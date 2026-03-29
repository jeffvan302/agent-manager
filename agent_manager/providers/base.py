"""Base provider interfaces."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agent_manager.config import ProviderConfig
from agent_manager.errors import (
    ConfigurationError,
    ProviderError,
    ProviderRequestError,
    ProviderResourceExhaustedError,
)
from agent_manager.types import (
    ProviderRequest,
    ProviderResult,
    ProviderStreamEvent,
    StructuredOutputSpec,
)
from agent_manager.version import __version__


@dataclass(slots=True)
class ProviderCapabilities:
    supports_tools: bool = False
    supports_streaming: bool = False
    supports_structured_output: bool = False
    supports_images: bool = False
    supports_system_messages: bool = True


class BaseProvider(ABC):
    """Normalized provider interface used by the runtime loop."""

    provider_name = "base"
    capabilities = ProviderCapabilities()

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig()

    @abstractmethod
    async def generate(self, request: ProviderRequest) -> ProviderResult:
        """Run one normalized generation request."""

    async def stream_generate(
        self,
        request: ProviderRequest,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Default streaming wrapper for providers without native token streams."""
        result = await self.generate(request)
        if result.text:
            yield ProviderStreamEvent(
                kind="text_delta",
                text=result.text,
                metadata={"provider": self.provider_name},
            )
        yield ProviderStreamEvent(
            kind="result",
            result=result,
            metadata={"provider": self.provider_name},
        )


class HTTPProvider(BaseProvider):
    """Shared HTTP helpers for JSON-based provider adapters."""

    default_base_url = ""
    default_api_key_env: str | None = None
    requires_api_key = False

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        if not self.config.base_url and self.default_base_url:
            self.config.base_url = self.default_base_url
        if not self.config.api_key_env and self.default_api_key_env:
            self.config.api_key_env = self.default_api_key_env

    def resolve_base_url(self) -> str:
        base_url = (self.config.base_url or self.default_base_url).rstrip("/")
        if not base_url:
            raise ConfigurationError(
                f"{self.provider_name} requires a base_url or default base URL."
            )
        return base_url

    def resolve_api_key(self, *, required: bool | None = None) -> str | None:
        required = self.requires_api_key if required is None else required
        explicit_key = self.config.settings.get("api_key")
        if isinstance(explicit_key, str) and explicit_key:
            return explicit_key

        env_name = self.config.api_key_env
        if env_name:
            env_value = os.getenv(env_name)
            if env_value:
                return env_value

        if required:
            detail = f" via {env_name}" if env_name else ""
            raise ConfigurationError(
                f"{self.provider_name} requires an API key{detail}."
            )
        return None

    def default_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"agent-manager/{__version__}",
        }
        configured_headers = self.config.settings.get("headers")
        if isinstance(configured_headers, Mapping):
            headers.update({str(key): str(value) for key, value in configured_headers.items()})
        return headers

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._build_url(path, query=query)
        request_headers = self.default_headers()
        if headers:
            request_headers.update(headers)
        timeout = float(self.config.settings.get("request_timeout_seconds", 60.0))
        max_attempts = max(int(self.config.settings.get("request_retries", 3)), 1)
        base_backoff_seconds = float(self.config.settings.get("request_retry_backoff_seconds", 0.5))

        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.to_thread(
                    self._request_json_blocking,
                    method,
                    url,
                    payload,
                    request_headers,
                    timeout,
                )
            except ProviderResourceExhaustedError:
                # Resource exhaustion is never retried automatically.
                raise
            except ProviderRequestError as exc:
                if not exc.retryable or attempt >= max_attempts:
                    raise
                await asyncio.sleep(base_backoff_seconds * attempt)

        raise ProviderError(f"{self.provider_name} request failed after retry exhaustion.")

    def _build_url(self, path: str, *, query: Mapping[str, Any] | None = None) -> str:
        base_url = self.resolve_base_url()
        normalized_path = path.lstrip("/")
        url = f"{base_url}/{normalized_path}"
        if query:
            filtered_query = {
                key: value
                for key, value in query.items()
                if value is not None
            }
            if filtered_query:
                url = f"{url}?{urlencode(filtered_query)}"
        return url

    def _request_json_blocking(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        request = Request(url=url, data=body, headers=dict(headers), method=method.upper())
        try:
            with urlopen(request, timeout=timeout) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise self._http_error_to_provider_error(
                exc.code, error_body, exc.headers,
            ) from exc
        except URLError as exc:
            raise ProviderRequestError(
                f"{self.provider_name} request failed: {exc.reason}",
                retryable=True,
            ) from exc

        if not raw_body:
            return {}

        try:
            return json.loads(raw_body)
        except JSONDecodeError as exc:
            raise ProviderError(
                f"{self.provider_name} returned a non-JSON response."
            ) from exc

    # ------------------------------------------------------------------
    # Out-of-resource detection (requirements 6.1 + 7.4)
    # ------------------------------------------------------------------

    def _http_error_to_provider_error(
        self,
        status_code: int,
        error_body: str,
        headers: Mapping[str, Any] | None,
    ) -> ProviderRequestError:
        """Classify an HTTP error as resource-exhaustion or generic failure.

        Provider adapters that need deeper classification can override this
        method, but the base implementation covers the most common patterns
        across OpenAI, Anthropic, Gemini, and local providers.
        """
        error_payload = self._parse_error_payload(error_body)
        resource_kind = self._classify_resource_exhaustion(
            status_code, error_body, error_payload,
        )
        retry_after_seconds = self._parse_retry_after(headers)

        if resource_kind is not None:
            message = self._resource_exhaustion_message(
                status_code, error_body, error_payload,
            )
            return ProviderResourceExhaustedError(
                message,
                provider=self.provider_name,
                kind=resource_kind,
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
                metadata=self._resource_exhaustion_metadata(
                    status_code, error_payload, retry_after_seconds,
                ),
            )

        return ProviderRequestError(
            f"{self.provider_name} API error ({status_code}): {error_body}",
            retryable=status_code in {408, 409, 429, 500, 502, 503, 504},
        )

    def _parse_error_payload(self, error_body: str) -> dict[str, Any] | None:
        if not error_body.strip():
            return None
        try:
            payload = json.loads(error_body)
        except JSONDecodeError:
            return None
        if isinstance(payload, Mapping):
            return dict(payload)
        return None

    def _classify_resource_exhaustion(
        self,
        status_code: int,
        error_body: str,
        error_payload: Mapping[str, Any] | None,
    ) -> str | None:
        """Return an exhaustion-kind string, or None if not a resource issue."""
        details = self._flatten_error_details(error_body, error_payload)

        # --- Quota / billing exhaustion ---
        quota_terms = {
            "insufficient_quota",
            "quota_exceeded",
            "credit_balance_too_low",
            "billing_hard_limit_reached",
            "credits_exhausted",
            "out_of_credits",
            "payment_required",
        }
        if any(term in details for term in quota_terms):
            return "quota_exhausted"
        if status_code == 402:
            return "quota_exhausted"

        # --- Rate limiting ---
        rate_limit_terms = {
            "rate_limit_exceeded",
            "rate_limited",
            "too_many_requests",
            "requests_per_minute",
            "tokens_per_minute",
        }
        if any(term in details for term in rate_limit_terms):
            return "rate_limited"
        if status_code == 429 and not any(term in details for term in quota_terms):
            return "rate_limited"

        # --- Model / capacity exhaustion ---
        capacity_terms = {
            "model_overloaded",
            "overloaded",
            "capacity",
            "server_overloaded",
            "model_not_available",
        }
        if any(term in details for term in capacity_terms):
            return "capacity_exhausted"
        if status_code == 529:
            return "capacity_exhausted"

        return None

    def _flatten_error_details(
        self,
        error_body: str,
        error_payload: Mapping[str, Any] | None,
    ) -> str:
        """Flatten an error payload into a lowercase string for term matching."""
        parts: list[str] = [error_body.lower()]
        if error_payload:
            # Walk common error envelope shapes.
            error_obj = error_payload.get("error")
            if isinstance(error_obj, Mapping):
                parts.append(str(error_obj.get("type", "")).lower())
                parts.append(str(error_obj.get("code", "")).lower())
                parts.append(str(error_obj.get("message", "")).lower())
            parts.append(str(error_payload.get("type", "")).lower())
            parts.append(str(error_payload.get("code", "")).lower())
            parts.append(str(error_payload.get("message", "")).lower())
        return " ".join(parts)

    def _parse_retry_after(self, headers: Mapping[str, Any] | None) -> float | None:
        """Extract Retry-After seconds from HTTP response headers."""
        if not headers:
            return None
        raw = None
        # Case-insensitive header lookup.
        for key in ("Retry-After", "retry-after", "x-ratelimit-reset-tokens"):
            value = headers.get(key) if hasattr(headers, "get") else None
            if value is not None:
                raw = value
                break
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _resource_exhaustion_message(
        self,
        status_code: int,
        error_body: str,
        error_payload: Mapping[str, Any] | None,
    ) -> str:
        """Build a human-readable message for the resource exhaustion error."""
        if error_payload:
            error_obj = error_payload.get("error")
            if isinstance(error_obj, Mapping):
                msg = error_obj.get("message")
                if isinstance(msg, str) and msg:
                    return f"{self.provider_name}: {msg}"
            msg = error_payload.get("message")
            if isinstance(msg, str) and msg:
                return f"{self.provider_name}: {msg}"
        snippet = error_body[:200] if error_body else "(no body)"
        return f"{self.provider_name} resource exhaustion ({status_code}): {snippet}"

    def _resource_exhaustion_metadata(
        self,
        status_code: int,
        error_payload: Mapping[str, Any] | None,
        retry_after_seconds: float | None,
    ) -> dict[str, Any]:
        """Gather provider metadata to help the caller decide what to do."""
        meta: dict[str, Any] = {"status_code": status_code}
        if retry_after_seconds is not None:
            meta["retry_after_seconds"] = retry_after_seconds
        if error_payload:
            error_obj = error_payload.get("error")
            if isinstance(error_obj, Mapping):
                for key in ("type", "code", "param"):
                    value = error_obj.get(key)
                    if value is not None:
                        meta[f"error_{key}"] = value
        return meta


# ------------------------------------------------------------------
# Shared helper functions used by provider adapters
# ------------------------------------------------------------------

def coerce_text(value: Any) -> str:
    """Extract human-readable text from provider-specific content payloads."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, Mapping):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    return str(value)


def maybe_parse_structured_output(
    text: str | None,
    spec: StructuredOutputSpec | None,
) -> Any | None:
    """Best-effort JSON extraction for structured-output requests."""
    if spec is None or text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate:
                candidates.append(candidate)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            continue
    return None


def coerce_arguments(value: Any) -> dict[str, Any]:
    """Normalize tool arguments into a dictionary."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except JSONDecodeError:
            return {"_raw_arguments": stripped}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {"value": value}


def ensure_tool_call_id(value: str | None = None) -> str:
    if value:
        return value
    return uuid.uuid4().hex


def message_tool_calls(metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not metadata:
        return []
    raw_calls = metadata.get("tool_calls", [])
    if not isinstance(raw_calls, Iterable) or isinstance(raw_calls, (str, bytes, dict)):
        return []
    calls: list[dict[str, Any]] = []
    for item in raw_calls:
        if isinstance(item, Mapping) and "name" in item:
            call_id = ensure_tool_call_id(str(item.get("id") or ""))
            calls.append(
                {
                    "id": call_id,
                    "name": str(item["name"]),
                    "arguments": coerce_arguments(item.get("arguments", {})),
                }
            )
    return calls
