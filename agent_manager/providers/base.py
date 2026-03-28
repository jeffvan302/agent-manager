"""Base provider interfaces."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agent_manager.config import ProviderConfig
from agent_manager.errors import ConfigurationError, ProviderError
from agent_manager.types import ProviderRequest, ProviderResult


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
        return await asyncio.to_thread(
            self._request_json_blocking,
            method,
            url,
            payload,
            request_headers,
            timeout,
        )

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
            raise ProviderError(
                f"{self.provider_name} API error ({exc.code}): {error_body}"
            ) from exc
        except URLError as exc:
            raise ProviderError(
                f"{self.provider_name} request failed: {exc.reason}"
            ) from exc

        if not raw_body:
            return {}

        try:
            return json.loads(raw_body)
        except JSONDecodeError as exc:
            raise ProviderError(
                f"{self.provider_name} returned a non-JSON response."
            ) from exc


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
