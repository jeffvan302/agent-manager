"""OpenAI-compatible chat completion providers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from agent_manager.providers.base import (
    HTTPProvider,
    ProviderCapabilities,
    coerce_arguments,
    coerce_text,
    ensure_tool_call_id,
    message_tool_calls,
    maybe_parse_structured_output,
)
from agent_manager.types import (
    Message,
    ProviderRequest,
    ProviderResult,
    ProviderStreamEvent,
    ToolCallRequest,
)


class OpenAICompatibleChatProvider(HTTPProvider):
    """Shared adapter for OpenAI-style chat completion APIs."""

    endpoint_path = "chat/completions"
    requires_api_key = False
    capabilities = ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_system_messages=True,
    )

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        payload = self._build_payload(request)
        response = await self._request_json(
            "POST",
            self.endpoint_path,
            payload=payload,
            headers=self._auth_headers(),
        )
        result = self._parse_response(response)
        if request.structured_output is not None:
            result.structured_output = maybe_parse_structured_output(
                result.text,
                request.structured_output,
            )
        return result

    async def stream_generate(
        self,
        request: ProviderRequest,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Native SSE streaming for OpenAI-compatible endpoints."""
        import asyncio
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError

        payload = self._build_payload(request)
        payload["stream"] = True
        url = self._build_url(self.endpoint_path)
        headers = {**self.default_headers(), **self._auth_headers()}
        body = json.dumps(payload).encode("utf-8")
        req = Request(url=url, data=body, headers=headers, method="POST")

        def _stream_blocking():
            chunks = []
            try:
                with urlopen(req, timeout=120) as resp:
                    buffer = ""
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace")
                        buffer += line
                        while "\n" in buffer:
                            event_line, buffer = buffer.split("\n", 1)
                            event_line = event_line.strip()
                            if not event_line or not event_line.startswith("data: "):
                                continue
                            data_str = event_line[6:]
                            if data_str == "[DONE]":
                                return chunks
                            try:
                                chunk = json.loads(data_str)
                                chunks.append(chunk)
                            except json.JSONDecodeError:
                                continue
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                raise self._http_error_to_provider_error(
                    exc.code, error_body, exc.headers,
                ) from exc
            return chunks

        chunks = await asyncio.to_thread(_stream_blocking)

        # Reassemble chunks into deltas and final result.
        collected_text = ""
        collected_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage = None

        for chunk in chunks:
            choices = chunk.get("choices", [])
            if not choices:
                if "usage" in chunk:
                    usage = chunk["usage"]
                continue
            delta = choices[0].get("delta", {})
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr

            # Text delta.
            content = delta.get("content")
            if content:
                collected_text += content
                yield ProviderStreamEvent(
                    kind="text_delta",
                    text=content,
                    metadata={"provider": self.provider_name},
                )

            # Tool call deltas.
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in collected_tool_calls:
                    collected_tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "name": "",
                        "arguments": "",
                    }
                if tc.get("id"):
                    collected_tool_calls[idx]["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    collected_tool_calls[idx]["name"] = func["name"]
                if func.get("arguments"):
                    collected_tool_calls[idx]["arguments"] += func["arguments"]

        # Build final result.
        tool_calls = []
        for _idx in sorted(collected_tool_calls.keys()):
            tc = collected_tool_calls[_idx]
            tool_calls.append(
                ToolCallRequest(
                    id=ensure_tool_call_id(tc["id"]),
                    name=tc["name"],
                    arguments=coerce_arguments(tc["arguments"]),
                )
            )

        stop = self._normalize_stop_reason(finish_reason, has_tool_calls=bool(tool_calls))
        result = ProviderResult(
            text=collected_text or None,
            tool_calls=tool_calls,
            stop_reason=stop,
            usage=usage,
            raw=None,
            metadata={"provider": self.provider_name, "streamed": True},
        )
        if request.structured_output is not None:
            result.structured_output = maybe_parse_structured_output(
                result.text, request.structured_output,
            )
        yield ProviderStreamEvent(
            kind="result",
            result=result,
            metadata={"provider": self.provider_name},
        )

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        api_key = self.resolve_api_key(required=self.requires_api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._to_openai_message(message) for message in request.messages],
        }

        if request.tools:
            payload["tools"] = [self._to_openai_tool(tool) for tool in request.tools]
            payload["tool_choice"] = "auto"
        if request.max_tokens is not None:
            self._set_output_token_limit(payload, request.max_tokens)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.structured_output is not None:
            payload["response_format"] = self._response_format(request)
        return payload

    def _set_output_token_limit(self, payload: dict[str, Any], max_tokens: int) -> None:
        payload["max_tokens"] = max_tokens

    def _response_format(self, request: ProviderRequest) -> dict[str, Any]:
        spec = request.structured_output
        if spec is None:
            return {"type": "json_object"}
        if spec.type == "json_schema" and spec.schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": spec.name or "response",
                    "schema": spec.schema,
                    "strict": spec.strict,
                },
            }
        return {"type": "json_object"}

    def _to_openai_message(self, message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role}

        if message.role == "tool":
            payload["content"] = message.content
            if message.name:
                payload["name"] = message.name
            tool_call_id = message.metadata.get("tool_call_id")
            if tool_call_id:
                payload["tool_call_id"] = str(tool_call_id)
            return payload

        if message.name and message.role != "assistant":
            payload["name"] = message.name

        if message.role == "assistant":
            tool_calls = message_tool_calls(message.metadata)
            if message.content:
                payload["content"] = message.content
            elif not tool_calls:
                payload["content"] = ""
            if tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=True),
                        },
                    }
                    for call in tool_calls
                ]
            return payload

        payload["content"] = message.content
        return payload

    def _to_openai_tool(self, tool: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": str(tool.get("description", "")),
                "parameters": dict(tool.get("input_schema", {})),
            },
        }

    def _parse_response(self, response: Mapping[str, Any]) -> ProviderResult:
        choice = {}
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, Mapping):
                choice = dict(first_choice)

        message: Mapping[str, Any] = {}
        raw_message = choice.get("message")
        if isinstance(raw_message, Mapping):
            message = raw_message

        raw_tool_calls = message.get("tool_calls", [])
        tool_calls: list[ToolCallRequest] = []
        if isinstance(raw_tool_calls, list):
            for item in raw_tool_calls:
                if not isinstance(item, Mapping):
                    continue
                function = item.get("function", {})
                if not isinstance(function, Mapping):
                    function = {}
                tool_calls.append(
                    ToolCallRequest(
                        id=ensure_tool_call_id(str(item.get("id") or "")),
                        name=str(function.get("name", "")),
                        arguments=coerce_arguments(function.get("arguments", {})),
                    )
                )

        finish_reason = choice.get("finish_reason")
        stop_reason = self._normalize_stop_reason(finish_reason, has_tool_calls=bool(tool_calls))
        return ProviderResult(
            text=coerce_text(message.get("content")),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=response.get("usage") if isinstance(response.get("usage"), Mapping) else None,
            raw=dict(response),
            metadata={
                "provider": self.provider_name,
                "model": response.get("model"),
            },
        )

    def _normalize_stop_reason(self, finish_reason: Any, *, has_tool_calls: bool) -> str | None:
        if has_tool_calls:
            return "tool_call"
        if finish_reason in {"stop", "completed"}:
            return "completed"
        if finish_reason is None:
            return None
        return str(finish_reason)


class OpenAIProvider(OpenAICompatibleChatProvider):
    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_api_key_env = "OPENAI_API_KEY"

    def _set_output_token_limit(self, payload: dict[str, Any], max_tokens: int) -> None:
        # OpenAI's chat-completions endpoint now prefers max_completion_tokens,
        # and newer reasoning / o-series models reject max_tokens entirely.
        payload["max_completion_tokens"] = max_tokens
    requires_api_key = True
