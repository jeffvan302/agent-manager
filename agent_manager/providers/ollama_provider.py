"""Ollama provider adapter."""

from __future__ import annotations

import asyncio
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
from agent_manager.types import Message, ProviderRequest, ProviderResult, ProviderStreamEvent, ToolCallRequest


class OllamaProvider(HTTPProvider):
    provider_name = "ollama"
    default_base_url = "http://localhost:11434/api"
    capabilities = ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_system_messages=True,
    )

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        payload = self._build_payload(request)
        response = await self._request_json("POST", "chat", payload=payload)
        result = self._parse_response(response)
        if request.structured_output is not None:
            result.structured_output = maybe_parse_structured_output(
                result.text,
                request.structured_output,
            )
        return result

    async def stream_generate(
        self, request: ProviderRequest
    ) -> AsyncIterator[ProviderStreamEvent]:
        payload = self._build_payload(request)
        payload["stream"] = True

        # Make blocking HTTP request in thread pool
        def _make_request() -> str:
            import urllib.request

            url = self._build_url("chat")
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as response:
                return response.read().decode("utf-8")

        response_text = await asyncio.to_thread(_make_request)

        # Parse newline-delimited JSON and aggregate results
        accumulated_content: list[str] = []
        final_response: dict[str, Any] | None = None

        for line in response_text.strip().split("\n"):
            if not line:
                continue

            chunk = json.loads(line)

            # Yield text delta events
            message = chunk.get("message", {})
            if isinstance(message, Mapping):
                content = message.get("content", "")
                if content:
                    accumulated_content.append(content)
                    yield ProviderStreamEvent(kind="text_delta", text=content)

            # On done, build final result
            if chunk.get("done", False):
                final_response = chunk
                break

        # Build and yield final result
        if final_response:
            full_text = "".join(accumulated_content)
            final_message = final_response.get("message", {})
            if not isinstance(final_message, Mapping):
                final_message = {}

            # Parse tool calls from final response
            raw_tool_calls = final_message.get("tool_calls", [])
            tool_calls: list[ToolCallRequest] = []
            if isinstance(raw_tool_calls, list):
                for index, item in enumerate(raw_tool_calls, start=1):
                    if not isinstance(item, Mapping):
                        continue
                    function = item.get("function", {})
                    if not isinstance(function, Mapping):
                        function = {}
                    tool_calls.append(
                        ToolCallRequest(
                            id=ensure_tool_call_id(
                                str(item.get("id") or f"{self.provider_name}-call-{index}")
                            ),
                            name=str(function.get("name", "")),
                            arguments=coerce_arguments(function.get("arguments", {})),
                        )
                    )

            usage = {
                key: value
                for key, value in final_response.items()
                if key.endswith("_count") or key.endswith("_duration")
            }

            result = ProviderResult(
                text=full_text,
                tool_calls=tool_calls,
                stop_reason=self._normalize_stop_reason(
                    final_response.get("done_reason"),
                    has_tool_calls=bool(tool_calls),
                    done=bool(final_response.get("done")),
                ),
                usage=usage or None,
                raw=dict(final_response),
                metadata={
                    "provider": self.provider_name,
                    "model": final_response.get("model"),
                },
            )

            if request.structured_output is not None:
                result.structured_output = maybe_parse_structured_output(
                    result.text,
                    request.structured_output,
                )

            yield ProviderStreamEvent(kind="result", result=result)

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._to_ollama_message(message) for message in request.messages],
            "stream": False,
        }
        if request.tools:
            payload["tools"] = [self._to_ollama_tool(tool) for tool in request.tools]
        if request.structured_output is not None:
            payload["format"] = request.structured_output.schema or "json"
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options
        return payload

    def _to_ollama_message(self, message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.name:
            payload["name"] = message.name

        if message.role == "assistant":
            tool_calls = message_tool_calls(message.metadata)
            if tool_calls:
                payload["tool_calls"] = [
                    {
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        }
                    }
                    for call in tool_calls
                ]
            return payload

        if message.role == "tool":
            payload["tool_name"] = message.name or message.metadata.get("tool_name") or "tool"
        return payload

    def _to_ollama_tool(self, tool: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": str(tool.get("description", "")),
                "parameters": dict(tool.get("input_schema", {})),
            },
        }

    def _parse_response(self, response: Mapping[str, Any]) -> ProviderResult:
        message = response.get("message", {})
        if not isinstance(message, Mapping):
            message = {}

        raw_tool_calls = message.get("tool_calls", [])
        tool_calls: list[ToolCallRequest] = []
        if isinstance(raw_tool_calls, list):
            for index, item in enumerate(raw_tool_calls, start=1):
                if not isinstance(item, Mapping):
                    continue
                function = item.get("function", {})
                if not isinstance(function, Mapping):
                    function = {}
                tool_calls.append(
                    ToolCallRequest(
                        id=ensure_tool_call_id(
                            str(item.get("id") or f"{self.provider_name}-call-{index}")
                        ),
                        name=str(function.get("name", "")),
                        arguments=coerce_arguments(function.get("arguments", {})),
                    )
                )

        usage = {
            key: value
            for key, value in response.items()
            if key.endswith("_count") or key.endswith("_duration")
        }
        return ProviderResult(
            text=coerce_text(message.get("content")),
            tool_calls=tool_calls,
            stop_reason=self._normalize_stop_reason(
                response.get("done_reason"),
                has_tool_calls=bool(tool_calls),
                done=bool(response.get("done")),
            ),
            usage=usage or None,
            raw=dict(response),
            metadata={
                "provider": self.provider_name,
                "model": response.get("model"),
            },
        )

    def _normalize_stop_reason(
        self,
        done_reason: Any,
        *,
        has_tool_calls: bool,
        done: bool,
    ) -> str | None:
        if has_tool_calls:
            return "tool_call"
        if done_reason in {"stop", "completed"} or done:
            return "completed"
        if done_reason is None:
            return None
        return str(done_reason)
