"""Anthropic provider adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agent_manager.providers.base import (
    HTTPProvider,
    ProviderCapabilities,
    coerce_arguments,
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


class AnthropicProvider(HTTPProvider):
    provider_name = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"
    default_api_key_env = "ANTHROPIC_API_KEY"
    requires_api_key = True
    capabilities = ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_structured_output=False,
        supports_system_messages=True,
    )

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        payload = self._build_payload(request)
        response = await self._request_json(
            "POST",
            "messages",
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
        payload = self._build_payload(request)
        payload["stream"] = True
        stream_events = await self._stream_sse_events(
            "messages",
            payload=payload,
            headers=self._auth_headers(),
        )

        text_parts: list[str] = []
        tool_buffers: dict[str, dict[str, Any]] = {}
        stop_reason: str | None = None
        usage: dict[str, Any] | None = None
        message_id: str | None = None
        model: str | None = None

        for event_type, data in stream_events:
            if event_type == "message_start":
                message = data.get("message", {})
                if isinstance(message, Mapping):
                    message_id = str(message.get("id") or "") or None
                    model = str(message.get("model") or "") or None
                    raw_usage = message.get("usage")
                    if isinstance(raw_usage, Mapping):
                        usage = dict(raw_usage)
                continue

            if event_type == "content_block_start":
                block = data.get("content_block", {})
                index = str(data.get("index", len(tool_buffers)))
                if not isinstance(block, Mapping):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = str(block.get("text") or "")
                    if text:
                        text_parts.append(text)
                        yield ProviderStreamEvent(
                            kind="text_delta",
                            text=text,
                            metadata={"provider": self.provider_name},
                        )
                elif block_type == "tool_use":
                    tool_buffers[index] = {
                        "id": ensure_tool_call_id(str(block.get("id") or "")),
                        "name": str(block.get("name") or ""),
                        "input": block.get("input"),
                        "input_chunks": [],
                    }
                continue

            if event_type == "content_block_delta":
                delta = data.get("delta", {})
                index = str(data.get("index", len(tool_buffers)))
                if not isinstance(delta, Mapping):
                    continue
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = str(delta.get("text") or "")
                    if text:
                        text_parts.append(text)
                        yield ProviderStreamEvent(
                            kind="text_delta",
                            text=text,
                            metadata={"provider": self.provider_name},
                        )
                elif delta_type == "input_json_delta":
                    buffer = tool_buffers.setdefault(
                        index,
                        {
                            "id": ensure_tool_call_id(),
                            "name": "",
                            "input": None,
                            "input_chunks": [],
                        },
                    )
                    partial_json = str(
                        delta.get("partial_json") or delta.get("text") or ""
                    )
                    if partial_json:
                        buffer["input_chunks"].append(partial_json)
                continue

            if event_type == "message_delta":
                delta = data.get("delta", {})
                if isinstance(delta, Mapping):
                    raw_stop_reason = delta.get("stop_reason")
                    if raw_stop_reason == "end_turn":
                        stop_reason = "completed"
                    elif raw_stop_reason == "tool_use":
                        stop_reason = "tool_call"
                    elif raw_stop_reason is not None:
                        stop_reason = str(raw_stop_reason)
                raw_usage = data.get("usage")
                if isinstance(raw_usage, Mapping):
                    usage = dict(raw_usage)

        tool_calls = self._finalize_stream_tool_calls(tool_buffers)
        final_stop_reason = self._normalize_stop_reason(
            stop_reason,
            has_tool_calls=bool(tool_calls),
        )
        result = ProviderResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=final_stop_reason,
            usage=usage,
            raw={},
            metadata={
                "provider": self.provider_name,
                "model": model,
                "id": message_id,
                "streamed": True,
            },
        )
        if request.structured_output is not None:
            result.structured_output = maybe_parse_structured_output(
                result.text,
                request.structured_output,
            )
        yield ProviderStreamEvent(
            kind="result",
            result=result,
            metadata={"provider": self.provider_name},
        )

    async def _stream_sse_events(
        self,
        path: str,
        *,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> list[tuple[str, dict[str, Any]]]:
        url = self._build_url(path)
        request_headers = {**self.default_headers(), **headers}
        request_body = json.dumps(payload).encode("utf-8")
        timeout = float(self.config.settings.get("request_timeout_seconds", 120.0))

        def _stream_blocking() -> list[tuple[str, dict[str, Any]]]:
            request = Request(
                url=url,
                data=request_body,
                headers=request_headers,
                method="POST",
            )
            parsed_events: list[tuple[str, dict[str, Any]]] = []
            current_event = ""
            current_data = ""
            try:
                with urlopen(request, timeout=timeout) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                        if not line:
                            event = self._decode_sse_event(current_event, current_data)
                            if event is not None:
                                parsed_events.append(event)
                            current_event = ""
                            current_data = ""
                            continue
                        if line.startswith("event: "):
                            current_event = line[7:].strip()
                            continue
                        if line.startswith("data: "):
                            data_line = line[6:]
                            current_data = (
                                f"{current_data}\n{data_line}"
                                if current_data
                                else data_line
                            )
                    event = self._decode_sse_event(current_event, current_data)
                    if event is not None:
                        parsed_events.append(event)
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                raise self._http_error_to_provider_error(
                    exc.code,
                    error_body,
                    exc.headers,
                ) from exc
            return parsed_events

        return await asyncio.to_thread(_stream_blocking)

    def _decode_sse_event(
        self,
        event_type: str,
        data_text: str,
    ) -> tuple[str, dict[str, Any]] | None:
        if not event_type or not data_text:
            return None
        try:
            payload = json.loads(data_text)
        except JSONDecodeError:
            return None
        if not isinstance(payload, Mapping):
            return None
        return event_type, dict(payload)

    def _finalize_stream_tool_calls(
        self,
        tool_buffers: Mapping[str, Mapping[str, Any]],
    ) -> list[ToolCallRequest]:
        def _sort_key(value: str) -> tuple[int, str]:
            return (0, value) if value.isdigit() else (1, value)

        tool_calls: list[ToolCallRequest] = []
        for index in sorted(tool_buffers.keys(), key=_sort_key):
            buffer = tool_buffers[index]
            name = str(buffer.get("name") or "")
            if not name:
                continue
            input_payload = buffer.get("input")
            if isinstance(input_payload, Mapping):
                arguments = coerce_arguments(dict(input_payload))
            else:
                arguments = coerce_arguments("".join(buffer.get("input_chunks", [])))
            tool_calls.append(
                ToolCallRequest(
                    id=ensure_tool_call_id(str(buffer.get("id") or "")),
                    name=name,
                    arguments=arguments,
                )
            )
        return tool_calls

    def _auth_headers(self) -> dict[str, str]:
        return {
            "x-api-key": str(self.resolve_api_key(required=True)),
            "anthropic-version": str(
                self.config.settings.get("anthropic_version", "2023-06-01")
            ),
        }

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        system_text, messages = self._to_anthropic_messages(request.messages)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or int(
                self.config.settings.get("max_tokens", 1024)
            ),
        }
        if system_text:
            payload["system"] = system_text
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [self._to_anthropic_tool(tool) for tool in request.tools]
        return payload

    def _to_anthropic_messages(
        self,
        messages: list[Message],
    ) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            role, content_blocks = self._to_anthropic_content(message)
            if not content_blocks:
                continue
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(content_blocks)
            else:
                converted.append({"role": role, "content": content_blocks})

        return "\n\n".join(system_parts), converted

    def _to_anthropic_content(self, message: Message) -> tuple[str, list[dict[str, Any]]]:
        if message.role == "tool":
            tool_use_id = message.metadata.get("tool_call_id")
            block = {
                "type": "tool_result",
                "tool_use_id": str(tool_use_id or ""),
                "content": message.content,
            }
            if message.metadata.get("is_error"):
                block["is_error"] = True
            return "user", [block]

        if message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message_tool_calls(message.metadata):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["arguments"],
                    }
                )
            return "assistant", blocks

        if message.content:
            return message.role, [{"type": "text", "text": message.content}]
        return message.role, []

    def _to_anthropic_tool(self, tool: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "name": str(tool["name"]),
            "description": str(tool.get("description", "")),
            "input_schema": dict(tool.get("input_schema", {})),
        }

    def _parse_response(self, response: Mapping[str, Any]) -> ProviderResult:
        raw_blocks = response.get("content", [])
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        if isinstance(raw_blocks, list):
            for block in raw_blocks:
                if not isinstance(block, Mapping):
                    continue
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                elif block_type == "tool_use":
                    tool_calls.append(
                        ToolCallRequest(
                            id=ensure_tool_call_id(str(block.get("id") or "")),
                            name=str(block.get("name", "")),
                            arguments=coerce_arguments(block.get("input", {})),
                        )
                    )

        stop_reason = response.get("stop_reason")
        normalized_stop_reason = self._normalize_stop_reason(
            stop_reason,
            has_tool_calls=bool(tool_calls),
        )
        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else None
        return ProviderResult(
            text="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            stop_reason=normalized_stop_reason,
            usage=usage,
            raw=dict(response),
            metadata={
                "provider": self.provider_name,
                "model": response.get("model"),
                "id": response.get("id"),
            },
        )

    def _normalize_stop_reason(
        self,
        stop_reason: Any,
        *,
        has_tool_calls: bool,
    ) -> str | None:
        if has_tool_calls:
            return "tool_call"
        if stop_reason == "end_turn":
            return "completed"
        if stop_reason is None:
            return None
        return str(stop_reason)
