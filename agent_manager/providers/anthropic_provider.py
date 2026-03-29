"""Anthropic provider adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_manager.providers.base import (
    HTTPProvider,
    ProviderCapabilities,
    coerce_arguments,
    ensure_tool_call_id,
    message_tool_calls,
    maybe_parse_structured_output,
)
from agent_manager.types import Message, ProviderRequest, ProviderResult, ToolCallRequest


class AnthropicProvider(HTTPProvider):
    provider_name = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"
    default_api_key_env = "ANTHROPIC_API_KEY"
    requires_api_key = True
    capabilities = ProviderCapabilities(
        supports_tools=True,
        supports_streaming=False,
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
        if tool_calls:
            normalized_stop_reason = "tool_call"
        elif stop_reason == "end_turn":
            normalized_stop_reason = "completed"
        elif stop_reason is None:
            normalized_stop_reason = None
        else:
            normalized_stop_reason = str(stop_reason)

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
