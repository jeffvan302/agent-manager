"""Google Gemini provider adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent_manager.providers.base import (
    HTTPProvider,
    ProviderCapabilities,
    coerce_arguments,
    ensure_tool_call_id,
    message_tool_calls,
)
from agent_manager.types import Message, ProviderRequest, ProviderResult, ToolCallRequest


class GeminiProvider(HTTPProvider):
    provider_name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    default_api_key_env = "GEMINI_API_KEY"
    requires_api_key = True
    capabilities = ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_system_messages=True,
    )

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        payload = self._build_payload(request)
        model_name = request.model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        response = await self._request_json(
            "POST",
            f"{model_name}:generateContent",
            payload=payload,
            headers=self._auth_headers(),
        )
        return self._parse_response(response)

    def _auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": str(self.resolve_api_key(required=True))}

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        system_text, contents = self._to_gemini_contents(request.messages)
        payload: dict[str, Any] = {
            "contents": contents,
        }

        if system_text:
            payload["systemInstruction"] = {
                "parts": [{"text": system_text}],
            }
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        self._to_gemini_function_declaration(tool)
                        for tool in request.tools
                    ]
                }
            ]

        generation_config: dict[str, Any] = {}
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    def _to_gemini_contents(
        self,
        messages: list[Message],
    ) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            content = self._to_gemini_content(message)
            if content is not None:
                contents.append(content)

        return "\n\n".join(system_parts), contents

    def _to_gemini_content(self, message: Message) -> dict[str, Any] | None:
        if message.role == "tool":
            response_payload = self._tool_response_payload(message)
            return {
                "role": "tool",
                "parts": [
                    {
                        "functionResponse": {
                            "name": message.name or "tool",
                            "response": response_payload,
                        }
                    }
                ],
            }

        if message.role == "assistant":
            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"text": message.content})
            for call in message_tool_calls(message.metadata):
                parts.append(
                    {
                        "functionCall": {
                            "name": call["name"],
                            "args": call["arguments"],
                        }
                    }
                )
            if not parts:
                return None
            return {"role": "model", "parts": parts}

        if not message.content:
            return None
        return {"role": "user", "parts": [{"text": message.content}]}

    def _tool_response_payload(self, message: Message) -> dict[str, Any]:
        try:
            parsed = json.loads(message.content)
        except json.JSONDecodeError:
            return {"result": message.content}
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}

    def _to_gemini_function_declaration(self, tool: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "name": str(tool["name"]),
            "description": str(tool.get("description", "")),
            "parameters": dict(tool.get("input_schema", {})),
        }

    def _parse_response(self, response: Mapping[str, Any]) -> ProviderResult:
        candidate: Mapping[str, Any] = {}
        candidates = response.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            if isinstance(first_candidate, Mapping):
                candidate = first_candidate

        content = candidate.get("content", {})
        if not isinstance(content, Mapping):
            content = {}
        parts = content.get("parts", [])

        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        if isinstance(parts, list):
            for index, part in enumerate(parts, start=1):
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                function_call = part.get("functionCall") or part.get("function_call")
                if isinstance(function_call, Mapping):
                    tool_calls.append(
                        ToolCallRequest(
                            id=ensure_tool_call_id(
                                str(function_call.get("id") or f"{self.provider_name}-call-{index}")
                            ),
                            name=str(function_call.get("name", "")),
                            arguments=coerce_arguments(
                                function_call.get("args", function_call.get("arguments", {}))
                            ),
                        )
                    )

        finish_reason = candidate.get("finishReason") or candidate.get("finish_reason")
        if tool_calls:
            normalized_stop_reason = "tool_call"
        elif finish_reason == "STOP":
            normalized_stop_reason = "completed"
        elif finish_reason is None:
            normalized_stop_reason = None
        else:
            normalized_stop_reason = str(finish_reason).lower()

        usage = response.get("usageMetadata") or response.get("usage_metadata")
        if not isinstance(usage, Mapping):
            usage = None
        return ProviderResult(
            text="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            stop_reason=normalized_stop_reason,
            usage=usage,
            raw=dict(response),
            metadata={
                "provider": self.provider_name,
                "model_version": response.get("modelVersion"),
                "response_id": response.get("responseId"),
            },
        )
