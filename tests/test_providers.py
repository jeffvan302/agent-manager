from __future__ import annotations

import asyncio
import unittest
from collections.abc import Mapping
from typing import Any

from agent_manager.config import ProviderConfig
from agent_manager.providers.anthropic_provider import AnthropicProvider
from agent_manager.providers.base import HTTPProvider
from agent_manager.providers.gemini_provider import GeminiProvider
from agent_manager.providers.lmstudio_provider import LMStudioProvider
from agent_manager.providers.ollama_provider import OllamaProvider
from agent_manager.providers.openai_provider import OpenAIProvider
from agent_manager.providers.vllm_provider import VLLMProvider
from agent_manager.errors import ProviderRequestError, ProviderResourceExhaustedError
from agent_manager.types import Message, ProviderRequest


BASE_MESSAGES = [
    Message(role="system", content="You are concise."),
    Message(role="user", content="Check the weather."),
    Message(
        role="assistant",
        content="I'll use the weather tool.",
        metadata={
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "lookup_weather",
                    "arguments": {"city": "Paris"},
                }
            ]
        },
    ),
    Message(
        role="tool",
        name="lookup_weather",
        content='{"temperature_c": 22}',
        metadata={"tool_call_id": "call_1"},
    ),
    Message(role="user", content="Summarize it."),
]

BASE_TOOLS = [
    {
        "name": "lookup_weather",
        "description": "Look up weather information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    }
]


class CaptureMixin:
    def __init__(self, response_payload: Mapping[str, Any], config: ProviderConfig) -> None:
        super().__init__(config)
        self.response_payload = dict(response_payload)
        self.last_method: str | None = None
        self.last_path: str | None = None
        self.last_payload: Mapping[str, Any] | None = None
        self.last_headers: Mapping[str, str] | None = None

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del query
        self.last_method = method
        self.last_path = path
        self.last_payload = payload or {}
        self.last_headers = headers or {}
        return dict(self.response_payload)


class CaptureOpenAIProvider(CaptureMixin, OpenAIProvider):
    pass


class CaptureAnthropicProvider(CaptureMixin, AnthropicProvider):
    pass


class CaptureGeminiProvider(CaptureMixin, GeminiProvider):
    pass


class CaptureOllamaProvider(CaptureMixin, OllamaProvider):
    pass


class CaptureLMStudioProvider(CaptureMixin, LMStudioProvider):
    pass


class CaptureVLLMProvider(CaptureMixin, VLLMProvider):
    pass


class CaptureStreamingAnthropicProvider(AnthropicProvider):
    def __init__(
        self,
        stream_events: list[tuple[str, dict[str, Any]]],
        config: ProviderConfig,
    ) -> None:
        super().__init__(config)
        self.stream_events = list(stream_events)
        self.last_path: str | None = None
        self.last_payload: Mapping[str, Any] | None = None
        self.last_headers: Mapping[str, str] | None = None

    async def _stream_sse_events(
        self,
        path: str,
        *,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> list[tuple[str, dict[str, Any]]]:
        self.last_path = path
        self.last_payload = payload
        self.last_headers = headers
        return list(self.stream_events)


class RetryableTestProvider(HTTPProvider):
    provider_name = "retry-test"
    default_base_url = "https://example.test"

    def __init__(self, failures_before_success: int, config: ProviderConfig) -> None:
        super().__init__(config)
        self.failures_before_success = failures_before_success
        self.calls = 0
        self.last_headers: Mapping[str, str] | None = None

    async def generate(self, request: ProviderRequest):
        del request
        raise NotImplementedError

    def _request_json_blocking(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        del method, url, payload, timeout
        self.calls += 1
        self.last_headers = headers
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise ProviderRequestError("transient failure", retryable=True)
        return {"ok": True}


class ErrorInspectingProvider(HTTPProvider):
    provider_name = "error-inspector"
    default_base_url = "https://example.test"

    async def generate(self, request: ProviderRequest):
        del request
        raise NotImplementedError


class ProviderAdapterTests(unittest.TestCase):
    def test_openai_adapter_translates_messages_tools_and_response(self) -> None:
        provider = CaptureOpenAIProvider(
            {
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "Checking now.",
                            "tool_calls": [
                                {
                                    "id": "call_openai_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            ProviderConfig(
                name="openai",
                model="gpt-4.1-mini",
                settings={"api_key": "test-key"},
            ),
        )
        request = ProviderRequest(
            model="gpt-4.1-mini",
            messages=BASE_MESSAGES,
            tools=BASE_TOOLS,
            max_tokens=64,
        )

        result = asyncio.run(provider.generate(request))

        self.assertEqual(provider.last_path, "chat/completions")
        self.assertEqual(provider.last_headers["Authorization"], "Bearer test-key")
        self.assertEqual(provider.last_payload["max_completion_tokens"], 64)
        self.assertNotIn("max_tokens", provider.last_payload)
        self.assertEqual(provider.last_payload["tools"][0]["function"]["name"], "lookup_weather")
        self.assertEqual(provider.last_payload["messages"][2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(provider.last_payload["messages"][3]["tool_call_id"], "call_1")
        self.assertEqual(result.stop_reason, "tool_call")
        self.assertEqual(result.tool_calls[0].id, "call_openai_1")
        self.assertEqual(result.tool_calls[0].arguments["city"], "Paris")

    def test_anthropic_adapter_translates_blocks_and_response(self) -> None:
        provider = CaptureAnthropicProvider(
            {
                "id": "msg_123",
                "model": "claude-3-7-sonnet",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "I'll check that."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "lookup_weather",
                        "input": {"city": "Paris"},
                    },
                ],
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
            ProviderConfig(
                name="anthropic",
                model="claude-3-7-sonnet",
                settings={"api_key": "anthropic-key"},
            ),
        )
        request = ProviderRequest(model="claude-3-7-sonnet", messages=BASE_MESSAGES, tools=BASE_TOOLS)

        result = asyncio.run(provider.generate(request))

        self.assertEqual(provider.last_path, "messages")
        self.assertEqual(provider.last_headers["x-api-key"], "anthropic-key")
        self.assertEqual(provider.last_payload["system"], "You are concise.")
        self.assertEqual(provider.last_payload["tools"][0]["input_schema"]["type"], "object")
        assistant_blocks = provider.last_payload["messages"][1]["content"]
        self.assertEqual(assistant_blocks[1]["type"], "tool_use")
        merged_user_blocks = provider.last_payload["messages"][2]["content"]
        self.assertEqual(merged_user_blocks[0]["type"], "tool_result")
        self.assertEqual(merged_user_blocks[1]["type"], "text")
        self.assertEqual(result.stop_reason, "tool_call")
        self.assertEqual(result.tool_calls[0].id, "toolu_123")

    def test_anthropic_streaming_yields_text_and_tool_calls(self) -> None:
        provider = CaptureStreamingAnthropicProvider(
            [
                (
                    "message_start",
                    {
                        "message": {
                            "id": "msg_stream_123",
                            "model": "claude-3-7-sonnet",
                            "usage": {"input_tokens": 12},
                        }
                    },
                ),
                (
                    "content_block_delta",
                    {"index": 0, "delta": {"type": "text_delta", "text": "I'll check"}},
                ),
                (
                    "content_block_delta",
                    {"index": 0, "delta": {"type": "text_delta", "text": " that."}},
                ),
                (
                    "content_block_start",
                    {
                        "index": 1,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_stream_123",
                            "name": "lookup_weather",
                        },
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "index": 1,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"city":"Paris"}',
                        },
                    },
                ),
                (
                    "message_delta",
                    {
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"input_tokens": 12, "output_tokens": 8},
                    },
                ),
            ],
            ProviderConfig(
                name="anthropic",
                model="claude-3-7-sonnet",
                settings={"api_key": "anthropic-key"},
            ),
        )
        request = ProviderRequest(model="claude-3-7-sonnet", messages=BASE_MESSAGES, tools=BASE_TOOLS)

        async def _collect():
            items = []
            async for item in provider.stream_generate(request):
                items.append(item)
            return items

        items = asyncio.run(_collect())

        self.assertEqual(provider.last_path, "messages")
        self.assertEqual(provider.last_headers["x-api-key"], "anthropic-key")
        self.assertEqual(items[0].kind, "text_delta")
        self.assertEqual(items[0].text, "I'll check")
        self.assertEqual(items[1].text, " that.")
        final_result = items[-1].result
        self.assertIsNotNone(final_result)
        self.assertEqual(final_result.text, "I'll check that.")
        self.assertEqual(final_result.stop_reason, "tool_call")
        self.assertEqual(final_result.tool_calls[0].id, "toolu_stream_123")
        self.assertEqual(final_result.tool_calls[0].arguments["city"], "Paris")

    def test_gemini_adapter_translates_function_calls(self) -> None:
        provider = CaptureGeminiProvider(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "I'll look it up."},
                                {
                                    "functionCall": {
                                        "name": "lookup_weather",
                                        "args": {"city": "Paris"},
                                    }
                                },
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 14, "candidatesTokenCount": 6},
                "modelVersion": "gemini-2.5-flash",
            },
            ProviderConfig(
                name="gemini",
                model="gemini-2.5-flash",
                settings={"api_key": "gemini-key"},
            ),
        )
        request = ProviderRequest(model="gemini-2.5-flash", messages=BASE_MESSAGES, tools=BASE_TOOLS)

        result = asyncio.run(provider.generate(request))

        self.assertEqual(provider.last_path, "models/gemini-2.5-flash:generateContent")
        self.assertEqual(provider.last_headers["x-goog-api-key"], "gemini-key")
        self.assertEqual(provider.last_payload["systemInstruction"]["parts"][0]["text"], "You are concise.")
        self.assertEqual(
            provider.last_payload["tools"][0]["functionDeclarations"][0]["name"],
            "lookup_weather",
        )
        self.assertEqual(provider.last_payload["contents"][1]["parts"][1]["functionCall"]["name"], "lookup_weather")
        self.assertEqual(provider.last_payload["contents"][2]["role"], "tool")
        self.assertEqual(result.stop_reason, "tool_call")
        self.assertEqual(result.tool_calls[0].arguments["city"], "Paris")

    def test_ollama_adapter_translates_messages_and_response(self) -> None:
        provider = CaptureOllamaProvider(
            {
                "model": "llama3.1",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 18,
                "eval_count": 5,
                "message": {
                    "role": "assistant",
                    "content": "I'll check.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup_weather",
                                "arguments": {"city": "Paris"},
                            }
                        }
                    ],
                },
            },
            ProviderConfig(name="ollama", model="llama3.1"),
        )
        request = ProviderRequest(model="llama3.1", messages=BASE_MESSAGES, tools=BASE_TOOLS)

        result = asyncio.run(provider.generate(request))

        self.assertEqual(provider.last_path, "chat")
        self.assertFalse(provider.last_payload["stream"])
        self.assertEqual(provider.last_payload["tools"][0]["function"]["name"], "lookup_weather")
        self.assertEqual(provider.last_payload["messages"][2]["tool_calls"][0]["function"]["name"], "lookup_weather")
        self.assertEqual(provider.last_payload["messages"][3]["tool_name"], "lookup_weather")
        self.assertEqual(result.stop_reason, "tool_call")
        self.assertEqual(result.tool_calls[0].name, "lookup_weather")
        self.assertEqual(result.usage["eval_count"], 5)

    def test_lmstudio_uses_openai_compatible_shape(self) -> None:
        provider = CaptureLMStudioProvider(
            {
                "model": "local-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Local answer.",
                        },
                    }
                ],
            },
            ProviderConfig(name="lmstudio", model="local-model"),
        )
        request = ProviderRequest(
            model="local-model",
            messages=BASE_MESSAGES[:2],
            tools=BASE_TOOLS,
            max_tokens=32,
        )

        result = asyncio.run(provider.generate(request))

        self.assertEqual(provider.resolve_base_url(), "http://localhost:1234/v1")
        self.assertEqual(provider.last_path, "chat/completions")
        self.assertNotIn("Authorization", provider.last_headers)
        self.assertEqual(provider.last_payload["max_tokens"], 32)
        self.assertNotIn("max_completion_tokens", provider.last_payload)
        self.assertEqual(result.text, "Local answer.")
        self.assertEqual(result.stop_reason, "completed")

    def test_vllm_uses_openai_compatible_shape_and_extra_body(self) -> None:
        provider = CaptureVLLMProvider(
            {
                "model": "NousResearch/Meta-Llama-3-8B-Instruct",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "vLLM answer.",
                        },
                    }
                ],
            },
            ProviderConfig(
                name="vllm",
                model="NousResearch/Meta-Llama-3-8B-Instruct",
                settings={
                    "extra_body": {
                        "top_k": 40,
                        "parallel_tool_calls": False,
                    }
                },
            ),
        )
        request = ProviderRequest(
            model="NousResearch/Meta-Llama-3-8B-Instruct",
            messages=BASE_MESSAGES[:2],
            tools=BASE_TOOLS,
            max_tokens=48,
        )

        result = asyncio.run(provider.generate(request))

        self.assertEqual(provider.resolve_base_url(), "http://localhost:8000/v1")
        self.assertEqual(provider.last_path, "chat/completions")
        self.assertNotIn("Authorization", provider.last_headers)
        self.assertEqual(provider.last_payload["max_tokens"], 48)
        self.assertNotIn("max_completion_tokens", provider.last_payload)
        self.assertEqual(provider.last_payload["top_k"], 40)
        self.assertEqual(provider.last_payload["parallel_tool_calls"], False)
        self.assertEqual(result.text, "vLLM answer.")
        self.assertEqual(result.stop_reason, "completed")

    def test_http_provider_retries_and_sets_user_agent(self) -> None:
        provider = RetryableTestProvider(
            failures_before_success=2,
            config=ProviderConfig(
                name="retry-test",
                model="n/a",
                settings={
                    "request_retries": 3,
                    "request_retry_backoff_seconds": 0,
                },
            ),
        )

        response = asyncio.run(provider._request_json("POST", "test", payload={"ok": True}))

        self.assertEqual(response["ok"], True)
        self.assertEqual(provider.calls, 3)
        self.assertTrue(provider.last_headers["User-Agent"].startswith("agent-manager/"))
        self.assertTrue(OpenAIProvider(ProviderConfig(settings={"api_key": "x"})).capabilities.supports_streaming)

    def test_resource_exhaustion_classification_quota(self) -> None:
        provider = ErrorInspectingProvider(ProviderConfig(name="error-inspector", model="n/a"))
        error = provider._http_error_to_provider_error(
            429,
            '{"error": {"type": "insufficient_quota", "message": "You exceeded your current quota."}}',
            {"Retry-After": "60"},
        )
        self.assertIsInstance(error, ProviderResourceExhaustedError)
        self.assertEqual(error.kind, "quota_exhausted")
        self.assertEqual(error.retry_after_seconds, 60.0)
        self.assertEqual(error.status_code, 429)
        self.assertFalse(error.retryable)

    def test_resource_exhaustion_classification_rate_limit(self) -> None:
        provider = ErrorInspectingProvider(ProviderConfig(name="error-inspector", model="n/a"))
        error = provider._http_error_to_provider_error(
            429,
            '{"error": {"type": "rate_limit_exceeded", "message": "Too many requests."}}',
            None,
        )
        self.assertIsInstance(error, ProviderResourceExhaustedError)
        self.assertEqual(error.kind, "rate_limited")

    def test_resource_exhaustion_classification_capacity(self) -> None:
        provider = ErrorInspectingProvider(ProviderConfig(name="error-inspector", model="n/a"))
        error = provider._http_error_to_provider_error(
            529,
            '{"error": {"type": "overloaded", "message": "Model is overloaded."}}',
            None,
        )
        self.assertIsInstance(error, ProviderResourceExhaustedError)
        self.assertEqual(error.kind, "capacity_exhausted")

    def test_generic_error_is_not_resource_exhausted(self) -> None:
        provider = ErrorInspectingProvider(ProviderConfig(name="error-inspector", model="n/a"))
        error = provider._http_error_to_provider_error(
            500,
            '{"error": {"message": "Internal server error"}}',
            None,
        )
        self.assertIsInstance(error, ProviderRequestError)
        self.assertNotIsInstance(error, ProviderResourceExhaustedError)
        self.assertTrue(error.retryable)

    def test_resource_exhaustion_not_retried_by_http_layer(self) -> None:
        """Resource-exhausted errors must bypass the retry loop."""

        class QuotaExhaustedProvider(HTTPProvider):
            provider_name = "quota-test"
            default_base_url = "https://example.test"
            calls = 0

            async def generate(self, request: ProviderRequest):
                raise NotImplementedError

            def _request_json_blocking(self, method, url, payload, headers, timeout):
                self.calls += 1
                raise ProviderResourceExhaustedError(
                    "quota hit",
                    provider="quota-test",
                    kind="quota_exhausted",
                    status_code=429,
                )

        provider = QuotaExhaustedProvider(
            ProviderConfig(
                name="quota-test",
                model="n/a",
                settings={"request_retries": 3, "request_retry_backoff_seconds": 0},
            )
        )
        with self.assertRaises(ProviderResourceExhaustedError):
            asyncio.run(provider._request_json("POST", "test"))
        # Should only be called once - no retries for resource exhaustion.
        self.assertEqual(provider.calls, 1)

    def test_resource_exhaustion_to_dict(self) -> None:
        error = ProviderResourceExhaustedError(
            "Quota exceeded.",
            provider="openai",
            kind="quota_exhausted",
            status_code=429,
            retry_after_seconds=30.0,
            metadata={"code": "insufficient_quota"},
        )
        d = error.to_dict()
        self.assertEqual(d["provider"], "openai")
        self.assertEqual(d["kind"], "quota_exhausted")
        self.assertEqual(d["status_code"], 429)
        self.assertEqual(d["retry_after_seconds"], 30.0)
        self.assertEqual(d["metadata"]["code"], "insufficient_quota")
        self.assertIn("Quota exceeded", d["message"])


if __name__ == "__main__":
    unittest.main()
