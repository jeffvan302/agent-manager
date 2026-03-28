"""LM Studio provider adapter using the OpenAI-compatible API surface."""

from __future__ import annotations

from agent_manager.providers.openai_provider import OpenAICompatibleChatProvider


class LMStudioProvider(OpenAICompatibleChatProvider):
    provider_name = "lmstudio"
    default_base_url = "http://localhost:1234/v1"
    default_api_key_env = "LMSTUDIO_API_KEY"
    requires_api_key = False

