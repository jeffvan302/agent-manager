"""vLLM provider adapter using the OpenAI-compatible API surface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_manager.providers.openai_provider import OpenAICompatibleChatProvider
from agent_manager.types import ProviderRequest


class VLLMProvider(OpenAICompatibleChatProvider):
    """First-class vLLM adapter with local-server defaults.

    vLLM exposes an OpenAI-compatible HTTP API, so this provider reuses the
    shared OpenAI chat-completions adapter while supplying vLLM-specific
    defaults and optional request-body passthrough fields.
    """

    provider_name = "vllm"
    default_base_url = "http://localhost:8000/v1"
    default_api_key_env = "VLLM_API_KEY"
    requires_api_key = False

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        payload = super()._build_payload(request)
        extra_body = self.config.settings.get("extra_body")
        if isinstance(extra_body, Mapping):
            payload.update({str(key): value for key, value in extra_body.items()})
        return payload
