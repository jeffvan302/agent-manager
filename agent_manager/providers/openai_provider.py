"""OpenAI provider adapter placeholder for Phase 2."""

from __future__ import annotations

from agent_manager.providers.base import BaseProvider
from agent_manager.types import ProviderRequest, ProviderResult


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        raise NotImplementedError("OpenAIProvider is planned for Phase 2.")
