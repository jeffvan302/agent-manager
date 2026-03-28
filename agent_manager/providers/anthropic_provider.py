"""Anthropic provider adapter placeholder for Phase 2."""

from __future__ import annotations

from agent_manager.providers.base import BaseProvider
from agent_manager.types import ProviderRequest, ProviderResult


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        raise NotImplementedError("AnthropicProvider is planned for Phase 2.")
