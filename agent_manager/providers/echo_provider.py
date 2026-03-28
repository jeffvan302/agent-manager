"""Minimal local provider used for Phase 1 smoke tests."""

from __future__ import annotations

from agent_manager.providers.base import BaseProvider, ProviderCapabilities
from agent_manager.types import ProviderRequest, ProviderResult


class EchoProvider(BaseProvider):
    """Return the latest user message as the provider response."""

    provider_name = "echo"
    capabilities = ProviderCapabilities(
        supports_tools=False,
        supports_streaming=False,
        supports_structured_output=False,
    )

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        latest_user_message = ""
        for message in reversed(request.messages):
            if message.role == "user":
                latest_user_message = message.content
                break

        if not latest_user_message:
            latest_user_message = "No user message was supplied."

        return ProviderResult(
            text=latest_user_message,
            stop_reason="completed",
            usage={
                "input_messages": len(request.messages),
                "model": request.model,
            },
            metadata={"provider": self.provider_name},
        )
