"""Base provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent_manager.config import ProviderConfig
from agent_manager.types import ProviderRequest, ProviderResult


@dataclass(slots=True)
class ProviderCapabilities:
    supports_tools: bool = False
    supports_streaming: bool = False
    supports_structured_output: bool = False
    supports_images: bool = False
    supports_system_messages: bool = True


class BaseProvider(ABC):
    """Normalized provider interface used by the runtime loop."""

    provider_name = "base"
    capabilities = ProviderCapabilities()

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig()

    @abstractmethod
    async def generate(self, request: ProviderRequest) -> ProviderResult:
        """Run one normalized generation request."""
