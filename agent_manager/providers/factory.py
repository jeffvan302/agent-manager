"""Provider factory and registry."""

from __future__ import annotations

from typing import Type

from agent_manager.config import ProviderConfig
from agent_manager.errors import ProviderNotFoundError
from agent_manager.providers.anthropic_provider import AnthropicProvider
from agent_manager.providers.base import BaseProvider
from agent_manager.providers.echo_provider import EchoProvider
from agent_manager.providers.gemini_provider import GeminiProvider
from agent_manager.providers.lmstudio_provider import LMStudioProvider
from agent_manager.providers.ollama_provider import OllamaProvider
from agent_manager.providers.openai_provider import OpenAIProvider
from agent_manager.providers.vllm_provider import VLLMProvider

ProviderType = Type[BaseProvider]

_PROVIDER_REGISTRY: dict[str, ProviderType] = {
    AnthropicProvider.provider_name: AnthropicProvider,
    EchoProvider.provider_name: EchoProvider,
    GeminiProvider.provider_name: GeminiProvider,
    LMStudioProvider.provider_name: LMStudioProvider,
    OllamaProvider.provider_name: OllamaProvider,
    OpenAIProvider.provider_name: OpenAIProvider,
    VLLMProvider.provider_name: VLLMProvider,
}


def register_provider(name: str, provider_cls: ProviderType) -> None:
    _PROVIDER_REGISTRY[name.lower()] = provider_cls


def available_providers() -> list[str]:
    return sorted(_PROVIDER_REGISTRY.keys())


def build_provider(config: ProviderConfig | str | None = None, **settings) -> BaseProvider:
    if config is None:
        provider_config = ProviderConfig()
    elif isinstance(config, str):
        provider_config = ProviderConfig(name=config, settings=dict(settings))
    else:
        provider_config = config
        if settings:
            provider_config.settings.update(settings)

    provider_name = provider_config.name.lower()
    provider_cls = _PROVIDER_REGISTRY.get(provider_name)
    if provider_cls is None:
        raise ProviderNotFoundError(
            f"Provider '{provider_config.name}' is not registered. "
            f"Available providers: {', '.join(available_providers())}"
        )
    return provider_cls(provider_config)
