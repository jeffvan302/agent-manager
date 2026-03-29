"""Lightweight token budget helpers for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

from agent_manager.config import RuntimeConfig
from agent_manager.types import Message


@dataclass(slots=True)
class TokenBudget:
    max_context_tokens: int
    reserved_output_tokens: int = 0

    @property
    def available_input_tokens(self) -> int:
        available = self.max_context_tokens - self.reserved_output_tokens
        return max(available, 0)


@dataclass(slots=True)
class ModelBudgetProfile:
    provider: str
    model: str
    max_context_tokens: int
    max_output_tokens: int
    chars_per_token: float = 4.0


KNOWN_MODEL_BUDGETS: dict[str, dict[str, tuple[int, int, float]]] = {
    "openai": {
        "gpt-4o-mini": (128000, 16384, 4.0),
        "gpt-4.1-mini": (128000, 16384, 4.0),
        "gpt-4.1": (1047576, 32768, 4.0),
    },
    "anthropic": {
        "claude-3-5-sonnet-latest": (200000, 8192, 4.0),
        "claude-3-7-sonnet-latest": (200000, 8192, 4.0),
    },
    "gemini": {
        "gemini-1.5-pro": (1048576, 8192, 4.0),
        "gemini-1.5-flash": (1048576, 8192, 4.0),
    },
    "ollama": {},
    "lmstudio": {},
    "echo": {},
}


def resolve_model_budget(config: RuntimeConfig) -> ModelBudgetProfile:
    provider = config.provider.name.strip().lower()
    model = config.provider.model.strip()
    model_key = model.lower()
    known_context, known_output, known_ratio = KNOWN_MODEL_BUDGETS.get(provider, {}).get(
        model_key,
        (
            config.runtime.max_context_tokens,
            config.runtime.max_output_tokens,
            4.0,
        ),
    )
    return ModelBudgetProfile(
        provider=provider,
        model=model,
        max_context_tokens=config.provider.resolved_max_context_tokens(known_context),
        max_output_tokens=config.provider.resolved_max_output_tokens(known_output),
        chars_per_token=config.provider.resolved_token_counter_chars_per_token(known_ratio),
    )


class SimpleTokenCounter:
    """Approximate token counting with provider/model-aware heuristics."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.chars_per_token = max(chars_per_token, 1.0)

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def count_message(self, message: Message) -> int:
        return self.estimate_text(message.content) + 4

    def count_messages(self, messages: list[Message]) -> int:
        return sum(self.count_message(message) for message in messages)
