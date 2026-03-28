"""Lightweight token budget helpers for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

from agent_manager.types import Message


@dataclass(slots=True)
class TokenBudget:
    max_context_tokens: int
    reserved_output_tokens: int = 0

    @property
    def available_input_tokens(self) -> int:
        available = self.max_context_tokens - self.reserved_output_tokens
        return max(available, 0)


class SimpleTokenCounter:
    """Approximate token counting until model-specific counters are added."""

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def count_message(self, message: Message) -> int:
        return self.estimate_text(message.content) + 4

    def count_messages(self, messages: list[Message]) -> int:
        return sum(self.count_message(message) for message in messages)

