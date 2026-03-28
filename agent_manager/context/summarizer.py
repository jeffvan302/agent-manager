"""Simple summary helpers used by the context pipeline."""

from __future__ import annotations

from agent_manager.types import Message


class SimpleSummarizer:
    """Very small placeholder summarizer until model-backed summaries exist."""

    def summarize_messages(self, messages: list[Message], limit: int = 4) -> str:
        trimmed = messages[-limit:]
        if not trimmed:
            return ""
        fragments = [f"{message.role}: {message.content}" for message in trimmed]
        return "\n".join(fragments)

