"""Context preparation result types."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_manager.types import ContextSection, Message


@dataclass(slots=True)
class PreparedTurn:
    messages: list[Message]
    sections: list[ContextSection]
    token_estimate: int
    dropped_sections: list[str] = field(default_factory=list)

