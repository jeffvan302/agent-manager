"""Context preparation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_manager.types import ContextSection, Message


@dataclass(slots=True)
class PreparedTurn:
    messages: list[Message] = field(default_factory=list)
    sections: list[ContextSection] = field(default_factory=list)
    token_estimate: int = 0
    dropped_sections: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
