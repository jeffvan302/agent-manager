"""Short-term memory helpers."""

from __future__ import annotations

from agent_manager.memory.base import MemoryEntry


class ShortTermMemory:
    """Simple in-memory container for the active task."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def recent(self, limit: int = 10) -> list[MemoryEntry]:
        return self._entries[-limit:]

