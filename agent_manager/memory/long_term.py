"""Long-term memory placeholder implementation."""

from __future__ import annotations

from agent_manager.memory.base import BaseMemoryStore, MemoryEntry


class InMemoryLongTermStore(BaseMemoryStore):
    """Placeholder long-term store used until persistence arrives."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def put(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def query(self, text: str) -> list[MemoryEntry]:
        query_text = text.lower()
        return [
            entry
            for entry in self._entries
            if query_text in entry.key.lower() or query_text in entry.value.lower()
        ]

