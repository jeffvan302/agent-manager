"""Base memory contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryEntry:
    key: str
    value: str
    source: str = "runtime"
    scope: str = "task"
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseMemoryStore(ABC):
    @abstractmethod
    def put(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""

    @abstractmethod
    def query(self, text: str) -> list[MemoryEntry]:
        """Return entries relevant to the query."""

