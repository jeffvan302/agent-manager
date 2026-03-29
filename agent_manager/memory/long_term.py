"""Long-term memory implementations with persistence."""

from __future__ import annotations

import json
import re
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from agent_manager.memory.base import BaseMemoryStore, MemoryEntry


class InMemoryLongTermStore(BaseMemoryStore):
    """In-memory long-term store for testing and ephemeral use."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def put(self, entry: MemoryEntry) -> None:
        # Upsert: replace if same key exists.
        self._entries = [e for e in self._entries if e.key != entry.key]
        self._entries.append(entry)

    def query(
        self,
        text: str,
        *,
        scope: str | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        query_text = text.lower()
        results: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            if not self._matches_filters(entry, scope=scope, tags=tags, min_confidence=min_confidence):
                continue
            score = self._score(query_text, entry)
            if score > 0:
                results.append((score, entry))
        results.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    @staticmethod
    def _matches_filters(
        entry: MemoryEntry,
        *,
        scope: str | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> bool:
        if scope and entry.scope != scope:
            return False
        if tags and not set(tags).intersection(entry.tags):
            return False
        if entry.confidence < min_confidence:
            return False
        return True

    @staticmethod
    def _score(query_text: str, entry: MemoryEntry) -> float:
        """Simple keyword overlap scoring."""
        query_words = set(query_text.split())
        entry_words = set(entry.key.lower().split()) | set(entry.value.lower().split())
        overlap = query_words & entry_words
        if not overlap:
            # Check substring match as fallback.
            if query_text in entry.key.lower() or query_text in entry.value.lower():
                return 0.5
            return 0.0
        return len(overlap) / max(len(query_words), 1)


class JsonFileLongTermStore(BaseMemoryStore):
    """Persistent long-term memory backed by a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._entries = [
                    MemoryEntry(**item) for item in data if isinstance(item, dict)
                ]
            except (json.JSONDecodeError, OSError, TypeError):
                self._entries = []

    def _save(self) -> None:
        """Atomic write: write to temp file then rename."""
        tmp_path = self.path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(
                    [e.to_dict() for e in self._entries],
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            tmp_path.replace(self.path)
        except OSError:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def put(self, entry: MemoryEntry) -> None:
        self._entries = [e for e in self._entries if e.key != entry.key]
        self._entries.append(entry)
        self._save()

    def query(
        self,
        text: str,
        *,
        scope: str | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        query_text = text.lower()
        results: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            if not InMemoryLongTermStore._matches_filters(
                entry, scope=scope, tags=tags, min_confidence=min_confidence
            ):
                continue
            score = InMemoryLongTermStore._score(query_text, entry)
            if score > 0:
                results.append((score, entry))
        results.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)


class SqliteLongTermStore(BaseMemoryStore):
    """Persistent long-term memory backed by SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'runtime',
                        timestamp TEXT NOT NULL,
                        scope TEXT NOT NULL DEFAULT 'task',
                        confidence REAL NOT NULL DEFAULT 1.0,
                        tags TEXT NOT NULL DEFAULT '[]',
                        metadata TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )

    def put(self, entry: MemoryEntry) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO memory (key, value, source, timestamp, scope, confidence, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        source = excluded.source,
                        timestamp = excluded.timestamp,
                        scope = excluded.scope,
                        confidence = excluded.confidence,
                        tags = excluded.tags,
                        metadata = excluded.metadata
                    """,
                    (
                        entry.key,
                        entry.value,
                        entry.source,
                        entry.timestamp,
                        entry.scope,
                        entry.confidence,
                        json.dumps(entry.tags),
                        json.dumps(entry.metadata),
                    ),
                )

    def query(
        self,
        text: str,
        *,
        scope: str | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        conditions = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if scope:
            conditions.append("scope = ?")
            params.append(scope)

        where_clause = " AND ".join(conditions)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT key, value, source, timestamp, scope, confidence, tags, metadata "
                f"FROM memory WHERE {where_clause}",
                params,
            ).fetchall()

        entries = [self._row_to_entry(row) for row in rows]

        # Filter by tags in Python (JSON array not easily queried in SQL).
        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if tag_set.intersection(e.tags)]

        # Score and rank.
        query_text = text.lower()
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in entries:
            score = InMemoryLongTermStore._score(query_text, entry)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def all_entries(self) -> list[MemoryEntry]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT key, value, source, timestamp, scope, confidence, tags, metadata FROM memory"
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    @staticmethod
    def _row_to_entry(row: tuple) -> MemoryEntry:
        return MemoryEntry(
            key=row[0],
            value=row[1],
            source=row[2],
            timestamp=row[3],
            scope=row[4],
            confidence=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            metadata=json.loads(row[7]) if row[7] else {},
        )
