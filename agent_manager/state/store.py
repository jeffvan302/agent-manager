"""State store implementations."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from pathlib import Path

from agent_manager.errors import CheckpointError
from agent_manager.types import LoopState


class StateStore(ABC):
    @abstractmethod
    def load(self, task_id: str) -> LoopState | None:
        """Load loop state for a task id."""

    @abstractmethod
    def save(self, state: LoopState) -> None:
        """Persist loop state."""


class InMemoryStateStore(StateStore):
    def __init__(self) -> None:
        self._states: dict[str, LoopState] = {}

    def load(self, task_id: str) -> LoopState | None:
        return self._states.get(task_id)

    def save(self, state: LoopState) -> None:
        self._states[state.task_id] = state


class JsonFileStateStore(StateStore):
    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path_for(self, task_id: str) -> Path:
        safe_task_id = self._safe_task_id(task_id)
        return self.base_path / f"{safe_task_id}.json"

    def _safe_task_id(self, task_id: str) -> str:
        trimmed = task_id.strip() or "task"
        normalized = re.sub(r"[^A-Za-z0-9._-]", "_", trimmed)
        normalized = normalized.strip("._") or "task"
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
        return f"{normalized[:80]}-{digest}"

    def load(self, task_id: str) -> LoopState | None:
        path = self._path_for(task_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CheckpointError(f"Failed to read checkpoint {path}") from exc
        return LoopState.from_dict(payload)

    def save(self, state: LoopState) -> None:
        path = self._path_for(state.task_id)
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(state.to_dict(), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except OSError as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise CheckpointError(f"Failed to write checkpoint {path}") from exc


class SqliteStateStore(StateStore):
    """Persist checkpoints in a local SQLite database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as conn:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS checkpoints (
                            task_id TEXT PRIMARY KEY,
                            saved_at TEXT NOT NULL,
                            payload TEXT NOT NULL
                        )
                        """
                    )
        except sqlite3.Error as exc:
            raise CheckpointError(f"Failed to initialize checkpoint database {self.path}") from exc

    def load(self, task_id: str) -> LoopState | None:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT payload FROM checkpoints WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise CheckpointError(f"Failed to read checkpoint from {self.path}") from exc
        if row is None:
            return None
        payload = json.loads(str(row[0]))
        return LoopState.from_dict(payload)

    def save(self, state: LoopState) -> None:
        saved_at = state.checkpoint_timestamps[-1] if state.checkpoint_timestamps else ""
        payload = json.dumps(state.to_dict(), ensure_ascii=True)
        try:
            with closing(self._connect()) as conn:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO checkpoints (task_id, saved_at, payload)
                        VALUES (?, ?, ?)
                        ON CONFLICT(task_id) DO UPDATE SET
                            saved_at = excluded.saved_at,
                            payload = excluded.payload
                        """,
                        (state.task_id, saved_at, payload),
                    )
        except sqlite3.Error as exc:
            raise CheckpointError(f"Failed to write checkpoint into {self.path}") from exc
