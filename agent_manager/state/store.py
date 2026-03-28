"""State store implementations."""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
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
        try:
            path.write_text(
                json.dumps(state.to_dict(), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise CheckpointError(f"Failed to write checkpoint {path}") from exc
