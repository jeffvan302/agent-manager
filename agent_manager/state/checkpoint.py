"""Checkpoint manager wrapper."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_manager.state.models import CheckpointRecord
from agent_manager.state.store import StateStore
from agent_manager.types import LoopState


class CheckpointManager:
    """Small wrapper around a state store to simplify runtime usage."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def save(self, state: LoopState) -> CheckpointRecord:
        self.store.save(state)
        return CheckpointRecord(
            task_id=state.task_id,
            saved_at=datetime.now(timezone.utc).isoformat(),
            state=state,
        )

    def load(self, task_id: str) -> LoopState | None:
        return self.store.load(task_id)

