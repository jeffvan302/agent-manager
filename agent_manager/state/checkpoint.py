"""Checkpoint manager wrapper."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_manager.observability import emitter, timed
from agent_manager.state.models import CheckpointRecord
from agent_manager.state.store import StateStore
from agent_manager.types import LoopState


class CheckpointManager:
    """Small wrapper around a state store to simplify runtime usage."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def save(self, state: LoopState) -> CheckpointRecord:
        saved_at = datetime.now(timezone.utc).isoformat()
        if not state.checkpoint_timestamps or state.checkpoint_timestamps[-1] != saved_at:
            state.checkpoint_timestamps.append(saved_at)
        with timed() as timing:
            self.store.save(state)
        emitter.checkpoint_save(
            task_id=state.task_id,
            step=state.step_index,
            duration_ms=timing["duration_ms"],
        )
        return CheckpointRecord(task_id=state.task_id, saved_at=saved_at, state=state)

    def load(self, task_id: str) -> LoopState | None:
        state = self.store.load(task_id)
        if state is not None:
            emitter.checkpoint_load(task_id=task_id, step=state.step_index)
        return state
