"""Serializable state-related models."""

from __future__ import annotations

from dataclasses import dataclass

from agent_manager.types import LoopState


@dataclass(slots=True)
class CheckpointRecord:
    task_id: str
    saved_at: str
    state: LoopState

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "saved_at": self.saved_at,
            "state": self.state.to_dict(),
        }
