"""Serializable service-layer request and response shapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RunRequest:
    prompt: str
    task_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RunResponse:
    task_id: str
    output_text: str
    status: str
    stop_reason: str

    def to_dict(self) -> dict:
        return asdict(self)

