"""Serializable service-layer request and response shapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class RunRequest:
    prompt: str
    task_id: str | None = None
    structured_output: dict[str, Any] | None = None
    stream: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RunResponse:
    task_id: str
    output_text: str
    status: str
    stop_reason: str
    structured_output: Any = None
    resource_exhaustion: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)
