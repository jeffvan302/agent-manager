"""Base tool interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 60
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolContext:
    task_id: str
    step_index: int
    tool_call_id: str | None = None
    working_directory: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    ok: bool
    output: dict[str, Any] | str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseTool(ABC):
    """Common base class for runtime-managed tools."""

    spec: ToolSpec

    @abstractmethod
    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool with normalized arguments."""
