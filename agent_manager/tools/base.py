"""Base tool interfaces."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

ToolOutput = dict[str, Any] | str
ToolHandler = Callable[
    [dict[str, Any], "ToolContext"],
    "ToolResult | ToolOutput | Awaitable[ToolResult | ToolOutput]",
]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    retry_count: int = 0
    retry_backoff_seconds: float = 0.0

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


def normalize_tool_result(
    tool_name: str,
    value: ToolResult | ToolOutput,
    *,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    """Normalize raw tool return values into a ToolResult."""

    if isinstance(value, ToolResult):
        if metadata:
            merged_metadata = dict(metadata)
            merged_metadata.update(value.metadata)
            value.metadata = merged_metadata
        return value

    normalized_output: ToolOutput
    if isinstance(value, (str, dict)):
        normalized_output = value
    else:
        normalized_output = {"value": value}

    return ToolResult(
        tool_name=tool_name,
        ok=True,
        output=normalized_output,
        metadata=dict(metadata or {}),
    )


class FunctionTool(BaseTool):
    """Adapter for exposing plain Python callables as runtime tools."""

    def __init__(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self.spec = spec
        self._handler = handler

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        value = self._handler(arguments, context)
        if inspect.isawaitable(value):
            value = await value
        return normalize_tool_result(self.spec.name, value)
