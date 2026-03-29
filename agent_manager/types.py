"""Shared serializable types used across the runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Message:
    role: str
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallRequest":
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=dict(data.get("arguments", {})),
        )


@dataclass(slots=True)
class ContextHint:
    key: str
    reason: str
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextHint":
        return cls(
            key=data["key"],
            reason=data["reason"],
            priority=int(data.get("priority", 0)),
        )


@dataclass(slots=True)
class ContextSection:
    key: str
    title: str
    content: str
    priority: int = 0
    token_estimate: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuredOutputSpec:
    type: str = "json_object"
    name: str | None = None
    schema: dict[str, Any] | None = None
    strict: bool = True
    prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StructuredOutputSpec | None":
        if data is None:
            return None
        return cls(
            type=str(data.get("type", "json_object")),
            name=data.get("name"),
            schema=dict(data.get("schema", {})) if isinstance(data.get("schema"), dict) else None,
            strict=bool(data.get("strict", True)),
            prompt=data.get("prompt"),
        )


@dataclass(slots=True)
class ProviderRequest:
    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    structured_output: StructuredOutputSpec | None = None
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [message.to_dict() for message in self.messages],
            "tools": list(self.tools),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "structured_output": (
                self.structured_output.to_dict() if self.structured_output is not None else None
            ),
            "stream": self.stream,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ProviderResult:
    text: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw: Any = None
    context_hints: list[ContextHint] = field(default_factory=list)
    structured_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "raw": self.raw,
            "context_hints": [hint.to_dict() for hint in self.context_hints],
            "structured_output": self.structured_output,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ProviderStreamEvent:
    kind: str
    text: str | None = None
    result: ProviderResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "result": self.result.to_dict() if self.result is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class LoopState:
    task_id: str
    goal: str
    step_index: int = 0
    messages: list[Message] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    step_history: list[dict[str, Any]] = field(default_factory=list)
    pending_subgoals: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_timestamps: list[str] = field(default_factory=list)
    structured_output_spec: StructuredOutputSpec | None = None
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "step_index": self.step_index,
            "messages": [message.to_dict() for message in self.messages],
            "summaries": list(self.summaries),
            "tool_observations": list(self.tool_observations),
            "step_history": list(self.step_history),
            "pending_subgoals": list(self.pending_subgoals),
            "errors": list(self.errors),
            "checkpoint_timestamps": list(self.checkpoint_timestamps),
            "structured_output_spec": (
                self.structured_output_spec.to_dict()
                if self.structured_output_spec is not None
                else None
            ),
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopState":
        return cls(
            task_id=data["task_id"],
            goal=data["goal"],
            step_index=int(data.get("step_index", 0)),
            messages=[Message.from_dict(item) for item in data.get("messages", [])],
            summaries=list(data.get("summaries", [])),
            tool_observations=list(data.get("tool_observations", [])),
            step_history=list(data.get("step_history", [])),
            pending_subgoals=list(data.get("pending_subgoals", [])),
            errors=list(data.get("errors", [])),
            checkpoint_timestamps=list(data.get("checkpoint_timestamps", [])),
            structured_output_spec=StructuredOutputSpec.from_dict(
                data.get("structured_output_spec")
            ),
            status=data.get("status", "running"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class AgentRunResult:
    task_id: str
    output_text: str
    state: LoopState
    stop_reason: str
    usage: dict[str, Any] | None = None
    structured_output: Any = None
    resource_exhaustion: dict[str, Any] | None = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "output_text": self.output_text,
            "state": self.state.to_dict(),
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "structured_output": self.structured_output,
            "resource_exhaustion": self.resource_exhaustion,
            "tool_results": list(self.tool_results),
            "events": list(self.events),
        }
