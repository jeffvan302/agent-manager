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
class ProviderRequest:
    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [message.to_dict() for message in self.messages],
            "tools": list(self.tools),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "raw": self.raw,
            "context_hints": [hint.to_dict() for hint in self.context_hints],
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
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "output_text": self.output_text,
            "state": self.state.to_dict(),
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "tool_results": list(self.tool_results),
            "events": list(self.events),
        }

