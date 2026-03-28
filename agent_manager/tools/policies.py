"""Policy layer for controlling tool usage by runtime profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_manager.errors import PolicyViolationError
from agent_manager.tools.base import ToolSpec


@dataclass(slots=True)
class ToolPolicyProfile:
    name: str
    allow_all: bool = False
    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    denied_tags: set[str] = field(default_factory=set)
    allowed_permissions: set[str] = field(default_factory=set)
    denied_permissions: set[str] = field(default_factory=set)


DEFAULT_PROFILES: dict[str, ToolPolicyProfile] = {
    "readonly": ToolPolicyProfile(
        name="readonly",
        allow_all=True,
        denied_tags={"write", "shell", "network"},
        denied_permissions={"filesystem:write", "process:execute", "network:request"},
    ),
    "local-dev": ToolPolicyProfile(name="local-dev", allow_all=True),
    "coding-agent": ToolPolicyProfile(name="coding-agent", allow_all=True),
    "unrestricted-lab": ToolPolicyProfile(name="unrestricted-lab", allow_all=True),
}


class PolicyEngine:
    """Validate tool calls against a named policy profile."""

    def __init__(self, profile: str | ToolPolicyProfile = "readonly") -> None:
        if isinstance(profile, ToolPolicyProfile):
            self.profile = profile
        else:
            self.profile = DEFAULT_PROFILES.get(
                profile,
                ToolPolicyProfile(name=profile, allow_all=True),
            )

    def assert_allowed(
        self,
        spec: ToolSpec,
        *,
        context: Any | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        del context, arguments
        if spec.name in self.profile.denied_tools:
            raise PolicyViolationError(
                f"Tool '{spec.name}' is blocked by profile '{self.profile.name}'."
            )

        blocked_tags = set(spec.tags) & self.profile.denied_tags
        if blocked_tags:
            raise PolicyViolationError(
                f"Tool '{spec.name}' uses blocked tags {sorted(blocked_tags)} "
                f"under profile '{self.profile.name}'."
            )

        blocked_permissions = set(spec.permissions) & self.profile.denied_permissions
        if blocked_permissions:
            raise PolicyViolationError(
                f"Tool '{spec.name}' requires blocked permissions "
                f"{sorted(blocked_permissions)} under profile '{self.profile.name}'."
            )

        if self.profile.allow_all:
            return

        if self.profile.allowed_permissions and not (
            set(spec.permissions) & self.profile.allowed_permissions
        ):
            raise PolicyViolationError(
                f"Tool '{spec.name}' does not have an allowed permission under "
                f"profile '{self.profile.name}'."
            )

        if spec.name not in self.profile.allowed_tools:
            raise PolicyViolationError(
                f"Tool '{spec.name}' is not allowed under profile '{self.profile.name}'."
            )
