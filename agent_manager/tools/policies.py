"""Policy layer for controlling tool usage by runtime profile."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_manager.errors import PolicyViolationError
from agent_manager.tools.base import ToolSpec

ApprovalHook = Callable[[ToolSpec, Any | None, dict[str, Any] | None], "ApprovalDecision | bool | str"]


@dataclass(slots=True)
class ApprovalDecision:
    approved: bool
    reason: str | None = None


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

DEFAULT_APPROVAL_TAGS = {"write", "shell", "network", "database"}
DEFAULT_APPROVAL_PERMISSIONS = {
    "filesystem:write",
    "process:execute",
    "network:request",
    "database:query",
    "database:write",
}


class PolicyEngine:
    """Validate tool calls against a named policy profile."""

    def __init__(
        self,
        profile: str | ToolPolicyProfile = "readonly",
        *,
        approval_hook: ApprovalHook | None = None,
        approval_tags: set[str] | None = None,
        approval_permissions: set[str] | None = None,
    ) -> None:
        if isinstance(profile, ToolPolicyProfile):
            self.profile = profile
        else:
            self.profile = DEFAULT_PROFILES.get(
                profile,
                ToolPolicyProfile(name=profile, allow_all=True),
            )
        self.approval_hook = approval_hook
        self.approval_tags = set(approval_tags or DEFAULT_APPROVAL_TAGS)
        self.approval_permissions = set(
            approval_permissions or DEFAULT_APPROVAL_PERMISSIONS
        )

    def assert_allowed(
        self,
        spec: ToolSpec,
        *,
        context: Any | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
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

        if self.approval_hook is not None and self._requires_approval(spec):
            decision = self.approval_hook(spec, context, arguments)
            if isinstance(decision, ApprovalDecision):
                approved = decision.approved
                reason = decision.reason
            elif isinstance(decision, bool):
                approved = decision
                reason = None
            else:
                approved = False
                reason = str(decision)
            if not approved:
                raise PolicyViolationError(
                    reason
                    or f"Tool '{spec.name}' requires approval under profile '{self.profile.name}'."
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

    def _requires_approval(self, spec: ToolSpec) -> bool:
        if set(spec.tags) & self.approval_tags:
            return True
        if set(spec.permissions) & self.approval_permissions:
            return True
        return False
