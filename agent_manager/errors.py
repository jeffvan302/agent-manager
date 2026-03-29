"""Shared error types for agent_manager."""

from __future__ import annotations

from typing import Any


class AgentManagerError(Exception):
    """Base package error."""


class ConfigurationError(AgentManagerError):
    """Raised when configuration cannot be loaded or validated."""


class ProviderError(AgentManagerError):
    """Base provider error."""


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered."""


class ProviderRequestError(ProviderError):
    """Raised when a provider request fails."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderResourceExhaustedError(ProviderRequestError):
    """Raised when a provider reports exhausted quota, capacity, or rate limits.

    This is a structured error distinct from transient failures.  It carries
    metadata so callers can decide whether to wait, switch providers, or
    surface the condition to the user.

    Attributes:
        provider: Name of the provider that raised the error.
        kind: Classification of the exhaustion type.  One of:
            ``"quota_exhausted"`` - API-key credits / billing limit reached.
            ``"rate_limited"`` - too many requests in a time window.
            ``"capacity_exhausted"`` - model or server overloaded.
            ``"resource_exhausted"`` - catch-all for other resource limits.
        status_code: The HTTP status code returned by the provider, if any.
        retry_after_seconds: Seconds the caller should wait before retrying,
            taken from the provider's ``Retry-After`` header when available.
        metadata: Arbitrary provider-specific details (e.g. remaining quota,
            reset timestamp, error codes).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        kind: str = "resource_exhausted",
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Resource exhaustion is NOT retryable by the automatic retry loop;
        # the caller should handle it explicitly.
        super().__init__(message, retryable=False)
        self.provider = provider
        self.kind = kind
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for structured logging and state storage."""
        return {
            "provider": self.provider,
            "kind": self.kind,
            "status_code": self.status_code,
            "retry_after_seconds": self.retry_after_seconds,
            "message": str(self),
            "metadata": dict(self.metadata),
        }


class ToolError(AgentManagerError):
    """Base tool error."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""


class PolicyViolationError(ToolError):
    """Raised when a tool is blocked by the active runtime profile."""


class LoopLimitExceededError(AgentManagerError):
    """Raised when the runtime reaches the maximum step count."""


class CheckpointError(AgentManagerError):
    """Raised when checkpoint storage fails."""
