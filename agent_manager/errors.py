"""Shared error types for agent_manager."""


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
