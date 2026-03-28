"""Public package exports for agent_manager."""

from agent_manager.config import LoggingConfig, ProviderConfig, RuntimeConfig, RuntimeLimits, load_config
from agent_manager.providers.factory import available_providers, build_provider, register_provider
from agent_manager.runtime.session import AgentSession
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from agent_manager.tools.registry import ToolRegistry
from agent_manager.types import (
    AgentRunResult,
    ContextHint,
    ContextSection,
    LoopState,
    Message,
    ProviderRequest,
    ProviderResult,
    ToolCallRequest,
)
from agent_manager.version import __version__

__all__ = [
    "__version__",
    "AgentRunResult",
    "AgentSession",
    "BaseTool",
    "ContextHint",
    "ContextSection",
    "LoggingConfig",
    "LoopState",
    "Message",
    "ProviderConfig",
    "ProviderRequest",
    "ProviderResult",
    "RuntimeConfig",
    "RuntimeLimits",
    "ToolCallRequest",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "available_providers",
    "build_provider",
    "load_config",
    "register_provider",
]

