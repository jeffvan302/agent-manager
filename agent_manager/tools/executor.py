"""Tool execution helpers."""

from __future__ import annotations

import asyncio
import inspect

from agent_manager.async_utils import run_sync
from agent_manager.tools.base import ToolContext, ToolResult
from agent_manager.tools.policies import PolicyEngine
from agent_manager.tools.registry import ToolRegistry
from agent_manager.types import ToolCallRequest


class ToolExecutor:
    """Execute normalized tool call requests against registered tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self.registry = registry
        self.policy_engine = policy_engine or PolicyEngine()

    async def execute_async(self, call: ToolCallRequest, context: ToolContext) -> ToolResult:
        tool = self.registry.get(call.name)
        self.policy_engine.assert_allowed(tool.spec)
        timeout = tool.spec.timeout_seconds if tool.spec.timeout_seconds > 0 else None

        if inspect.iscoroutinefunction(tool.invoke):
            invocation = tool.invoke(call.arguments, context)
        else:
            invocation = asyncio.to_thread(tool.invoke, call.arguments, context)

        return await asyncio.wait_for(invocation, timeout=timeout)

    def execute(self, call: ToolCallRequest, context: ToolContext) -> ToolResult:
        return run_sync(self.execute_async(call, context))
