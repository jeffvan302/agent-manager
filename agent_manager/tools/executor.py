"""Tool execution helpers."""

from __future__ import annotations

import asyncio
import time

from agent_manager.async_utils import run_sync
from agent_manager.errors import PolicyViolationError
from agent_manager.observability import emitter
from agent_manager.tools.base import ToolContext, ToolResult, normalize_tool_result
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
        try:
            self.policy_engine.assert_allowed(
                tool.spec,
                context=context,
                arguments=call.arguments,
            )
        except PolicyViolationError as exc:
            emitter.tool_policy_violation(tool_name=tool.spec.name, reason=str(exc))
            raise
        timeout = tool.spec.timeout_seconds if tool.spec.timeout_seconds > 0 else None
        max_attempts = max(tool.spec.retry_count + 1, 1)
        backoff_seconds = max(tool.spec.retry_backoff_seconds, 0.0)

        for attempt in range(1, max_attempts + 1):
            started_at = time.perf_counter()
            emitter.tool_call(
                tool_name=tool.spec.name,
                tool_call_id=context.tool_call_id or "",
                arguments=call.arguments,
            )
            try:
                invocation = tool.invoke(call.arguments, context)
                result = await asyncio.wait_for(invocation, timeout=timeout)
                normalized = normalize_tool_result(tool.spec.name, result)
                normalized.metadata.setdefault("attempts", attempt)
                emitter.tool_result(
                    tool_name=tool.spec.name,
                    ok=normalized.ok,
                    duration_ms=(time.perf_counter() - started_at) * 1000.0,
                    error=normalized.error,
                )
                return normalized
            except asyncio.TimeoutError:
                duration_ms = (time.perf_counter() - started_at) * 1000.0
                emitter.tool_result(
                    tool_name=tool.spec.name,
                    ok=False,
                    duration_ms=duration_ms,
                    error=f"Tool '{tool.spec.name}' timed out after {timeout} seconds.",
                )
                raise TimeoutError(
                    f"Tool '{tool.spec.name}' timed out after {timeout} seconds."
                )
            except PolicyViolationError:
                raise
            except Exception as exc:
                if attempt >= max_attempts:
                    duration_ms = (time.perf_counter() - started_at) * 1000.0
                    emitter.tool_result(
                        tool_name=tool.spec.name,
                        ok=False,
                        duration_ms=duration_ms,
                        error=str(exc),
                    )
                    return ToolResult(
                        tool_name=tool.spec.name,
                        ok=False,
                        output={},
                        error=str(exc),
                        metadata={"attempts": attempt},
                    )
                if backoff_seconds > 0:
                    await asyncio.sleep(backoff_seconds * attempt)

        return ToolResult(
            tool_name=tool.spec.name,
            ok=False,
            output={},
            error=f"Tool '{tool.spec.name}' failed without producing a result.",
            metadata={"attempts": max_attempts},
        )

    def execute(self, call: ToolCallRequest, context: ToolContext) -> ToolResult:
        return run_sync(self.execute_async(call, context))
