"""Minimal agent loop implementation."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from agent_manager.async_utils import run_sync
from agent_manager.config import RuntimeConfig
from agent_manager.context.pipeline import PreCallPipeline
from agent_manager.errors import PolicyViolationError
from agent_manager.observability import get_logger
from agent_manager.providers.base import BaseProvider
from agent_manager.runtime.events import RuntimeEvent
from agent_manager.state.checkpoint import CheckpointManager
from agent_manager.tools.base import ToolContext, ToolResult
from agent_manager.tools.executor import ToolExecutor
from agent_manager.tools.registry import ToolRegistry
from agent_manager.types import (
    AgentRunResult,
    LoopState,
    Message,
    ProviderRequest,
    ProviderResult,
    ToolCallRequest,
)


class AgentLoop:
    """Run an iterative provider/tool loop with checkpoint hooks."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        provider: BaseProvider,
        tools: ToolRegistry,
        tool_executor: ToolExecutor,
        context_pipeline: PreCallPipeline,
        checkpoints: CheckpointManager,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tools = tools
        self.tool_executor = tool_executor
        self.context_pipeline = context_pipeline
        self.checkpoints = checkpoints
        self.logger = get_logger("runtime.loop")
        self._interrupt_requested = False

    async def run_async(self, goal: str, task_id: str | None = None) -> AgentRunResult:
        self._interrupt_requested = False
        initial_state = LoopState(
            task_id=task_id or uuid.uuid4().hex,
            goal=goal,
            messages=[Message(role="user", content=goal)],
        )
        return await self.run_state_async(initial_state)

    def run(self, goal: str, task_id: str | None = None) -> AgentRunResult:
        return run_sync(self.run_async(goal, task_id=task_id))

    async def run_state_async(self, state: LoopState) -> AgentRunResult:
        events: list[RuntimeEvent] = [
            RuntimeEvent("loop.started", {"task_id": state.task_id, "goal": state.goal})
        ]
        tool_results: list[dict] = []
        last_result = ProviderResult(text="")
        deadline = self._deadline()
        consecutive_failures = int(state.metadata.get("consecutive_failures", 0))

        while state.step_index < self.config.runtime.max_steps:
            stop_reason = self._should_stop(deadline)
            if stop_reason is not None:
                return self._stop(
                    state,
                    stop_reason=stop_reason,
                    last_result=last_result,
                    tool_results=tool_results,
                    events=events,
                )

            self.checkpoints.save(state)
            prepared = self.context_pipeline.prepare(state, self.config)
            events.append(
                RuntimeEvent(
                    "context.prepared",
                    {
                        "task_id": state.task_id,
                        "step_index": state.step_index,
                        "token_estimate": prepared.token_estimate,
                        "dropped_sections": list(prepared.dropped_sections),
                    },
                )
            )

            try:
                request = ProviderRequest(
                    model=self.config.provider.model,
                    messages=prepared.messages,
                    tools=self.tools.provider_definitions(),
                    max_tokens=self.config.runtime.max_output_tokens,
                    metadata={
                        "task_id": state.task_id,
                        "step_index": state.step_index,
                        "profile": self.config.profile,
                    },
                )
                self.logger.info(
                    "provider request",
                    extra={
                        "event": "provider.request",
                        "details": {
                            "provider": self.provider.provider_name,
                            "task_id": state.task_id,
                            "step_index": state.step_index,
                        },
                    },
                )
                last_result = await self._generate_with_timeout(request, deadline)
                consecutive_failures = 0
                state.metadata["consecutive_failures"] = 0
                state.metadata.pop("last_error", None)
                events.append(
                    RuntimeEvent(
                        "provider.completed",
                        {
                            "provider": self.provider.provider_name,
                            "stop_reason": last_result.stop_reason,
                        },
                    )
                )
            except TimeoutError as exc:
                state.metadata["last_error"] = str(exc) or "Loop timed out."
                return self._stop(
                    state,
                    stop_reason="timeout",
                    last_result=last_result,
                    tool_results=tool_results,
                    events=events,
                )
            except Exception as exc:
                consecutive_failures = self._record_failure(state, exc, events)
                if consecutive_failures >= self.config.runtime.max_consecutive_failures:
                    return self._stop(
                        state,
                        stop_reason="repeated_failure",
                        last_result=last_result,
                        tool_results=tool_results,
                        events=events,
                    )
                state.step_index += 1
                continue

            if last_result.text:
                state.messages.append(
                    Message(
                        role="assistant",
                        content=last_result.text,
                        metadata={"stop_reason": last_result.stop_reason or "completed"},
                    )
                )

            if last_result.tool_calls:
                for call in last_result.tool_calls:
                    stop_reason = self._should_stop(deadline)
                    if stop_reason is not None:
                        return self._stop(
                            state,
                            stop_reason=stop_reason,
                            last_result=last_result,
                            tool_results=tool_results,
                            events=events,
                        )
                    try:
                        result = await self._execute_tool_call(call, state, deadline)
                        consecutive_failures = 0
                        state.metadata["consecutive_failures"] = 0
                        state.metadata.pop("last_error", None)
                    except PolicyViolationError as exc:
                        state.metadata["last_error"] = str(exc)
                        events.append(
                            RuntimeEvent(
                                "tool.policy_violation",
                                {
                                    "task_id": state.task_id,
                                    "step_index": state.step_index,
                                    "tool_name": call.name,
                                    "tool_call_id": call.id,
                                },
                            )
                        )
                        return self._stop(
                            state,
                            stop_reason="policy_violation",
                            last_result=last_result,
                            tool_results=tool_results,
                            events=events,
                        )
                    except TimeoutError as exc:
                        state.metadata["last_error"] = str(exc) or "Tool execution timed out."
                        return self._stop(
                            state,
                            stop_reason="timeout",
                            last_result=last_result,
                            tool_results=tool_results,
                            events=events,
                        )
                    except Exception as exc:
                        consecutive_failures = self._record_failure(state, exc, events)
                        if consecutive_failures >= self.config.runtime.max_consecutive_failures:
                            return self._stop(
                                state,
                                stop_reason="repeated_failure",
                                last_result=last_result,
                                tool_results=tool_results,
                                events=events,
                            )
                        state.step_index += 1
                        break
                    tool_results.append(result.to_dict())
                    state.tool_observations.append(result.to_dict())
                    state.messages.append(
                        Message(
                            role="tool",
                            name=result.tool_name,
                            content=self._stringify_tool_output(result),
                            metadata={"tool_call_id": call.id},
                        )
                    )
                else:
                    state.step_index += 1
                    continue
                state.step_index += 1
                continue

            return self._stop(
                state,
                stop_reason=last_result.stop_reason or "completed",
                last_result=last_result,
                tool_results=tool_results,
                events=events,
                advance_step=True,
            )

        return self._stop(
            state,
            stop_reason="max_steps_exceeded",
            last_result=last_result,
            tool_results=tool_results,
            events=events,
        )

    def run_state(self, state: LoopState) -> AgentRunResult:
        return run_sync(self.run_state_async(state))

    async def _execute_tool_call(
        self,
        call: ToolCallRequest,
        state: LoopState,
        deadline: float | None,
    ) -> ToolResult:
        context = ToolContext(
            task_id=state.task_id,
            step_index=state.step_index,
            tool_call_id=call.id,
            working_directory=str(Path.cwd()),
        )
        result = await self._execute_tool_with_timeout(call, context, deadline)
        result.metadata.setdefault("tool_call_id", call.id)
        return result

    def _stringify_tool_output(self, result: ToolResult) -> str:
        if isinstance(result.output, str):
            return result.output
        return json.dumps(result.output, ensure_ascii=True)

    def request_interrupt(self) -> None:
        self._interrupt_requested = True

    def _deadline(self) -> float | None:
        timeout_seconds = self.config.runtime.timeout_seconds
        if timeout_seconds <= 0:
            return None
        return time.monotonic() + timeout_seconds

    def _remaining_time(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        return max(remaining, 0.0)

    def _should_stop(self, deadline: float | None) -> str | None:
        if self._interrupt_requested:
            return "user_interrupted"
        remaining = self._remaining_time(deadline)
        if remaining is not None and remaining <= 0:
            return "timeout"
        return None

    async def _generate_with_timeout(
        self,
        request: ProviderRequest,
        deadline: float | None,
    ) -> ProviderResult:
        remaining = self._remaining_time(deadline)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Loop timeout reached before provider call.")
        return await asyncio.wait_for(self.provider.generate(request), timeout=remaining)

    async def _execute_tool_with_timeout(
        self,
        call: ToolCallRequest,
        context: ToolContext,
        deadline: float | None,
    ) -> ToolResult:
        remaining = self._remaining_time(deadline)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Loop timeout reached before tool execution.")
        task = self.tool_executor.execute_async(call, context)
        if remaining is None:
            return await task
        return await asyncio.wait_for(task, timeout=remaining)

    def _record_failure(
        self,
        state: LoopState,
        exc: Exception,
        events: list[RuntimeEvent],
    ) -> int:
        failures = int(state.metadata.get("consecutive_failures", 0)) + 1
        state.metadata["consecutive_failures"] = failures
        state.metadata["last_error"] = str(exc)
        events.append(
            RuntimeEvent(
                "loop.failure",
                {
                    "task_id": state.task_id,
                    "step_index": state.step_index,
                    "error": str(exc),
                    "consecutive_failures": failures,
                },
            )
        )
        self.checkpoints.save(state)
        return failures

    def _stop(
        self,
        state: LoopState,
        *,
        stop_reason: str,
        last_result: ProviderResult,
        tool_results: list[dict],
        events: list[RuntimeEvent],
        advance_step: bool = False,
    ) -> AgentRunResult:
        if advance_step:
            state.step_index += 1
        state.status = stop_reason
        self.checkpoints.save(state)
        events.append(
            RuntimeEvent(
                f"loop.{stop_reason}",
                {"task_id": state.task_id, "step_index": state.step_index},
            )
        )
        output_text = last_result.text or ""
        return AgentRunResult(
            task_id=state.task_id,
            output_text=output_text,
            state=state,
            stop_reason=stop_reason,
            usage=last_result.usage,
            tool_results=tool_results,
            events=[event.to_dict() for event in events],
        )
