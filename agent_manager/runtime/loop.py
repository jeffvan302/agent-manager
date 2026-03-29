"""Agent loop with checkpointing, event streaming, and structured output."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_manager.async_utils import run_sync
from agent_manager.config import RuntimeConfig
from agent_manager.context.pipeline import PreCallPipeline
from agent_manager.errors import PolicyViolationError, ProviderResourceExhaustedError
from agent_manager.observability import get_logger
from agent_manager.providers.base import BaseProvider, maybe_parse_structured_output
from agent_manager.runtime.events import RuntimeEvent
from agent_manager.runtime.planner import Planner
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
    StructuredOutputSpec,
    ToolCallRequest,
)

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]


class AgentLoop:
    """Run the provider/tool loop."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        provider: BaseProvider,
        tools: ToolRegistry,
        tool_executor: ToolExecutor,
        context_pipeline: PreCallPipeline,
        checkpoints: CheckpointManager,
        working_directory: str | Path | None = None,
        tool_context_metadata: dict[str, Any] | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tools = tools
        self.tool_executor = tool_executor
        self.context_pipeline = context_pipeline
        self.checkpoints = checkpoints
        self.working_directory = str(
            Path(working_directory or Path.cwd()).resolve(strict=False)
        )
        self.tool_context_metadata = dict(tool_context_metadata or {})
        self.planner = planner or Planner()
        self.logger = get_logger("runtime.loop")
        self._interrupt_requested = False

    async def run_async(
        self,
        goal: str,
        task_id: str | None = None,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
        event_sink: EventSink | None = None,
    ) -> AgentRunResult:
        self._interrupt_requested = False
        spec = self._coerce_structured_output(structured_output)
        state = LoopState(
            task_id=task_id or uuid.uuid4().hex,
            goal=goal,
            messages=[Message(role="user", content=goal)],
            structured_output_spec=spec,
        )
        return await self.run_state_async(
            state,
            structured_output=spec,
            event_sink=event_sink,
        )

    def run(
        self,
        goal: str,
        task_id: str | None = None,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> AgentRunResult:
        return run_sync(
            self.run_async(goal, task_id=task_id, structured_output=structured_output)
        )

    async def stream_async(
        self,
        goal: str,
        task_id: str | None = None,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue()

        async def _sink(event: RuntimeEvent) -> None:
            await queue.put(event)

        async def _producer() -> None:
            try:
                await self.run_async(
                    goal,
                    task_id=task_id,
                    structured_output=structured_output,
                    event_sink=_sink,
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(_producer())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event.to_dict()
            await task
        finally:
            if not task.done():
                self.request_interrupt()
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def run_state_async(
        self,
        state: LoopState,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
        event_sink: EventSink | None = None,
    ) -> AgentRunResult:
        requested_spec = self._coerce_structured_output(structured_output)
        spec = requested_spec or state.structured_output_spec
        state.structured_output_spec = spec
        events: list[RuntimeEvent] = []
        await self._emit_event(
            events,
            RuntimeEvent("loop.started", {"task_id": state.task_id, "goal": state.goal}),
            event_sink,
        )
        await self._maybe_plan(state, events, event_sink)
        last_result = ProviderResult(text="")
        tool_results: list[dict[str, Any]] = []
        deadline = self._deadline()
        consecutive_failures = int(state.metadata.get("consecutive_failures", 0))

        while state.step_index < self.config.runtime.max_steps:
            step_record = {
                "step_index": state.step_index,
                "started_at": self._timestamp(),
                "provider": self.provider.provider_name,
                "model": self.config.provider.model,
            }
            stop_reason = self._should_stop(deadline)
            if stop_reason is not None:
                step_record["status"] = stop_reason
                self._record_step(state, step_record)
                return await self._stop(
                    state,
                    stop_reason=stop_reason,
                    last_result=last_result,
                    tool_results=tool_results,
                    events=events,
                    event_sink=event_sink,
                )

            self.checkpoints.save(state)
            prepared = await self.context_pipeline.prepare_async(state, self.config)
            state.metadata["prepared_context"] = {
                "sections": [section.key for section in prepared.sections],
                "dropped_sections": list(prepared.dropped_sections),
                "token_estimate": prepared.token_estimate,
                "pre_call_functions": list(prepared.metadata.get("pre_call_functions", [])),
                "executed_steps": list(prepared.metadata.get("executed_steps", [])),
            }
            step_record["prepared_context"] = dict(state.metadata["prepared_context"])
            self.checkpoints.save(state)
            await self._emit_event(
                events,
                RuntimeEvent(
                    "context.prepared",
                    {
                        "task_id": state.task_id,
                        "step_index": state.step_index,
                        "token_estimate": prepared.token_estimate,
                        "sections": [section.key for section in prepared.sections],
                        "dropped_sections": list(prepared.dropped_sections),
                    },
                ),
                event_sink,
            )

            request_messages = self._messages_for_request(prepared.messages, spec)
            request = ProviderRequest(
                model=self.config.provider.model,
                messages=request_messages,
                tools=self.tools.provider_definitions(),
                max_tokens=self.config.provider.resolved_max_output_tokens(
                    self.config.runtime.max_output_tokens
                ),
                structured_output=spec,
                stream=event_sink is not None,
                metadata={
                    "task_id": state.task_id,
                    "step_index": state.step_index,
                    "profile": self.config.profile,
                },
            )
            step_record["request"] = {
                "message_count": len(request.messages),
                "tool_count": len(request.tools),
                "structured_output": spec.to_dict() if spec is not None else None,
            }
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
            await self._emit_event(
                events,
                RuntimeEvent(
                    "provider.requested",
                    {
                        "provider": self.provider.provider_name,
                        "task_id": state.task_id,
                        "step_index": state.step_index,
                        "tool_count": len(request.tools),
                    },
                ),
                event_sink,
            )

            try:
                last_result = await self._generate_with_timeout(
                    request,
                    deadline,
                    events=events,
                    event_sink=event_sink,
                )
            except TimeoutError as exc:
                self._append_error(state, "timeout", str(exc) or "Loop timed out.")
                step_record["status"] = "timeout"
                step_record["error"] = str(exc)
                self._record_step(state, step_record)
                return await self._stop(
                    state,
                    stop_reason="timeout",
                    last_result=last_result,
                    tool_results=tool_results,
                    events=events,
                    event_sink=event_sink,
                )
            except ProviderResourceExhaustedError as exc:
                state.metadata["resource_exhaustion"] = exc.to_dict()
                self._append_error(
                    state,
                    "resource_exhausted",
                    str(exc),
                    metadata=exc.to_dict(),
                )
                step_record["status"] = "resource_exhausted"
                step_record["error"] = exc.to_dict()
                self._record_step(state, step_record)
                await self._emit_event(
                    events,
                    RuntimeEvent(
                        "provider.resource_exhausted",
                        {
                            "task_id": state.task_id,
                            "step_index": state.step_index,
                            "provider": exc.provider or self.provider.provider_name,
                            "kind": exc.kind,
                            "status_code": exc.status_code,
                            "retry_after_seconds": exc.retry_after_seconds,
                        },
                    ),
                    event_sink,
                )
                return await self._stop(
                    state,
                    stop_reason="resource_exhausted",
                    last_result=last_result,
                    tool_results=tool_results,
                    events=events,
                    event_sink=event_sink,
                )
            except Exception as exc:
                consecutive_failures = await self._record_failure(
                    state,
                    exc,
                    events,
                    event_sink=event_sink,
                )
                step_record["status"] = "failed"
                step_record["error"] = str(exc)
                self._record_step(state, step_record)
                if consecutive_failures >= self.config.runtime.max_consecutive_failures:
                    return await self._stop(
                        state,
                        stop_reason="repeated_failure",
                        last_result=last_result,
                        tool_results=tool_results,
                        events=events,
                        event_sink=event_sink,
                    )
                state.step_index += 1
                continue

            if spec is not None and last_result.structured_output is None:
                last_result.structured_output = maybe_parse_structured_output(
                    last_result.text,
                    spec,
                )
                if last_result.structured_output is None and last_result.text:
                    self._append_error(
                        state,
                        "structured_output_parse_failed",
                        "Structured output was requested but the response was not valid JSON.",
                        metadata={"non_fatal": True},
                    )
                    await self._emit_event(
                        events,
                        RuntimeEvent(
                            "provider.structured_output_unparsed",
                            {
                                "task_id": state.task_id,
                                "step_index": state.step_index,
                            },
                        ),
                        event_sink,
                    )

            self._apply_result_state(state, last_result)
            consecutive_failures = 0
            state.metadata["consecutive_failures"] = 0
            state.metadata.pop("last_error", None)
            step_record["provider_result"] = {
                "stop_reason": last_result.stop_reason,
                "usage": last_result.usage,
                "tool_calls": [call.to_dict() for call in last_result.tool_calls],
                "structured_output": last_result.structured_output,
            }
            await self._emit_event(
                events,
                RuntimeEvent(
                    "provider.completed",
                    {
                        "provider": self.provider.provider_name,
                        "stop_reason": last_result.stop_reason,
                        "structured_output": last_result.structured_output is not None,
                    },
                ),
                event_sink,
            )

            if last_result.text or last_result.tool_calls:
                state.messages.append(self._assistant_message_from_result(last_result))

            if last_result.tool_calls:
                step_record["tool_calls"] = [call.to_dict() for call in last_result.tool_calls]
                for call in last_result.tool_calls:
                    stop_reason = self._should_stop(deadline)
                    if stop_reason is not None:
                        step_record["status"] = stop_reason
                        self._record_step(state, step_record)
                        return await self._stop(
                            state,
                            stop_reason=stop_reason,
                            last_result=last_result,
                            tool_results=tool_results,
                            events=events,
                            event_sink=event_sink,
                        )
                    await self._emit_event(
                        events,
                        RuntimeEvent(
                            "tool.requested",
                            {
                                "task_id": state.task_id,
                                "step_index": state.step_index,
                                "tool_name": call.name,
                                "tool_call_id": call.id,
                            },
                        ),
                        event_sink,
                    )
                    try:
                        result = await self._execute_tool_call(call, state, deadline)
                    except PolicyViolationError as exc:
                        self._append_error(
                            state,
                            "policy_violation",
                            str(exc),
                            metadata={"tool_name": call.name, "tool_call_id": call.id},
                        )
                        step_record["status"] = "policy_violation"
                        step_record["error"] = str(exc)
                        self._record_step(state, step_record)
                        await self._emit_event(
                            events,
                            RuntimeEvent(
                                "tool.policy_violation",
                                {
                                    "task_id": state.task_id,
                                    "step_index": state.step_index,
                                    "tool_name": call.name,
                                    "tool_call_id": call.id,
                                },
                            ),
                            event_sink,
                        )
                        return await self._stop(
                            state,
                            stop_reason="policy_violation",
                            last_result=last_result,
                            tool_results=tool_results,
                            events=events,
                            event_sink=event_sink,
                        )
                    except TimeoutError as exc:
                        self._append_error(
                            state,
                            "timeout",
                            str(exc) or "Tool execution timed out.",
                            metadata={"tool_name": call.name, "tool_call_id": call.id},
                        )
                        step_record["status"] = "timeout"
                        step_record["error"] = str(exc)
                        self._record_step(state, step_record)
                        return await self._stop(
                            state,
                            stop_reason="timeout",
                            last_result=last_result,
                            tool_results=tool_results,
                            events=events,
                            event_sink=event_sink,
                        )
                    except Exception as exc:
                        consecutive_failures = await self._record_failure(
                            state,
                            exc,
                            events,
                            event_sink=event_sink,
                        )
                        if consecutive_failures >= self.config.runtime.max_consecutive_failures:
                            step_record["status"] = "repeated_failure"
                            step_record["error"] = str(exc)
                            self._record_step(state, step_record)
                            return await self._stop(
                                state,
                                stop_reason="repeated_failure",
                                last_result=last_result,
                                tool_results=tool_results,
                                events=events,
                                event_sink=event_sink,
                            )
                        continue

                    if result.ok:
                        consecutive_failures = 0
                        state.metadata["consecutive_failures"] = 0
                        state.metadata.pop("last_error", None)
                    else:
                        consecutive_failures = await self._record_failure(
                            state,
                            RuntimeError(result.error or f"Tool '{result.tool_name}' failed."),
                            events,
                            event_sink=event_sink,
                        )
                    payload = result.to_dict()
                    tool_results.append(payload)
                    state.tool_observations.append(payload)
                    step_record.setdefault("tool_results", []).append(payload)
                    state.messages.append(
                        Message(
                            role="tool",
                            name=result.tool_name,
                            content=self._stringify_tool_output(result),
                            metadata={
                                "tool_call_id": call.id,
                                "is_error": not result.ok,
                            },
                        )
                    )
                    await self._emit_event(
                        events,
                        RuntimeEvent(
                            "tool.completed",
                            {
                                "task_id": state.task_id,
                                "step_index": state.step_index,
                                "tool_name": result.tool_name,
                                "tool_call_id": call.id,
                                "ok": result.ok,
                            },
                        ),
                        event_sink,
                    )
                    if not result.ok and (
                        consecutive_failures >= self.config.runtime.max_consecutive_failures
                    ):
                        step_record["status"] = "repeated_failure"
                        self._record_step(state, step_record)
                        return await self._stop(
                            state,
                            stop_reason="repeated_failure",
                            last_result=last_result,
                            tool_results=tool_results,
                            events=events,
                            event_sink=event_sink,
                        )
                step_record["status"] = "tool_call"
                self._record_step(state, step_record)
                state.step_index += 1
                continue

            step_record["status"] = last_result.stop_reason or "completed"
            self._record_step(state, step_record)
            return await self._stop(
                state,
                stop_reason=last_result.stop_reason or "completed",
                last_result=last_result,
                tool_results=tool_results,
                events=events,
                event_sink=event_sink,
                advance_step=True,
            )

        return await self._stop(
            state,
            stop_reason="max_steps_exceeded",
            last_result=last_result,
            tool_results=tool_results,
            events=events,
            event_sink=event_sink,
        )

    def run_state(
        self,
        state: LoopState,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> AgentRunResult:
        return run_sync(self.run_state_async(state, structured_output=structured_output))

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
            working_directory=self.working_directory,
            metadata=dict(self.tool_context_metadata),
        )
        result = await self._execute_tool_with_timeout(call, context, deadline)
        result.metadata.setdefault("tool_call_id", call.id)
        return result

    def _stringify_tool_output(self, result: ToolResult) -> str:
        if result.error:
            return json.dumps(
                {"ok": result.ok, "output": result.output, "error": result.error},
                ensure_ascii=True,
            )
        if isinstance(result.output, str):
            return result.output
        return json.dumps(result.output, ensure_ascii=True)

    def _assistant_message_from_result(self, result: ProviderResult) -> Message:
        metadata = {"stop_reason": result.stop_reason or "completed"}
        if result.tool_calls:
            metadata["tool_calls"] = [call.to_dict() for call in result.tool_calls]
        if result.structured_output is not None:
            metadata["structured_output"] = result.structured_output
        return Message(role="assistant", content=result.text or "", metadata=metadata)

    def request_interrupt(self) -> None:
        self._interrupt_requested = True

    def _deadline(self) -> float | None:
        if self.config.runtime.timeout_seconds <= 0:
            return None
        return time.monotonic() + self.config.runtime.timeout_seconds

    def _remaining_time(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(deadline - time.monotonic(), 0.0)

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
        *,
        events: list[RuntimeEvent],
        event_sink: EventSink | None,
    ) -> ProviderResult:
        remaining = self._remaining_time(deadline)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Loop timeout reached before provider call.")
        if event_sink is None:
            return await asyncio.wait_for(self.provider.generate(request), timeout=remaining)
        return await asyncio.wait_for(
            self._stream_provider(request, events, event_sink),
            timeout=remaining,
        )

    async def _stream_provider(
        self,
        request: ProviderRequest,
        events: list[RuntimeEvent],
        event_sink: EventSink,
    ) -> ProviderResult:
        final_result: ProviderResult | None = None
        chunks: list[str] = []
        async for item in self.provider.stream_generate(request):
            if item.kind == "text_delta" and item.text:
                chunks.append(item.text)
                await self._emit_event(
                    events,
                    RuntimeEvent(
                        "provider.text_delta",
                        {
                            "task_id": request.metadata.get("task_id"),
                            "step_index": request.metadata.get("step_index"),
                            "delta": item.text,
                        },
                    ),
                    event_sink,
                )
            elif item.kind == "result" and item.result is not None:
                final_result = item.result
        if final_result is None:
            final_result = ProviderResult(text="".join(chunks), stop_reason="completed")
        elif not final_result.text and chunks:
            final_result.text = "".join(chunks)
        await self._emit_event(
            events,
            RuntimeEvent(
                "provider.result",
                {
                    "provider": self.provider.provider_name,
                    "task_id": request.metadata.get("task_id"),
                    "step_index": request.metadata.get("step_index"),
                    **self._result_payload(final_result),
                },
            ),
            event_sink,
        )
        return final_result

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

    async def _record_failure(
        self,
        state: LoopState,
        exc: Exception,
        events: list[RuntimeEvent],
        *,
        event_sink: EventSink | None,
    ) -> int:
        failures = int(state.metadata.get("consecutive_failures", 0)) + 1
        state.metadata["consecutive_failures"] = failures
        self._append_error(
            state,
            type(exc).__name__,
            str(exc),
            metadata={"consecutive_failures": failures},
        )
        await self._emit_event(
            events,
            RuntimeEvent(
                "loop.failure",
                {
                    "task_id": state.task_id,
                    "step_index": state.step_index,
                    "error": str(exc),
                    "consecutive_failures": failures,
                },
            ),
            event_sink,
        )
        self.checkpoints.save(state)
        return failures

    async def _stop(
        self,
        state: LoopState,
        *,
        stop_reason: str,
        last_result: ProviderResult,
        tool_results: list[dict[str, Any]],
        events: list[RuntimeEvent],
        event_sink: EventSink | None,
        advance_step: bool = False,
    ) -> AgentRunResult:
        if advance_step:
            state.step_index += 1
        state.status = stop_reason
        self.checkpoints.save(state)
        terminal_payload = {
            "task_id": state.task_id,
            "step_index": state.step_index,
            "stop_reason": stop_reason,
            "resource_exhaustion": state.metadata.get("resource_exhaustion"),
            "tool_results": list(tool_results),
            **self._result_payload(last_result),
        }
        await self._emit_event(
            events,
            RuntimeEvent(
                f"loop.{stop_reason}",
                terminal_payload,
            ),
            event_sink,
        )
        return AgentRunResult(
            task_id=state.task_id,
            output_text=last_result.text or "",
            state=state,
            stop_reason=stop_reason,
            usage=last_result.usage,
            structured_output=last_result.structured_output,
            resource_exhaustion=state.metadata.get("resource_exhaustion"),
            tool_results=tool_results,
            events=[event.to_dict() for event in events],
        )

    async def _emit_event(
        self,
        events: list[RuntimeEvent],
        event: RuntimeEvent,
        event_sink: EventSink | None,
    ) -> None:
        events.append(event)
        if event_sink is None:
            return
        value = event_sink(event)
        if asyncio.iscoroutine(value):
            await value

    def _append_error(
        self,
        state: LoopState,
        error_type: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        state.metadata["last_error"] = message
        state.errors.append(
            {
                "type": error_type,
                "message": message,
                "step_index": state.step_index,
                "timestamp": self._timestamp(),
                "metadata": dict(metadata or {}),
            }
        )

    def _record_step(self, state: LoopState, step_record: dict[str, Any]) -> None:
        step_record.setdefault("finished_at", self._timestamp())
        state.step_history.append(step_record)

    def _apply_result_state(self, state: LoopState, result: ProviderResult) -> None:
        pending = result.metadata.get("pending_subgoals")
        if isinstance(pending, list):
            state.pending_subgoals = [str(item) for item in pending]

    async def _maybe_plan(
        self,
        state: LoopState,
        events: list[RuntimeEvent],
        event_sink: EventSink | None,
    ) -> None:
        if state.metadata.get("current_plan"):
            return
        plan = self.planner.build_plan(
            state.goal,
            available_tools=self.tools.names(),
        )
        if not plan:
            return
        state.metadata["current_plan"] = list(plan)
        state.metadata["planner"] = type(self.planner).__name__
        if not state.pending_subgoals:
            state.pending_subgoals = list(plan)
        self.checkpoints.save(state)
        await self._emit_event(
            events,
            RuntimeEvent(
                "loop.planned",
                {
                    "task_id": state.task_id,
                    "step_index": state.step_index,
                    "planner": type(self.planner).__name__,
                    "plan": list(plan),
                },
            ),
            event_sink,
        )

    def _messages_for_request(
        self,
        messages: list[Message],
        structured_output: StructuredOutputSpec | None,
    ) -> list[Message]:
        rendered = list(messages)
        if structured_output is None or self.provider.capabilities.supports_structured_output:
            return rendered
        instruction = self._structured_output_instruction(structured_output)
        if not instruction:
            return rendered
        return [Message(role="system", content=instruction), *rendered]

    def _structured_output_instruction(self, structured_output: StructuredOutputSpec) -> str:
        parts = ["Return only valid JSON."]
        if structured_output.type == "json_schema" and structured_output.schema:
            parts.append(
                f"Match this JSON schema exactly: {json.dumps(structured_output.schema, ensure_ascii=True)}"
            )
        if structured_output.prompt:
            parts.append(structured_output.prompt)
        return "\n".join(parts)

    def _coerce_structured_output(
        self,
        value: StructuredOutputSpec | dict[str, Any] | None,
    ) -> StructuredOutputSpec | None:
        if value is None:
            return None
        if isinstance(value, StructuredOutputSpec):
            return value
        return StructuredOutputSpec.from_dict(value)

    def _result_payload(self, result: ProviderResult) -> dict[str, Any]:
        return {
            "output_text": result.text or "",
            "structured_output": result.structured_output,
            "usage": result.usage,
            "provider_stop_reason": result.stop_reason,
            "tool_calls": [call.to_dict() for call in result.tool_calls],
        }

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()
