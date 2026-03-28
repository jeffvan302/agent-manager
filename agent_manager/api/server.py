"""Headless service abstraction for future HTTP or IPC layers."""

from __future__ import annotations

from agent_manager.api.schemas import RunRequest, RunResponse
from agent_manager.runtime.session import AgentSession


class AgentService:
    """Small service wrapper around an AgentSession."""

    def __init__(self, session: AgentSession) -> None:
        self.session = session

    async def run_async(self, request: RunRequest) -> RunResponse:
        result = await self.session.run_async(request.prompt, task_id=request.task_id)
        return RunResponse(
            task_id=result.task_id,
            output_text=result.output_text,
            status=result.state.status,
            stop_reason=result.stop_reason,
        )

    def run(self, request: RunRequest) -> RunResponse:
        result = self.session.run(request.prompt, task_id=request.task_id)
        return RunResponse(
            task_id=result.task_id,
            output_text=result.output_text,
            status=result.state.status,
            stop_reason=result.stop_reason,
        )
