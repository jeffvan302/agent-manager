"""High-level session object exposed to library users."""

from __future__ import annotations

from agent_manager.config import RuntimeConfig, load_config
from agent_manager.context.pipeline import PreCallPipeline
from agent_manager.observability import configure_logging, get_logger
from agent_manager.providers.base import BaseProvider
from agent_manager.providers.factory import build_provider
from agent_manager.runtime.loop import AgentLoop
from agent_manager.state.checkpoint import CheckpointManager
from agent_manager.state.store import JsonFileStateStore, StateStore
from agent_manager.tools.executor import ToolExecutor
from agent_manager.tools.policies import PolicyEngine
from agent_manager.tools.registry import ToolRegistry
from agent_manager.types import AgentRunResult


class AgentSession:
    """User-facing runtime facade for one configured agent manager session."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        provider: BaseProvider | None = None,
        tools: ToolRegistry | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        self.config = config or load_config()
        configure_logging(
            level=self.config.logging.level,
            json_output=self.config.logging.json_output,
        )
        self.logger = get_logger("runtime.session")
        self.provider = provider or build_provider(self.config.provider)
        self.tools = tools or ToolRegistry()
        self.policy_engine = PolicyEngine(self.config.profile)
        self.tool_executor = ToolExecutor(self.tools, self.policy_engine)
        self.state_store = state_store or JsonFileStateStore(
            self.config.resolved_state_dir()
        )
        self.checkpoints = CheckpointManager(self.state_store)
        self.context_pipeline = PreCallPipeline()
        self.loop = AgentLoop(
            config=self.config,
            provider=self.provider,
            tools=self.tools,
            tool_executor=self.tool_executor,
            context_pipeline=self.context_pipeline,
            checkpoints=self.checkpoints,
        )

    async def run_async(self, prompt: str, task_id: str | None = None) -> AgentRunResult:
        self._log_run()
        return await self.loop.run_async(prompt, task_id=task_id)

    def run(self, prompt: str, task_id: str | None = None) -> AgentRunResult:
        self._log_run()
        return self.loop.run(prompt, task_id=task_id)

    async def resume_async(self, task_id: str) -> AgentRunResult:
        state = self.checkpoints.load(task_id)
        if state is None:
            raise FileNotFoundError(f"No checkpoint found for task '{task_id}'.")
        return await self.loop.run_state_async(state)

    def resume(self, task_id: str) -> AgentRunResult:
        return self.loop.run_state(self._load_state(task_id))

    def _load_state(self, task_id: str):
        state = self.checkpoints.load(task_id)
        if state is None:
            raise FileNotFoundError(f"No checkpoint found for task '{task_id}'.")
        return state

    def _log_run(self) -> None:
        self.logger.info(
            "session run",
            extra={
                "event": "session.run",
                "details": {"provider": self.provider.provider_name},
            },
        )
