"""High-level session object exposed to library users."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agent_manager.config import RuntimeConfig, load_config
from agent_manager.context.functions import PreCallFunction
from agent_manager.context.pipeline import PreCallPipeline
from agent_manager.memory.base import BaseMemoryStore
from agent_manager.memory.retrieval import BaseRetriever
from agent_manager.observability import configure_logging, get_logger
from agent_manager.plugins.base import Plugin
from agent_manager.plugins.registry import PluginRegistry
from agent_manager.providers.base import BaseProvider
from agent_manager.providers.factory import build_provider, register_provider
from agent_manager.runtime.loop import AgentLoop
from agent_manager.runtime.planner import Planner
from agent_manager.state.checkpoint import CheckpointManager
from agent_manager.state.store import JsonFileStateStore, SqliteStateStore, StateStore
from agent_manager.tools.builtins import register_builtin_tools
from agent_manager.tools.base import BaseTool
from agent_manager.tools.executor import ToolExecutor
from agent_manager.tools.policies import (
    DEFAULT_PROFILES,
    ApprovalHook,
    PolicyEngine,
    ToolPolicyProfile,
)
from agent_manager.tools.registry import ToolRegistry
from agent_manager.tools.web_search import BaseWebSearcher
from agent_manager.types import AgentRunResult, StructuredOutputSpec


class AgentSession:
    """User-facing runtime facade for one configured agent manager session."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        provider: BaseProvider | None = None,
        tools: ToolRegistry | None = None,
        state_store: StateStore | None = None,
        retriever: BaseRetriever | None = None,
        memory_store: BaseMemoryStore | None = None,
        include_builtin_tools: bool = True,
        working_directory: str | Path | None = None,
        tool_context_metadata: dict[str, Any] | None = None,
        context_pipeline: PreCallPipeline | None = None,
        pre_call_functions: dict[str, PreCallFunction] | None = None,
        approval_hook: ApprovalHook | None = None,
        web_searcher: BaseWebSearcher | None = None,
        plugins: list[Plugin] | PluginRegistry | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.config = config or load_config()
        configure_logging(
            level=self.config.logging.level,
            json_output=self.config.logging.json_output,
        )
        self.logger = get_logger("runtime.session")
        self.provider = provider or build_provider(self.config.provider)
        self.working_directory = Path(working_directory or Path.cwd()).resolve(strict=False)
        self.tool_context_metadata = dict(tool_context_metadata or {})
        self.tool_context_metadata.setdefault(
            "filesystem_roots",
            [str(self.working_directory)],
        )
        self.tools = self._build_tool_registry(
            tools=tools,
            retriever=retriever,
            web_searcher=web_searcher,
            include_builtin_tools=include_builtin_tools,
        )
        self.policy_engine = self._build_policy_engine(approval_hook=approval_hook)
        self.tool_executor = ToolExecutor(self.tools, self.policy_engine)
        self.state_store = state_store or self._build_state_store()
        self.checkpoints = CheckpointManager(self.state_store)
        self.retriever = retriever
        self.memory_store = memory_store
        self.retrievers: dict[str, BaseRetriever] = {}
        if retriever is not None:
            self.retrievers["default"] = retriever
        self.context_pipeline = context_pipeline or PreCallPipeline(
            retriever=retriever,
            memory_store=memory_store,
            custom_functions=pre_call_functions,
        )
        if context_pipeline is not None and pre_call_functions:
            for name, fn in pre_call_functions.items():
                self.context_pipeline.register_function(name, fn)
        self.plugins = PluginRegistry()
        if isinstance(plugins, PluginRegistry):
            self.plugins = plugins
        elif plugins:
            self.plugins.register_many(plugins)
        self.planner = planner or Planner()
        self.providers: dict[str, type[BaseProvider]] = {
            self.provider.provider_name: type(self.provider)
        }
        self.loop = AgentLoop(
            config=self.config,
            provider=self.provider,
            tools=self.tools,
            tool_executor=self.tool_executor,
            context_pipeline=self.context_pipeline,
            checkpoints=self.checkpoints,
            working_directory=self.working_directory,
            tool_context_metadata=self.tool_context_metadata,
            planner=self.planner,
        )
        self.plugins.apply_all(self)

    async def run_async(
        self,
        prompt: str,
        task_id: str | None = None,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> AgentRunResult:
        self._log_run()
        return await self.loop.run_async(
            prompt,
            task_id=task_id,
            structured_output=structured_output,
        )

    def run(
        self,
        prompt: str,
        task_id: str | None = None,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> AgentRunResult:
        self._log_run()
        return self.loop.run(
            prompt,
            task_id=task_id,
            structured_output=structured_output,
        )

    async def stream_async(
        self,
        prompt: str,
        task_id: str | None = None,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        self._log_run()
        async for event in self.loop.stream_async(
            prompt,
            task_id=task_id,
            structured_output=structured_output,
        ):
            yield event

    async def resume_async(
        self,
        task_id: str,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> AgentRunResult:
        state = self.checkpoints.load(task_id)
        if state is None:
            raise FileNotFoundError(f"No checkpoint found for task '{task_id}'.")
        return await self.loop.run_state_async(
            state,
            structured_output=structured_output,
        )

    def resume(
        self,
        task_id: str,
        *,
        structured_output: StructuredOutputSpec | dict[str, Any] | None = None,
    ) -> AgentRunResult:
        return self.loop.run_state(
            self._load_state(task_id),
            structured_output=structured_output,
        )

    def request_interrupt(self) -> None:
        self.loop.request_interrupt()

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

    def _build_tool_registry(
        self,
        *,
        tools: ToolRegistry | None,
        retriever: BaseRetriever | None,
        web_searcher: BaseWebSearcher | None,
        include_builtin_tools: bool,
    ) -> ToolRegistry:
        if tools is not None and not include_builtin_tools:
            return tools

        registry = ToolRegistry()
        if include_builtin_tools:
            register_builtin_tools(
                registry,
                retriever=retriever,
                web_searcher=web_searcher,
            )
        if tools is not None:
            registry.register_many(tools.all(), replace=True)
        return registry

    def _build_state_store(self) -> StateStore:
        backend = self.config.resolved_checkpoint_backend()
        if backend == "sqlite":
            try:
                return SqliteStateStore(self.config.resolved_state_path())
            except Exception:
                self.logger.warning(
                    "SQLite state store unavailable (filesystem may not support locking), "
                    "falling back to JSON file store."
                )
                return JsonFileStateStore(self.config.resolved_state_dir())
        return JsonFileStateStore(self.config.resolved_state_dir())

    def _build_policy_engine(
        self,
        *,
        approval_hook: ApprovalHook | None,
    ) -> PolicyEngine:
        base_profile = DEFAULT_PROFILES.get(
            self.config.profile,
            ToolPolicyProfile(name=self.config.profile, allow_all=True),
        )
        configured_policy = self.config.tool_policy
        configured_allowed_tools = {
            tool_name
            for tool_name in configured_policy.allowed_tools
            if str(tool_name).strip()
        }
        profile = ToolPolicyProfile(
            name=base_profile.name,
            allow_all=base_profile.allow_all and not configured_allowed_tools,
            allowed_tools=set(base_profile.allowed_tools) | set(configured_allowed_tools),
            denied_tools=set(base_profile.denied_tools) | {
                tool_name
                for tool_name in configured_policy.denied_tools
                if str(tool_name).strip()
            },
            denied_tags=set(base_profile.denied_tags) | {
                tag
                for tag in configured_policy.denied_tags
                if str(tag).strip()
            },
            allowed_permissions=set(base_profile.allowed_permissions),
            denied_permissions={
                *base_profile.denied_permissions,
                *(
                permission
                for permission in configured_policy.denied_permissions
                if str(permission).strip()
                ),
            },
        )
        return PolicyEngine(profile, approval_hook=approval_hook)

    def register_tool(self, tool: BaseTool, *, replace: bool = True) -> None:
        self.tools.register(tool, replace=replace)

    def register_pre_call_function(self, name: str, fn: PreCallFunction) -> None:
        self.context_pipeline.register_function(name, fn)

    def register_retriever(
        self,
        name: str,
        retriever: BaseRetriever,
        *,
        make_default: bool = False,
    ) -> None:
        self.retrievers[name] = retriever
        if make_default or self.context_pipeline.runtime.retriever is None:
            self.retriever = retriever
            self.context_pipeline.runtime.retriever = retriever

    def register_provider_adapter(
        self,
        name: str,
        provider_cls: type[BaseProvider],
    ) -> None:
        register_provider(name, provider_cls)
        self.providers[name] = provider_cls

    def register_plugin(self, plugin: Plugin) -> None:
        self.plugins.register(plugin)
        plugin.assert_available()
        plugin.register(self)
