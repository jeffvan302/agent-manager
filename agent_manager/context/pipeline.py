"""Pre-call context pipeline with registry-driven steps."""

from __future__ import annotations

from collections.abc import Mapping

from agent_manager.async_utils import run_sync
from agent_manager.config import RuntimeConfig
from agent_manager.context.assembler import ContextAssembler
from agent_manager.context.functions import (
    PreCallFunction,
    PreCallFunctionRegistry,
    PreCallRuntime,
    default_pre_call_functions,
    run_pre_call_function,
)
from agent_manager.context.sections import PreparedTurn
from agent_manager.memory.base import BaseMemoryStore
from agent_manager.memory.retrieval import BaseRetriever
from agent_manager.types import LoopState


class PreCallPipeline:
    """Configurable context preparation pipeline for provider calls."""

    def __init__(
        self,
        *,
        assembler: ContextAssembler | None = None,
        registry: PreCallFunctionRegistry | None = None,
        configured_steps: list[str] | None = None,
        retriever: BaseRetriever | None = None,
        memory_store: BaseMemoryStore | None = None,
        custom_functions: Mapping[str, PreCallFunction] | None = None,
    ) -> None:
        self.assembler = assembler or ContextAssembler()
        self.registry = registry or PreCallFunctionRegistry(default_pre_call_functions())
        if custom_functions:
            self.registry.register_many(custom_functions)
        self.configured_steps = list(configured_steps) if configured_steps is not None else None
        self.runtime = PreCallRuntime(
            assembler=self.assembler,
            retriever=retriever,
            memory_store=memory_store,
        )

    def register_function(self, name: str, fn: PreCallFunction) -> None:
        self.registry.register(name, fn)

    async def prepare_async(self, state: LoopState, config: RuntimeConfig) -> PreparedTurn:
        prepared = self.assembler.new_prepared_turn()
        step_names = self.configured_steps or list(config.context.pre_call_functions)
        prepared.metadata["pre_call_functions"] = list(step_names)

        for name, fn in self.registry.resolve(step_names):
            prepared = await run_pre_call_function(
                fn,
                state,
                prepared,
                config,
                self.runtime,
            )
            prepared.metadata.setdefault("executed_steps", []).append(name)

        if not prepared.messages:
            prepared = self.assembler.finalize_turn(state, prepared, config)
        elif prepared.token_estimate <= 0:
            prepared.token_estimate = self.assembler.token_counter.count_messages(
                prepared.messages
            )
        return prepared

    def prepare(self, state: LoopState, config: RuntimeConfig) -> PreparedTurn:
        return run_sync(self.prepare_async(state, config))
