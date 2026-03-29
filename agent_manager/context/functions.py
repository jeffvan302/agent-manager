"""Pre-call context pipeline functions and registries."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass

from agent_manager.config import RuntimeConfig
from agent_manager.context.assembler import ContextAssembler
from agent_manager.context.sections import PreparedTurn
from agent_manager.errors import ConfigurationError
from agent_manager.memory.base import BaseMemoryStore
from agent_manager.memory.retrieval import BaseRetriever
from agent_manager.types import LoopState

PreCallFunction = Callable[
    [LoopState, PreparedTurn, RuntimeConfig, "PreCallRuntime"],
    PreparedTurn | Awaitable[PreparedTurn],
]


@dataclass(slots=True)
class PreCallRuntime:
    assembler: ContextAssembler
    retriever: BaseRetriever | None = None
    memory_store: BaseMemoryStore | None = None


class PreCallFunctionRegistry:
    """Register named functions used by the pre-call context pipeline."""

    def __init__(self, functions: Mapping[str, PreCallFunction] | None = None) -> None:
        self._functions: dict[str, PreCallFunction] = {}
        for name, fn in (functions or {}).items():
            self.register(name, fn)

    def register(self, name: str, fn: PreCallFunction) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise ConfigurationError("Pre-call function names must not be empty.")
        self._functions[normalized_name] = fn

    def register_many(self, functions: Mapping[str, PreCallFunction]) -> None:
        for name, fn in functions.items():
            self.register(name, fn)

    def get(self, name: str) -> PreCallFunction:
        try:
            return self._functions[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"Pre-call function '{name}' is not registered. "
                f"Available functions: {', '.join(self.names())}"
            ) from exc

    def resolve(self, names: Iterable[str]) -> list[tuple[str, PreCallFunction]]:
        return [(name, self.get(name)) for name in names]

    def names(self) -> list[str]:
        return sorted(self._functions.keys())


async def run_pre_call_function(
    fn: PreCallFunction,
    state: LoopState,
    prepared: PreparedTurn,
    config: RuntimeConfig,
    runtime: PreCallRuntime,
) -> PreparedTurn:
    value = fn(state, prepared, config, runtime)
    if inspect.isawaitable(value):
        value = await value
    return value


async def collect_recent_messages(
    state: LoopState,
    prepared: PreparedTurn,
    config: RuntimeConfig,
    runtime: PreCallRuntime,
) -> PreparedTurn:
    prepared.sections.extend(runtime.assembler.build_core_sections(state, config))
    prepared.sections.extend(runtime.assembler.build_recent_message_sections(state, config))
    return prepared


async def summarize_history(
    state: LoopState,
    prepared: PreparedTurn,
    config: RuntimeConfig,
    runtime: PreCallRuntime,
) -> PreparedTurn:
    summary = runtime.assembler.build_summary_section(state, config)
    if summary is not None:
        prepared.sections.append(summary)
    return prepared


async def inject_retrieval(
    state: LoopState,
    prepared: PreparedTurn,
    config: RuntimeConfig,
    runtime: PreCallRuntime,
) -> PreparedTurn:
    sections = await runtime.assembler.build_retrieval_sections(
        state,
        config,
        runtime.retriever,
    )
    prepared.sections.extend(sections)
    return prepared


async def inject_memory_facts(
    state: LoopState,
    prepared: PreparedTurn,
    config: RuntimeConfig,
    runtime: PreCallRuntime,
) -> PreparedTurn:
    sections = await runtime.assembler.build_memory_sections(
        state,
        config,
        runtime.memory_store,
    )
    prepared.sections.extend(sections)
    return prepared


async def apply_token_budget(
    state: LoopState,
    prepared: PreparedTurn,
    config: RuntimeConfig,
    runtime: PreCallRuntime,
) -> PreparedTurn:
    del state
    prepared.sections = runtime.assembler.with_estimates(prepared.sections, config)
    prepared.sections, prepared.dropped_sections = runtime.assembler.fit_sections_to_budget(
        prepared.sections,
        config,
    )
    prepared.metadata["budget_applied"] = True
    return prepared


async def finalize_messages(
    state: LoopState,
    prepared: PreparedTurn,
    config: RuntimeConfig,
    runtime: PreCallRuntime,
) -> PreparedTurn:
    return runtime.assembler.finalize_turn(state, prepared, config)


def default_pre_call_functions() -> dict[str, PreCallFunction]:
    return {
        "collect_recent_messages": collect_recent_messages,
        "summarize_history": summarize_history,
        "inject_retrieval": inject_retrieval,
        "inject_memory_facts": inject_memory_facts,
        "apply_token_budget": apply_token_budget,
        "finalize_messages": finalize_messages,
    }
