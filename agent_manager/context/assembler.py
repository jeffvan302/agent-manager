"""Helpers for inspectable context assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from agent_manager.config import RuntimeConfig
from agent_manager.context.budget import (
    SimpleTokenCounter,
    TokenBudget,
    resolve_model_budget,
)
from agent_manager.context.sections import PreparedTurn
from agent_manager.context.summarizer import SimpleSummarizer
from agent_manager.memory.base import BaseMemoryStore
from agent_manager.memory.retrieval import BaseRetriever
from agent_manager.types import ContextSection, LoopState, Message


class ContextAssembler:
    """Build and render inspectable context sections."""

    def __init__(
        self,
        token_counter: SimpleTokenCounter | None = None,
        summarizer: SimpleSummarizer | None = None,
    ) -> None:
        self.token_counter = token_counter or SimpleTokenCounter()
        self.summarizer = summarizer or SimpleSummarizer()

    def new_prepared_turn(self) -> PreparedTurn:
        return PreparedTurn()

    def build_core_sections(
        self,
        state: LoopState,
        config: RuntimeConfig,
    ) -> list[ContextSection]:
        sections: list[ContextSection] = []

        if config.system_prompt:
            sections.append(
                ContextSection(
                    key="system_instructions",
                    title="System Instructions",
                    content=config.system_prompt,
                    priority=100,
                )
            )

        sections.append(
            ContextSection(
                key="goal",
                title="Goal",
                content=state.goal,
                priority=90,
            )
        )

        if state.metadata.get("current_plan"):
            sections.append(
                ContextSection(
                    key="current_plan",
                    title="Current Plan",
                    content=str(state.metadata["current_plan"]),
                    priority=85,
                )
            )

        sections.append(
            ContextSection(
                key="execution_constraints",
                title="Execution Constraints",
                content=(
                    f"Profile: {config.profile}\n"
                    f"Step: {state.step_index}\n"
                    f"Max output tokens: {resolve_model_budget(config).max_output_tokens}\n"
                    f"Model: {config.provider.name}/{config.provider.model}"
                ),
                priority=82,
            )
        )
        return self.with_estimates(sections, config)

    def build_recent_message_sections(
        self,
        state: LoopState,
        config: RuntimeConfig,
    ) -> list[ContextSection]:
        history_window = max(config.context.history_window, 0)
        if history_window == 0:
            return []

        recent_messages = state.messages[-history_window:]
        start_index = max(len(state.messages) - len(recent_messages), 0)
        sections: list[ContextSection] = []
        for offset, message in enumerate(recent_messages, start=1):
            order = start_index + offset
            sections.append(
                ContextSection(
                    key=f"recent_message_{order}",
                    title=f"Recent {message.role.title()} Message",
                    content=message.content,
                    priority=self._message_priority(message),
                    metadata={
                        "kind": "message",
                        "message": message.to_dict(),
                        "order": order,
                        "role": message.role,
                    },
                )
            )
        return self.with_estimates(sections, config)

    def build_summary_section(
        self,
        state: LoopState,
        config: RuntimeConfig,
    ) -> ContextSection | None:
        if state.summaries:
            return self._estimated_section(
                ContextSection(
                    key="summary",
                    title="Summary",
                    content="\n".join(state.summaries),
                    priority=80,
                ),
                config,
            )

        history_window = max(config.context.history_window, 0)
        trigger = max(config.context.summary_trigger_messages, history_window)
        if len(state.messages) <= trigger or history_window <= 0:
            return None

        summary = self.summarizer.summarize_messages(state.messages[:-history_window])
        if not summary:
            return None

        return self._estimated_section(
            ContextSection(
                key="summary",
                title="Summary",
                content=summary,
                priority=78,
            ),
            config,
        )

    async def build_retrieval_sections(
        self,
        state: LoopState,
        config: RuntimeConfig,
        retriever: BaseRetriever | None,
    ) -> list[ContextSection]:
        if retriever is None:
            return []

        query = self._lookup_query(state)
        if not query:
            return []

        results = await asyncio.to_thread(
            retriever.retrieve,
            query,
            max(config.context.retrieval_top_k, 0),
            None,
        )
        sections = [
            ContextSection(
                key=f"retrieved_{index}",
                title=f"Retrieved Knowledge {index}",
                content=result.content,
                priority=70,
                metadata={
                    "kind": "retrieval",
                    "retrieval_id": result.id,
                    "score": result.score,
                    "source_metadata": dict(result.metadata),
                },
            )
            for index, result in enumerate(results, start=1)
        ]
        return self.with_estimates(sections, config)

    async def build_memory_sections(
        self,
        state: LoopState,
        config: RuntimeConfig,
        memory_store: BaseMemoryStore | None,
    ) -> list[ContextSection]:
        if memory_store is None:
            return []

        query = self._lookup_query(state)
        if not query:
            return []

        entries = await asyncio.to_thread(memory_store.query, query)
        sections = [
            ContextSection(
                key=f"memory_fact_{index}",
                title=f"Memory Fact {index}",
                content=entry.value,
                priority=68,
                metadata={
                    "kind": "memory",
                    "memory_key": entry.key,
                    "source": entry.source,
                    "scope": entry.scope,
                    "confidence": entry.confidence,
                    "tags": list(entry.tags),
                    "entry_metadata": dict(entry.metadata),
                },
            )
            for index, entry in enumerate(entries[: max(config.context.max_memory_facts, 0)], start=1)
        ]
        return self.with_estimates(sections, config)

    def fit_sections_to_budget(
        self,
        sections: list[ContextSection],
        config: RuntimeConfig,
    ) -> tuple[list[ContextSection], list[str]]:
        profile = resolve_model_budget(config)
        budget = TokenBudget(
            max_context_tokens=profile.max_context_tokens,
            reserved_output_tokens=profile.max_output_tokens,
        )
        fitted: list[ContextSection] = []
        dropped: list[str] = []
        current_tokens = 0

        for section in sorted(sections, key=lambda item: item.priority, reverse=True):
            estimate = section.token_estimate or 0
            if current_tokens + estimate <= budget.available_input_tokens:
                fitted.append(section)
                current_tokens += estimate
            else:
                dropped.append(section.key)

        fitted.sort(key=self._section_sort_key)
        return fitted, dropped

    def render_messages(
        self,
        state: LoopState,
        sections: list[ContextSection],
    ) -> list[Message]:
        system_sections: list[ContextSection] = []
        message_sections: list[ContextSection] = []

        for section in sections:
            if section.metadata.get("kind") == "message":
                message_sections.append(section)
            else:
                system_sections.append(section)

        messages: list[Message] = []
        if system_sections:
            system_parts = [
                f"{section.title}:\n{section.content}"
                for section in sorted(system_sections, key=self._section_sort_key)
                if section.content
            ]
            if system_parts:
                messages.append(Message(role="system", content="\n\n".join(system_parts)))

        if message_sections:
            for section in sorted(
                message_sections,
                key=lambda item: int(item.metadata.get("order", 0)),
            ):
                raw_message = section.metadata.get("message")
                if isinstance(raw_message, dict):
                    messages.append(Message.from_dict(raw_message))
        else:
            messages.append(Message(role="user", content=state.goal))

        return messages

    def finalize_turn(
        self,
        state: LoopState,
        prepared: PreparedTurn,
        config: RuntimeConfig,
    ) -> PreparedTurn:
        if not prepared.sections and not prepared.metadata.get("budget_applied"):
            prepared.sections = self.build_core_sections(state, config)
        if not prepared.messages:
            prepared.messages = self.render_messages(state, prepared.sections)
        prepared.token_estimate = self._token_counter_for(config).count_messages(
            prepared.messages
        )
        return prepared

    def with_estimates(
        self,
        sections: Iterable[ContextSection],
        config: RuntimeConfig,
    ) -> list[ContextSection]:
        return [self._estimated_section(section, config) for section in sections]

    def _estimated_section(
        self,
        section: ContextSection,
        config: RuntimeConfig,
    ) -> ContextSection:
        if section.token_estimate is None:
            section.token_estimate = self._token_counter_for(config).estimate_text(
                section.content
            )
        return section

    def _token_counter_for(self, config: RuntimeConfig) -> SimpleTokenCounter:
        profile = resolve_model_budget(config)
        return SimpleTokenCounter(chars_per_token=profile.chars_per_token)

    def _lookup_query(self, state: LoopState) -> str:
        recent_user_messages = [
            message.content
            for message in state.messages
            if message.role == "user" and message.content
        ]
        if recent_user_messages:
            return recent_user_messages[-1]
        return state.goal

    def _message_priority(self, message: Message) -> int:
        if message.role == "user":
            return 76
        if message.role == "assistant":
            return 74
        if message.role == "tool":
            return 72
        return 70

    def _section_sort_key(self, section: ContextSection) -> tuple[int, str]:
        order = int(section.metadata.get("order", 0))
        return (-section.priority, order, section.key)
