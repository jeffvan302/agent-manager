"""Build inspectable context sections before a provider call."""

from __future__ import annotations

from agent_manager.config import RuntimeConfig
from agent_manager.context.budget import SimpleTokenCounter, TokenBudget
from agent_manager.context.sections import PreparedTurn
from agent_manager.context.summarizer import SimpleSummarizer
from agent_manager.types import ContextSection, LoopState, Message


class ContextAssembler:
    """Turn loop state into a small, inspectable request payload."""

    def __init__(
        self,
        token_counter: SimpleTokenCounter | None = None,
        summarizer: SimpleSummarizer | None = None,
        history_window: int = 8,
    ) -> None:
        self.token_counter = token_counter or SimpleTokenCounter()
        self.summarizer = summarizer or SimpleSummarizer()
        self.history_window = history_window

    def prepare(self, state: LoopState, config: RuntimeConfig) -> PreparedTurn:
        sections = self._build_sections(state, config)
        budget = TokenBudget(
            max_context_tokens=config.runtime.max_context_tokens,
            reserved_output_tokens=config.runtime.max_output_tokens,
        )
        fitted_sections, dropped_sections = self._fit_to_budget(sections, budget)
        messages = self._render_messages(state, fitted_sections)
        token_estimate = self.token_counter.count_messages(messages)
        return PreparedTurn(
            messages=messages,
            sections=fitted_sections,
            token_estimate=token_estimate,
            dropped_sections=dropped_sections,
        )

    def _build_sections(self, state: LoopState, config: RuntimeConfig) -> list[ContextSection]:
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

        if state.summaries:
            sections.append(
                ContextSection(
                    key="summary",
                    title="Summary",
                    content="\n".join(state.summaries),
                    priority=80,
                )
            )
        elif len(state.messages) > self.history_window:
            summary = self.summarizer.summarize_messages(state.messages[:-self.history_window])
            if summary:
                sections.append(
                    ContextSection(
                        key="summary",
                        title="Summary",
                        content=summary,
                        priority=75,
                    )
                )

        recent_messages = state.messages[-self.history_window :]
        for index, message in enumerate(recent_messages, start=1):
            sections.append(
                ContextSection(
                    key=f"recent_message_{index}",
                    title=f"Recent {message.role.title()} Message",
                    content=message.content,
                    priority=60,
                    metadata={"role": message.role},
                )
            )

        for section in sections:
            if section.token_estimate is None:
                section.token_estimate = self.token_counter.estimate_text(section.content)
        return sections

    def _fit_to_budget(
        self,
        sections: list[ContextSection],
        budget: TokenBudget,
    ) -> tuple[list[ContextSection], list[str]]:
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

        fitted.sort(key=lambda item: item.priority, reverse=True)
        return fitted, dropped

    def _render_messages(
        self,
        state: LoopState,
        sections: list[ContextSection],
    ) -> list[Message]:
        system_parts = [
            f"{section.title}:\n{section.content}"
            for section in sections
            if section.key in {"system_instructions", "goal", "summary"}
        ]

        messages: list[Message] = []
        if system_parts:
            messages.append(Message(role="system", content="\n\n".join(system_parts)))

        recent_messages = state.messages[-self.history_window :]
        if recent_messages:
            messages.extend(recent_messages)
        else:
            messages.append(Message(role="user", content=state.goal))

        return messages

