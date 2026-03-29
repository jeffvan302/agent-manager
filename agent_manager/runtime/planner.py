"""Lightweight task planner used by the runtime loop."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Planner:
    """Build a small actionable plan from the current goal and toolset."""

    max_steps: int = 5
    research_keywords: tuple[str, ...] = (
        "research",
        "search",
        "find",
        "compare",
        "look up",
        "investigate",
    )
    coding_keywords: tuple[str, ...] = (
        "code",
        "implement",
        "fix",
        "debug",
        "refactor",
        "write",
        "build",
        "test",
    )
    documentation_keywords: tuple[str, ...] = (
        "document",
        "docs",
        "readme",
        "requirements",
        "phase",
        "spec",
        "summarize",
    )
    generic_opening_steps: list[str] = field(
        default_factory=lambda: [
            "Understand the goal and identify the highest-priority context.",
        ]
    )

    def build_plan(
        self,
        goal: str,
        *,
        available_tools: list[str] | None = None,
    ) -> list[str]:
        normalized_goal = goal.strip().lower()
        tools = set(available_tools or [])
        plan: list[str] = list(self.generic_opening_steps)

        if self._matches_any(normalized_goal, self.research_keywords):
            if "web_search" in tools:
                plan.append("Search approved sources and capture the most relevant findings.")
            else:
                plan.append("Gather the most relevant external and internal information.")

        if self._matches_any(normalized_goal, self.coding_keywords):
            if {"read_file", "write_file"} & tools:
                plan.append("Inspect the affected files and update the implementation.")
            else:
                plan.append("Inspect the implementation details and prepare the needed code changes.")
            if "run_shell" in tools:
                plan.append("Run validation commands or tests after making changes.")

        if self._matches_any(normalized_goal, self.documentation_keywords):
            plan.append("Draft or update the documentation to match the implementation.")

        if len(plan) == 1:
            plan.extend(
                [
                    "Assemble the most relevant context for the current task.",
                    "Produce the next best response or action with the available tools.",
                ]
            )

        plan.append("Review the result and stop when the goal is satisfied.")
        return self._dedupe_and_limit(plan)

    def _matches_any(self, goal: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in goal for keyword in keywords)

    def _dedupe_and_limit(self, steps: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for step in steps:
            if step in seen:
                continue
            seen.add(step)
            deduped.append(step)
            if len(deduped) >= self.max_steps:
                break
        return deduped
