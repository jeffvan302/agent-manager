"""Planner placeholder."""

from __future__ import annotations


class Planner:
    """A hook point for planning logic added in later phases."""

    def build_plan(self, goal: str) -> list[str]:
        del goal
        return []

