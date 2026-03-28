"""Tool registry used by the runtime."""

from __future__ import annotations

from agent_manager.errors import ToolNotFoundError
from agent_manager.tools.base import BaseTool, ToolSpec


class ToolRegistry:
    """Register and expose tools without coupling them to providers."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.spec.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.") from exc

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def definitions(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def provider_definitions(self) -> list[dict]:
        return [spec.to_dict() for spec in self.definitions()]

