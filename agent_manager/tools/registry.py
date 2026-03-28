"""Tool registry used by the runtime."""

from __future__ import annotations

from collections.abc import Iterable

from agent_manager.errors import ToolNotFoundError
from agent_manager.tools.base import BaseTool, FunctionTool, ToolHandler, ToolSpec


class ToolRegistry:
    """Register and expose tools without coupling them to providers."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool, *, replace: bool = True) -> BaseTool:
        if not replace and tool.spec.name in self._tools:
            raise ValueError(f"Tool '{tool.spec.name}' is already registered.")
        self._tools[tool.spec.name] = tool
        return tool

    def register_many(self, tools: Iterable[BaseTool], *, replace: bool = True) -> None:
        for tool in tools:
            self.register(tool, replace=replace)

    def register_callable(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        replace: bool = True,
    ) -> FunctionTool:
        tool = FunctionTool(spec, handler)
        self.register(tool, replace=replace)
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.") from exc

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def definitions(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def provider_definitions(self) -> list[dict]:
        return [spec.to_dict() for spec in self.definitions()]

    def __iter__(self):
        return iter(self._tools.values())
