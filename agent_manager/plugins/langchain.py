"""Optional adapters for LangChain-style tools."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable, Mapping
from typing import Any

from agent_manager.plugins.base import Plugin
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec, normalize_tool_result


class LangChainToolAdapter(BaseTool):
    """Adapt a LangChain-like tool object into the agent_manager tool contract."""

    def __init__(self, tool: Any) -> None:
        self.tool = tool
        self.spec = ToolSpec(
            name=str(getattr(tool, "name", "langchain_tool")),
            description=str(getattr(tool, "description", "")),
            input_schema=self._input_schema(tool),
            tags=["integration", "langchain"],
        )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        result = await self._invoke_tool(arguments)
        return normalize_tool_result(self.spec.name, result)

    async def _invoke_tool(self, arguments: dict[str, Any]) -> Any:
        if hasattr(self.tool, "ainvoke"):
            value = self.tool.ainvoke(arguments)
            if inspect.isawaitable(value):
                return await value
            return value
        if hasattr(self.tool, "invoke"):
            return await asyncio.to_thread(self.tool.invoke, arguments)
        if hasattr(self.tool, "run"):
            return await asyncio.to_thread(self.tool.run, arguments)
        if callable(self.tool):
            return await asyncio.to_thread(self.tool, arguments)
        raise TypeError(
            f"LangChain-like tool '{self.spec.name}' does not expose invoke/ainvoke/run/callable."
        )

    def _input_schema(self, tool: Any) -> dict[str, Any]:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            if hasattr(args_schema, "model_json_schema"):
                return dict(args_schema.model_json_schema())
            if hasattr(args_schema, "schema"):
                return dict(args_schema.schema())

        raw_args = getattr(tool, "args", None)
        if isinstance(raw_args, Mapping):
            return {
                "type": "object",
                "properties": {
                    str(name): {"type": "string"}
                    for name in raw_args.keys()
                },
            }
        return {"type": "object", "properties": {}}


class LangChainToolsPlugin(Plugin):
    """Register one or more LangChain-like tools on an AgentSession."""

    name = "langchain-tools"
    description = "Expose LangChain-style tools through the unified tool registry."

    def __init__(self, tools: Iterable[Any], *, replace: bool = True) -> None:
        self._tools = list(tools)
        self._replace = replace

    def register(self, target: Any) -> None:
        for tool in self._tools:
            target.register_tool(
                LangChainToolAdapter(tool),
                replace=self._replace,
            )
