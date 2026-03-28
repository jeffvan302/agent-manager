"""Built-in tools exposed by the runtime."""

from __future__ import annotations

from agent_manager.memory.retrieval import BaseRetriever
from agent_manager.tools.base import BaseTool
from agent_manager.tools.builtins.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from agent_manager.tools.builtins.http import HttpRequestTool
from agent_manager.tools.builtins.retrieval import RetrieveDocumentsTool
from agent_manager.tools.builtins.shell import RunShellCommandTool
from agent_manager.tools.registry import ToolRegistry


def default_builtin_tools(*, retriever: BaseRetriever | None = None) -> list[BaseTool]:
    tools: list[BaseTool] = [
        ListDirectoryTool(),
        ReadFileTool(),
        WriteFileTool(),
        RunShellCommandTool(),
        HttpRequestTool(),
    ]
    if retriever is not None:
        tools.append(RetrieveDocumentsTool(retriever))
    return tools


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    retriever: BaseRetriever | None = None,
    replace: bool = True,
) -> ToolRegistry:
    registry.register_many(
        default_builtin_tools(retriever=retriever),
        replace=replace,
    )
    return registry


__all__ = [
    "HttpRequestTool",
    "ListDirectoryTool",
    "ReadFileTool",
    "RetrieveDocumentsTool",
    "RunShellCommandTool",
    "WriteFileTool",
    "default_builtin_tools",
    "register_builtin_tools",
]
