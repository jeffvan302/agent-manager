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
from agent_manager.tools.builtins.web import WebSearchTool
from agent_manager.tools.registry import ToolRegistry
from agent_manager.tools.web_search import (
    BaseWebSearcher,
    BraveWebSearcher,
    DuckDuckGoWebSearcher,
    GoogleSearchToolWebSearcher,
    SerpAPIWebSearcher,
    TavilyWebSearcher,
    available_web_search_backends,
)


def default_builtin_tools(
    *,
    retriever: BaseRetriever | None = None,
    web_searcher: BaseWebSearcher | None = None,
    include_web_search: bool = True,
) -> list[BaseTool]:
    tools: list[BaseTool] = [
        ListDirectoryTool(),
        ReadFileTool(),
        WriteFileTool(),
        RunShellCommandTool(),
        HttpRequestTool(),
    ]
    if include_web_search:
        tools.append(WebSearchTool(web_searcher or GoogleSearchToolWebSearcher()))
    if retriever is not None:
        tools.append(RetrieveDocumentsTool(retriever))
    return tools


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    retriever: BaseRetriever | None = None,
    web_searcher: BaseWebSearcher | None = None,
    include_web_search: bool = True,
    replace: bool = True,
) -> ToolRegistry:
    registry.register_many(
        default_builtin_tools(
            retriever=retriever,
            web_searcher=web_searcher,
            include_web_search=include_web_search,
        ),
        replace=replace,
    )
    return registry


__all__ = [
    "HttpRequestTool",
    "ListDirectoryTool",
    "ReadFileTool",
    "RetrieveDocumentsTool",
    "RunShellCommandTool",
    "WebSearchTool",
    "WriteFileTool",
    "BaseWebSearcher",
    "BraveWebSearcher",
    "DuckDuckGoWebSearcher",
    "GoogleSearchToolWebSearcher",
    "SerpAPIWebSearcher",
    "TavilyWebSearcher",
    "available_web_search_backends",
    "default_builtin_tools",
    "register_builtin_tools",
]
