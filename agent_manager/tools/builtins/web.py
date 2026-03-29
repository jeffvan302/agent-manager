"""Built-in web search tool."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from agent_manager.tools.web_search import BaseWebSearcher


class WebSearchTool(BaseTool):
    spec = ToolSpec(
        name="web_search",
        description="Search the web through the configured search backend.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {"type": "array"},
            },
            "required": ["query", "results"],
        },
        tags=["network", "web"],
        permissions=["network:request"],
        timeout_seconds=20.0,
    )

    def __init__(self, searcher: BaseWebSearcher) -> None:
        self.searcher = searcher

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        query = str(arguments["query"])
        limit = max(int(arguments.get("limit", 5)), 1)
        results = await asyncio.to_thread(self.searcher.search, query, limit=limit)
        return ToolResult(
            tool_name=self.spec.name,
            ok=True,
            output={
                "query": query,
                "results": [result.to_dict() for result in results],
            },
            metadata={"result_count": len(results)},
        )
