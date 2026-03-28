"""Built-in retrieval tool."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from agent_manager.memory.retrieval import BaseRetriever
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec


class RetrieveDocumentsTool(BaseTool):
    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever
        self.spec = ToolSpec(
            name="retrieve_documents",
            description="Retrieve relevant knowledge chunks for the current task.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "metadata_filter": {"type": "object"},
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
            tags=["retrieval", "rag"],
            permissions=["memory:read"],
        )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        query = str(arguments["query"])
        top_k = max(int(arguments.get("top_k", 5)), 1)
        metadata_filter = arguments.get("metadata_filter")
        normalized_filter = (
            {str(key): value for key, value in metadata_filter.items()}
            if isinstance(metadata_filter, Mapping)
            else None
        )

        results = await asyncio.to_thread(
            self.retriever.retrieve,
            query,
            top_k,
            normalized_filter,
        )
        return ToolResult(
            tool_name=self.spec.name,
            ok=True,
            output={
                "query": query,
                "results": [item.to_dict() for item in results],
            },
            metadata={"result_count": len(results)},
        )
