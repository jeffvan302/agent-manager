"""Optional adapters for LlamaIndex-style retrievers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_manager.memory.retrieval import BaseRetriever, RetrievalResult
from agent_manager.plugins.base import Plugin
from agent_manager.tools.builtins.retrieval import RetrieveDocumentsTool


class LlamaIndexRetrieverAdapter(BaseRetriever):
    """Wrap a LlamaIndex-style retriever with the agent_manager retriever interface."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        raw_items = self.retriever.retrieve(query)
        results: list[RetrievalResult] = []
        for index, item in enumerate(raw_items, start=1):
            content = self._content(item)
            metadata = self._metadata(item)
            if metadata_filter and not self._matches_metadata(metadata, metadata_filter):
                continue
            results.append(
                RetrievalResult(
                    id=self._item_id(item, index),
                    content=content,
                    score=self._score(item),
                    metadata=metadata,
                )
            )
            if len(results) >= max(top_k, 0):
                break
        return results

    def _content(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(getattr(item, "text", None), str):
            return item.text
        node = getattr(item, "node", None)
        if node is not None:
            if callable(getattr(node, "get_content", None)):
                return str(node.get_content())
            if isinstance(getattr(node, "text", None), str):
                return node.text
        if callable(getattr(item, "get_content", None)):
            return str(item.get_content())
        return str(item)

    def _metadata(self, item: Any) -> dict[str, Any]:
        if isinstance(getattr(item, "metadata", None), Mapping):
            return dict(item.metadata)
        node = getattr(item, "node", None)
        if isinstance(getattr(node, "metadata", None), Mapping):
            return dict(node.metadata)
        return {}

    def _score(self, item: Any) -> float:
        score = getattr(item, "score", 0.0)
        try:
            return float(score or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _item_id(self, item: Any, index: int) -> str:
        for attribute in ("node_id", "id_", "id"):
            value = getattr(item, attribute, None)
            if value:
                return str(value)
        node = getattr(item, "node", None)
        for attribute in ("node_id", "id_", "id"):
            value = getattr(node, attribute, None)
            if value:
                return str(value)
        return f"llamaindex-{index}"

    def _matches_metadata(
        self,
        metadata: Mapping[str, Any],
        metadata_filter: Mapping[str, Any],
    ) -> bool:
        for key, expected in metadata_filter.items():
            if metadata.get(key) != expected:
                return False
        return True


class LlamaIndexRetrievalPlugin(Plugin):
    """Register a LlamaIndex-style retriever and retrieval tool on a session."""

    name = "llamaindex-retrieval"
    description = "Expose a LlamaIndex-style retriever through the unified retrieval tool."

    def __init__(
        self,
        retriever: Any,
        *,
        retriever_name: str = "llamaindex",
        set_default: bool = True,
    ) -> None:
        self._adapter = LlamaIndexRetrieverAdapter(retriever)
        self._retriever_name = retriever_name
        self._set_default = set_default

    def register(self, target: Any) -> None:
        target.register_retriever(
            self._retriever_name,
            self._adapter,
            make_default=self._set_default,
        )
        target.register_tool(RetrieveDocumentsTool(self._adapter), replace=True)
