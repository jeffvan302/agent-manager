"""Optional adapter for ChromaDB vector store retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_manager.errors import ConfigurationError
from agent_manager.memory.retrieval import BaseRetriever, RetrievalResult
from agent_manager.plugins.base import Plugin
from agent_manager.tools.builtins.retrieval import RetrieveDocumentsTool


class ChromaRetrieverAdapter(BaseRetriever):
    """Wrap a ChromaDB collection as an agent_manager retriever.

    Usage::

        import chromadb
        client = chromadb.Client()
        collection = client.get_or_create_collection("docs")
        retriever = ChromaRetrieverAdapter(collection)
    """

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": max(top_k, 1),
        }
        if metadata_filter:
            kwargs["where"] = dict(metadata_filter)

        raw = self._collection.query(**kwargs)
        return self._parse_results(raw)

    def _parse_results(self, raw: Any) -> list[RetrievalResult]:
        """Parse the ChromaDB QueryResult dict into RetrievalResult items."""
        results: list[RetrievalResult] = []

        ids_list = raw.get("ids") or [[]]
        documents_list = raw.get("documents") or [[]]
        metadatas_list = raw.get("metadatas") or [[]]
        distances_list = raw.get("distances") or [[]]

        # Chroma wraps results in an outer list (one per query text).
        ids = ids_list[0] if ids_list else []
        documents = documents_list[0] if documents_list else []
        metadatas = metadatas_list[0] if metadatas_list else []
        distances = distances_list[0] if distances_list else []

        for index in range(len(ids)):
            doc_id = str(ids[index]) if index < len(ids) else f"chroma-{index}"
            content = str(documents[index]) if index < len(documents) else ""
            metadata = dict(metadatas[index]) if index < len(metadatas) and isinstance(metadatas[index], Mapping) else {}
            distance = float(distances[index]) if index < len(distances) else 0.0
            # ChromaDB returns distances; convert to a similarity-like score.
            score = 1.0 / (1.0 + distance)
            results.append(
                RetrievalResult(
                    id=doc_id,
                    content=content,
                    score=score,
                    metadata=metadata,
                )
            )
        return results


class ChromaRetrievalPlugin(Plugin):
    """Register a ChromaDB collection as a retriever and retrieval tool."""

    name = "chroma-retrieval"
    description = "Expose a ChromaDB collection through the unified retrieval tool."

    def __init__(
        self,
        collection: Any,
        *,
        retriever_name: str = "chroma",
        set_default: bool = True,
    ) -> None:
        self._adapter = ChromaRetrieverAdapter(collection)
        self._retriever_name = retriever_name
        self._set_default = set_default

    def is_available(self) -> bool:
        # The collection object is already provided; no import needed to use it.
        return True

    def register(self, target: Any) -> None:
        target.register_retriever(
            self._retriever_name,
            self._adapter,
            make_default=self._set_default,
        )
        target.register_tool(RetrieveDocumentsTool(self._adapter), replace=True)
