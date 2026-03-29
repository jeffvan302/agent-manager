"""Optional adapter for FAISS-based vector retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_manager.memory.retrieval import BaseRetriever, RetrievalResult
from agent_manager.plugins.base import Plugin
from agent_manager.tools.builtins.retrieval import RetrieveDocumentsTool


@dataclass(slots=True)
class FAISSDocument:
    """A document stored alongside a FAISS index entry."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FAISSRetrieverAdapter(BaseRetriever):
    """Wrap a FAISS index and document store as an agent_manager retriever.

    This adapter expects a pre-built FAISS index and a parallel list of
    ``FAISSDocument`` entries that correspond to the vectors in the index.
    An embedding function is required to convert query strings to vectors.

    Usage::

        import faiss
        import numpy as np

        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        docs = []

        # ... add vectors and documents ...

        retriever = FAISSRetrieverAdapter(
            index=index,
            documents=docs,
            embed_fn=my_embedding_function,  # str -> np.ndarray
        )
    """

    def __init__(
        self,
        *,
        index: Any,
        documents: list[FAISSDocument],
        embed_fn: Any,
    ) -> None:
        self._index = index
        self._documents = list(documents)
        self._embed_fn = embed_fn

    def add(self, vector: Any, document: FAISSDocument) -> None:
        """Add a single vector + document pair."""
        import numpy as np

        vec = np.array(vector, dtype="float32")
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        self._index.add(vec)
        self._documents.append(document)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        import numpy as np

        query_vector = self._embed_fn(query)
        query_vector = np.array(query_vector, dtype="float32")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Retrieve more candidates when filtering so we can discard non-matches.
        fetch_k = top_k * 3 if metadata_filter else top_k
        distances, indices = self._index.search(query_vector, min(fetch_k, len(self._documents)))

        results: list[RetrievalResult] = []
        for distance, idx in zip(distances[0], indices[0]):
            if int(idx) < 0:
                continue
            doc = self._documents[int(idx)]
            if metadata_filter and not self._matches(doc.metadata, metadata_filter):
                continue
            score = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievalResult(
                    id=doc.id,
                    content=doc.content,
                    score=score,
                    metadata=dict(doc.metadata),
                )
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
        for key, expected in filters.items():
            if metadata.get(key) != expected:
                return False
        return True


class FAISSRetrievalPlugin(Plugin):
    """Register a FAISS-backed retriever and retrieval tool on a session."""

    name = "faiss-retrieval"
    description = "Expose a FAISS index through the unified retrieval tool."

    def __init__(
        self,
        *,
        index: Any,
        documents: list[FAISSDocument],
        embed_fn: Any,
        retriever_name: str = "faiss",
        set_default: bool = True,
    ) -> None:
        self._adapter = FAISSRetrieverAdapter(
            index=index, documents=documents, embed_fn=embed_fn,
        )
        self._retriever_name = retriever_name
        self._set_default = set_default

    def is_available(self) -> bool:
        # The index and embed_fn are already provided; no import needed to use them.
        return True

    def register(self, target: Any) -> None:
        target.register_retriever(
            self._retriever_name,
            self._adapter,
            make_default=self._set_default,
        )
        target.register_tool(RetrieveDocumentsTool(self._adapter), replace=True)
