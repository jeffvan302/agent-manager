"""Core chunking, embedding, and in-memory vector indexing helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from agent_manager.memory.retrieval import BaseRetriever, RetrievalResult

EmbeddingFunction = Callable[[str], list[float]]


@dataclass(slots=True)
class DocumentChunk:
    id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_retrieval_result(self, score: float = 0.0) -> RetrievalResult:
        return RetrievalResult(
            id=self.id,
            content=self.content,
            score=score,
            metadata=dict(self.metadata),
        )


class TextChunker:
    """Split large documents into overlapping chunks for retrieval."""

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        self.chunk_size = max(chunk_size, 1)
        self.overlap = max(min(overlap, self.chunk_size - 1), 0)

    def chunk_document(
        self,
        document_id: str,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        normalized = text.strip()
        if not normalized:
            return []
        chunks: list[DocumentChunk] = []
        start = 0
        index = 1
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            content = normalized[start:end].strip()
            if content:
                chunk_metadata = dict(metadata or {})
                chunk_metadata.update(
                    {
                        "document_id": document_id,
                        "chunk_index": index,
                        "char_start": start,
                        "char_end": end,
                    }
                )
                chunks.append(
                    DocumentChunk(
                        id=f"{document_id}:chunk:{index}",
                        document_id=document_id,
                        content=content,
                        metadata=chunk_metadata,
                    )
                )
            if end >= len(normalized):
                break
            start = max(end - self.overlap, 0)
            index += 1
        return chunks


class HashEmbeddingProvider:
    """Small dependency-free embedding provider for local indexing and tests."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = max(dimensions, 8)

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return vector
        for token in tokens:
            slot = hash(token) % self.dimensions
            vector[slot] += 1.0
        length = sqrt(sum(value * value for value in vector)) or 1.0
        return [value / length for value in vector]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class InMemoryVectorRetriever(BaseRetriever):
    """Simple vector retriever backed by in-memory embeddings."""

    def __init__(self, embed_fn: EmbeddingFunction) -> None:
        self._embed_fn = embed_fn
        self._entries: list[tuple[DocumentChunk, list[float]]] = []

    def index_chunks(self, chunks: Iterable[DocumentChunk]) -> list[DocumentChunk]:
        indexed: list[DocumentChunk] = []
        for chunk in chunks:
            vector = self._embed_fn(chunk.content)
            self._entries.append((chunk, list(vector)))
            indexed.append(chunk)
        return indexed

    def index_document(
        self,
        document_id: str,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        chunker: TextChunker | None = None,
    ) -> list[DocumentChunk]:
        chunker = chunker or TextChunker()
        chunks = chunker.chunk_document(document_id, text, metadata=metadata)
        return self.index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_vector = self._embed_fn(query)
        scored: list[RetrievalResult] = []
        for chunk, vector in self._entries:
            if metadata_filter and not self._matches(chunk.metadata, metadata_filter):
                continue
            score = self._cosine_similarity(query_vector, vector)
            scored.append(chunk.to_retrieval_result(score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(top_k, 0)]

    @staticmethod
    def _matches(
        metadata: Mapping[str, Any],
        metadata_filter: Mapping[str, Any],
    ) -> bool:
        for key, expected in metadata_filter.items():
            if metadata.get(key) != expected:
                return False
        return True

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = sqrt(sum(value * value for value in left)) or 1.0
        right_norm = sqrt(sum(value * value for value in right)) or 1.0
        return dot / (left_norm * right_norm)
