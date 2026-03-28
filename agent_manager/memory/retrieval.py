"""Retrieval contracts and a simple in-memory retriever."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return retrieval results for the query."""


class InMemoryKeywordRetriever(BaseRetriever):
    """Small keyword-overlap retriever for local documents and tests."""

    def __init__(self, items: list[RetrievalResult] | None = None) -> None:
        self._items: list[RetrievalResult] = list(items or [])

    def add(
        self,
        *,
        item_id: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._items.append(
            RetrievalResult(
                id=item_id,
                content=content,
                metadata=dict(metadata or {}),
            )
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_terms = {term for term in query.lower().split() if term}
        matches: list[RetrievalResult] = []

        for item in self._items:
            if metadata_filter and not self._matches_metadata(item, metadata_filter):
                continue
            content_terms = {term for term in item.content.lower().split() if term}
            overlap = len(query_terms & content_terms)
            if overlap <= 0 and query_terms:
                continue
            score = float(overlap) if query_terms else 1.0
            matches.append(
                RetrievalResult(
                    id=item.id,
                    content=item.content,
                    score=score,
                    metadata=dict(item.metadata),
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: max(top_k, 0)]

    def _matches_metadata(
        self,
        item: RetrievalResult,
        metadata_filter: Mapping[str, Any],
    ) -> bool:
        for key, expected in metadata_filter.items():
            if item.metadata.get(key) != expected:
                return False
        return True
