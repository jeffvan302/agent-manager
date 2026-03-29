"""Web search abstractions used by the built-in search tool."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseWebSearcher(ABC):
    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        """Return normalized search results."""


class DuckDuckGoWebSearcher(BaseWebSearcher):
    """Small no-key search adapter backed by DuckDuckGo's instant-answer API."""

    endpoint = "https://api.duckduckgo.com/"

    def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        params = urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
        )
        request = Request(
            url=f"{self.endpoint}?{params}",
            headers={"Accept": "application/json", "User-Agent": "agent-manager/web-search"},
            method="GET",
        )
        with urlopen(request, timeout=20.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return self._parse_payload(payload, limit=limit)

    def _parse_payload(
        self,
        payload: Mapping[str, Any],
        *,
        limit: int,
    ) -> list[WebSearchResult]:
        results: list[WebSearchResult] = []
        abstract_text = payload.get("AbstractText")
        abstract_url = payload.get("AbstractURL")
        heading = payload.get("Heading")
        if isinstance(abstract_text, str) and abstract_text and isinstance(abstract_url, str):
            results.append(
                WebSearchResult(
                    title=str(heading or "Instant Answer"),
                    url=abstract_url,
                    snippet=abstract_text,
                    source="duckduckgo",
                )
            )

        related_topics = payload.get("RelatedTopics", [])
        if isinstance(related_topics, list):
            for topic in related_topics:
                if len(results) >= limit:
                    break
                result = self._topic_to_result(topic)
                if result is not None:
                    results.append(result)
        return results[: max(limit, 0)]

    def _topic_to_result(self, topic: Any) -> WebSearchResult | None:
        if not isinstance(topic, Mapping):
            return None
        if "Topics" in topic and isinstance(topic.get("Topics"), list):
            for nested in topic["Topics"]:
                result = self._topic_to_result(nested)
                if result is not None:
                    return result
            return None
        text = topic.get("Text")
        url = topic.get("FirstURL")
        if not isinstance(text, str) or not isinstance(url, str):
            return None
        title = text.split(" - ", 1)[0]
        return WebSearchResult(
            title=title,
            url=url,
            snippet=text,
            source="duckduckgo",
        )
