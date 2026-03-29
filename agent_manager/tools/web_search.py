"""Web search abstractions used by the built-in search tool."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agent_manager.config import WebSearchToolConfig
from agent_manager.errors import ConfigurationError
from agent_manager.version import __version__


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


class ConfiguredWebSearcher(BaseWebSearcher):
    """Reusable base for HTTP-backed search providers."""

    source = "web"
    default_endpoint = ""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        timeout_seconds: float = 20.0,
        max_results: int = 5,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.endpoint = endpoint or self.default_endpoint
        self.timeout_seconds = max(float(timeout_seconds), 1.0)
        self.max_results = max(int(max_results), 1)
        self.settings = dict(settings or {})

    def _bounded_limit(self, requested_limit: int) -> int:
        return max(1, min(int(requested_limit), self.max_results))

    def _user_agent(self) -> str:
        return f"agent-manager/{__version__} web-search"

    def _request_json(
        self,
        *,
        url: str,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        final_url = url
        if params:
            final_url = f"{url}?{urlencode(params, doseq=True)}"
        request_headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent(),
        }
        if headers:
            request_headers.update(headers)
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")

        request = Request(
            url=final_url,
            data=data,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content = response.read().decode("utf-8")
        except HTTPError as exc:  # pragma: no cover - network failure path
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.source} search API error ({exc.code}): {error_body}"
            ) from exc
        except URLError as exc:  # pragma: no cover - network failure path
            raise RuntimeError(f"{self.source} search request failed: {exc.reason}") from exc

        decoded = json.loads(content)
        if not isinstance(decoded, Mapping):
            raise RuntimeError(f"{self.source} search API returned a non-object payload.")
        return decoded


class APIKeyWebSearcher(ConfiguredWebSearcher):
    """Reusable helper for providers that need API-key auth."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_key_env = api_key_env

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            value = os.getenv(self.api_key_env)
            if value:
                return value
            raise ConfigurationError(
                f"{self.source} search requires an API key via {self.api_key_env}."
            )
        raise ConfigurationError(f"{self.source} search requires an API key.")


class DuckDuckGoWebSearcher(ConfiguredWebSearcher):
    """Small no-key search adapter backed by DuckDuckGo's instant-answer API."""

    source = "duckduckgo"
    default_endpoint = "https://api.duckduckgo.com/"

    def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        bounded_limit = self._bounded_limit(limit)
        payload = self._request_json(
            url=self.endpoint,
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            },
        )
        return self._parse_payload(payload, limit=bounded_limit)

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
                    source=self.source,
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
        return results[:limit]

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
            source=self.source,
        )


class SerpAPIWebSearcher(APIKeyWebSearcher):
    """SerpAPI-backed web search."""

    source = "serpapi"
    default_endpoint = "https://serpapi.com/search.json"

    def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        bounded_limit = self._bounded_limit(limit)
        params = {
            "engine": str(self.settings.get("engine", "google")),
            "q": query,
            "api_key": self.resolved_api_key(),
            "num": bounded_limit,
            **self._string_params(self.settings),
        }
        payload = self._request_json(url=self.endpoint, params=params)
        return self._parse_payload(payload, limit=bounded_limit)

    def _parse_payload(
        self,
        payload: Mapping[str, Any],
        *,
        limit: int,
    ) -> list[WebSearchResult]:
        results: list[WebSearchResult] = []

        answer_box = payload.get("answer_box")
        if isinstance(answer_box, Mapping):
            answer_text = answer_box.get("snippet") or answer_box.get("answer")
            answer_url = answer_box.get("link") or answer_box.get("displayed_link")
            title = answer_box.get("title") or "Answer Box"
            if isinstance(answer_text, str) and answer_text and isinstance(answer_url, str):
                results.append(
                    WebSearchResult(
                        title=str(title),
                        url=answer_url,
                        snippet=answer_text,
                        source=self.source,
                        metadata={"result_type": "answer_box"},
                    )
                )

        organic_results = payload.get("organic_results", [])
        if isinstance(organic_results, list):
            for item in organic_results:
                if len(results) >= limit:
                    break
                if not isinstance(item, Mapping):
                    continue
                title = item.get("title")
                url = item.get("link")
                snippet = item.get("snippet") or item.get("snippet_highlighted_words")
                if not isinstance(title, str) or not isinstance(url, str):
                    continue
                if isinstance(snippet, list):
                    snippet_text = " ".join(str(part) for part in snippet)
                else:
                    snippet_text = str(snippet or "")
                results.append(
                    WebSearchResult(
                        title=title,
                        url=url,
                        snippet=snippet_text,
                        source=self.source,
                    )
                )
        return results[:limit]

    def _string_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        excluded = {"engine"}
        return {
            str(key): value
            for key, value in params.items()
            if key not in excluded and value is not None
        }


class TavilyWebSearcher(APIKeyWebSearcher):
    """Tavily-backed search with normalized results."""

    source = "tavily"
    default_endpoint = "https://api.tavily.com/search"

    def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        bounded_limit = self._bounded_limit(limit)
        payload = self._request_json(
            url=self.endpoint,
            method="POST",
            headers={"Authorization": f"Bearer {self.resolved_api_key()}"},
            payload={
                "query": query,
                "max_results": bounded_limit,
                **self._request_settings(),
            },
        )
        return self._parse_payload(payload, limit=bounded_limit)

    def _request_settings(self) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in self.settings.items()
            if value is not None
        }

    def _parse_payload(
        self,
        payload: Mapping[str, Any],
        *,
        limit: int,
    ) -> list[WebSearchResult]:
        results: list[WebSearchResult] = []
        items = payload.get("results", [])
        if isinstance(items, list):
            for item in items:
                if len(results) >= limit:
                    break
                if not isinstance(item, Mapping):
                    continue
                title = item.get("title")
                url = item.get("url")
                snippet = item.get("content")
                if not isinstance(title, str) or not isinstance(url, str):
                    continue
                results.append(
                    WebSearchResult(
                        title=title,
                        url=url,
                        snippet=str(snippet or ""),
                        source=self.source,
                        metadata={
                            "score": item.get("score"),
                            "favicon": item.get("favicon"),
                        },
                    )
                )
        if results:
            return results[:limit]

        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            return [
                WebSearchResult(
                    title="Tavily Answer",
                    url="https://docs.tavily.com/",
                    snippet=answer,
                    source=self.source,
                    metadata={"result_type": "answer"},
                )
            ]
        return []


class BraveWebSearcher(APIKeyWebSearcher):
    """Brave Search API adapter."""

    source = "brave"
    default_endpoint = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        bounded_limit = self._bounded_limit(limit)
        payload = self._request_json(
            url=self.endpoint,
            headers={"X-Subscription-Token": self.resolved_api_key()},
            params={
                "q": query,
                "count": bounded_limit,
                **self._query_settings(),
            },
        )
        return self._parse_payload(payload, limit=bounded_limit)

    def _query_settings(self) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in self.settings.items()
            if value is not None
        }

    def _parse_payload(
        self,
        payload: Mapping[str, Any],
        *,
        limit: int,
    ) -> list[WebSearchResult]:
        web = payload.get("web", {})
        if not isinstance(web, Mapping):
            return []
        items = web.get("results", [])
        if not isinstance(items, list):
            return []

        results: list[WebSearchResult] = []
        for item in items:
            if len(results) >= limit:
                break
            if not isinstance(item, Mapping):
                continue
            title = item.get("title")
            url = item.get("url")
            snippet = item.get("description")
            if not isinstance(title, str) or not isinstance(url, str):
                continue
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=str(snippet or ""),
                    source=self.source,
                    metadata={
                        "extra_snippets": item.get("extra_snippets"),
                        "language": item.get("language"),
                    },
                )
            )
        return results[:limit]


def available_web_search_backends() -> list[str]:
    return ["duckduckgo", "serpapi", "tavily", "brave"]


def build_web_searcher(
    config: WebSearchToolConfig | Mapping[str, Any] | None = None,
) -> BaseWebSearcher:
    if config is None:
        normalized = WebSearchToolConfig()
    elif isinstance(config, WebSearchToolConfig):
        normalized = config
    else:
        normalized = WebSearchToolConfig.from_dict(config)

    backend = normalized.backend.strip().lower()
    common_kwargs = {
        "endpoint": normalized.endpoint,
        "timeout_seconds": normalized.timeout_seconds,
        "max_results": normalized.max_results,
        "settings": normalized.settings,
    }
    if backend == "duckduckgo":
        return DuckDuckGoWebSearcher(**common_kwargs)
    if backend == "serpapi":
        return SerpAPIWebSearcher(
            api_key=normalized.api_key,
            api_key_env=normalized.api_key_env,
            **common_kwargs,
        )
    if backend == "tavily":
        return TavilyWebSearcher(
            api_key=normalized.api_key,
            api_key_env=normalized.api_key_env,
            **common_kwargs,
        )
    if backend == "brave":
        return BraveWebSearcher(
            api_key=normalized.api_key,
            api_key_env=normalized.api_key_env,
            **common_kwargs,
        )
    raise ConfigurationError(
        f"Unsupported web search backend '{normalized.backend}'. "
        f"Available backends: {', '.join(available_web_search_backends())}."
    )


__all__ = [
    "APIKeyWebSearcher",
    "BaseWebSearcher",
    "BraveWebSearcher",
    "ConfiguredWebSearcher",
    "DuckDuckGoWebSearcher",
    "SerpAPIWebSearcher",
    "TavilyWebSearcher",
    "WebSearchResult",
    "available_web_search_backends",
    "build_web_searcher",
]
