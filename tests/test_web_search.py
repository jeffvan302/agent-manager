from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_manager import AgentSession, RuntimeConfig
from agent_manager.tools.builtins.web import WebSearchTool
from agent_manager.tools.web_search import (
    BraveWebSearcher,
    SerpAPIWebSearcher,
    TavilyWebSearcher,
    build_web_searcher,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class WebSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="agent_manager_web_search_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_web_searcher_returns_configured_backend(self) -> None:
        searcher = build_web_searcher(
            {
                "backend": "serpapi",
                "api_key": "test-key",
                "max_results": 7,
            }
        )

        self.assertIsInstance(searcher, SerpAPIWebSearcher)
        self.assertEqual(searcher.max_results, 7)

    def test_session_can_disable_web_search_tool(self) -> None:
        session = AgentSession(
            config=RuntimeConfig.from_dict(
                {
                    "profile": "local-dev",
                    "state_backend": "json",
                    "state_dir": str(self.temp_dir / "state"),
                    "tools": {"web_search": {"enabled": False}},
                }
            ),
            working_directory=self.temp_dir,
        )

        self.assertFalse(session.tools.has("web_search"))

    def test_session_uses_configured_web_search_backend(self) -> None:
        session = AgentSession(
            config=RuntimeConfig.from_dict(
                {
                    "profile": "local-dev",
                    "state_backend": "json",
                    "state_dir": str(self.temp_dir / "state"),
                    "tools": {
                        "web_search": {
                            "backend": "brave",
                            "api_key": "brave-key",
                        }
                    },
                }
            ),
            working_directory=self.temp_dir,
        )

        tool = session.tools.get("web_search")

        self.assertIsInstance(tool, WebSearchTool)
        self.assertIsInstance(tool.searcher, BraveWebSearcher)

    def test_serpapi_search_builds_expected_query_and_parses_results(self) -> None:
        captured_request = {}

        def fake_urlopen(request, timeout=0.0):
            captured_request["url"] = request.full_url
            captured_request["timeout"] = timeout
            return FakeHTTPResponse(
                {
                    "organic_results": [
                        {
                            "title": "Example Result",
                            "link": "https://example.com/result",
                            "snippet": "Useful result snippet.",
                        }
                    ]
                }
            )

        searcher = SerpAPIWebSearcher(api_key="serp-key", settings={"engine": "google"})
        with patch("agent_manager.tools.web_search.urlopen", new=fake_urlopen):
            results = searcher.search("budget gpu", limit=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Example Result")
        self.assertIn("engine=google", captured_request["url"])
        self.assertIn("q=budget+gpu", captured_request["url"])
        self.assertIn("api_key=serp-key", captured_request["url"])
        self.assertIn("num=3", captured_request["url"])

    def test_tavily_search_posts_authorized_json_and_parses_results(self) -> None:
        captured_request = {}

        def fake_urlopen(request, timeout=0.0):
            captured_request["url"] = request.full_url
            captured_request["timeout"] = timeout
            captured_request["method"] = request.get_method()
            captured_request["body"] = json.loads(request.data.decode("utf-8"))
            captured_request["headers"] = {
                key.lower(): value for key, value in request.header_items()
            }
            return FakeHTTPResponse(
                {
                    "results": [
                        {
                            "title": "Tavily Result",
                            "url": "https://example.com/tavily",
                            "content": "Tavily snippet",
                            "score": 0.98,
                        }
                    ]
                }
            )

        searcher = TavilyWebSearcher(
            api_key="tvly-test",
            settings={"search_depth": "basic", "topic": "general"},
        )
        with patch("agent_manager.tools.web_search.urlopen", new=fake_urlopen):
            results = searcher.search("agent manager", limit=4)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Tavily Result")
        self.assertEqual(captured_request["method"], "POST")
        self.assertEqual(
            captured_request["headers"]["authorization"],
            "Bearer tvly-test",
        )
        self.assertEqual(captured_request["body"]["query"], "agent manager")
        self.assertEqual(captured_request["body"]["max_results"], 4)
        self.assertEqual(captured_request["body"]["search_depth"], "basic")

    def test_brave_search_uses_subscription_header_and_parses_results(self) -> None:
        captured_request = {}

        def fake_urlopen(request, timeout=0.0):
            captured_request["url"] = request.full_url
            captured_request["timeout"] = timeout
            captured_request["headers"] = {
                key.lower(): value for key, value in request.header_items()
            }
            return FakeHTTPResponse(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Brave Result",
                                "url": "https://example.com/brave",
                                "description": "Brave snippet",
                                "extra_snippets": ["More context"],
                            }
                        ]
                    }
                }
            )

        searcher = BraveWebSearcher(api_key="brave-test", settings={"extra_snippets": True})
        with patch("agent_manager.tools.web_search.urlopen", new=fake_urlopen):
            results = searcher.search("python agent", limit=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Brave Result")
        self.assertEqual(
            captured_request["headers"]["x-subscription-token"],
            "brave-test",
        )
        self.assertIn("q=python+agent", captured_request["url"])
        self.assertIn("count=2", captured_request["url"])


if __name__ == "__main__":
    unittest.main()
