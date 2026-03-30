from __future__ import annotations

import asyncio
import json
import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from agent_manager import AgentSession, RuntimeConfig
from agent_manager.memory import InMemoryKeywordRetriever
from agent_manager.providers.base import BaseProvider
from agent_manager.tools import ToolContext, ToolExecutor, ToolRegistry, ToolSpec
from agent_manager.tools.base import BaseTool, ToolResult
from agent_manager.tools.builtins.http import HttpRequestTool
from agent_manager.types import ProviderRequest, ProviderResult, ToolCallRequest


def make_workspace_temp_dir() -> Path:
    root = Path.cwd() / ".tmp_tests"
    root.mkdir(exist_ok=True)
    temp_dir = root / uuid4().hex
    temp_dir.mkdir()
    return temp_dir


class FlakyTool(BaseTool):
    spec = ToolSpec(
        name="flaky_tool",
        description="Fail once and then succeed.",
        retry_count=1,
    )

    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        del arguments, context
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient tool failure")
        return ToolResult(tool_name=self.spec.name, ok=True, output={"done": True})


class ExplodingTool(BaseTool):
    spec = ToolSpec(
        name="explode_tool",
        description="Always fail.",
        retry_count=1,
    )

    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        del arguments, context
        raise RuntimeError("tool exploded")


class ToolFailureProvider(BaseProvider):
    provider_name = "tool-failure-provider"

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[ProviderRequest] = []

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ProviderResult(
                tool_calls=[
                    ToolCallRequest(
                        id="explode-1",
                        name="explode_tool",
                        arguments={},
                    )
                ],
                stop_reason="tool_call",
            )
        return ProviderResult(text="handled tool failure", stop_reason="completed")


class ToolSystemTests(unittest.TestCase):
    def test_registry_can_wrap_sync_callable_tools(self) -> None:
        registry = ToolRegistry()
        registry.register_callable(
            ToolSpec(
                name="echo_arguments",
                description="Return the provided arguments.",
            ),
            lambda arguments, context: {
                "arguments": arguments,
                "tool_call_id": context.tool_call_id,
            },
        )
        executor = ToolExecutor(registry)

        result = asyncio.run(
            executor.execute_async(
                ToolCallRequest(
                    id="call-1",
                    name="echo_arguments",
                    arguments={"value": "hello"},
                ),
                ToolContext(task_id="task-1", step_index=0, tool_call_id="call-1"),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.output["arguments"]["value"], "hello")
        self.assertEqual(result.output["tool_call_id"], "call-1")

    def test_executor_retries_tools_before_succeeding(self) -> None:
        tool = FlakyTool()
        executor = ToolExecutor(ToolRegistry([tool]))

        result = asyncio.run(
            executor.execute_async(
                ToolCallRequest(id="call-2", name="flaky_tool", arguments={}),
                ToolContext(task_id="task-1", step_index=0, tool_call_id="call-2"),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(tool.calls, 2)
        self.assertEqual(result.metadata["attempts"], 2)

    def test_executor_returns_failed_tool_result_after_retries(self) -> None:
        executor = ToolExecutor(ToolRegistry([ExplodingTool()]))

        result = asyncio.run(
            executor.execute_async(
                ToolCallRequest(id="call-3", name="explode_tool", arguments={}),
                ToolContext(task_id="task-1", step_index=0, tool_call_id="call-3"),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "tool exploded")
        self.assertEqual(result.metadata["attempts"], 2)

    def test_builtin_filesystem_tools_stay_within_working_directory(self) -> None:
        temp_dir = make_workspace_temp_dir()
        outside_dir = make_workspace_temp_dir()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {
                        "state_dir": str(temp_dir / "state"),
                        "profile": "local-dev",
                    }
                ),
                include_builtin_tools=True,
                working_directory=temp_dir,
            )
            write_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="write-1",
                    name="write_file",
                    arguments={
                        "path": "notes/example.txt",
                        "content": "phase 3",
                        "create_parents": True,
                    },
                ),
                ToolContext(
                    task_id="task-1",
                    step_index=0,
                    tool_call_id="write-1",
                    working_directory=str(temp_dir),
                    metadata={"filesystem_roots": [str(temp_dir)]},
                ),
            )
            read_outside_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="read-1",
                    name="read_file",
                    arguments={"path": str(outside_dir / "secret.txt")},
                ),
                ToolContext(
                    task_id="task-1",
                    step_index=1,
                    tool_call_id="read-1",
                    working_directory=str(temp_dir),
                    metadata={"filesystem_roots": [str(temp_dir)]},
                ),
            )
            self.assertTrue(write_result.ok)
            self.assertTrue((temp_dir / "notes" / "example.txt").exists())
            self.assertFalse(read_outside_result.ok)
            self.assertIn("outside the allowed filesystem roots", read_outside_result.error)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_retrieval_tool_is_registered_when_retriever_is_provided(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            retriever = InMemoryKeywordRetriever()
            retriever.add(
                item_id="doc-1",
                content="agent manager phase three retrieval support",
                metadata={"scope": "docs"},
            )
            session = AgentSession(
                config=RuntimeConfig.from_dict({"state_dir": str(temp_dir / "state")}),
                retriever=retriever,
                working_directory=temp_dir,
            )
            result = session.tool_executor.execute(
                ToolCallRequest(
                    id="retrieve-1",
                    name="retrieve_documents",
                    arguments={
                        "query": "phase retrieval",
                        "metadata_filter": {"scope": "docs"},
                    },
                ),
                ToolContext(
                    task_id="task-1",
                    step_index=0,
                    tool_call_id="retrieve-1",
                    working_directory=str(temp_dir),
                    metadata={"filesystem_roots": [str(temp_dir)]},
                ),
            )
            self.assertTrue(session.tools.has("retrieve_documents"))
            self.assertTrue(result.ok)
            self.assertEqual(result.metadata["result_count"], 1)
            self.assertEqual(result.output["results"][0]["id"], "doc-1")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_http_request_uses_google_fetch_tool_for_browser_gets(self) -> None:
        tool = HttpRequestTool()

        def fake_get_page(url: str, *, headless: bool, cookie_file: str | None):
            self.assertEqual(url, "https://example.com")
            self.assertTrue(headless)
            self.assertEqual(cookie_file, "cookies.json")
            return {
                "url": url,
                "text": "rendered page text",
                "html": "<html><body>rendered</body></html>",
            }

        def fake_get_pdf(**kwargs):  # pragma: no cover - should not be used in this test
            raise AssertionError("get_pdf should not be called for text fetches")

        fake_module = types.SimpleNamespace(get_page=fake_get_page, get_pdf=fake_get_pdf)
        with patch.dict(sys.modules, {"google_search_tool": fake_module}):
            result = asyncio.run(
                tool.invoke(
                    {
                        "url": "https://example.com",
                        "method": "GET",
                        "cookie_file": "cookies.json",
                    },
                    ToolContext(task_id="task-http", step_index=0, tool_call_id="http-1"),
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.output["body"], "rendered page text")
        self.assertEqual(result.output["engine"], "browser")
        self.assertEqual(result.output["content_format"], "text")

    def test_http_request_can_return_pdf_from_google_fetch_tool(self) -> None:
        tool = HttpRequestTool()

        def fake_get_page(**kwargs):  # pragma: no cover - should not be used in this test
            raise AssertionError("get_page should not be called for pdf fetches")

        def fake_get_pdf(
            url: str,
            *,
            output_path: str | None,
            headless: bool,
            cookie_file: str | None,
            page_format: str,
            print_background: bool,
        ):
            self.assertEqual(url, "https://example.com/report")
            self.assertEqual(output_path, "report.pdf")
            self.assertEqual(page_format, "Letter")
            self.assertFalse(print_background)
            self.assertTrue(headless)
            self.assertEqual(cookie_file, "cookies.json")
            return {
                "url": url,
                "pdf_bytes": b"%PDF-demo%",
                "path": "report.pdf",
            }

        fake_module = types.SimpleNamespace(get_page=fake_get_page, get_pdf=fake_get_pdf)
        with patch.dict(sys.modules, {"google_search_tool": fake_module}):
            result = asyncio.run(
                tool.invoke(
                    {
                        "url": "https://example.com/report",
                        "response_format": "pdf",
                        "output_path": "report.pdf",
                        "cookie_file": "cookies.json",
                        "page_format": "Letter",
                        "print_background": False,
                    },
                    ToolContext(task_id="task-http", step_index=0, tool_call_id="http-2"),
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.output["engine"], "browser")
        self.assertEqual(result.output["content_format"], "pdf")
        self.assertEqual(result.output["path"], "report.pdf")
        self.assertEqual(result.output["headers"]["Content-Type"], "application/pdf")
        self.assertTrue(result.output["body"])

    def test_http_request_keeps_raw_backend_for_post_requests(self) -> None:
        tool = HttpRequestTool()

        def fake_request(
            method: str,
            url: str,
            headers: dict[str, str],
            data: bytes | None,
            timeout: float,
            max_chars: int,
        ) -> dict[str, object]:
            self.assertEqual(method, "POST")
            self.assertEqual(url, "https://api.example.com/items")
            self.assertEqual(headers["Content-Type"], "application/json")
            self.assertEqual(json.loads(data.decode("utf-8")), {"name": "demo"})
            self.assertEqual(timeout, 30.0)
            self.assertEqual(max_chars, 20000)
            return {
                "url": url,
                "status": 200,
                "headers": {"Content-Type": "application/json"},
                "body": "{\"ok\":true}",
                "truncated": False,
                "engine": "raw",
                "content_format": "text",
                "path": None,
            }

        tool._request = fake_request  # type: ignore[method-assign]
        result = asyncio.run(
            tool.invoke(
                {
                    "url": "https://api.example.com/items",
                    "method": "POST",
                    "json": {"name": "demo"},
                },
                ToolContext(task_id="task-http", step_index=0, tool_call_id="http-3"),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.output["engine"], "raw")

    def test_loop_records_failed_tool_observation_and_continues(self) -> None:
        temp_dir = make_workspace_temp_dir()
        provider = ToolFailureProvider()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {
                        "state_dir": str(temp_dir),
                        "runtime": {"max_steps": 3, "max_consecutive_failures": 3},
                    }
                ),
                provider=provider,
                tools=ToolRegistry([ExplodingTool()]),
                include_builtin_tools=False,
                working_directory=temp_dir,
            )
            result = session.run("trigger a tool failure")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.stop_reason, "completed")
        self.assertFalse(result.tool_results[0]["ok"])
        self.assertEqual(result.state.tool_observations[0]["error"], "tool exploded")
        self.assertTrue(provider.requests[1].messages[-1].metadata["is_error"])


if __name__ == "__main__":
    unittest.main()
