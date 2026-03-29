from __future__ import annotations

import asyncio
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_manager import AgentSession, RuntimeConfig
from agent_manager.context.budget import resolve_model_budget
from agent_manager.errors import PolicyViolationError
from agent_manager.memory import (
    HashEmbeddingProvider,
    InMemoryVectorRetriever,
    MemoryEntry,
    TextChunker,
)
from agent_manager.providers.base import BaseProvider, ProviderCapabilities
from agent_manager.runtime.planner import Planner
from agent_manager.tools import ToolContext, WebSearchResult
from agent_manager.tools.builtins.filesystem import WriteFileTool
from agent_manager.tools.web_search import BaseWebSearcher
from agent_manager.types import (
    LoopState,
    Message,
    ProviderRequest,
    ProviderResult,
    ProviderStreamEvent,
    StructuredOutputSpec,
    ToolCallRequest,
)


def make_workspace_temp_dir() -> Path:
    root = Path.cwd() / ".tmp_tests"
    root.mkdir(exist_ok=True)
    temp_dir = root / uuid4().hex
    temp_dir.mkdir()
    return temp_dir


class StructuredProvider(BaseProvider):
    provider_name = "structured-provider"
    capabilities = ProviderCapabilities(supports_structured_output=False)

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        del request
        return ProviderResult(
            text='{"answer":"ok"}',
            stop_reason="completed",
            usage={"input_tokens": 3, "output_tokens": 2},
            metadata={"pending_subgoals": ["review changes"]},
        )

    async def stream_generate(self, request: ProviderRequest):
        del request
        yield ProviderStreamEvent(kind="text_delta", text='{"answer":')
        yield ProviderStreamEvent(kind="text_delta", text='"ok"}')
        yield ProviderStreamEvent(
            kind="result",
            result=ProviderResult(
                text='{"answer":"ok"}',
                structured_output={"answer": "ok"},
                stop_reason="completed",
                usage={"input_tokens": 3, "output_tokens": 2},
            ),
        )


class RecordingStructuredProvider(StructuredProvider):
    def __init__(self) -> None:
        super().__init__()
        self.request_specs: list[dict | None] = []

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.request_specs.append(
            request.structured_output.to_dict() if request.structured_output is not None else None
        )
        return await super().generate(request)


class SlowStreamingProvider(BaseProvider):
    provider_name = "slow-streaming-provider"
    capabilities = ProviderCapabilities(supports_streaming=True)

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        del request
        return ProviderResult(text="slow fallback", stop_reason="completed")

    async def stream_generate(self, request: ProviderRequest):
        del request
        yield ProviderStreamEvent(kind="text_delta", text="partial")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class PlainProvider(BaseProvider):
    provider_name = "plain-provider"

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        del request
        return ProviderResult(text="done", stop_reason="completed")


class FixedPlanner(Planner):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def build_plan(
        self,
        goal: str,
        *,
        available_tools: list[str] | None = None,
    ) -> list[str]:
        self.calls.append(
            {
                "goal": goal,
                "available_tools": list(available_tools or []),
            }
        )
        return [
            "Inspect the current task context.",
            "Make the required change.",
            "Verify the result.",
        ]


class FakeWebSearcher(BaseWebSearcher):
    def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title="Agent Manager Docs",
                url="https://example.com/docs",
                snippet=f"Result for {query}",
                source="fake",
            )
        ][:limit]


class RequirementsCompletionTests(unittest.TestCase):
    def test_sqlite_store_is_default_and_persists_checkpoint_state(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "provider": {"name": "echo", "model": "echo-v1"},
                }
            )
            session = AgentSession(config=config)
            result = session.run("persist to sqlite")
            sqlite_path = config.resolved_state_path()
            loaded = session.state_store.load(result.task_id)
            sqlite_exists = sqlite_path.exists()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(sqlite_exists)
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.checkpoint_timestamps)

    def test_structured_output_and_pending_subgoals_are_recorded(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict({"state_dir": str(temp_dir)}),
                provider=StructuredProvider(),
                include_builtin_tools=False,
            )
            result = session.run(
                "Return JSON",
                structured_output={
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.structured_output, {"answer": "ok"})
        self.assertEqual(result.state.pending_subgoals, ["review changes"])
        self.assertTrue(result.state.step_history)

    def test_stream_async_emits_provider_text_events(self) -> None:
        temp_dir = make_workspace_temp_dir()
        events: list[dict] = []

        async def _collect() -> None:
            session = AgentSession(
                config=RuntimeConfig.from_dict({"state_dir": str(temp_dir)}),
                provider=StructuredProvider(),
                include_builtin_tools=False,
            )
            async for event in session.stream_async("stream json"):
                events.append(event)

        try:
            asyncio.run(_collect())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        event_names = [event["name"] for event in events]
        self.assertIn("provider.text_delta", event_names)
        self.assertIn("loop.completed", event_names)

    def test_resume_restores_persisted_structured_output_spec(self) -> None:
        temp_dir = make_workspace_temp_dir()
        provider = RecordingStructuredProvider()
        spec = StructuredOutputSpec(
            type="json_schema",
            name="answer",
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        )
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict({"state_dir": str(temp_dir)}),
                provider=provider,
                include_builtin_tools=False,
            )
            state = LoopState(
                task_id="resume-structured",
                goal="Return JSON",
                messages=[Message(role="user", content="Return JSON")],
                structured_output_spec=spec,
            )
            session.checkpoints.save(state)
            result = session.resume(state.task_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.structured_output, {"answer": "ok"})
        self.assertEqual(provider.request_specs, [spec.to_dict()])

    def test_stream_async_surfaces_terminal_result_payload(self) -> None:
        temp_dir = make_workspace_temp_dir()
        events: list[dict] = []

        async def _collect() -> None:
            session = AgentSession(
                config=RuntimeConfig.from_dict({"state_dir": str(temp_dir)}),
                provider=StructuredProvider(),
                include_builtin_tools=False,
            )
            async for event in session.stream_async("stream json"):
                events.append(event)

        try:
            asyncio.run(_collect())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        provider_result = next(event for event in events if event["name"] == "provider.result")
        completed = next(event for event in events if event["name"] == "loop.completed")
        self.assertEqual(provider_result["details"]["output_text"], '{"answer":"ok"}')
        self.assertEqual(completed["details"]["output_text"], '{"answer":"ok"}')
        self.assertEqual(completed["details"]["structured_output"], {"answer": "ok"})
        self.assertEqual(completed["details"]["usage"], {"input_tokens": 3, "output_tokens": 2})

    def test_closing_stream_early_cancels_background_run(self) -> None:
        temp_dir = make_workspace_temp_dir()
        provider = SlowStreamingProvider()

        async def _close_early() -> None:
            session = AgentSession(
                config=RuntimeConfig.from_dict({"state_dir": str(temp_dir)}),
                provider=provider,
                include_builtin_tools=False,
            )
            stream = session.stream_async("stream slowly")
            while True:
                event = await anext(stream)
                if event["name"] == "provider.text_delta":
                    break
            await asyncio.wait_for(stream.aclose(), timeout=1.0)

        try:
            asyncio.run(_close_early())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(provider.cancelled)

    def test_planner_is_wired_into_session_and_loop(self) -> None:
        temp_dir = make_workspace_temp_dir()
        planner = FixedPlanner()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict({"state_dir": str(temp_dir)}),
                provider=PlainProvider(),
                include_builtin_tools=False,
                planner=planner,
            )
            result = session.run("Implement the documentation update.")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(planner.calls)
        self.assertEqual(
            result.state.metadata["current_plan"],
            [
                "Inspect the current task context.",
                "Make the required change.",
                "Verify the result.",
            ],
        )
        self.assertEqual(
            result.state.pending_subgoals,
            [
                "Inspect the current task context.",
                "Make the required change.",
                "Verify the result.",
            ],
        )
        self.assertEqual(result.state.metadata["planner"], "FixedPlanner")

    def test_model_budget_uses_provider_specific_overrides(self) -> None:
        config = RuntimeConfig.from_dict(
            {
                "provider": {
                    "name": "openai",
                    "model": "gpt-4o-mini",
                    "settings": {
                        "model_context_tokens": 4096,
                        "model_max_output_tokens": 256,
                        "token_counter_chars_per_token": 2.0,
                    },
                }
            }
        )

        profile = resolve_model_budget(config)

        self.assertEqual(profile.max_context_tokens, 4096)
        self.assertEqual(profile.max_output_tokens, 256)
        self.assertEqual(profile.chars_per_token, 2.0)

    def test_memory_entry_has_timestamp_by_default(self) -> None:
        entry = MemoryEntry(key="pref", value="concise")
        self.assertTrue(entry.timestamp)

    def test_vector_retriever_supports_chunking_indexing_and_filtering(self) -> None:
        embedder = HashEmbeddingProvider(dimensions=32)
        retriever = InMemoryVectorRetriever(embedder.embed)
        chunker = TextChunker(chunk_size=24, overlap=4)
        chunks = retriever.index_document(
            "doc-1",
            "agent manager supports retrieval over indexed chunks",
            metadata={"scope": "docs"},
            chunker=chunker,
        )
        results = retriever.retrieve("indexed retrieval", top_k=2, metadata_filter={"scope": "docs"})

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(results)
        self.assertEqual(results[0].metadata["scope"], "docs")

    def test_approval_hook_can_block_dangerous_tool_calls(self) -> None:
        tool = WriteFileTool()
        session = AgentSession(
            config=RuntimeConfig.from_dict({"profile": "local-dev"}),
            include_builtin_tools=False,
            approval_hook=lambda spec, context, arguments: (
                False if spec.name == "write_file" else True
            ),
        )
        session.register_tool(tool)
        with self.assertRaises(PolicyViolationError):
            session.tool_executor.execute(
                ToolCallRequest(
                    id="write-1",
                    name="write_file",
                    arguments={"path": "blocked.txt", "content": "nope"},
                ),
                ToolContext(
                    task_id="task-1",
                    step_index=0,
                    tool_call_id="write-1",
                    working_directory=str(Path.cwd()),
                ),
            )

    def test_web_search_tool_abstraction_can_be_registered(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {"state_dir": str(temp_dir), "profile": "local-dev"}
                ),
                web_searcher=FakeWebSearcher(),
                working_directory=temp_dir,
            )
            result = session.tool_executor.execute(
                ToolCallRequest(
                    id="search-1",
                    name="web_search",
                    arguments={"query": "agent manager", "limit": 1},
                ),
                ToolContext(
                    task_id="task-1",
                    step_index=0,
                    tool_call_id="search-1",
                    working_directory=str(temp_dir),
                ),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["result_count"], 1)
        self.assertEqual(result.output["results"][0]["source"], "fake")


if __name__ == "__main__":
    unittest.main()
