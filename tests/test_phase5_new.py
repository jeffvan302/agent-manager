"""Tests for Phase 5 additions: MCP, Chroma, FAISS, pgvector, embeddings, export."""

from __future__ import annotations

import shutil
import unittest
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4
from typing import Any

from agent_manager import AgentSession, RuntimeConfig
from agent_manager.plugins import (
    Plugin,
    PluginRegistry,
)
from agent_manager.plugins.mcp import MCPToolAdapter, MCPToolsPlugin
from agent_manager.plugins.chroma import ChromaRetrieverAdapter, ChromaRetrievalPlugin
from agent_manager.plugins.faiss_store import (
    FAISSDocument,
    FAISSRetrieverAdapter,
    FAISSRetrievalPlugin,
)
from agent_manager.plugins.pgvector import PgVectorRetrieverAdapter
from agent_manager.plugins.embeddings import (
    BaseEmbeddingProvider,
    CallableEmbeddingProvider,
)
from agent_manager.plugins.export import (
    export_all,
    to_langchain_tool_definition,
    to_mcp_tool_definition,
    to_openai_function,
    to_openapi_schema,
)
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from agent_manager.types import ToolCallRequest


def make_workspace_temp_dir() -> Path:
    root = Path.cwd() / ".tmp_tests"
    root.mkdir(exist_ok=True)
    temp_dir = root / uuid4().hex
    temp_dir.mkdir()
    return temp_dir


# ---------------------------------------------------------------------------
# Fakes / stubs
# ---------------------------------------------------------------------------

class FakeMCPClient:
    """Mimics an MCP client with a call_tool method."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._responses = responses or {}

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if name in self._responses:
            return self._responses[name]
        return {
            "content": [{"type": "text", "text": f"result for {name}"}],
            "isError": False,
        }


class FakeChromaCollection:
    """Mimics a ChromaDB collection.query() response."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    def query(self, query_texts: list[str], n_results: int, **kwargs) -> dict:
        ids = [[d["id"] for d in self._docs[:n_results]]]
        documents = [[d["content"] for d in self._docs[:n_results]]]
        metadatas = [[d.get("metadata", {}) for d in self._docs[:n_results]]]
        distances = [[d.get("distance", 0.5) for d in self._docs[:n_results]]]
        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }


class FakeFAISSIndex:
    """Minimal FAISS index stub using numpy."""

    def __init__(self, dimension: int) -> None:
        import numpy as np
        self._dimension = dimension
        self._vectors: list[Any] = []

    def add(self, vectors: Any) -> None:
        import numpy as np
        for row in vectors:
            self._vectors.append(np.array(row, dtype="float32"))

    def search(self, query: Any, k: int) -> tuple:
        import numpy as np
        if not self._vectors:
            return np.array([[]]), np.array([[]])
        distances = []
        for vec in self._vectors:
            dist = float(np.sum((query[0] - vec) ** 2))
            distances.append(dist)
        indices = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
        return (
            np.array([[distances[i] for i in indices]], dtype="float32"),
            np.array([indices], dtype="int64"),
        )


class FakePgCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def execute(self, sql: str, params: list) -> None:
        pass

    def fetchall(self) -> list[tuple]:
        return self._rows

    def close(self) -> None:
        pass


class FakePgConnection:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def cursor(self) -> FakePgCursor:
        return FakePgCursor(self._rows)


# ---------------------------------------------------------------------------
# MCP adapter tests
# ---------------------------------------------------------------------------

class MCPAdapterTests(unittest.TestCase):
    def test_mcp_tool_adapter_registers_and_executes(self) -> None:
        client = FakeMCPClient()
        definition = {
            "name": "mcp_search",
            "description": "Search the knowledge base.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        adapter = MCPToolAdapter(tool_definition=definition, client=client)

        self.assertEqual(adapter.spec.name, "mcp_search")
        self.assertIn("mcp", adapter.spec.tags)
        self.assertEqual(
            adapter.spec.input_schema["properties"]["query"]["type"], "string"
        )

    def test_mcp_tool_invocation_returns_text_content(self) -> None:
        import asyncio

        async def _run():
            client = FakeMCPClient()
            definition = {"name": "echo", "description": "Echo tool."}
            adapter = MCPToolAdapter(tool_definition=definition, client=client)
            result = await adapter.invoke(
                {"text": "hello"},
                ToolContext(task_id="t", step_index=0),
            )
            return result, client

        result, client = asyncio.new_event_loop().run_until_complete(_run())
        self.assertTrue(result.ok)
        self.assertIn("result for echo", result.output)
        self.assertEqual(client.calls[0], ("echo", {"text": "hello"}))

    def test_mcp_tool_error_response(self) -> None:
        import asyncio

        async def _run():
            client = FakeMCPClient(
                responses={
                    "fail_tool": {
                        "content": [{"type": "text", "text": "something broke"}],
                        "isError": True,
                    }
                }
            )
            adapter = MCPToolAdapter(
                tool_definition={"name": "fail_tool", "description": ""},
                client=client,
            )
            return await adapter.invoke({}, ToolContext(task_id="t", step_index=0))

        result = asyncio.new_event_loop().run_until_complete(_run())
        self.assertFalse(result.ok)
        self.assertIn("something broke", result.error)

    def test_mcp_tools_plugin_registers_tools_on_session(self) -> None:
        temp_dir = make_workspace_temp_dir()
        client = FakeMCPClient()
        definitions = [
            {"name": "tool_a", "description": "Tool A."},
            {"name": "tool_b", "description": "Tool B."},
        ]
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {"state_dir": str(temp_dir), "profile": "local-dev"}
                ),
                include_builtin_tools=False,
                plugins=[MCPToolsPlugin(client=client, tool_definitions=definitions)],
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(session.tools.has("tool_a"))
        self.assertTrue(session.tools.has("tool_b"))


# ---------------------------------------------------------------------------
# Chroma adapter tests
# ---------------------------------------------------------------------------

class ChromaAdapterTests(unittest.TestCase):
    def test_chroma_retriever_returns_results(self) -> None:
        collection = FakeChromaCollection([
            {"id": "doc-1", "content": "First document", "metadata": {"source": "wiki"}, "distance": 0.2},
            {"id": "doc-2", "content": "Second document", "metadata": {"source": "faq"}, "distance": 0.8},
        ])
        retriever = ChromaRetrieverAdapter(collection)
        results = retriever.retrieve("test query", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "doc-1")
        self.assertEqual(results[0].content, "First document")
        self.assertEqual(results[0].metadata["source"], "wiki")
        # Score should be higher for lower distance
        self.assertGreater(results[0].score, results[1].score)

    def test_chroma_retriever_respects_top_k(self) -> None:
        collection = FakeChromaCollection([
            {"id": "doc-1", "content": "A", "distance": 0.1},
            {"id": "doc-2", "content": "B", "distance": 0.2},
            {"id": "doc-3", "content": "C", "distance": 0.3},
        ])
        retriever = ChromaRetrieverAdapter(collection)
        results = retriever.retrieve("test", top_k=1)
        self.assertEqual(len(results), 1)

    def test_chroma_plugin_registers_retriever_and_tool(self) -> None:
        temp_dir = make_workspace_temp_dir()
        collection = FakeChromaCollection([
            {"id": "doc-1", "content": "Test doc", "distance": 0.1},
        ])
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {"state_dir": str(temp_dir), "profile": "local-dev"}
                ),
                include_builtin_tools=False,
                plugins=[ChromaRetrievalPlugin(collection)],
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertIn("chroma", session.retrievers)
        self.assertTrue(session.tools.has("retrieve_documents"))


# ---------------------------------------------------------------------------
# FAISS adapter tests
# ---------------------------------------------------------------------------

class FAISSAdapterTests(unittest.TestCase):
    def _make_retriever(self) -> FAISSRetrieverAdapter:
        import numpy as np

        index = FakeFAISSIndex(dimension=3)
        docs = [
            FAISSDocument(id="d1", content="About cats", metadata={"topic": "animals"}),
            FAISSDocument(id="d2", content="About dogs", metadata={"topic": "animals"}),
            FAISSDocument(id="d3", content="About python", metadata={"topic": "code"}),
        ]
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype="float32")

        for vec, doc in zip(vectors, docs):
            index.add(vec.reshape(1, -1))

        def embed_fn(text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

        return FAISSRetrieverAdapter(index=index, documents=docs, embed_fn=embed_fn)

    def test_faiss_retriever_returns_ranked_results(self) -> None:
        retriever = self._make_retriever()
        results = retriever.retrieve("cats", top_k=3)
        self.assertEqual(len(results), 3)
        # First result should be closest to [1,0,0] which is d1
        self.assertEqual(results[0].id, "d1")

    def test_faiss_retriever_respects_top_k(self) -> None:
        retriever = self._make_retriever()
        results = retriever.retrieve("test", top_k=1)
        self.assertEqual(len(results), 1)

    def test_faiss_retriever_metadata_filter(self) -> None:
        retriever = self._make_retriever()
        results = retriever.retrieve("test", top_k=3, metadata_filter={"topic": "code"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "d3")

    def test_faiss_plugin_registers_on_session(self) -> None:
        import numpy as np

        temp_dir = make_workspace_temp_dir()
        index = FakeFAISSIndex(dimension=3)
        index.add(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
        docs = [FAISSDocument(id="d1", content="test")]

        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {"state_dir": str(temp_dir), "profile": "local-dev"}
                ),
                include_builtin_tools=False,
                plugins=[FAISSRetrievalPlugin(
                    index=index,
                    documents=docs,
                    embed_fn=lambda t: [1.0, 0.0, 0.0],
                )],
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertIn("faiss", session.retrievers)
        self.assertTrue(session.tools.has("retrieve_documents"))


# ---------------------------------------------------------------------------
# pgvector adapter tests
# ---------------------------------------------------------------------------

class PgVectorAdapterTests(unittest.TestCase):
    def test_pgvector_retriever_returns_results(self) -> None:
        rows = [
            ("pg-1", "First PostgreSQL doc", {"category": "db"}, 0.3),
            ("pg-2", "Second PostgreSQL doc", {"category": "db"}, 0.7),
        ]
        conn = FakePgConnection(rows)
        retriever = PgVectorRetrieverAdapter(
            connection=conn,
            table_name="docs",
            embed_fn=lambda t: [0.1, 0.2, 0.3],
        )
        results = retriever.retrieve("test query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "pg-1")
        self.assertEqual(results[0].content, "First PostgreSQL doc")
        self.assertEqual(results[0].metadata["category"], "db")

    def test_pgvector_score_inversely_related_to_distance(self) -> None:
        rows = [
            ("a", "Close doc", {}, 0.1),
            ("b", "Far doc", {}, 2.0),
        ]
        conn = FakePgConnection(rows)
        retriever = PgVectorRetrieverAdapter(
            connection=conn, table_name="t", embed_fn=lambda t: [0.0],
        )
        results = retriever.retrieve("q", top_k=2)
        self.assertGreater(results[0].score, results[1].score)


# ---------------------------------------------------------------------------
# Embedding provider tests
# ---------------------------------------------------------------------------

class EmbeddingProviderTests(unittest.TestCase):
    def test_callable_embedding_provider(self) -> None:
        fn = lambda text: [float(len(text)), 0.5, 0.1]
        provider = CallableEmbeddingProvider(fn)
        vec = provider.embed("hello")
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec[0], 5.0)

    def test_callable_embedding_batch(self) -> None:
        fn = lambda text: [float(len(text))]
        provider = CallableEmbeddingProvider(fn)
        results = provider.embed_batch(["a", "ab", "abc"])
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], [1.0])
        self.assertEqual(results[2], [3.0])


# ---------------------------------------------------------------------------
# Export bridge tests
# ---------------------------------------------------------------------------

SAMPLE_SPEC = ToolSpec(
    name="search_docs",
    description="Search documents by keyword.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    output_schema={
        "type": "object",
        "properties": {"results": {"type": "array"}},
    },
    tags=["retrieval"],
)


class ExportBridgeTests(unittest.TestCase):
    def test_to_openai_function(self) -> None:
        result = to_openai_function(SAMPLE_SPEC)
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "search_docs")
        self.assertEqual(
            result["function"]["parameters"]["properties"]["query"]["type"], "string"
        )

    def test_to_mcp_tool_definition(self) -> None:
        result = to_mcp_tool_definition(SAMPLE_SPEC)
        self.assertEqual(result["name"], "search_docs")
        self.assertIn("inputSchema", result)
        self.assertEqual(result["inputSchema"]["required"], ["query"])

    def test_to_langchain_tool_definition(self) -> None:
        result = to_langchain_tool_definition(SAMPLE_SPEC)
        self.assertEqual(result["name"], "search_docs")
        self.assertIn("args_schema", result)
        self.assertIn("return_schema", result)

    def test_to_openapi_schema(self) -> None:
        result = to_openapi_schema(SAMPLE_SPEC)
        self.assertEqual(result["operationId"], "search_docs")
        self.assertIn("requestBody", result)
        self.assertIn("responses", result)
        self.assertEqual(result["tags"], ["retrieval"])

    def test_export_all_openai(self) -> None:
        specs = [SAMPLE_SPEC]
        results = export_all(specs, format="openai")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["function"]["name"], "search_docs")

    def test_export_all_mcp(self) -> None:
        results = export_all([SAMPLE_SPEC], format="mcp")
        self.assertEqual(results[0]["name"], "search_docs")

    def test_export_all_unknown_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            export_all([SAMPLE_SPEC], format="unknown_format")

    def test_spec_without_output_schema(self) -> None:
        spec = ToolSpec(name="simple", description="A simple tool.")
        # OpenAPI should not have responses key
        result = to_openapi_schema(spec)
        self.assertNotIn("responses", result)
        # LangChain should not have return_schema key
        lc_result = to_langchain_tool_definition(spec)
        self.assertNotIn("return_schema", lc_result)


# ---------------------------------------------------------------------------
# Integration: multiple plugin types on one session
# ---------------------------------------------------------------------------

class MultiPluginIntegrationTests(unittest.TestCase):
    def test_mcp_and_chroma_plugins_together(self) -> None:
        temp_dir = make_workspace_temp_dir()
        mcp_client = FakeMCPClient()
        chroma_coll = FakeChromaCollection([
            {"id": "c1", "content": "ChromaDoc", "distance": 0.1},
        ])
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {"state_dir": str(temp_dir), "profile": "local-dev"}
                ),
                include_builtin_tools=False,
                plugins=[
                    MCPToolsPlugin(
                        client=mcp_client,
                        tool_definitions=[{"name": "mcp_echo", "description": "Echo."}],
                    ),
                    ChromaRetrievalPlugin(chroma_coll),
                ],
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(session.tools.has("mcp_echo"))
        self.assertTrue(session.tools.has("retrieve_documents"))
        self.assertIn("chroma", session.retrievers)
        self.assertIn("mcp-tools", session.plugins.names())
        self.assertIn("chroma-retrieval", session.plugins.names())


if __name__ == "__main__":
    unittest.main()
