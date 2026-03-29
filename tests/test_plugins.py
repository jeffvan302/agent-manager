from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_manager import AgentSession, RuntimeConfig
from agent_manager.plugins import (
    LangChainToolsPlugin,
    LlamaIndexRetrievalPlugin,
    OpenAPIOperation,
    OpenAPIToolsPlugin,
    Plugin,
    TinyDBToolsPlugin,
)
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from agent_manager.types import ToolCallRequest


def make_workspace_temp_dir() -> Path:
    root = Path.cwd() / ".tmp_tests"
    root.mkdir(exist_ok=True)
    temp_dir = root / uuid4().hex
    temp_dir.mkdir()
    return temp_dir


class FakeArgsSchema:
    @staticmethod
    def model_json_schema() -> dict:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }


class FakeLangChainTool:
    name = "chain_echo"
    description = "Echo the provided value."
    args_schema = FakeArgsSchema()

    def invoke(self, arguments: dict) -> dict:
        return {"echo": arguments["value"]}


class FakeLlamaNode:
    def __init__(self) -> None:
        self.text = "retrieved from llamaindex"
        self.metadata = {"source": "knowledge-base"}
        self.score = 0.9
        self.node_id = "node-1"


class FakeLlamaRetriever:
    def retrieve(self, query: str):
        del query
        return [FakeLlamaNode()]


class FakeHttpTool(BaseTool):
    spec = ToolSpec(
        name="fake_http_tool",
        description="Capture outgoing OpenAPI requests.",
        tags=["network"],
        permissions=["network:request"],
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        del context
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_name="http_request",
            ok=True,
            output={
                "url": arguments["url"],
                "status": 200,
                "headers": arguments.get("headers", {}),
                "body": '{"ok": true}',
                "truncated": False,
            },
        )


class PluginCaptureTool(BaseTool):
    spec = ToolSpec(
        name="plugin_tool",
        description="Registered by a plugin.",
    )

    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        del arguments, context
        return ToolResult(tool_name=self.spec.name, ok=True, output={"ok": True})


class SessionToolPlugin(Plugin):
    name = "session-tool-plugin"

    def register(self, target) -> None:
        target.register_tool(PluginCaptureTool(), replace=True)


class FakeTinyDocument(dict):
    def __init__(self, data: dict, doc_id: int) -> None:
        super().__init__(data)
        self.doc_id = doc_id


class FakeTinyTable:
    def __init__(self) -> None:
        self._documents: list[FakeTinyDocument] = []
        self._next_id = 1

    def insert(self, document: dict) -> int:
        doc_id = self._next_id
        self._next_id += 1
        self._documents.append(FakeTinyDocument(dict(document), doc_id))
        return doc_id

    def all(self) -> list[FakeTinyDocument]:
        return list(self._documents)

    def search(self, cond) -> list[FakeTinyDocument]:
        return [document for document in self._documents if cond(document)]

    def get(self, cond=None, doc_id=None):
        if doc_id is not None:
            for document in self._documents:
                if document.doc_id == doc_id:
                    return document
            return None
        if cond is None:
            return self._documents[0] if self._documents else None
        for document in self._documents:
            if cond(document):
                return document
        return None

    def update(self, fields: dict, cond=None, doc_ids=None) -> list[int]:
        matched = self._matching_documents(cond=cond, doc_ids=doc_ids)
        for document in matched:
            document.update(fields)
        return [document.doc_id for document in matched]

    def remove(self, cond=None, doc_ids=None) -> list[int]:
        matched = self._matching_documents(cond=cond, doc_ids=doc_ids)
        matched_ids = {document.doc_id for document in matched}
        self._documents = [
            document for document in self._documents if document.doc_id not in matched_ids
        ]
        return sorted(matched_ids)

    def upsert(self, document: dict, cond=None) -> list[int]:
        matched = self.search(cond) if cond is not None else []
        if matched:
            for existing in matched:
                existing.update(document)
            return [existing.doc_id for existing in matched]
        return [self.insert(document)]

    def _matching_documents(self, *, cond=None, doc_ids=None) -> list[FakeTinyDocument]:
        if doc_ids is not None:
            ids = {int(item) for item in doc_ids}
            return [document for document in self._documents if document.doc_id in ids]
        if cond is None:
            return []
        return self.search(cond)


class FakeTinyDB:
    def __init__(self) -> None:
        self._tables: dict[str, FakeTinyTable] = {}

    def table(self, name: str) -> FakeTinyTable:
        if name not in self._tables:
            self._tables[name] = FakeTinyTable()
        return self._tables[name]


class PluginAdapterTests(unittest.TestCase):
    def test_langchain_tools_plugin_registers_and_executes_tool(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {
                        "state_dir": str(temp_dir),
                        "profile": "local-dev",
                    }
                ),
                include_builtin_tools=False,
                plugins=[LangChainToolsPlugin([FakeLangChainTool()])],
            )
            result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tool-1",
                    name="chain_echo",
                    arguments={"value": "phase-5"},
                ),
                ToolContext(task_id="task-1", step_index=0, tool_call_id="tool-1"),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(session.tools.has("chain_echo"))
        self.assertEqual(result.output["echo"], "phase-5")

    def test_llamaindex_plugin_registers_default_retriever_and_tool(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {
                        "state_dir": str(temp_dir),
                        "profile": "local-dev",
                    }
                ),
                include_builtin_tools=False,
                plugins=[LlamaIndexRetrievalPlugin(FakeLlamaRetriever())],
            )
            result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tool-2",
                    name="retrieve_documents",
                    arguments={"query": "anything"},
                ),
                ToolContext(task_id="task-1", step_index=0, tool_call_id="tool-2"),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertIn("llamaindex", session.retrievers)
        self.assertIsNotNone(session.retriever)
        self.assertEqual(result.output["results"][0]["id"], "node-1")
        self.assertEqual(result.output["results"][0]["metadata"]["source"], "knowledge-base")

    def test_openapi_tools_plugin_registers_operation_tool(self) -> None:
        temp_dir = make_workspace_temp_dir()
        http_tool = FakeHttpTool()
        operation = OpenAPIOperation(
            name="get_pet",
            description="Fetch a pet by id.",
            base_url="https://api.example.test/v1",
            path="/pets/{pet_id}",
            input_schema={
                "type": "object",
                "properties": {
                    "pet_id": {"type": "string"},
                    "query": {"type": "object"},
                },
                "required": ["pet_id"],
            },
        )
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {
                        "state_dir": str(temp_dir),
                        "profile": "local-dev",
                    }
                ),
                include_builtin_tools=False,
                plugins=[OpenAPIToolsPlugin([operation], http_tool=http_tool)],
            )
            result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tool-3",
                    name="get_pet",
                    arguments={
                        "pet_id": "pet-123",
                        "query": {"include": "owner"},
                    },
                ),
                ToolContext(task_id="task-1", step_index=0, tool_call_id="tool-3"),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(session.tools.has("get_pet"))
        self.assertEqual(
            http_tool.calls[0]["url"],
            "https://api.example.test/v1/pets/pet-123?include=owner",
        )
        self.assertEqual(result.output["url"], http_tool.calls[0]["url"])

    def test_session_can_register_plugin_after_initialization(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {
                        "state_dir": str(temp_dir),
                        "profile": "local-dev",
                    }
                ),
                include_builtin_tools=False,
            )
            session.register_plugin(SessionToolPlugin())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(session.tools.has("plugin_tool"))
        self.assertIn("session-tool-plugin", session.plugins.names())

    def test_tinydb_tools_plugin_registers_configured_wrapper(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            session = AgentSession(
                config=RuntimeConfig.from_dict(
                    {
                        "state_dir": str(temp_dir),
                        "profile": "local-dev",
                    }
                ),
                include_builtin_tools=False,
                plugins=[
                    TinyDBToolsPlugin(
                        database=FakeTinyDB(),
                        table_name="notes",
                        tool_name="notes_db",
                        allowed_operations=[
                            "insert",
                            "search",
                            "get",
                            "update",
                            "remove",
                            "count",
                        ],
                    )
                ],
            )
            context = ToolContext(task_id="task-1", step_index=0, tool_call_id="tiny-1")
            insert_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tiny-1",
                    name="notes_db",
                    arguments={
                        "action": "insert",
                        "document": {"kind": "todo", "title": "Ship TinyDB tool"},
                    },
                ),
                context,
            )
            search_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tiny-2",
                    name="notes_db",
                    arguments={
                        "action": "search",
                        "query": {"kind": "todo"},
                    },
                ),
                ToolContext(task_id="task-1", step_index=1, tool_call_id="tiny-2"),
            )
            update_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tiny-3",
                    name="notes_db",
                    arguments={
                        "action": "update",
                        "query": {"title": "Ship TinyDB tool"},
                        "fields": {"done": True},
                    },
                ),
                ToolContext(task_id="task-1", step_index=2, tool_call_id="tiny-3"),
            )
            get_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tiny-4",
                    name="notes_db",
                    arguments={
                        "action": "get",
                        "query": {"done": True},
                    },
                ),
                ToolContext(task_id="task-1", step_index=3, tool_call_id="tiny-4"),
            )
            count_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tiny-5",
                    name="notes_db",
                    arguments={
                        "action": "count",
                        "query": {"kind": "todo"},
                    },
                ),
                ToolContext(task_id="task-1", step_index=4, tool_call_id="tiny-5"),
            )
            disallowed_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tiny-6",
                    name="notes_db",
                    arguments={"action": "all"},
                ),
                ToolContext(task_id="task-1", step_index=5, tool_call_id="tiny-6"),
            )
            remove_result = session.tool_executor.execute(
                ToolCallRequest(
                    id="tiny-7",
                    name="notes_db",
                    arguments={
                        "action": "remove",
                        "query": {"done": True},
                    },
                ),
                ToolContext(task_id="task-1", step_index=6, tool_call_id="tiny-7"),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertTrue(session.tools.has("notes_db"))
        self.assertTrue(insert_result.ok)
        self.assertEqual(search_result.output["count"], 1)
        self.assertEqual(search_result.output["documents"][0]["title"], "Ship TinyDB tool")
        self.assertEqual(update_result.output["count"], 1)
        self.assertEqual(get_result.output["document"]["done"], True)
        self.assertEqual(count_result.output["count"], 1)
        self.assertFalse(disallowed_result.ok)
        self.assertIn("Allowed actions", disallowed_result.error)
        self.assertEqual(remove_result.output["count"], 1)


if __name__ == "__main__":
    unittest.main()
