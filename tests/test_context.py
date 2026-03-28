from __future__ import annotations

import asyncio
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_manager import AgentSession, RuntimeConfig
from agent_manager.context import PreCallPipeline
from agent_manager.memory import InMemoryKeywordRetriever, InMemoryLongTermStore, MemoryEntry
from agent_manager.providers.base import BaseProvider
from agent_manager.types import ContextSection, LoopState, Message, ProviderRequest, ProviderResult


def make_workspace_temp_dir() -> Path:
    root = Path.cwd() / ".tmp_tests"
    root.mkdir(exist_ok=True)
    temp_dir = root / uuid4().hex
    temp_dir.mkdir()
    return temp_dir


async def inject_custom_note(state, prepared, config, runtime):
    del state, config, runtime
    prepared.sections.append(
        ContextSection(
            key="custom_note",
            title="Custom Note",
            content="Prefer incremental changes with verification.",
            priority=69,
        )
    )
    return prepared


async def inject_large_low_priority_section(state, prepared, config, runtime):
    del state, config, runtime
    prepared.sections.append(
        ContextSection(
            key="huge_context",
            title="Huge Context",
            content="x" * 2000,
            priority=1,
        )
    )
    return prepared


class CaptureProvider(BaseProvider):
    provider_name = "capture-provider"

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[ProviderRequest] = []

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.requests.append(request)
        return ProviderResult(text="context-ready", stop_reason="completed")


class ContextPipelineTests(unittest.TestCase):
    def test_runtime_config_loads_context_settings(self) -> None:
        config = RuntimeConfig.from_dict(
            {
                "context": {
                    "history_window": 3,
                    "summary_trigger_messages": 5,
                    "pre_call_functions": [
                        "collect_recent_messages",
                        "finalize_messages",
                    ],
                }
            }
        )

        self.assertEqual(config.context.history_window, 3)
        self.assertEqual(config.context.summary_trigger_messages, 5)
        self.assertEqual(
            config.context.pre_call_functions,
            ["collect_recent_messages", "finalize_messages"],
        )

    def test_pipeline_uses_configured_functions_and_custom_step(self) -> None:
        config = RuntimeConfig.from_dict(
            {
                "context": {
                    "pre_call_functions": [
                        "collect_recent_messages",
                        "inject_custom_note",
                        "apply_token_budget",
                        "finalize_messages",
                    ]
                }
            }
        )
        pipeline = PreCallPipeline(
            custom_functions={"inject_custom_note": inject_custom_note}
        )
        state = LoopState(
            task_id="task-1",
            goal="Inspect the project",
            messages=[Message(role="user", content="Inspect the project")],
        )

        prepared = asyncio.run(pipeline.prepare_async(state, config))

        self.assertEqual(
            prepared.metadata["executed_steps"],
            [
                "collect_recent_messages",
                "inject_custom_note",
                "apply_token_budget",
                "finalize_messages",
            ],
        )
        self.assertIn("custom_note", [section.key for section in prepared.sections])
        self.assertIn("Custom Note:\nPrefer incremental changes with verification.", prepared.messages[0].content)

    def test_pipeline_injects_retrieval_and_memory_sections(self) -> None:
        config = RuntimeConfig.from_dict({})
        retriever = InMemoryKeywordRetriever()
        retriever.add(
            item_id="doc-1",
            content="Repository conventions favor verification and small steps.",
            metadata={"scope": "docs"},
        )
        memory_store = InMemoryLongTermStore()
        memory_store.put(
            MemoryEntry(
                key="conventions",
                value="Conventions include concise comments and focused patches.",
                source="team",
                scope="workspace",
            )
        )
        pipeline = PreCallPipeline(retriever=retriever, memory_store=memory_store)
        state = LoopState(
            task_id="task-1",
            goal="conventions",
            messages=[Message(role="user", content="conventions")],
        )

        prepared = asyncio.run(pipeline.prepare_async(state, config))

        section_keys = [section.key for section in prepared.sections]
        self.assertIn("retrieved_1", section_keys)
        self.assertIn("memory_fact_1", section_keys)
        self.assertIn("Retrieved Knowledge 1", prepared.messages[0].content)
        self.assertIn("Memory Fact 1", prepared.messages[0].content)

    def test_token_budget_drops_low_priority_sections(self) -> None:
        config = RuntimeConfig.from_dict(
            {
                "runtime": {
                    "max_context_tokens": 120,
                    "max_output_tokens": 40,
                },
                "context": {
                    "pre_call_functions": [
                        "collect_recent_messages",
                        "inject_large_low_priority_section",
                        "apply_token_budget",
                        "finalize_messages",
                    ]
                },
            }
        )
        pipeline = PreCallPipeline(
            custom_functions={
                "inject_large_low_priority_section": inject_large_low_priority_section
            }
        )
        state = LoopState(
            task_id="task-1",
            goal="Keep the important context only",
            messages=[Message(role="user", content="Keep the important context only")],
        )

        prepared = asyncio.run(pipeline.prepare_async(state, config))

        self.assertIn("huge_context", prepared.dropped_sections)
        self.assertNotIn("Huge Context", prepared.messages[0].content)

    def test_session_uses_custom_pre_call_function_in_provider_request(self) -> None:
        temp_dir = make_workspace_temp_dir()
        provider = CaptureProvider()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "provider": {"name": "echo", "model": "echo-v1"},
                    "runtime": {"max_steps": 2},
                    "context": {
                        "pre_call_functions": [
                            "collect_recent_messages",
                            "inject_custom_note",
                            "apply_token_budget",
                            "finalize_messages",
                        ]
                    },
                }
            )
            session = AgentSession(
                config=config,
                provider=provider,
                include_builtin_tools=False,
                working_directory=temp_dir,
                pre_call_functions={"inject_custom_note": inject_custom_note},
            )
            result = session.run("Prepare request context")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(provider.requests[0].messages[0].role, "system")
        self.assertIn(
            "Custom Note:\nPrefer incremental changes with verification.",
            provider.requests[0].messages[0].content,
        )
        self.assertEqual(
            result.state.metadata["prepared_context"]["executed_steps"],
            [
                "collect_recent_messages",
                "inject_custom_note",
                "apply_token_budget",
                "finalize_messages",
            ],
        )


if __name__ == "__main__":
    unittest.main()
