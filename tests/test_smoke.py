from __future__ import annotations

import asyncio
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_manager import AgentSession, RuntimeConfig
from agent_manager.errors import ProviderResourceExhaustedError
from agent_manager.providers.base import BaseProvider
from agent_manager.providers.factory import available_providers, build_provider
from agent_manager.state.store import JsonFileStateStore
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from agent_manager.tools.registry import ToolRegistry
from agent_manager.types import ProviderRequest, ProviderResult, ToolCallRequest


def make_workspace_temp_dir() -> Path:
    root = Path.cwd() / ".tmp_tests"
    root.mkdir(exist_ok=True)
    temp_dir = root / uuid4().hex
    temp_dir.mkdir()
    return temp_dir


class CaptureTool(BaseTool):
    spec = ToolSpec(
        name="capture_context",
        description="Capture the tool context for testing.",
    )

    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.spec.name,
            ok=True,
            output={"arguments": arguments, "tool_call_id": context.tool_call_id},
        )


class SlowTool(BaseTool):
    spec = ToolSpec(
        name="slow_tool",
        description="Sleep longer than the configured timeout.",
        timeout_seconds=0.01,
    )

    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        del arguments, context
        await asyncio.sleep(0.05)
        return ToolResult(tool_name=self.spec.name, ok=True, output={"done": True})


class BlockedTool(BaseTool):
    spec = ToolSpec(
        name="blocked_tool",
        description="A tool blocked by the readonly profile.",
        tags=["shell"],
    )

    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        del arguments, context
        return ToolResult(tool_name=self.spec.name, ok=True, output={"done": True})


class ToolCallingProvider(BaseProvider):
    provider_name = "test-tool-caller"

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        del request
        if self._calls == 0:
            self._calls += 1
            return ProviderResult(
                tool_calls=[
                    ToolCallRequest(
                        id="provider-call-123",
                        name="capture_context",
                        arguments={"value": "hello"},
                    )
                ],
                stop_reason="tool_call",
            )
        return ProviderResult(text="done", stop_reason="completed")


class SingleToolCallProvider(BaseProvider):
    provider_name = "single-tool-caller"

    def __init__(self, tool_name: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self._calls = 0

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        del request
        if self._calls == 0:
            self._calls += 1
            return ProviderResult(
                tool_calls=[
                    ToolCallRequest(
                        id=f"{self.tool_name}-call-1",
                        name=self.tool_name,
                        arguments={},
                    )
                ],
                stop_reason="tool_call",
            )
        return ProviderResult(text="done", stop_reason="completed")


class SlowProvider(BaseProvider):
    provider_name = "slow-provider"

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        del request
        await asyncio.sleep(0.05)
        return ProviderResult(text="too slow", stop_reason="completed")


class FailingProvider(BaseProvider):
    provider_name = "failing-provider"

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        del request
        raise RuntimeError("provider boom")


class ResourceExhaustedProvider(BaseProvider):
    """Provider that raises a resource exhaustion error on the first call."""

    provider_name = "resource-exhausted-provider"

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        del request
        raise ProviderResourceExhaustedError(
            "Quota exceeded.",
            provider="resource-exhausted-provider",
            kind="quota_exhausted",
            status_code=429,
            retry_after_seconds=60.0,
        )


class SmokeTests(unittest.TestCase):
    def test_echo_provider_is_registered(self) -> None:
        self.assertIn("echo", available_providers())
        self.assertIn("openai", available_providers())
        self.assertIn("vllm", available_providers())

    def test_scaffolded_provider_can_be_built(self) -> None:
        provider = build_provider("openai")
        self.assertEqual(provider.provider_name, "openai")
        vllm = build_provider("vllm")
        self.assertEqual(vllm.provider_name, "vllm")

    def test_runtime_can_complete_a_basic_run(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "provider": {"name": "echo", "model": "echo-v1"},
                    "runtime": {"max_steps": 2},
                }
            )
            session = AgentSession(config=config)
            result = session.run("Phase one smoke test")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.state.status, "completed")
        self.assertIn("Phase one smoke test", result.output_text)

    def test_runtime_can_complete_an_async_run(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "provider": {"name": "echo", "model": "echo-v1"},
                    "runtime": {"max_steps": 2},
                }
            )
            session = AgentSession(config=config)
            result = asyncio.run(session.run_async("Phase one async smoke test"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.state.status, "completed")
        self.assertIn("Phase one async smoke test", result.output_text)

    def test_tool_call_id_is_preserved(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "provider": {"name": "echo", "model": "echo-v1"},
                    "runtime": {"max_steps": 3},
                }
            )
            tools = ToolRegistry([CaptureTool()])
            session = AgentSession(
                config=config,
                provider=ToolCallingProvider(),
                tools=tools,
            )
            result = session.run("Run tool flow")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.state.status, "completed")
        self.assertEqual(result.tool_results[0]["metadata"]["tool_call_id"], "provider-call-123")
        self.assertEqual(result.tool_results[0]["output"]["tool_call_id"], "provider-call-123")
        self.assertEqual(result.state.messages[-3].metadata["tool_calls"][0]["id"], "provider-call-123")
        self.assertEqual(result.state.messages[-2].metadata["tool_call_id"], "provider-call-123")

    def test_config_can_load_from_toml(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config_path = temp_dir / "runtime.toml"
            config_path.write_text(
                '\n'.join(
                    [
                        '[provider]',
                        'name = "echo"',
                        'model = "test-model"',
                        '',
                        '[runtime]',
                        'max_steps = 3',
                    ]
                ),
                encoding="utf-8",
            )

            config = RuntimeConfig.from_file(config_path, apply_env=False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(config.provider.model, "test-model")
        self.assertEqual(config.runtime.max_steps, 3)

    def test_state_store_sanitizes_task_id(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            store = JsonFileStateStore(temp_dir)
            config = RuntimeConfig.from_dict({"state_dir": str(temp_dir)})
            session = AgentSession(config=config, state_store=store)
            task_id = "../unsafe:task/id"
            result = session.run("sanitize me", task_id=task_id)
            files = list(temp_dir.glob("*.json"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.task_id, task_id)
        self.assertEqual(len(files), 1)
        self.assertNotIn("..", files[0].name)
        self.assertNotIn("/", files[0].name)
        self.assertNotIn("\\", files[0].name)

    def test_provider_timeout_stops_cleanly(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "runtime": {"timeout_seconds": 0.01, "max_steps": 2},
                }
            )
            session = AgentSession(config=config, provider=SlowProvider())
            result = session.run("timeout test")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.stop_reason, "timeout")
        self.assertEqual(result.state.status, "timeout")

    def test_policy_violation_stops_cleanly(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "profile": "readonly",
                    "runtime": {"max_steps": 2},
                }
            )
            tools = ToolRegistry([BlockedTool()])
            session = AgentSession(
                config=config,
                provider=SingleToolCallProvider("blocked_tool"),
                tools=tools,
            )
            result = session.run("policy test")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.stop_reason, "policy_violation")
        self.assertEqual(result.state.status, "policy_violation")

    def test_repeated_failure_stops_cleanly(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "runtime": {
                        "max_steps": 5,
                        "max_consecutive_failures": 2,
                    },
                }
            )
            session = AgentSession(config=config, provider=FailingProvider())
            result = session.run("failure test")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.stop_reason, "repeated_failure")
        self.assertEqual(result.state.status, "repeated_failure")
        self.assertEqual(result.state.metadata["consecutive_failures"], 2)

    def test_tool_timeout_stops_cleanly(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "runtime": {"timeout_seconds": 2, "max_steps": 2},
                }
            )
            tools = ToolRegistry([SlowTool()])
            session = AgentSession(
                config=config,
                provider=SingleToolCallProvider("slow_tool"),
                tools=tools,
            )
            result = session.run("tool timeout test")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.stop_reason, "timeout")
        self.assertEqual(result.state.status, "timeout")

    def test_resource_exhaustion_stops_cleanly(self) -> None:
        """Verify that a provider resource exhaustion error stops the loop
        with the correct stop_reason and exposes the error details."""
        temp_dir = make_workspace_temp_dir()
        try:
            config = RuntimeConfig.from_dict(
                {
                    "state_dir": str(temp_dir),
                    "runtime": {"max_steps": 3},
                }
            )
            session = AgentSession(
                config=config,
                provider=ResourceExhaustedProvider(),
            )
            result = session.run("resource exhaustion test")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.stop_reason, "resource_exhausted")
        self.assertEqual(result.state.status, "resource_exhausted")
        self.assertIsNotNone(result.resource_exhaustion)
        self.assertEqual(result.resource_exhaustion["kind"], "quota_exhausted")
        self.assertEqual(result.resource_exhaustion["status_code"], 429)
        self.assertEqual(result.resource_exhaustion["retry_after_seconds"], 60.0)
        self.assertEqual(
            result.resource_exhaustion["provider"],
            "resource-exhausted-provider",
        )


if __name__ == "__main__":
    unittest.main()
