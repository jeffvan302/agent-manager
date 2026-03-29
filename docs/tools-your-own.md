# Build Your Own Tool

`agent-manager` lets you add tools in two main ways:

1. subclass `BaseTool`
2. register a plain callable with `ToolRegistry.register_callable()`

Use `BaseTool` when the tool has its own state or helper methods. Use a callable for small utility tools.

## Option 1: subclass `BaseTool`

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec


class EchoPathTool(BaseTool):
    spec = ToolSpec(
        name="echo_path",
        description="Return the provided path along with the current task id.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["task_id", "path"],
        },
        tags=["example"],
    )

    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.spec.name,
            ok=True,
            output={
                "task_id": context.task_id,
                "path": arguments["path"],
            },
        )


session = AgentSession(config=RuntimeConfig.from_dict({"profile": "coding-agent"}))
session.register_tool(EchoPathTool())
```

## Option 2: register a callable

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.tools.base import ToolSpec

session = AgentSession(config=RuntimeConfig.from_dict({"profile": "coding-agent"}))

session.tools.register_callable(
    ToolSpec(
        name="echo_value",
        description="Echo a string value and the active task id.",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        },
        tags=["example"],
    ),
    lambda arguments, context: {
        "value": arguments["value"],
        "task_id": context.task_id,
    },
)
```

## Recommended `ToolSpec` fields

The current tool contract supports:

- `name`
- `description`
- `input_schema`
- `output_schema`
- `tags`
- `permissions`
- `timeout_seconds`
- `retry_count`
- `retry_backoff_seconds`

Example for a network-aware tool:

```python
from agent_manager.tools.base import ToolSpec

ToolSpec(
    name="lookup_release_notes",
    description="Fetch release notes from an approved API endpoint.",
    input_schema={
        "type": "object",
        "properties": {
            "version": {"type": "string"},
        },
        "required": ["version"],
    },
    tags=["network"],
    permissions=["network:request"],
    timeout_seconds=15.0,
    retry_count=2,
    retry_backoff_seconds=0.5,
)
```

## How tool execution works

When the model asks for a tool:

1. the runtime matches the tool name against the registry
2. policy checks run
3. the tool executes with a `ToolContext`
4. the result is normalized to `ToolResult`
5. the tool observation is appended to loop state

The `ToolContext` includes:

- `task_id`
- `step_index`
- `tool_call_id`
- `working_directory`
- `metadata`

## Registering tools through plugins

If you want reusable packaging, create a plugin that registers tools onto a session.

```python
from agent_manager.plugins import Plugin


class ExampleToolPlugin(Plugin):
    name = "example-tool-plugin"
    description = "Register a reusable example tool."

    def register(self, target) -> None:
        target.register_tool(EchoPathTool(), replace=True)
```

Usage:

```python
session = AgentSession(
    config=RuntimeConfig.from_dict({"profile": "coding-agent"}),
    plugins=[ExampleToolPlugin()],
)
```

## Testing your own tool

The easiest test is direct execution through the executor.

```python
from agent_manager.tools.base import ToolContext
from agent_manager.types import ToolCallRequest

call = ToolCallRequest(
    id="tool-test-1",
    name="echo_path",
    arguments={"path": "src/app.py"},
)

context = ToolContext(
    task_id="tool-test",
    step_index=0,
    tool_call_id="tool-test-1",
    working_directory=str(session.working_directory),
)

result = session.tool_executor.execute(call, context)
assert result.ok is True
assert result.output["path"] == "src/app.py"
```

## Good design tips

- keep tool names stable and explicit
- make `input_schema` small and easy for the model to satisfy
- use `permissions` and `tags` so policy decisions stay clear
- return structured dictionaries whenever possible
- put side effects behind explicit arguments rather than hidden behavior
- keep write and network tools separate from read-only tools
