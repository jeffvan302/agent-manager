# Tools Overview

This guide documents the built-in tools and the integration-style tools already supported by `agent-manager`.

## How tools are registered

By default, `AgentSession` registers the built-in tools when `include_builtin_tools=True`.

```python
from agent_manager import AgentSession, RuntimeConfig

session = AgentSession(
    config=RuntimeConfig.from_dict({"profile": "coding-agent"})
)

print(session.tools.names())
```

Important note:

- `retrieve_documents` is only registered when the session has a retriever.

## Built-in tools

### `list_directory`

Purpose:

- list files and folders inside the allowed working-directory scope

Typical arguments:

```json
{
  "path": ".",
  "recursive": true
}
```

Example:

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.tools.base import ToolContext
from agent_manager.types import ToolCallRequest

session = AgentSession(config=RuntimeConfig.from_dict({"profile": "coding-agent"}))
result = session.tool_executor.execute(
    ToolCallRequest(id="tool-1", name="list_directory", arguments={"path": ".", "recursive": False}),
    ToolContext(task_id="demo", step_index=0, tool_call_id="tool-1", working_directory=str(session.working_directory)),
)
print(result.output)
```

### `read_file`

Purpose:

- read UTF-8 text files inside the allowed filesystem scope

Typical arguments:

```json
{
  "path": "requirements.md",
  "max_chars": 4000
}
```

### `write_file`

Purpose:

- write UTF-8 text files inside the allowed filesystem scope

Typical arguments:

```json
{
  "path": "notes/todo.md",
  "content": "# Next steps",
  "create_parents": true
}
```

### `run_shell_command`

Purpose:

- run a shell command in the session working directory

Typical arguments:

```json
{
  "command": "python -m unittest discover -s tests",
  "timeout_seconds": 120
}
```

Notes:

- the built-in tool blocks obviously dangerous command patterns
- shell access can be denied by policy

### `http_request`

Purpose:

- fetch a page through the browser-backed `google_search_tool` engine for normal `GET` page requests, or send a raw HTTP request for API-style calls

Typical arguments:

```json
{
  "url": "https://httpbin.org/get",
  "method": "GET"
}
```

Browser-backed page fetch example:

```json
{
  "url": "https://example.com",
  "method": "GET",
  "response_format": "text",
  "headless": true,
  "cookie_file": "google_cookies.json"
}
```

Rendered HTML example:

```json
{
  "url": "https://example.com",
  "response_format": "html"
}
```

PDF export example:

```json
{
  "url": "https://example.com/report",
  "response_format": "pdf",
  "output_path": "report.pdf",
  "page_format": "Letter",
  "print_background": false
}
```

Raw API request example:

```json
{
  "url": "https://httpbin.org/post",
  "method": "POST",
  "engine": "raw",
  "headers": {
    "Accept": "application/json"
  },
  "json": {
    "name": "demo"
  }
}
```

### `web_search`

Purpose:

- search the web through the configured search backend

Configurable backends:

- `google`
- `duckduckgo`
- `serpapi`
- `tavily`
- `brave`

Default backend:

- `google` via the installed `google_search_tool`

Typical arguments:

```json
{
  "query": "TinyDB Python examples",
  "limit": 5
}
```

Configuration example:

```toml
[tools.web_search]
enabled = true
backend = "tavily"
api_key_env = "TAVILY_API_KEY"
timeout_seconds = 20
max_results = 5

[tools.web_search.settings]
search_depth = "basic"
topic = "general"
```

Setup guide:

- [Web Search Tool Setup](./tools-web-search.md)

### `retrieve_documents`

Purpose:

- retrieve indexed chunks from an attached retriever

Typical arguments:

```json
{
  "query": "context pipeline requirements",
  "top_k": 3,
  "metadata_filter": {
    "source": "requirements.md"
  }
}
```

Setup guide:

- [Retrieval Tool Setup](./tools-retrieval.md)

## More involved tool setups

These are the tool setups that deserve their own guides.

- [Retrieval Tool Setup](./tools-retrieval.md)
- [Web Search Tool Setup](./tools-web-search.md)
- [TinyDB Tool Setup](./tools-tinydb.md)
- [Build Your Own Tool](./tools-your-own.md)

## Integration and plugin tools

The codebase also supports plugin-style tool or retrieval integrations.

### TinyDB

- plugin: `TinyDBToolsPlugin`
- adapter: `TinyDBToolAdapter`
- guide: [TinyDB Tool Setup](./tools-tinydb.md)

### OpenAPI operations

- plugin: `OpenAPIToolsPlugin`
- adapter: `OpenAPIToolAdapter`
- lets you expose API operations as regular tools

### MCP tools

- plugin: `MCPToolsPlugin`
- adapter: `MCPToolAdapter`
- lets MCP server tools appear as runtime tools

### LangChain tools

- plugin: `LangChainToolsPlugin`
- adapter: `LangChainToolAdapter`

### Retrieval integrations

- `LlamaIndexRetrievalPlugin`
- `ChromaRetrievalPlugin`
- `FAISSRetrievalPlugin`
- `PgVectorRetrievalPlugin`

## Example: direct tool execution

Tools are usually called by the model, but you can execute them directly for testing.

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.tools.base import ToolContext
from agent_manager.types import ToolCallRequest

session = AgentSession(config=RuntimeConfig.from_dict({"profile": "coding-agent"}))

call = ToolCallRequest(
    id="manual-1",
    name="http_request",
    arguments={
        "url": "https://httpbin.org/get",
        "method": "GET",
    },
)
context = ToolContext(
    task_id="manual-tool-test",
    step_index=0,
    tool_call_id="manual-1",
    working_directory=str(session.working_directory),
)

result = session.tool_executor.execute(call, context)
print(result.ok)
print(result.output)
```

## CLI tool testing

You can also test tools directly from the command line without going through a model run.

List the tools available under a config:

```bash
tool-test --config test-conn.toml --list
```

Show the schema for a tool:

```bash
tool-test --config test-conn.toml --schema web_search
```

Run a tool with JSON input:

```bash
tool-test --config test-conn.toml http_request "{\"url\":\"https://httpbin.org/get\",\"method\":\"GET\"}"
```

Run a tool with a plain string input when the tool has exactly one required field:

```bash
tool-test --config test-conn.toml web_search "budget GPU for yolo12 training"
tool-test --config test-conn.toml read_file "requirements.md"
tool-test --config test-conn.toml run_shell_command "python -V"
```

Load input from a JSON file:

```bash
tool-test --config test-conn.toml write_file @tool-input.json
```

Notes:

- `tool-test` reuses the current runtime config and tool policy
- if you use `profile="readonly"`, some tools may still be blocked unless explicitly allowed
- `retrieve_documents` is only available when a retriever is configured in the session setup

## Tool policies

Tool access is shaped by the runtime profile and optional policy overrides.

Default profile names in the code:

- `readonly`
- `local-dev`
- `coding-agent`
- `unrestricted-lab`

Example policy override:

```python
from agent_manager import RuntimeConfig

config = RuntimeConfig.from_dict(
    {
        "profile": "coding-agent",
        "tool_policy": {
            "denied_tools": ["write_file"],
            "denied_permissions": ["process:execute"],
        },
    }
)
```

## Next docs

- [Build Your Own Tool](./tools-your-own.md)
- [Context Manager](./context-manager.md)
- [Advanced Two-Agent Coding Example](./advanced-two-agent-coding.md)
