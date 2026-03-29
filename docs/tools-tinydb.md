# TinyDB Tool Setup

This guide documents the TinyDB wrapper plugin added to `agent-manager`.

## What it is

The TinyDB integration exposes a TinyDB database or table as a normal runtime tool.

Classes:

- `TinyDBToolsPlugin`
- `TinyDBToolAdapter`

## Install

```bash
pip install -e .[tinydb]
```

## Supported actions

The TinyDB wrapper supports:

- `insert`
- `upsert`
- `get`
- `search`
- `all`
- `count`
- `update`
- `remove`

Safety rule:

- `update` and `remove` require either `query` or `doc_ids`

## Example 1: wrap a TinyDB file

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.plugins import TinyDBToolsPlugin

session = AgentSession(
    config=RuntimeConfig.from_dict({"profile": "coding-agent"}),
    plugins=[
        TinyDBToolsPlugin(
            path=".agent_manager/notes.json",
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

print(session.tools.names())
```

## Example 2: wrap an existing TinyDB object

```python
from tinydb import TinyDB

from agent_manager import AgentSession, RuntimeConfig
from agent_manager.plugins import TinyDBToolsPlugin

db = TinyDB("app-data.json")

session = AgentSession(
    config=RuntimeConfig.from_dict({"profile": "coding-agent"}),
    plugins=[
        TinyDBToolsPlugin(
            database=db,
            table_name="jobs",
            tool_name="jobs_db",
        )
    ],
)
```

## Direct execution example

```python
from agent_manager.tools.base import ToolContext
from agent_manager.types import ToolCallRequest

insert_result = session.tool_executor.execute(
    ToolCallRequest(
        id="tiny-1",
        name="notes_db",
        arguments={
            "action": "insert",
            "document": {
                "kind": "todo",
                "title": "Document TinyDB integration",
                "status": "open",
            },
        },
    ),
    ToolContext(
        task_id="tinydb-demo",
        step_index=0,
        tool_call_id="tiny-1",
        working_directory=str(session.working_directory),
    ),
)

search_result = session.tool_executor.execute(
    ToolCallRequest(
        id="tiny-2",
        name="notes_db",
        arguments={
            "action": "search",
            "query": {"kind": "todo", "status": "open"},
        },
    ),
    ToolContext(
        task_id="tinydb-demo",
        step_index=1,
        tool_call_id="tiny-2",
        working_directory=str(session.working_directory),
    ),
)

print(insert_result.output)
print(search_result.output)
```

## Tool input shape

Typical input fields:

```json
{
  "action": "search",
  "document": {},
  "fields": {},
  "query": {
    "kind": "todo"
  },
  "doc_ids": [1, 2],
  "limit": 20
}
```

Usage notes:

- `document` is used by `insert` and `upsert`
- `fields` is used by `update`
- `query` is a simple field-to-value mapping
- dot notation works for nested lookups such as `"author.name": "Theo"`
- `limit` is mainly useful for `search`

## Recommended patterns

Good TinyDB uses:

- local notes
- job queues
- cached research results
- lightweight automation state
- small internal catalogs

Consider a different database when:

- you need concurrent writes from many processes
- you need joins or relational queries
- the dataset is large

## Using it from a model

Once registered, the model sees TinyDB as just another tool.

Example prompt:

```text
Use notes_db to insert a planning note for the TinyDB guide, then search for open todo items and summarize them.
```
