# Context Manager

This guide explains how `agent-manager` assembles context before each provider call and how that interacts with looping, retrieval, summaries, and checkpoints.

## Why context is explicit

The runtime does not treat context as just a growing chat log.

Instead, it builds a prepared turn from:

- system instructions
- the user goal
- recent messages
- summaries
- task state
- retrieved documents
- memory facts
- tool observations

This matches the design goals in `requirements.md`: context is assembled, inspectable, and configurable.

## The default pre-call pipeline

The default configured steps are:

1. `collect_recent_messages`
2. `summarize_history`
3. `inject_retrieval`
4. `inject_memory_facts`
5. `apply_token_budget`
6. `finalize_messages`

Those names come directly from the runtime config and registry.

## Basic configuration example

```python
from agent_manager import AgentSession, RuntimeConfig

config = RuntimeConfig.from_dict(
    {
        "provider": {"name": "ollama", "model": "llama3.1"},
        "profile": "coding-agent",
        "context": {
            "history_window": 10,
            "summary_trigger_messages": 12,
            "retrieval_top_k": 4,
            "max_memory_facts": 6,
            "pre_call_functions": [
                "collect_recent_messages",
                "summarize_history",
                "inject_retrieval",
                "apply_token_budget",
                "finalize_messages",
            ],
        },
    }
)

session = AgentSession(config=config)
```

## What each built-in context function does

### `collect_recent_messages`

- adds the core system and goal sections
- adds recent conversational history based on `history_window`

### `summarize_history`

- builds a summary section when the conversation grows
- helps compress old context before the next provider call

### `inject_retrieval`

- asks the active retriever for relevant chunks
- converts them into attributable context sections

### `inject_memory_facts`

- loads memory facts from the configured memory store

### `apply_token_budget`

- estimates token use
- drops lower-priority sections when the request is too large
- preserves higher-priority sections as much as possible

### `finalize_messages`

- converts the prepared sections into the actual message list sent to the provider

## Custom context functions

You can register your own pre-call functions without changing the core loop.

### Example: inject active file hints

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.context.functions import PreCallRuntime
from agent_manager.context.sections import PreparedTurn
from agent_manager.types import ContextSection, LoopState


async def inject_active_files(
    state: LoopState,
    prepared: PreparedTurn,
    config: RuntimeConfig,
    runtime: PreCallRuntime,
) -> PreparedTurn:
    del config, runtime
    active_files = state.metadata.get("active_files", [])
    if active_files:
        prepared.sections.append(
            ContextSection(
                key="active-files",
                title="Active Files",
                content="\\n".join(str(item) for item in active_files),
                priority=70,
                metadata={"source": "custom"},
            )
        )
    return prepared


session = AgentSession(
    config=RuntimeConfig.from_dict(
        {
            "profile": "coding-agent",
            "context": {
                "pre_call_functions": [
                    "collect_recent_messages",
                    "inject_active_files",
                    "apply_token_budget",
                    "finalize_messages",
                ]
            },
        }
    ),
    pre_call_functions={"inject_active_files": inject_active_files},
)
```

### Registering a custom function later

```python
session.register_pre_call_function("inject_active_files", inject_active_files)
```

## How looping works

At a high level, each loop iteration does the following:

1. load or resume loop state
2. prepare context through the pre-call pipeline
3. send the request to the provider
4. inspect whether the provider returned text, structured output, or tool calls
5. execute requested tools
6. append tool observations and assistant messages
7. update summaries, checkpoints, and loop metadata
8. stop or continue

## Stop conditions

The implementation supports the stop conditions described by the requirements:

- completed
- user interruption
- max steps reached
- repeated failure
- tool policy violation
- timeout
- resource exhaustion

## Resume and checkpoints

Checkpointed state includes the goal, messages, summaries, tool observations, step history, errors, and structured-output spec.

Example:

```python
result = session.run(
    "Build a phased implementation checklist for the provider layer.",
    task_id="provider-checklist",
)

later = session.resume("provider-checklist")
```

## Streaming

If you want to observe loop progress as it happens, use `stream_async()`.

```python
import asyncio

from agent_manager import AgentSession, RuntimeConfig


async def main() -> None:
    session = AgentSession(
        config=RuntimeConfig.from_dict(
            {"provider": {"name": "echo", "model": "echo-v1"}}
        )
    )
    async for event in session.stream_async("Summarize the runtime architecture."):
        print(event)


asyncio.run(main())
```

## When to customize the context pipeline

Customize it when you need:

- different behavior for research vs coding runs
- repo-specific context injection
- memory suppression for privacy or cost reasons
- tighter token control for smaller local models
- stronger prioritization for system instructions or retrieved facts

## Practical advice

- keep the pipeline short at first
- add retrieval only when you have useful indexed content
- add custom steps for domain-specific context, not for generic message passing
- prefer high-value sections over large raw dumps
- test with streaming on, so you can see where the loop spends time
