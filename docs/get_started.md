# Get Started

This guide shows how to use `agent-manager` as a Python library, CLI, and headless runtime. It also explains how the common agent patterns from `requirements.md` map to the current codebase.

## The mental model

`agent-manager` does not create separate subclasses like `CodingAgent` or `ResearchAgent`.

Instead, you build an agent by combining:

- a provider configuration
- a runtime profile
- built-in and custom tools
- retrieval and memory components
- a context pipeline
- prompts and system instructions

The main entry point is `AgentSession`.

## Install

Basic install:

```bash
pip install -e .
```

Useful optional extras:

```bash
pip install -e .[anthropic]
pip install -e .[openai]
pip install -e .[local-providers]
pip install -e .[tinydb]
pip install -e .[langchain]
pip install -e .[llamaindex]
```

## Smallest working example

This uses the built-in `echo` provider, so it works without an API key.

```python
from agent_manager import AgentSession, RuntimeConfig

config = RuntimeConfig.from_dict(
    {
        "provider": {"name": "echo", "model": "echo-v1"},
        "profile": "readonly",
    }
)

session = AgentSession(config=config)
result = session.run("Say hello from the agent runtime.")
print(result.output_text)
```

## CLI quick start

```bash
agent-manager "Inspect the project and suggest the next task."
```

Override the configured provider or model:

```bash
agent-manager --provider ollama --model llama3.1 "Summarize the repository."
```

Stream runtime events:

```bash
agent-manager --stream "Walk the repo and explain the main modules."
```

Return a JSON result:

```bash
agent-manager --json "Explain what this runtime does."
```

## Agent patterns

The requirements define four main use cases. Here is how to build each one with the current implementation.

### 1. Coding agent

Best fit:

- `profile="coding-agent"` or `profile="local-dev"`
- built-in file and shell tools
- optional retrieval for architecture notes, docs, or code standards
- any supported backend, including local sources like Ollama, LM Studio, and vLLM

```python
from agent_manager import AgentSession, RuntimeConfig

config = RuntimeConfig.from_dict(
    {
        "provider": {"name": "ollama", "model": "llama3.1"},
        "profile": "coding-agent",
        "system_prompt": "You are a careful coding assistant. Read before you write.",
    }
)

session = AgentSession(config=config)
result = session.run("Inspect the repository and propose a safe refactor.")
print(result.output_text)
```

Local vLLM variant:

```python
config = RuntimeConfig.from_dict(
    {
        "provider": {
            "name": "vllm",
            "model": "NousResearch/Meta-Llama-3-8B-Instruct",
            "base_url": "http://localhost:8000/v1",
        },
        "profile": "coding-agent",
    }
)
```

Simple prompt:

```text
Inspect the codebase, identify the best place to add a TinyDB-backed note store, and explain the change before editing.
```

### 2. Research agent

Best fit:

- `profile="coding-agent"` with write and shell access denied
- built-in `web_search` and `http_request`
- optional retrieval for local notes

```python
from agent_manager import AgentSession, RuntimeConfig

config = RuntimeConfig.from_dict(
    {
        "provider": {"name": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
        "profile": "coding-agent",
        "tool_policy": {
            "denied_permissions": ["filesystem:write", "process:execute"],
        },
        "system_prompt": "You are a research agent. Prefer cited, tool-backed answers.",
    }
)

session = AgentSession(config=config)
result = session.run("Research TinyDB usage patterns and summarize the tradeoffs.")
print(result.output_text)
```

Simple prompt:

```text
Search for recent TinyDB examples, compare them to our plugin design, and return a short list of recommendations with links.
```

### 3. Document or knowledge agent

Best fit:

- a retriever attached to the session
- `retrieve_documents` enabled automatically
- context pipeline steps for retrieval injection

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.memory import HashEmbeddingProvider, InMemoryVectorRetriever, TextChunker

embedder = HashEmbeddingProvider(dimensions=64)
retriever = InMemoryVectorRetriever(embedder.embed)
retriever.index_document(
    "requirements",
    "Agent-manager should support multi-provider orchestration, tools, memory, and checkpoints.",
    metadata={"source": "requirements.md"},
    chunker=TextChunker(chunk_size=120, overlap=20),
)

config = RuntimeConfig.from_dict(
    {
        "provider": {"name": "echo", "model": "echo-v1"},
        "profile": "readonly",
    }
)

session = AgentSession(config=config, retriever=retriever)
result = session.run("What does this system need to support?")
print(result.output_text)
```

Simple prompt:

```text
Answer using retrieved project knowledge first, then explain what is still missing.
```

### 4. Automation agent

Best fit:

- checkpoints enabled
- resumable task ids
- structured output if you want machine-readable state
- optional streaming for monitoring

```python
from agent_manager import AgentSession, RuntimeConfig

config = RuntimeConfig.from_dict(
    {
        "provider": {"name": "anthropic", "model": "claude-3-5-sonnet-latest", "api_key_env": "ANTHROPIC_API_KEY"},
        "profile": "coding-agent",
        "runtime": {
            "max_steps": 10,
            "timeout_seconds": 600,
        },
    }
)

session = AgentSession(config=config)
result = session.run(
    "Create an implementation plan for improving the provider adapters.",
    task_id="provider-improvement-plan",
)
print(result.output_text)
```

Resume later:

```python
resumed = session.resume("provider-improvement-plan")
print(resumed.output_text)
```

Simple prompt:

```text
Break this task into steps, use tools when needed, checkpoint progress, and stop when the result is ready for human review.
```

## Structured output example

`AgentSession.run()` and `run_async()` accept a structured schema.

```python
from agent_manager import AgentSession, RuntimeConfig

session = AgentSession(
    config=RuntimeConfig.from_dict({"provider": {"name": "echo", "model": "echo-v1"}})
)

result = session.run(
    "Return a project summary.",
    structured_output={
        "type": "json_schema",
        "name": "project_summary",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "next_step": {"type": "string"},
            },
            "required": ["summary", "next_step"],
        },
        "strict": True,
    },
)

print(result.structured_output)
```

## Headless service example

You can also wrap a session in `AgentService`.

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.api.schemas import RunRequest
from agent_manager.api.server import AgentService

session = AgentSession(
    config=RuntimeConfig.from_dict({"provider": {"name": "echo", "model": "echo-v1"}})
)
service = AgentService(session)

response = service.run(RunRequest(prompt="Summarize the runtime architecture."))
print(response.output_text)
```

## Next docs

- [Tools Overview](./tools.md)
- [Context Manager](./context-manager.md)
- [Providers and Connectivity](./providers-and-connectivity.md)
- [Configuration](./configuration.md)
