# Retrieval Tool Setup

This guide shows how to enable `retrieve_documents` and how to build a simple local retrieval pipeline.

## What retrieval gives you

Once a retriever is attached to `AgentSession`:

- the `retrieve_documents` tool is registered
- the context pipeline can inject retrieved chunks before each provider call
- the model can search project-specific knowledge instead of relying only on the immediate prompt

## Minimal in-memory retrieval setup

```python
from agent_manager import AgentSession, RuntimeConfig
from agent_manager.memory import HashEmbeddingProvider, InMemoryVectorRetriever, TextChunker

embedder = HashEmbeddingProvider(dimensions=64)
retriever = InMemoryVectorRetriever(embedder.embed)
chunker = TextChunker(chunk_size=300, overlap=40)

retriever.index_document(
    "requirements",
    """
    The runtime must support multi-provider adapters, tool execution,
    context management, retrieval, checkpoints, and structured output.
    """,
    metadata={"source": "requirements.md", "kind": "spec"},
    chunker=chunker,
)

config = RuntimeConfig.from_dict(
    {
        "provider": {"name": "echo", "model": "echo-v1"},
        "profile": "readonly",
    }
)

session = AgentSession(config=config, retriever=retriever)
result = session.run("What are the main runtime capabilities?")
print(result.output_text)
```

## Using the tool directly

```python
from agent_manager.tools.base import ToolContext
from agent_manager.types import ToolCallRequest

call = ToolCallRequest(
    id="retrieve-1",
    name="retrieve_documents",
    arguments={
        "query": "runtime capabilities",
        "top_k": 3,
        "metadata_filter": {"kind": "spec"},
    },
)

context = ToolContext(
    task_id="retrieval-demo",
    step_index=0,
    tool_call_id="retrieve-1",
    working_directory=str(session.working_directory),
)

result = session.tool_executor.execute(call, context)
print(result.output)
```

## How it works in the code

The current in-repo retrieval stack includes:

- `TextChunker` for overlapping document chunks
- `HashEmbeddingProvider` for dependency-free embeddings
- `InMemoryVectorRetriever` for vector-based retrieval
- `InMemoryKeywordRetriever` for lightweight keyword retrieval

These are good for local testing and implementation examples.

## Context-pipeline integration

The default context pipeline includes `inject_retrieval`.

That means retrieval can be used in two ways:

1. explicitly by the model through `retrieve_documents`
2. automatically during pre-call context assembly

You can tune retrieval behavior through config:

```python
from agent_manager import RuntimeConfig

config = RuntimeConfig.from_dict(
    {
        "context": {
            "retrieval_top_k": 4,
            "pre_call_functions": [
                "collect_recent_messages",
                "inject_retrieval",
                "apply_token_budget",
                "finalize_messages",
            ],
        }
    }
)
```

## Production-oriented retrieval plugins

If you do not want the built-in in-memory retriever, the project also provides plugin adapters for:

- LlamaIndex
- Chroma
- FAISS
- pgvector

Those plugins let you keep the same `retrieve_documents` behavior while swapping the backing store.

## Good inputs for retrieval

Good things to index:

- `requirements.md`
- design notes
- architecture docs
- style guides
- API schemas
- internal runbooks

Good metadata to attach:

- `source`
- `kind`
- `module`
- `version`

## When to use retrieval vs file tools

Prefer `retrieve_documents` when:

- the knowledge base is larger than a few files
- you want semantic lookup
- you want chunk-level grounding

Prefer `read_file` when:

- you need the exact live file contents
- you want the full text of a small file
- the model needs precise line-by-line edits
