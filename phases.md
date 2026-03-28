# phases.md - Implementation Phases for agent-manager

## Purpose

This document turns [requirements.md](/C:/Users/TheunisvanNiekerk/Code/agent-manager/requirements.md) into an implementation roadmap for `agent-manager`.

The intent is to:

- sequence the work into practical phases
- keep the architecture deployable as a reusable Python library
- plan for local and hosted model providers without rewriting the core
- define interfaces early so later phases fit into the same runtime

## Naming and Packaging

- Project/distribution name: `agent-manager`
- Python package name: `agent_manager`
- Target runtime: Python `3.11+`
- Primary outputs:
  - reusable Python library
  - headless local service
  - CLI entry point

## Cross-Phase Architecture Commitments

These decisions should be treated as stable from the beginning so later phases extend the project instead of reshaping it.

### 1. Provider-agnostic core

The runtime owns orchestration, tools, memory, context, and state. Providers only translate between the internal contract and external APIs.

### 2. Explicit context assembly

Context is not a raw message log. Every request should be built from named sections such as system instructions, task state, retrieved knowledge, recent observations, and summaries.

### 3. Normalized tool calling

Providers may differ in how they request tools, but the runtime should expose one internal `ToolCallRequest` contract and one `ToolResult` contract.

### 4. Checkpoint-first long-running loops

Every loop should be resumable. State must be serializable and saved before and after risky or expensive operations.

### 5. Optional integrations stay optional

Provider SDKs, retrieval backends, and third-party AI tooling should be loaded through extras or plugins so the base package stays lightweight.

## Requirements Coverage by Phase

| Requirement area | Phase |
| --- | --- |
| Reusable library structure | Phase 1 |
| Provider abstraction | Phase 2 |
| Tool system | Phase 3 |
| Context management and loop preparation | Phase 4 |
| Existing AI tooling as built-ins/plugins | Phase 5 |
| State, checkpointing, observability, safety | Starts in Phase 1 and expands through Phases 2-4 |

## Recommended Repository Shape

This is the structure to grow toward. Phase 1 should create the skeleton, even if many modules only contain interfaces and stubs.

```text
agent_manager/
  __init__.py
  version.py
  config.py
  types.py
  errors.py

  providers/
    __init__.py
    base.py
    factory.py
    openai_provider.py
    anthropic_provider.py
    gemini_provider.py
    ollama_provider.py
    lmstudio_provider.py

  tools/
    __init__.py
    base.py
    registry.py
    executor.py
    policies.py
    builtins/
      filesystem.py
      shell.py
      http.py
      retrieval.py

  context/
    __init__.py
    assembler.py
    budget.py
    sections.py
    summarizer.py
    pipeline.py

  memory/
    __init__.py
    base.py
    short_term.py
    long_term.py
    retrieval.py

  state/
    __init__.py
    models.py
    store.py
    checkpoint.py

  runtime/
    __init__.py
    loop.py
    session.py
    planner.py
    events.py

  plugins/
    __init__.py
    base.py
    registry.py

  api/
    __init__.py
    server.py
    schemas.py

  cli/
    __init__.py
    main.py

tests/
docs/
requirements.md
phases.md
pyproject.toml
README.md
```

## Phase 1 - Deployable Library Foundation

### Objective

Build `agent-manager` as a clean Python package that can be installed into other projects and used without committing to any single provider or tool backend.

### Why this phase matters

Your first phase should solve the packaging and library problem before model-specific work begins. If we skip this, later provider and tool work will get trapped inside an application-shaped codebase instead of a reusable runtime.

### Deliverables

- `pyproject.toml` with package metadata, optional dependency groups, and CLI entry point
- `agent_manager/` package with stable top-level interfaces
- basic config loading from environment plus file
- shared internal types for messages, tool calls, context sections, and loop state
- minimal runtime session and loop skeleton
- logging and error model
- test layout with smoke tests for imports and package creation
- README examples for library usage

### Implementation notes

- Keep `agent_manager/__init__.py` very small and export only public APIs.
- Separate the distribution name from the import name:
  - pip package: `agent-manager`
  - Python import: `agent_manager`
- Start with `pydantic` models or dataclasses for all state that crosses subsystem boundaries.
- Add optional extras early:
  - `openai`
  - `anthropic`
  - `gemini`
  - `retrieval`
  - `dev`

### Planned public API

The public API should be small at first and should not expose provider-specific details.

```python
from agent_manager.runtime.session import AgentSession
from agent_manager.config import RuntimeConfig
from agent_manager.providers.factory import build_provider
from agent_manager.tools.registry import ToolRegistry

config = RuntimeConfig.from_env()
provider = build_provider(config.provider)
tools = ToolRegistry()

session = AgentSession(config=config, provider=provider, tools=tools)
result = session.run("Summarize the repository and propose next actions.")
print(result.output_text)
```

### Foundation types to define now

These are worth creating in Phase 1 even if they are only partially used until later.

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None


class ToolCallRequest(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class ProviderResult(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    stop_reason: str | None = None
    usage: dict | None = None
    raw: object | None = None
    requested_context_keys: list[str] = Field(default_factory=list)


class ContextSection(BaseModel):
    key: str
    title: str
    content: str
    priority: int = 0
    token_estimate: int | None = None


class LoopState(BaseModel):
    task_id: str
    goal: str
    step_index: int = 0
    messages: list[Message] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)
    tool_observations: list[dict] = Field(default_factory=list)
    status: str = "running"
```

### Packaging snippet

This is an illustrative example for project shape. Exact SDK package names and versions should be confirmed when Phase 2 implementation begins.

```toml
[project]
name = "agent-manager"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "pydantic",
  "httpx",
  "typer",
  "tenacity",
  "orjson",
]

[project.optional-dependencies]
openai = ["openai"]
anthropic = ["anthropic"]
gemini = ["<gemini-sdk-package>"]
retrieval = ["chromadb", "sentence-transformers"]
dev = ["pytest", "pytest-asyncio", "respx", "ruff"]

[project.scripts]
agent-manager = "agent_manager.cli.main:app"
```

### Phase 1 exit criteria

- package installs locally with `pip install -e .`
- another Python project can import `agent_manager`
- runtime config, provider factory, and tool registry are present
- the repo has a stable internal module layout that future phases can extend

## Phase 2 - Multi-Provider Wrapper Layer

### Objective

Add a provider adapter layer for major platforms and local model runtimes so the same orchestrator can work with OpenAI, Anthropic, Gemini, LM Studio, and Ollama.

### Scope

- OpenAI adapter
- Anthropic adapter
- Gemini adapter
- Ollama adapter
- LM Studio adapter
- provider capability metadata
- normalized request and response translation
- model tool-call normalization
- streaming/event normalization

### Design rule

The runtime should never branch on provider behavior outside the adapter layer. Provider quirks stay inside `agent_manager/providers/`.

### Important design choice

Phase 2 should define a capability model because not all providers support the same features in the same way.

```python
from pydantic import BaseModel


class ProviderCapabilities(BaseModel):
    supports_tools: bool = False
    supports_streaming: bool = False
    supports_structured_output: bool = False
    supports_images: bool = False
    supports_system_messages: bool = True
```

### Base provider contract

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class ProviderRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[dict] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    metadata: dict = Field(default_factory=dict)


class BaseProvider(ABC):
    provider_name: str
    capabilities: ProviderCapabilities

    @abstractmethod
    async def generate(self, request: ProviderRequest) -> ProviderResult:
        raise NotImplementedError
```

### Tool and context hints

You mentioned that a model may need to tell us what tool to execute and possibly what context it needs next. That should be part of the normalized provider response so the loop can decide what to fetch before the next step.

```python
class ContextHint(BaseModel):
    key: str
    reason: str
    priority: int = 0


class ProviderResult(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    stop_reason: str | None = None
    usage: dict | None = None
    raw: object | None = None
    context_hints: list[ContextHint] = Field(default_factory=list)
```

### Adapter example

Each adapter should do three things:

1. translate the internal request into provider format
2. call the remote or local backend
3. translate the response back into `ProviderResult`

```python
class OpenAIProvider(BaseProvider):
    provider_name = "openai"
    capabilities = ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_structured_output=True,
    )

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        payload = self._to_openai_payload(request)
        response = await self._client.responses.create(**payload)
        return self._from_openai_response(response)
```

### Implementation sequencing inside Phase 2

Start with the providers that reduce future work:

1. OpenAI
2. LM Studio
3. Ollama
4. Anthropic
5. Gemini

This order is useful because OpenAI and LM Studio can often share message/tool schema ideas, while Ollama clarifies local-runtime differences early.

### Phase 2 exit criteria

- switching providers is config-only
- provider responses become the same internal `ProviderResult`
- tool requests from each provider normalize to `ToolCallRequest`
- local and cloud models can run through the same runtime session

## Phase 3 - Unified Tool System

### Objective

Create one tool contract that works for your own Python tools, built-in tools, and retrieval/RAG workflows.

### Scope

- base tool interface
- tool registry
- tool executor
- tool policies and runtime profiles
- observation/result normalization
- sync and async support
- built-in tool packages
- retrieval as a tool or tool-backed service

### Design rule

Providers do not execute tools. They only request them. The runtime validates, authorizes, executes, records, and returns tool results.

### Tool interfaces

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict
    output_schema: dict | None = None
    tags: list[str] = Field(default_factory=list)
    timeout_seconds: int = 60


class ToolContext(BaseModel):
    task_id: str
    step_index: int
    working_directory: str | None = None
    metadata: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    ok: bool
    output: dict | str
    error: str | None = None
    metadata: dict = Field(default_factory=dict)


class BaseTool(ABC):
    spec: ToolSpec

    @abstractmethod
    async def invoke(self, arguments: dict, context: ToolContext) -> ToolResult:
        raise NotImplementedError
```

### Registry and executor

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def definitions(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]
```

```python
class ToolExecutor:
    def __init__(self, registry: ToolRegistry, policy_engine) -> None:
        self.registry = registry
        self.policy_engine = policy_engine

    async def execute(self, call: ToolCallRequest, context: ToolContext) -> ToolResult:
        tool = self.registry.get(call.name)
        self.policy_engine.assert_allowed(tool.spec, context)
        return await tool.invoke(call.arguments, context)
```

### Retrieval and RAG in Phase 3

RAG should not be hard-coded into the loop. It should plug in through either:

- a retrieval service used by the context pipeline
- a first-class tool such as `retrieve_documents`

Example retrieval tool interface:

```python
class RetrieveDocumentsTool(BaseTool):
    spec = ToolSpec(
        name="retrieve_documents",
        description="Retrieve relevant knowledge chunks for the current task.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        tags=["retrieval", "rag"],
    )
```

### Policy planning

Phase 3 should also establish the profiles described in [requirements.md](/C:/Users/TheunisvanNiekerk/Code/agent-manager/requirements.md):

- `readonly`
- `local-dev`
- `coding-agent`
- `unrestricted-lab`

### Phase 3 exit criteria

- the model can request a normalized tool call
- the runtime validates and executes the requested tool
- tool results are added back into loop state as normalized observations
- custom Python tools and RAG-style tools share the same execution path

## Phase 4 - Context Management Between Loops

### Objective

Add a pre-agent-call context preparation pipeline so every loop step can rebuild the right context before the model runs.

### Why this matters

This is where the runtime becomes more than a chat wrapper. It begins to manage state, summaries, retrieval, and token budgeting deliberately between loop iterations.

### Scope

- pre-call context pipeline
- token budgeting
- context section prioritization
- summary generation
- message pruning
- retrieval injection
- working memory updates
- checkpoint-aware loop preparation

### Recommended design

Model the pre-call flow as an explicit pipeline with steps that can be enabled or disabled by config.

```python
class PreparedTurn(BaseModel):
    messages: list[Message]
    sections: list[ContextSection]
    token_estimate: int
    dropped_sections: list[str] = Field(default_factory=list)


class PreCallStep(ABC):
    @abstractmethod
    async def run(self, state: LoopState, config) -> LoopState:
        raise NotImplementedError


class ContextAssembler:
    def __init__(self, section_builders, token_counter, summarizer) -> None:
        self.section_builders = section_builders
        self.token_counter = token_counter
        self.summarizer = summarizer

    async def prepare(self, state: LoopState, query: str, config) -> PreparedTurn:
        sections = await self._build_sections(state, query, config)
        fitted_sections = await self._fit_to_budget(sections, config)
        messages = self._render_messages(fitted_sections)
        return PreparedTurn(
            messages=messages,
            sections=fitted_sections,
            token_estimate=self.token_counter.count_messages(messages),
        )
```

### Suggested pre-call pipeline

1. load checkpoint/state
2. refresh short-term memory view
3. fetch long-term memory or retrieval matches
4. summarize older history if over threshold
5. assemble context sections by priority
6. estimate tokens and trim to budget
7. persist "prepared" state before provider call

### Context section examples

- `system_instructions`
- `user_goal`
- `current_plan`
- `recent_messages`
- `tool_observations`
- `retrieved_knowledge`
- `memory_facts`
- `active_files`
- `execution_constraints`

### Planning for model-aware budgets

Different models have different context windows, and some providers count tokens differently. That means Phase 4 should depend on provider capability/config data introduced in Phase 2.

```python
class ModelProfile(BaseModel):
    provider: str
    model: str
    max_context_tokens: int
    max_output_tokens: int
    tool_call_overhead: int = 0
```

### Loop integration example

```python
async def run_step(self, state: LoopState) -> LoopState:
    prepared = await self.context_assembler.prepare(
        state=state,
        query=state.goal,
        config=self.config,
    )
    result = await self.provider.generate(
        ProviderRequest(
            model=self.config.provider.model,
            messages=prepared.messages,
            tools=self.tools.definitions(),
        )
    )
    return await self._handle_provider_result(state, result)
```

### Phase 4 exit criteria

- context is assembled by explicit pipeline steps
- summaries and retrieval can be injected before a provider call
- loop iterations can carry forward context in a controlled way
- the runtime can prune or compress state when token pressure increases

## Phase 5 - Built-In AI Tooling and Plugin Adapters

### Objective

Add optional integrations for existing AI-related tools so `agent-manager` can act as both a runtime and a composition layer.

### Scope

- plugin interface for third-party tools
- optional built-in adapters for common AI tooling ecosystems
- import/export bridges between `agent-manager` tools and external tool definitions
- optional retrieval backends and embedding providers
- optional model gateway integrations

### Design rule

These integrations should extend the platform, not define the platform. The core runtime must still work without them.

### Recommended plugin contract

```python
from abc import ABC, abstractmethod


class Plugin(ABC):
    name: str

    @abstractmethod
    def register(self, app) -> None:
        raise NotImplementedError
```

### Candidate built-in adapters

- MCP-compatible tool bridge
- LangChain-compatible tool adapter
- LlamaIndex retriever adapter
- vector store adapters for Chroma, FAISS, and pgvector
- HTTP/OpenAPI tool adapter for external services

### Example plugin registration

```python
class ChromaRetrievalPlugin(Plugin):
    name = "chroma-retrieval"

    def register(self, app) -> None:
        app.retrievers.register("chroma", ChromaRetriever)
        app.tools.register(RetrieveDocumentsTool())
```

### Guardrails for this phase

- keep plugins behind extras or entry points
- avoid coupling the core loop to any third-party framework
- prefer adapters around stable interfaces instead of deep framework inheritance
- require capability checks so missing dependencies fail clearly

### Phase 5 exit criteria

- third-party AI tools can be plugged in without changing runtime internals
- built-in integrations remain optional
- the runtime can expose external tools, retrievers, or providers through the same internal contracts

## Suggested Milestone Alignment

If you want to map this document back to implementation milestones, this is the cleanest sequence:

1. Phase 1 + initial state/logging skeleton
2. Phase 2 provider adapters
3. Phase 3 tool registry and executor
4. Phase 4 context/memory/checkpoint preparation
5. Phase 5 plugin ecosystem and built-in AI integrations

## Risks to Watch Early

### Risk 1: Provider-specific behavior leaking into the loop

Mitigation: keep adapter translation code isolated and normalize aggressively.

### Risk 2: Tool schema incompatibility across providers

Mitigation: define one internal schema and add provider-specific translators.

### Risk 3: Context management becoming an unstructured message pile

Mitigation: keep named context sections and token-budgeted assembly from the start.

### Risk 4: RAG getting embedded too deeply in the core

Mitigation: expose retrieval as a service or tool, not as mandatory runtime logic.

### Risk 5: Plugin complexity overwhelming the MVP

Mitigation: make Phase 5 optional and keep the core runtime complete before plugin expansion.

## Recommended Immediate Next Steps

The best implementation order from this document is:

1. create the Phase 1 package skeleton and `pyproject.toml`
2. add the shared types and base interfaces from this roadmap
3. implement one minimal runtime session and provider factory
4. then begin the first provider adapter in Phase 2

## Summary

This roadmap keeps `agent-manager` aligned with [requirements.md](/C:/Users/TheunisvanNiekerk/Code/agent-manager/requirements.md) while incorporating your requested phases:

- Phase 1 establishes a deployable library foundation
- Phase 2 adds unified model wrappers
- Phase 3 adds a unified tool execution layer
- Phase 4 adds explicit context management between loops
- Phase 5 adds optional built-in AI tool integrations

The most important design choice is to lock in the internal contracts early so providers, tools, memory, and plugins all extend the same runtime instead of competing with it.
