# Requirements.md — Local-First Python Agent Runtime

## 1. Purpose

Build a local-first Python **agent runtime / orchestrator** that can:

- run against multiple model providers
- manage context and memory
- define and execute tools
- support long-running agent loops
- remain portable across local and hosted LLM backends

The system should treat the model as a pluggable reasoning engine and keep orchestration, tool execution, memory, and state management under application control.

---

## 2. Goals

The runtime must:

- support **multiple providers** through a shared interface
- work with **local models** and **cloud APIs**
- allow developers to define tools in Python
- support **structured tool calling**
- manage **short-term context**, **summaries**, and **retrieval**
- support **long-running tasks** through iterative loops
- maintain **clear separation** between:
  - model providers
  - orchestration logic
  - tools
  - memory
  - storage
  - transport / API surface
- allow provider switching via configuration rather than code changes
- be usable both as:
  - a Python library
  - a headless local service

---

## 3. Non-Goals

The first version should **not** try to be:

- a UI product
- a hosted multi-tenant platform
- a no-code builder
- a distributed workflow engine
- a benchmark suite
- a fully autonomous unsafe system with unrestricted execution

These may be added later, but they are out of scope for the initial framework.

---

## 4. Core Use Cases

### 4.1 Coding Agent
- inspect files
- plan implementation steps
- write code
- run tests
- read errors
- retry with updated context

### 4.2 Research Agent
- search the web through approved tools
- summarize findings
- cite sources
- maintain working notes and task state

### 4.3 Document / Knowledge Agent
- load documents
- chunk and index content
- retrieve relevant passages
- answer questions with source grounding

### 4.4 Automation Agent
- accept a long-running goal
- break it into steps
- call tools repeatedly
- checkpoint progress
- recover from interruption

---

## 5. High-Level Architecture

```text
User / API / CLI
    ↓
Agent Runtime / Orchestrator
    ├── Context Manager
    ├── Planner / Loop Controller
    ├── Tool Registry + Tool Executor
    ├── Memory Manager
    ├── Retrieval Layer
    ├── State Store / Checkpoints
    └── Provider Adapter Layer
            ├── OpenAI Adapter
            ├── Anthropic Adapter
            ├── Ollama Adapter
            └── LM Studio Adapter
```

---

## 6. Functional Requirements

## 6.1 Provider Abstraction

The framework must provide a provider interface that normalizes:

- chat completion requests
- system / user / assistant messages
- tool definitions
- tool call responses
- structured output requests
- streaming tokens or events
- usage metadata
- errors and retryable failures

### Required providers
- OpenAI
- Anthropic
- Ollama
- LM Studio

### Requirements
- provider selection must be config-driven
- providers must be swappable without changing orchestrator logic
- provider responses must be converted into a normalized internal format
- provider-specific quirks must be isolated inside adapter modules

### Example normalized response model

```python
class NormalizedResult:
    text: str | None
    tool_calls: list
    stop_reason: str | None
    usage: dict | None
    raw: object | None
```

---

## 6.2 Agent Loop

The runtime must support an iterative control loop:

1. load state
2. build context
3. call model
4. detect normal response vs tool request
5. execute tool if requested
6. append observations
7. update memory / summaries / checkpoints
8. repeat until completion or stop condition

### Requirements
- maximum step count must be configurable
- loop timeout must be configurable
- stop conditions must include:
  - completed
  - user interruption
  - max steps reached
  - repeated failure
  - tool policy violation
- the loop must support both synchronous and async execution patterns

---

## 6.3 Tool System

The framework must allow Python-defined tools with metadata.

### A tool must include:
- name
- description
- input schema
- callable implementation
- optional output schema
- permissions / policy tags
- timeout
- retry policy

### Tool categories
- file read / write
- shell / process execution
- web search
- HTTP requests
- database access
- memory retrieval
- embeddings / vector search
- custom application tools

### Requirements
- tools must be registrable dynamically
- tools must be invokable by normalized tool-call events
- tool execution results must be marshaled into a normalized observation format
- unsafe tools must be disableable by policy
- tool access should be scoped by runtime profile

### Example normalized tool interface

```python
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    output_schema: dict | None = None
```

---

## 6.4 Context Management

The framework must treat context as an explicitly assembled object, not an implicit chat log.

### Context sources
- system instructions
- user request
- recent conversation history
- summaries of earlier history
- task state
- retrieved documents
- tool results
- working memory
- relevant files / code snippets

### Requirements
- context assembly must be modular and inspectable
- each context section should be attributable and optionally logged
- pre-call context preparation must support a configurable pipeline of named functions/steps
- the runtime must include standard context-distillation functions such as recent-history selection, summary generation, retrieval injection, and token-budget fitting
- applications must be able to register custom pre-call context functions without modifying the core loop
- the active pre-call functions should be selectable by configuration so different runtime profiles can use different context preparation behavior
- the runtime must support:
  - sliding windows
  - summarization
  - message pruning
  - relevance-based retrieval
  - token budgeting
- context builders must be model-aware because token limits vary by provider / model

### Required capabilities
- estimate token usage before request
- truncate or compress when needed
- preserve high-priority instructions under pressure
- keep tool results separate from user-visible output
- allow custom context-distillation hooks to add, remove, transform, or reorder context sections before the provider call

### Example pre-call pipeline contract

```python
class PreCallContextStep:
    name: str

    async def run(self, state, prepared_context, config):
        ...
```

---

## 6.5 Memory

Memory should be separated into layers.

### Short-term memory
- current task state
- recent steps
- recent observations
- active files and artifacts

### Long-term memory
- user preferences
- stable facts
- prior task summaries
- reusable learned patterns

### Requirements
- short-term memory should be stored in structured state
- long-term memory should be queryable independently of the chat history
- memory writes should be explicit, not automatic by default
- memory entries should support metadata:
  - source
  - timestamp
  - confidence
  - scope
  - tags

---

## 6.6 Retrieval

The framework should support optional retrieval-augmented generation.

### Requirements
- support document chunking
- support embedding and indexing
- support top-k retrieval
- support metadata filtering
- retrieved chunks must be tracked and attributable
- retrieval must be optional and replaceable

### Candidate backends
- FAISS
- Chroma
- pgvector
- local SQLite-backed metadata store

---

## 6.7 State and Checkpointing

The runtime must preserve enough state to resume long-running jobs.

### State must include:
- task id
- current goal
- step history
- summaries
- tool outputs
- pending subgoals
- errors
- checkpoint timestamps

### Requirements
- checkpoint persistence must survive process restarts
- resumable runs must reconstruct the loop state
- checkpoint storage should default to SQLite or JSON files
- state must be serializable

---

## 6.8 Observability

The framework must provide strong introspection.

### Required logging / tracing
- provider request / response metadata
- token estimates and actual usage when available
- tool call requests and results
- timing
- retry attempts
- summarization events
- retrieval selections
- checkpoint saves / loads

### Requirements
- logs should support structured JSON output
- developer mode should allow verbose trace inspection
- secrets must be redacted
- sensitive tool outputs must be maskable

---

## 6.9 Safety and Policy Controls

The runtime must include a policy layer for tool usage and execution boundaries.

### Requirements
- tool allowlist / denylist
- optional approval hooks before dangerous actions
- filesystem scope restrictions
- network access restrictions
- shell execution restrictions
- maximum subprocess duration
- configurable safe mode profiles

### Example runtime profiles
- readonly
- local-dev
- coding-agent
- unrestricted-lab

---

## 6.10 Output Modes

The framework should support multiple output modes:

- plain text response
- structured JSON response
- tool call
- streamed tokens
- event stream for UI / CLI

### Requirements
- the orchestrator must distinguish:
  - internal reasoning artifacts
  - tool requests
  - tool results
  - final user-facing output
- final output formatting should be handled separately from loop internals

---

## 6.11 Configuration

The system must be configurable by file and environment variables.

### Suggested formats
- YAML
- TOML
- JSON
- `.env`

### Configurable values
- provider and model
- token limits
- timeouts
- tool policies
- logging level
- checkpoint storage
- retrieval backend
- summarization thresholds

---

## 7. Non-Functional Requirements

## 7.1 Extensibility
- adding a new provider should require only a new adapter
- adding a new tool should not require loop changes
- adding a new memory backend should not affect tool execution logic

## 7.2 Portability
- must run locally on macOS, Linux, and Windows
- Python version target: **3.11+**

## 7.3 Performance
- requests should avoid unnecessary full-history replay
- context building should be incremental where practical
- expensive summarization should be triggered by policy thresholds

## 7.4 Reliability
- transient provider failures should be retryable
- tool crashes should be isolated and reported cleanly
- checkpoints should reduce loss during interruption

## 7.5 Testability
- provider adapters must be mockable
- tool execution must be unit-testable
- state transitions must be testable without live model calls

---

## 8. Suggested Project Structure

```text
agent_runtime/
  __init__.py
  config.py
  types.py
  events.py

  providers/
    base.py
    openai_provider.py
    anthropic_provider.py
    ollama_provider.py
    lmstudio_provider.py

  tools/
    base.py
    registry.py
    executor.py
    builtins/
      filesystem.py
      shell.py
      web.py
      http.py

  context/
    builder.py
    budget.py
    summarizer.py
    selectors.py

  memory/
    base.py
    short_term.py
    long_term.py
    retrieval.py

  state/
    models.py
    store.py
    checkpoint.py

  runtime/
    loop.py
    planner.py
    policies.py
    session.py

  api/
    server.py
    schemas.py

  cli/
    main.py

tests/
docs/
requirements.md
pyproject.toml
README.md
```

---

## 9. Recommended Internal Interfaces

## 9.1 Provider Interface

```python
class BaseProvider:
    def generate(self, messages, tools=None, **kwargs):
        raise NotImplementedError
```

## 9.2 Tool Interface

```python
class BaseTool:
    name: str
    description: str

    def invoke(self, arguments: dict) -> dict | str:
        raise NotImplementedError
```

## 9.3 State Store Interface

```python
class StateStore:
    def load(self, task_id: str): ...
    def save(self, task_id: str, state): ...
```

## 9.4 Context Builder Interface

```python
class ContextBuilder:
    def build(self, state, query, retrieved_docs=None) -> list[dict]:
        ...
```

---

## 10. MVP Scope

The first release should include:

- provider adapters:
  - OpenAI
  - Ollama
  - LM Studio
- basic Anthropic adapter if time permits
- one iterative agent loop
- tool registry
- filesystem tools
- shell execution tool
- simple web search tool abstraction
- token-aware context builder
- SQLite checkpoint store
- JSON structured logs
- CLI interface
- minimal Python API

### Explicitly defer
- multi-agent coordination
- GUI
- distributed execution
- advanced planning trees
- human approval UI
- hosted deployment control plane

---

## 11. Suggested Dependencies

### Core
- `pydantic`
- `httpx`
- `tenacity`
- `typer`
- `rich`
- `sqlalchemy` or `sqlite-utils`
- `orjson`
- `tiktoken` or provider-specific token counters where available

### Optional provider SDKs
- `openai`
- `anthropic`

### Optional retrieval
- `faiss-cpu`
- `chromadb`
- `sentence-transformers`

### Testing
- `pytest`
- `pytest-asyncio`
- `respx`

---

## 12. Acceptance Criteria

The framework is acceptable when it can:

1. run the same task against Ollama and LM Studio by changing only config
2. run at least one cloud provider through the same orchestrator contract
3. define Python tools and expose them to a model
4. detect and execute tool calls
5. maintain short-term state across multiple loop iterations
6. summarize older interaction history when token budget is exceeded
7. checkpoint and resume a task after interruption
8. log provider calls, tool calls, and task transitions
9. prevent restricted tools from running under a safe profile
10. expose both a Python API and a CLI entry point

---

## 13. Suggested Milestones

### Milestone 1: Provider Foundation
- normalized provider interface
- OpenAI-compatible adapter
- Ollama adapter
- LM Studio adapter

### Milestone 2: Core Runtime
- task state model
- loop controller
- tool detection
- tool execution

### Milestone 3: Context + Memory
- context builder
- token budgeting
- summarization
- checkpointing

### Milestone 4: Tooling + Safety
- filesystem tool
- shell tool
- policy controls
- runtime profiles

### Milestone 5: Retrieval + Docs
- embedding pipeline
- local vector search
- source attribution

### Milestone 6: Developer Experience
- CLI
- config loading
- structured logs
- tests
- examples

---

## 14. Design Principles

- **Provider-agnostic core**
- **Explicit context assembly**
- **Structured state over raw chat logs**
- **Tools as first-class capabilities**
- **Safety by policy, not by assumption**
- **Resumable long-running execution**
- **Observable by default**
- **Simple MVP, extensible internals**

---

## 15. Open Questions

These should be decided before implementation hardens:

- Will tool calls be model-native, prompt-protocol-based, or both?
- Will shell execution be built-in or plugin-only?
- Will retrieval be mandatory for coding agents?
- What is the default state store: SQLite, JSON, or both?
- Will the runtime expose a REST API, gRPC API, or CLI only in v1?
- How much provider-specific behavior should be surfaced vs hidden?

---

## 16. Summary

This project is a **local-first, multi-provider Python agent runtime** with:

- pluggable model backends
- explicit context management
- tool orchestration
- memory and retrieval
- long-running iterative execution
- resumable state
- safety controls
- developer-friendly interfaces

The system should prioritize clarity, portability, and control over excessive abstraction.
