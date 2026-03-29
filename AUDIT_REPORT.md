# Agent Manager — Requirements Compliance Audit Report

**Date:** March 29, 2026
**Scope:** Full code audit against requirements.md
**Test Results:** 47 passed, 28 failed (all failures caused by SQLite on OneDrive-mounted filesystem)

---

## Executive Summary

The agent_manager package implements a substantial portion of the requirements.md specification. The project structure closely follows the suggested layout (Section 8), all four required provider adapters exist, the agent loop is functional, tools are registrable and executable, and context assembly with a pre-call pipeline is in place. However, several areas remain incomplete or have bugs that would prevent production use. The most significant gaps are: no native streaming, a stub summarizer, no persistent long-term memory, an incomplete observability layer, and zero declared core dependencies in pyproject.toml.

**Overall Grade: B-** — Architecturally sound, core loop works, but critical subsystems are stubs or incomplete.

---

## Test Suite Status

- **47 tests pass** — covering providers, tool execution, context pipeline, retrieval, indexing, embeddings, export bridges, and plugin adapters at the unit level.
- **28 tests fail** — all due to `sqlite3.OperationalError: disk I/O error` because the project lives on a OneDrive-synced directory. SQLite requires a POSIX-compatible filesystem with file locking; cloud-synced folders break this. **These are environment failures, not code bugs**, but the code should fall back to JSON state store more gracefully.

---

## Section-by-Section Compliance

### 6.1 Provider Abstraction — MOSTLY COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| Normalized response model (text, tool_calls, stop_reason, usage, raw) | PASS | ProviderResult in types.py covers all fields plus extras |
| OpenAI adapter | PASS | Full message/tool translation and response normalization |
| Anthropic adapter | PASS | Content-block parsing, tool_use detection |
| Ollama adapter | PASS | Message format and done_reason normalization |
| LM Studio adapter | PASS | OpenAI-compatible shape reuse |
| Config-driven provider selection | PASS | Factory pattern with registry in factory.py |
| Swappable without orchestrator changes | PASS | Abstract BaseProvider interface |
| Provider-specific quirks isolated | PASS | Each adapter normalizes independently |
| Out-of-resource detection | PASS | ProviderResourceExhaustedError with kind/metadata/retry_after |
| Structured failure exposure | PASS | Distinct exception type, not retried, to_dict() serialization |
| Streaming support | FAIL | All providers set supports_streaming=False; only a base-class fake-stream fallback exists |
| Sync + async API | PARTIAL | Only async generate(); no sync wrapper on providers |

**Bugs/Issues:**
- `ProviderRequest.stream` field is accepted but never checked — dead code.
- `_parse_retry_after()` extracts the header but the retry loop ignores it — always uses fixed backoff.
- `maybe_parse_structured_output()` is duplicated in every provider instead of being centralized.

### 6.2 Agent Loop — COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| Iterative loop (load → context → model → detect → execute → observe → update → repeat) | PASS | loop.py implements full cycle |
| Configurable max_steps | PASS | RuntimeLimits.max_steps checked each iteration |
| Configurable timeout | PASS | Deadline tracking with asyncio |
| Stop: completed | PASS | |
| Stop: user interruption | PASS | request_interrupt() sets flag |
| Stop: max steps reached | PASS | |
| Stop: repeated failure | PASS | max_consecutive_failures tracked |
| Stop: tool policy violation | PASS | PolicyViolationError caught and surfaces |
| Sync and async execution | PASS | run() and run_async() both exist; run() wraps async |

**Issues:**
- No per-tool retry limit — a consistently failing tool retries indefinitely until max_consecutive_failures.
- No tool-call deduplication if the model returns the same call twice.
- Resume validation is minimal — can resume after policy_violation, which may not be intended.

### 6.3 Tool System — COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| ToolSpec with name, description, input_schema | PASS | Also has output_schema, tags, permissions, timeout, retry |
| Dynamic registration | PASS | ToolRegistry.register(), register_callable() |
| Normalized observation format | PASS | ToolResult with ok, output, error, metadata |
| Unsafe tools disableable by policy | PASS | ToolPolicyEngine with deny lists |
| Tool access scoped by runtime profile | PASS | DEFAULT_PROFILES: readonly, local-dev, coding-agent, unrestricted-lab |
| Approval hooks | PASS | ApprovalHook callback pattern |
| Filesystem tools | PASS | Read, write, list with path scoping |
| Shell tool | PASS | Async subprocess with timeout |
| Web search tool | PASS | DuckDuckGo searcher abstraction |
| HTTP tool | PASS | GET/POST with response truncation |
| Retrieval tool | PASS | Wraps BaseRetriever |

**Issues:**
- Shell tool has no command filtering — accepts any command including `rm -rf`.
- Filesystem tools don't prevent symlink escapes from allowed roots.
- HTTP tool returns status=0 for client-side network errors, which is confusing.
- Executor retry backoff is linear (1s, 2s, 3s), not exponential as the field name `retry_backoff_seconds` suggests.

### 6.4 Context Management — MOSTLY COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| Modular, inspectable assembly | PASS | ContextAssembler with named sections |
| Pre-call pipeline of named steps | PASS | PreCallPipeline with registry |
| Standard distillation functions | PASS | 6 built-in functions in functions.py |
| Custom pre-call functions registrable | PASS | register() on pipeline, no core loop changes needed |
| Configurable per profile | PASS | config.context.pre_call_functions selects active steps |
| Sliding windows | PARTIAL | Simple tail slice, not true sliding window |
| Summarization | FAIL | SimpleSummarizer just concatenates last 4 messages — not real summarization |
| Message pruning | PARTIAL | Token budget drops low-priority sections |
| Relevance-based retrieval | PASS | Retrieval injection with top-k |
| Token budgeting | PASS | ModelBudgetProfile with known model limits |
| Model-aware token limits | PASS | KNOWN_MODEL_BUDGETS covers major models |
| Token estimation before request | PASS | SimpleTokenCounter (chars/4 heuristic) |
| Preserve high-priority under pressure | PARTIAL | Priority sorting exists but system instructions have no guaranteed floor |

**Critical Issue:** SimpleSummarizer (summarizer.py) is a stub — it concatenates messages instead of producing actual summaries. This means acceptance criterion #6 ("summarize older history when token budget exceeded") is not truly met.

### 6.5 Memory — PARTIALLY COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| Short-term: current task, recent steps, observations | PASS | ShortTermMemory in-memory store |
| Long-term: preferences, facts, summaries, patterns | FAIL | InMemoryLongTermStore only — no persistence |
| Short-term in structured state | PASS | |
| Long-term queryable independently | PARTIAL | Substring matching only, no metadata filters |
| Explicit memory writes | PASS | Manual add() calls |
| Metadata: source, timestamp, confidence, scope, tags | PASS | MemoryEntry has all fields |

**Critical Issue:** Long-term memory has no persistent storage backend. InMemoryLongTermStore loses all data on restart. No SQLite, JSON, or vector DB implementation exists for long-term memory.

### 6.6 Retrieval — MOSTLY COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| Document chunking | PASS | TextChunker with configurable size/overlap |
| Embedding and indexing | PASS | HashEmbeddingProvider + InMemoryVectorRetriever |
| Top-k retrieval | PASS | |
| Metadata filtering | PASS | |
| Retrieved chunks tracked/attributable | PASS | DocumentChunk with metadata |
| Retrieval optional and replaceable | PASS | Abstract BaseRetriever interface |
| FAISS backend | PASS | Plugin adapter |
| Chroma backend | PASS | Plugin adapter |
| pgvector backend | PASS | Plugin adapter |

**Issue:** All in-memory retrievers lose data on restart. No built-in persistent retrieval store (the plugin adapters for Chroma/FAISS/pgvector delegate to external libraries that handle persistence).

### 6.7 State and Checkpointing — COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| State includes task_id, goal, step_history, summaries, tool_outputs, subgoals, errors, timestamps | PASS | LoopState model |
| Checkpoint survives process restart | PASS | SqliteStateStore and JsonFileStateStore |
| Resumable runs reconstruct state | PASS | resume_async() loads and continues |
| SQLite or JSON default | PASS | Both implemented, SQLite is default |
| Serializable state | PASS | to_dict()/from_dict() on LoopState |

**Issue:** JsonFileStateStore doesn't use atomic writes (temp file + rename). Crash during write could corrupt state. SQLite doesn't work on cloud-synced filesystems.

### 6.8 Observability — INCOMPLETE (Skeleton Only)

| Requirement | Status | Notes |
|---|---|---|
| Provider request/response metadata | FAIL | Not logged by observability module |
| Token estimates and actual usage | FAIL | Not tracked centrally |
| Tool call requests and results | FAIL | Not logged |
| Timing | FAIL | No duration tracking |
| Retry attempts | FAIL | Not logged |
| Summarization events | FAIL | Not logged |
| Retrieval selections | FAIL | Not logged |
| Checkpoint saves/loads | FAIL | Not logged |
| Structured JSON output | PASS | JsonLogFormatter exists |
| Developer mode verbose trace | FAIL | Not implemented |
| Secret redaction | FAIL | Not implemented |
| Sensitive output masking | FAIL | Not implemented |

**Critical Issue:** observability.py provides only the JSON log formatter infrastructure. None of the required event types are actually emitted. The runtime loop emits RuntimeEvents, but these aren't integrated with the observability module.

### 6.9 Safety and Policy Controls — COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| Tool allowlist/denylist | PASS | ToolPolicyProfile |
| Approval hooks | PASS | ApprovalHook callback |
| Filesystem scope restrictions | PASS | resolve_scoped_path() |
| Network access restrictions | PARTIAL | Policy tags exist but no network-specific enforcement |
| Shell execution restrictions | PARTIAL | Policy can block shell tool but no command-level filtering |
| Maximum subprocess duration | PASS | Timeout enforcement |
| Configurable safe mode profiles | PASS | readonly, local-dev, coding-agent, unrestricted-lab |

### 6.10 Output Modes — MOSTLY COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| Plain text response | PASS | |
| Structured JSON response | PASS | structured_output support |
| Tool call | PASS | |
| Streamed tokens | PARTIAL | Base-class fake streaming only |
| Event stream for UI/CLI | PASS | stream_async() with RuntimeEvent |

### 6.11 Configuration — MOSTLY COMPLETE

| Requirement | Status | Notes |
|---|---|---|
| YAML | FAIL | Not supported (only TOML and JSON) |
| TOML | PASS | Via tomllib/tomli |
| JSON | PASS | |
| .env | PASS | AGENT_MANAGER_ prefix environment variables |
| Provider and model | PASS | |
| Token limits | PASS | |
| Timeouts | PASS | |
| Tool policies | FAIL | Not configurable via config file |
| Logging level | PASS | |
| Checkpoint storage | PASS | state_backend: sqlite/json |
| Retrieval backend | FAIL | Not configurable |
| Summarization thresholds | PARTIAL | Only summary_trigger_messages |

---

## Acceptance Criteria (Section 12) — Status

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Same task against Ollama and LM Studio by changing config | PASS | Both adapters exist, factory swaps them |
| 2 | At least one cloud provider through same contract | PASS | OpenAI and Anthropic adapters |
| 3 | Define Python tools and expose to model | PASS | ToolRegistry + provider_definitions() |
| 4 | Detect and execute tool calls | PASS | Loop detects tool_calls, executor runs them |
| 5 | Short-term state across loop iterations | PASS | LoopState persists across steps |
| 6 | Summarize older history when token budget exceeded | FAIL | Summarizer is a stub (concatenation only) |
| 7 | Checkpoint and resume after interruption | PASS | SqliteStateStore + resume_async() |
| 8 | Log provider calls, tool calls, task transitions | PARTIAL | RuntimeEvents emitted but not structured logging |
| 9 | Prevent restricted tools under safe profile | PASS | Policy engine blocks denied tools |
| 10 | Python API and CLI entry point | PASS | AgentSession + agent-manager CLI |

---

## pyproject.toml Issues

**Critical: `dependencies = []`** — The package declares zero core dependencies. While the code uses mostly stdlib (json, sqlite3, urllib, asyncio, dataclasses), it requires `tomli` as a fallback for Python < 3.11 TOML support, and tests need `pytest`. The suggested dependencies from requirements.md (pydantic, httpx, tenacity, typer, rich, sqlalchemy, orjson, tiktoken) are not used — the code chose stdlib alternatives instead. This is a valid architectural choice but means:

- No request validation (no pydantic)
- No HTTP/2 or connection pooling (urllib instead of httpx)
- No structured retry library (custom retry loops instead of tenacity)
- No rich CLI output (argparse instead of typer/rich)
- No fast JSON (json instead of orjson)
- No accurate token counting (char heuristic instead of tiktoken)

The `[project.optional-dependencies]` section is also incomplete: `retrieval = []` is empty, `dev` is missing `pytest-asyncio`, and `all` doesn't include `mcp`, `langchain-core`, or `llama-index-core`.

---

## Top Priority Fixes

### P0 — Blocking Issues

1. **Implement real summarization** — Replace SimpleSummarizer stub with LLM-backed or extractive summarization. This is acceptance criterion #6.
2. **Add persistent long-term memory** — InMemoryLongTermStore needs a SQLite or JSON-file backend.
3. **Declare core dependencies** — At minimum add `tomli` (for <3.11 compat) to dependencies. Consider adding tiktoken for accurate token counting.
4. **Fix SQLite on network filesystems** — Either auto-detect and fall back to JSON, or document the limitation clearly.

### P1 — Important Gaps

5. **Implement native streaming** — OpenAI, Anthropic, and Ollama all support streaming natively. Currently all providers fake it.
6. **Build out observability** — The JSON formatter exists but no events are logged. Wire RuntimeEvents into structured logging with timing, token usage, and tool execution tracking.
7. **Add YAML config support** — Requirements specify YAML as first format option.
8. **Add tool policy to config** — Tool allow/deny lists should be configurable from file, not just code.
9. **Fix retry-after header usage** — The header is parsed but ignored in the retry loop.

### P2 — Quality Improvements

10. **Atomic file writes** — JsonFileStateStore should write to temp then rename.
11. **Shell command filtering** — Add command validation or approval hooks for dangerous commands.
12. **Symlink prevention** — Filesystem tools should reject paths that resolve through symlinks outside allowed roots.
13. **Secret redaction in logs** — API keys and sensitive content must be masked.
14. **Token counter accuracy** — The chars/4 heuristic is rough; consider optional tiktoken integration.

---

## Architecture Strengths

- Clean separation of concerns across providers, tools, context, memory, state, and runtime
- Provider factory pattern makes adding new backends trivial
- Pre-call pipeline is well-designed and genuinely extensible
- Policy engine with approval hooks is production-appropriate
- ProviderResourceExhaustedError handling is thorough and well-structured
- Plugin system (MCP, LangChain, LlamaIndex, OpenAPI) adds significant extensibility
- Export bridges for cross-framework tool definition sharing
- Dual async/sync API at the session level

---

## Conclusion

The agent_manager package has a solid architectural foundation that follows the requirements document's design principles closely. The provider abstraction, tool system, agent loop, policy engine, and context pipeline are all functional and well-designed. The main gaps are in subsystems that require deeper integration: real summarization (needs an LLM call), observability (needs event wiring), persistent long-term memory (needs a storage backend), and native streaming (needs provider-specific implementation). Addressing the P0 items above would bring the project to a genuinely functional MVP that meets all 10 acceptance criteria.
