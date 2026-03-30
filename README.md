# agent-manager

`agent-manager` is a local-first Python agent runtime that can run as a reusable library, a headless local service, and a CLI.

The current implementation includes:

- provider adapters for `openai`, `anthropic`, `gemini`, `ollama`, `lmstudio`, `vllm`, and `echo`
- a resumable agent loop with sync, async, and streaming execution
- built-in tools for filesystem, shell, HTTP, web search, and retrieval
- context assembly with a configurable pre-call pipeline
- checkpointing, retrieval, plugin integrations, and a CLI/API layer

## Install

```bash
pip install -e .
```

## Quick Start

```python
from agent_manager import AgentSession, RuntimeConfig

session = AgentSession(config=RuntimeConfig.from_env())
result = session.run("Summarize the goals of this project.")

print(result.output_text)
```

The default provider is `echo`, which is useful for smoke tests before connecting a real model backend.

## CLI

```bash
agent-manager "Plan the next implementation step."
```

Optional configuration can be supplied with `--config path/to/config.toml`.

## Documentation

See the docs folder for implementation guides:

- `docs/get_started.md`
- `docs/config-tool.md`
- `docs/tools.md`
- `docs/context-manager.md`
- `docs/providers-and-connectivity.md`
- `docs/advanced-two-agent-coding.md`

## Config Wizard

Generate and test TOML configs with:

```bash
agent-manager-config
```

## Tool Testing

Test registered tools directly with:

```bash
tool-test --list
tool-test read_file "requirements.md"
tool-test web_search "budget GPU for yolo12 training"
```

`web_search` can now be configured through `[tools.web_search]` in TOML. The default
backend is `google` via your installed `google_search_tool`, and you can also switch to
`duckduckgo`, `serpapi`, `tavily`, or `brave`.

## Package Layout

```text
agent_manager/
  providers/
  tools/
  context/
  memory/
  state/
  runtime/
  plugins/
  api/
  cli/
```

## Local provider examples

Ollama:

```bash
agent-manager --provider ollama --model llama3.1 "Summarize this repository."
```

vLLM:

```bash
agent-manager --provider vllm --model NousResearch/Meta-Llama-3-8B-Instruct "Summarize this repository."
```
