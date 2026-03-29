# Configuration

`agent-manager` can be configured from:

- Python dictionaries through `RuntimeConfig.from_dict()`
- environment variables through `RuntimeConfig.from_env()`
- config files through `RuntimeConfig.from_file()` or `load_config()`
- the interactive wizard through `agent-manager-config`

Supported config file formats:

- TOML
- JSON
- YAML

## Minimal config

```toml
profile = "readonly"
system_prompt = "You are a helpful local-first agent runtime."

[provider]
name = "echo"
model = "echo-v1"
```

## Recommended local-dev config

```toml
profile = "local-dev"
system_prompt = "You are a careful coding assistant."
state_backend = "sqlite"
state_path = ".agent_manager/state.sqlite3"

[provider]
name = "ollama"
model = "llama3.1"
base_url = "http://localhost:11434"

[runtime]
max_steps = 8
timeout_seconds = 300
max_consecutive_failures = 3
max_context_tokens = 8192
max_output_tokens = 1024

[context]
history_window = 8
summary_trigger_messages = 8
retrieval_top_k = 3
max_memory_facts = 5
pre_call_functions = [
  "collect_recent_messages",
  "summarize_history",
  "inject_retrieval",
  "inject_memory_facts",
  "apply_token_budget",
  "finalize_messages",
]

[tools.web_search]
enabled = true
backend = "duckduckgo"
timeout_seconds = 20
max_results = 5
```

## Provider config

Provider fields currently supported by `ProviderConfig`:

- `name`
- `model`
- `base_url`
- `api_key_env`
- `settings`

Example:

```toml
[provider]
name = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"

[provider.settings]
request_timeout_seconds = 60
request_retries = 3
request_retry_backoff_seconds = 0.5
max_output_tokens = 1200
model_context_tokens = 128000
```

You can also store the raw key directly in TOML:

```toml
[provider]
name = "openai"
model = "gpt-4o-mini"

[provider.settings]
api_key = "your-openai-key"
```

The provider lookup order is:

1. `provider.settings.api_key`
2. environment variable named by `provider.api_key_env`

### vLLM example

```toml
[provider]
name = "vllm"
model = "NousResearch/Meta-Llama-3-8B-Instruct"
base_url = "http://localhost:8000/v1"
api_key_env = "VLLM_API_KEY"

[provider.settings]
request_timeout_seconds = 120
request_retries = 3
request_retry_backoff_seconds = 0.5
extra_body = { top_k = 40, parallel_tool_calls = false }
```

## Runtime limits

`runtime` controls the loop:

- `max_steps`
- `timeout_seconds`
- `max_consecutive_failures`
- `max_context_tokens`
- `max_output_tokens`

## Context config

`context` controls the pre-call pipeline:

- `history_window`
- `summary_trigger_messages`
- `retrieval_top_k`
- `max_memory_facts`
- `pre_call_functions`

## Tool policy config

`tool_policy` can tighten or loosen the active runtime profile.

```toml
[tool_policy]
allowed_tools = []
denied_tools = ["write_file"]
denied_tags = ["network"]
denied_permissions = ["process:execute"]
```

## Tool implementation config

`tools` controls how certain built-in tools are implemented.

Current structured tool config:

- `tools.web_search`

Example with Tavily:

```toml
[tools.web_search]
enabled = true
backend = "tavily"
api_key_env = "TAVILY_API_KEY"
endpoint = "https://api.tavily.com/search"
timeout_seconds = 20
max_results = 5

[tools.web_search.settings]
search_depth = "basic"
topic = "general"
include_answer = false
```

Example with Brave Search:

```toml
[tools.web_search]
enabled = true
backend = "brave"
api_key_env = "BRAVE_SEARCH_API_KEY"

[tools.web_search.settings]
extra_snippets = true
country = "us"
search_lang = "en"
```

Example with SerpAPI:

```toml
[tools.web_search]
enabled = true
backend = "serpapi"
api_key_env = "SERPAPI_API_KEY"

[tools.web_search.settings]
engine = "google"
gl = "us"
hl = "en"
```

Notes:

- `tool_policy` decides whether the agent is allowed to call `web_search`
- `[tools.web_search]` decides which backend actually handles the search
- `tools.web_search.enabled = false` removes the built-in `web_search` tool from default registration

## State backend

Supported backends in the current implementation:

- `sqlite`
- `json`

Relevant fields:

- `state_backend`
- `state_path`
- `state_dir`

Example:

```toml
state_backend = "json"
state_dir = ".agent_manager/state"
```

## Environment variable overrides

The runtime reads `AGENT_MANAGER_*` environment variables.

Common ones:

- `AGENT_MANAGER_PROVIDER`
- `AGENT_MANAGER_MODEL`
- `AGENT_MANAGER_BASE_URL`
- `AGENT_MANAGER_API_KEY_ENV`
- `AGENT_MANAGER_PROFILE`
- `AGENT_MANAGER_SYSTEM_PROMPT`
- `AGENT_MANAGER_STATE_DIR`
- `AGENT_MANAGER_STATE_PATH`
- `AGENT_MANAGER_LOG_LEVEL`
- `AGENT_MANAGER_LOG_JSON`
- `AGENT_MANAGER_ALLOWED_TOOLS`
- `AGENT_MANAGER_DENIED_TOOLS`
- `AGENT_MANAGER_DENIED_TAGS`
- `AGENT_MANAGER_DENIED_PERMISSIONS`

Example:

```bash
set AGENT_MANAGER_PROVIDER=ollama
set AGENT_MANAGER_MODEL=llama3.1
set AGENT_MANAGER_BASE_URL=http://localhost:11434
agent-manager "Summarize this repository."
```

## Loading from Python

```python
from agent_manager import RuntimeConfig

config = RuntimeConfig.from_file("agent-manager.toml")
```

Or:

```python
from agent_manager import RuntimeConfig

config = RuntimeConfig.from_env()
```

## Good configuration habits

- keep provider secrets in environment variables
- use `readonly` or policy overrides for research-style runs
- use `sqlite` for resumable local work
- tune `max_context_tokens` to the real model you are using
- keep `pre_call_functions` explicit per profile so behavior stays predictable

If you want a guided way to create the TOML file, see [Configuration Tool](./config-tool.md).

For a backend-by-backend guide to web search, see [Web Search Tool Setup](./tools-web-search.md).
