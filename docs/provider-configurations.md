# Provider Configurations

This document gives copy-paste configuration examples for each first-class provider in `agent-manager`.

These examples match the current provider adapters in the codebase.

## Common shape

Every provider uses the same base config shape:

```toml
[provider]
name = "provider-name"
model = "model-name"
base_url = "https://example.test"
api_key_env = "ENV_VAR_NAME"

[provider.settings]
request_timeout_seconds = 60
request_retries = 3
request_retry_backoff_seconds = 0.5
```

Notes:

- `base_url` is optional when the provider has a built-in default
- `api_key_env` is optional when the provider has a built-in default or does not require an API key
- `provider.settings` is optional
- when you want to store the raw secret in the file, use `provider.settings.api_key`

## `echo`

Use this for smoke tests with no external dependency.

```toml
[provider]
name = "echo"
model = "echo-v1"
```

## `openai`

```toml
[provider]
name = "openai"
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[provider.settings]
request_timeout_seconds = 60
request_retries = 3
request_retry_backoff_seconds = 0.5
```

Direct-key variant:

```toml
[provider]
name = "openai"
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"

[provider.settings]
api_key = "your-openai-key"
```

## `anthropic`

```toml
[provider]
name = "anthropic"
model = "claude-3-5-sonnet-latest"
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"

[provider.settings]
request_timeout_seconds = 60
request_retries = 3
request_retry_backoff_seconds = 0.5
```

## `gemini`

```toml
[provider]
name = "gemini"
model = "gemini-1.5-pro"
base_url = "https://generativelanguage.googleapis.com/v1beta"
api_key_env = "GEMINI_API_KEY"

[provider.settings]
request_timeout_seconds = 60
request_retries = 3
request_retry_backoff_seconds = 0.5
```

## `ollama`

```toml
[provider]
name = "ollama"
model = "llama3.1"
base_url = "http://localhost:11434/api"

[provider.settings]
request_timeout_seconds = 120
request_retries = 2
request_retry_backoff_seconds = 0.25
```

## `lmstudio`

LM Studio uses its OpenAI-compatible local server.

```toml
[provider]
name = "lmstudio"
model = "local-model"
base_url = "http://localhost:1234/v1"
api_key_env = "LMSTUDIO_API_KEY"

[provider.settings]
request_timeout_seconds = 120
request_retries = 2
request_retry_backoff_seconds = 0.25
```

Notes:

- `LMSTUDIO_API_KEY` is optional in practice because the adapter does not require an API key by default
- keep `base_url` pointed at the LM Studio OpenAI-compatible endpoint

## `vllm`

vLLM uses its OpenAI-compatible local server and supports extra request-body fields through `provider.settings.extra_body`.

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

Notes:

- `VLLM_API_KEY` is optional unless you started the vLLM server with an API key
- the served model needs a chat template for chat-completions-style usage
- `extra_body` is the right place for vLLM-specific request parameters

## Minimal examples by environment

### Hosted providers

```toml
[provider]
name = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
```

```toml
[provider]
name = "anthropic"
model = "claude-3-5-sonnet-latest"
api_key_env = "ANTHROPIC_API_KEY"
```

```toml
[provider]
name = "gemini"
model = "gemini-1.5-pro"
api_key_env = "GEMINI_API_KEY"
```

### Local providers

```toml
[provider]
name = "ollama"
model = "llama3.1"
base_url = "http://localhost:11434/api"
```

```toml
[provider]
name = "lmstudio"
model = "local-model"
base_url = "http://localhost:1234/v1"
```

```toml
[provider]
name = "vllm"
model = "NousResearch/Meta-Llama-3-8B-Instruct"
base_url = "http://localhost:8000/v1"
```

## Full runtime example

```toml
profile = "coding-agent"
system_prompt = "You are a careful coding assistant."
state_backend = "sqlite"
state_path = ".agent_manager/state.sqlite3"

[provider]
name = "ollama"
model = "llama3.1"
base_url = "http://localhost:11434/api"

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
```

## Related docs

- [Providers and Connectivity](./providers-and-connectivity.md)
- [Configuration](./configuration.md)
- [Get Started](./get_started.md)
