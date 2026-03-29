# Providers and Connectivity

This guide shows which providers are implemented, how to configure them, and how to test that your models are reachable.

If you want copy-paste config blocks for every first-class provider, see [Provider Configurations](./provider-configurations.md).

## Implemented provider names

Use one of these names in config:

- `echo`
- `openai`
- `anthropic`
- `gemini`
- `ollama`
- `lmstudio`
- `vllm`

## Provider setup examples

### Echo

Useful for smoke tests with no external dependency.

```toml
[provider]
name = "echo"
model = "echo-v1"
```

### OpenAI

```toml
[provider]
name = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
```

### Anthropic

```toml
[provider]
name = "anthropic"
model = "claude-3-5-sonnet-latest"
api_key_env = "ANTHROPIC_API_KEY"
```

### Gemini

```toml
[provider]
name = "gemini"
model = "gemini-1.5-pro"
api_key_env = "GEMINI_API_KEY"
```

### Ollama

```toml
[provider]
name = "ollama"
model = "llama3.1"
base_url = "http://localhost:11434"
```

### LM Studio

LM Studio is used through its OpenAI-compatible local API.

```toml
[provider]
name = "lmstudio"
model = "local-model"
base_url = "http://localhost:1234/v1"
```

### vLLM

vLLM is supported as a first-class provider name and uses the OpenAI-compatible server interface.

```toml
[provider]
name = "vllm"
model = "NousResearch/Meta-Llama-3-8B-Instruct"
base_url = "http://localhost:8000/v1"
api_key_env = "VLLM_API_KEY"

[provider.settings]
extra_body = { top_k = 40, parallel_tool_calls = false }
```

## Quick connectivity tests

### 1. CLI smoke test

The simplest check:

```bash
agent-manager --provider echo --model echo-v1 "Reply with OK."
```

Then switch to your real provider:

```bash
agent-manager --provider ollama --model llama3.1 "Reply with OK."
```

Or with vLLM:

```bash
agent-manager --provider vllm --model NousResearch/Meta-Llama-3-8B-Instruct "Reply with OK."
```

### 2. Python smoke test

```python
from agent_manager import AgentSession, RuntimeConfig

session = AgentSession(
    config=RuntimeConfig.from_dict(
        {
            "provider": {"name": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "profile": "readonly",
        }
    )
)

result = session.run("Reply with exactly: OK")
print(result.output_text)
```

### 3. Streaming smoke test

```bash
agent-manager --provider openai --model gpt-4o-mini --stream "Reply with a short streamed response."
```

### 4. Structured output smoke test

Create `schema.json`:

```json
{
  "type": "object",
  "properties": {
    "status": {"type": "string"},
    "message": {"type": "string"}
  },
  "required": ["status", "message"]
}
```

Run:

```bash
agent-manager --provider openai --model gpt-4o-mini --structured-schema schema.json --json "Return a JSON status."
```

## Testing local model connectivity

### Ollama checklist

1. confirm Ollama is running
2. confirm the model is pulled locally
3. set `base_url` to the local Ollama API if needed
4. run a CLI smoke test

### LM Studio checklist

1. start the LM Studio local server
2. confirm the OpenAI-compatible endpoint is enabled
3. set `base_url` to the local server URL
4. run a CLI smoke test

### vLLM checklist

1. start the vLLM OpenAI-compatible server, usually on `http://localhost:8000/v1`
2. confirm the served model has a chat template
3. set `base_url` and `model` in the runtime config
4. if you started vLLM with an API key, set `VLLM_API_KEY`
5. run a CLI smoke test

## Testing provider fail states

The implementation includes resource-exhaustion handling.

That means quota, rate-limit, or capacity issues should surface as structured runtime failures rather than generic crashes.

Practical checks:

- intentionally use a bad API key and confirm you get a clear provider error
- intentionally point to a stopped local model server and confirm the request fails cleanly
- if possible, test a rate-limited environment and confirm retry behavior

## Provider-switching example

The rest of the runtime does not need to change when the provider changes.

```python
from agent_manager import AgentSession, RuntimeConfig

config = RuntimeConfig.from_dict(
    {
        "provider": {"name": "vllm", "model": "NousResearch/Meta-Llama-3-8B-Instruct"},
        "profile": "coding-agent",
    }
)

session = AgentSession(config=config)
print(session.run("Summarize the project.").output_text)

config.provider.name = "lmstudio"
config.provider.base_url = "http://localhost:1234/v1"
session = AgentSession(config=config)
print(session.run("Summarize the project.").output_text)
```

## Recommended rollout order

If you are connecting a new environment, test in this order:

1. `echo`
2. your real provider with a one-line prompt
3. streaming
4. structured output
5. tool use
6. long-running resumable tasks

## Related docs

- [Get Started](./get_started.md)
- [Configuration](./configuration.md)
- [Tools Overview](./tools.md)
- [Context Manager](./context-manager.md)
