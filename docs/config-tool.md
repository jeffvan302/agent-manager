# Configuration Tool

`agent-manager` now includes an interactive configuration wizard that can:

- load an existing JSON, TOML, or YAML config
- guide you through provider, runtime, context, logging, policy, and state settings
- test a provider connection before you save
- write a clean TOML config file for the runtime

## Command

After installing the package, run:

```bash
agent-manager-config
```

You can also import an existing config first:

```bash
agent-manager-config --import-config agent-manager.yaml
```

Or choose a default output path:

```bash
agent-manager-config --output configs/local-dev.toml
```

## What the wizard edits

The menu covers:

- provider settings
- runtime limits
- context settings
- logging
- tool implementation settings
- tool policy
- general profile and state storage settings

In the tool policy section, the wizard now lists the known built-in tool names so users can see valid values while editing `allowed_tools` and `denied_tools`.

In the tools section, the wizard can configure the built-in `web_search` tool backend, including:

- `google`
- `duckduckgo`
- `serpapi`
- `tavily`
- `brave`

The `google` backend now assumes your installed `google_search_tool` package and its browser setup from [google_search_tool.md](../google_search_tool.md).

For provider authentication, the wizard now supports both:

- `provider.api_key_env` for an environment variable name such as `OPENAI_API_KEY`
- `provider.settings.api_key` for storing the raw key directly in the TOML

## Provider testing

One of the menu options runs a live provider connection test with the current config.

That test:

- builds the selected provider adapter
- sends a short prompt
- shows the stop reason and returned text
- surfaces config, auth, network, and resource-exhaustion errors clearly

This is especially useful for:

- OpenAI
- Anthropic
- Gemini
- Ollama
- LM Studio
- vLLM

## Provider templates

The wizard includes recommended starter templates for all first-class providers:

- `echo`
- `openai`
- `anthropic`
- `gemini`
- `ollama`
- `lmstudio`
- `vllm`

These templates fill in a recommended model, base URL, and API key environment variable where appropriate.

## Tooltips and editing behavior

Each editable field shows:

- the current value
- a short explanation
- the expected format

Editing tips:

- press `Enter` to keep the current value
- type `?` while editing a field to see its help text again
- type `-` on clearable fields to unset them

Important provider note:

- `provider.api_key_env` expects the name of an environment variable, not the secret value
- if you want the secret stored directly in the file, use `provider.settings.api_key`

## Example workflow

1. Run `agent-manager-config`
2. Choose `3` to apply a provider template
3. Choose `2` to edit provider values like model or base URL
4. Choose `4` to test the provider connection
5. Edit runtime, context, and tools settings
6. Choose `11` to preview the TOML
7. Choose `12` to save the file
8. Choose `13` to view usage examples for the generated config

## Using the generated TOML file

### CLI

```bash
agent-manager --config configs/local-dev.toml "Summarize the repository."
```

### Python

```python
from agent_manager import AgentSession, load_config

config = load_config("configs/local-dev.toml")
session = AgentSession(config=config)
result = session.run("Reply with OK")
print(result.output_text)
```

### Example TOML with raw API key

```toml
[provider]
name = "openai"
model = "gpt-4o-mini"

[provider.settings]
api_key = "your-openai-key"
```

This works because the provider layer checks `provider.settings.api_key` before falling back to `provider.api_key_env`.

### Environment variable

```bash
set AGENT_MANAGER_CONFIG=configs/local-dev.toml
agent-manager "Reply with OK"
```

## When this tool is most useful

Use the wizard when you want to:

- turn an older JSON or YAML config into TOML
- confirm a provider is reachable before starting a run
- help teammates create a correct config without reading every provider doc
- standardize runtime and context settings across projects

## Related docs

- [Configuration](./configuration.md)
- [Provider Configurations](./provider-configurations.md)
- [Providers and Connectivity](./providers-and-connectivity.md)
- [Web Search Tool Setup](./tools-web-search.md)
