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
- tool policy
- general profile and state storage settings

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

## Example workflow

1. Run `agent-manager-config`
2. Choose `3` to apply a provider template
3. Choose `2` to edit provider values like model or base URL
4. Choose `4` to test the provider connection
5. Edit runtime and context settings
6. Choose `10` to preview the TOML
7. Choose `11` to save the file
8. Choose `12` to view usage examples for the generated config

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
