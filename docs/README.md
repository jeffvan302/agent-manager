# Documentation

This folder documents the current `agent-manager` implementation and maps it back to the project goals in `requirements.md`.

## Start here

- [Get Started](./get_started.md)
- [Configuration](./configuration.md)
- [Configuration Tool](./config-tool.md)
- [Provider Configurations](./provider-configurations.md)
- [Providers and Connectivity](./providers-and-connectivity.md)
- [Context Manager](./context-manager.md)
- [Advanced Two-Agent Coding Example](./advanced-two-agent-coding.md)

## Tools

- [Tools Overview](./tools.md)
- [Build Your Own Tool](./tools-your-own.md)
- [Retrieval Tool Setup](./tools-retrieval.md)
- [TinyDB Tool Setup](./tools-tinydb.md)

## What these docs assume

- Python `3.11+`
- `agent-manager` installed from this repository
- a configured provider if you want to run a real model instead of the `echo` provider

## Useful starting commands

```bash
pip install -e .
agent-manager "Summarize the project goals."
```

For provider-specific installation and connectivity checks, see [Providers and Connectivity](./providers-and-connectivity.md).
