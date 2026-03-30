# Web Search Tool Setup

`agent-manager` can now configure the built-in `web_search` tool directly from TOML through `[tools.web_search]`.

This controls:

- whether the tool is registered at all
- which backend powers it
- where the backend endpoint lives
- how API keys are loaded
- backend-specific request settings

## Supported backends

- `google`
- `duckduckgo`
- `serpapi`
- `tavily`
- `brave`

## Default Google example

When nothing else is configured, `agent-manager` now defaults to `google` through the installed `google_search_tool` package.

```toml
[tools.web_search]
enabled = true
backend = "google"
timeout_seconds = 20
max_results = 5

[tools.web_search.settings]
headless = true
cookie_file = "google_cookies.json"
```

Setup guide:

- see [google_search_tool.md](../google_search_tool.md)

Notes:

- import path is `google_search_tool`
- this backend does not require an API key
- it is now the default when no other web-search backend is configured

## DuckDuckGo example

```toml
[tools.web_search]
enabled = true
backend = "duckduckgo"
endpoint = "https://api.duckduckgo.com/"
timeout_seconds = 20
max_results = 5
```

Notes:

- no API key is required
- this is now an explicit fallback backend rather than the default

## SerpAPI example

```toml
[tools.web_search]
enabled = true
backend = "serpapi"
api_key_env = "SERPAPI_API_KEY"
endpoint = "https://serpapi.com/search.json"
timeout_seconds = 20
max_results = 5

[tools.web_search.settings]
engine = "google"
gl = "us"
hl = "en"
```

Or store the key directly:

```toml
[tools.web_search]
backend = "serpapi"
api_key = "your-serpapi-key"
```

## Tavily example

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

Useful Tavily settings include:

- `search_depth`
- `topic`
- `include_answer`
- `include_raw_content`
- `include_domains`
- `exclude_domains`

## Brave Search example

```toml
[tools.web_search]
enabled = true
backend = "brave"
api_key_env = "BRAVE_SEARCH_API_KEY"
endpoint = "https://api.search.brave.com/res/v1/web/search"
timeout_seconds = 20
max_results = 5

[tools.web_search.settings]
extra_snippets = true
country = "us"
search_lang = "en"
```

## Wizard support

The `agent-manager-config` wizard now has a dedicated **Tools** section.

In that section you can edit:

- `tools.web_search.enabled`
- `tools.web_search.backend`
- `tools.web_search.endpoint`
- `tools.web_search.api_key_env`
- `tools.web_search.api_key`
- `tools.web_search.timeout_seconds`
- `tools.web_search.max_results`
- `tools.web_search.settings`

When you switch the backend, the wizard fills in a recommended endpoint and API key environment variable name for that backend.

## Testing the tool

Use `tool-test` to verify the configured backend without involving a model:

```bash
tool-test --config test-conn.toml --schema web_search
tool-test --config test-conn.toml web_search "budget GPU for yolo12 training"
```

If you want a web-only research profile, combine it with policy settings like:

```toml
profile = "readonly"

[tool_policy]
allowed_tools = ["web_search", "http_request"]
denied_permissions = ["filesystem:write", "process:execute"]
```

## Using it from Python

```python
from agent_manager import AgentSession, load_config

config = load_config("test-conn.toml")
session = AgentSession(config=config)
result = session.run("Find recent GPU recommendations for YOLO training.")
print(result.output_text)
```

## Notes

- `tools.web_search.enabled = false` removes the built-in `web_search` tool from default registration
- custom code can still inject a different `web_searcher` directly into `AgentSession(...)`
- `tool_policy` controls access to the tool, while `[tools.web_search]` controls how the tool is implemented
