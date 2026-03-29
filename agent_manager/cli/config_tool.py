"""Interactive configuration wizard for generating TOML config files."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from agent_manager.config import ContextConfig, ProviderConfig, RuntimeConfig
from agent_manager.errors import (
    ConfigurationError,
    ProviderError,
    ProviderRequestError,
    ProviderResourceExhaustedError,
)
from agent_manager.providers.base import HTTPProvider
from agent_manager.providers.factory import available_providers, build_provider
from agent_manager.tools.policies import DEFAULT_PROFILES
from agent_manager.types import Message, ProviderRequest

DEFAULT_CONNECTION_TEST_PROMPT = "Reply with exactly: OK"

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "echo": {
        "model": "echo-v1",
        "description": "Local smoke-test provider with no external connection.",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "description": "Hosted OpenAI-compatible provider.",
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-latest",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "description": "Hosted Anthropic Messages API provider.",
    },
    "gemini": {
        "model": "gemini-1.5-pro",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GEMINI_API_KEY",
        "description": "Hosted Google Gemini provider.",
    },
    "ollama": {
        "model": "llama3.1",
        "base_url": "http://localhost:11434/api",
        "description": "Local Ollama server.",
    },
    "lmstudio": {
        "model": "local-model",
        "base_url": "http://localhost:1234/v1",
        "api_key_env": "LMSTUDIO_API_KEY",
        "description": "Local LM Studio OpenAI-compatible server.",
    },
    "vllm": {
        "model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "VLLM_API_KEY",
        "description": "Local vLLM OpenAI-compatible server.",
    },
}

PROFILE_HELP: dict[str, str] = {
    "readonly": "Blocks write, shell, and network tools by default.",
    "local-dev": "Permissive local development profile.",
    "coding-agent": "General coding profile with tool access enabled.",
    "unrestricted-lab": "Permissive lab profile for experiments.",
}


@dataclass(slots=True)
class FieldSpec:
    label: str
    help_text: str
    getter: Callable[[RuntimeConfig], Any]
    setter: Callable[[RuntimeConfig, str], None]
    clearer: Callable[[RuntimeConfig], None] | None = None


def _clean_for_toml(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _clean_for_toml(item)
            if normalized is None:
                continue
            if isinstance(normalized, Mapping) and not normalized:
                continue
            if isinstance(normalized, list) and not normalized:
                continue
            cleaned[str(key)] = normalized
        return cleaned
    if isinstance(value, list):
        return [_clean_for_toml(item) for item in value if item is not None]
    if value is None:
        return None
    return value


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _write_toml_table(lines: list[str], table_name: str, data: Mapping[str, Any]) -> None:
    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if table_name:
        lines.append(f"[{table_name}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_value(value)}")
    if table_name and (scalar_items or table_items):
        lines.append("")

    for key, value in table_items:
        child_name = f"{table_name}.{key}" if table_name else str(key)
        _write_toml_table(lines, child_name, value)


def runtime_config_to_toml(config: RuntimeConfig) -> str:
    data = _clean_for_toml(config.to_dict())
    if not isinstance(data, Mapping):
        raise TypeError("RuntimeConfig.to_dict() must produce a mapping.")

    lines = [
        "# Generated by agent-manager-config",
        "# Edit this file directly or regenerate it with the configuration wizard.",
        "",
    ]
    top_level_scalars: dict[str, Any] = {}
    top_level_tables: dict[str, Mapping[str, Any]] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            top_level_tables[str(key)] = value
        else:
            top_level_scalars[str(key)] = value

    for key, value in top_level_scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    if top_level_scalars:
        lines.append("")

    for key, value in top_level_tables.items():
        _write_toml_table(lines, key, value)

    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def save_runtime_config_toml(config: RuntimeConfig, path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.suffix.lower() not in {".toml", ".tml"}:
        file_path = file_path.with_suffix(".toml")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(runtime_config_to_toml(config), encoding="utf-8")
    return file_path


def load_runtime_config_for_wizard(path: str | Path) -> RuntimeConfig:
    return RuntimeConfig.from_file(path, apply_env=False)


def provider_connection_probe(
    config: RuntimeConfig,
    *,
    prompt: str = DEFAULT_CONNECTION_TEST_PROMPT,
) -> dict[str, Any]:
    provider_config = ProviderConfig.from_dict(config.provider.to_dict())
    provider = build_provider(provider_config)
    request = ProviderRequest(
        model=provider_config.model,
        messages=[Message(role="user", content=prompt)],
        max_tokens=max(min(config.runtime.max_output_tokens, 64), 1),
        temperature=0.0,
    )

    result = provider.generate_sync(request)
    response_text = result.text or ""
    if not response_text and result.structured_output is not None:
        response_text = json.dumps(result.structured_output, ensure_ascii=False)
    summary = {
        "success": True,
        "provider": provider.provider_name,
        "model": provider_config.model,
        "stop_reason": result.stop_reason,
        "output_text": response_text,
        "usage": result.usage,
    }
    if isinstance(provider, HTTPProvider):
        summary["base_url"] = provider.resolve_base_url()
    return summary


def config_usage_text(config_path: str | Path) -> str:
    path = Path(config_path)
    return "\n".join(
        [
            f"Generated config file: {path}",
            "",
            "CLI usage:",
            f'  agent-manager --config "{path}" "Summarize the repository."',
            "",
            "Python usage:",
            "  from agent_manager import AgentSession, load_config",
            f'  config = load_config(r"{path}")',
            "  session = AgentSession(config=config)",
            '  result = session.run("Reply with OK")',
            "  print(result.output_text)",
            "",
            "Environment usage:",
            f'  set AGENT_MANAGER_CONFIG={path}',
            '  agent-manager "Reply with OK"',
        ]
    )


def _display_value(value: Any, *, max_length: int = 72) -> str:
    if isinstance(value, list):
        text = ", ".join(str(item) for item in value)
    elif isinstance(value, Mapping):
        text = json.dumps(value, ensure_ascii=False)
    elif value is None:
        text = "(unset)"
    else:
        text = str(value)
    if not text:
        return "(empty)"
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("Enter yes/no, true/false, or 1/0.")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _set_provider_name(config: RuntimeConfig, raw: str) -> None:
    name = raw.strip().lower()
    if name not in available_providers():
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {', '.join(available_providers())}"
        )
    config.provider.name = name


def _set_provider_model(config: RuntimeConfig, raw: str) -> None:
    value = raw.strip()
    if not value:
        raise ValueError("Model must not be empty.")
    config.provider.model = value


def _set_optional_provider_field(field_name: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        setattr(config.provider, field_name, raw.strip() or None)
    return _setter


def _clear_optional_provider_field(field_name: str) -> Callable[[RuntimeConfig], None]:
    def _clearer(config: RuntimeConfig) -> None:
        setattr(config.provider, field_name, None)
    return _clearer


def _set_provider_setting_number(
    key: str,
    caster: Callable[[str], Any],
    *,
    allow_clear: bool = True,
) -> FieldSpec:
    def _getter(config: RuntimeConfig) -> Any:
        return config.provider.settings.get(key)

    def _setter(config: RuntimeConfig, raw: str) -> None:
        config.provider.settings[key] = caster(raw.strip())

    def _clearer(config: RuntimeConfig) -> None:
        if allow_clear:
            config.provider.settings.pop(key, None)

    return FieldSpec(
        label=key,
        help_text=f"Provider setting '{key}'. Leave unset to use the adapter default.",
        getter=_getter,
        setter=_setter,
        clearer=_clearer if allow_clear else None,
    )


def _set_provider_settings_json(config: RuntimeConfig, raw: str) -> None:
    config.provider.settings["extra_body"] = _parse_json_object(raw)


def _clear_provider_settings_json(config: RuntimeConfig) -> None:
    config.provider.settings.pop("extra_body", None)


def _set_top_level_string(attribute: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        value = raw.strip()
        if not value:
            raise ValueError(f"{attribute} must not be empty.")
        setattr(config, attribute, value)
    return _setter


def _set_top_level_optional_path(attribute: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        value = raw.strip()
        if not value:
            raise ValueError(f"{attribute} must not be empty.")
        setattr(config, attribute, value)
    return _setter


def _set_state_backend(config: RuntimeConfig, raw: str) -> None:
    backend = raw.strip().lower()
    if backend not in {"sqlite", "json"}:
        raise ValueError("State backend must be 'sqlite' or 'json'.")
    config.state_backend = backend


def _set_runtime_int(attribute: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        setattr(config.runtime, attribute, int(raw.strip()))
    return _setter


def _set_runtime_float(attribute: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        setattr(config.runtime, attribute, float(raw.strip()))
    return _setter


def _set_logging_bool(attribute: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        setattr(config.logging, attribute, _parse_bool(raw))
    return _setter


def _set_logging_string(attribute: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        value = raw.strip()
        if not value:
            raise ValueError(f"{attribute} must not be empty.")
        setattr(config.logging, attribute, value)
    return _setter


def _set_context_int(attribute: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        setattr(config.context, attribute, int(raw.strip()))
    return _setter


def _set_context_functions(config: RuntimeConfig, raw: str) -> None:
    values = _parse_csv(raw)
    if not values:
        raise ValueError("Pre-call functions list must not be empty.")
    config.context.pre_call_functions = values


def _set_policy_list(attribute: str) -> Callable[[RuntimeConfig, str], None]:
    def _setter(config: RuntimeConfig, raw: str) -> None:
        setattr(config.tool_policy, attribute, _parse_csv(raw))
    return _setter


def _clear_policy_list(attribute: str) -> Callable[[RuntimeConfig], None]:
    def _clearer(config: RuntimeConfig) -> None:
        setattr(config.tool_policy, attribute, [])
    return _clearer


def provider_fields() -> list[FieldSpec]:
    return [
        FieldSpec(
            label="provider.name",
            help_text=(
                "The provider adapter to use. Available values: "
                + ", ".join(available_providers())
            ),
            getter=lambda config: config.provider.name,
            setter=_set_provider_name,
        ),
        FieldSpec(
            label="provider.model",
            help_text="The model identifier sent to the provider.",
            getter=lambda config: config.provider.model,
            setter=_set_provider_model,
        ),
        FieldSpec(
            label="provider.base_url",
            help_text=(
                "Optional provider base URL. Leave unset to use the adapter default. "
                "Useful for local servers like Ollama, LM Studio, and vLLM."
            ),
            getter=lambda config: config.provider.base_url,
            setter=_set_optional_provider_field("base_url"),
            clearer=_clear_optional_provider_field("base_url"),
        ),
        FieldSpec(
            label="provider.api_key_env",
            help_text=(
                "Environment variable name that stores the provider API key. "
                "Leave unset when the provider does not require authentication."
            ),
            getter=lambda config: config.provider.api_key_env,
            setter=_set_optional_provider_field("api_key_env"),
            clearer=_clear_optional_provider_field("api_key_env"),
        ),
        _set_provider_setting_number("request_timeout_seconds", float),
        _set_provider_setting_number("request_retries", int),
        _set_provider_setting_number("request_retry_backoff_seconds", float),
        FieldSpec(
            label="provider.settings.extra_body",
            help_text=(
                "Optional JSON object merged into the provider request body. "
                "Useful for provider-specific knobs like vLLM extra parameters."
            ),
            getter=lambda config: config.provider.settings.get("extra_body"),
            setter=_set_provider_settings_json,
            clearer=_clear_provider_settings_json,
        ),
    ]


def runtime_fields() -> list[FieldSpec]:
    return [
        FieldSpec(
            label="runtime.max_steps",
            help_text="Maximum number of loop iterations before the run stops.",
            getter=lambda config: config.runtime.max_steps,
            setter=_set_runtime_int("max_steps"),
        ),
        FieldSpec(
            label="runtime.timeout_seconds",
            help_text="Overall timeout for the agent loop.",
            getter=lambda config: config.runtime.timeout_seconds,
            setter=_set_runtime_float("timeout_seconds"),
        ),
        FieldSpec(
            label="runtime.max_consecutive_failures",
            help_text="Maximum repeated failures before the loop stops.",
            getter=lambda config: config.runtime.max_consecutive_failures,
            setter=_set_runtime_int("max_consecutive_failures"),
        ),
        FieldSpec(
            label="runtime.max_context_tokens",
            help_text="Budget used when assembling model input context.",
            getter=lambda config: config.runtime.max_context_tokens,
            setter=_set_runtime_int("max_context_tokens"),
        ),
        FieldSpec(
            label="runtime.max_output_tokens",
            help_text="Requested output-token budget for the provider call.",
            getter=lambda config: config.runtime.max_output_tokens,
            setter=_set_runtime_int("max_output_tokens"),
        ),
    ]


def context_fields() -> list[FieldSpec]:
    defaults = ", ".join(ContextConfig().pre_call_functions)
    return [
        FieldSpec(
            label="context.history_window",
            help_text="How many recent messages are carried directly into the next call.",
            getter=lambda config: config.context.history_window,
            setter=_set_context_int("history_window"),
        ),
        FieldSpec(
            label="context.summary_trigger_messages",
            help_text="Message count that encourages summary creation.",
            getter=lambda config: config.context.summary_trigger_messages,
            setter=_set_context_int("summary_trigger_messages"),
        ),
        FieldSpec(
            label="context.retrieval_top_k",
            help_text="How many retrieved items are pulled into context.",
            getter=lambda config: config.context.retrieval_top_k,
            setter=_set_context_int("retrieval_top_k"),
        ),
        FieldSpec(
            label="context.max_memory_facts",
            help_text="Maximum number of memory facts injected into a turn.",
            getter=lambda config: config.context.max_memory_facts,
            setter=_set_context_int("max_memory_facts"),
        ),
        FieldSpec(
            label="context.pre_call_functions",
            help_text=(
                "Comma-separated names of pre-call context steps. "
                f"Defaults: {defaults}"
            ),
            getter=lambda config: config.context.pre_call_functions,
            setter=_set_context_functions,
        ),
    ]


def logging_fields() -> list[FieldSpec]:
    return [
        FieldSpec(
            label="logging.level",
            help_text="Logging level such as DEBUG, INFO, WARNING, or ERROR.",
            getter=lambda config: config.logging.level,
            setter=_set_logging_string("level"),
        ),
        FieldSpec(
            label="logging.json_output",
            help_text="Emit logs as JSON lines instead of plain text.",
            getter=lambda config: config.logging.json_output,
            setter=_set_logging_bool("json_output"),
        ),
    ]


def policy_fields() -> list[FieldSpec]:
    return [
        FieldSpec(
            label="tool_policy.allowed_tools",
            help_text="Comma-separated tool names that are explicitly allowed.",
            getter=lambda config: config.tool_policy.allowed_tools,
            setter=_set_policy_list("allowed_tools"),
            clearer=_clear_policy_list("allowed_tools"),
        ),
        FieldSpec(
            label="tool_policy.denied_tools",
            help_text="Comma-separated tool names that are blocked.",
            getter=lambda config: config.tool_policy.denied_tools,
            setter=_set_policy_list("denied_tools"),
            clearer=_clear_policy_list("denied_tools"),
        ),
        FieldSpec(
            label="tool_policy.denied_tags",
            help_text="Comma-separated tool tags that are blocked.",
            getter=lambda config: config.tool_policy.denied_tags,
            setter=_set_policy_list("denied_tags"),
            clearer=_clear_policy_list("denied_tags"),
        ),
        FieldSpec(
            label="tool_policy.denied_permissions",
            help_text="Comma-separated permission strings that are blocked.",
            getter=lambda config: config.tool_policy.denied_permissions,
            setter=_set_policy_list("denied_permissions"),
            clearer=_clear_policy_list("denied_permissions"),
        ),
    ]


def top_level_fields() -> list[FieldSpec]:
    profiles = ", ".join(DEFAULT_PROFILES.keys())
    profile_help = "; ".join(f"{key}: {value}" for key, value in PROFILE_HELP.items())
    return [
        FieldSpec(
            label="profile",
            help_text=f"Runtime profile. Built-in profiles: {profiles}. {profile_help}",
            getter=lambda config: config.profile,
            setter=_set_top_level_string("profile"),
        ),
        FieldSpec(
            label="system_prompt",
            help_text="Top-level system instruction inserted into the context.",
            getter=lambda config: config.system_prompt,
            setter=_set_top_level_string("system_prompt"),
        ),
        FieldSpec(
            label="state_backend",
            help_text="Checkpoint backend: sqlite or json.",
            getter=lambda config: config.state_backend,
            setter=_set_state_backend,
        ),
        FieldSpec(
            label="state_path",
            help_text="SQLite checkpoint file path.",
            getter=lambda config: config.state_path,
            setter=_set_top_level_optional_path("state_path"),
        ),
        FieldSpec(
            label="state_dir",
            help_text="Directory used by the JSON checkpoint backend.",
            getter=lambda config: config.state_dir,
            setter=_set_top_level_optional_path("state_dir"),
        ),
    ]


class ConfigWizard:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        source_path: str | Path | None = None,
        output_path: str | Path | None = None,
        input_fn: Callable[[str], str] = input,
        out: TextIO | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.source_path = Path(source_path) if source_path else None
        self.output_path = Path(output_path) if output_path else self._default_output_path()
        self.input_fn = input_fn
        self.out = out or sys.stdout
        self.dirty = False

    def _default_output_path(self) -> Path:
        if self.source_path is not None:
            return self.source_path.with_suffix(".toml")
        return Path("agent-manager.generated.toml")

    def run(self) -> int:
        while True:
            self._print_main_menu()
            choice = self.input_fn("Choose an option: ").strip().lower()
            if choice == "1":
                self._load_existing_config()
            elif choice == "2":
                self._edit_section("Provider", provider_fields())
            elif choice == "3":
                self._apply_provider_template()
            elif choice == "4":
                self._test_provider_connection()
            elif choice == "5":
                self._edit_section("Runtime", runtime_fields())
            elif choice == "6":
                self._edit_section("Context", context_fields())
            elif choice == "7":
                self._edit_section("Logging", logging_fields())
            elif choice == "8":
                self._edit_section("Tool Policy", policy_fields())
            elif choice == "9":
                self._edit_section("General and State", top_level_fields())
            elif choice == "10":
                self._preview_toml()
            elif choice == "11":
                self._save_toml()
            elif choice == "12":
                self._show_usage_examples()
            elif choice in {"0", "q", "quit", "exit"}:
                if self.dirty and not self._confirm("You have unsaved changes. Exit anyway? [y/N]: "):
                    continue
                self.out.write("Exiting configuration wizard.\n")
                return 0
            else:
                self.out.write("Unknown option. Choose a menu number.\n")

    def _print_main_menu(self) -> None:
        self.out.write("\n=== agent-manager configuration wizard ===\n")
        self.out.write(
            f"Provider: {self.config.provider.name} | Model: {self.config.provider.model} | "
            f"Profile: {self.config.profile}\n"
        )
        self.out.write(f"Output TOML: {self.output_path}\n")
        if self.source_path is not None:
            self.out.write(f"Loaded from: {self.source_path}\n")
        self.out.write("\n")
        self.out.write("1. Load existing config file\n")
        self.out.write("2. Edit provider\n")
        self.out.write("3. Apply recommended provider template\n")
        self.out.write("4. Test provider connection\n")
        self.out.write("5. Edit runtime\n")
        self.out.write("6. Edit context\n")
        self.out.write("7. Edit logging\n")
        self.out.write("8. Edit tool policy\n")
        self.out.write("9. Edit general and state settings\n")
        self.out.write("10. Preview generated TOML\n")
        self.out.write("11. Save TOML config\n")
        self.out.write("12. Show how to use the generated config\n")
        self.out.write("0. Exit\n\n")

    def _edit_section(self, title: str, fields: list[FieldSpec]) -> None:
        while True:
            self.out.write(f"\n--- {title} ---\n")
            for index, field in enumerate(fields, start=1):
                self.out.write(
                    f"{index}. {field.label}: {_display_value(field.getter(self.config))}\n"
                )
            self.out.write("h. Show help for a field\n")
            self.out.write("b. Back to main menu\n")
            choice = self.input_fn("Choose a field: ").strip().lower()
            if choice == "b":
                return
            if choice == "h":
                self._show_field_help(fields)
                continue
            if not choice.isdigit():
                self.out.write("Choose a field number, 'h', or 'b'.\n")
                continue
            index = int(choice) - 1
            if index < 0 or index >= len(fields):
                self.out.write("That field number is out of range.\n")
                continue
            self._edit_field(fields[index])

    def _show_field_help(self, fields: list[FieldSpec]) -> None:
        raw = self.input_fn("Field number for help: ").strip()
        if not raw.isdigit():
            self.out.write("Enter a field number.\n")
            return
        index = int(raw) - 1
        if index < 0 or index >= len(fields):
            self.out.write("That field number is out of range.\n")
            return
        field = fields[index]
        self.out.write(f"\n{field.label}\n{field.help_text}\n")

    def _edit_field(self, field: FieldSpec) -> None:
        current = field.getter(self.config)
        self.out.write(f"\n{field.label}\n")
        self.out.write(f"Current value: {_display_value(current)}\n")
        self.out.write(f"Help: {field.help_text}\n")
        self.out.write("Enter a new value. Leave blank to keep the current value.\n")
        if field.clearer is not None:
            self.out.write("Type '-' to clear this value.\n")
        while True:
            raw = self.input_fn("> ")
            if raw == "":
                return
            if raw.strip() == "?":
                self.out.write(field.help_text + "\n")
                continue
            if raw.strip() == "-" and field.clearer is not None:
                field.clearer(self.config)
                self.dirty = True
                self.out.write(f"Cleared {field.label}.\n")
                return
            try:
                field.setter(self.config, raw)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.out.write(f"Invalid value: {exc}\n")
                continue
            self.dirty = True
            self.out.write(f"Updated {field.label}.\n")
            return

    def _load_existing_config(self) -> None:
        raw_path = self.input_fn("Path to existing JSON/TOML/YAML config: ").strip()
        if not raw_path:
            self.out.write("No path entered.\n")
            return
        try:
            config = load_runtime_config_for_wizard(raw_path)
        except (ConfigurationError, OSError, json.JSONDecodeError) as exc:
            self.out.write(f"Failed to load config: {exc}\n")
            return
        self.config = config
        self.source_path = Path(raw_path)
        self.output_path = self.source_path.with_suffix(".toml")
        self.dirty = True
        self.out.write(f"Loaded config from {self.source_path}.\n")

    def _apply_provider_template(self) -> None:
        self.out.write("\nAvailable provider templates:\n")
        names = available_providers()
        for index, name in enumerate(names, start=1):
            preset = PROVIDER_PRESETS.get(name, {})
            description = str(preset.get("description", ""))
            self.out.write(f"{index}. {name}: {description}\n")
        raw = self.input_fn("Choose a provider template: ").strip()
        if not raw.isdigit():
            self.out.write("Enter a provider number.\n")
            return
        index = int(raw) - 1
        if index < 0 or index >= len(names):
            self.out.write("That provider number is out of range.\n")
            return
        provider_name = names[index]
        preset = PROVIDER_PRESETS.get(provider_name, {})
        self.config.provider.name = provider_name
        if preset.get("model"):
            self.config.provider.model = str(preset["model"])
        self.config.provider.base_url = preset.get("base_url")
        self.config.provider.api_key_env = preset.get("api_key_env")
        self.dirty = True
        self.out.write(
            f"Applied recommended settings for {provider_name}. "
            f"Model: {self.config.provider.model}\n"
        )

    def _test_provider_connection(self) -> None:
        raw_prompt = self.input_fn(
            f"Connection test prompt [{DEFAULT_CONNECTION_TEST_PROMPT}]: "
        ).strip()
        prompt = raw_prompt or DEFAULT_CONNECTION_TEST_PROMPT
        self.out.write("Testing provider connection...\n")
        try:
            summary = provider_connection_probe(self.config, prompt=prompt)
        except ProviderResourceExhaustedError as exc:
            self.out.write(
                "Provider reported resource exhaustion:\n"
                f"{json.dumps(exc.to_dict(), indent=2, ensure_ascii=False)}\n"
            )
            return
        except (ConfigurationError, ProviderRequestError, ProviderError, OSError) as exc:
            self.out.write(f"Connection test failed: {exc}\n")
            return
        except Exception as exc:  # pragma: no cover - defensive CLI fallback
            self.out.write(f"Unexpected provider error: {exc}\n")
            return

        self.out.write("Connection test succeeded.\n")
        self.out.write(f"Provider: {summary['provider']}\n")
        self.out.write(f"Model: {summary['model']}\n")
        if "base_url" in summary:
            self.out.write(f"Base URL: {summary['base_url']}\n")
        self.out.write(f"Stop reason: {summary.get('stop_reason')}\n")
        self.out.write(f"Output: {_display_value(summary.get('output_text', ''), max_length=120)}\n")
        if summary.get("usage") is not None:
            self.out.write(
                "Usage: "
                f"{json.dumps(summary['usage'], ensure_ascii=False)}\n"
            )

    def _preview_toml(self) -> None:
        self.out.write("\n--- TOML Preview ---\n")
        self.out.write(runtime_config_to_toml(self.config))

    def _save_toml(self) -> None:
        raw_path = self.input_fn(f"Save path [{self.output_path}]: ").strip()
        target = Path(raw_path) if raw_path else self.output_path
        try:
            saved_path = save_runtime_config_toml(self.config, target)
        except OSError as exc:
            self.out.write(f"Failed to save TOML: {exc}\n")
            return
        self.output_path = saved_path
        self.dirty = False
        self.out.write(f"Saved TOML config to {saved_path}\n")

    def _show_usage_examples(self) -> None:
        self.out.write("\n--- Config Usage ---\n")
        self.out.write(config_usage_text(self.output_path) + "\n")

    def _confirm(self, prompt: str) -> bool:
        answer = self.input_fn(prompt).strip().lower()
        return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-manager-config",
        description="Interactive wizard for generating agent-manager TOML configs.",
    )
    parser.add_argument(
        "--import-config",
        help="Optional JSON, TOML, or YAML config to load before the wizard starts.",
    )
    parser.add_argument(
        "--output",
        help="Optional default output path for the generated TOML file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config: RuntimeConfig | None = None
    source_path: str | Path | None = None
    if args.import_config:
        source_path = args.import_config
        try:
            config = load_runtime_config_for_wizard(args.import_config)
        except (ConfigurationError, OSError, json.JSONDecodeError) as exc:
            parser.error(f"Failed to load import config: {exc}")

    wizard = ConfigWizard(
        config=config,
        source_path=source_path,
        output_path=args.output,
    )
    return wizard.run()


if __name__ == "__main__":
    raise SystemExit(main())
