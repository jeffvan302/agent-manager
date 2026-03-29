"""Configuration loading for the agent_manager runtime."""

from __future__ import annotations

import json
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redefine]
try:
    import yaml as _yaml
except ModuleNotFoundError:
    _yaml = None  # type: ignore[assignment]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from agent_manager.errors import ConfigurationError


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid integer for {field_name}: {value}") from exc


def _parse_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid number for {field_name}: {value}") from exc


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class ProviderConfig:
    name: str = "echo"
    model: str = "echo-v1"
    base_url: str | None = None
    api_key_env: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ProviderConfig":
        data = data or {}
        return cls(
            name=str(data.get("name", "echo")),
            model=str(data.get("model", "echo-v1")),
            base_url=data.get("base_url"),
            api_key_env=data.get("api_key_env"),
            settings=dict(data.get("settings", {})),
        )

    def resolved_max_context_tokens(self, fallback: int) -> int:
        value = (
            self.settings.get("model_context_tokens")
            or self.settings.get("context_window")
            or self.settings.get("max_context_tokens")
        )
        return int(value) if value is not None else fallback

    def resolved_max_output_tokens(self, fallback: int) -> int:
        value = (
            self.settings.get("model_max_output_tokens")
            or self.settings.get("max_output_tokens")
            or self.settings.get("output_limit")
        )
        return int(value) if value is not None else fallback

    def resolved_token_counter_chars_per_token(self, fallback: float = 4.0) -> float:
        value = self.settings.get("token_counter_chars_per_token")
        return float(value) if value is not None else fallback


@dataclass(slots=True)
class RuntimeLimits:
    max_steps: int = 6
    timeout_seconds: float = 300.0
    max_consecutive_failures: int = 3
    max_context_tokens: int = 8192
    max_output_tokens: int = 1024

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeLimits":
        data = data or {}
        return cls(
            max_steps=int(data.get("max_steps", 6)),
            timeout_seconds=float(data.get("timeout_seconds", 300.0)),
            max_consecutive_failures=int(data.get("max_consecutive_failures", 3)),
            max_context_tokens=int(data.get("max_context_tokens", 8192)),
            max_output_tokens=int(data.get("max_output_tokens", 1024)),
        )


@dataclass(slots=True)
class LoggingConfig:
    level: str = "WARNING"
    json_output: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "LoggingConfig":
        data = data or {}
        return cls(
            level=str(data.get("level", "WARNING")),
            json_output=bool(data.get("json_output", False)),
        )


@dataclass(slots=True)
class ContextConfig:
    history_window: int = 8
    summary_trigger_messages: int = 8
    retrieval_top_k: int = 3
    max_memory_facts: int = 5
    pre_call_functions: list[str] = field(
        default_factory=lambda: [
            "collect_recent_messages",
            "summarize_history",
            "inject_retrieval",
            "inject_memory_facts",
            "apply_token_budget",
            "finalize_messages",
        ]
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ContextConfig":
        data = data or {}
        raw_functions = data.get("pre_call_functions")
        if isinstance(raw_functions, list):
            pre_call_functions = [str(item) for item in raw_functions if str(item).strip()]
        else:
            pre_call_functions = cls().pre_call_functions
        return cls(
            history_window=int(data.get("history_window", 8)),
            summary_trigger_messages=int(data.get("summary_trigger_messages", 8)),
            retrieval_top_k=int(data.get("retrieval_top_k", 3)),
            max_memory_facts=int(data.get("max_memory_facts", 5)),
            pre_call_functions=pre_call_functions,
        )


@dataclass(slots=True)
class ToolPolicyConfig:
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    denied_tags: list[str] = field(default_factory=list)
    denied_permissions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ToolPolicyConfig":
        data = data or {}
        return cls(
            allowed_tools=list(data.get("allowed_tools", [])),
            denied_tools=list(data.get("denied_tools", [])),
            denied_tags=list(data.get("denied_tags", [])),
            denied_permissions=list(data.get("denied_permissions", [])),
        )


@dataclass(slots=True)
class RuntimeConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    runtime: RuntimeLimits = field(default_factory=RuntimeLimits)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    tool_policy: ToolPolicyConfig = field(default_factory=ToolPolicyConfig)
    profile: str = "readonly"
    system_prompt: str = "You are a helpful local-first agent runtime."
    state_backend: str = "sqlite"
    state_path: str = ".agent_manager/state.sqlite3"
    state_dir: str = ".agent_manager/state"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeConfig":
        data = data or {}
        return cls(
            provider=ProviderConfig.from_dict(data.get("provider")),
            runtime=RuntimeLimits.from_dict(data.get("runtime")),
            logging=LoggingConfig.from_dict(data.get("logging")),
            context=ContextConfig.from_dict(data.get("context")),
            tool_policy=ToolPolicyConfig.from_dict(data.get("tool_policy")),
            profile=str(data.get("profile", "readonly")),
            system_prompt=str(
                data.get("system_prompt", "You are a helpful local-first agent runtime.")
            ),
            state_backend=str(data.get("state_backend", "sqlite")).lower(),
            state_path=str(data.get("state_path", ".agent_manager/state.sqlite3")),
            state_dir=str(data.get("state_dir", ".agent_manager/state")),
            extra=dict(data.get("extra", {})),
        )

    @classmethod
    def from_env(cls, prefix: str = "AGENT_MANAGER_") -> "RuntimeConfig":
        config = cls()
        config.apply_env_overrides(os.environ, prefix=prefix)
        return config

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        prefix: str = "AGENT_MANAGER_",
        apply_env: bool = True,
    ) -> "RuntimeConfig":
        file_path = Path(path)
        if not file_path.exists():
            raise ConfigurationError(f"Config file does not exist: {file_path}")

        if file_path.suffix.lower() == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
        elif file_path.suffix.lower() in {".toml", ".tml"}:
            data = tomllib.loads(file_path.read_text(encoding="utf-8"))
        elif file_path.suffix.lower() in {".yaml", ".yml"}:
            if _yaml is None:
                raise ConfigurationError(
                    "PyYAML is required for YAML config files. Install with: pip install pyyaml"
                )
            data = _yaml.safe_load(file_path.read_text(encoding="utf-8"))
        else:
            raise ConfigurationError(
                f"Unsupported config format for {file_path}. Use .json, .toml, or .yaml."
            )

        config = cls.from_dict(data)
        if apply_env:
            config.apply_env_overrides(os.environ, prefix=prefix)
        return config

    def apply_env_overrides(
        self,
        env: Mapping[str, str],
        *,
        prefix: str = "AGENT_MANAGER_",
    ) -> None:
        if f"{prefix}PROVIDER" in env:
            self.provider.name = env[f"{prefix}PROVIDER"]
        if f"{prefix}MODEL" in env:
            self.provider.model = env[f"{prefix}MODEL"]
        if f"{prefix}BASE_URL" in env:
            self.provider.base_url = env[f"{prefix}BASE_URL"]
        if f"{prefix}API_KEY_ENV" in env:
            self.provider.api_key_env = env[f"{prefix}API_KEY_ENV"]
        if f"{prefix}PROFILE" in env:
            self.profile = env[f"{prefix}PROFILE"]
        if f"{prefix}SYSTEM_PROMPT" in env:
            self.system_prompt = env[f"{prefix}SYSTEM_PROMPT"]
        if f"{prefix}STATE_DIR" in env:
            self.state_dir = env[f"{prefix}STATE_DIR"]
        if f"{prefix}STATE_BACKEND" in env:
            self.state_backend = env[f"{prefix}STATE_BACKEND"].strip().lower()
        if f"{prefix}STATE_PATH" in env:
            self.state_path = env[f"{prefix}STATE_PATH"]
        if f"{prefix}MAX_STEPS" in env:
            self.runtime.max_steps = _parse_int(env[f"{prefix}MAX_STEPS"], "MAX_STEPS")
        if f"{prefix}TIMEOUT_SECONDS" in env:
            self.runtime.timeout_seconds = _parse_float(
                env[f"{prefix}TIMEOUT_SECONDS"],
                "TIMEOUT_SECONDS",
            )
        if f"{prefix}MAX_CONSECUTIVE_FAILURES" in env:
            self.runtime.max_consecutive_failures = _parse_int(
                env[f"{prefix}MAX_CONSECUTIVE_FAILURES"],
                "MAX_CONSECUTIVE_FAILURES",
            )
        if f"{prefix}MAX_CONTEXT_TOKENS" in env:
            self.runtime.max_context_tokens = _parse_int(
                env[f"{prefix}MAX_CONTEXT_TOKENS"],
                "MAX_CONTEXT_TOKENS",
            )
        if f"{prefix}MAX_OUTPUT_TOKENS" in env:
            self.runtime.max_output_tokens = _parse_int(
                env[f"{prefix}MAX_OUTPUT_TOKENS"],
                "MAX_OUTPUT_TOKENS",
            )
        if f"{prefix}LOG_LEVEL" in env:
            self.logging.level = env[f"{prefix}LOG_LEVEL"]
        if f"{prefix}LOG_JSON" in env:
            self.logging.json_output = _parse_bool(env[f"{prefix}LOG_JSON"])
        if f"{prefix}CONTEXT_HISTORY_WINDOW" in env:
            self.context.history_window = _parse_int(
                env[f"{prefix}CONTEXT_HISTORY_WINDOW"],
                "CONTEXT_HISTORY_WINDOW",
            )
        if f"{prefix}CONTEXT_SUMMARY_TRIGGER_MESSAGES" in env:
            self.context.summary_trigger_messages = _parse_int(
                env[f"{prefix}CONTEXT_SUMMARY_TRIGGER_MESSAGES"],
                "CONTEXT_SUMMARY_TRIGGER_MESSAGES",
            )
        if f"{prefix}CONTEXT_RETRIEVAL_TOP_K" in env:
            self.context.retrieval_top_k = _parse_int(
                env[f"{prefix}CONTEXT_RETRIEVAL_TOP_K"],
                "CONTEXT_RETRIEVAL_TOP_K",
            )
        if f"{prefix}CONTEXT_MAX_MEMORY_FACTS" in env:
            self.context.max_memory_facts = _parse_int(
                env[f"{prefix}CONTEXT_MAX_MEMORY_FACTS"],
                "CONTEXT_MAX_MEMORY_FACTS",
            )
        if f"{prefix}CONTEXT_PRE_CALL_FUNCTIONS" in env:
            self.context.pre_call_functions = _parse_csv_list(
                env[f"{prefix}CONTEXT_PRE_CALL_FUNCTIONS"]
            )
        if f"{prefix}DENIED_TOOLS" in env:
            self.tool_policy.denied_tools = _parse_csv_list(
                env[f"{prefix}DENIED_TOOLS"]
            )
        if f"{prefix}ALLOWED_TOOLS" in env:
            self.tool_policy.allowed_tools = _parse_csv_list(
                env[f"{prefix}ALLOWED_TOOLS"]
            )
        if f"{prefix}DENIED_TAGS" in env:
            self.tool_policy.denied_tags = _parse_csv_list(
                env[f"{prefix}DENIED_TAGS"]
            )
        if f"{prefix}DENIED_PERMISSIONS" in env:
            self.tool_policy.denied_permissions = _parse_csv_list(
                env[f"{prefix}DENIED_PERMISSIONS"]
            )

    def resolved_state_dir(self, base_path: str | Path | None = None) -> Path:
        path = Path(self.state_dir)
        if path.is_absolute():
            return path
        if base_path is None:
            base_path = Path.cwd()
        return Path(base_path) / path

    def resolved_state_path(self, base_path: str | Path | None = None) -> Path:
        if (
            self.state_path == ".agent_manager/state.sqlite3"
            and self.state_dir != ".agent_manager/state"
        ):
            path = Path(self.state_dir) / "checkpoints.sqlite3"
        else:
            path = Path(self.state_path)
        if path.is_absolute():
            return path
        if base_path is None:
            base_path = Path.cwd()
        return Path(base_path) / path

    def resolved_checkpoint_backend(self) -> str:
        backend = self.state_backend.strip().lower()
        if backend not in {"sqlite", "json"}:
            raise ConfigurationError(
                f"Unsupported state backend '{self.state_backend}'. Use 'sqlite' or 'json'."
            )
        return backend


def load_config(path: str | Path | None = None, *, prefix: str = "AGENT_MANAGER_") -> RuntimeConfig:
    """Load configuration from a file or the current environment."""

    if path is None:
        path = os.getenv(f"{prefix}CONFIG")
    if path:
        return RuntimeConfig.from_file(path, prefix=prefix, apply_env=True)
    return RuntimeConfig.from_env(prefix=prefix)
