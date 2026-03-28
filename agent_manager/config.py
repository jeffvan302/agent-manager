"""Configuration loading for the agent_manager runtime."""

from __future__ import annotations

import json
import os
import tomllib
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


@dataclass(slots=True)
class RuntimeLimits:
    max_steps: int = 6
    timeout_seconds: int = 300
    max_consecutive_failures: int = 3
    max_context_tokens: int = 8192
    max_output_tokens: int = 1024

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeLimits":
        data = data or {}
        return cls(
            max_steps=int(data.get("max_steps", 6)),
            timeout_seconds=int(data.get("timeout_seconds", 300)),
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
class RuntimeConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    runtime: RuntimeLimits = field(default_factory=RuntimeLimits)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profile: str = "readonly"
    system_prompt: str = "You are a helpful local-first agent runtime."
    state_dir: str = ".agent_manager/state"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeConfig":
        data = data or {}
        return cls(
            provider=ProviderConfig.from_dict(data.get("provider")),
            runtime=RuntimeLimits.from_dict(data.get("runtime")),
            logging=LoggingConfig.from_dict(data.get("logging")),
            profile=str(data.get("profile", "readonly")),
            system_prompt=str(
                data.get("system_prompt", "You are a helpful local-first agent runtime.")
            ),
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
        else:
            raise ConfigurationError(
                f"Unsupported config format for {file_path}. Use .json or .toml."
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
        if f"{prefix}MAX_STEPS" in env:
            self.runtime.max_steps = _parse_int(env[f"{prefix}MAX_STEPS"], "MAX_STEPS")
        if f"{prefix}TIMEOUT_SECONDS" in env:
            self.runtime.timeout_seconds = _parse_int(
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

    def resolved_state_dir(self, base_path: str | Path | None = None) -> Path:
        path = Path(self.state_dir)
        if path.is_absolute():
            return path
        if base_path is None:
            base_path = Path.cwd()
        return Path(base_path) / path


def load_config(path: str | Path | None = None, *, prefix: str = "AGENT_MANAGER_") -> RuntimeConfig:
    """Load configuration from a file or the current environment."""

    if path is None:
        path = os.getenv(f"{prefix}CONFIG")
    if path:
        return RuntimeConfig.from_file(path, prefix=prefix, apply_env=True)
    return RuntimeConfig.from_env(prefix=prefix)
