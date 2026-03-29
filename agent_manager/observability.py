"""Structured observability for the agent_manager runtime.

Provides JSON-structured logging, event emission helpers, secret redaction,
and timing utilities for provider calls, tool execution, and state operations.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


# ------------------------------------------------------------------
# Secret redaction
# ------------------------------------------------------------------

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(sk-[A-Za-z0-9]{20,})", re.ASCII),           # OpenAI keys
    re.compile(r"(sk-ant-[A-Za-z0-9\-]{20,})", re.ASCII),     # Anthropic keys
    re.compile(r"(AIza[A-Za-z0-9_\-]{30,})", re.ASCII),       # Google API keys
    re.compile(r"(Bearer\s+[A-Za-z0-9\-._~+/]+=*)", re.ASCII),  # Bearer tokens
    re.compile(r"(x-api-key:\s*\S+)", re.IGNORECASE),          # x-api-key header values
]

_SENSITIVE_KEYS = frozenset({
    "api_key", "api_key_env", "authorization", "x-api-key", "token",
    "secret", "password", "credentials", "private_key",
})


def redact_secrets(value: Any, *, depth: int = 0) -> Any:
    """Recursively redact secrets from dicts, lists, and strings."""
    if depth > 10:
        return value
    if isinstance(value, str):
        result = value
        for pattern in _SECRET_PATTERNS:
            result = pattern.sub("***REDACTED***", result)
        return result
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = redact_secrets(val, depth=depth + 1)
        return redacted
    if isinstance(value, (list, tuple)):
        return type(value)(redact_secrets(item, depth=depth + 1) for item in value)
    return value


# ------------------------------------------------------------------
# Sensitive output masking
# ------------------------------------------------------------------

_MASK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # emails
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{16}\b"),              # credit card numbers
]


def mask_sensitive_output(text: str) -> str:
    """Mask common PII patterns in tool output."""
    result = text
    for pattern in _MASK_PATTERNS:
        result = pattern.sub("***MASKED***", result)
    return result


# ------------------------------------------------------------------
# JSON log formatter
# ------------------------------------------------------------------

class JsonLogFormatter(logging.Formatter):
    """Emit compact JSON log lines with secret redaction."""

    def __init__(self, *, redact: bool = True) -> None:
        super().__init__()
        self._redact = redact

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "details"):
            details = record.details
            if self._redact:
                details = redact_secrets(details)
            payload["details"] = details
        if hasattr(record, "duration_ms"):
            payload["duration_ms"] = record.duration_ms
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, default=str)


# ------------------------------------------------------------------
# Logger setup
# ------------------------------------------------------------------

def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    *,
    redact_secrets_in_logs: bool = True,
) -> logging.Logger:
    """Configure the package root logger."""
    logger = logging.getLogger("agent_manager")
    logger.setLevel(level.upper())
    logger.propagate = False

    formatter: logging.Formatter
    if json_output:
        formatter = JsonLogFormatter(redact=redact_secrets_in_logs)
    else:
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(formatter)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger for package modules."""
    return logging.getLogger(f"agent_manager.{name}")


# ------------------------------------------------------------------
# Structured event emitters
# ------------------------------------------------------------------

class ObservabilityEmitter:
    """Centralised event logging for the runtime.

    All structured events (provider calls, tool calls, checkpoints, etc.)
    should be emitted through this class so that the observability contract
    stays consistent regardless of which module triggers the event.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or get_logger("observability")

    def _emit(self, level: int, message: str, *, event: str, details: dict[str, Any] | None = None, duration_ms: float | None = None) -> None:
        extra: dict[str, Any] = {"event": event}
        if details:
            extra["details"] = details
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        self._log.log(level, message, extra=extra)

    # ---- Provider events ----

    def provider_request(
        self,
        *,
        provider: str,
        model: str,
        message_count: int,
        tool_count: int = 0,
        token_estimate: int | None = None,
    ) -> None:
        self._emit(logging.INFO, f"provider request to {provider}/{model}", event="provider.request", details={
            "provider": provider,
            "model": model,
            "message_count": message_count,
            "tool_count": tool_count,
            "token_estimate": token_estimate,
        })

    def provider_response(
        self,
        *,
        provider: str,
        model: str,
        stop_reason: str | None,
        usage: dict[str, Any] | None,
        duration_ms: float,
        has_tool_calls: bool = False,
    ) -> None:
        self._emit(logging.INFO, f"provider response from {provider}/{model}", event="provider.response", details={
            "provider": provider,
            "model": model,
            "stop_reason": stop_reason,
            "usage": usage,
            "has_tool_calls": has_tool_calls,
        }, duration_ms=duration_ms)

    def provider_error(
        self,
        *,
        provider: str,
        error: str,
        retryable: bool = False,
        attempt: int = 1,
    ) -> None:
        self._emit(logging.WARNING, f"provider error ({provider}): {error}", event="provider.error", details={
            "provider": provider,
            "error": error,
            "retryable": retryable,
            "attempt": attempt,
        })

    def provider_retry(
        self,
        *,
        provider: str,
        attempt: int,
        backoff_seconds: float,
        reason: str = "",
    ) -> None:
        self._emit(logging.INFO, f"retrying {provider} (attempt {attempt})", event="provider.retry", details={
            "provider": provider,
            "attempt": attempt,
            "backoff_seconds": backoff_seconds,
            "reason": reason,
        })

    # ---- Tool events ----

    def tool_call(
        self,
        *,
        tool_name: str,
        tool_call_id: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        safe_args = redact_secrets(arguments) if arguments else {}
        self._emit(logging.INFO, f"tool call: {tool_name}", event="tool.call", details={
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "arguments": safe_args,
        })

    def tool_result(
        self,
        *,
        tool_name: str,
        ok: bool,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        self._emit(
            logging.INFO if ok else logging.WARNING,
            f"tool result: {tool_name} ({'ok' if ok else 'error'})",
            event="tool.result",
            details={
                "tool_name": tool_name,
                "ok": ok,
                "error": error,
            },
            duration_ms=duration_ms,
        )

    def tool_policy_violation(
        self,
        *,
        tool_name: str,
        reason: str,
    ) -> None:
        self._emit(logging.WARNING, f"tool policy violation: {tool_name}", event="tool.policy_violation", details={
            "tool_name": tool_name,
            "reason": reason,
        })

    # ---- Context events ----

    def context_assembled(
        self,
        *,
        section_count: int,
        token_estimate: int,
        dropped_sections: list[str] | None = None,
    ) -> None:
        self._emit(logging.DEBUG, f"context assembled: {section_count} sections, ~{token_estimate} tokens", event="context.assembled", details={
            "section_count": section_count,
            "token_estimate": token_estimate,
            "dropped_sections": dropped_sections or [],
        })

    def summarization(
        self,
        *,
        input_messages: int,
        output_length: int,
        duration_ms: float,
    ) -> None:
        self._emit(logging.INFO, f"summarized {input_messages} messages", event="context.summarization", details={
            "input_messages": input_messages,
            "output_length": output_length,
        }, duration_ms=duration_ms)

    def retrieval(
        self,
        *,
        query: str,
        results_count: int,
        duration_ms: float,
    ) -> None:
        self._emit(logging.DEBUG, f"retrieval: {results_count} results", event="context.retrieval", details={
            "query_preview": query[:100],
            "results_count": results_count,
        }, duration_ms=duration_ms)

    # ---- State events ----

    def checkpoint_save(
        self,
        *,
        task_id: str,
        step: int,
        duration_ms: float,
    ) -> None:
        self._emit(logging.INFO, f"checkpoint saved: {task_id} (step {step})", event="state.checkpoint_save", details={
            "task_id": task_id,
            "step": step,
        }, duration_ms=duration_ms)

    def checkpoint_load(
        self,
        *,
        task_id: str,
        step: int,
    ) -> None:
        self._emit(logging.INFO, f"checkpoint loaded: {task_id} (step {step})", event="state.checkpoint_load", details={
            "task_id": task_id,
            "step": step,
        })

    # ---- Loop events ----

    def loop_step(
        self,
        *,
        task_id: str,
        step: int,
        status: str,
    ) -> None:
        self._emit(logging.DEBUG, f"loop step {step}: {status}", event="loop.step", details={
            "task_id": task_id,
            "step": step,
            "status": status,
        })

    def loop_complete(
        self,
        *,
        task_id: str,
        stop_reason: str,
        total_steps: int,
        duration_ms: float,
    ) -> None:
        self._emit(logging.INFO, f"loop complete: {stop_reason} after {total_steps} steps", event="loop.complete", details={
            "task_id": task_id,
            "stop_reason": stop_reason,
            "total_steps": total_steps,
        }, duration_ms=duration_ms)


# ------------------------------------------------------------------
# Timing context manager
# ------------------------------------------------------------------

@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """Context manager that records elapsed wall-clock time in ms.

    Usage::

        with timed() as t:
            do_work()
        print(t["duration_ms"])
    """
    result: dict[str, float] = {"duration_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["duration_ms"] = (time.perf_counter() - start) * 1000.0


# Singleton emitter for convenience.
emitter = ObservabilityEmitter()
