"""Basic logging helpers used by the runtime."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Emit compact JSON log lines for automation and trace capture."""

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
            payload["details"] = record.details
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str = "INFO", json_output: bool = False) -> logging.Logger:
    """Configure the package root logger."""

    logger = logging.getLogger("agent_manager")
    logger.setLevel(level.upper())
    logger.propagate = False

    formatter: logging.Formatter
    if json_output:
        formatter = JsonLogFormatter()
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

