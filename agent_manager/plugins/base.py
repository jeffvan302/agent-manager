"""Base plugin contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_manager.errors import ConfigurationError


class Plugin(ABC):
    """Register optional runtime extensions without changing the core."""

    name = "plugin"
    description = ""

    def is_available(self) -> bool:
        return True

    def assert_available(self) -> None:
        if not self.is_available():
            raise ConfigurationError(f"Plugin '{self.name}' is not available.")

    @abstractmethod
    def register(self, target: Any) -> None:
        """Attach the plugin to a session or application object."""
