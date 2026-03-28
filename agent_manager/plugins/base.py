"""Base plugin contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Plugin(ABC):
    """Register optional runtime extensions without changing the core."""

    name = "plugin"

    @abstractmethod
    def register(self, target: Any) -> None:
        """Attach the plugin to a session or application object."""

