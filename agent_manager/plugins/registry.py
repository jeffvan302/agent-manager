"""Plugin registry."""

from __future__ import annotations

from agent_manager.plugins.base import Plugin


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        self._plugins[plugin.name] = plugin

    def apply_all(self, target: object) -> None:
        for plugin in self._plugins.values():
            plugin.register(target)

