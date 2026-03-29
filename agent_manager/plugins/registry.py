"""Plugin registry."""

from __future__ import annotations

from collections.abc import Iterable

from agent_manager.plugins.base import Plugin


class PluginRegistry:
    def __init__(self, plugins: Iterable[Plugin] | None = None) -> None:
        self._plugins: dict[str, Plugin] = {}
        for plugin in plugins or []:
            self.register(plugin)

    def register(self, plugin: Plugin) -> None:
        self._plugins[plugin.name] = plugin

    def register_many(self, plugins: Iterable[Plugin]) -> None:
        for plugin in plugins:
            self.register(plugin)

    def get(self, name: str) -> Plugin:
        return self._plugins[name]

    def names(self) -> list[str]:
        return sorted(self._plugins.keys())

    def apply(self, name: str, target: object) -> None:
        plugin = self.get(name)
        plugin.assert_available()
        plugin.register(target)

    def apply_all(self, target: object) -> None:
        for plugin in self._plugins.values():
            plugin.assert_available()
            plugin.register(target)
