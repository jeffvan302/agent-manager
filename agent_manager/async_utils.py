"""Helpers for bridging sync and async runtime APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def run_sync(awaitable: Awaitable[T]) -> T:
    """Run an awaitable from sync code when no event loop is active."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("A running event loop is active. Use the async API instead.")

