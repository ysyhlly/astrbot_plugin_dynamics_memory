"""Shared write-side fence for source status changes and stale pruning."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class SourceStateGuard:
    """Serialize status transitions with graph rebuilds and vector pruning.

    Lock order for operations that need more than this guard is:

    1. ``SourceStateGuard``
    2. ``GraphMemoryManager._rebuild_gate``
    3. FAISS persister ``_mutation_lock``

    Reentrant acquisition by the owning asyncio task lets a status operation
    call a guarded helper without releasing the fence. Other tasks remain
    blocked until the outer operation commits or rolls back.

    The global lock is intentionally simple: archived/unarchived edits and
    maintenance are rare, while a per-source lock registry would need cleanup
    and deterministic multi-source acquisition to avoid deadlocks.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task | None = None
        self._depth = 0

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("source-state guard requires an asyncio task")
        if self._owner is task:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return

        async with self._lock:
            self._owner = task
            self._depth = 1
            try:
                yield
            finally:
                self._depth = 0
                self._owner = None


__all__ = ["SourceStateGuard"]
