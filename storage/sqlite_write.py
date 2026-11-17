"""Short, serialized plugin-owned SQLite writes; no network call inside a transaction."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from weakref import WeakValueDictionary
import aiosqlite
from .sqlite_connection import BUSY_TIMEOUT_MS, open_connection


_PATH_WRITE_LOCKS: WeakValueDictionary[tuple[int, str], asyncio.Lock] = (
    WeakValueDictionary()
)


def _get_write_lock(owner, path, separate: bool) -> asyncio.Lock:
    if separate:
        loop_id = id(asyncio.get_running_loop())
        key = (loop_id, str(Path(path).resolve()))
        lock = _PATH_WRITE_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _PATH_WRITE_LOCKS[key] = lock
        return lock
    lock = getattr(owner, "_sqlite_mutation_lock", None)
    if lock is None:
        lock = owner._sqlite_mutation_lock = asyncio.Lock()
    return lock


async def settle(awaitable):
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A second cancellation must not abandon a queued commit, rollback or
        # close while SQLite's worker still owns the transaction/connection.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


@asynccontextmanager
async def write_transaction(owner):
    path = getattr(owner, "db_path", None)
    separate = isinstance(path, (str, Path)) and str(path) not in ("", ":memory:")
    lock = _get_write_lock(owner, path, separate)
    async with lock:
        # open_connection drains and closes an opening worker on cancellation.
        # A private writer never inherits a concurrent reader's WAL snapshot.
        db = await open_connection(str(path)) if separate else owner.db_connection
        if db is None:
            raise RuntimeError("Memory database is unavailable")
        db.row_factory = aiosqlite.Row
        try:
            await db.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            for attempt in range(4):
                try:
                    if separate or not getattr(db, "in_transaction", False):
                        await db.execute("BEGIN IMMEDIATE")
                    break
                except aiosqlite.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 3:
                        raise
                    await asyncio.sleep(0.05 * 2**attempt)
            try:
                yield db
                await settle(db.commit())
            except BaseException:
                await settle(db.rollback())
                raise
        finally:
            if separate:
                await settle(db.close())
