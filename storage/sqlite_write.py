"""Short, serialized plugin-owned SQLite writes; no network call inside a transaction."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import aiosqlite
from .sqlite_connection import open_connection


async def settle(awaitable):
    task=asyncio.ensure_future(awaitable)
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
    lock=getattr(owner,'_sqlite_mutation_lock',None)
    if lock is None:
        lock=owner._sqlite_mutation_lock=asyncio.Lock()
    async with lock:
        path=getattr(owner,'db_path',None)
        separate=isinstance(path,(str,Path)) and str(path) not in ('',':memory:')
        # open_connection drains and closes an opening worker on cancellation.
        # A private writer never inherits a concurrent reader's WAL snapshot.
        db=await open_connection(str(path)) if separate else owner.db_connection
        if db is None:
            raise RuntimeError('Memory database is unavailable')
        db.row_factory=aiosqlite.Row
        try:
            await db.execute('PRAGMA busy_timeout=1000')
            for attempt in range(4):
                try:
                    if separate or not getattr(db,'in_transaction',False):
                        await db.execute('BEGIN IMMEDIATE')
                    break
                except aiosqlite.OperationalError as exc:
                    if 'locked' not in str(exc).lower() or attempt==3:
                        raise
                    await asyncio.sleep(.05*2**attempt)
            try:
                yield db
                await settle(db.commit())
            except BaseException:
                await settle(db.rollback())
                raise
        finally:
            if separate:
                await settle(db.close())
