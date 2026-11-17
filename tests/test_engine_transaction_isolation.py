"""Real SQLite regressions for the engine's reader/writer ownership boundary."""
import asyncio

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.tests.test_memory_engine import _FakeFaissDB
from astrbot_plugin_dynamics_memory.storage.sqlite_connection import open_connection
from astrbot_plugin_dynamics_memory.storage.sqlite_write import settle


@pytest.mark.asyncio
async def test_journal_write_does_not_upgrade_a_stale_reader_snapshot(tmp_path):
    engine = MemoryEngine(str(tmp_path / 'memory.db'), _FakeFaissDB(),
                          config={'write_op_repair_enabled': False})
    await engine.initialize()
    cursor = None
    try:
        first = await engine._start_write_op('add', {'sample': 1})
        await engine._start_write_op('add', {'sample': 2})
        # A concurrent recall pauses between execute() and fetchall(). Another
        # store commits a document/index change before the journal is advanced.
        cursor = await engine.db_connection.execute('SELECT id FROM memory_write_ops ORDER BY id')
        assert (await cursor.fetchone())[0] == first
        async with aiosqlite.connect(engine.db_path) as other:
            await other.execute('UPDATE memory_write_ops SET updated_at=updated_at+1 WHERE id=?', (first,))
            await other.commit()
        await engine._advance_write_op(first, 'document_indexed', memory_id=42)
        await cursor.close()
        cursor = None
        async with aiosqlite.connect(engine.db_path) as check:
            async with check.execute('SELECT step,memory_id FROM memory_write_ops WHERE id=?', (first,)) as row:
                assert await row.fetchone() == ('document_indexed', 42)
    finally:
        if cursor is not None:
            await cursor.close()
        await engine.close()


@pytest.mark.asyncio
async def test_engine_reader_cannot_see_uncommitted_publication(tmp_path):
    engine = MemoryEngine(str(tmp_path / 'memory.db'), _FakeFaissDB(),
                          config={'write_op_repair_enabled': False})
    await engine.initialize()
    entered = asyncio.Event()
    release = asyncio.Event()
    try:
        op = await engine._start_write_op('add', {})
        async def publish():
            async with engine._write_transaction() as db:
                await db.execute("UPDATE memory_write_ops SET status='completed' WHERE id=?", (op,))
                entered.set()
                await release.wait()
        task = asyncio.create_task(publish())
        await entered.wait()
        try:
            async with engine.db_connection.execute('SELECT status FROM memory_write_ops WHERE id=?', (op,)) as cursor:
                assert (await cursor.fetchone())[0] == 'pending'
        finally:
            release.set()
            await task
        async with engine.db_connection.execute('SELECT status FROM memory_write_ops WHERE id=?', (op,)) as cursor:
            assert (await cursor.fetchone())[0] == 'completed'
    finally:
        release.set()
        await engine.close()


@pytest.mark.asyncio
async def test_engine_writer_rolls_back_after_cancellation(tmp_path):
    engine = MemoryEngine(str(tmp_path / 'memory.db'), _FakeFaissDB(),
                          config={'write_op_repair_enabled': False})
    await engine.initialize()
    entered = asyncio.Event()
    try:
        op = await engine._start_write_op('add', {})
        async def pending():
            async with engine._write_transaction() as db:
                await db.execute("UPDATE memory_write_ops SET step='not_committed' WHERE id=?", (op,))
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(pending())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await engine._advance_write_op(op, 'after_cancel')
        async with aiosqlite.connect(engine.db_path) as check:
            async with check.execute('SELECT step FROM memory_write_ops WHERE id=?', (op,)) as row:
                assert await row.fetchone() == ('after_cancel',)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_open_finishes_journal_mode_cursor_before_another_writer(tmp_path):
    path = str(tmp_path / 'journal.db')
    first = await open_connection(path, apply_journal_mode=True)
    try:
        async with aiosqlite.connect(path, timeout=0.1) as second:
            await second.execute('CREATE TABLE sample(id INTEGER)')
            await second.commit()
    finally:
        await first.close()


@pytest.mark.asyncio
async def test_repeated_cancel_waits_for_sqlite_cleanup():
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    async def cleanup():
        entered.set()
        await release.wait()
        completed.set()
    task = asyncio.create_task(settle(cleanup()))
    try:
        await entered.wait()
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
