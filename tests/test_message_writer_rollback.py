from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore


@pytest.mark.asyncio
async def test_cancelled_message_write_waits_for_queued_commit_and_stays_atomic(
    tmp_path,
):
    store = ConversationStore(str(tmp_path / "message-cancel.sqlite3"))
    await store.initialize()
    await store.connection.execute("PRAGMA journal_mode=DELETE")
    await store.connection.execute("PRAGMA busy_timeout=2000")
    blocker = await aiosqlite.connect(store.db_path)
    await blocker.execute("PRAGMA journal_mode=DELETE")
    await blocker.execute("BEGIN")
    async with blocker.execute("SELECT COUNT(*) FROM messages") as cursor:
        await cursor.fetchone()
    commit_started = asyncio.Event()
    real_commit = store.connection.commit

    async def queued_commit():
        driver_commit = asyncio.create_task(real_commit())
        # Let aiosqlite enqueue the real sqlite3.Connection.commit call before
        # signaling the test to cancel the writer.
        await asyncio.sleep(0)
        commit_started.set()
        await driver_commit

    store.connection.commit = queued_commit
    message = Message(
        id=0,
        session_id="synthetic:FriendMessage:user",
        role="user",
        content="synthetic original",
        sender_id="synthetic-user",
        platform="synthetic-adapter",
        timestamp=1_700_000_000,
    )
    writer = asyncio.create_task(store.add_message(message))
    await commit_started.wait()
    writer.cancel()
    await asyncio.sleep(0)
    assert not writer.done()
    await blocker.rollback()
    with pytest.raises(asyncio.CancelledError):
        await writer

    # The driver commit had already been queued. The caller observes cancellation
    # only after it finishes, leaving a complete message/session transaction.
    await store.connection.execute("SELECT 1")
    async with store.connection.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (message.session_id,)
    ) as cursor:
        assert (await cursor.fetchone())[0] == 1
    async with store.connection.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id=?", (message.session_id,)
    ) as cursor:
        assert (await cursor.fetchone())[0] == 1
    store.connection.commit = real_commit
    await blocker.close()
    await store.close()


@pytest.mark.asyncio
async def test_cancelled_message_write_before_insert_rolls_back_and_cannot_write_late(
    tmp_path,
):
    store = ConversationStore(str(tmp_path / "message-precommit-cancel.sqlite3"))
    await store.initialize()
    insert_started = asyncio.Event()
    release_insert = asyncio.Event()
    real_execute = store.connection.execute

    async def paused_execute(sql, *args, **kwargs):
        if "INSERT INTO messages" in str(sql):
            insert_started.set()
            await release_insert.wait()
        return await real_execute(sql, *args, **kwargs)

    store.connection.execute = paused_execute
    message = Message(
        id=0,
        session_id="synthetic:FriendMessage:user",
        role="user",
        content="synthetic original",
        sender_id="synthetic-user",
        platform="synthetic-adapter",
        timestamp=1_700_000_000,
    )
    writer = asyncio.create_task(store.add_message(message))
    await insert_started.wait()
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer
    release_insert.set()
    store.connection.execute = real_execute
    async with store.connection.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (message.session_id,)
    ) as cursor:
        assert (await cursor.fetchone())[0] == 0
    async with store.connection.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id=?", (message.session_id,)
    ) as cursor:
        assert (await cursor.fetchone())[0] == 0
    await store.close()
