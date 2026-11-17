"""Keep local source protection while adopting bounded/anchored reflection."""

import asyncio
import json
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.managers.memory_engine_write_ops import (
    MemoryEngineWriteOpsMixin,
)
from tests.event_handler_support import _make_real_handler
from tests.test_reflection_backlog import (
    _seed_messages,
    _configure_processor,
    _build_plan,
    _run_scheduled,
)


@pytest.mark.asyncio
async def test_trim_during_model_call_commits_anchor_then_continues_remaining_input(
    tmp_path,
):
    handler, manager, store = await _make_real_handler(tmp_path)
    reflection = handler._memory_reflection
    reflection.config_manager = ConfigManager(
        {"reflection_engine": {"max_messages_per_segment": 4}}
    )
    sid = "test:group:trim-inflight"
    await _seed_messages(manager, sid, [1000.0 + i for i in range(10)])
    await manager.update_session_metadata(sid, "last_summarized_index", 4)
    calls = []

    async def summarize(**kwargs):
        calls.append(len(kwargs["messages"]))
        if len(calls) == 1:
            assert await manager.trim_session_messages(sid, 4) == 4
        return "summary", {"topics": ["puzzle"]}, 0.9

    _configure_processor(handler, side_effect=summarize)
    try:
        plan = await _build_plan(handler, sid, 10)
        await _run_scheduled(handler, plan)
        assert calls == [4, 2]
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 6
        assert await store.get_message_count(sid) == 6
        for call in handler.memory_engine.add_memory.call_args_list:
            assert callable(call.kwargs["source_validator"])
            assert callable(call.kwargs["source_publish_lease"])
            window = call.kwargs["metadata"]["source_window"]
            assert (
                window["end_message_id"] > 0
                and len(window["snapshot_fingerprint"]) == 64
            )
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["edit", "delete", "rollback"])
async def test_invalidated_snapshot_cannot_commit_or_advance(tmp_path, damage):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:source-change"
    await _seed_messages(manager, sid, [1000.0 + i for i in range(4)])

    async def summarize(**kwargs):
        mid = kwargs["messages"][0].id
        if damage == "edit":
            await store.connection.execute(
                "UPDATE messages SET content='edited' WHERE id=?", (mid,)
            )
            await store.connection.commit()
        elif damage == "delete":
            await store.connection.execute("DELETE FROM messages WHERE id=?", (mid,))
            await store.connection.commit()
        else:
            await manager.bump_session_revision(sid)
        return "summary", {"topics": ["puzzle"]}, 0.9

    _configure_processor(handler, side_effect=summarize)
    try:
        plan = await _build_plan(handler, sid, 4)
        await _run_scheduled(handler, plan)
        handler.memory_engine.add_memory.assert_not_awaited()
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 0
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["policy", "provider_terminal", "retry_exhausted"])
async def test_paused_input_keeps_cursor_and_original_messages(tmp_path, stage):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:paused"
    await _seed_messages(manager, sid, [1000.0 + i for i in range(8)])
    pending = {
        "start_index": 0,
        "end_index": 8,
        "retry_count": 3,
        "failure_stage": stage,
        "retry_after": time.time() + 900,
    }
    await manager.update_session_metadata(sid, "pending_summary", pending)
    try:
        plan = await handler._memory_reflection._build_reflection_plan_for_cursor(
            sid, 8, 0, pending, 0, "persona", sid
        )
        assert plan is None
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 0
        assert await store.get_message_count(sid) == 8
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_timestamp_ties_and_concurrent_append_do_not_skip_later_rows(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:timestamps"
    await _seed_messages(manager, sid, [1000.0] * 6)
    try:
        start, rows = await manager.get_summary_window(sid, 4)
        assert start == 0 and len(rows) == 4
        await manager.add_message(
            session_id=sid,
            role="user",
            content="later",
            sender_id="u1",
            timestamp=1000.0,
        )
        assert await manager.commit_summary_cursor(sid, rows[-1], 0)
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 4
        _, remaining = await manager.get_summary_window(sid, 4)
        assert len(remaining) == 3
        assert remaining[-1].content == "later"
        await manager.bump_session_revision(sid)
        assert not await manager.commit_summary_cursor(sid, remaining[-1], 0)
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 4
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_missing_or_foreign_anchor_never_advances_cursor(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:anchor-boundary"
    await _seed_messages(manager, sid, [1000.0, 1001.0])
    try:
        _, rows = await manager.get_summary_window(sid, 4)
        assert not await manager.commit_summary_cursor("other", rows[-1], 0)
        await store.connection.execute(
            "DELETE FROM messages WHERE id=?", (rows[-1].id,)
        )
        await store.connection.commit()
        assert not await manager.commit_summary_cursor(sid, rows[-1], 0)
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 0
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_cancellation_after_durable_cursor_does_not_delete_committed_memory(
    tmp_path,
):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:durable-cursor"
    await _seed_messages(manager, sid, [1000.0 + i for i in range(4)])
    _configure_processor(handler)
    handler.memory_engine.delete_memory = AsyncMock(return_value=1)
    original = store._commit_write_transaction

    async def commit_then_cancel():
        await original()
        raise asyncio.CancelledError()

    store._commit_write_transaction = commit_then_cancel
    try:
        plan = await _build_plan(handler, sid, 4)
        with pytest.raises(asyncio.CancelledError):
            await handler._memory_reflection._storage_task(
                sid, plan.history_messages, "persona", 0, 4, 0, expected_revision=0
            )
        handler.memory_engine.delete_memory.assert_not_awaited()
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 4
    finally:
        store._commit_write_transaction = original
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_old_offset_is_not_enough_to_reuse_another_summary(tmp_path):
    async with aiosqlite.connect(tmp_path / "dedup.db") as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            "CREATE TABLE documents(id INTEGER PRIMARY KEY, metadata TEXT)"
        )
        await db.execute("CREATE TABLE memory_sources(memory_id INTEGER PRIMARY KEY)")
        await db.execute(
            "INSERT INTO documents VALUES(1,?)",
            (
                json.dumps(
                    {
                        "source_session_id": "room",
                        "source_window": {
                            "start_index": 4,
                            "end_index": 8,
                            "snapshot_fingerprint": "original",
                        },
                    }
                ),
            ),
        )
        engine = NS(db_connection=db)
        await MemoryEngineWriteOpsMixin._create_write_ops_table(engine)
        plan = await (
            await db.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM documents WHERE json_valid(metadata) AND json_extract(metadata,'$.source_session_id')=? AND json_extract(metadata,'$.source_window.snapshot_fingerprint')=?",
                ("room", "original"),
            )
        ).fetchall()
        assert any("idx_summary_snapshot_v1" in row[3] for row in plan)
        assert (
            await MemoryEngineWriteOpsMixin.stored_summary_window(
                engine, "room", 4, 8, snapshot_fingerprint="changed-input"
            )
            is None
        )
        assert (
            await MemoryEngineWriteOpsMixin.stored_summary_window(
                engine, "room", 0, 4, snapshot_fingerprint="original"
            )
            == 1
        )
        assert (
            await MemoryEngineWriteOpsMixin.stored_summary_window(
                engine, "other", 0, 4, snapshot_fingerprint="original"
            )
            is None
        )


@pytest.mark.asyncio
async def test_verified_stored_snapshot_advances_without_another_model_call(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:stored-snapshot"
    await _seed_messages(manager, sid, [1000.0 + i for i in range(4)])
    _configure_processor(handler)
    handler.memory_engine.stored_summary_window = AsyncMock(return_value=123)
    try:
        plan = await _build_plan(handler, sid, 4)
        await _run_scheduled(handler, plan)
        handler.memory_processor.process_conversation.assert_not_awaited()
        handler.memory_engine.add_memory.assert_not_awaited()
        fingerprint = handler.memory_engine.stored_summary_window.call_args.kwargs[
            "snapshot_fingerprint"
        ]
        assert len(fingerprint) == 64
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 4
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_transient_retry_resumes_same_input_after_cooldown(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:resume"
    await _seed_messages(manager, sid, [1000.0 + i for i in range(8)])
    pending = {
        "start_index": 0,
        "end_index": 8,
        "retry_count": 3,
        "failure_stage": "retry_exhausted",
        "retry_after": time.time() - 1,
    }
    await manager.update_session_metadata(sid, "pending_summary", pending)
    try:
        handler._memory_reflection.config_manager = ConfigManager(
            {"reflection_engine": {"max_messages_per_segment": 4}}
        )
        plan = await handler._memory_reflection._build_reflection_plan_for_cursor(
            sid, 8, 0, pending, 0, "persona", sid
        )
        assert (plan.start_index, plan.end_index, plan.retry_count) == (0, 8, 0)
        assert await store.get_message_count(sid) == 8
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 0
    finally:
        await handler.shutdown()
        await store.close()


def test_segment_setting_is_preserved_by_runtime_config():
    assert ConfigManager({}).get("reflection_engine.max_messages_per_segment") == 120
    assert (
        ConfigManager({"reflection_engine": {"max_messages_per_segment": 240}}).get(
            "reflection_engine.max_messages_per_segment"
        )
        == 240
    )
