"""#385: bounded reflection input, convergent retention and serialized writes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.managers.consolidation_manager import (
    MemoryConsolidationManager,
)
from tests.event_handler_support import _make_real_handler
from tests.test_reflection_backlog import (
    _seed_messages,
    _configure_processor,
    _build_plan,
    _run_scheduled,
)


@pytest.mark.asyncio
async def test_long_same_day_backlog_is_consumed_in_bounded_chunks(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    cfg = ConfigManager(
        {
            "reflection_engine": {
                "max_messages_per_segment": 4,
                "max_segments_per_run": 10,
            },
            "filtering_settings": {"use_persona_filtering": False},
        }
    )
    handler._memory_reflection.config_manager = cfg
    sid = "test:group:bounded"
    await _seed_messages(manager, sid, [1_780_000_000.0 + i for i in range(10)])
    _configure_processor(handler)
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    try:
        plan = await _build_plan(handler, sid, 10)
        assert plan.end_index == 4
        await _run_scheduled(handler, plan)
        assert [
            len(call.kwargs["messages"])
            for call in handler.memory_processor.process_conversation.await_args_list
        ] == [4, 4, 2]
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 10
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [0, 10])
async def test_message_cap_preserves_unsummarized_overflow(tmp_path, cursor):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:retention"
    await _seed_messages(manager, sid, [1_780_000_000.0 + i for i in range(160)])
    await manager.update_session_metadata(sid, "last_summarized_index", cursor)
    revision = await manager.get_session_revision(sid)
    try:
        await handler._message_utils.enforce_message_limit(sid)
        assert await store.get_message_count(sid) == 160 - cursor
        assert await manager.get_session_metadata(sid, "last_summarized_index", -1) == 0
        assert await manager.get_session_revision(sid) == revision
        messages = await manager.get_messages_range(sid, 0, 160 - cursor)
        assert [message.content for message in messages] == [
            f"message-{i}" for i in range(cursor, 160)
        ]
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_consolidation_waits_for_reflection_writes_but_not_its_llm(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    engine = handler.memory_engine
    engine.archive_memories = AsyncMock(return_value=2)
    cfg = ConfigManager({"memory_consolidation": {"enabled": True}})
    merger = Mock()
    merger.merge_memories = AsyncMock(
        return_value={
            "summary": "merged",
            "key_facts": [],
            "topics": [],
            "importance": 0.5,
        }
    )
    consolidation = MemoryConsolidationManager(engine, merger, cfg)
    handler._memory_reflection.consolidation_manager = consolidation
    group = [
        {"id": 1, "metadata": {"session_id": "test:group:a"}},
        {"id": 2, "metadata": {"session_id": "test:group:a"}},
    ]
    try:
        async with consolidation.write_lock:
            task = asyncio.create_task(consolidation._consolidate_group(group, {}))
            await asyncio.sleep(0)
            assert merger.merge_memories.await_count == 1
            engine.add_memory.assert_not_awaited()
        await task
        engine.add_memory.assert_awaited_once()
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_consolidation_is_deferred_until_reflection_backlog_finishes(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:deferred"
    await _seed_messages(manager, sid, [1_780_000_000.0 + i for i in range(4)])
    _configure_processor(handler)
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    consolidation = SimpleNamespace(write_lock=asyncio.Lock(), maybe_run=AsyncMock())
    handler._memory_reflection.consolidation_manager = consolidation
    try:
        plan = await _build_plan(handler, sid, 4)
        consolidation.maybe_run.assert_not_awaited()
        await _run_scheduled(handler, plan)
        await asyncio.gather(*list(handler._storage_tasks))
        consolidation.maybe_run.assert_awaited_once_with("reflection")
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 4
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cursor, deleted, force, committed, expected",
    [(4, 4, False, True, 2), (0, 6, True, False, 0)],
)
async def test_partial_summary_requires_its_message_anchor_after_head_trim(
    tmp_path, cursor, deleted, force, committed, expected
):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:anchor"
    await _seed_messages(manager, sid, [1_780_000_000.0 + i for i in range(10)])
    await manager.update_session_metadata(sid, "last_summarized_index", cursor)
    snapshot = await manager.get_messages_range(sid, cursor, 6)
    revision = await manager.get_session_revision(sid)
    try:
        assert await manager.trim_session_messages(sid, deleted, delete_unsummarized=force) == deleted
        assert await manager.commit_summary_cursor(sid, snapshot[-1], revision) is committed
        witness = await manager.get_session_metadata(sid, "last_summary_message_v1")
        if committed:
            assert witness == {"id": snapshot[-1].id, "timestamp": snapshot[-1].timestamp}
        else:
            assert witness is None
            assert await store.get_message_by_id(sid, snapshot[-1].id) is None
        assert (
            await manager.get_session_metadata(sid, "last_summarized_index", -1)
            == expected
        )
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [1, 3])
async def test_pending_retry_never_skips_unsummarized_chunks(tmp_path, retries, monkeypatch):
    handler, manager, store = await _make_real_handler(tmp_path)
    handler._memory_reflection.config_manager = ConfigManager(
        {"reflection_engine": {"max_messages_per_segment": 4}}
    )
    sid = "test:group:pending-range"
    await _seed_messages(manager, sid, [1_780_000_000.0 + i for i in range(10)])
    reflection = handler._memory_reflection
    initial_pending = {"start_index": 6, "end_index": 10, "retry_count": retries}
    await manager.update_session_metadata(sid, "pending_summary", initial_pending)
    try:
        plan = await reflection._build_reflection_plan_for_cursor(
            sid, 10, 0, initial_pending, None, None, sid,
        )
        if retries == 3:
            assert plan is None
            pending = await manager.get_session_metadata(sid, "pending_summary")
            assert (pending["start_index"], pending["end_index"]) == (0, 4)
            assert pending["retry_count"] == 0
            assert pending["exhausted_retry_count"] == 3
            assert pending["failure_stage"] == "retry_exhausted"
            assert await reflection._build_reflection_plan_for_cursor(
                sid, 10, 0, pending, None, None, sid,
            ) is None
            monkeypatch.setattr(
                "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection.time",
                SimpleNamespace(time=lambda: pending["retry_after"] + 1),
            )
            plan = await reflection._build_reflection_plan_for_cursor(
                sid, 10, 0, pending, None, None, sid,
            )
            assert plan.retry_count == 0
        assert (plan.start_index, plan.end_index) == (0, 4)
        assert [message.content for message in plan.history_messages] == [
            "message-0", "message-1", "message-2", "message-3",
        ]
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 0
        assert await store.get_message_count(sid) == 10
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_snapshot_refreshes_a_cursor_that_moved_before_history_fetch(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:snapshot-shift"
    await _seed_messages(manager, sid, [1_780_000_000.0 + i for i in range(10)])
    await manager.update_session_metadata(sid, "last_summarized_index", 4)
    try:
        await manager.trim_session_messages(sid, 4)
        plan = await handler._memory_reflection._build_reflection_plan_for_cursor(
            sid, 10, 4, None, 0, None, sid
        )
        assert plan.start_index == 0
        assert plan.history_messages[0].content == "message-4"
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_native_retry_keeps_failed_rows_after_trim_while_model_runs(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    reflection = handler._memory_reflection
    reflection.config_manager = ConfigManager(
        {"reflection_engine": {"max_messages_per_segment": 4}}
    )
    sid = "test:group:trim-during-model"
    await _seed_messages(manager, sid, [1_780_000_000.0 + i for i in range(10)])
    await manager.update_session_metadata(sid, "last_summarized_index", 4)
    entered, release = asyncio.Event(), asyncio.Event()

    async def fail_after_trim(**kwargs):
        entered.set()
        await release.wait()
        raise ConnectionResetError("synthetic transport failure")

    _configure_processor(handler, side_effect=fail_after_trim)
    try:
        revision = await manager.get_session_revision(sid)
        plan = await reflection._build_reflection_plan_for_cursor(
            sid, 10, 4, None, revision, None, sid,
        )
        assert [message.content for message in plan.history_messages] == [
            "message-4", "message-5", "message-6", "message-7",
        ]
        assert await reflection._schedule_reflection(plan)
        await asyncio.wait_for(entered.wait(), 2)
        assert await manager.trim_session_messages(sid, 4) == 4
        release.set()
        await asyncio.gather(*list(handler._storage_tasks))
        pending = await manager.get_session_metadata(sid, "pending_summary")
        assert (pending["start_index"], pending["end_index"], pending["retry_count"]) == (0, 4, 1)
        assert await manager.get_session_metadata(sid, "last_summarized_index") == 0
        handler.memory_processor.process_conversation.assert_awaited_once()
        handler.memory_engine.add_memory.assert_not_awaited()

        retry = await reflection._build_reflection_plan_for_cursor(
            sid, 6, 0, pending, revision, None, sid,
        )
        assert retry.history_messages == plan.history_messages
        _configure_processor(handler)
        handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
        await _run_scheduled(handler, retry)
        assert [
            [message.content for message in call.kwargs["messages"]]
            for call in handler.memory_processor.process_conversation.await_args_list
        ] == [
            ["message-4", "message-5", "message-6", "message-7"],
            ["message-8", "message-9"],
        ]
        assert await manager.get_session_metadata(sid, "last_summarized_index") == 6
        assert await manager.get_session_metadata(sid, "pending_summary") is None
    finally:
        release.set()
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_cancelled_native_pending_write_rolls_back_and_releases_sqlite(tmp_path, monkeypatch):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:pending-cancel"
    await _seed_messages(manager, sid, [1_780_000_000.0, 1_780_000_001.0])
    revision = await manager.get_session_revision(sid)
    messages = await manager.get_messages_range(sid, 0, 2)
    pending = {"start_index": 0, "end_index": 2, "retry_count": 1, "failure_stage": "model"}
    entered = asyncio.Event()

    async def wait_before_commit():
        entered.set()
        await asyncio.Event().wait()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(store, "_commit_write_transaction", wait_before_commit)
            task = asyncio.create_task(manager.write_native_pending_summary(sid, messages, pending, revision))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not store.connection.in_transaction
        assert await manager.get_session_metadata(sid, "pending_summary") is None
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 0

        await store.update_message_metadata(messages[0].id, {"edited": True})
        assert not await manager.write_native_pending_summary(sid, messages, pending, revision)
        assert await manager.get_session_metadata(sid, "pending_summary") is None
        fresh = await manager.get_messages_range(sid, 0, 2)
        assert await manager.write_native_pending_summary(sid, fresh, pending, revision)
        assert await manager.get_session_metadata(sid, "pending_summary") == pending
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backfill_at", ["model", "after_publication"])
async def test_native_summary_cannot_advance_past_a_backfilled_row(tmp_path, backfill_at):
    handler, manager, store = await _make_real_handler(tmp_path)
    reflection = handler._memory_reflection
    sid = "test:group:backfill-in-reflection"
    start = 1_780_000_000.0
    await _seed_messages(manager, sid, [start, start + 2])
    entered, release = asyncio.Event(), asyncio.Event()
    model_inputs = []
    publications = []
    validation = []
    engine = handler.memory_engine
    engine.find_similar_memory = AsyncMock(return_value=None)
    engine.delete_memory = AsyncMock(side_effect=lambda memory_id: publications.remove(memory_id) is None)

    async def summarize(*, messages, **kwargs):
        model_inputs.append([message.content for message in messages])
        if backfill_at == "model" and len(model_inputs) == 1:
            entered.set()
            await release.wait()
        return "summary", {"topics": ["t1"]}, 0.6

    async def publish(*, source_validator, source_publish_lease, **kwargs):
        validation.append(await source_validator())
        async with source_publish_lease() as valid:
            assert validation[-1] is valid
            if not valid:
                return None
            publications.append(1)
        if backfill_at == "after_publication" and len(model_inputs) == 1:
            entered.set()
            await release.wait()
        return 1

    _configure_processor(handler, side_effect=summarize)
    engine.add_memory = AsyncMock(side_effect=publish)
    try:
        assert await reflection._schedule_reflection(await _build_plan(handler, sid, 2))
        await asyncio.wait_for(entered.wait(), 2)
        await manager.add_message(sid, "user", "backfilled", timestamp=start + 1)
        release.set()
        await asyncio.gather(*list(handler._storage_tasks))
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 0
        assert await manager.get_session_metadata(sid, "pending_summary") is None
        assert publications == []
        if backfill_at == "after_publication":
            assert validation == [True]
            engine.delete_memory.assert_awaited_once_with(1)
        else:
            assert validation == [False]
            engine.delete_memory.assert_not_awaited()

        await _run_scheduled(handler, await _build_plan(handler, sid, 3))
        assert model_inputs == [["message-0", "message-1"], ["message-0", "backfilled", "message-1"]]
        assert await manager.get_session_metadata(sid, "last_summarized_index") == 3
        assert await manager.get_session_metadata(sid, "pending_summary") is None
        assert publications == [1]
    finally:
        release.set()
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["after_delete", "after_commit"])
async def test_native_trim_keeps_transaction_and_cache_consistent_on_cancellation(tmp_path, monkeypatch, cancel_at):
    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:group:trim-cancel"
    start = 1_780_000_000.0
    await _seed_messages(manager, sid, [start + i for i in range(4)])
    await manager.update_session_metadata(sid, "last_summarized_index", 2)
    await manager.update_session_metadata(sid, "pending_summary", {"start_index": 2, "end_index": 4})
    assert len(await manager.get_messages(sid)) == 4
    entered, release = asyncio.Event(), asyncio.Event()
    execute = store.connection.execute
    commit = store.connection.commit
    rollback = store.connection.rollback
    rollback_started, rollback_release = asyncio.Event(), asyncio.Event()

    def pause_after_delete(statement, parameters=None):
        if "DELETE FROM messages" not in statement:
            return execute(statement, parameters)

        async def deleted():
            cursor = await execute(statement, parameters)
            entered.set()
            await release.wait()
            return cursor

        return deleted()

    async def pause_after_commit():
        await commit()
        entered.set()
        await release.wait()

    async def pause_before_rollback():
        rollback_started.set()
        await rollback_release.wait()
        await rollback()

    try:
        with monkeypatch.context() as patch:
            if cancel_at == "after_delete":
                patch.setattr(store.connection, "execute", pause_after_delete)
                patch.setattr(store.connection, "rollback", pause_before_rollback)
            else:
                patch.setattr(store.connection, "commit", pause_after_commit)
            task = asyncio.create_task(manager.trim_session_messages(sid, 2))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            if cancel_at == "after_delete":
                await asyncio.wait_for(rollback_started.wait(), 2)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                rollback_release.set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not store.connection.in_transaction
        kept_from = 0 if cancel_at == "after_delete" else 2
        assert [message.content for message in await manager.get_messages(sid)] == [
            f"message-{i}" for i in range(kept_from, 4)
        ]
        await manager.add_message(sid, "user", "next message", timestamp=start + 10)
        session = await store.get_session(sid)
        assert session.message_count == await store.get_message_count(sid) == 5 - kept_from
        assert session.metadata["last_summarized_index"] == 2 - kept_from
        assert session.metadata["pending_summary"] == {"start_index": 2 - kept_from, "end_index": 4 - kept_from}
    finally:
        release.set()
        rollback_release.set()
        await handler.shutdown()
        await store.close()
