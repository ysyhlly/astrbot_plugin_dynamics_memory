"""Managed input provenance and actual-delivery persistence regressions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler
from astrbot_plugin_dynamics_memory.core.memory_source import serialize_source_messages, restore_source_messages
from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message


SCOPE = {"platform": "qq", "umo": "qq:GroupMessage:1", "persona_id": "anan", "instance_id": "one"}


def envelope(event_id="a", kind="message.received", text="我觉得她在生气", scope=None):
    return {"schema_version": 1, "event_id": event_id, "type": kind,
            "scope": scope or SCOPE, "actor_id": "u1", "text": text,
            "occurred_at": "2026-10-10T10:00:00+08:00", "data": {},
            "source_ids": [event_id], "fact_kind": "observation"}


@pytest.mark.asyncio
async def test_actual_segments_only_duplicates_survive_restart(tmp_path):
    store = ConversationStore(str(tmp_path / "memory.db"))
    await store.initialize()
    try:
        assert await store.ingest_narrative_event(envelope())
        assert not await store.ingest_narrative_event(envelope())
        assert await store.ingest_narrative_event(envelope("sent-0", "delivery.success", "第一段"))
        for kind in ("delivery.failure", "delivery.uncertain", "draft.created"):
            assert not await store.ingest_narrative_event(envelope(kind, kind, "未发草稿"))
        assert await store.get_message_count(scope_key(SCOPE)) == 2
    finally:
        await store.close()
    store = ConversationStore(str(tmp_path / "memory.db"))
    await store.initialize()
    try:
        assert not await store.ingest_narrative_event(envelope("sent-0", "delivery.success", "第一段"))
        items = await store.recall_narrative_events(SCOPE, "生气")
        assert len(items) == 2
        assert items[0]["text"] == "用户发言：我觉得她在生气"
        assert items[0]["fact_kind"] == "observation"
        assert {item["source_id"] for item in items} == {"a", "sent-0"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_concurrent_duplicate_and_scope_isolation(tmp_path):
    store = ConversationStore(str(tmp_path / "memory.db"))
    await store.initialize()
    try:
        results = await asyncio.gather(*(store.ingest_narrative_event(envelope()) for _ in range(6)))
        assert results.count(True) == 1
        for changed in ({"umo": "qq:GroupMessage:2"}, {"platform": "qq2"}, {"persona_id": "other"}, {"instance_id": "two"}):
            scope = {**SCOPE, **changed}
            assert await store.recall_narrative_events(scope, "") == []
        assert len(await store.get_messages(scope_key(SCOPE))) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_transaction_retries_without_losing_evidence(tmp_path):
    store = ConversationStore(str(tmp_path / "memory.db"))
    await store.initialize()
    original = store.connection.execute

    async def broken(sql, *args, **kwargs):
        if "INSERT INTO messages" in sql:
            raise RuntimeError("injected storage failure")
        return await original(sql, *args, **kwargs)

    try:
        store.connection.execute = broken
        with pytest.raises(RuntimeError):
            await store.ingest_narrative_event(envelope())
        store.connection.execute = original
        assert await store.ingest_narrative_event(envelope())
        assert await store.get_message_count(scope_key(SCOPE)) == 1
    finally:
        store.connection.execute = original
        await store.close()


def test_scope_cannot_fall_back_to_group_or_unknown_persona():
    with pytest.raises(ValueError):
        scope_key({"umo": "1"})
    with pytest.raises(ValueError):
        scope_key({**SCOPE, "persona_id": None})


@pytest.mark.asyncio
async def test_receipt_summary_reads_committed_evidence_without_appending_drafts():
    handler = EventHandler.__new__(EventHandler)
    handler.config_manager = SimpleNamespace(get=lambda key, default=None: default)
    handler._memory_reflection = SimpleNamespace(schedule_recorded_scope=AsyncMock())
    await handler.reflect_recorded_scope(SCOPE)
    handler._memory_reflection.schedule_recorded_scope.assert_awaited_once_with(SCOPE)


def test_summary_source_roundtrip_keeps_fact_category_scope_and_event_id():
    message = Message(id=1, session_id=scope_key(SCOPE), role="user",
                      content="我猜他生气了", sender_id="u1", platform="qq",
                      metadata={"fact_kind": "observation", "scope": SCOPE, "source_event_id": "real-1"})
    restored = restore_source_messages(serialize_source_messages([message]))[0]
    assert restored.metadata["fact_kind"] == "observation"
    assert restored.metadata["scope"] == SCOPE
    assert restored.metadata["source_event_id"] == "real-1"
