"""The private hook and its effect guard look up fresh document and atom state."""

import asyncio
import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest
import pytest_asyncio

from astrbot_plugin_dynamics_memory.core import plugin_helpers_mixin
from astrbot_plugin_dynamics_memory.core.event_handler_modules import memory_recall
from astrbot_plugin_dynamics_memory.core.managers import memory_engine_recall
from astrbot_plugin_dynamics_memory.core.memory_scope import current_adapter_access
from astrbot_plugin_dynamics_memory.core.plugin_helpers_mixin import PluginHelpersMixin
from astrbot_plugin_dynamics_memory.core.retrieval.bm25_retriever import BM25Retriever
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import HybridResult

from .test_atom_injection_lifecycle import _RecallEngine, _create_atoms
from .test_event_handler import _make_event

# Pytest resolves these fixtures and the handler's dependencies by name.
from .test_event_handler import (  # noqa: F401
    conversation_manager,
    handler,
    memory_engine,
    memory_processor,
)


_SESSION = "test:FriendMessage:sid-1"
_PERSONA = "persona_1"
_PARENT = "Original parent with REMOVED_FACT and LIVE_FACT"


@pytest.fixture
def private_hook(handler, monkeypatch):  # noqa: F811
    event = _make_event()
    event.unified_msg_origin = _SESSION
    event.get_platform_id.return_value = "test"
    event._chat_dynamics_owned_request = True
    event._cd_owned_original_input_v1 = True
    event.get_extra.return_value = None
    scope = {"platform": "test", "umo": _SESSION, "persona_id": _PERSONA, "instance_id": "test"}
    receipt = {
        "schema_version": 1, "provider": "astrbot_plugin_chat_dynamics",
        "status": "skipped", "reason": "cd_owned_request", "scope": scope,
        "purpose": "reply", "card_version": "fixture",
    }
    req = SimpleNamespace(system_prompt="", contexts=[], extra_user_content_parts=[],
                          _cd_memory_feedback_v1=receipt)
    recall = handler._memory_recall
    recall.message_utils.resolve_original_input = AsyncMock(
        return_value=SimpleNamespace(status="valid", query_text="What do you remember?")
    )
    recall._store_owned_original_messages = AsyncMock(return_value=True)
    recall._schedule_message_limit = lambda _session: "scheduled"
    recall.config_manager._config["recall_engine"]["top_k"] = 5
    persona = AsyncMock(return_value=_PERSONA)
    monkeypatch.setattr(memory_recall, "get_persona_id", persona)
    monkeypatch.setattr(plugin_helpers_mixin, "get_persona_id", persona)
    plugin = PluginHelpersMixin()
    plugin.context = handler.context
    plugin.config_manager = recall.config_manager
    plugin.initializer = SimpleNamespace(conversation_manager=handler.conversation_manager)
    plugin.peek_evidence_revision = lambda _scope: {"available": True, "correction_revision": 0}
    plugin._narrative_scope_allowed = AsyncMock(return_value=True)
    plugin._fresh_cd_owned_worldbook_receipt = AsyncMock(return_value=receipt)

    async def allow_frames(_event, _scope, frames):
        return [dict(frame) for frame in frames]

    plugin._filter_native_frames = AsyncMock(side_effect=allow_frames)
    recall.summary_guard = plugin._install_private_summary_guard
    return SimpleNamespace(handler=handler, plugin=plugin, event=event, req=req)


@pytest_asyncio.fixture
async def source_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "summaries.db")
    async with aiosqlite.connect(db_path) as connection:
        now = time.time()
        await _create_atoms(connection, [
            (11, 1, "LIVE_FACT", "active", now + 3600),
            (12, 1, "SECOND_LIVE_FACT", "active", now + 3600),
            (13, 1, "REMOVED_FACT", "forgotten", now + 3600),
        ])
        metadata = {
            "adapter_id": "test", "session_id": _SESSION, "source_session_id": _SESSION,
            "persona_id": _PERSONA, "interaction_type": "private_chat", "source_platform": "test",
            "memory_visibility": "private", "importance": 0.7, "create_time": now,
        }
        documents = [(1, _PARENT, {
            **metadata, "atom_types": ["factual"], "key_facts": ["REMOVED_FACT"],
            "persona_summary": "Another REMOVED_FACT",
        })]
        documents.extend((i, f"Legacy line {i}", dict(metadata)) for i in range(2, 6))
        await connection.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)")
        await connection.executemany(
            "INSERT INTO documents VALUES (?, ?, ?)",
            [(i, content, json.dumps(meta)) for i, content, meta in documents],
        )
        await connection.commit()
        engine = _RecallEngine(connection)
        engine.bm25_retriever = BM25Retriever(db_path, None, cache_reads=True)

        async def retrieve_documents(_storage, *, metadata_filters, ids, limit):
            # Also exercise the legacy get_memory path under the baseline.
            async with connection.execute(
                "SELECT id, text, metadata FROM documents WHERE id = ?", ids
            ) as cursor:
                row = await cursor.fetchone()
            return [{"id": row[0], "text": row[1], "metadata": json.loads(row[2])}] if row else []

        monkeypatch.setattr(memory_engine_recall, "retrieve_documents", retrieve_documents)
        engine.faiss_db = None
        candidates = [
            HybridResult(i, 1 - i * 0.05, 0.8, 0.8, None, content, meta)
            for i, content, meta in documents
        ]
        async def search(**_kwargs):
            return await engine._apply_atom_policy(candidates)

        engine.search_memories = AsyncMock(side_effect=search)
        engine.schedule_recall_access_touch = lambda _docs, _atoms: None
        try:
            yield SimpleNamespace(engine=engine, connection=connection, candidates=candidates)
        finally:
            await engine.bm25_retriever.close()


async def _recall(private_hook, source_db, *, budget=None):
    private_hook.plugin.initializer.memory_engine = source_db.engine
    private_hook.handler._memory_recall.memory_engine = source_db.engine
    await private_hook.handler.handle_memory_recall(private_hook.event, private_hook.req, budget=budget)


@pytest.mark.asyncio
async def test_plain_and_atom_documents_share_one_accepted_summary(private_hook, source_db):
    await _recall(private_hook, source_db)
    req = private_hook.req
    assert req._lm_private_hook_outcome_v1 == "ready"
    assert len(req.extra_user_content_parts) == 1
    summary = req.extra_user_content_parts[0].text
    assert "LIVE_FACT SECOND_LIVE_FACT" in summary
    assert "REMOVED_FACT" not in summary
    assert all(f"Legacy line {i}" in summary for i in range(2, 6))
    frozen = req._lm_private_summary_sources_v1
    assert frozen["segment_item_count"] == 5
    first = frozen["source_documents"][0]
    assert first["document_text_sha256"] == hashlib.sha256(_PARENT.encode()).hexdigest()
    assert first["projected_text_sha256"] == hashlib.sha256(b"LIVE_FACT\nSECOND_LIVE_FACT").hexdigest()
    assert [atom[0] for atom in first["atom_proof"]] == [11, 12]
    assert await req._lm_private_summary_validate_v1() is True
    frame = private_hook.plugin._filter_native_frames.call_args.args[2][0]
    assert frame["text"] == "LIVE_FACT\nSECOND_LIVE_FACT"
    assert frame["source_kind"] == "legacy_summary"
    assert frame["provenance_complete"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("sql", [
    "UPDATE memory_atoms SET content = 'changed' WHERE id = 11",
    "UPDATE memory_atoms SET status = 'forgotten' WHERE id = 11",
    "UPDATE memory_atoms SET expires_at = 0 WHERE id = 11",
    "UPDATE memory_atoms SET parent_memory_id = 2 WHERE id = 11",
    "DELETE FROM memory_atoms WHERE id = 11",
    "UPDATE memory_atoms SET id = 14 WHERE id = 11",
    "UPDATE documents SET text = 'changed' WHERE id = 1",
    "UPDATE documents SET metadata = json_set(metadata, '$.superseded_by', 5) WHERE id = 1",
    "UPDATE documents SET metadata = json_set(metadata, '$.persona_summary', 'changed') WHERE id = 2",
    "UPDATE documents SET metadata = json_set(metadata, '$.source_session_id', 'test:FriendMessage:other') WHERE id = 1",
    "UPDATE documents SET metadata = json_set(metadata, '$.session_id', 'test:FriendMessage:other') WHERE id = 1",
    "UPDATE documents SET metadata = json_set(metadata, '$.forgotten', json('true')) WHERE id = 1",
    "UPDATE documents SET metadata = json_set(metadata, '$.adapter_id', 'outside') WHERE id = 1",
    "DELETE FROM documents WHERE id = 1",
    "INSERT INTO memory_atoms VALUES (14, 2, 'New atom', 'active', 2000000000)",
])
async def test_frozen_summary_rejects_changed_sources(private_hook, source_db, sql):
    private_hook.plugin.config_manager._config["filtering_settings"]["shared_adapter_ids"] = ["test"]
    await _recall(private_hook, source_db)
    validator = private_hook.req._lm_private_summary_validate_v1
    assert await validator() is True
    await source_db.connection.execute(sql)
    await source_db.connection.commit()
    assert await validator() is False


@pytest.mark.asyncio
async def test_frozen_summary_rechecks_adapter_sharing_outside_hook(private_hook, source_db):
    filtering = private_hook.plugin.config_manager._config["filtering_settings"]
    filtering["shared_adapter_ids"] = ["test", "other"]
    await source_db.connection.execute(
        "UPDATE documents SET metadata = json_set(metadata, '$.adapter_id', 'other') WHERE id = 1"
    )
    await source_db.connection.commit()
    source_db.candidates[0].metadata["adapter_id"] = "other"
    await _recall(private_hook, source_db)
    validator = private_hook.req._lm_private_summary_validate_v1
    assert current_adapter_access.get() is None
    assert await validator() is True
    filtering["shared_adapter_ids"] = ["test"]
    assert await validator() is False
    assert current_adapter_access.get() is None


@pytest.mark.asyncio
async def test_plain_documents_work_without_an_atom_table(private_hook, source_db):
    await source_db.connection.execute("DROP TABLE memory_atoms")
    await source_db.connection.commit()
    del source_db.candidates[0]
    await _recall(private_hook, source_db)
    req = private_hook.req
    assert req._lm_private_hook_outcome_v1 == "ready"
    assert len(req.extra_user_content_parts) == 1
    assert await req._lm_private_summary_validate_v1() is True


@pytest.mark.asyncio
async def test_access_statistics_do_not_invalidate_frozen_summary(private_hook, source_db):
    await _recall(private_hook, source_db)
    validator = private_hook.req._lm_private_summary_validate_v1
    await source_db.connection.execute(
        "UPDATE documents SET metadata = json_set(metadata, '$.access_count', 5, '$.last_access_time', 10)"
    )
    await source_db.connection.commit()
    assert await validator() is True


@pytest.mark.asyncio
async def test_reapplying_atom_policy_preserves_document_identity(source_db):
    engine = source_db.engine
    first = await engine._apply_atom_policy(source_db.candidates)
    await source_db.connection.execute("UPDATE memory_atoms SET status='forgotten' WHERE id=11")
    await source_db.connection.commit()
    updated = await engine._apply_atom_policy(first)
    assert updated[0].content == "SECOND_LIVE_FACT"
    assert updated[0].atom_projection.document_sha256 == hashlib.sha256(_PARENT.encode()).hexdigest()
    assert [atom[0] for atom in updated[0].atom_projection.atoms] == [12]


@pytest.mark.asyncio
async def test_final_snapshot_rejects_atom_deleted_during_blocking_check(private_hook, source_db):
    async def delete_during_validation(_event, _scope, frames):
        await source_db.connection.execute("DELETE FROM memory_atoms WHERE id = 11")
        await source_db.connection.commit()
        return frames

    private_hook.plugin._filter_native_frames.side_effect = delete_during_validation
    await _recall(private_hook, source_db)
    req = private_hook.req
    assert private_hook.plugin._filter_native_frames.await_count == 2
    assert req._lm_private_hook_outcome_v1 == "partial"
    assert req.extra_user_content_parts == []
    assert getattr(req, "_lm_private_summary_sources_v1", None) is None
    assert getattr(req, "_lm_private_summary_validate_v1", None) is None
    assert not req._lm_private_segment_refs_v1


@pytest.mark.asyncio
async def test_shared_source_scope_is_checked_once_per_pass_and_fresh_each_effect(
    private_hook, source_db,
):
    await _recall(private_hook, source_db)
    guard = private_hook.req._lm_private_summary_validate_v1
    scope_allowed = private_hook.plugin._narrative_scope_allowed
    scope_allowed.reset_mock()
    assert await guard() is True
    assert scope_allowed.await_count == 2
    assert all(call.args[0]["umo"] == _SESSION for call in scope_allowed.await_args_list)

    scope_allowed.reset_mock()
    scope_allowed.side_effect = [True, False]
    assert await guard() is False
    assert scope_allowed.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["forget_atom", "expire_atom", "delete_parent", "correction_revision"])
async def test_deferred_final_snapshot_rejects_mid_effect_revocation(private_hook, source_db, change):
    await _recall(private_hook, source_db)
    guard = private_hook.req._lm_private_summary_validate_v1
    assert guard._lm_summary_deferred_final_owner_v1 == (private_hook.req, private_hook.event)
    finalize = await guard(defer_final=True)
    assert callable(finalize)
    if change == "correction_revision":
        private_hook.plugin.peek_evidence_revision = lambda _scope: {
            "available": True, "correction_revision": 1,
        }
    else:
        statements = {
            "forget_atom": "UPDATE memory_atoms SET status='forgotten' WHERE id=11",
            "expire_atom": "UPDATE memory_atoms SET expires_at=0 WHERE id=11",
            "delete_parent": "DELETE FROM documents WHERE id=2",
        }
        await source_db.connection.execute(statements[change])
        await source_db.connection.commit()
    assert await finalize() is False


@pytest.mark.asyncio
async def test_deferred_finalizer_can_only_be_used_once_by_its_owner_task(private_hook, source_db):
    await _recall(private_hook, source_db)
    guard = private_hook.req._lm_private_summary_validate_v1
    finalize = await guard(defer_final=True)
    assert await asyncio.create_task(finalize()) is False
    assert await finalize() is True
    assert await finalize() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_deferred_finalizer_is_consumed_when_final_read_fails(private_hook, source_db, failure):
    await _recall(private_hook, source_db)
    guard = private_hook.req._lm_private_summary_validate_v1
    finalize = await guard(defer_final=True)
    good_snapshot = source_db.engine.get_summary_source_snapshots
    source_db.engine.get_summary_source_snapshots = AsyncMock(side_effect=failure)
    with pytest.raises(failure):
        await finalize()

    source_db.engine.get_summary_source_snapshots = good_snapshot
    assert await finalize() is False
    assert await guard() is True
