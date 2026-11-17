from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection import (
    _OWNED_DRAFT_LINEAGE_KEY,
    _OWNED_MEMORY_SOURCE_VALIDATE_ATTR,
)
from astrbot_plugin_dynamics_memory.core.memory_source import serialize_source_messages
from astrbot_plugin_dynamics_memory.core.utils.json_utils import safe_json_dict

from ..event_handler_support import _make_real_handler
from .test_guarded_publication_recovery import open_engine, publication_row

SESSION_ID = "qq:FriendMessage:synthetic-window"
FIRST_TIMESTAMP = 1_760_000_000
START = 6
END = 9


@pytest_asyncio.fixture
async def reflection_window(tmp_path):
    async with open_engine(tmp_path) as engine:
        handler, manager, store = await _make_real_handler(tmp_path, memory_engine=engine)
        try:
            for index in range(START):
                await manager.add_message(
                    SESSION_ID, "user", f"Earlier message {index}",
                    timestamp=FIRST_TIMESTAMP + index,
                )
            await manager.update_session_metadata(SESSION_ID, "last_summarized_index", START)
            for index, (role, content, metadata) in enumerate([
                ("user", "The seed catalog arrived.", {}),
                ("assistant", "I will look up storage instructions.", {_OWNED_DRAFT_LINEAGE_KEY: "synthetic-token"}),
                ("user", "Keep the seeds in a cool place.", {}),
            ], START):
                await manager.add_message(
                    SESSION_ID, role, content, metadata=metadata,
                    sender_id="seed-bot" if role == "assistant" else "seed-keeper", platform="qq",
                    timestamp=FIRST_TIMESTAMP + 2 * index,
                )
            revision = await manager.get_session_revision(SESSION_ID)
            window = await store.get_messages_range(SESSION_ID, offset=START, limit=END - START)
            pending = {"start_index": START, "end_index": END, "retry_count": 1}
            assert await manager.write_native_pending_summary(SESSION_ID, window, pending, revision)
            reflection = handler._memory_reflection
            plan = await reflection._build_reflection_plan_for_cursor(
                SESSION_ID, END, START, pending, revision, "persona-test", SESSION_ID,
            )
            assert (plan.start_index, plan.end_index) == (START, END)
            yield reflection, plan, manager, engine
        finally:
            await handler.shutdown()
            await store.close()


async def _run_reflection(reflection, plan, input_rows, after_model=None):
    async def summarize(*, messages, **kwargs):
        if after_model is not None:
            await after_model()
        return "A seed catalog arrived and the seeds need cool storage.", {"topics": ["seeds"]}, 0.9

    processor = reflection.memory_processor
    processor.process_conversation = AsyncMock(side_effect=summarize)
    processor.classify_atoms_from_metadata = Mock(return_value=[])
    result = await reflection._storage_task(
        plan.session_id, plan.history_messages, plan.persona_id, plan.start_index,
        plan.end_index, plan.retry_count, plan.memory_scope, plan.expected_revision,
    )
    processor.process_conversation.assert_awaited_once()
    assert processor.process_conversation.await_args.kwargs["messages"] == input_rows
    return result


def _before_publication(monkeypatch, engine, callback):
    save_source = engine.save_memory_source

    async def save_then_change_source(*args, **kwargs):
        await save_source(*args, **kwargs)
        await callback()

    monkeypatch.setattr(engine, "save_memory_source", save_then_change_source)


@pytest.mark.asyncio
async def test_pending_window_with_unavailable_assistant_lineage_publishes_user_sources(reflection_window):
    reflection, plan, manager, engine = reflection_window
    user_rows = [plan.history_messages[0], plan.history_messages[2]]

    assert await _run_reflection(reflection, plan, user_rows) is True
    assert await manager.get_session_metadata(SESSION_ID, "last_summarized_index") == END
    assert await manager.get_session_metadata(SESSION_ID, "pending_summary") is None
    assert await manager.store.get_message_count(SESSION_ID) == END

    op_id, doc_id, status, step = await publication_row(engine)
    assert (status, step) == ("completed", "source_published")
    record = await engine.get_memory_record(doc_id)
    metadata = safe_json_dict(record["metadata"])
    assert metadata["source_publication_verified_v1"] == {"write_op_id": op_id, "memory_id": doc_id}
    assert metadata["source_window"]["message_count"] == END - START
    assert [ref["id"] for ref in metadata["source_row_refs_v1"]] == [row.id for row in user_rows]
    assert await engine.get_memory_source(doc_id) == serialize_source_messages(user_rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["after_model", "publication"])
async def test_assistant_lineage_revoked_after_selection_cannot_publish(reflection_window, monkeypatch, stage):
    reflection, plan, manager, engine = reflection_window
    assistant = plan.history_messages[1]
    event = SimpleNamespace()
    validator = AsyncMock(return_value=True)
    setattr(event, _OWNED_MEMORY_SOURCE_VALIDATE_ATTR, validator)
    reflection._remember_owned_source_lineage(
        assistant.metadata[_OWNED_DRAFT_LINEAGE_KEY], event, validator, assistant,
    )

    async def revoke():
        setattr(event, _OWNED_MEMORY_SOURCE_VALIDATE_ATTR, None)

    if stage == "publication":
        _before_publication(monkeypatch, engine, revoke)
    assert not await _run_reflection(
        reflection, plan, plan.history_messages,
        after_model=revoke if stage == "after_model" else None,
    )
    assert await manager.get_session_metadata(SESSION_ID, "last_summarized_index") == START
    assert engine.faiss_db.embedding_storage.index.ntotal == 0
    if stage == "publication":
        _, doc_id, status, step = await publication_row(engine)
        assert (status, step) == ("failed", "source_publication_rejected")
        assert await engine.get_memory_record(doc_id) is None
    else:
        pending = await manager.get_session_metadata(SESSION_ID, "pending_summary")
        assert (pending["failure_stage"], pending["error_code"]) == ("source_unavailable", "source_changed")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["edit", "delete", "backfill", "cursor", "revision", "tail"])
async def test_publication_checks_full_window_and_cursor(reflection_window, monkeypatch, change):
    reflection, plan, manager, engine = reflection_window
    first, excluded, last = plan.history_messages

    async def change_window():
        if change in {"edit", "delete"}:
            # Exercise the raw row check without using a revision bump.
            statement = (
                "UPDATE messages SET content='Changed assistant draft' WHERE id=?"
                if change == "edit" else "DELETE FROM messages WHERE id=?"
            )
            async with manager.store._write_lock:
                await manager.store.connection.execute(statement, (excluded.id,))
                await manager.store.connection.commit()
        elif change in {"backfill", "tail"}:
            timestamp = first.timestamp + 1 if change == "backfill" else last.timestamp + 1
            await manager.add_message(SESSION_ID, "user", "Newly received message", timestamp=timestamp)
        elif change == "cursor":
            assert await manager.commit_summary_cursor(
                SESSION_ID, last, plan.expected_revision, window_messages=plan.history_messages,
            )
        else:
            await manager.bump_session_revision(SESSION_ID)

    _before_publication(monkeypatch, engine, change_window)
    result = await _run_reflection(reflection, plan, [first, last])
    _, doc_id, status, step = await publication_row(engine)
    cursor = await manager.get_session_metadata(SESSION_ID, "last_summarized_index")
    if change == "tail":
        assert result is True
        assert ((status, step), cursor) == (("completed", "source_published"), END)
        assert await manager.store.get_message_count(SESSION_ID) == END + 1
    else:
        assert result is False
        assert (status, step) == ("failed", "source_publication_rejected")
        assert await engine.get_memory_record(doc_id) is None
        assert cursor == (END if change == "cursor" else START)
