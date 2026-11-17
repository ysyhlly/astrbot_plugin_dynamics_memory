"""Trimming must retain unsummarized messages even when old cursor metadata is corrupt."""

import json

import pytest

from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata,deleted",
    [
        ("bad-json", 0),
        ("[]", 0),
        ('{"last_summarized_index":"bad"}', 0),
        ('{"last_summarized_index":99}', 0),
        (
            json.dumps(
                {
                    "last_summarized_index": 2,
                    "pending_summary": {"start_index": 2, "end_index": 4},
                }
            ),
            2,
        ),
        (
            json.dumps(
                {
                    "last_summarized_index": 2,
                    "pending_summary": {"start_index": "bad", "end_index": 4},
                }
            ),
            2,
        ),
    ],
)
async def test_retained_trimming_reconciles_only_safe_prefix(
    tmp_path, metadata, deleted
):
    store = ConversationStore(str(tmp_path / "conversation.db"))
    await store.initialize()
    try:
        for i in range(4):
            await store.add_message(
                Message(
                    id=0,
                    session_id="s",
                    role="user",
                    sender_id="u1",
                    content=f"line-{i}",
                    timestamp=100 + i,
                    metadata={},
                )
            )
        await store.connection.execute(
            "UPDATE sessions SET metadata=?,message_count=999 WHERE session_id='s'",
            (metadata,),
        )
        await store.connection.commit()
        assert await store.trim_session_messages("missing", 10) == 0
        assert await store.trim_session_messages("s", 0) == 0
        assert await store.trim_session_messages("s", 10) == deleted
        rows = await store.get_messages("s", limit=20)
        assert {row.content for row in rows} == {f"line-{i}" for i in range(deleted, 4)}
        if deleted:
            session = await store.get_session("s")
            assert (
                session.message_count == 2
                and session.metadata["last_summarized_index"] == 0
            )
            pending = session.metadata.get("pending_summary")
            if "bad" in metadata:
                assert pending is None
            else:
                assert pending["start_index"] == 0 and pending["end_index"] == 2
    finally:
        await store.close()
