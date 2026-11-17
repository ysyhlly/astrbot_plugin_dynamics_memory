"""Retained migration runs once and never remaps already-qualified sessions."""

import json
import pytest
from tests.integration.test_real_db_end_to_end import real_db_stack as real_db_stack


@pytest.mark.asyncio
async def test_legacy_prefix_migration_is_idempotent_and_preserves_qualified_rows(
    real_db_stack,
):
    engine = real_db_stack["memory_engine"]
    ids = []
    for session in (
        "leaf",
        "astrbot!leaf",
        "webchat!astrbot!leaf",
        "other:FriendMessage:leaf",
    ):
        ids.append(
            await engine.add_memory(
                "source " + session, session_id=session, persona_id="p"
            )
        )
    target = "webchat:FriendMessage:webchat!astrbot!leaf"
    await engine._migrate_session_data_if_needed(target)
    for mid, old in zip(ids[:3], ("leaf", "astrbot!leaf", "webchat!astrbot!leaf")):
        record = await engine.get_memory(mid)
        meta = record["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        assert meta["session_id"] == target and meta["old_session_id"] == old
    record = await engine.get_memory(ids[3])
    assert json.loads(record["metadata"])["session_id"] == "other:FriendMessage:leaf"
    assert target in engine._runtime_migration_checked_sessions
    await engine._migrate_session_data_if_needed(target)
    await engine._migrate_session_data_if_needed("malformed")
    assert (
        json.loads((await engine.get_memory(ids[0]))["metadata"])["old_session_id"]
        == "leaf"
    )
