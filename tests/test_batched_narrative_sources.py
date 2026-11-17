"""Exact evidence recovery stays scoped and fresh when batched."""

import json
import pytest
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from .test_narrative_api import envelope, SCOPE


@pytest.mark.asyncio
async def test_batched_sources_keep_order_duplicates_and_recheck_deleted_rows(tmp_path):
    store = ConversationStore(str(tmp_path / "sources.db"))
    await store.initialize()
    try:
        for i in range(205):
            await store.ingest_narrative_event(envelope(str(i), text=str(i)))
        ids = [str(i) for i in reversed(range(205))] + ["0"]
        metadata = {
            "source_window": {"session_id": scope_key(SCOPE)},
            "narrative_provenance": [
                {"source_id": i, "scope": dict(SCOPE), "fact_kind": "observation"}
                for i in ids
            ],
        }
        queries = []
        await store.connection.set_trace_callback(queries.append)
        rows = await store.load_narrative_sources(metadata)
        assert [row["content"] for row in rows] == ids
        assert (
            len(
                [
                    q
                    for q in queries
                    if q.startswith("SELECT event_key,session_id,envelope")
                ]
            )
            == 2
        )
        await store.connection.execute(
            "DELETE FROM narrative_events WHERE event_key=?", (scope_key(SCOPE) + ":0",)
        )
        await store.connection.commit()
        assert await store.load_narrative_sources(metadata) is None
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["scope", "kind", "event_id", "missing", "row_scope"]
)
async def test_batch_never_accepts_foreign_or_forged_sources(tmp_path, damage):
    store = ConversationStore(str(tmp_path / "sources.db"))
    await store.initialize()
    try:
        await store.ingest_narrative_event(envelope("a"))
        item = {"source_id": "a", "scope": dict(SCOPE), "fact_kind": "observation"}
        metadata = {
            "source_window": {"session_id": scope_key(SCOPE)},
            "narrative_provenance": [item],
        }
        if damage == "scope":
            item["scope"]["umo"] = "qq:GroupMessage:2"
        elif damage == "kind":
            item["fact_kind"] = "receipt"
        elif damage == "missing":
            item["source_id"] = "missing"
        elif damage == "row_scope":
            await store.connection.execute(
                "UPDATE narrative_events SET session_id=?", ("foreign",)
            )
            await store.connection.commit()
        else:
            forged = envelope("wrong")
            await store.connection.execute(
                "UPDATE narrative_events SET envelope=?", (json.dumps(forged),)
            )
            await store.connection.commit()
        assert await store.load_narrative_sources(metadata) is None
    finally:
        await store.close()
