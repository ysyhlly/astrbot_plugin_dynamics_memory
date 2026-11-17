import json
from types import SimpleNamespace

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core.retrieval.narrative_history import (
    ensure_scope_index,
    search_local,
    scope_expressions,
)


@pytest.mark.asyncio
async def test_index_preserves_origin_persona_lifecycle_and_malformed_metadata(
    tmp_path,
):
    async with aiosqlite.connect(tmp_path / "index.db") as db:
        await db.execute(
            "CREATE TABLE documents(id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
        )
        docs = [
            {"session_id": "room", "persona_id": "anan"},
            {
                "source_session_id": "room",
                "session_id": "old",
                "persona_id": "夏目安安",
            },
            {
                "source_session_id": "room",
                "source_window": {"session_id": "other"},
                "persona_id": "anan",
            },
            {"session_id": "other", "persona_id": "anan"},
            {"session_id": "room", "persona_id": "different"},
            {"session_id": "room", "persona_id": "anan", "status": "deleted"},
            {"session_id": "room", "persona_id": None},
        ]
        for i, meta in enumerate(docs, 1):
            await db.execute(
                "INSERT INTO documents VALUES(?,?,?)", (i, "饮料偏好", json.dumps(meta))
            )
        await db.execute("INSERT INTO documents VALUES(8,'malformed','{broken')")
        await ensure_scope_index(db)
        await ensure_scope_index(db)

        async def policy(rows):
            return rows

        engine = SimpleNamespace(
            db_connection=db,
            text_processor=None,
            _filter_by_retrieval_policy=lambda rows: rows,
            _apply_atom_policy=policy,
        )
        result = await search_local(engine, "饮料", 20, ("room",), ("anan", "夏目安安"))
        assert {r.doc_id for r in result} == {1, 2, 7}
        _, origin, status, persona = scope_expressions("d.")
        plan = await (
            await db.execute(
                f"EXPLAIN QUERY PLAN SELECT id FROM documents d WHERE {origin}=? AND {status}=? AND {persona}=?",
                ("room", "active", "anan"),
            )
        ).fetchall()
        assert any("idx_narrative_scope_v1" in row[3] for row in plan)
        await db.execute(
            "UPDATE documents SET metadata=? WHERE id=1",
            (json.dumps({"session_id": "other", "persona_id": "anan"}),),
        )
        result = await search_local(engine, "饮料", 20, ("room",), ("anan", "夏目安安"))
        assert {r.doc_id for r in result} == {2, 7}
