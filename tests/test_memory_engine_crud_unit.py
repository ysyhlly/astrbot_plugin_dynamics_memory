"""Unit tests covering remaining branches of MemoryEngineCrudMixin."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core.maintenance.source_state_guard import (
    SourceStateGuard,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine_crud import (
    MemoryEngineCrudMixin,
)


class _DummyCrudEngine(MemoryEngineCrudMixin):
    def __init__(self, db_connection=None):
        self.db_connection = db_connection
        self.config = {
            "memory_dedup": {"merge_sources": True, "max_source_messages": 10}
        }
        self.atom_enabled = True
        self.atom_store = None
        self.hybrid_retriever = SimpleNamespace(
            update_metadata=AsyncMock(return_value=True),
        )
        self.index_writer = SimpleNamespace(
            update_metadata=AsyncMock(return_value=True),
            delete_memory=AsyncMock(return_value=True),
        )
        self.graph_memory_manager = SimpleNamespace(
            index_memory=AsyncMock(return_value=True),
            delete_memory=AsyncMock(return_value=True),
            source_graph_index_complete=AsyncMock(return_value=True),
        )
        self.source_state_guard = SourceStateGuard()
        self.invalidated = 0

    def _write_db(self):
        return self._write_transaction()

    def _write_transaction(self):
        engine = self

        class _Tx:
            async def __aenter__(self):
                return engine.db_connection

            async def __aexit__(self, exc_type, exc, tb):
                if exc_type is None:
                    await engine.db_connection.commit()
                else:
                    await engine.db_connection.rollback()

        return _Tx()

    def _invalidate_search_cache(self):
        self.invalidated += 1

    def _is_memory_vector_indexed(self, _memory_id):
        return True

    async def get_memory(self, memory_id: int):
        cursor = await self.db_connection.execute(
            "SELECT id, doc_id, text, metadata FROM documents WHERE id=?",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "doc_id": row["doc_id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]),
        }

    async def get_memory_record(self, memory_id: int):
        return await self.get_memory(memory_id)

    async def get_memory_source(self, memory_id: int):
        cursor = await self.db_connection.execute(
            "SELECT source_json FROM memory_sources WHERE memory_id=?",
            (memory_id,),
        )
        row = await cursor.fetchone()
        return json.loads(row["source_json"]) if row else []

    async def save_memory_source(self, memory_id: int, source_messages: list):
        now = time.time()
        await self.db_connection.execute(
            "INSERT OR REPLACE INTO memory_sources(memory_id, source_json, created_at, updated_at) VALUES(?,?,?,?)",
            (memory_id, json.dumps(source_messages), now, now),
        )
        await self.db_connection.commit()


@pytest.mark.asyncio
async def test_crud_superseded_decay_and_merge_duplicate(tmp_path):
    no_db = _DummyCrudEngine(None)
    assert await no_db.mark_memories_superseded([1, 2], 3) == 0
    assert await no_db.merge_duplicate_memory(1, importance=0.5) is None

    db = await aiosqlite.connect(str(tmp_path / "crud.db"))
    db.row_factory = aiosqlite.Row
    try:
        await db.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, doc_id TEXT, text TEXT, metadata TEXT)"
        )
        await db.execute(
            "CREATE TABLE memory_sources (memory_id INTEGER PRIMARY KEY, source_json TEXT, created_at REAL, updated_at REAL)"
        )
        await db.executemany(
            "INSERT INTO documents(id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
            [
                (
                    1,
                    "d1",
                    "likes coffee",
                    json.dumps({"status": "active", "importance": 0.5}),
                ),
                (
                    2,
                    "d2",
                    "likes coffee",
                    json.dumps({"status": "active", "importance": 0.5}),
                ),
                (
                    3,
                    "d3",
                    "merged body",
                    json.dumps(
                        {
                            "status": "active",
                            "previous_id": 4,
                            "guarded_merge_snapshot_v1": "fp-1",
                        }
                    ),
                ),
                (
                    4,
                    "d4",
                    "old body",
                    json.dumps({"status": "archived", "superseded_by": 3}),
                ),
            ],
        )
        await db.commit()

        engine = _DummyCrudEngine(db)
        engine.update_memory = AsyncMock(return_value=True)
        assert await engine.mark_memories_superseded([1, 2, 1], 1) == 1

        # Test _decay_updates_for_rows pure calculation + noop skip + protected skip
        rows = [
            {"id": 1, "metadata": json.dumps({"importance": 1.0, "access_count": 5})},
            {
                "id": 2,
                "metadata": json.dumps(
                    {
                        "importance": 0.01,
                        "access_count": 0,
                        "last_access_time": 0.0,
                    }
                ),
            },
            {
                "id": 3,
                "metadata": json.dumps(
                    {
                        "importance": 0.6,
                        "access_count": 4,
                        "last_access_time": 100.0,
                    }
                ),
            },
        ]
        updates, skipped = engine._decay_updates_for_rows(
            rows,
            decay_rate=0.01,
            days=1,
            access_window_start=50.0,
            max_access_count=10.0,
            access_decay_multiplier=0.5,
            protected_threshold=1.0,
        )
        assert len(updates) == 1 and skipped == 1

        # Test rollback_guarded_summary_merge
        assert await engine.rollback_guarded_summary_merge(3, "wrong-fp") is False
        assert await engine.rollback_guarded_summary_merge(3, "fp-1") is True
        d4 = await engine.get_memory(4)
        assert d4["metadata"]["status"] == "active"

        # Test merge_duplicate_memory without content change + source merge
        engine.replace_memory = AsyncMock(return_value=10)
        res_id = await engine.merge_duplicate_memory(
            1,
            content=None,
            importance=0.8,
            metadata={"topics": ["coffee"], "source_window": {"start": 0, "end": 1}},
            source_messages=[
                {"id": 1, "content": "msg1"},
                {"id": 1, "content": "msg1"},
            ],
        )
        assert res_id == 1

        # Content changed -> replace_memory raises -> marks index rebuild needed
        engine._mark_index_rebuild_needed = AsyncMock()
        engine.replace_memory = AsyncMock(side_effect=RuntimeError("replace boom"))
        assert (
            await engine.merge_duplicate_memory(
                1, content="completely new summary text", importance=0.8
            )
            is None
        )
        engine._mark_index_rebuild_needed.assert_awaited_once()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_crud_status_guard_and_update_edge_branches(tmp_path):
    db = await aiosqlite.connect(str(tmp_path / "crud2.db"))
    db.row_factory = aiosqlite.Row
    try:
        await db.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, doc_id TEXT, text TEXT, metadata TEXT)"
        )
        await db.execute(
            "INSERT INTO documents(id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
            (
                1,
                "d1",
                "archived mem",
                json.dumps({"status": "archived", "topics": ["a"]}),
            ),
        )
        await db.commit()
        engine = _DummyCrudEngine(db)

        assert await engine._update_memory_status_guarded(1, {"status": "bad"}) is False
        assert (
            await engine._update_memory_status_guarded(999, {"status": "active"})
            is False
        )

        # Reactivation graph indexing deferred -> marks repair needed & returns False
        engine.graph_memory_manager.index_memory = AsyncMock(return_value=False)
        assert (
            await engine._update_memory_status_guarded(1, {"status": "active"}) is False
        )
        row = await engine.get_memory(1)
        assert row["metadata"]["needs_graph_repair"] is True

        # Reactivation with vector missing -> marks repair needed & returns False
        engine.graph_memory_manager.index_memory = AsyncMock(return_value=True)
        engine._is_memory_vector_indexed = lambda _id: False
        assert (
            await engine._update_memory_status_guarded(1, {"status": "active"}) is False
        )

        # Reactivation with graph incomplete -> marks repair needed & returns False
        engine._is_memory_vector_indexed = lambda _id: True
        engine.graph_memory_manager.source_graph_index_complete = AsyncMock(
            return_value=False
        )
        assert (
            await engine._update_memory_status_guarded(1, {"status": "active"}) is False
        )

        # _delete_memory with index_writer=None & empty content update
        engine.index_writer = None
        assert await engine._delete_memory(1) is False
        assert await engine._update_memory(1, {"content": "   "}, None) is False
        assert (
            await engine._update_memory(1, {"metadata": {"note": "x"}}, None) is False
        )
    finally:
        await db.close()
