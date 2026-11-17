"""Targeted unit tests covering remaining branches of MemoryEngineWriteOpsMixin."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core.managers.memory_engine_write_ops import (
    MemoryEngineWriteOpsMixin,
)
from astrbot_plugin_dynamics_memory.core.maintenance.source_state_guard import (
    SourceStateGuard,
)
from astrbot_plugin_dynamics_memory.core.models.memory_atom import (
    AtomStatus,
    AtomType,
    DecayType,
    MemoryAtom,
)


class _DummyWriteOpsEngine(MemoryEngineWriteOpsMixin):
    def __init__(self, db_connection=None):
        self.db_connection = db_connection
        self.config = {}
        self.faiss_db = SimpleNamespace(delete=AsyncMock(return_value=True))
        self.atom_enabled = True
        self.atom_store = None
        self.hybrid_retriever = SimpleNamespace(
            update_metadata=AsyncMock(return_value=True),
            delete_memory=AsyncMock(return_value=True),
        )
        self.dual_route_retriever = None
        self.graph_memory_manager = None
        self._write_op_max_retries = 3
        self.source_state_guard = SourceStateGuard()
        self.source_loader = None
        self.invalidated = 0

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

    async def delete_memory(self, memory_id: int):
        await self.db_connection.execute(
            "DELETE FROM documents WHERE id=?", (memory_id,)
        )
        await self.db_connection.commit()
        return True

    async def _delete_graph_and_atoms_for_batch(self, memory_ids: list[int]):
        self.deleted_batch_graph_atoms = list(memory_ids)


async def _create_schema(db):
    await db.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, doc_id TEXT, text TEXT, metadata TEXT)"
    )
    await db.execute(
        "CREATE TABLE memory_sources (memory_id INTEGER PRIMARY KEY, source_json TEXT, created_at REAL, updated_at REAL)"
    )
    await db.execute(
        "CREATE VIRTUAL TABLE livingmemory_memories_fts USING fts5(doc_id, text)"
    )
    await db.commit()


@pytest.mark.asyncio
async def test_none_db_guards_and_repair_retained_sources_edges(tmp_path):
    no_db = _DummyWriteOpsEngine(None)
    assert await no_db.repair_retained_sources() == {
        "repaired": 0,
        "available": False,
    }
    assert await no_db.stored_summary_window("s1", 0, 1) is None
    await no_db._create_write_ops_table()
    assert await no_db._start_write_op("add") is None
    await no_db._advance_write_op(None, "step")
    assert await no_db._repair_incomplete_write_ops() == 0
    assert await no_db._delete_document_indexes_for_batch([]) == 0
    assert await no_db.cleanup_write_ops() == 0

    db = await aiosqlite.connect(str(tmp_path / "ops.db"))
    db.row_factory = aiosqlite.Row
    try:
        await _create_schema(db)
        engine = _DummyWriteOpsEngine(db)
        assert await engine.cleanup_write_ops() == 0
        await engine._create_write_ops_table()

        prov_a = [{"event_id": "e1"}]
        prov_b = [{"event_id": "e2"}]
        await db.executemany(
            "INSERT INTO documents(id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
            [
                (1, "d1", "bad", json.dumps({"narrative_provenance": prov_a})),
                (2, "d2", "empty", json.dumps({"narrative_provenance": prov_a})),
                (3, "d3", "changed", json.dumps({"narrative_provenance": prov_a})),
                (4, "d4", "ok", json.dumps({"narrative_provenance": prov_a})),
                (
                    5,
                    "d5",
                    "snap",
                    json.dumps(
                        {
                            "source_session_id": "s1",
                            "source_window": {"snapshot_fingerprint": "fp-1"},
                            "status": "active",
                            "has_source": 0,
                        }
                    ),
                ),
            ],
        )
        await db.commit()
        assert (
            await engine.stored_summary_window("s1", 0, 1, snapshot_fingerprint="fp-1")
            == 5
        )

        calls = 0

        async def loader(meta):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyError("missing event")
            if calls == 2:
                return []
            if calls == 3:
                await db.execute(
                    "UPDATE documents SET metadata=? WHERE id=3",
                    (json.dumps({"narrative_provenance": prov_b}),),
                )
                await db.commit()
                return [{"content": "stale"}]
            return [{"content": "restored"}]

        engine.source_loader = loader
        res = await engine.repair_retained_sources(limit=10)
        assert res["available"] is True
        assert res["repaired"] == 1
        assert res["source_unavailable"] == 3
        assert engine.invalidated >= 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_write_ops_serialize_deserialize_requeue_and_cleanup(tmp_path):
    db = await aiosqlite.connect(str(tmp_path / "ops2.db"))
    db.row_factory = aiosqlite.Row
    try:
        await _create_schema(db)
        engine = _DummyWriteOpsEngine(db)
        await engine._create_write_ops_table()

        atom = MemoryAtom(
            parent_memory_id=5,
            atom_type=AtomType.FACTUAL,
            content="prefers tea",
            entities=["tea"],
            importance=0.8,
            confidence=0.9,
            decay_type=DecayType.EXPONENTIAL,
            status=AtomStatus.ACTIVE,
            session_id="s1",
            persona_id="p1",
            metadata={"k": "v"},
        )
        serialized = engine._serialize_atom_for_repair(atom)
        rebuilt = engine._deserialize_atom_from_repair(serialized, 9, "s2", "p2")
        assert rebuilt is not None and rebuilt.parent_memory_id == 9

        assert (
            engine._deserialize_atom_from_repair({"content": "  "}, 1, None, None)
            is None
        )
        bad_enums = engine._deserialize_atom_from_repair(
            {
                "content": "valid",
                "atom_type": "invalid_type",
                "decay_type": "invalid_decay",
                "status": "invalid_status",
            },
            1,
            "s-fb",
            "p-fb",
        )
        assert bad_enums.atom_type == AtomType.UNKNOWN
        assert bad_enums.decay_type == DecayType.EXPONENTIAL
        assert bad_enums.status == AtomStatus.ACTIVE

        # Failed sqlite lock op requeue & cleanup_write_ops
        op_lock = await engine._start_write_op("add", {"a": 1})
        await db.execute(
            "UPDATE memory_write_ops SET payload='{bad' WHERE id=?", (op_lock,)
        )
        await db.commit()
        await engine._advance_write_op(
            op_lock,
            "failed_step",
            status="failed",
            error="database is locked",
            payload_patch={"b": 2},
        )
        assert await engine.requeue_sqlite_failures() == 1
        old_ts = time.time() - 30 * 86400
        await db.execute(
            "UPDATE memory_write_ops SET status='completed', updated_at=? WHERE id=?",
            (old_ts, op_lock),
        )
        await db.commit()
        assert await engine.cleanup_write_ops(retention_days=14) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_repair_add_published_delete_and_batch_error_branches(tmp_path):
    db = await aiosqlite.connect(str(tmp_path / "ops3.db"))
    db.row_factory = aiosqlite.Row
    try:
        await _create_schema(db)
        engine = _DummyWriteOpsEngine(db)
        await engine._create_write_ops_table()

        # Ambiguous unpublished candidates (>1)
        op_ambig = await engine._start_write_op(
            "add", {"source_validation_required": True}
        )
        await db.executemany(
            "INSERT INTO documents(id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
            [
                (
                    10,
                    "d10",
                    "t10",
                    json.dumps(
                        {"write_op_id": op_ambig, "status": "pending_source_validation"}
                    ),
                ),
                (
                    11,
                    "d11",
                    "t11",
                    json.dumps(
                        {"write_op_id": op_ambig, "status": "pending_source_validation"}
                    ),
                ),
            ],
        )
        await db.commit()
        assert (
            await engine._repair_add_write_op(
                op_ambig, None, {"source_validation_required": True}
            )
            is False
        )

        # Published source write with graph failure -> records source_published_graph_pending
        op_pub = await engine._start_write_op(
            "add", {"source_validation_required": True}, memory_id=12
        )
        pub_meta = {
            "write_op_id": op_pub,
            "status": "active",
            "source_publication_verified_v1": {"write_op_id": op_pub, "memory_id": 12},
        }
        await db.execute(
            "INSERT INTO documents(id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
            (12, "d12", "published text", json.dumps(pub_meta)),
        )
        await db.commit()
        engine.atom_store = SimpleNamespace(
            get_by_parent=AsyncMock(return_value=[]),
            insert_many=AsyncMock(),
            delete_by_parent=AsyncMock(),
        )
        engine.graph_memory_manager = SimpleNamespace(
            index_memory=AsyncMock(return_value=False),
            delete_memory=AsyncMock(),
        )
        assert (
            await engine._repair_add_write_op(
                op_pub, 12, {"source_validation_required": True}
            )
            is False
        )
        engine.graph_memory_manager.index_memory = AsyncMock(
            side_effect=RuntimeError("graph boom")
        )
        assert (
            await engine._repair_add_write_op(
                op_pub, 12, {"source_validation_required": True}
            )
            is False
        )
        engine.graph_memory_manager.index_memory = AsyncMock(return_value=True)
        assert await engine._repair_incomplete_write_ops() >= 1

        # Add repair with failed_atoms dedup + missing document + loader fallback
        op_missing = await engine._start_write_op("add", {}, memory_id=9999)
        assert await engine._repair_add_write_op(op_missing, 9999, {}) is False

        existing_atom = MemoryAtom(
            parent_memory_id=13,
            atom_type=AtomType.FACTUAL,
            content="already there",
            session_id="s1",
            persona_id="p1",
        )
        engine.atom_store.get_by_parent = AsyncMock(return_value=[existing_atom])
        await db.execute(
            "INSERT INTO documents(id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
            (
                13,
                "d13",
                "mem 13",
                json.dumps(
                    {"has_source": True, "session_id": "s1", "persona_id": "p1"}
                ),
            ),
        )
        await db.commit()
        engine.source_loader = AsyncMock(return_value=[{"content": "loaded"}])
        op_atoms = await engine._start_write_op("add", {}, memory_id=13)
        assert (
            await engine._repair_add_write_op(
                op_atoms,
                13,
                {
                    "failed_atoms": [
                        {
                            "content": "already there",
                            "atom_type": "factual",
                            "session_id": "s1",
                            "persona_id": "p1",
                        },
                        {
                            "content": "new atom",
                            "atom_type": "factual",
                            "session_id": "s1",
                            "persona_id": "p1",
                        },
                    ]
                },
            )
            is True
        )
        engine.atom_store.insert_many.assert_awaited_once()

        # Delete repair when document still exists
        await db.execute(
            "INSERT INTO documents(id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
            (14, "d14", "to delete", "{}"),
        )
        await db.commit()
        op_del = await engine._start_write_op("delete", {}, memory_id=14)
        assert await engine._repair_delete_write_op(op_del, 14) is True
        engine.hybrid_retriever.delete_memory.assert_awaited_with(14)
        engine.graph_memory_manager.delete_memory.assert_awaited_with(14)
        engine.atom_store.delete_by_parent.assert_awaited_with(14)

        # Batch delete faiss failure raises RuntimeError
        await db.execute(
            "INSERT INTO documents(id, doc_id, text, metadata) VALUES (?, ?, ?, ?)",
            (15, "d15", "batch fail", "{}"),
        )
        await db.commit()
        engine.faiss_db.delete = AsyncMock(return_value=False)
        with pytest.raises(RuntimeError, match="Vector deletion was not confirmed"):
            await engine._delete_document_indexes_for_batch([15])

        # _repair_incomplete_write_ops DB exception guard and CancelledError propagation
        engine_err = _DummyWriteOpsEngine(
            SimpleNamespace(
                execute=AsyncMock(side_effect=RuntimeError("db read failure"))
            )
        )
        assert await engine_err._repair_incomplete_write_ops() == 0

        engine_cancel = _DummyWriteOpsEngine(
            SimpleNamespace(execute=AsyncMock(side_effect=asyncio.CancelledError()))
        )
        with pytest.raises(asyncio.CancelledError):
            await engine_cancel._repair_incomplete_write_ops()
    finally:
        await db.close()
