"""Batch 2 regression tests: P0-2, P0-3, P1-#8, P1-#9, P1-#12, P2-#15, P2-#18."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core import faiss_async_persist
from astrbot_plugin_dynamics_memory.core.faiss_async_persist import (
    _ReadersWriterLock,
    install_async_persist,
)
from astrbot_plugin_dynamics_memory.core.managers import backup_manager as bm_mod
from astrbot_plugin_dynamics_memory.core.managers.backup_manager import (
    BackupManager,
    _backup_sqlite_file,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.models.graph_models import (
    GraphEdge,
    GraphEntry,
    GraphNode,
)
from astrbot_plugin_dynamics_memory.core.models.memory_atom import (
    AtomType,
    MemoryAtom,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.backup_handler import (
    BackupHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.memory_handler_update import (
    MemoryHandlerUpdateMixin,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import PageApiUtils
from astrbot_plugin_dynamics_memory.core.processors.text_processor import TextProcessor
from astrbot_plugin_dynamics_memory.core.retrieval.bm25_retriever import BM25Retriever
from astrbot_plugin_dynamics_memory.storage.atom_store import AtomStore
from astrbot_plugin_dynamics_memory.storage.graph_store import GraphStore
from astrbot_plugin_dynamics_memory.storage.sqlite_connection import _PLUGIN_DB_FILES


class _StubFaissDB:
    """Minimal FaissVecDB stub writing documents rows to the same SQLite DB."""

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._next_id = 1

    async def insert(self, content: str, metadata: dict, id: int | None = None):
        doc_id = id if id is not None else self._next_id
        self._next_id = max(self._next_id, int(doc_id) + 1)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO documents "
                "(id, doc_id, text, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
                (
                    int(doc_id),
                    f"doc-{doc_id}",
                    content,
                    json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            await db.commit()
        return int(doc_id)

    async def delete_documents(self, doc_ids):
        ids = [int(i) for i in doc_ids]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"DELETE FROM documents WHERE id IN ({placeholders})", ids
                )
                await db.commit()
        return len(ids)


# ---------------------------------------------------------------------------
# P0-2 & P2-#18: GraphStore shared-edge re-attribution & indexed orphan cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p02_delete_memory_reattributes_shared_edge_without_cascading_other_memory(
    tmp_path: Path,
):
    db_path = str(tmp_path / "graph_p02.db")
    store = GraphStore(db_path)
    await store.initialize()

    alice = GraphNode("person", "Alice", "alice")
    bob = GraphNode("person", "Bob", "bob")
    node_map = await store.upsert_nodes([alice, bob])

    # Memory 101 creates the edge; memory 102 semantically merges into the same edge_id!
    edge_101 = GraphEdge(alice.node_key, bob.node_key, "collaborates_with", 101)
    edge_102 = GraphEdge(alice.node_key, bob.node_key, "collaborates_with", 102)
    edge_id_101 = await store.add_edge(edge_101, node_map)
    edge_id_102 = await store.add_edge(edge_102, node_map)
    assert edge_id_102 == edge_id_101

    entry_101 = await store.add_entry(
        GraphEntry(
            entry_key="fact:101:1",
            source_memory_id=101,
            session_id="s1",
            persona_id="p1",
            entry_type="fact",
            content="Alice collaborates with Bob on project Alpha",
            node_keys=[alice.node_key, bob.node_key],
            relation_type="collaborates_with",
        ),
        node_map,
        edge_id=edge_id_101,
    )
    await store.update_entry_vector_doc_id(entry_101, 5001)

    entry_102 = await store.add_entry(
        GraphEntry(
            entry_key="fact:102:1",
            source_memory_id=102,
            session_id="s1",
            persona_id="p1",
            entry_type="fact",
            content="Alice and Bob shipped project Alpha milestone",
            node_keys=[alice.node_key, bob.node_key],
            relation_type="collaborates_with",
        ),
        node_map,
        edge_id=edge_id_102,
    )
    await store.update_entry_vector_doc_id(entry_102, 5002)

    # Delete memory 101 -> shared edge MUST survive, re-attributed to 102,
    # and memory 102's graph_entry + FTS row MUST stay intact (no cascade ghost!).
    removed_vectors = await store.delete_memory(101)
    assert removed_vectors == [5001]

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        edge_row = await (
            await db.execute(
                "SELECT id, source_memory_id FROM graph_edges WHERE id = ?",
                (edge_id_101,),
            )
        ).fetchone()
        assert edge_row is not None
        assert int(edge_row["source_memory_id"]) == 102

        entries = await (
            await db.execute("SELECT id, source_memory_id FROM graph_entries")
        ).fetchall()
        assert [(int(r["id"]), int(r["source_memory_id"])) for r in entries] == [
            (entry_102, 102)
        ]

        fts_rows = await (
            await db.execute("SELECT entry_id FROM livingmemory_graph_entries_fts")
        ).fetchall()
        assert [int(r["entry_id"]) for r in fts_rows] == [entry_102]
        assert entry_101 not in [int(r["entry_id"]) for r in fts_rows]

    # Delete memory 102 -> edge, entries, FTS, and orphaned nodes all vanish cleanly.
    removed_vectors_2 = await store.delete_memory(102)
    assert removed_vectors_2 == [5002]

    async with aiosqlite.connect(db_path) as db:
        for table in (
            "graph_entries",
            "graph_edges",
            "graph_nodes",
            "graph_entry_nodes",
            "livingmemory_graph_entries_fts",
        ):
            cnt = (
                await (await db.execute(f"SELECT COUNT(*) FROM {table}")).fetchone()
            )[0]
            assert cnt == 0, f"expected {table} to be empty, got {cnt}"


@pytest.mark.asyncio
async def test_p02_batch_delete_memories_shared_edge_partial_and_full(tmp_path: Path):
    db_path = str(tmp_path / "graph_p02_batch.db")
    store = GraphStore(db_path)
    await store.initialize()

    rust = GraphNode("topic", "Rust", "rust")
    wasm = GraphNode("topic", "WASM", "wasm")
    node_map = await store.upsert_nodes([rust, wasm])

    edge_id = await store.add_edge(
        GraphEdge(rust.node_key, wasm.node_key, "targets", 201), node_map
    )
    assert (
        await store.add_edge(
            GraphEdge(rust.node_key, wasm.node_key, "targets", 202), node_map
        )
        == edge_id
    )
    assert (
        await store.add_edge(
            GraphEdge(rust.node_key, wasm.node_key, "targets", 203), node_map
        )
        == edge_id
    )

    e201 = await store.add_entry(
        GraphEntry(
            "e:201",
            201,
            "s1",
            "p1",
            "fact",
            "m201",
            node_keys=[rust.node_key, wasm.node_key],
        ),
        node_map,
        edge_id=edge_id,
    )
    e202 = await store.add_entry(
        GraphEntry(
            "e:202",
            202,
            "s1",
            "p1",
            "fact",
            "m202",
            node_keys=[rust.node_key, wasm.node_key],
        ),
        node_map,
        edge_id=edge_id,
    )
    e203 = await store.add_entry(
        GraphEntry(
            "e:203",
            203,
            "s1",
            "p1",
            "fact",
            "m203",
            node_keys=[rust.node_key, wasm.node_key],
        ),
        node_map,
        edge_id=edge_id,
    )
    await store.update_entry_vector_doc_ids({e201: 6001, e202: 6002, e203: 6003})

    removed = await store.batch_delete_memories([201, 202])
    assert removed == {201: [6001], 202: [6002]}

    async with aiosqlite.connect(db_path) as db:
        edge_owner = (
            await (
                await db.execute(
                    "SELECT source_memory_id FROM graph_edges WHERE id = ?", (edge_id,)
                )
            ).fetchone()
        )[0]
        assert int(edge_owner) == 203
        remaining = await (await db.execute("SELECT id FROM graph_entries")).fetchall()
        assert [int(r[0]) for r in remaining] == [e203]

    removed_last = await store.batch_delete_memories([203])
    assert removed_last == {203: [6003]}

    async with aiosqlite.connect(db_path) as db:
        assert (
            await (await db.execute("SELECT COUNT(*) FROM graph_edges")).fetchone()
        )[0] == 0
        assert (
            await (await db.execute("SELECT COUNT(*) FROM graph_nodes")).fetchone()
        )[0] == 0


@pytest.mark.asyncio
async def test_p218_orphan_node_cleanup_uses_indexed_edge_id_and_targets_candidates(
    tmp_path: Path,
):
    db_path = str(tmp_path / "graph_p218.db")
    store = GraphStore(db_path)
    await store.initialize()

    unrelated = GraphNode("topic", "Unrelated", "unrelated")
    shared = GraphNode("topic", "Shared", "shared")
    exclusive = GraphNode("topic", "Exclusive", "exclusive")
    node_map = await store.upsert_nodes([unrelated, shared, exclusive])

    edge_301 = await store.add_edge(
        GraphEdge(shared.node_key, exclusive.node_key, "relates_to", 301), node_map
    )
    await store.add_entry(
        GraphEntry(
            "e:301",
            301,
            "s1",
            "p1",
            "fact",
            "m301",
            node_keys=[shared.node_key, exclusive.node_key],
        ),
        node_map,
        edge_id=edge_301,
    )
    await store.add_entry(
        GraphEntry(
            "e:302", 302, "s1", "p1", "summary", "m302", node_keys=[shared.node_key]
        ),
        node_map,
    )

    await store.delete_memory(301)

    async with aiosqlite.connect(db_path) as db:
        idx_row = await (
            await db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_graph_entries_edge_id'"
            )
        ).fetchone()
        assert idx_row is not None

        remaining_ids = {
            int(r[0])
            for r in await (await db.execute("SELECT id FROM graph_nodes")).fetchall()
        }
        assert node_map[exclusive.node_key] not in remaining_ids
        assert node_map[shared.node_key] in remaining_ids
        assert node_map[unrelated.node_key] in remaining_ids

        # Verify EXPLAIN QUERY PLAN uses idx_graph_entries_edge_id (no SCAN graph_entries).
        plan_rows = await (
            await db.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT edge_id, MIN(source_memory_id) FROM graph_entries WHERE edge_id IN (?) GROUP BY edge_id",
                (edge_301,),
            )
        ).fetchall()
        plan_text = " ".join(str(r[3]) for r in plan_rows)
        assert "idx_graph_entries_edge_id" in plan_text

        # Verify orphan node deletion plan uses PRIMARY KEY + indexed NOT EXISTS lookups.
        orphan_plan = await (
            await db.execute(
                "EXPLAIN QUERY PLAN "
                "DELETE FROM graph_nodes WHERE id IN (?) "
                "AND NOT EXISTS (SELECT 1 FROM graph_edges WHERE source_node_id = graph_nodes.id) "
                "AND NOT EXISTS (SELECT 1 FROM graph_edges WHERE target_node_id = graph_nodes.id) "
                "AND NOT EXISTS (SELECT 1 FROM graph_entry_nodes WHERE node_id = graph_nodes.id)",
                (node_map[shared.node_key],),
            )
        ).fetchall()
        orphan_text = " ".join(str(r[3]) for r in orphan_plan)
        assert "SCAN graph_nodes" not in orphan_text
        assert "SCAN graph_edges" not in orphan_text
        assert "SCAN graph_entry_nodes" not in orphan_text


# ---------------------------------------------------------------------------
# P0-3: Unified path write lock, safe VACUUM, auto_vacuum=INCREMENTAL detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p03_concurrent_writes_across_stores_and_vacuum_serialized(
    tmp_path: Path,
):
    db_path = tmp_path / "livingmemory.db"
    graph_docs_path = tmp_path / "livingmemory_graph_documents.db"

    # Seed companion DB with auto_vacuum = NONE (0) to test detection + VACUUM upgrade.
    with sqlite3.connect(str(graph_docs_path)) as conn:
        conn.execute("PRAGMA auto_vacuum = NONE")
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT)")
        conn.execute("INSERT INTO documents (id, text) VALUES (1, 'seed')")
        conn.commit()

    engine = MemoryEngine(
        db_path=str(db_path), faiss_db=_StubFaissDB(db_path), config={}
    )
    await engine.initialize()
    graph_store = GraphStore(str(db_path))
    await graph_store.initialize()
    atom_store = AtomStore(str(db_path))
    await atom_store.initialize()
    bm25 = BM25Retriever(str(db_path), TextProcessor())
    await bm25.initialize()

    try:
        # First detect companion auto_vacuum=NONE without running VACUUM.
        no_vac = await engine.maintain_storage(vacuum=False)
        assert no_vac["success"] is True
        comp_before = no_vac["companion_dbs"]["livingmemory_graph_documents.db"]
        assert comp_before["auto_vacuum_mode"] == 0
        assert comp_before["needs_vacuum_for_incremental"] is True

        # Concurrently hammer GraphStore, AtomStore, BM25Retriever, and maintain_storage(vacuum=True).
        async def _write_graph(i: int):
            node = GraphNode("topic", f"K{i}", f"k{i}")
            nid = await graph_store.upsert_node(node)
            await graph_store.add_entry(
                GraphEntry(
                    f"g:{i}",
                    1000 + i,
                    "s1",
                    "p1",
                    "summary",
                    f"graph text {i}",
                    node_keys=[node.node_key],
                ),
                {node.node_key: nid},
            )

        async def _write_atom(i: int):
            await atom_store.insert(
                MemoryAtom(
                    parent_memory_id=2000 + i,
                    atom_type=AtomType.PREFERENCE,
                    content=f"atom content {i}",
                    session_id="s1",
                )
            )

        async def _write_bm25(i: int):
            await bm25.add_document(3000 + i, f"bm25 content {i}")

        tasks = (
            [_write_graph(i) for i in range(6)]
            + [_write_atom(i) for i in range(6)]
            + [_write_bm25(i) for i in range(6)]
            + [engine.maintain_storage(vacuum=True)]
        )
        results = await asyncio.gather(*tasks)
        vac_report = results[-1]
        assert vac_report["success"] is True
        assert vac_report["auto_vacuum_mode"] == 2
        assert vac_report["needs_vacuum_for_incremental"] is False

        comp_after = vac_report["companion_dbs"]["livingmemory_graph_documents.db"]
        assert comp_after["auto_vacuum_mode"] == 2
        assert comp_after["needs_vacuum_for_incremental"] is False
    finally:
        await engine.close()


# ---------------------------------------------------------------------------
# P1-#8 & P2-#15: Decay no-op + corrupt JSON guard + storage caps + safe backup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p18_p215_decay_skips_compact_noop_and_survives_corrupt_json(
    tmp_path: Path,
):
    db_path = tmp_path / "decay_p18.db"
    engine = MemoryEngine(
        db_path=str(db_path), faiss_db=_StubFaissDB(db_path), config={}
    )
    await engine.initialize()
    try:
        # Drop any expression indexes on documents.metadata to simulate a legacy DB
        # containing a pre-existing malformed JSON row.
        idx_rows = await (
            await engine.db_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='documents' AND sql IS NOT NULL"
            )
        ).fetchall()
        for idx_row in idx_rows:
            await engine.db_connection.execute(f'DROP INDEX IF EXISTS "{idx_row[0]}"')

        # Row 1: compact JSON already at floor -> MUST be skipped as no-op!
        compact_floor = json.dumps(
            {"importance": 0.01, "access_count": 0, "last_access_time": 1.0},
            separators=(",", ":"),
        )
        # Row 2: malformed JSON -> MUST NOT abort decay!
        corrupt_raw = "{not-valid-json"
        # Row 3: normal active row -> MUST decay cleanly!
        active_meta = json.dumps(
            {"importance": 0.5, "access_count": 4, "last_access_time": 1.0},
            separators=(",", ":"),
        )
        await engine.db_connection.executemany(
            "INSERT INTO documents (id, doc_id, text, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
            [
                (1, "d1", "floor", compact_floor),
                (2, "d2", "corrupt", corrupt_raw),
                (3, "d3", "active", active_meta),
            ],
        )
        await engine.db_connection.commit()

        # Pure computation check: compact floor produces 0 updates, corrupt row skipped safely.
        updates, skipped = engine._decay_updates_for_rows(
            [
                {"id": 1, "metadata": compact_floor},
                {"id": 2, "metadata": corrupt_raw},
            ],
            decay_rate=0.1,
            days=1,
            access_window_start=time.time() - 86400 * 30,
            max_access_count=10.0,
            access_decay_multiplier=0.5,
            protected_threshold=0.6,
        )
        assert updates == []
        assert skipped == 2

        # End-to-end apply_daily_decay must succeed despite row 2's corrupt JSON.
        affected = await engine.apply_daily_decay(decay_rate=0.1, days=1)
        assert affected >= 1

        row3 = await (
            await engine.db_connection.execute(
                "SELECT metadata FROM documents WHERE id = 3"
            )
        ).fetchone()
        meta3 = json.loads(row3["metadata"])
        assert meta3["importance"] < 0.5
        assert meta3["access_count"] == 2
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_p18_bounds_provenance_sources_history_and_drops_legacy_fts(
    tmp_path: Path,
):
    db_path = tmp_path / "bloat_p18.db"
    engine = MemoryEngine(
        db_path=str(db_path), faiss_db=_StubFaissDB(db_path), config={}
    )
    await engine.initialize()
    try:
        # 1. narrative_provenance capped at 64 entries.
        provenance = [{"event_id": f"e-{i}"} for i in range(100)]
        mid = await engine.add_memory(
            content="bounded provenance",
            session_id="s1",
            persona_id="p1",
            importance=0.5,
            metadata={"narrative_provenance": provenance},
        )
        rec = await engine.get_memory_record(mid)
        assert rec is not None
        meta_obj = (
            rec["metadata"]
            if isinstance(rec["metadata"], dict)
            else json.loads(rec["metadata"])
        )
        stored_prov = meta_obj["narrative_provenance"]
        assert len(stored_prov) == 64
        assert stored_prov[0]["event_id"] == "e-36"
        assert stored_prov[-1]["event_id"] == "e-99"

        # 2. save_memory_source caps single row <= 256 KiB.
        huge_messages = [
            {"role": "user", "content": f"msg-{i}-" + ("x" * 10000)} for i in range(40)
        ]
        await engine.save_memory_source(mid, huge_messages)
        row = await (
            await engine.db_connection.execute(
                "SELECT source_json FROM memory_sources WHERE memory_id = ?", (mid,)
            )
        ).fetchone()
        raw_bytes = row["source_json"].encode("utf-8")
        assert len(raw_bytes) <= 256 * 1024
        parsed_msgs = json.loads(row["source_json"])
        assert isinstance(parsed_msgs, list) and len(parsed_msgs) > 0

        # 3. prune_memory_sources removes orphans & expired rows and flips has_source=false.
        old_ts = time.time() - 200 * 86400
        await engine.db_connection.execute(
            "INSERT INTO memory_sources (memory_id, source_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (999999, "[]", old_ts, old_ts),  # orphan
        )
        await engine.db_connection.execute(
            "UPDATE memory_sources SET updated_at = ? WHERE memory_id = ?",
            (old_ts, mid),
        )
        await engine.db_connection.commit()

        prune_stats = await engine.prune_memory_sources(
            archived_retention_days=90,
            active_retention_days=180,
            max_total_bytes=100 * 1024 * 1024,
        )
        assert prune_stats["orphans_removed"] == 1
        assert prune_stats["expired_removed"] == 1
        rec_after = await engine.get_memory_record(mid)
        meta_after = (
            rec_after["metadata"]
            if isinstance(rec_after["metadata"], dict)
            else json.loads(rec_after["metadata"])
        )
        assert meta_after.get("has_source") is False

        # 4. maintain_storage drops legacy FTS backup table.
        await engine.db_connection.execute(
            "CREATE TABLE livingmemory_legacy_documents_fts_backup (id INTEGER, text TEXT)"
        )
        await engine.db_connection.commit()
        report = await engine.maintain_storage(vacuum=False)
        assert report["success"] is True
        legacy = await (
            await engine.db_connection.execute(
                "SELECT name FROM sqlite_master WHERE name='livingmemory_legacy_documents_fts_backup'"
            )
        ).fetchone()
        assert legacy is None

        # 5. update_history value truncation.
        truncated = PageApiUtils._history_value("A" * 500)
        assert len(truncated) <= 241
        assert truncated.endswith("…")
    finally:
        await engine.close()


def test_p18_backup_sqlite_no_copy2_fallback_and_sidecar_meta(
    tmp_path: Path, monkeypatch
):
    assert "evidence_sidecar.sqlite3" in _PLUGIN_DB_FILES

    # Build a real SQLite DB file.
    db_src = tmp_path / "livingmemory.db"
    with sqlite3.connect(str(db_src)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'ok')")
        conn.commit()

    # 1. Verify online backup passes pages=256 and raises without shutil.copy2 on sqlite3.Error.
    captured_kwargs: list[dict] = []
    real_connect = sqlite3.connect

    class _FailingBackupConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, *a, **kw):
            return self._inner.execute(*a, **kw)

        def backup(self, target, **kw):
            captured_kwargs.append(kw)
            raise sqlite3.OperationalError("database is locked")

        def close(self):
            self._inner.close()

    def _patched_connect(path, *a, **kw):
        conn = real_connect(path, *a, **kw)
        if Path(path) == db_src:
            return _FailingBackupConn(conn)
        return conn

    copy_called = False

    def _forbid_copy2(*a, **kw):
        nonlocal copy_called
        copy_called = True
        raise AssertionError(
            "shutil.copy2 MUST NOT be called as fallback for live SQLite DB"
        )

    monkeypatch.setattr(bm_mod.sqlite3, "connect", _patched_connect)
    monkeypatch.setattr(bm_mod.shutil, "copy2", _forbid_copy2)

    dst = tmp_path / "out.db"
    with pytest.raises(sqlite3.OperationalError):
        _backup_sqlite_file(db_src, dst)
    assert captured_kwargs and captured_kwargs[0].get("pages") == 256
    assert not copy_called
    assert not dst.exists()

    # Also ensure backup_if_needed aborts cleanly without writing .plugin_version or archive.
    mgr = BackupManager(str(tmp_path), compress=True)
    with pytest.raises(sqlite3.OperationalError):
        mgr.backup_if_needed()
    assert not (tmp_path / ".plugin_version").exists()
    assert not list((tmp_path / "backups").glob("*.tar.gz"))


def test_p18_backup_sidecar_meta_and_schema_bak_pruning(tmp_path: Path, monkeypatch):
    db_src = tmp_path / "livingmemory.db"
    with sqlite3.connect(str(db_src)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.commit()
    (tmp_path / ".plugin_version").write_text("1.0.0", encoding="utf-8")

    mgr = BackupManager(str(tmp_path), compress=True)
    archive_path = mgr.backup_if_needed()
    assert archive_path is not None
    meta_sidecar = Path(archive_path).parent / "v1.0.0.meta.json"
    assert meta_sidecar.is_file()

    # list_backups MUST read from sidecar .meta.json without opening .tar.gz!
    def _forbid_tar_open(*a, **kw):
        raise AssertionError(
            "list_backups should use .meta.json sidecar instead of tarfile.open"
        )

    monkeypatch.setattr(tarfile, "open", _forbid_tar_open)
    listed = BackupManager.list_backups(str(tmp_path))
    assert len(listed) == 1
    assert listed[0]["name"] == "v1.0.0"
    assert listed[0]["previous_version"] == "1.0.0"

    # Stale evidence_sidecar.sqlite3.schema-*.bak files should be pruned keeping newest 1.
    old_bak1 = tmp_path / "evidence_sidecar.sqlite3.schema-v1-20260101T000000Z.bak"
    old_bak2 = tmp_path / "evidence_sidecar.sqlite3.schema-v1-20260102T000000Z.bak"
    new_bak = tmp_path / "evidence_sidecar.sqlite3.schema-v2-20261124T000000Z.bak"
    ages_days = (30, 10, 1)
    for age, p in zip(ages_days, (old_bak1, old_bak2, new_bak)):
        p.write_bytes(b"bak-data")
        mtime = time.time() - age * 86400
        os.utime(p, (mtime, mtime))

    prune_stats = BackupManager.prune(str(tmp_path), keep_days=7, max_total_mb=500)
    assert prune_stats["files_removed"] >= 2
    assert not old_bak1.exists()
    assert not old_bak2.exists()
    assert new_bak.exists()


@pytest.mark.asyncio
async def test_p18_backup_handler_offloads_list_backups(tmp_path: Path, monkeypatch):
    handler = BackupHandler(PageApiUtils(), str(tmp_path))
    called_via_thread = False
    real_to_thread = asyncio.to_thread

    async def _spy_to_thread(fn, *args, **kwargs):
        nonlocal called_via_thread
        if fn is BackupManager.list_backups:
            called_via_thread = True
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _spy_to_thread)
    res = await handler.list_backups()
    assert res["status"] == "ok"
    assert called_via_thread is True


# ---------------------------------------------------------------------------
# P1-#9: AsyncIndexPersister double CancelledError & writer-preferring RWLock
# ---------------------------------------------------------------------------


class _MockVec:
    def reshape(self, *args):
        return self


class _MockIndex:
    def __init__(self, started: threading.Event, release: threading.Event):
        self._started = started
        self._release = release
        self.ntotal = 1

    def search(self, vector, k):
        self._started.set()
        assert self._release.wait(3.0)
        return [[0.0]], [[1]]


class _MockStorage:
    def __init__(self, path: Path, index: _MockIndex):
        self.path = str(path)
        self.index = index
        self.mutations = 0

    async def save_index(self) -> None:
        return None

    async def insert_batch(self, vectors, ids) -> None:
        self.mutations += 1
        self.index.ntotal += len(ids)


@pytest.mark.asyncio
async def test_p19_double_cancel_keeps_read_lock_held_until_worker_finishes(
    tmp_path: Path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()
    storage = _MockStorage(tmp_path / "mock.index", _MockIndex(started, release))
    persister = install_async_persist(storage, debounce_seconds=600)

    try:
        # 1. Double-cancel search() while worker thread is blocked.
        search_task = asyncio.create_task(persister.search(_MockVec(), 1))
        assert await asyncio.to_thread(started.wait, 2.0)

        writer_task = asyncio.create_task(storage.insert_batch([[0.3, 0.4]], [2]))
        search_task.cancel()
        await asyncio.sleep(0.01)
        search_task.cancel()  # Second CancelledError!
        await asyncio.sleep(0.02)

        # Writer MUST still be blocked because the worker thread is still inside index.search!
        assert not writer_task.done()
        assert storage.mutations == 0

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await search_task
        await writer_task
        assert storage.mutations == 1

        # 2. Double-cancel _do_flush() while _write_index_atomic is blocked.
        flush_started = threading.Event()
        flush_release = threading.Event()

        def _blocking_write(index, path):
            flush_started.set()
            assert flush_release.wait(3.0)

        monkeypatch.setattr(faiss_async_persist, "_write_index_atomic", _blocking_write)
        try:
            persister._dirty = True
            flush_task = asyncio.create_task(persister._do_flush(force=True))
            assert await asyncio.to_thread(flush_started.wait, 2.0)

            writer_task2 = asyncio.create_task(storage.insert_batch([[0.5, 0.6]], [3]))
            flush_task.cancel()
            await asyncio.sleep(0.01)
            flush_task.cancel()  # Second CancelledError!
            await asyncio.sleep(0.02)

            assert not writer_task2.done()
            assert storage.mutations == 1

            flush_release.set()
            with pytest.raises(asyncio.CancelledError):
                await flush_task
            await writer_task2
            assert storage.mutations == 2
        finally:
            flush_release.set()
    finally:
        release.set()
        await persister.aclose()


@pytest.mark.asyncio
async def test_p19_rwlock_writer_preference_prevents_starvation():
    lock = _ReadersWriterLock()
    order: list[str] = []
    r1_entered = asyncio.Event()
    r1_release = asyncio.Event()

    async def reader1():
        async with lock.read():
            order.append("r1_enter")
            r1_entered.set()
            await r1_release.wait()
            order.append("r1_exit")

    async def writer():
        async with lock.write():
            order.append("w_enter")
            order.append("w_exit")

    async def reader2():
        async with lock.read():
            order.append("r2_enter")
            order.append("r2_exit")

    t_r1 = asyncio.create_task(reader1())
    await r1_entered.wait()

    t_w = asyncio.create_task(writer())
    await asyncio.sleep(0.01)
    t_r2 = asyncio.create_task(reader2())
    await asyncio.sleep(0.01)

    r1_release.set()
    await asyncio.gather(t_r1, t_w, t_r2)

    assert order.index("w_enter") < order.index("r2_enter")


# ---------------------------------------------------------------------------
# P1-#12: batch_update_memories status transitions maintain indexes
# ---------------------------------------------------------------------------


class _BatchUpdateHarness(MemoryHandlerUpdateMixin):
    def __init__(self):
        self.utils = PageApiUtils()


@pytest.mark.asyncio
async def test_p112_batch_update_memories_status_maintains_indexes(monkeypatch):
    from astrbot_plugin_dynamics_memory.core.page_api_modules import (
        memory_handler_update as mod,
    )

    records = {
        1: {"id": 1, "metadata": {"status": "active"}},
        2: {"id": 2, "metadata": {"status": "archived"}},
        3: {"id": 3, "metadata": {"status": "active"}},
    }
    engine = SimpleNamespace(
        archive_memories=AsyncMock(return_value=1),
        restore_memory=AsyncMock(return_value=True),
        delete_memory=AsyncMock(return_value=True),
        update_memory=AsyncMock(return_value=True),
        get_memory_record=AsyncMock(side_effect=lambda mid: records.get(mid)),
    )
    handler = _BatchUpdateHarness()

    async def _run(payload: dict):
        monkeypatch.setattr(
            mod,
            "request",
            SimpleNamespace(get_json=AsyncMock(return_value=payload)),
        )
        return await handler.batch_update_memories(engine)

    # Archive -> archive_memories([1])
    res_arch = await _run({"memory_ids": [1], "field": "status", "value": "archived"})
    assert res_arch["status"] == "ok"
    assert res_arch["data"]["updated_count"] == 1
    engine.archive_memories.assert_awaited_once_with([1])
    engine.update_memory.assert_not_awaited()

    # Restore archived -> restore_memory(2)
    res_rest = await _run({"memory_ids": [2], "field": "status", "value": "active"})
    assert res_rest["status"] == "ok"
    assert res_rest["data"]["updated_count"] == 1
    engine.restore_memory.assert_awaited_once_with(2)
    engine.update_memory.assert_not_awaited()

    # Delete -> delete_memory(3)
    res_del = await _run({"memory_ids": [3], "field": "status", "value": "deleted"})
    assert res_del["status"] == "ok"
    assert res_del["data"]["updated_count"] == 1
    engine.delete_memory.assert_awaited_once_with(3)
    engine.update_memory.assert_not_awaited()
