"""Regression tests for the P0/P1 optimization wave in dynamics_memory.

Covers:
1. [P0] Cold-start retry on_initialized callback, @filter.on_astrbot_loaded wakeup,
   entry-point idempotent _ensure_runtime_components() self-healing, silent provider
   lookup via find_provider_silently, and 300s slow-probe instead of permanent lockup.
2. [P0] EvidenceSidecar.prune_old(), multi-SQLite daily maintenance
   (conversations.db, livingmemory_graph_documents.db, evidence_sidecar.sqlite3),
   and BackupManager SQLite online backup + tar.gz compression + include_faiss=False
   + evidence_sidecar inclusion + legacy compatibility.
3. [P1] MemorySearchTool concurrent multi-read_scopes + shared QueryEmbeddingCache +
   remote_deadline + concurrent include_source batching + bounded _json_result char
   budget; embedding telemetry reporting; and synchronous read-only get_series_health().
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from astrbot_plugin_dynamics_memory import series_panels
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.managers.backup_manager import (
    _BACKUP_INFO_FILE,
    BackupManager,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine_stats import (
    MemoryEngineStatsMixin,
)
from astrbot_plugin_dynamics_memory.core.memory_scope import MemoryReadScope
from astrbot_plugin_dynamics_memory.core.passive_group_capture import (
    get_active_plugin,
    set_active_plugin,
)
from astrbot_plugin_dynamics_memory.core.plugin_initializer import PluginInitializer
from astrbot_plugin_dynamics_memory.core.retrieval.vector_retriever import (
    VectorRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.vector_search import (
    QueryEmbeddingCache,
    query_embeddings,
    retrieve_vectors,
)
from astrbot_plugin_dynamics_memory.core.schedulers.decay_scheduler import (
    DecayScheduler,
)
from astrbot_plugin_dynamics_memory.core.tools.memory_search_tool import (
    DEFAULT_RESULT_CHAR_BUDGET,
    SOURCE_RESULT_CHAR_BUDGET,
    MemorySearchTool,
    _json_result,
)
from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar
from astrbot_plugin_dynamics_memory.storage.sqlite_connection import open_connection


# ==============================================================================
# 1. [P0] Cold-start retry callback, wakeup, self-healing, silent lookup, slow-probe
# ==============================================================================


@pytest.mark.asyncio
async def test_retry_initialization_invokes_on_initialized_callback(
    monkeypatch, tmp_path
):
    """Background retry completing must fire on_initialized immediately."""
    ctx = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={}),
        get_all_embedding_providers=lambda: [],
        get_all_providers=lambda: [],
        get_provider_by_id=Mock(return_value=None),
    )
    callback_called = asyncio.Event()

    async def on_ready():
        callback_called.set()

    init = PluginInitializer(
        ctx, ConfigManager(), str(tmp_path), on_initialized=on_ready
    )

    attempts = 0

    async def fake_providers(silent=True):
        nonlocal attempts
        attempts += 1
        if attempts >= 2:
            init.embedding_provider = object()
            init.llm_provider = object()

    async def fake_complete():
        init._initialization_complete = True

    monkeypatch.setattr(init, "_initialize_providers", fake_providers)
    monkeypatch.setattr(init, "_complete_initialization", fake_complete)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    await init._retry_initialization()

    assert init.is_initialized is True
    assert callback_called.is_set() is True


@pytest.mark.asyncio
async def test_on_astrbot_loaded_wakes_sleeping_retry_immediately(tmp_path):
    """@filter.on_astrbot_loaded wakes a sleeping retry task without waiting for backoff."""
    ctx = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={}),
        get_all_stars=lambda: {},
        get_all_embedding_providers=lambda: [],
        get_all_providers=lambda: [],
        get_provider_by_id=Mock(return_value=None),
    )
    ready_event = asyncio.Event()
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = ctx
    plugin._series_enabled = True
    plugin._terminating = False
    plugin.config_manager = ConfigManager()
    plugin.event_handler = None
    plugin.command_handler = None
    plugin.evidence_service = None
    plugin._background_tasks = set()
    plugin._component_init_lock = asyncio.Lock()
    plugin._llm_tools_registered = False

    async def on_ready():
        ready_event.set()

    init = PluginInitializer(
        ctx, plugin.config_manager, str(tmp_path), on_initialized=on_ready
    )
    init._long_retry_interval = 300.0
    plugin.initializer = init

    providers_available = False

    async def fake_providers(silent=True):
        if providers_available:
            init.embedding_provider = object()
            init.llm_provider = object()

    async def fake_complete():
        init._initialization_complete = True

    init._initialize_providers = fake_providers
    init._complete_initialization = fake_complete

    # Start background retry loop — it sleeps on 2.0s first backoff
    init._start_retry_task_if_needed()
    await asyncio.sleep(0.05)
    assert init.is_initialized is False

    # Providers become ready and AstrBot finishes loading -> immediate wake
    providers_available = True
    started = time.monotonic()
    await plugin.on_astrbot_loaded()
    await asyncio.wait_for(ready_event.wait(), timeout=1.0)
    elapsed = time.monotonic() - started

    assert init.is_initialized is True
    assert elapsed < 0.8
    await init.stop_background_tasks()


@pytest.mark.asyncio
async def test_entry_points_self_heal_missing_runtime_components():
    """handle_memory_recall, get_evidence_view, recall_context self-heal missing components."""
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = SimpleNamespace(get_all_stars=lambda: {})
    plugin._series_enabled = True
    plugin._terminating = False
    plugin.config_manager = ConfigManager()
    plugin.initializer = SimpleNamespace(
        is_initialized=True,
        is_failed=False,
        error_message=None,
        memory_engine=Mock(),
        memory_processor=Mock(),
        conversation_manager=Mock(),
    )
    plugin.event_handler = None
    plugin.evidence_service = None
    plugin._component_init_lock = asyncio.Lock()
    plugin._ensure_runtime_components = AsyncMock(return_value=False)

    # 1. handle_memory_recall
    event = Mock()
    event.get_extra = Mock(return_value=None)
    req = SimpleNamespace()
    await plugin.handle_memory_recall(event, req)
    assert plugin._ensure_runtime_components.await_count == 1

    # 2. get_evidence_view
    plugin._narrative_scope_allowed = AsyncMock(return_value=True)
    view = await plugin.get_evidence_view(
        {
            "platform": "qq",
            "umo": "qq:GroupMessage:1",
            "persona_id": "p",
            "instance_id": "p",
        }
    )
    assert view["available"] is False
    assert plugin._ensure_runtime_components.await_count == 2

    # 3. recall_context
    plugin._narrative_scope_allowed = AsyncMock(return_value=False)
    res = await plugin.recall_context(
        {
            "platform": "qq",
            "umo": "qq:GroupMessage:1",
            "persona_id": "p",
            "instance_id": "p",
        },
        "hello",
    )
    assert res["available"] is False
    assert plugin._ensure_runtime_components.await_count == 3


@pytest.mark.asyncio
async def test_cold_start_provider_lookup_uses_find_provider_silently(tmp_path):
    """During cold start, missing provider lookup never calls noisy context.get_provider_by_id."""
    get_by_id = Mock(
        side_effect=AssertionError("should not call noisy get_provider_by_id")
    )
    emb_stub = SimpleNamespace(id="emb-1")
    ctx = SimpleNamespace(
        provider_manager=SimpleNamespace(
            inst_map={"emb-1": emb_stub},
            embedding_provider_inst_map={},
        ),
        get_all_providers=lambda: [],
        get_all_embedding_providers=lambda: [],
        get_all_rerank_providers=lambda: [],
        get_all_stt_providers=lambda: [],
        get_provider_by_id=get_by_id,
    )
    init = PluginInitializer(ctx, ConfigManager(), str(tmp_path))

    assert init._get_provider_by_id("missing-pid", silent=True) is None
    assert init._get_provider_by_id("missing-pid", silent=False) is None
    assert init._get_provider_by_id("emb-1", silent=True) is emb_stub
    get_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_retry_slow_probe_300s_and_external_reset(monkeypatch, tmp_path):
    """Exceeding max attempts switches to 300s slow probe instead of permanent lockup, and reset_retry recovers."""
    ctx = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={}),
        get_all_embedding_providers=lambda: [],
        get_all_providers=lambda: [],
        get_provider_by_id=Mock(return_value=None),
    )
    init = PluginInitializer(ctx, ConfigManager(), str(tmp_path))
    init._max_provider_attempts = 2
    observed_delays: list[float] = []
    entered_slow_probe = asyncio.Event()
    providers_ready = False

    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float):
        observed_delays.append(delay)
        if delay >= 300.0:
            entered_slow_probe.set()
            # Park until woken by reset_retry()
            await asyncio.Event().wait()
        await real_sleep(0)

    async def fake_providers(silent=True):
        if providers_ready:
            init.embedding_provider = object()
            init.llm_provider = object()

    async def fake_complete():
        init._initialization_complete = True

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    init._initialize_providers = fake_providers
    init._complete_initialization = fake_complete

    init._start_retry_task_if_needed()
    await asyncio.wait_for(entered_slow_probe.wait(), timeout=2.0)

    assert init.is_failed is False
    assert init.in_slow_probe_mode is True
    assert 300.0 in observed_delays
    assert init.error_message is not None

    # External reset wakes slow probe immediately and completes initialization
    providers_ready = True
    init.reset_retry()
    assert await init.ensure_initialized(timeout=2.0) is True
    assert init.is_initialized is True
    assert init.in_slow_probe_mode is False
    await init.stop_background_tasks()


# ==============================================================================
# 2. [P0] EvidenceSidecar.prune_old, multi-SQLite maintenance, BackupManager
# ==============================================================================


@pytest.mark.asyncio
async def test_evidence_sidecar_prune_old_cleans_expired_and_preserves_active(tmp_path):
    """prune_old removes expired invalidated materials, orphan atoms/corrections, keeps active & supersede."""
    db_path = tmp_path / "evidence_sidecar.sqlite3"
    sidecar = EvidenceSidecar(db_path)
    await sidecar.initialize()
    try:
        now = time.time()
        old_ts = now - 45 * 86400

        with sqlite3.connect(str(db_path)) as conn:
            # 1. Active material + active atom -> MUST BE PRESERVED
            conn.execute(
                """INSERT INTO evidence_materials(
                    material_id,scope_key,material_kind,role,conversation_id,snapshot_fingerprint,
                    source_position,source_family_id,text_hash,projection_key,card_version,
                    status,diagnostics_code,created_at,updated_at
                ) VALUES ('mat_active','sc1','event','user','c1','fp1',0,'','h1','pk1','v1','ready','',?,?)""",
                (old_ts, old_ts),
            )
            conn.execute(
                """INSERT INTO evidence_atoms(
                    atom_id,scope_key,material_id,kind,status,root_ids_json,
                    root_material_ids_json,atom_json,revision,created_at,updated_at
                ) VALUES ('atm_active','sc1','mat_active','fact','active','[]','[]','{}',1,?,?)""",
                (old_ts, old_ts),
            )

            # 2. Expired invalidated material + invalidated atom -> MUST BE PRUNED
            conn.execute(
                """INSERT INTO evidence_materials(
                    material_id,scope_key,material_kind,role,conversation_id,snapshot_fingerprint,
                    source_position,source_family_id,text_hash,projection_key,card_version,
                    status,diagnostics_code,created_at,updated_at
                ) VALUES ('mat_expired','sc1','event','user','c1','fp1',1,'','h2','pk1','v1','invalidated','',?,?)""",
                (old_ts, old_ts),
            )
            conn.execute(
                """INSERT INTO evidence_atoms(
                    atom_id,scope_key,material_id,kind,status,root_ids_json,
                    root_material_ids_json,atom_json,revision,created_at,updated_at
                ) VALUES ('atm_expired','sc1','mat_expired','fact','invalidated','[]','[]','{}',1,?,?)""",
                (old_ts, old_ts),
            )

            # 3. Supersede case: old invalidated material referenced by active replacement atom -> PRESERVED
            conn.execute(
                """INSERT INTO evidence_materials(
                    material_id,scope_key,material_kind,role,conversation_id,snapshot_fingerprint,
                    source_position,source_family_id,text_hash,projection_key,card_version,
                    status,diagnostics_code,created_at,updated_at
                ) VALUES ('mat_superseded','sc1','event','user','c1','fp1',2,'','h3','pk1','v1','invalidated','',?,?)""",
                (old_ts, old_ts),
            )
            conn.execute(
                """INSERT INTO evidence_atoms(
                    atom_id,scope_key,material_id,kind,status,root_ids_json,
                    root_material_ids_json,atom_json,revision,created_at,updated_at
                ) VALUES ('atm_replacement','sc1','mat_superseded','fact','active','[]','[]','{}',2,?,?)""",
                (old_ts, old_ts),
            )

            # 4. Orphan atom whose material does not exist -> MUST BE PRUNED
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute(
                """INSERT INTO evidence_atoms(
                    atom_id,scope_key,material_id,kind,status,root_ids_json,
                    root_material_ids_json,atom_json,revision,created_at,updated_at
                ) VALUES ('atm_orphan','sc1','mat_missing','fact','active','[]','[]','{}',1,?,?)""",
                (now, now),
            )

            # 5. Orphan correction (target & replacement missing, older than retention) -> MUST BE PRUNED
            conn.execute(
                """INSERT INTO evidence_corrections(
                    correction_id,scope_key,correction_revision,target_id,action,
                    replacement_atom_id,affected_ids_json,reason,action_id,created_at
                ) VALUES ('cor_orphan','sc1',1,'atm_gone','withdraw',NULL,'[]','old',NULL,?)""",
                (old_ts,),
            )
            # 6. Active correction pointing to existing replacement atom -> PRESERVED
            conn.execute(
                """INSERT INTO evidence_corrections(
                    correction_id,scope_key,correction_revision,target_id,action,
                    replacement_atom_id,affected_ids_json,reason,action_id,created_at
                ) VALUES ('cor_live','sc1',2,'atm_old','supersede','atm_replacement','[]','keep',NULL,?)""",
                (old_ts,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO evidence_scope_revisions(scope_key,correction_revision) VALUES ('sc1',2)"
            )
            conn.commit()

        stats = await sidecar.prune_old(retention_days=30, now=now)
        assert stats["atoms_pruned"] == 2  # atm_expired + atm_orphan
        assert stats["materials_pruned"] == 1  # mat_expired
        assert stats["corrections_pruned"] == 1  # cor_orphan
        assert stats["wal_checkpointed"] == 1

        with sqlite3.connect(str(db_path)) as conn:
            atoms = {r[0] for r in conn.execute("SELECT atom_id FROM evidence_atoms")}
            mats = {
                r[0] for r in conn.execute("SELECT material_id FROM evidence_materials")
            }
            cors = {
                r[0]
                for r in conn.execute("SELECT correction_id FROM evidence_corrections")
            }
            rev = conn.execute(
                "SELECT correction_revision FROM evidence_scope_revisions WHERE scope_key='sc1'"
            ).fetchone()[0]

        assert atoms == {"atm_active", "atm_replacement"}
        assert mats == {"mat_active", "mat_superseded"}
        assert cors == {"cor_live"}
        assert rev == 2
        wal_path = Path(f"{db_path}-wal")
        if wal_path.exists():
            assert wal_path.stat().st_size == 0
    finally:
        await sidecar.close()


def _create_wal_sqlite(db_path: Path, table_sql: str, rows: list[tuple]) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute(table_sql)
    for row in rows:
        placeholders = ",".join("?" for _ in row)
        conn.execute(f"INSERT INTO items VALUES ({placeholders})", row)
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_maintain_storage_and_decay_scheduler_checkpoint_all_companion_dbs(
    tmp_path,
):
    """Daily maintenance checkpoints primary DB plus conversations, graph_documents, and sidecar."""
    primary = tmp_path / "livingmemory.db"
    conv_db = tmp_path / "conversations.db"
    graph_db = tmp_path / "livingmemory_graph_documents.db"
    sidecar_db = tmp_path / "evidence_sidecar.sqlite3"

    for path in (primary, conv_db, graph_db):
        _create_wal_sqlite(
            path,
            "CREATE TABLE items (id INTEGER PRIMARY KEY, payload TEXT)",
            [(i, f"data-{i}-" + "x" * 200) for i in range(1, 20)],
        )

    sidecar = EvidenceSidecar(sidecar_db)
    await sidecar.initialize()
    try:

        class _DummyEngine(MemoryEngineStatsMixin):
            def __init__(self, db_path_str: str, conn):
                self.db_path = db_path_str
                self.db_connection = conn
                self.config = {
                    "auto_cleanup_enabled": False,
                    "graph_document_path": str(graph_db),
                }

            async def apply_daily_decay(self, rate, days):
                return 0

        conn = await open_connection(str(primary), apply_journal_mode=True)
        try:
            engine = _DummyEngine(str(primary), conn)
            result = await engine.maintain_storage()
            assert result["success"] is True
            assert set(result["companion_dbs"].keys()) >= {
                "conversations.db",
                "livingmemory_graph_documents.db",
                "evidence_sidecar.sqlite3",
            }
            for name in (
                "conversations.db",
                "livingmemory_graph_documents.db",
                "evidence_sidecar.sqlite3",
            ):
                assert result["companion_dbs"][name]["success"] is True

            # Also verify DecayScheduler._execute_decay runs companion maintenance + sidecar prune
            scheduler = DecayScheduler(
                memory_engine=engine,
                decay_rate=0.0,
                data_dir=str(tmp_path),
                backup_enabled=False,
                evidence_sidecar=sidecar,
            )
            ok = await scheduler._execute_decay(1)
            assert ok is True

            # Verify no memory/conversation/graph rows were lost
            for path in (primary, conv_db, graph_db):
                with sqlite3.connect(str(path)) as check_conn:
                    count = check_conn.execute("SELECT COUNT(*) FROM items").fetchone()[
                        0
                    ]
                assert count == 19
        finally:
            await conn.close()
    finally:
        await sidecar.close()


def test_backup_manager_online_backup_tar_gz_wal_consistency_and_compression(tmp_path):
    """BackupManager uses SQLite online backup, includes sidecar, skips FAISS by default, writes tar.gz."""
    # Create WAL-mode SQLite DBs and keep a writer connection open so WAL stays uncheckpointed
    open_conns = []
    try:
        for db_name in (
            "livingmemory.db",
            "conversations.db",
            "livingmemory_graph_documents.db",
            "evidence_sidecar.sqlite3",
        ):
            db_path = tmp_path / db_name
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA wal_autocheckpoint=0")
            conn.execute("CREATE TABLE t (k TEXT PRIMARY KEY, v TEXT)")
            conn.execute(
                "INSERT INTO t VALUES ('wal_key', ?)",
                ("wal_consistent_payload_" + "A" * 2048,),
            )
            conn.commit()
            open_conns.append(conn)

        # Create huge rebuildable FAISS index files
        (tmp_path / "livingmemory.index").write_bytes(b"F" * 200_000)
        (tmp_path / "livingmemory_graph.index").write_bytes(b"G" * 200_000)
        (tmp_path / "decay_state.json").write_text('{"last_decay_date":"2026-11-23"}')

        mgr = BackupManager(str(tmp_path))
        mgr.version_file.write_text("2.9.0", encoding="utf-8")
        archive = mgr.backup_if_needed()
        assert archive is not None
        archive_path = Path(archive)
        assert archive_path.is_file()
        assert archive_path.name == "v2.9.0.tar.gz"
        # Compressed archive skipping FAISS is drastically smaller than raw data
        assert archive_path.stat().st_size < 50_000

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as tar:
            names = {Path(m.name).name for m in tar.getmembers() if m.isfile()}
            assert {
                "livingmemory.db",
                "conversations.db",
                "livingmemory_graph_documents.db",
                "evidence_sidecar.sqlite3",
                "decay_state.json",
                _BACKUP_INFO_FILE,
            } <= names
            assert "livingmemory.index" not in names
            assert "livingmemory_graph.index" not in names
            tar.extractall(extract_dir, filter="data")

        # Every extracted SQLite DB must contain the committed WAL row (online backup consistency)
        for db_name in (
            "livingmemory.db",
            "conversations.db",
            "livingmemory_graph_documents.db",
            "evidence_sidecar.sqlite3",
        ):
            with sqlite3.connect(str(extract_dir / db_name)) as restored:
                row = restored.execute("SELECT v FROM t WHERE k='wal_key'").fetchone()
            assert row is not None and row[0].startswith("wal_consistent_payload_")

        # list_backups reads BOTH new tar.gz archives and legacy directories
        legacy_dir = tmp_path / "backups" / "v1.0.0"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / _BACKUP_INFO_FILE).write_text(
            json.dumps({"previous_version": "1.0.0", "plugin_version": "2.0.0"}),
            encoding="utf-8",
        )
        listed = BackupManager.list_backups(str(tmp_path))
        by_name = {entry["name"]: entry for entry in listed}
        assert by_name["v2.9.0"]["format"] == "tar.gz"
        assert by_name["v1.0.0"]["format"] == "directory"

        # include_faiss=True explicitly includes .index files
        mgr.version_file.write_text("2.9.1", encoding="utf-8")
        with_faiss = mgr.backup_if_needed(include_faiss=True)
        assert with_faiss is not None
        with tarfile.open(with_faiss, "r:gz") as tar:
            names = {Path(m.name).name for m in tar.getmembers() if m.isfile()}
            assert "livingmemory.index" in names
            assert "livingmemory_graph.index" in names
    finally:
        for conn in open_conns:
            conn.close()


# ==============================================================================
# 3. [P1] MemorySearchTool concurrency/budget + Embedding telemetry + get_series_health
# ==============================================================================


@pytest.mark.asyncio
async def test_memory_search_tool_concurrent_scopes_shared_embedding_cache_and_remote_deadline():
    """Multiple read_scopes execute concurrently and share one QueryEmbeddingCache + remote_deadline."""
    seen_contexts = []

    async def slow_search(**kwargs):
        ctx = kwargs["execution_context"]
        seen_contexts.append(ctx)
        await asyncio.sleep(0.08)
        return []

    engine = Mock()
    engine.search_memories = AsyncMock(side_effect=slow_search)
    tool = MemorySearchTool(
        context=Mock(),
        config_manager=ConfigManager(),
        memory_engine=engine,
    )

    event = Mock()
    event.unified_msg_origin = "qq:FriendMessage:u1"
    event.get_extra = Mock(return_value=None)
    event._cd_tool_memory_sources_register_v1 = None
    run_context = Mock()
    run_context.context.event = event

    multi_policy = SimpleNamespace(
        read_scopes=(
            MemoryReadScope(
                session_ids=("qq:FriendMessage:u1",),
                interaction_types=("private_chat",),
                require_persona=True,
                read_only=False,
            ),
            MemoryReadScope(
                session_ids=None,
                interaction_types=("group_chat",),
                source_session_id="qq:GroupMessage:g1",
                require_persona=True,
                read_only=True,
            ),
            MemoryReadScope(
                session_ids=None,
                interaction_types=("group_chat",),
                source_session_id="qq:GroupMessage:g2",
                require_persona=True,
                read_only=True,
            ),
        ),
    )

    with (
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
            new=AsyncMock(return_value="persona-a"),
        ),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.resolve_verified_group_sessions",
            new=AsyncMock(return_value=("qq:GroupMessage:g1", "qq:GroupMessage:g2")),
        ),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.resolve_memory_access_policy",
            return_value=multi_policy,
        ),
    ):
        started = time.monotonic()
        raw = await tool.call(run_context, query="shared topic")
        elapsed = time.monotonic() - started

    data = json.loads(raw)
    assert data["status"] == "empty"
    assert len(seen_contexts) == 3
    # 3 scopes x 0.08s concurrently should complete well under serial 0.24s
    assert elapsed < 0.20
    # All scopes share the exact same QueryEmbeddingCache and remote_deadline
    caches = {id(ctx.embedding_cache) for ctx in seen_contexts}
    assert len(caches) == 1
    assert isinstance(seen_contexts[0].embedding_cache, QueryEmbeddingCache)
    assert seen_contexts[0].remote_deadline is not None
    assert all(
        ctx.remote_deadline == seen_contexts[0].remote_deadline for ctx in seen_contexts
    )


@pytest.mark.asyncio
async def test_memory_search_tool_include_source_runs_concurrently():
    """include_source=True batches/parallelizes source ID and row lookups."""
    memories = [
        SimpleNamespace(
            doc_id=i,
            content=f"summary {i}",
            final_score=1.0 - i * 0.1,
            metadata={
                "session_id": "qq:FriendMessage:u1",
                "persona_id": "persona-a",
                "importance": 0.8,
                "status": "active",
            },
        )
        for i in (1, 2, 3)
    ]
    engine = Mock()
    engine.search_memories = AsyncMock(return_value=memories)

    async def fake_source_ids(_eng, doc_id, _meta, _umo):
        await asyncio.sleep(0.04)
        return [doc_id * 10 + 1, doc_id * 10 + 2]

    async def fake_verified_row(_scope, doc_id, row_id, **_kwargs):
        await asyncio.sleep(0.05)
        return {
            "role": "user",
            "content": f"orig doc={doc_id} row={row_id}",
            "session_id": "qq:FriendMessage:u1",
            "sender_id": "u1",
            "sender_name": "Alice",
            "platform": "qq",
            "timestamp": 1700000000.0 + row_id,
        }

    fake_plugin = SimpleNamespace(
        initializer=SimpleNamespace(memory_engine=engine),
        peek_evidence_revision=lambda _s: {"available": True, "correction_revision": 1},
        _native_scope_from_request=lambda req_s, umo: {**req_s, "umo": umo},
        _native_memory_source_row_ids=fake_source_ids,
        _native_verified_source_row=fake_verified_row,
        _native_read_scopes=lambda *a, **k: [],
    )
    tool = MemorySearchTool(
        context=Mock(),
        config_manager=ConfigManager(),
        memory_engine=engine,
    )
    event = Mock()
    event.unified_msg_origin = "qq:FriendMessage:u1"
    event.get_extra = Mock(return_value=None)
    event._cd_tool_memory_sources_register_v1 = None
    run_context = Mock()
    run_context.context.event = event

    with (
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_active_plugin",
            return_value=fake_plugin,
        ),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
            new=AsyncMock(return_value="persona-a"),
        ),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.make_memory_source_validator",
            return_value=AsyncMock(return_value=True),
        ),
    ):
        started = time.monotonic()
        raw = await tool.call(run_context, query="orig", include_source=True)
        elapsed = time.monotonic() - started

    data = json.loads(raw)
    assert data["status"] == "ok"
    assert len(data["results"]) == 3
    for item in data["results"]:
        assert len(item["source_messages"]) == 2
    # Serial would be 3*0.04 + 6*0.05 = 0.42s; concurrent semaphore(4) finishes much faster
    assert elapsed < 0.30


def test_memory_search_tool_json_result_char_budget_prioritizes_high_score():
    """_json_result enforces 6000/12000 budgets, keeps high-score items, and truncates long text."""
    assert DEFAULT_RESULT_CHAR_BUDGET == 6000
    assert SOURCE_RESULT_CHAR_BUDGET == 12000

    payload = {
        "status": "ok",
        "query": "预算测试",
        "count": 10,
        "results": [
            {
                "id": idx,
                "score": 1.0 - idx * 0.05,
                "content": f"高分记忆#{idx} " + ("长文本内容" * 300),
                "source_messages": [
                    {"role": "user", "content": f"原文#{idx} " + ("详细对话" * 150)}
                    for _ in range(3)
                ],
            }
            for idx in range(10)
        ],
        "diagnostics": {"elapsed_ms": 12},
    }

    normal_out = _json_result(payload, max_chars=DEFAULT_RESULT_CHAR_BUDGET)
    assert len(normal_out) <= DEFAULT_RESULT_CHAR_BUDGET
    parsed = json.loads(normal_out)
    assert parsed["diagnostics"]["truncated"] is True
    assert parsed["results"][0]["id"] == 0  # Highest score memory preserved first
    assert parsed["count"] == len(parsed["results"])

    source_out = _json_result(payload, max_chars=SOURCE_RESULT_CHAR_BUDGET)
    assert len(source_out) <= SOURCE_RESULT_CHAR_BUDGET
    parsed_source = json.loads(source_out)
    assert parsed_source["results"][0]["id"] == 0
    # Larger budget preserves more content than smaller budget
    assert len(source_out) >= len(normal_out)


@pytest.mark.asyncio
async def test_embedding_telemetry_reports_duration_and_outcomes():
    """vector_search and vector_retriever report embedding usage via record_series_llm_usage."""
    recorded: list[dict] = []

    class _Telemetry:
        def record(self, **kwargs):
            recorded.append(kwargs)

    class _FakePlugin:
        def __init__(self, ctx):
            self.context = ctx

    fake_ctx = SimpleNamespace(_chat_dynamics_series_telemetry=_Telemetry())
    fake_plugin = _FakePlugin(fake_ctx)
    prev_plugin = get_active_plugin()
    set_active_plugin(fake_plugin)
    try:
        emb_provider = SimpleNamespace(
            provider_config={
                "id": "openai-emb",
                "embedding_model": "text-embedding-3-small",
            },
            get_embedding=AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4]),
        )

        # 1. retrieve_vectors direct & via QueryEmbeddingCache
        fake_index = SimpleNamespace(ntotal=1)

        async def fake_search(_vec, _k):
            return np.asarray([[0.1]], dtype=np.float32), np.asarray(
                [[1]], dtype=np.int64
            )

        fake_db = SimpleNamespace(
            embedding_provider=emb_provider,
            embedding_storage=SimpleNamespace(
                index=fake_index, dimension=4, search=fake_search
            ),
            document_storage=SimpleNamespace(
                get_documents=AsyncMock(
                    return_value=[
                        {"id": 1, "text": "hello", "metadata": {"status": "active"}}
                    ]
                )
            ),
        )

        hits = await retrieve_vectors(fake_db, "test query", 3, {})
        assert len(hits) == 1

        cache = QueryEmbeddingCache(asyncio.get_running_loop().time() + 5.0)
        token = query_embeddings.set(cache)
        try:
            await retrieve_vectors(fake_db, "cached query", 3, {})
            # Second call with same query hits cache -> no extra telemetry record
            await retrieve_vectors(fake_db, "cached query", 3, {})
        finally:
            query_embeddings.reset(token)

        # 2. VectorRetriever.add_document success & failure
        insert_db = SimpleNamespace(
            embedding_provider=emb_provider,
            insert=AsyncMock(return_value=42),
        )
        retriever = VectorRetriever(insert_db)
        assert await retriever.add_document("short memory", {"importance": 0.9}) == 42

        failing_provider = SimpleNamespace(
            provider_config={"id": "broken-emb", "model": "bad-model"},
            get_embedding=AsyncMock(side_effect=RuntimeError("embed failed")),
        )
        failing_db = SimpleNamespace(
            embedding_provider=failing_provider,
            embedding_storage=SimpleNamespace(index=fake_index, dimension=4),
            document_storage=SimpleNamespace(get_documents=AsyncMock(return_value=[])),
        )
        with pytest.raises(RuntimeError, match="embed failed"):
            await retrieve_vectors(failing_db, "boom", 3, {})

        assert len(recorded) == 4
        for entry in recorded:
            assert entry["plugin_id"] == "astrbot_plugin_dynamics_memory"
            assert entry["purpose"] == "embedding"
            assert entry["generation_seconds"] >= 0.0
        assert recorded[0]["provider_id"] == "openai-emb"
        assert recorded[0]["model"] == "text-embedding-3-small"
        assert recorded[0]["ok"] is True
        assert recorded[3]["provider_id"] == "broken-emb"
        assert recorded[3]["ok"] is False
        assert recorded[3]["error_type"] == "RuntimeError"
    finally:
        set_active_plugin(prev_plugin)


def test_get_series_health_sync_readonly_and_panel_compatible():
    """get_series_health() is synchronous, read-only, and compatible with series_panels.read_plugin_health."""
    assert not inspect.iscoroutinefunction(LivingMemoryPlugin.get_series_health)

    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = SimpleNamespace(
        get_all_stars=lambda: {},
        provider_manager=SimpleNamespace(inst_map={}),
        get_all_providers=lambda: [],
        get_all_embedding_providers=lambda: [],
    )
    plugin._series_enabled = True
    plugin._terminating = False
    plugin._llm_tools_registered = True
    plugin.config_manager = ConfigManager(
        {
            "provider_settings": {
                "embedding_provider_id": "emb-main",
                "llm_provider_id": "llm-main",
            }
        }
    )

    # Healthy ready state
    plugin.initializer = SimpleNamespace(
        is_initialized=True,
        is_failed=False,
        error_message=None,
        _provider_check_attempts=1,
        in_slow_probe_mode=False,
        embedding_provider=SimpleNamespace(provider_config={"id": "emb-main"}),
        llm_provider=SimpleNamespace(provider_config={"id": "llm-main"}),
        index_maintenance_status={"state": "ready"},
    )
    plugin.event_handler = object()
    plugin.evidence_service = object()

    health = series_panels.read_plugin_health("astrbot_plugin_dynamics_memory", plugin)
    assert health is not None
    assert health["ready"] is True
    assert health["degraded"] is False
    assert health["issues"] == []
    roles = {p["role"]: p for p in health["providers"]}
    assert roles["embedding"]["status"] == "ready"
    assert roles["embedding"]["provider_id"] == "emb-main"
    assert roles["llm"]["status"] == "ready"
    assert roles["llm"]["provider_id"] == "llm-main"

    # Degraded cold-start slow-probe state with missing providers
    plugin.initializer = SimpleNamespace(
        is_initialized=False,
        is_failed=False,
        error_message="Provider 初始化超时（已转入低频探测）",
        _provider_check_attempts=61,
        in_slow_probe_mode=True,
        embedding_provider=None,
        llm_provider=None,
        index_maintenance_status={"state": "idle"},
    )
    plugin.event_handler = None
    plugin.evidence_service = None

    degraded = series_panels.read_plugin_health(
        "astrbot_plugin_dynamics_memory", plugin
    )
    assert degraded is not None
    assert degraded["ready"] is False
    assert degraded["degraded"] is True
    assert "embedding_provider_missing" in degraded["issues"]
    assert "llm_provider_missing" in degraded["issues"]
    assert "initializing_slow_probe" in degraded["issues"]
    assert degraded["metrics"]["slow_probe_mode"] is True
