"""Focused regression tests for R2/R3 optimizations in astrbot_plugin_dynamics_memory."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest
from astrbot.core.message.components import At, Plain
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall import (
    should_skip_group_recall_query,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.message_utils import (
    MessageUtils,
)
from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    history_material_id,
    normalize_host_material,
)
from astrbot_plugin_dynamics_memory.core.managers.backup_manager import BackupManager
from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
    ConversationManager,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
from astrbot_plugin_dynamics_memory.core.models.graph_models import (
    GraphEntry,
    GraphNode,
)
from astrbot_plugin_dynamics_memory.core.processors.memory_processor import (
    MemoryProcessor,
)
from astrbot_plugin_dynamics_memory.core.processors.memory_processor_parse import (
    check_summary_quality,
)
from astrbot_plugin_dynamics_memory.core.retrieval.dual_route_retriever import (
    DualRouteRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.graph_keyword_retriever import (
    GraphKeywordRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import HybridResult
from astrbot_plugin_dynamics_memory.core.schedulers.decay_scheduler import (
    DecayScheduler,
)
from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from astrbot_plugin_dynamics_memory.storage.db_migration import DBMigration
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar
from astrbot_plugin_dynamics_memory.storage.graph_store import GraphStore
from astrbot_plugin_dynamics_memory.storage.sqlite_connection import open_connection


# ==============================================================================
# A.1 Storage & backup governance (P0)
# ==============================================================================


@pytest.mark.asyncio
async def test_auto_vacuum_incremental_enabled_on_fresh_dbs(tmp_path: Path):
    db_path = tmp_path / "fresh.db"
    conn = await open_connection(str(db_path), apply_journal_mode=True)
    try:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        await conn.commit()
        cur = await conn.execute("PRAGMA auto_vacuum")
        row = await cur.fetchone()
        await cur.close()
        assert int(row[0]) == 2  # INCREMENTAL
    finally:
        await conn.close()

    sidecar_path = tmp_path / "evidence_sidecar.sqlite3"
    sidecar = EvidenceSidecar(sidecar_path)
    await sidecar.initialize()
    try:
        with sqlite3.connect(str(sidecar_path)) as sconn:
            av = sconn.execute("PRAGMA auto_vacuum").fetchone()[0]
        assert int(av) == 2
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_db_migration_compressed_backup_and_scheduler_prune(tmp_path: Path):
    db_path = tmp_path / "livingmemory.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO docs VALUES (1, ?)", ("hello " * 200,))
        conn.commit()

    migration = DBMigration(str(db_path))
    backup_str = await migration.create_backup()
    assert backup_str is not None
    backup_file = Path(backup_str)
    assert backup_file.is_file()
    assert backup_file.name.endswith(".db.gz")

    # Verify valid gzip containing a SQLite database
    raw = gzip.decompress(backup_file.read_bytes())
    assert raw.startswith(b"SQLite format 3\x00")
    # Verify no .tmp_* staging files remain
    backups_dir = tmp_path / "backups"
    assert not any(p.name.startswith(".tmp_") for p in backups_dir.iterdir())

    # Plant stale .bak / .tmp_* files and an older giant uncompressed .db backup
    (backups_dir / "stale.bak").write_bytes(b"b" * 100)
    (backups_dir / ".tmp_aborted.db").write_bytes(b"t" * 100)
    old_raw = backups_dir / "livingmemory_backup_20200101_000000.db"
    old_raw.write_bytes(b"x" * (2 * 1024 * 1024))
    old_ts = time.time() - 3600
    os.utime(old_raw, (old_ts, old_ts))

    # DecayScheduler._run_backup() invokes BackupManager.prune() and removes debris
    scheduler = DecayScheduler(
        memory_engine=Mock(),
        decay_rate=0.0,
        data_dir=str(tmp_path),
        db_migration=migration,
        backup_enabled=True,
        backup_keep_days=7,
    )
    await scheduler._run_backup()
    assert not (backups_dir / "stale.bak").exists()
    assert not (backups_dir / ".tmp_aborted.db").exists()

    # Cap enforcement prefers protecting compressed .db.gz over legacy uncompressed .db
    stats = BackupManager.prune(str(tmp_path), keep_days=7, max_total_mb=1)
    assert not old_raw.exists()
    assert any(p.name.endswith(".db.gz") for p in backups_dir.iterdir())
    assert stats["bytes_removed"] >= 2 * 1024 * 1024


def test_history_material_id_deduplicates_across_snapshots_and_positions():
    id1 = history_material_id("sc:1", "conv:1", "snap-turn-1", 0, "明天一起吃饭")
    id2 = history_material_id("sc:1", "conv:1", "snap-turn-2", 5, "明天一起吃饭")
    id3 = history_material_id("sc:1", "conv:1", "snap-turn-2", 5, "换一条消息")
    assert id1 == id2
    assert id1 != id3

    d1, _ = normalize_host_material(
        "sc:1",
        {"role": "user", "text": "明天一起吃饭", "position": 0},
        conversation_id="conv:1",
        snapshot_fingerprint="snap-1",
    )
    d2, _ = normalize_host_material(
        "sc:1",
        {"role": "user", "text": "明天一起吃饭", "position": 3},
        conversation_id="conv:1",
        snapshot_fingerprint="snap-2",
    )
    assert d1["material_id"] == d2["material_id"]
    assert d1["source_locator"]["snapshot_fingerprint"] == "snap-1"
    assert d2["source_locator"]["snapshot_fingerprint"] == "snap-2"


@pytest.mark.asyncio
async def test_evidence_sidecar_prune_old_cleans_stale_queue_and_per_scope_cap(
    tmp_path: Path,
):
    db_path = tmp_path / "evidence_sidecar.sqlite3"
    sidecar = EvidenceSidecar(db_path)
    await sidecar.initialize()
    try:
        now = time.time()
        old_ts = now - 25 * 86400
        with sqlite3.connect(str(db_path)) as conn:
            # Aged ready/deferred/owner_wait/queued materials without atoms -> pruned
            for idx, st in enumerate(("ready", "queued", "deferred", "owner_wait")):
                mid = f"mat_old_{idx}"
                conn.execute(
                    """INSERT INTO evidence_materials(
                        material_id,scope_key,material_kind,role,conversation_id,snapshot_fingerprint,
                        source_position,source_family_id,text_hash,projection_key,card_version,
                        status,diagnostics_code,created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        mid,
                        "sc1",
                        "host_record_only",
                        "user",
                        "c1",
                        "fp",
                        idx,
                        "",
                        f"h{idx}",
                        "pk",
                        "v1",
                        st,
                        "",
                        old_ts,
                        old_ts,
                    ),
                )
                conn.execute(
                    """INSERT INTO evidence_derivation_queue(
                        material_id,scope_key,umo,source_locator_json,correction_revision,
                        projection_key,text_hash,status,available_at,lease_id,lease_until,created_at,updated_at
                    ) VALUES (?,?,?,?,0,'pk',?,?,0,'',0,?,?)""",
                    (
                        mid,
                        "sc1",
                        "u1",
                        "{}",
                        f"h{idx}",
                        "deferred" if st == "ready" else st,
                        old_ts,
                        old_ts,
                    ),
                )
            # 10 recent unreferenced materials in sc2 to test per-scope cap=4
            for idx in range(10):
                mid = f"mat_cap_{idx}"
                ts = now - (10 - idx)
                conn.execute(
                    """INSERT INTO evidence_materials(
                        material_id,scope_key,material_kind,role,conversation_id,snapshot_fingerprint,
                        source_position,source_family_id,text_hash,projection_key,card_version,
                        status,diagnostics_code,created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        mid,
                        "sc2",
                        "host_record_only",
                        "user",
                        "c2",
                        "fp",
                        idx,
                        "",
                        f"hc{idx}",
                        "pk",
                        "v1",
                        "deferred",
                        "",
                        ts,
                        ts,
                    ),
                )
            conn.commit()

        stats = await sidecar.prune_old(
            retention_days=20, now=now, queue_retention_days=14, max_rows_per_scope=4
        )
        assert stats["queue_pruned"] == 4
        assert stats["materials_pruned"] == 4 + 6  # 4 aged in sc1 + 6 capped in sc2
        with sqlite3.connect(str(db_path)) as conn:
            sc1_cnt = conn.execute(
                "SELECT COUNT(*) FROM evidence_materials WHERE scope_key='sc1'"
            ).fetchone()[0]
            sc2_cnt = conn.execute(
                "SELECT COUNT(*) FROM evidence_materials WHERE scope_key='sc2'"
            ).fetchone()[0]
        assert sc1_cnt == 0
        assert sc2_cnt == 4
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_enforce_message_limit_hard_ceiling_and_narrative_prune(tmp_path: Path):
    db_path = tmp_path / "conversations.db"
    store = ConversationStore(str(db_path))
    await store.initialize()
    manager = ConversationManager(
        store=store, max_cache_size=10, context_window_size=20
    )
    sid = "test:GroupMessage:stalled"
    try:
        # Verify redundant idx_session_id and unused idx_msg_timestamp are dropped,
        # while both distinct active indexes idx_msg_session and idx_msg_session_id exist.
        async with store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ) as cur:
            indexes = {str(r[0]) for r in await cur.fetchall()}
        assert "idx_msg_session" in indexes
        assert "idx_msg_session_id" in indexes
        assert "idx_session_id" not in indexes
        assert "idx_msg_timestamp" not in indexes

        now = time.time()
        for i in range(200):
            await manager.add_message(
                session_id=sid,
                role="user",
                content=f"msg-{i}",
                sender_id="u1",
                timestamp=now + i,
            )
        await manager.update_session_metadata(sid, "last_summarized_index", 0)
        utils = MessageUtils(
            ConfigManager(
                {
                    "session_manager": {
                        "max_messages_per_session": 100,
                        "hard_max_messages_per_session": 150,
                        "cleanup_batch_size": 25,
                    }
                }
            ),
            manager,
        )
        await utils.enforce_message_limit(sid)
        remaining = await store.get_message_count(sid)
        assert remaining == 150  # Trimmed 50 unsummarized rows down to hard ceiling

        # Seed narrative_events (old + excess recent) and prune via sync helper
        async with store.connection.execute("SELECT 1"):
            pass
        for i in range(8):
            ts = now - 40 * 86400 if i < 3 else now - (10 - i)
            await store.connection.execute(
                "INSERT INTO narrative_events VALUES (?, ?, ?, ?, ?)",
                (f"ev_{i}", sid, f"id_{i}", "{}", ts),
            )
        await store.connection.commit()

        pruned = ConversationStore.prune_narrative_events_file_sync(
            db_path, retention_days=30, max_per_session=3, now=now
        )
        assert pruned == 5  # 3 expired + 2 over cap=3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_purge_expired_archived_memories_and_strip_write_ops_payloads(
    tmp_path: Path,
):
    db_path = tmp_path / "livingmemory.db"
    engine = MemoryEngine.__new__(MemoryEngine)
    engine.db_path = str(db_path)
    engine.config = {"archived_retention_days": 30}
    engine.db_connection = await open_connection(str(db_path), apply_journal_mode=True)
    engine.db_connection.row_factory = sqlite3.Row
    engine._db_write_lock = asyncio.Lock()
    try:
        await engine.db_connection.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, doc_id TEXT, text TEXT, metadata TEXT)"
        )
        await engine.db_connection.execute(
            """CREATE TABLE memory_write_ops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                op_type TEXT NOT NULL, memory_id INTEGER,
                status TEXT NOT NULL DEFAULT 'pending', step TEXT NOT NULL DEFAULT 'started',
                payload TEXT DEFAULT '{}', error TEXT, retry_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            )"""
        )
        now = time.time()
        old_arch = now - 45 * 86400
        recent_arch = now - 5 * 86400
        await engine.db_connection.execute(
            "INSERT INTO documents VALUES (1, 'd1', 'old archived', ?)",
            (json.dumps({"status": "archived", "archived_at": old_arch}),),
        )
        await engine.db_connection.execute(
            "INSERT INTO documents VALUES (2, 'd2', 'recent archived', ?)",
            (json.dumps({"status": "archived", "archived_at": recent_arch}),),
        )
        await engine.db_connection.execute(
            "INSERT INTO documents VALUES (3, 'd3', 'active', ?)",
            (json.dumps({"status": "active", "create_time": old_arch}),),
        )
        # Write ops: 1 completed 5d ago with bulky payload, 1 completed 20d ago, 1 pending 40d ago
        big_payload = json.dumps({"blob": "x" * 1024})
        await engine.db_connection.execute(
            "INSERT INTO memory_write_ops(op_type,status,payload,created_at,updated_at) VALUES ('add','completed',?,?,?)",
            (big_payload, now - 5 * 86400, now - 5 * 86400),
        )
        await engine.db_connection.execute(
            "INSERT INTO memory_write_ops(op_type,status,payload,created_at,updated_at) VALUES ('add','completed',?,?,?)",
            (big_payload, now - 20 * 86400, now - 20 * 86400),
        )
        await engine.db_connection.execute(
            "INSERT INTO memory_write_ops(op_type,status,payload,created_at,updated_at) VALUES ('add','pending',?,?,?)",
            (big_payload, now - 40 * 86400, now - 40 * 86400),
        )
        await engine.db_connection.commit()

        deleted_ids: list[int] = []

        async def _fake_batch_delete(ids):
            deleted_ids.extend(ids)
            return len(ids)

        engine.batch_delete_memories = _fake_batch_delete
        purged = await engine.purge_expired_archived_memories()
        assert purged == 1
        assert deleted_ids == [1]

        ops_deleted = await engine.cleanup_write_ops(
            retention_days=14, payload_retention_days=3
        )
        assert ops_deleted == 2  # 20d completed + 40d stale pending
        cur = await engine.db_connection.execute("SELECT payload FROM memory_write_ops")
        remaining_ops = await cur.fetchall()
        assert len(remaining_ops) == 1
        assert remaining_ops[0]["payload"] == "{}"  # Bulky payload stripped!
    finally:
        await engine.db_connection.close()


# ==============================================================================
# A.2 Reflection quality gate & retry backoff (P1)
# ==============================================================================


@pytest.mark.asyncio
async def test_summary_quality_idiom_exemptions_and_rich_summary_downgrade():
    # Idioms/compounds must NOT be flagged
    idiom_data = {
        "summary": "张三问群里有没有人玩过新游戏，李四说还有人没下班，总有人爱熬夜，告诉所有人明天照常开会，没有人反对，如果有人遇到问题可以随时联系。",
        "key_facts": ["张三询问新游戏", "李四提到加班"],
        "participants": ["张三", "李四"],
        "importance": 0.6,
    }
    chk = check_summary_quality(idiom_data)
    assert chk.reason_code is None
    assert chk.matched_generic_terms == ()

    # Rich summary (>=100 chars, >=3 facts, non-empty participants) with single anonymous hit -> degraded warning, NOT hard rejection
    rich_summary = (
        "张三、李四和王五在群里详细讨论了周末聚餐安排，确定周六晚上七点在市中心火锅店集合，"
        "王五负责提前预约包间并确认菜单与饮料清单，李四负责统计大家忌口和到场时间，"
        "中途有人提议顺便带桌游活跃气氛，最后三人一致同意按此方案执行并提醒准时出发。"
    )
    assert len(rich_summary) >= 100
    rich_data = {
        "summary": rich_summary,
        "key_facts": [
            "周六晚七点火锅店聚餐",
            "王五负责预约包间",
            "李四负责统计忌口",
        ],
        "participants": ["张三", "李四", "王五"],
        "importance": 0.7,
    }
    chk_rich = check_summary_quality(rich_data)
    assert chk_rich.reason_code is None
    assert chk_rich.degraded_warning == "summary_generic_reference_degraded"

    # MemoryProcessor stores it with summary_quality="degraded" instead of raising
    class _StubLLM:
        async def text_chat(self, **_kw):
            return SimpleNamespace(
                completion_text=json.dumps(rich_data, ensure_ascii=False)
            )

    processor = MemoryProcessor(llm_provider=_StubLLM())
    msgs = [
        Message(
            id=1,
            session_id="qq:GroupMessage:1",
            role="user",
            content="周六聚餐",
            sender_id="1",
            sender_name="张三",
            group_id="1",
            timestamp=time.time(),
        )
    ]
    _content, meta, _imp = await processor.process_conversation(
        msgs, is_group_chat=True
    )
    assert meta["summary_quality"] == "degraded"
    assert meta["summary_quality_warning"] == "summary_generic_reference_degraded"

    # Sparse summary with anonymous hit still hard-rejects on first pass,
    # and accepts degraded draft when allow_degraded_quality=True
    sparse_data = {
        "summary": "先是有人提议改期，张三和李四随后确认时间。",
        "key_facts": ["讨论改期"],
        "participants": ["张三", "李四"],
        "importance": 0.6,
    }
    assert check_summary_quality(sparse_data).reason_code == "summary_generic_reference"

    class _SparseLLM:
        async def text_chat(self, **_kw):
            return SimpleNamespace(
                completion_text=json.dumps(sparse_data, ensure_ascii=False)
            )

    sparse_proc = MemoryProcessor(llm_provider=_SparseLLM())
    _c2, meta2, _i2 = await sparse_proc.process_conversation(
        msgs, is_group_chat=True, allow_degraded_quality=True
    )
    assert meta2["summary_quality"] == "degraded"


@pytest.mark.asyncio
async def test_reflection_exponential_retry_backoff_and_cursor_deadlock_escape(
    tmp_path: Path,
):
    from tests.test_reflection_backlog import _make_real_handler, _seed_messages

    handler, manager, store = await _make_real_handler(tmp_path)
    sid = "test:GroupMessage:backoff"
    now = time.time()
    await _seed_messages(manager, sid, [now - 100 + i for i in range(4)])
    reflection = handler._memory_reflection
    msgs = await store.get_session_messages_asc(sid)
    try:
        # Retry 1 -> ~60s backoff
        t0 = time.time()
        await reflection._record_pending_summary(
            sid,
            0,
            4,
            0,
            None,
            failure_stage="model",
            error_code="err",
            window_messages=msgs,
        )
        p1 = await manager.get_session_metadata(sid, "pending_summary")
        assert 50 <= p1["retry_after"] - t0 <= 70
        # Backoff enforced for failure_stage="model"
        assert (
            await reflection._build_reflection_plan_for_cursor(
                sid, 4, 0, p1, None, None, sid
            )
            is None
        )

        # Retry 2 -> ~300s backoff
        t1 = time.time()
        await reflection._record_pending_summary(
            sid,
            0,
            4,
            1,
            None,
            failure_stage="model",
            error_code="err",
            window_messages=msgs,
        )
        p2 = await manager.get_session_metadata(sid, "pending_summary")
        assert 290 <= p2["retry_after"] - t1 <= 310

        # Retry 3 -> >=900s cooldown, exhausted_cycles=1
        t2 = time.time()
        await reflection._record_pending_summary(
            sid,
            0,
            4,
            2,
            None,
            failure_stage="model",
            error_code="err",
            window_messages=msgs,
        )
        p3 = await manager.get_session_metadata(sid, "pending_summary")
        assert p3["failure_stage"] == "retry_exhausted"
        assert p3["exhausted_cycles"] == 1
        assert p3["retry_after"] - t2 >= 890

        # Second exhaustion cycle -> advances cursor so last_summarized_index never deadlocks
        await reflection._record_pending_summary(
            sid,
            0,
            4,
            2,
            None,
            failure_stage="model",
            error_code="err",
            window_messages=msgs,
        )
        assert await manager.get_session_metadata(sid, "last_summarized_index", 0) == 4
        assert await manager.get_session_metadata(sid, "pending_summary", None) is None
    finally:
        await handler.shutdown()
        await store.close()


# ==============================================================================
# A.3 Recall latency, noise filtering & graph optimization (P1)
# ==============================================================================


def test_should_skip_group_recall_query_filters_noise_and_keeps_valid():
    def _group_ev(comps, self_id="10000"):
        return SimpleNamespace(
            get_message_type=lambda: MessageType.GROUP_MESSAGE,
            get_self_id=lambda: self_id,
            message_obj=SimpleNamespace(message=comps),
            get_messages=lambda: comps,
        )

    def _private_ev(comps, self_id="10000"):
        return SimpleNamespace(
            get_message_type=lambda: MessageType.FRIEND_MESSAGE,
            get_self_id=lambda: self_id,
            message_obj=SimpleNamespace(message=comps),
            get_messages=lambda: comps,
        )

    # At only other bot/user -> skip
    assert (
        should_skip_group_recall_query(
            _group_ev([At(qq="3889001246"), Plain(text="今日运势")]),
            "今日运势",
        )
        == "at_other"
    )
    # At self_id -> do NOT skip
    assert (
        should_skip_group_recall_query(
            _group_ev([At(qq="10000"), Plain(text="帮我查一下")]),
            "帮我查一下",
        )
        is None
    )
    # At all -> do NOT skip
    assert (
        should_skip_group_recall_query(
            _group_ev([At(qq="all"), Plain(text="大家明天开会")]),
            "大家明天开会",
        )
        is None
    )
    # Command prefixes -> skip
    for cmd in ("/今日运势", "!ping", ".help", "！菜单"):
        assert (
            should_skip_group_recall_query(_group_ev([Plain(text=cmd)]), cmd)
            == "command_prefix"
        )
    # Trivially short (<2 meaningful chars) -> skip
    for short in ("嗯", "?", "。。。", "1"):
        assert (
            should_skip_group_recall_query(_group_ev([Plain(text=short)]), short)
            == "trivial_short"
        )
    # Private messages with command/short are NOT filtered by group noise filter
    assert (
        should_skip_group_recall_query(_private_ev([Plain(text="/help")]), "/help")
        is None
    )
    assert should_skip_group_recall_query(_private_ev([Plain(text="嗯")]), "嗯") is None


@pytest.mark.asyncio
async def test_main_handle_memory_recall_skips_projection_on_excluded_or_noise():
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.context = SimpleNamespace(get_all_stars=lambda: {})
    plugin._series_enabled = True
    plugin._terminating = False
    plugin.config_manager = ConfigManager(
        {"access_control": {"whitelist_enabled": True, "allowed_ids": "99999"}}
    )
    plugin.initializer = SimpleNamespace(is_initialized=True)
    plugin._install_private_injection_getter = Mock()
    plugin._prepare_native_memory_projection = AsyncMock(
        side_effect=AssertionError("projection must not run for excluded/noise events")
    )
    plugin._remember_blocking_owner = AsyncMock()
    plugin.event_handler = SimpleNamespace(
        message_utils=SimpleNamespace(
            resolve_original_input=AsyncMock(
                return_value=SimpleNamespace(status="not_owned", query_text="/今日运势")
            )
        ),
        handle_memory_recall=AsyncMock(),
    )

    # 1. Policy denied (not in whitelist) -> skips projection
    ev_denied = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:11111",
        get_group_id=lambda: "11111",
        get_sender_id=lambda: "222",
        get_self_id=lambda: "10000",
        get_message_type=lambda: MessageType.GROUP_MESSAGE,
        get_extra=lambda *_a, **_kw: None,
        message_obj=SimpleNamespace(message=[Plain(text="你好呀")]),
    )
    req1 = ProviderRequest(prompt="你好呀")
    await plugin.handle_memory_recall(ev_denied, req1)
    assert getattr(req1, "_lm_private_hook_outcome_v1", None) == "policy_denied"
    plugin._prepare_native_memory_projection.assert_not_awaited()

    # 2. Allowed group, but command prefix -> skips projection
    ev_cmd = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:99999",
        get_group_id=lambda: "99999",
        get_sender_id=lambda: "222",
        get_self_id=lambda: "10000",
        get_message_type=lambda: MessageType.GROUP_MESSAGE,
        get_extra=lambda *_a, **_kw: None,
        message_obj=SimpleNamespace(message=[Plain(text="/今日运势")]),
    )
    req2 = ProviderRequest(prompt="/今日运势")
    await plugin.handle_memory_recall(ev_cmd, req2)
    assert (
        getattr(req2, "_lm_private_hook_outcome_v1", None) == "skipped_command_prefix"
    )
    plugin._prepare_native_memory_projection.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_store_indexed_prefix_concurrent_reads_and_sub_budget(
    tmp_path: Path,
):
    db_path = tmp_path / "graph_opt.db"
    store = GraphStore(str(db_path), cache_reads=True)
    await store.initialize()
    try:
        node_ids = await store.upsert_nodes(
            [
                GraphNode("topic", "杭州西湖", "杭州西湖"),
                GraphNode("topic", "上海外滩", "上海外滩"),
            ]
        )
        await store.add_entry(
            GraphEntry(
                entry_key="e1",
                source_memory_id=10,
                session_id="s1",
                persona_id="p1",
                entry_type="topic",
                content="杭州西湖游玩记录",
                node_keys=["topic:杭州西湖"],
            ),
            node_ids,
        )

        # 1. Verify 2-char token uses B-tree index range instead of full-table LIKE scan
        lo, hi = store._prefix_range_bounds("杭州")
        async with store._read_connect() as db:
            cur = await db.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM graph_nodes WHERE canonical_value >= ? AND canonical_value < ?",
                (lo, hi),
            )
            plan = " ".join(str(r) for r in await cur.fetchall())
        assert "idx_graph_nodes_canonical" in plan

        matched = await store.search_nodes_by_tokens(["杭州"], limit=5)
        assert [m["canonical_value"] for m in matched] == ["杭州西湖"]

        # 2. Concurrent WAL reads execute in parallel across pool connections (not serialized behind 1 lock)
        in_flight = 0
        max_in_flight = 0

        async def _slow_read():
            nonlocal in_flight, max_in_flight
            async with store._read_connect() as db:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                try:
                    await asyncio.sleep(0.05)
                    cur = await db.execute("SELECT COUNT(*) FROM graph_nodes")
                    return (await cur.fetchone())[0]
                finally:
                    in_flight -= 1

        counts = await asyncio.gather(_slow_read(), _slow_read(), _slow_read())
        assert counts == [2, 2, 2]
        assert max_in_flight >= 2

        # 3. GraphKeywordRetriever caps token count cleanly
        selected = GraphKeywordRetriever._select_node_tokens(
            [f"词{i:02d}" for i in range(30)], max_tokens=12
        )
        assert len(selected) == 12
    finally:
        await store.close()

    # 4. DualRouteRetriever enforces tight graph sub-budget and degrades gracefully
    doc_result = HybridResult(
        doc_id=42,
        final_score=0.9,
        rrf_score=0.9,
        bm25_score=0.8,
        vector_score=0.9,
        content="主文档命中",
        metadata={"importance": 0.8},
    )

    class _FastDocRetriever:
        async def search(self, *_a, **_kw):
            return [doc_result]

    class _SlowGraphRetriever:
        async def search(self, *_a, **_kw):
            await asyncio.sleep(1.5)
            return []

    async def _loader(doc_id: int):
        return {"id": doc_id, "text": "主文档命中", "metadata": "{}"}

    dual = DualRouteRetriever(
        document_retriever=_FastDocRetriever(),
        graph_retriever=_SlowGraphRetriever(),
        memory_loader=_loader,
        config={"graph_route_timeout_seconds": 0.15, "retrieval_timeout_seconds": 5.0},
    )
    started = time.monotonic()
    res = await dual.search("测试查询", k=3)
    elapsed = time.monotonic() - started
    assert [r.doc_id for r in res] == [42]
    assert elapsed < 0.8


@pytest.mark.asyncio
async def test_graph_read_pool_discards_connection_on_cancellation(tmp_path: Path):
    """Cancelling an in-flight read (e.g. graph route timeout) must discard, not reuse, the connection."""
    db_path = tmp_path / "graph_cancel.db"
    store = GraphStore(str(db_path), cache_reads=True)
    await store.initialize()
    try:
        await store.upsert_nodes([GraphNode("topic", "杭州西湖", "杭州西湖")])

        entered = asyncio.Event()
        cancelled_conn_holder: list[object] = []

        async def _cancellable_read():
            async with store._read_connect() as db:
                cancelled_conn_holder.append(db)
                entered.set()
                await asyncio.sleep(5.0)

        task = asyncio.create_task(_cancellable_read())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(cancelled_conn_holder) == 1
        cancelled_conn = cancelled_conn_holder[0]
        # Cancelled connection must NOT be returned to the pool
        assert cancelled_conn not in store._read_pool
        assert len(store._read_pool) == 0

        # Subsequent read opens a fresh healthy connection and succeeds cleanly
        async with store._read_connect() as fresh_db:
            assert fresh_db is not cancelled_conn
            cur = await fresh_db.execute("SELECT COUNT(*) FROM graph_nodes")
            row = await cur.fetchone()
            assert row[0] == 1
        assert len(store._read_pool) == 1
        assert store._read_pool[0] is not cancelled_conn
    finally:
        await store.close()
        assert not store._closing_tasks
