"""Wave 2 comprehensive unit & integration tests for astrbot_plugin_dynamics_memory."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
from astrbot_plugin_dynamics_memory.core.managers.memory_engine_recall import (
    MemoryEngineRecallMixin,
)
from astrbot_plugin_dynamics_memory.core.memory_scope import (
    fetch_group_culture_context,
)
from astrbot_plugin_dynamics_memory.core.models.graph_models import (
    GraphEdge,
    GraphEntry,
    GraphNode,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.governance_handler import (
    GovernanceHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.graph_handler import (
    GraphHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.settings_handler import (
    HOT_RELOAD_PATHS,
    SettingsHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import PageApiUtils
from astrbot_plugin_dynamics_memory.core.self_service import (
    CorrectionQueueStore,
    MemberMemoryCardService,
)
from astrbot_plugin_dynamics_memory.core.series import SeriesMixin
from astrbot_plugin_dynamics_memory.core.utils.formatting import (
    format_memories_for_injection,
)
from astrbot_plugin_dynamics_memory.storage.graph_store import GraphStore
from quart import Quart


class _FakeConfig:
    def __init__(self, data: dict | None = None):
        self.data = data or {}

    def get(self, key: str, default=None):
        cur = self.data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def apply_updates(self, values: dict):
        for path, val in values.items():
            cur = self.data
            parts = path.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = val
        return self.data


class _FakeEvent:
    def __init__(
        self,
        *,
        umo: str = "aiocqhttp:GroupMessage:10001",
        sender_id: str = "u100",
        sender_name: str = "Alice",
        platform: str = "aiocqhttp",
    ):
        self.unified_msg_origin = umo
        self._sender_id = sender_id
        self._sender_name = sender_name
        self._platform = platform

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return self._sender_name

    def get_platform_name(self):
        return self._platform

    def get_platform_id(self):
        return self._platform


async def _create_docs_db(db_path: Path) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            text TEXT,
            metadata TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    await conn.commit()
    return conn


async def _insert_doc(
    conn: aiosqlite.Connection,
    text: str,
    metadata: dict,
    *,
    doc_id: str | None = None,
) -> int:
    cur = await conn.execute(
        """
        INSERT INTO documents (doc_id, text, metadata, created_at, updated_at)
        VALUES (?, ?, ?, '2026-01-01', '2026-01-01')
        """,
        (doc_id or "", text, json.dumps(metadata, ensure_ascii=False)),
    )
    await conn.commit()
    return int(cur.lastrowid)


class _SimpleMemoryEngine(MemoryEngineRecallMixin):
    def __init__(self, db_connection: aiosqlite.Connection, graph_store=None):
        self.db_connection = db_connection
        self.graph_store = graph_store
        self.updated_calls: list[tuple[int, str | None, dict | None]] = []
        self.archived_ids: list[int] = []

    async def get_memory_by_id(self, memory_id: int) -> dict | None:
        cur = await self.db_connection.execute(
            "SELECT id, doc_id, text, metadata FROM documents WHERE id = ?",
            (int(memory_id),),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "doc_id": row["doc_id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"] or "{}"),
        }

    async def update_memory(
        self,
        memory_id: int,
        content: str | None = None,
        metadata_updates: dict | None = None,
    ) -> bool:
        mem = await self.get_memory_by_id(memory_id)
        if not mem:
            return False
        new_text = content if content is not None else mem["text"]
        new_meta = {**mem["metadata"], **(metadata_updates or {})}
        await self.db_connection.execute(
            "UPDATE documents SET text = ?, metadata = ? WHERE id = ?",
            (new_text, json.dumps(new_meta, ensure_ascii=False), int(memory_id)),
        )
        await self.db_connection.commit()
        self.updated_calls.append((int(memory_id), content, metadata_updates))
        return True

    async def archive_memories(self, memory_ids: list[int]) -> None:
        for mid in memory_ids:
            await self.update_memory(mid, metadata_updates={"status": "archived"})
            self.archived_ids.append(int(mid))


# =====================================================================
# 1. Self-Service /我的记忆 Card & /纠错 Correction Queue
# =====================================================================


@pytest.mark.asyncio
async def test_member_card_strict_isolation_and_png_render(tmp_path: Path):
    conn = await _create_docs_db(tmp_path / "docs.db")
    try:
        now = time.time()
        # Alice's own memory in group 10001 (contains raw 9-digit QQ to verify masking)
        id_alice = await _insert_doc(
            conn,
            "Alice 喜欢在周末去咖啡馆画画，常联系 123456789",
            {
                "session_id": "aiocqhttp:GroupMessage:10001",
                "sender_id": "u100",
                "status": "active",
                "importance": 0.85,
                "topics": ["咖啡", "画画"],
                "create_time": now,
                "participant_identities": [
                    {"sender_id": "u100", "display_name": "Alice"}
                ],
            },
        )
        # Shared memory in group 10001 involving Alice & Bob
        id_shared = await _insert_doc(
            conn,
            "Alice 和 Bob 一起通关了联机副本",
            {
                "session_id": "aiocqhttp:GroupMessage:10001",
                "status": "active",
                "importance": 0.75,
                "topics": ["联机游戏"],
                "create_time": now,
                "participant_identities": [
                    {"sender_id": "u100", "display_name": "Alice"},
                    {"sender_id": "u200", "display_name": "Bob"},
                ],
            },
        )
        # Bob's private memory in same group (MUST NOT LEAK to Alice)
        await _insert_doc(
            conn,
            "Bob 的秘密计划",
            {
                "session_id": "aiocqhttp:GroupMessage:10001",
                "sender_id": "u200",
                "status": "active",
                "importance": 0.9,
                "create_time": now,
                "participant_identities": [
                    {"sender_id": "u200", "display_name": "Bob"}
                ],
            },
        )
        # Alice's memory in another group 99999 (MUST NOT LEAK into group 10001)
        await _insert_doc(
            conn,
            "Alice 在别的群说过的话",
            {
                "session_id": "aiocqhttp:GroupMessage:99999",
                "sender_id": "u100",
                "status": "active",
                "importance": 0.9,
                "create_time": now,
            },
        )

        engine = _SimpleMemoryEngine(conn)
        svc = MemberMemoryCardService(temp_dir=tmp_path / "cards")
        event = _FakeEvent(
            umo="aiocqhttp:GroupMessage:10001", sender_id="u100", sender_name="Alice"
        )
        cfg = _FakeConfig({"filtering_settings": {"memory_scope_mode": "session"}})

        card = await svc.collect_member_card_data(engine, event, cfg, context=None)
        assert card["allowed"] is True
        assert card["total_count"] == 2
        assert [m["id"] for m in card["personal_memories"]] == [id_alice]
        assert [m["id"] for m in card["shared_highlights"]] == [id_shared]
        assert "123456789" not in card["personal_memories"][0]["summary"]
        assert "***789" in card["personal_memories"][0]["summary"]
        assert "咖啡" in card["top_topics"]

        # Render PNG image via Pillow
        png_path = svc.render_card_image(card, output_path=tmp_path / "out.png")
        assert png_path is not None and png_path.is_file()
        assert png_path.stat().st_size > 1000

        # Render text fallback
        txt = svc.render_card_text(card)
        assert f"#{id_alice}" in txt
        assert "/纠错" in txt
        assert "Bob 的秘密计划" not in txt
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_member_card_respects_privacy_mute(tmp_path: Path):
    conn = await _create_docs_db(tmp_path / "docs.db")
    try:
        engine = _SimpleMemoryEngine(conn)
        svc = MemberMemoryCardService(temp_dir=tmp_path / "cards")
        event = _FakeEvent(
            umo="aiocqhttp:GroupMessage:10001", sender_id="u100", sender_name="Alice"
        )
        cfg = _FakeConfig({})

        # Fake chat_dynamics owner returning muted=True
        cd_owner = SimpleNamespace(
            get_series_privacy_muted=lambda umo, mid, module: {"muted": True}
        )
        star_meta = SimpleNamespace(
            name="astrbot_plugin_chat_dynamics", activated=True, star_cls=cd_owner
        )
        fake_ctx = SimpleNamespace(get_all_stars=lambda: [star_meta])

        card = await svc.collect_member_card_data(engine, event, cfg, context=fake_ctx)
        assert card["allowed"] is False
        assert card["reason"] == "privacy_muted"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_member_correction_queue_submit_and_admin_review(tmp_path: Path):
    conn = await _create_docs_db(tmp_path / "docs.db")
    try:
        now = time.time()
        id_alice = await _insert_doc(
            conn,
            "Alice 喜欢喝拿铁咖啡",
            {
                "session_id": "aiocqhttp:GroupMessage:10001",
                "sender_id": "u100",
                "status": "active",
                "importance": 0.7,
                "create_time": now,
            },
        )
        id_bob = await _insert_doc(
            conn,
            "Bob 喜欢骑行",
            {
                "session_id": "aiocqhttp:GroupMessage:10001",
                "sender_id": "u200",
                "status": "active",
                "importance": 0.7,
                "create_time": now,
            },
        )
        engine = _SimpleMemoryEngine(conn)
        store = CorrectionQueueStore(tmp_path / "corrections.sqlite3")
        svc = MemberMemoryCardService()
        cfg = _FakeConfig({})
        alice_event = _FakeEvent(
            umo="aiocqhttp:GroupMessage:10001", sender_id="u100", sender_name="Alice"
        )

        # Cannot submit correction for Bob's memory
        denied = await svc.submit_member_correction(
            memory_engine=engine,
            correction_store=store,
            event=alice_event,
            config_manager=cfg,
            context=None,
            memory_id=id_bob,
            description="我想乱改别人的记忆",
        )
        assert denied["ok"] is False

        # Submit for Alice's own memory
        submitted = await svc.submit_member_correction(
            memory_engine=engine,
            correction_store=store,
            event=alice_event,
            config_manager=cfg,
            context=None,
            memory_id=id_alice,
            description="Alice 其实更喜欢喝乌龙茶，不常喝咖啡",
        )
        assert submitted["ok"] is True
        pid = submitted["proposal"]["id"]

        # Admin reviews & approves via GovernanceHandler
        app = Quart(__name__)
        plugin = SimpleNamespace(correction_store=store)
        handler = GovernanceHandler(PageApiUtils(), plugin)

        async with app.test_request_context("/corrections?status=pending"):
            listed = await handler.list_corrections()
            assert listed["status"] == "ok"
            assert listed["data"]["pending_count"] == 1

        async with app.test_request_context(
            "/corrections/review",
            method="POST",
            json={
                "id": pid,
                "decision": "approve",
                "replacement": "Alice 更喜欢喝乌龙茶，平时很少喝咖啡",
                "review_note": "核实通过",
            },
        ):
            reviewed = await handler.review_correction(engine)
            assert reviewed["status"] == "ok"
            assert reviewed["data"]["proposal"]["status"] == "approved"

        updated_mem = await engine.get_memory_by_id(id_alice)
        assert updated_mem["text"] == "Alice 更喜欢喝乌龙茶，平时很少喝咖啡"
        assert (
            updated_mem["metadata"]["update_history"][-1]["source"]
            == "member_correction"
        )
    finally:
        await conn.close()


# =====================================================================
# 2. Group Culture Feeding Memory Recall
# =====================================================================


@pytest.mark.asyncio
async def test_group_culture_context_bridge_and_formatting():
    # Chronicle owner returning approved + pending entries
    async def _fake_get_culture(umo, query="", limit=4):
        return {
            "items": [
                {
                    "term": "夜宵局",
                    "meaning": "每晚十一点后群里约点烧烤",
                    "status": "approved",
                },
                {"term": "待审黑话", "meaning": "不应泄露给召回", "status": "pending"},
                {
                    "term": "摸鱼大师",
                    "meaning": "特指工作日下午准点冒泡的群友",
                    "status": "approved",
                },
            ]
        }

    chronicle_owner = SimpleNamespace(get_culture_context=_fake_get_culture)
    star_meta = SimpleNamespace(
        name="astrbot_plugin_group_chronicle", activated=True, star_cls=chronicle_owner
    )
    fake_ctx = SimpleNamespace(get_all_stars=lambda: [star_meta])

    items = await fetch_group_culture_context(
        fake_ctx, "aiocqhttp:GroupMessage:10001", query="今晚吃什么", limit=4
    )
    assert len(items) == 2
    assert items[0]["term"] == "夜宵局"
    assert all(it["term"] != "待审黑话" for it in items)

    # Guard against unittest.mock.Mock returning non-list truthy mocks
    mock_ctx = MagicMock()
    assert (
        await fetch_group_culture_context(mock_ctx, "aiocqhttp:GroupMessage:10001")
        == []
    )

    # Format memories with culture items
    memories = [
        {
            "id": 1,
            "content": "大家上周五去了老王烧烤店",
            "score": 0.9,
            "metadata": {"importance": 0.8, "create_time": time.time()},
        }
    ]
    plain_text = format_memories_for_injection(memories)
    with_culture = format_memories_for_injection(memories, culture_items=items)
    assert "夜宵局" not in plain_text
    assert "群内常用梗/黑话参考" in with_culture
    assert "夜宵局：每晚十一点后群里约点烧烤" in with_culture


# =====================================================================
# 3. Health Checkup, Dedup Merge, Bulk Governance & Quick Filters
# =====================================================================


@pytest.mark.asyncio
async def test_governance_health_merge_and_bulk_actions(tmp_path: Path):
    conn = await _create_docs_db(tmp_path / "docs.db")
    gstore = GraphStore(str(tmp_path / "graph.db"))
    await gstore.initialize()
    try:
        now = time.time()
        old_ts = now - 60 * 86400
        # Pair of near-duplicates
        id_d1 = await _insert_doc(
            conn,
            "小明非常喜欢周末去城东公园骑公路自行车锻炼身体",
            {
                "session_id": "g1",
                "status": "active",
                "importance": 0.8,
                "create_time": now,
            },
        )
        id_d2 = await _insert_doc(
            conn,
            "小明非常喜欢周末去城东公园骑公路自行车锻炼",
            {
                "session_id": "g1",
                "status": "active",
                "importance": 0.5,
                "create_time": now,
            },
        )
        # Pair of contradictions (shared term "香菜", opposing polarity)
        id_c1 = await _insert_doc(
            conn,
            "小红特别喜欢吃香菜火锅",
            {
                "session_id": "g1",
                "status": "active",
                "importance": 0.7,
                "topics": ["香菜"],
                "create_time": now,
            },
        )
        id_c2 = await _insert_doc(
            conn,
            "小红非常讨厌香菜，一点都不吃香菜",
            {
                "session_id": "g1",
                "status": "active",
                "importance": 0.7,
                "topics": ["香菜"],
                "create_time": now,
            },
        )
        # Dead memory (60 days old, 0 access, low importance)
        id_dead = await _insert_doc(
            conn,
            "某天路过超市买了一瓶矿泉水",
            {
                "session_id": "g1",
                "status": "active",
                "importance": 0.3,
                "access_count": 0,
                "create_time": old_ts,
            },
        )
        # Orphan graph node
        orphan_id = await gstore.upsert_node(
            GraphNode(
                node_type="topic",
                value="孤立话题",
                canonical_value="孤立话题",
            )
        )

        engine = _SimpleMemoryEngine(conn, graph_store=gstore)
        handler = GovernanceHandler(
            PageApiUtils(), SimpleNamespace(correction_store=None)
        )
        app = Quart(__name__)

        async with app.test_request_context("/governance/health?stale_days=30"):
            report = await handler.get_health_report(engine)
            assert report["status"] == "ok"
            data = report["data"]
            assert data["summary"]["duplicate_count"] >= 1
            assert data["summary"]["contradiction_count"] >= 1
            assert any(
                {c["memory_a_id"], c["memory_b_id"]} == {id_c1, id_c2}
                for c in data["contradictions"]
            )
            assert any(m["id"] == id_dead for m in data["dead_memories"])
            assert any(n["id"] == orphan_id for n in data["orphan_nodes"])

        # Dry-run merge preview
        async with app.test_request_context(
            "/governance/merge",
            method="POST",
            json={"keep_id": id_d1, "merge_ids": [id_d2], "dry_run": True},
        ):
            prev = await handler.merge_duplicates(engine)
            assert prev["status"] == "ok"
            assert prev["data"]["dry_run"] is True
            assert (await engine.get_memory_by_id(id_d2))["metadata"][
                "status"
            ] == "active"

        # Execute merge
        async with app.test_request_context(
            "/governance/merge",
            method="POST",
            json={"keep_id": id_d1, "merge_ids": [id_d2], "dry_run": False},
        ):
            merged = await handler.merge_duplicates(engine)
            assert merged["status"] == "ok"
            assert id_d2 in merged["data"]["archived_ids"]
            assert (await engine.get_memory_by_id(id_d2))["metadata"][
                "status"
            ] == "archived"

        # Bulk downweight & archive on dead memory
        async with app.test_request_context(
            "/governance/bulk_action",
            method="POST",
            json={"ids": [id_dead], "action": "downweight", "factor": 0.5},
        ):
            dw = await handler.bulk_action(engine)
            assert dw["status"] == "ok"
            assert (await engine.get_memory_by_id(id_dead))["metadata"][
                "importance"
            ] == pytest.approx(0.15)

        async with app.test_request_context(
            "/governance/bulk_action",
            method="POST",
            json={"ids": [id_dead], "action": "archive"},
        ):
            arc = await handler.bulk_action(engine)
            assert arc["status"] == "ok"
            assert (await engine.get_memory_by_id(id_dead))["metadata"][
                "status"
            ] == "archived"
    finally:
        await gstore.close()
        await conn.close()


@pytest.mark.asyncio
async def test_list_memories_page_wave2_quick_filters(tmp_path: Path):
    conn = await _create_docs_db(tmp_path / "docs.db")
    try:
        now = time.time()
        old = now - 45 * 86400
        await _insert_doc(
            conn,
            "事实记忆 A",
            {
                "memory_type": "FACTUAL",
                "importance": 0.25,
                "access_count": 0,
                "create_time": old,
                "memory_origin": "auto",
                "has_source": True,
                "source_message_count": 3,
                "status": "active",
            },
        )
        await _insert_doc(
            conn,
            "事实记忆 B",
            {
                "memory_type": "FACT",
                "importance": 0.9,
                "access_count": 5,
                "create_time": now,
                "memory_origin": "agent",
                "has_source": False,
                "status": "active",
            },
        )
        engine = _SimpleMemoryEngine(conn)

        # Canonical type group GROUP:FACT matches both FACT and FACTUAL
        res_group = await engine.list_memories_page(memory_type="GROUP:FACT")
        assert res_group["total"] == 2

        # Low importance + zero access + stale + has_source + origin
        res_filtered = await engine.list_memories_page(
            importance_max=0.35,
            zero_access=True,
            stale_days=30,
            memory_origin="auto",
            has_source=True,
        )
        assert res_filtered["total"] == 1
        assert res_filtered["items"][0]["text"] == "事实记忆 A"

        res_no_src = await engine.list_memories_page(
            has_source=False, memory_origin="agent"
        )
        assert res_no_src["total"] == 1
        assert res_no_src["items"][0]["text"] == "事实记忆 B"
    finally:
        await conn.close()


# =====================================================================
# 4. Graph Node Type Filter, Synonym Merge & Orphan Cleanup
# =====================================================================


@pytest.mark.asyncio
async def test_graph_store_merge_nodes_cleanup_orphans_and_type_filter(tmp_path: Path):
    gstore = GraphStore(str(tmp_path / "graph.db"))
    await gstore.initialize()
    try:
        ids = await gstore.upsert_nodes(
            [
                GraphNode("person", "Alice", "alice"),
                GraphNode("person", "爱丽丝", "爱丽丝"),
                GraphNode("topic", "乌龙茶", "乌龙茶"),
                GraphNode("topic", "画画", "画画"),
                GraphNode("topic", "孤立节点X", "孤立节点x"),
            ]
        )
        alice_id = ids["person:alice"]
        alias_id = ids["person:爱丽丝"]
        tea_id = ids["topic:乌龙茶"]
        orphan_id = ids["topic:孤立节点x"]

        await gstore.add_edge(
            GraphEdge(
                source_key="person:alice",
                target_key="topic:画画",
                relation_type="likes",
                source_memory_id=9,
                weight=1.0,
                confidence=0.9,
            ),
            ids,
        )
        edge_id = await gstore.add_edge(
            GraphEdge(
                source_key="person:爱丽丝",
                target_key="topic:乌龙茶",
                relation_type="likes",
                source_memory_id=10,
                weight=1.0,
                confidence=0.8,
            ),
            ids,
        )
        await gstore.add_entries(
            [
                GraphEntry(
                    entry_key="ent1",
                    source_memory_id=10,
                    session_id="g1",
                    persona_id="p1",
                    entry_type="relation",
                    relation_type="likes",
                    content="爱丽丝喜欢乌龙茶",
                    node_keys=["person:爱丽丝", "topic:乌龙茶"],
                )
            ],
            ids,
            {"person:爱丽丝|likes|topic:乌龙茶|10": edge_id},
        )

        # Orphan detection & cleanup must only remove orphan_x
        orphans = await gstore.list_orphan_nodes()
        assert [o["id"] for o in orphans] == [orphan_id]
        cleaned = await gstore.cleanup_orphan_nodes()
        assert cleaned["deleted_count"] == 1
        assert cleaned["deleted_ids"] == [orphan_id]

        # Merge synonym node person:alice_alias into person:alice
        merged = await gstore.merge_nodes(alice_id, [alias_id])
        assert merged["target_node_id"] == alice_id
        assert merged["merged_node_ids"] == [alias_id]
        assert merged["relinked_edges"] == 1
        assert merged["relinked_entries"] == 1

        # Verify graph_handler node type filter
        snapshot = {
            "nodes": [
                {"id": alice_id, "type": "person"},
                {"id": tea_id, "type": "topic"},
            ],
            "edges": [{"source": alice_id, "target": tea_id}],
            "entries": [],
            "memories": [],
        }
        filtered = GraphHandler._filter_snapshot_by_node_types(snapshot, ["person"])
        assert [n["id"] for n in filtered["nodes"]] == [alice_id]
        assert filtered["edges"] == []
    finally:
        await gstore.close()


# =====================================================================
# 5. Batch Conflict Resolution & Settings Hot vs Reload Classification
# =====================================================================


@pytest.mark.asyncio
async def test_series_batch_resolve_and_pending_count(monkeypatch):
    class _FakeConflictStore:
        def __init__(self):
            self.resolved = []

        async def list_series_conflicts(self):
            return {
                "items": [
                    {"id": "c1", "revision": 1, "status": "pending"},
                    {"id": "c2", "revision": 2, "status": "pending"},
                    {"id": "c3", "revision": 1, "status": "resolved"},
                ]
            }

        async def resolve_series_conflict(self, cid, rev, reason, actor):
            self.resolved.append((cid, rev, reason, actor))
            return {"id": cid, "status": "resolved"}

    class _DummySeries(SeriesMixin):
        def __init__(self):
            self.context = MagicMock()
            self.store = _FakeConflictStore()
            self.initializer = SimpleNamespace(
                conversation_manager=SimpleNamespace(store=self.store)
            )

        async def _series_admin(self, *, check_csrf: bool = False):
            return "admin_user"

        async def _ensure_plugin_ready(self, wait: bool = False):
            return True, ""

    dummy = _DummySeries()
    listed = await dummy.series_conflict_list()
    assert listed["status"] == "ok"
    assert listed["data"]["pending_count"] == 2

    fake_web_mod = SimpleNamespace(
        request=SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "items": [{"id": "c1", "revision": 1}, {"id": "c2", "revision": 2}],
                    "reason": "批量确认保留原文",
                }
            )
        )
    )
    monkeypatch.setitem(sys.modules, "astrbot.api.web", fake_web_mod)
    batch = await dummy.series_resolve_conflicts_batch()
    assert batch["status"] == "ok"
    assert batch["data"]["resolved_count"] == 2
    assert len(dummy.store.resolved) == 2


@pytest.mark.asyncio
async def test_settings_handler_hot_vs_reload_classification():
    cfg = _FakeConfig(
        {
            "recall_engine": {"top_k": 5},
            "graph_memory": {"enabled": True},
        }
    )
    plugin = SimpleNamespace(config_manager=cfg, initializer=None)
    handler = SettingsHandler(PageApiUtils(), plugin)
    app = Quart(__name__)

    assert "recall_engine.top_k" in HOT_RELOAD_PATHS
    assert "graph_memory.enabled" not in HOT_RELOAD_PATHS

    # Hot-only change
    async with app.test_request_context(
        "/config/update",
        method="POST",
        json={"values": {"recall_engine.top_k": 8, "graph_memory.enabled": True}},
    ):
        res = await handler.update_config()
        assert res["status"] == "ok"
        assert res["data"]["hot_applied"] == ["recall_engine.top_k"]
        assert res["data"]["reload_required"] == []

    # Reload-required change
    async with app.test_request_context(
        "/config/update",
        method="POST",
        json={"values": {"recall_engine.top_k": 8, "graph_memory.enabled": False}},
    ):
        res2 = await handler.update_config()
        assert res2["status"] == "ok"
        assert res2["data"]["reload_required"] == ["graph_memory.enabled"]
        assert res2["data"]["hot_applied"] == []
