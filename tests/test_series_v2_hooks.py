"""Unit tests for Dynamics Memory v2.0 cross-plugin hooks and series_panels presence."""

from __future__ import annotations

import inspect
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

from astrbot_plugin_dynamics_memory import series_panels
from astrbot_plugin_dynamics_memory.core.series import PLUGIN, SeriesMixin
from astrbot_plugin_dynamics_memory.storage.schema import GRAPH_SCHEMA_STATEMENTS


def _init_memory_db(db_path: Path, docs: list[tuple[int, str, dict]], nodes=()):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT, metadata TEXT)"
        )
        for doc_id, content, meta in docs:
            conn.execute(
                "INSERT INTO documents (id, content, metadata) VALUES (?, ?, ?)",
                (doc_id, content, json.dumps(meta, ensure_ascii=False)),
            )
        for stmt in GRAPH_SCHEMA_STATEMENTS:
            if "fts5" in stmt.lower():
                continue
            conn.execute(stmt)
        now = datetime.now(timezone.utc).isoformat()
        for node in nodes:
            if len(node) == 5:
                node_id, node_type, node_value, session_id, node_persona = node
            else:
                node_id, node_type, node_value, session_id = node
                node_persona = ""
            conn.execute(
                "INSERT INTO graph_nodes (id, node_key, node_type, node_value, canonical_value, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, '{}', ?, ?)",
                (
                    node_id,
                    f"{node_type}:{node_value.lower()}",
                    node_type,
                    node_value,
                    node_value.lower(),
                    now,
                    now,
                ),
            )
            if session_id or node_persona:
                conn.execute(
                    "INSERT INTO graph_entries (id, entry_key, source_memory_id, session_id, persona_id, entry_type, content, metadata, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 'fact', ?, '{}', ?, ?)",
                    (
                        node_id,
                        f"entry:{node_id}",
                        1,
                        session_id,
                        node_persona,
                        node_value,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    "INSERT INTO graph_entry_nodes (entry_id, node_id) VALUES (?, ?)",
                    (node_id, node_id),
                )
        conn.commit()
    finally:
        conn.close()


def _init_conflicts_db(db_path: Path, statuses: list[str]):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """CREATE TABLE series_source_conflicts (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                original_hash TEXT NOT NULL, incoming_hash TEXT NOT NULL, incoming_envelope TEXT NOT NULL,
                created_at REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                revision INTEGER NOT NULL DEFAULT 0, reviewer TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT ''
            )"""
        )
        for idx, status in enumerate(statuses):
            conn.execute(
                "INSERT INTO series_source_conflicts (id, session_id, event_id, original_hash, incoming_hash, incoming_envelope, created_at, status) "
                "VALUES (?, 'qq:GroupMessage:1', ?, 'a', 'b', '{}', ?, ?)",
                (f"c{idx}", f"ev{idx}", time.time(), status),
            )
        conn.commit()
    finally:
        conn.close()


def _make_owner(
    tmp_path: Path,
    *,
    enabled: bool = True,
    initialized: bool = True,
    stars=None,
    config: dict | None = None,
):
    class _Cfg:
        def __init__(self, data):
            self._data = data or {}

        def get(self, key, default=None):
            return self._data.get(key, default)

    owner = SeriesMixin()
    owner.context = NS(get_all_stars=lambda: stars if stars is not None else {})
    owner._data_dir = str(tmp_path)
    owner._series_enabled = enabled
    owner._terminating = False
    owner.initializer = NS(is_initialized=initialized, data_dir=str(tmp_path))
    owner.config_manager = _Cfg(config or {})
    return owner


def test_get_series_summary_states_and_standalone_safety(tmp_path):
    assert not inspect.iscoroutinefunction(SeriesMixin.get_series_summary)
    now = time.time()
    _init_memory_db(
        tmp_path / "livingmemory.db",
        [
            (
                1,
                "喜欢拿铁",
                {"status": "active", "create_time": now, "update_time": now},
            ),
            (2, "旧记录", {"status": "archived", "create_time": now}),
        ],
    )

    owner = _make_owner(tmp_path, enabled=True, initialized=True)
    summary = owner.get_series_summary()
    assert summary["badge"] == {"count": 0, "tone": "accent", "label": "运行正常"}
    assert "1 条活跃记忆" in summary["pulse"]["headline"]
    assert summary["pulse"]["updated_at"]

    _init_conflicts_db(
        tmp_path / "conversations.db", ["pending", "pending", "reviewed"]
    )
    conflict_summary = owner.get_series_summary()
    assert conflict_summary["badge"]["count"] == 2
    assert conflict_summary["badge"]["tone"] == "warm"
    assert "2 条来源冲突" in conflict_summary["badge"]["label"]

    legacy_owner = _make_owner(
        tmp_path,
        stars={
            "old": NS(
                name="astrbot_plugin_livingmemory",
                activated=True,
                star_cls=NS(),
            )
        },
    )
    danger_summary = legacy_owner.get_series_summary()
    assert danger_summary["badge"]["count"] == 1
    assert danger_summary["badge"]["tone"] == "danger"

    disabled_owner = _make_owner(
        tmp_path / "empty_dir", enabled=False, initialized=False
    )
    disabled_summary = disabled_owner.get_series_summary()
    assert disabled_summary["badge"]["tone"] == "muted"
    assert disabled_summary["badge"]["count"] == 0

    booting_owner = _make_owner(tmp_path / "empty_dir", enabled=True, initialized=False)
    assert booting_owner.get_series_summary()["badge"]["tone"] == "info"


def test_series_search_items_filters_scope_and_entities(tmp_path):
    now = time.time()
    _init_memory_db(
        tmp_path / "livingmemory.db",
        [
            (
                10,
                "小明最喜欢抹茶拿铁",
                {
                    "status": "active",
                    "memory_type": "PREFERENCE",
                    "session_id": "qq:GroupMessage:100",
                    "importance": 0.9,
                    "create_time": now,
                },
            ),
            (
                11,
                "已删除的抹茶记录",
                {
                    "status": "deleted",
                    "memory_type": "FACT",
                    "session_id": "qq:GroupMessage:100",
                    "importance": 0.9,
                    "create_time": now,
                },
            ),
            (
                12,
                "另一个群的抹茶蛋糕",
                {
                    "status": "active",
                    "memory_type": "FACT",
                    "session_id": "qq:GroupMessage:200",
                    "importance": 0.5,
                    "create_time": now,
                },
            ),
        ],
        nodes=[
            (1, "topic", "抹茶研究所", "qq:GroupMessage:100"),
            (2, "topic", "抹茶甜点铺", "qq:GroupMessage:200"),
        ],
    )
    owner = _make_owner(tmp_path)

    assert owner.series_search_items("") == []
    assert owner.series_search_items("%_") == []

    scoped = owner.series_search_items("抹茶", scope="qq:GroupMessage:100", limit=10)
    assert len(scoped) == 2
    assert {item["kind"] for item in scoped} == {"memory", "entity"}
    for item in scoped:
        assert item["plugin_id"] == PLUGIN
        assert item["page"] == "dashboard"
        assert isinstance(item["deep_link"], dict)
        assert item["title"] and item["subtitle"]

    all_items = owner.series_search_items("抹茶", limit=2)
    assert len(all_items) == 2

    empty_owner = _make_owner(tmp_path / "missing_dir")
    assert empty_owner.series_search_items("抹茶") == []


def test_get_series_scope_status_whitelist_and_counts(tmp_path):
    now = time.time()
    old = now - 86400 * 3
    _init_memory_db(
        tmp_path / "livingmemory.db",
        [
            (
                1,
                "今天的事实",
                {
                    "status": "active",
                    "session_id": "qq:GroupMessage:42",
                    "persona_id": "anan",
                    "create_time": now,
                },
            ),
            (
                2,
                "三天前的事实",
                {
                    "status": "active",
                    "session_id": "qq:GroupMessage:42",
                    "persona_id": "anan",
                    "create_time": old,
                },
            ),
        ],
    )
    owner = _make_owner(tmp_path)
    status = owner.get_series_scope_status("qq:GroupMessage:42")
    assert status["enabled"] is True
    assert status["supported"] is False
    assert status["persona_id"] == "anan"
    assert status["activity_today"] == 1
    assert "共 2 条记忆" in status["summary"]
    assert status["version"]

    wl_owner = _make_owner(
        tmp_path,
        config={
            "access_control.whitelist_enabled": True,
            "access_control.allowed_ids": "999",
        },
    )
    denied = wl_owner.get_series_scope_status("qq:GroupMessage:42")
    assert denied["enabled"] is False
    assert "白名单" in denied["summary"]


def test_get_journal_events_date_persona_scope_and_contract(tmp_path):
    now = time.time()
    yesterday = now - 86400 * 2
    _init_memory_db(
        tmp_path / "livingmemory.db",
        [
            (
                21,
                "和大家约好周末去露营",
                {
                    "status": "active",
                    "memory_type": "EVENT",
                    "session_id": "qq:GroupMessage:42",
                    "persona_id": "anan",
                    "importance": 0.9,
                    "create_time": now,
                },
            ),
            (
                22,
                "普通记录",
                {
                    "status": "active",
                    "memory_type": "FACT",
                    "session_id": "qq:GroupMessage:42",
                    "persona_id": "other",
                    "importance": 0.4,
                    "create_time": now - 10,
                },
            ),
            (
                23,
                "前天的旧记忆",
                {
                    "status": "active",
                    "memory_type": "FACT",
                    "session_id": "qq:GroupMessage:42",
                    "persona_id": "anan",
                    "importance": 0.8,
                    "create_time": yesterday,
                },
            ),
        ],
    )
    owner = _make_owner(tmp_path)
    events = owner.get_journal_events(
        persona="anan", scope="qq:GroupMessage:42", date="today", limit=10
    )
    assert len(events) == 1
    ev = events[0]
    assert ev["id"] == "memory:21"
    assert ev["category"] == "memory"
    assert ev["source_label"] == "群间记忆"
    assert ev["tone"] == "warm"
    assert ev["target_plugin"] == PLUGIN
    assert ev["target_route"] == "dashboard"
    assert ev["target_id"] == "21"
    assert ev["occurred_at"]
    assert "露营" in ev["title"]

    assert owner.get_journal_events(date="not-a-date") == []
    assert _make_owner(tmp_path / "missing").get_journal_events() == []


def test_series_panels_presence_badge_pulse_cache_and_catalog(tmp_path):
    series_panels.invalidate_summary_cache()
    calls = {"count": 0}

    class _GoodPlugin:
        def get_dynamics_capabilities(self):
            return {"schema_version": 1, "ready": True, "enabled": True}

        def get_series_summary(self):
            calls["count"] += 1
            return {
                "badge": {"count": 3, "tone": "warm", "label": "3 项待审"},
                "pulse": {
                    "headline": "今日已审 5 条",
                    "updated_at": "2026-11-21T02:00:00Z",
                },
            }

    class _BrokenPlugin:
        def get_dynamics_capabilities(self):
            return {"schema_version": 1, "ready": True, "enabled": True}

        def get_series_summary(self):
            raise RuntimeError("boom")

    class _AsyncPlugin:
        def get_dynamics_capabilities(self):
            return {"schema_version": 1, "ready": True, "enabled": True}

        async def get_series_summary(self):
            return {"badge": {"count": 9, "tone": "danger", "label": "async"}}

    good = _GoodPlugin()
    broken = _BrokenPlugin()
    async_plug = _AsyncPlugin()

    ctx = NS(
        get_all_stars=lambda: {
            "learning": NS(
                name="astrbot_plugin_dynamics_learning",
                activated=True,
                version="1.0.0",
                star_cls=good,
            ),
            "horizons": NS(
                name="astrbot_plugin_dynamics_horizons",
                activated=True,
                version="1.0.0",
                star_cls=broken,
            ),
            "world": NS(
                name="astrbot_plugin_dynamics_world",
                activated=True,
                version="1.0.0",
                star_cls=async_plug,
            ),
        }
    )

    first = series_panels.presence(ctx, PLUGIN, root=tmp_path)
    by_id = {row["id"]: row for row in first["plugins"]}
    assert by_id["astrbot_plugin_chat_dynamics"]["label"] == "对话主控"
    assert by_id["astrbot_plugin_group_chronicle"]["label"] == "群文化沿革"

    assert by_id["astrbot_plugin_dynamics_learning"]["badge"] == {
        "count": 3,
        "tone": "warm",
        "label": "3 项待审",
    }
    assert (
        by_id["astrbot_plugin_dynamics_learning"]["pulse"]["headline"]
        == "今日已审 5 条"
    )
    assert by_id["astrbot_plugin_dynamics_horizons"]["badge"] is None
    assert by_id["astrbot_plugin_dynamics_horizons"]["pulse"] is None
    assert by_id["astrbot_plugin_dynamics_world"]["badge"] is None
    assert calls["count"] == 1

    # Cached within TTL
    second = series_panels.presence(ctx, PLUGIN, root=tmp_path)
    assert calls["count"] == 1
    assert (
        next(
            r
            for r in second["plugins"]
            if r["id"] == "astrbot_plugin_dynamics_learning"
        )["badge"]["count"]
        == 3
    )

    series_panels.invalidate_summary_cache()
    series_panels.presence(ctx, PLUGIN, root=tmp_path)
    assert calls["count"] == 2


def test_journal_and_search_strict_persona_isolation_and_unbound_scope(tmp_path):
    now = time.time()
    scope = "qq:GroupMessage:777"
    _init_memory_db(
        tmp_path / "livingmemory.db",
        [
            (
                31,
                "角色A记录的抹茶偏好",
                {
                    "status": "active",
                    "memory_type": "PREFERENCE",
                    "session_id": scope,
                    "persona_id": "persona_a",
                    "importance": 0.9,
                    "create_time": now,
                },
            ),
            (
                32,
                "角色B记录的抹茶偏好",
                {
                    "status": "active",
                    "memory_type": "PREFERENCE",
                    "session_id": scope,
                    "persona_id": "persona_b",
                    "importance": 0.9,
                    "create_time": now,
                },
            ),
        ],
        nodes=[
            (101, "topic", "抹茶A基地", scope, "persona_a"),
            (102, "topic", "抹茶B基地", scope, "persona_b"),
        ],
    )
    owner = _make_owner(tmp_path)

    # Cross-persona switch in the same group strictly isolates each persona.
    events_a = owner.get_journal_events(persona="persona_a", scope=scope, date="today")
    assert [ev["id"] for ev in events_a] == ["memory:31"]
    events_b = owner.get_journal_events(persona="persona_b", scope=scope, date="today")
    assert [ev["id"] for ev in events_b] == ["memory:32"]

    # Specified group with empty or unbound persona MUST NOT relax to all personas.
    assert owner.get_journal_events(persona="", scope=scope, date="today") == []
    assert owner.get_journal_events(persona="__none__", scope=scope, date="today") == []
    assert owner.get_journal_events(persona="[%None]", scope=scope, date="today") == []

    # Explicit global query (no scope, no persona) keeps global visibility.
    global_events = owner.get_journal_events(persona="", scope="", date="today")
    assert {ev["id"] for ev in global_events} == {"memory:31", "memory:32"}

    # Federated search respects optional persona filter for both memories and graph entities.
    hits_a = owner.series_search_items(
        "抹茶", scope=scope, limit=10, persona="persona_a"
    )
    assert {h["title"] for h in hits_a} == {"角色A记录的抹茶偏好", "抹茶A基地"}
    hits_b = owner.series_search_items(
        "抹茶", scope=scope, limit=10, persona="persona_b"
    )
    assert {h["title"] for h in hits_b} == {"角色B记录的抹茶偏好", "抹茶B基地"}
    assert owner.series_search_items("抹茶", scope=scope, persona="__none__") == []


def test_journal_date_boundaries_consistent_across_timezones(tmp_path, monkeypatch):
    from datetime import timedelta

    from astrbot_plugin_dynamics_memory.core import series as series_mod

    # 2026-11-20 15:30 UTC == 2026-11-20 23:30+08:00 (previous Shanghai day)
    # 2026-11-20 16:30 UTC == 2026-11-21 00:30+08:00 (current Shanghai day)
    prev_shanghai_ts = datetime(2026, 11, 20, 15, 30, tzinfo=timezone.utc).timestamp()
    curr_shanghai_ts = datetime(2026, 11, 20, 16, 30, tzinfo=timezone.utc).timestamp()
    now_utc_ts = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc).timestamp()

    scope = "qq:GroupMessage:888"
    _init_memory_db(
        tmp_path / "livingmemory.db",
        [
            (
                41,
                "上海昨日深夜记忆",
                {
                    "status": "active",
                    "session_id": scope,
                    "persona_id": "anan",
                    "create_time": prev_shanghai_ts,
                },
            ),
            (
                42,
                "上海今日凌晨记忆",
                {
                    "status": "active",
                    "session_id": scope,
                    "persona_id": "anan",
                    "create_time": curr_shanghai_ts,
                },
            ),
        ],
    )
    owner = _make_owner(tmp_path)

    # Simulate UTC host clock at 2026-11-20 17:00 UTC (2026-11-21 01:00+08:00).
    monkeypatch.setattr(series_mod.time, "time", lambda: now_utc_ts)

    # Even if host timezone were UTC or UTC-5, _day_bounds uses UTC+8 consistently with World.
    expected_start = datetime(
        2026, 11, 21, 0, 0, tzinfo=timezone(timedelta(hours=8))
    ).timestamp()
    assert series_mod._day_bounds("today") == (expected_start, expected_start + 86400.0)
    assert series_mod._day_bounds("2026-11-21") == (
        expected_start,
        expected_start + 86400.0,
    )

    today_events = owner.get_journal_events(persona="anan", scope=scope, date="today")
    assert [ev["id"] for ev in today_events] == ["memory:42"]

    date_events = owner.get_journal_events(
        persona="anan", scope=scope, date="2026-11-21"
    )
    assert [ev["id"] for ev in date_events] == ["memory:42"]

    prev_events = owner.get_journal_events(
        persona="anan", scope=scope, date="2026-11-20"
    )
    assert [ev["id"] for ev in prev_events] == ["memory:41"]

    # Explicit start_ts/end_ts from World overrides date parsing cleanly.
    explicit = owner.get_journal_events(
        persona="anan",
        scope=scope,
        date="2026-11-21",
        start_ts=prev_shanghai_ts - 60,
        end_ts=prev_shanghai_ts + 60,
    )
    assert [ev["id"] for ev in explicit] == ["memory:41"]

    # Invalid explicit bounds return empty list safely.
    assert (
        owner.get_journal_events(
            persona="anan", scope=scope, start_ts=100.0, end_ts=100.0
        )
        == []
    )
    assert (
        owner.get_journal_events(
            persona="anan", scope=scope, start_ts=100.0, end_ts=None
        )
        == []
    )
    assert (
        owner.get_journal_events(
            persona="anan", scope=scope, start_ts=float("nan"), end_ts=200.0
        )
        == []
    )

    # activity_today in scope status also respects UTC+8 boundary.
    status = owner.get_series_scope_status(scope)
    assert status["activity_today"] == 1
