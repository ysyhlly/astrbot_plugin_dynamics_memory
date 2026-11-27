"""Wave 3 unit & integration tests: Dynamics Event Bus topic archival,
cross-5-plugin milestone keepsake bridge, and Provider stub verification.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from astrbot_plugin_dynamics_memory.core.event_bus_milestones import (
    EventBusAndMilestoneService,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.governance_handler import (
    GovernanceHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import PageApiUtils
from quart import Quart


def _make_ctx(plugins: dict[str, object]):
    stars = [
        SimpleNamespace(name=name, activated=True, star_cls=owner)
        for name, owner in plugins.items()
    ]
    return SimpleNamespace(get_all_stars=lambda: stars)


@pytest.mark.asyncio
async def test_event_bus_subscribe_unsubscribe_and_topic_archived_privacy_dedup(
    tmp_path: Path,
):
    subscribed: dict[str, tuple[str, object]] = {}
    unsubscribed: list[str] = []

    class _FakeChatDynamics:
        def subscribe_dynamics_event(self, event_type, callback, *, owner=""):
            token = f"tok-{event_type}-{len(subscribed) + 1}"
            subscribed[token] = (event_type, callback)
            return token

        def unsubscribe_dynamics_event(self, token):
            unsubscribed.append(token)
            return subscribed.pop(token, None) is not None

        def get_series_privacy_muted(self, umo, member_id, scope=""):
            # u_muted is privacy-muted for memory
            return {"muted": member_id == "u_muted"}

    added_memories: list[dict] = []

    class _FakeEngine:
        async def add_memory(
            self, *, content, session_id=None, importance=0.5, metadata=None
        ):
            doc_id = len(added_memories) + 101
            added_memories.append(
                {
                    "id": doc_id,
                    "content": content,
                    "session_id": session_id,
                    "importance": importance,
                    "metadata": metadata or {},
                }
            )
            return doc_id

    cd = _FakeChatDynamics()
    plugin = SimpleNamespace(
        context=_make_ctx({"astrbot_plugin_chat_dynamics": cd}),
        initializer=SimpleNamespace(
            memory_engine=_FakeEngine(), conversation_manager=None
        ),
        series_runtime_allowed=lambda: True,
    )

    svc = EventBusAndMilestoneService(plugin, tmp_path / "milestones.sqlite3")
    assert svc.ensure_subscribed() is True
    assert len(subscribed) == 1

    # 1. All participants muted -> skipped cleanly
    res_muted = await svc.handle_topic_archived(
        {
            "umo": "aiocqhttp:GroupMessage:10001",
            "topic_id": "t_muted",
            "summary": "仅静音成员参与的讨论",
            "participants": [{"sender_id": "u_muted", "display_name": "MutedUser"}],
            "message_count": 6,
        }
    )
    assert res_muted["accepted"] is False
    assert res_muted["reason"] == "all_participants_muted"
    assert len(added_memories) == 0

    # 2. Mixed participants -> muted member stripped, non-muted archived with high SNR
    res_ok = await svc.handle_topic_archived(
        {
            "umo": "aiocqhttp:GroupMessage:10001",
            "topic_id": "t_game_night",
            "summary": "群友约定周五晚上一起联机开荒新版本副本",
            "participants": [
                {"sender_id": "u_muted", "display_name": "MutedUser"},
                {"sender_id": "u_alice", "display_name": "Alice"},
            ],
            "message_count": 12,
        }
    )
    assert res_ok["accepted"] is True
    assert res_ok["doc_id"] == 101
    assert res_ok["filtered_muted_count"] == 1
    assert added_memories[0]["metadata"]["memory_origin"] == "topic_archived"
    assert added_memories[0]["metadata"]["participants"] == ["Alice"]

    # 3. Duplicate topic_id -> rejected without creating second memory
    res_dup = await svc.handle_topic_archived(
        {
            "umo": "aiocqhttp:GroupMessage:10001",
            "topic_id": "t_game_night",
            "summary": "群友约定周五晚上一起联机开荒新版本副本",
            "participants": [{"sender_id": "u_alice", "display_name": "Alice"}],
            "message_count": 12,
        }
    )
    assert res_dup["accepted"] is False
    assert res_dup["reason"] == "duplicate"
    assert len(added_memories) == 1

    # 4. Unsubscribe on terminate
    assert svc.unsubscribe() is True
    assert len(unsubscribed) == 1
    assert len(subscribed) == 0


@pytest.mark.asyncio
async def test_milestone_keepsake_cross_plugin_chain_and_daily_rate_limit(
    tmp_path: Path,
):
    visual_calls: list[dict] = []
    world_calls: list[dict] = []
    bus_events: list[tuple[str, dict]] = []
    updated_meta: dict[int, dict] = {}

    class _FakeVisual:
        async def generate_series_artwork(self, prompt, **kwargs):
            visual_calls.append({"prompt": prompt, **kwargs})
            return {"ok": True, "image_path": "/data/polaroid_42.png"}

    class _FakeWorld:
        async def register_external_keepsake(self, **kwargs):
            world_calls.append(kwargs)
            return {
                "ok": True,
                "keepsake_id": "ks_99",
                "deep_link": "astrbot_plugin_dynamics_world?page=cabin&keepsake=ks_99",
            }

    class _FakeChatDynamics:
        async def emit_dynamics_event(self, event_type, payload):
            bus_events.append((event_type, payload))
            return 2

    class _FakeEngine:
        async def get_memory_by_id(self, memory_id: int):
            return {
                "id": memory_id,
                "text": "大家一起庆祝建群一周年并通宵通关合作副本",
                "metadata": {
                    "session_id": "aiocqhttp:GroupMessage:20002",
                    "persona_id": "p_main",
                    "participants": ["Alice", "Bob"],
                    "topics": ["一周年庆功"],
                    "importance": 0.92,
                },
            }

        async def update_memory(self, memory_id, content=None, metadata_updates=None):
            updated_meta[memory_id] = metadata_updates or {}
            return True

    engine = _FakeEngine()
    plugin = SimpleNamespace(
        context=_make_ctx(
            {
                "astrbot_plugin_visual_media": _FakeVisual(),
                "astrbot_plugin_dynamics_world": _FakeWorld(),
                "astrbot_plugin_chat_dynamics": _FakeChatDynamics(),
            }
        ),
        initializer=SimpleNamespace(memory_engine=engine),
        series_runtime_allowed=lambda: True,
    )
    svc = EventBusAndMilestoneService(plugin, tmp_path / "ms.sqlite3")
    plugin.event_bus_service = svc

    meta = {
        "session_id": "aiocqhttp:GroupMessage:20002",
        "persona_id": "p_main",
        "participants": ["Alice", "Bob"],
        "topics": ["一周年庆功"],
    }
    assert svc.is_milestone_candidate("大家庆祝建群一周年", meta, 0.92) is True
    assert (
        svc.is_milestone_candidate("今天吃了午饭", {"participants": ["Alice"]}, 0.5)
        is False
    )

    # First milestone of the day succeeds & bridges visual_media + dynamics_world + chat_dynamics
    first = await svc.trigger_milestone_keepsake(
        memory_id=42,
        umo="aiocqhttp:GroupMessage:20002",
        content="大家一起庆祝建群一周年并通宵通关合作副本",
        metadata=meta,
    )
    assert first["ok"] is True
    ms = first["milestone"]
    assert ms["status"] == "synced_world"
    assert ms["image_path"] == "/data/polaroid_42.png"
    assert ms["keepsake_id"] == "ks_99"
    assert ms["event_emitted"] is True

    assert visual_calls[0]["style"] == "polaroid"
    assert visual_calls[0]["source_plugin"] == "dynamics_memory"
    assert world_calls[0]["source_plugin"] == "dynamics_memory"
    assert world_calls[0]["image_path"] == "/data/polaroid_42.png"
    assert bus_events[0][0] == "milestone_achieved"
    assert updated_meta[42]["milestone_keepsake"]["keepsake_id"] == "ks_99"

    # Second milestone same group same day blocked by daily rate limit (max 1/group/day)
    second = await svc.trigger_milestone_keepsake(
        memory_id=43,
        umo="aiocqhttp:GroupMessage:20002",
        content="同一天另一件高重要度大事纪念日",
        metadata=meta,
    )
    assert second["ok"] is False
    assert second["reason"] == "daily_limit_reached"

    # GovernanceHandler Page API endpoints: GET milestones + POST milestones/trigger (with force)
    handler = GovernanceHandler(PageApiUtils(), plugin)
    app = Quart(__name__)
    async with app.test_request_context("/milestones"):
        listed = await handler.list_milestones()
        assert listed["status"] == "ok"
        assert listed["data"]["total"] == 1

    async with app.test_request_context(
        "/milestones/trigger",
        method="POST",
        json={"memory_id": 44, "force": True},
    ):
        forced = await handler.trigger_milestone(engine)
        assert forced["status"] == "ok"
        assert forced["data"]["milestone"]["memory_id"] == 44


@pytest.mark.asyncio
async def test_event_bus_guards_against_mock_objects(tmp_path: Path):
    """MagicMock contexts must never produce bogus tokens or keepsake IDs."""
    plugin = SimpleNamespace(
        context=MagicMock(),
        initializer=None,
        series_runtime_allowed=lambda: True,
    )
    svc = EventBusAndMilestoneService(plugin, tmp_path / "mock.sqlite3")
    assert svc.ensure_subscribed() is False

    res = await svc.trigger_milestone_keepsake(
        memory_id=55,
        umo="aiocqhttp:GroupMessage:30003",
        content="一起庆祝百天纪念日",
        metadata={"participants": ["Alice", "Bob"]},
    )
    assert res["ok"] is True
    assert res["milestone"]["image_path"] == ""
    assert res["milestone"]["keepsake_id"] == ""
    assert res["milestone"]["status"] == "recorded"
