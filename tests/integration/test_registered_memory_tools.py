"""Run actual tools against the public evidence service and mutable member policy."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from tests.integration.test_public_evidence_pipeline import (
    pipeline as pipeline,
    event,
    SCOPE,
)
from tests.test_blocking_policy import FakeBlockingService
from astrbot_plugin_dynamics_memory.core.tools.memory_search_tool import (
    MemorySearchTool,
)
from astrbot_plugin_dynamics_memory.core.tools import memory_search_tool
from astrbot_plugin_dynamics_memory.core.recall_runtime import BoundedRecallTaskPool
from astrbot_plugin_dynamics_memory.core import tool_recall_guard


@pytest.mark.asyncio
async def test_managed_tool_returns_only_current_atoms_and_rechecks_member_policy(
    pipeline, monkeypatch
):
    p, model, store = pipeline
    blocker = FakeBlockingService()
    p.context.get_all_stars = lambda: [
        NS(name="astrbot_plugin_autonomous_blocking", star_cls=blocker, activated=True)
    ]
    platform = NS(
        meta=lambda: NS(id="fixture", name="fixture"),
        status=NS(value="running"),
        get_self_id=lambda: "bot",
    )
    p.context.get_platform_inst = lambda name: platform
    callbacks = []
    extra = {
        "_narrative_managed_v1": {"scope": dict(SCOPE)},
        "_cd_memory_feedback_v1": {"scope": dict(SCOPE), "card_version": ""},
    }
    host_event = NS(
        unified_msg_origin=SCOPE["umo"],
        get_platform_id=lambda: "fixture",
        get_platform_name=lambda: "fixture",
        get_self_id=lambda: "bot",
        get_sender_id=lambda: "reader-user",
        get_extra=lambda key, default=None: extra.get(key, default),
        _cd_tool_memory_sources_register_v1=lambda callback, **kw: (
            callbacks.append(callback) or True
        ),
    )
    assert await p._blocking_policy_gate.observe_trusted_event(host_event, SCOPE)
    p.evidence_service.material_policy = p._evidence_material_policy
    p.evidence_service.owner_readiness = p._blocking_policy_gate.owner_ready
    raw = event()
    await p.ingest_events([raw])
    source = {**raw, "source_kind": "event", "event_type": raw["type"]}
    view = await p.get_evidence_view(SCOPE, source_refs=[source])
    for _ in range(100):
        if view["atoms"]:
            break
        await asyncio.sleep(0.01)
        view = await p.get_evidence_view(SCOPE, source_refs=[source])
    assert view["atoms"]
    p.initializer.memory_engine = NS(
        search_narrative_history=AsyncMock(return_value=[]),
        search_narrative_semantic=AsyncMock(return_value=[]),
        get_memory=AsyncMock(return_value=None),
    )
    p._recall_runtime = BoundedRecallTaskPool()
    monkeypatch.setattr(memory_search_tool, "get_active_plugin", lambda: p)
    monkeypatch.setattr(tool_recall_guard, "get_active_plugin", lambda: p)
    monkeypatch.setattr(
        memory_search_tool, "get_persona_id", AsyncMock(return_value="reader")
    )
    monkeypatch.setattr(
        tool_recall_guard, "get_persona_id", AsyncMock(return_value="reader")
    )
    tool = MemorySearchTool(
        context=p.context,
        config_manager=p.config_manager,
        memory_engine=p.initializer.memory_engine,
        conversation_manager=p.initializer.conversation_manager,
        recall_runtime=p._recall_runtime,
    )
    try:
        result = json.loads(
            await tool.call(
                NS(context=NS(event=host_event)),
                query="coffee",
                k=3,
                include_source=True,
            )
        )
        assert result["status"] in {"ok", "partial"} and result["results"]
        assert callbacks and await callbacks[0]()
        blocker.rules = [
            {"target_uid": "reader-user", "mode": "shield", "active": True}
        ]
        assert not await callbacks[0]()
        blocked = json.loads(
            await tool.call(NS(context=NS(event=host_event)), query="coffee", k=3)
        )
        assert not blocked["results"]
        assert model.text_chat.await_count == 1
    finally:
        await p._recall_runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["blocking", "blocking_disabled"])
async def test_ordinary_tool_revalidates_exact_source_and_summary(
    pipeline, monkeypatch, policy
):
    import copy
    from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
    from astrbot_plugin_dynamics_memory.core.memory_source import (
        build_source_row_refs_v1,
    )
    from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
        ConversationManager,
    )
    from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
        HybridResult,
    )

    p, model, store = pipeline
    scope = {**SCOPE, "umo": "fixture:FriendMessage:user-1"}
    row = Message(
        id=0,
        session_id=scope["umo"],
        role="user",
        content="I prefer coffee.",
        sender_id="user-1",
        sender_name="Reader",
        platform="fixture",
        metadata={},
    )
    row.id = await store.add_message(row)
    metadata = {
        "status": "active",
        "session_id": scope["umo"],
        "source_session_id": scope["umo"],
        "persona_id": "reader",
        "interaction_type": "private_chat",
        "memory_visibility": "private",
        "source_row_refs_v1": build_source_row_refs_v1([row], scope["umo"]),
    }
    engine = NS(
        get_memory=AsyncMock(
            side_effect=lambda mid: {
                "text": "summary",
                "metadata": copy.deepcopy(metadata),
            }
        ),
        get_summary_source_snapshots=AsyncMock(
            side_effect=lambda mids: {
                mid: ("summary", copy.deepcopy(metadata), ())
                for mid in mids
                if mid == 9
            }
        ),
        get_memory_source=AsyncMock(return_value=[]),
        search_memories=AsyncMock(
            return_value=[HybridResult(9, 0.9, 0.1, None, 0.9, "summary", metadata)]
        ),
    )
    p.initializer.memory_engine = engine
    p.initializer.conversation_manager = ConversationManager(store)
    callbacks = []
    host_event = NS(
        unified_msg_origin=scope["umo"],
        get_platform_id=lambda: "fixture",
        get_platform_name=lambda: "fixture",
        get_self_id=lambda: "bot",
        get_sender_id=lambda: "user-1",
        get_extra=lambda *args: None,
        _cd_tool_memory_sources_register_v1=lambda callback, **kw: (
            callbacks.append(callback) or True
        ),
    )
    if policy == "blocking":
        authority = NS(
            enabled=True,
            resolve_context=lambda **kw: {
                "ready": True,
                "umo": scope["umo"],
                "owner": {"self_id": "bot", "platform_instance_id": "fixture"},
            },
            filter_material=lambda context, bundle, purpose: bundle,
        )
        name = "astrbot_plugin_autonomous_blocking"
    else:
        authority = NS(enabled=False)
        name = "astrbot_plugin_autonomous_blocking"
    p.context.get_all_stars = lambda: [
        NS(name=name, activated=True, star_cls=authority)
    ]
    monkeypatch.setattr(memory_search_tool, "get_active_plugin", lambda: p)
    monkeypatch.setattr(tool_recall_guard, "get_active_plugin", lambda: p)
    monkeypatch.setattr(
        memory_search_tool, "get_persona_id", AsyncMock(return_value="reader")
    )
    monkeypatch.setattr(
        tool_recall_guard, "get_persona_id", AsyncMock(return_value="reader")
    )
    runtime = BoundedRecallTaskPool()
    tool = MemorySearchTool(
        context=p.context,
        config_manager=p.config_manager,
        memory_engine=engine,
        conversation_manager=p.initializer.conversation_manager,
        recall_runtime=runtime,
    )
    try:
        result = json.loads(
            await tool.call(
                NS(context=NS(event=host_event)), query="coffee", include_source=True
            )
        )
        assert result["status"] == "ok", result
        assert result["results"][0]["source_messages"][0]["content"] == row.content
        assert callbacks and await callbacks[0]()
        metadata["status"] = "archived"
        assert not await callbacks[0]()
        model.text_chat.assert_not_awaited()
    finally:
        await runtime.close()
