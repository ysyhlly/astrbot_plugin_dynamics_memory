"""Real archive -> durable evidence -> correction; models are deterministic spies."""

import asyncio
import copy
import hashlib
import json
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from astrbot_plugin_dynamics_memory import main
from astrbot_plugin_dynamics_memory.core import plugin_helpers_mixin
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.blocking_policy import BlockingPolicyGate
from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    normalize_event_material,
    normalize_host_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar

SCOPE = dict(
    platform="fixture",
    umo="fixture:GroupMessage:room",
    persona_id="reader",
    instance_id="reader",
)


@pytest_asyncio.fixture
async def pipeline(tmp_path, monkeypatch):
    from astrbot_plugin_dynamics_memory.core import plugin_helpers_mixin

    monkeypatch.setattr(
        plugin_helpers_mixin, "is_session_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        plugin_helpers_mixin,
        "is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        plugin_helpers_mixin, "get_persona_id", AsyncMock(return_value="reader")
    )
    from astrbot_plugin_dynamics_memory.core.event_handler_modules import memory_recall

    monkeypatch.setattr(
        memory_recall, "get_persona_id", AsyncMock(return_value="reader")
    )
    store = ConversationStore(str(tmp_path / "history.db"))
    await store.initialize()

    async def derive(**kwargs):
        rows = json.loads(kwargs["prompt"])["materials"]
        return NS(
            completion_text=json.dumps(
                {
                    "materials": [
                        {
                            "material_id": r["material_id"],
                            "atoms": [
                                {
                                    "kind": "reported_claim",
                                    "claim": r["text"],
                                    "span_start": 0,
                                    "span_end": len(r["text"]),
                                    "conditions": "",
                                    "polarity": "positive",
                                }
                            ],
                        }
                        for r in rows
                    ]
                }
            )
        )

    provider = NS(
        provider_config={"id": "fixture-only"}, text_chat=AsyncMock(side_effect=derive)
    )
    processor = NS(_get_current_llm_provider=lambda: provider)
    manager = NS(
        store=store,
        invalidate_cache=AsyncMock(),
        get_conversation=AsyncMock(return_value=None),
    )
    p = main.LivingMemoryPlugin.__new__(main.LivingMemoryPlugin)
    p._terminating = False
    p._series_enabled = True
    p.context = NS(get_all_stars=lambda: [], get_config=lambda *args: {})
    p.config_manager = ConfigManager()
    p.initializer = NS(
        is_initialized=True,
        _providers_ready=True,
        memory_processor=processor,
        conversation_manager=manager,
    )
    p.event_handler = None
    p._blocking_policy_gate = BlockingPolicyGate(p.context)
    p.evidence_service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "evidence.db"),
        scope_key,
        p._derive_evidence_materials,
        material_loader=p._load_evidence_material,
        event_loader=p._load_archived_evidence_event,
        scheduler_poll_seconds=0.01,
    )
    await p.evidence_service.initialize()
    try:
        yield p, provider, store
    finally:
        await p.evidence_service.close()
        await store.close()


def event(identifier="one", text="I prefer coffee while reading."):
    return dict(
        schema_version=1,
        type="message.received",
        event_id=identifier,
        scope=dict(SCOPE),
        actor_id="reader-user",
        text=text,
        occurred_at=time.time(),
    )


@pytest.mark.asyncio
async def test_archive_derivation_source_reference_and_correction(pipeline):
    p, provider, store = pipeline
    original = event()
    assert (await p.ingest_events([original]))["accepted"] == 1
    assert (await p.ingest_events([original]))["accepted"] == 0
    raw = {**original, "source_kind": "event", "event_type": original["type"]}
    view = await p.get_evidence_view(SCOPE, source_refs=[raw])
    for _ in range(100):
        if view["atoms"]:
            break
        await asyncio.sleep(0.01)
        view = await p.get_evidence_view(SCOPE, source_refs=[raw])
    assert view["atoms"] and provider.text_chat.await_count == 1
    atom = view["atoms"][0]
    assert atom["source_refs"][0]["author_id"] == "reader-user"
    assert atom["kind"] == "reported_claim" and atom["status"] != "verified"
    descriptor, _ = normalize_event_material(scope_key(SCOPE), SCOPE, raw)
    loaded = await p._load_evidence_material(descriptor)
    assert loaded["text"] == original["text"]
    revision = view["correction_revision"]
    result = await p.correct_insight(
        atom["atom_id"], revision, "dispute", reason="user corrected interpretation"
    )
    assert result["correction_revision"] > revision
    assert (await store.load_narrative_event(SCOPE, original["event_id"]))[
        "text"
    ] == original["text"]
    assert await p._load_evidence_material(descriptor) is None
    with pytest.raises((ValueError, RuntimeError)):
        await p.correct_insight(
            atom["atom_id"], revision, "withdraw", reason="stale update"
        )


@pytest.mark.asyncio
async def test_scope_denial_and_drafts_cannot_be_archived(pipeline, monkeypatch):
    p, provider, store = pipeline
    assert (await p.ingest_events([{**event(), "type": "draft.created"}]))[
        "accepted"
    ] == 0
    monkeypatch.setattr(
        plugin_helpers_mixin, "is_session_enabled", AsyncMock(return_value=False)
    )
    assert (await p.ingest_events([event()]))["accepted"] == 0
    assert not (await p.get_evidence_view(SCOPE, source_refs=[]))["available"]
    provider.text_chat.assert_not_awaited()
    monkeypatch.setattr(
        plugin_helpers_mixin, "is_session_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        plugin_helpers_mixin,
        "is_plugin_enabled_for_session",
        AsyncMock(return_value=False),
    )
    assert not await p._narrative_scope_allowed(SCOPE)
    monkeypatch.setattr(
        plugin_helpers_mixin,
        "is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    p.context.get_config = lambda *args: {
        "platform_settings": {
            "enable_id_white_list": True,
            "id_whitelist": ["different"],
        }
    }
    assert not await p._narrative_scope_allowed(SCOPE)
    p._series_enabled = False
    assert not await p._narrative_scope_allowed(SCOPE)


@pytest.mark.asyncio
async def test_host_history_locator_checks_fingerprint_position_and_shape(pipeline):
    p, _, _ = pipeline
    rows = [{"role": "user", "content": "source statement"}]
    history = json.dumps(rows)
    fingerprint = hashlib.sha256(history.encode()).hexdigest()
    p.initializer.conversation_manager.get_conversation = AsyncMock(
        return_value=NS(history=history)
    )
    raw = dict(
        source_kind="host_history",
        conversation_id="conversation",
        snapshot_fingerprint=fingerprint,
        position=0,
        role="user",
        text=rows[0]["content"],
    )
    descriptor, _ = normalize_host_material(
        scope_key(SCOPE),
        raw,
        conversation_id="conversation",
        snapshot_fingerprint=fingerprint,
    )
    descriptor["source_locator"]["scope"] = dict(SCOPE)
    result = await p._load_evidence_material(descriptor)
    assert result and result["text"] == "source statement"
    for field, value in [
        ("position", -1),
        ("position", True),
        ("conversation_id", ""),
        ("snapshot_fingerprint", "changed"),
    ]:
        changed = copy.deepcopy(descriptor)
        changed["source_locator"][field] = value
        assert await p._load_evidence_material(changed) is None
    for malformed in (
        "not json",
        json.dumps({"not": "a list"}),
        json.dumps([{"role": "user", "content": "source statement", "tool_calls": []}]),
    ):
        changed = copy.deepcopy(descriptor)
        changed["source_locator"]["snapshot_fingerprint"] = hashlib.sha256(
            malformed.encode()
        ).hexdigest()
        p.initializer.conversation_manager.get_conversation.return_value = NS(
            history=malformed
        )
        assert await p._load_evidence_material(changed) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["scope", "revision", "event_type", "missing_event", "kind"]
)
async def test_source_locators_fail_closed_on_tampering(pipeline, mutation):
    p, _, _ = pipeline
    raw = event()
    await p.ingest_events([raw])
    descriptor, _ = normalize_event_material(
        scope_key(SCOPE),
        SCOPE,
        {**raw, "source_kind": "event", "event_type": raw["type"]},
    )
    if mutation == "scope":
        descriptor["source_locator"]["scope"]["umo"] = "fixture:GroupMessage:other"
    elif mutation == "revision":
        descriptor["correction_revision"] = 999
    elif mutation == "event_type":
        descriptor["source_locator"]["event_type"] = "delivery.success"
    elif mutation == "missing_event":
        descriptor["source_locator"]["event_id"] = "absent"
    else:
        descriptor["source_locator"]["kind"] = "unsupported"
    assert await p._load_evidence_material(descriptor) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["unready", "missing", "decision_model", "bad_json", "empty", "transport"],
)
async def test_derivation_does_not_fallback_or_publish_unverified_output(
    pipeline, failure
):
    p, provider, _ = pipeline
    raw = event()
    _, material = normalize_event_material(
        scope_key(SCOPE),
        SCOPE,
        {**raw, "source_kind": "event", "event_type": raw["type"]},
    )
    if failure == "unready":
        p.initializer._providers_ready = False
    if failure == "missing":
        p.initializer.memory_processor = None
    if failure == "decision_model":
        provider.is_systemone_provider = True
    if failure == "bad_json":
        provider.text_chat.side_effect = None
        provider.text_chat.return_value = NS(completion_text="not json")
    if failure == "empty":
        provider.text_chat.side_effect = None
        provider.text_chat.return_value = NS(completion_text="")
    if failure == "transport":
        provider.text_chat.side_effect = ConnectionError("fixture connection failed")
    expected = ConnectionError if failure == "transport" else RuntimeError
    with pytest.raises(expected):
        await p._derive_evidence_materials([material])
    if failure in {"unready", "missing", "decision_model"}:
        provider.text_chat.assert_not_awaited()
    else:
        assert provider.text_chat.await_count == 1


@pytest.mark.asyncio
async def test_derivation_discards_result_after_provider_replacement(pipeline):
    p, provider, _ = pipeline
    original = provider.text_chat.side_effect

    async def replaced(**kwargs):
        result = await original(**kwargs)
        p.initializer.memory_processor._get_current_llm_provider = lambda: NS()
        return result

    provider.text_chat.side_effect = replaced
    raw = event()
    _, material = normalize_event_material(
        scope_key(SCOPE),
        SCOPE,
        {**raw, "source_kind": "event", "event_type": raw["type"]},
    )
    with pytest.raises(RuntimeError, match="summarizer_changed"):
        await p._derive_evidence_materials([material])


@pytest.mark.asyncio
async def test_native_projector_is_current_registered_owner_and_receipt_is_bound(
    pipeline,
):
    p, _, _ = pipeline
    extras = {}
    event = NS(
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.update({key: value}),
    )
    req = NS()
    receipt = {
        "schema_version": 1,
        "provider": "astrbot_plugin_chat_dynamics",
        "scope": dict(SCOPE),
        "card_version": "card",
    }
    owner = NS(prepare_memory_request=AsyncMock(return_value=receipt))
    entry = NS(name="astrbot_plugin_chat_dynamics", star_cls=owner, activated=True)
    p.context.get_all_stars = lambda: [entry]
    assert await p._prepare_native_memory_projection(event, req) == receipt
    assert extras["_cd_memory_feedback_v1"] == receipt
    assert await p._prepare_native_memory_projection(event, req) == receipt
    assert owner.prepare_memory_request.await_count == 1
    extras.clear()
    entry.activated = False
    assert await p._prepare_native_memory_projection(event, NS()) is None
    entry.activated = True
    owner.prepare_memory_request.side_effect = RuntimeError("reloaded")
    assert await p._prepare_native_memory_projection(event, NS()) is None


@pytest.mark.asyncio
async def test_native_filter_rejects_forged_frames_and_wrong_owner(pipeline):
    p, _, _ = pipeline
    trusted = {"schema_version": 1, "projection_restricted": True}
    event = NS(
        get_extra=lambda key, default=None: (
            trusted if key == "_autonomous_blocking_context_v1" else default
        ),
        get_platform_id=lambda: "fixture",
        get_self_id=lambda: "bot",
    )
    frames = [
        {"atom_id": "one", "text": "real source", "author_uid": "user"},
        {"atom_id": "two", "text": "blocked source", "author_uid": "other"},
    ]
    service = NS(
        enabled=True,
        resolve_context=lambda **kw: {
            "ready": True,
            "umo": SCOPE["umo"],
            "owner": {"self_id": "bot", "platform_instance_id": "fixture"},
        },
        filter_material=lambda *a: {
            "quotes": [
                frames[0],
                {"atom_id": "two", "text": "forged", "author_uid": "other"},
            ]
        },
    )
    p.context.get_all_stars = lambda: [
        NS(name="astrbot_plugin_autonomous_blocking", star_cls=service, activated=True)
    ]
    assert await p._filter_native_frames(event, SCOPE, frames) == [frames[0]]
    service.resolve_context = lambda **kw: {
        "ready": True,
        "umo": SCOPE["umo"],
        "owner": {"self_id": "different", "platform_instance_id": "fixture"},
    }
    assert await p._filter_native_frames(event, SCOPE, frames) is False
    service.enabled = False
    assert await p._filter_native_frames(event, SCOPE, frames) == frames


@pytest.mark.asyncio
async def test_private_summary_collects_exact_source_without_second_model_call(
    pipeline,
):
    from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
    from astrbot_plugin_dynamics_memory.core.memory_source import (
        build_source_row_refs_v1,
    )
    from astrbot_plugin_dynamics_memory.core.native_recall import (
        NATIVE_MEMORY_REQUEST_KEY,
    )

    p, provider, store = pipeline
    scope = {**SCOPE, "umo": "fixture:FriendMessage:user-1"}
    row = Message(
        id=0,
        session_id=scope["umo"],
        role="user",
        content="I prefer coffee.",
        sender_id="user-1",
        sender_name="Reader",
        platform="fixture",
        timestamp=1234,
        metadata={},
    )
    row.id = await store.add_message(row)
    metadata = {
        "status": "active",
        "session_id": scope["umo"],
        "source_session_id": scope["umo"],
        "persona_id": "reader",
        "source_platform": "fixture",
        "interaction_type": "private_chat",
        "memory_visibility": "private",
        "source_row_refs_v1": build_source_row_refs_v1([row], scope["umo"]),
    }
    p.initializer.memory_engine = NS(
        get_memory=AsyncMock(
            side_effect=lambda mid: (
                {"text": "untrusted summary", "metadata": copy.deepcopy(metadata)}
                if mid == 9
                else None
            )
        ),
        get_memory_source=AsyncMock(return_value=[]),
    )
    p.evidence_service.conversation_row_loader = p._load_conversation_row_evidence
    receipt = {
        "schema_version": 1,
        "provider": "astrbot_plugin_chat_dynamics",
        "status": "prepared",
        "purpose": "reply",
        "scope": scope,
        "card_version": "card",
    }
    event = NS(
        unified_msg_origin=scope["umo"],
        get_sender_id=lambda: "user-1",
        get_platform_name=lambda: "fixture",
        get_platform_id=lambda: "fixture",
        get_self_id=lambda: "bot",
        get_extra=lambda *a: None,
    )
    req = NS(_cd_memory_feedback_v1=receipt)
    blocker = NS(
        enabled=True,
        resolve_context=lambda **kw: {
            "ready": True,
            "umo": scope["umo"],
            "owner": {"self_id": "bot", "platform_instance_id": "fixture"},
        },
        filter_material=lambda context, bundle, purpose: bundle,
    )
    p.context.get_all_stars = lambda: [
        NS(name="astrbot_plugin_autonomous_blocking", star_cls=blocker, activated=True)
    ]
    await p._native_collect_summary_views(
        event, req, [{"id": 9, "content": "untrusted summary"}], persona_id="reader"
    )
    request = getattr(req, NATIVE_MEMORY_REQUEST_KEY)
    assert len(request["source_views"]) == 1
    atoms = request["source_views"][0]["evidence_view"]["atoms"]
    assert (
        atoms
        and atoms[0]["claim"] == row.content
        and atoms[0]["subject_ids"] == ["user-1"]
    )
    provider.text_chat.assert_not_awaited()
    await p._native_collect_summary_views(event, req, [{"id": 9}], persona_id="reader")
    assert len(request["source_views"]) == 1
    metadata["status"] = "archived"
    new_req = NS(_cd_memory_feedback_v1=receipt)
    await p._native_collect_summary_views(
        event, new_req, [{"id": 9}], persona_id="reader"
    )
    assert not getattr(new_req, NATIVE_MEMORY_REQUEST_KEY)["source_views"]


@pytest.mark.asyncio
async def test_legacy_summary_projection_never_becomes_verified_fact(pipeline):
    p, _, _ = pipeline
    scope = {**SCOPE, "umo": "fixture:FriendMessage:user"}
    receipt = {
        "schema_version": 1,
        "provider": "astrbot_plugin_chat_dynamics",
        "status": "prepared",
        "purpose": "reply",
        "scope": scope,
    }
    event = NS(get_extra=lambda *a: None)
    req = NS(_cd_memory_feedback_v1=receipt)
    p.get_evidence_view = AsyncMock(
        return_value={
            "atoms": [
                {
                    "atom_id": "legacy",
                    "source_status": "legacy_summary",
                    "status": "reported_unverified",
                    "claim": "old claim",
                    "confidence": 0.9,
                },
                {
                    "atom_id": "wrong",
                    "source_status": "exact_event",
                    "status": "verified",
                    "claim": "do not flatten this",
                },
            ]
        }
    )
    result = await p._native_memory_evidence_filter(
        event, req, [{"id": 9, "content": "summary"}], "query", "reader"
    )
    assert len(result) == 1 and result[0]["score"] == 0.35
    assert "未核实" in result[0]["content"]
    req._cd_memory_feedback_v1 = {**receipt, "purpose": "invalid"}
    assert (
        await p._native_memory_evidence_filter(event, req, [], "query", "reader")
        is None
    )


def test_injection_report_is_bound_to_exact_request_and_contains_no_source_text():
    from astrbot_plugin_dynamics_memory.core.recall_runtime import (
        remember_private_segment,
    )

    p = main.LivingMemoryPlugin.__new__(main.LivingMemoryPlugin)
    event = NS()
    part = NS(text="private source body")
    req = NS(extra_user_content_parts=[part], _lm_private_hook_outcome_v1="ready")
    remember_private_segment(req, part, source_level="verified_quote", item_count=1)
    p._install_private_injection_getter(event, req)
    getter = req._lm_private_injection_v1
    assert getter(NS(), event) is None and getter(req, NS()) is None
    value = getter(req, event)
    assert (
        value["segments"][0]["sha256"] == hashlib.sha256(part.text.encode()).hexdigest()
    )
    assert part.text not in repr(value)
    req._lm_private_hook_outcome_v1 = "forged outcome"
    assert getter(req, event)["hook_outcome"] == "error"


@pytest.mark.asyncio
async def test_native_hook_uses_authentic_input_and_revalidates_quote_before_delivery(
    pipeline,
):
    from astrbot.api.platform import MessageType
    from astrbot.core.message.components import Plain
    from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
    from astrbot_plugin_dynamics_memory.core.memory_source import (
        build_source_row_refs_v1,
    )
    from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
        ConversationManager,
    )
    from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler
    from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
        HybridResult,
    )
    from astrbot_plugin_dynamics_memory.core.recall_runtime import BoundedRecallTaskPool
    from astrbot_plugin_dynamics_memory.core.native_recall import (
        NATIVE_MEMORY_VALIDATE_KEY,
    )

    p, provider, store = pipeline
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
                "text": "unverified summary",
                "metadata": copy.deepcopy(metadata),
            }
        ),
        get_memory_source=AsyncMock(return_value=[]),
        search_narrative_history=AsyncMock(return_value=[]),
        search_narrative_semantic=AsyncMock(return_value=[]),
    )
    engine.search_memories = AsyncMock(
        return_value=[
            HybridResult(9, 0.9, 0.1, None, 0.9, "unverified summary", metadata)
        ]
    )
    manager = ConversationManager(store)
    p.initializer.memory_engine = engine
    p.initializer.conversation_manager = manager
    p.evidence_service.conversation_row_loader = p._load_conversation_row_evidence
    p._recall_runtime = BoundedRecallTaskPool()
    receipt = {
        "schema_version": 1,
        "provider": "astrbot_plugin_chat_dynamics",
        "status": "prepared",
        "purpose": "reply",
        "scope": scope,
        "card_version": "card",
    }
    extra = {
        "_cd_memory_feedback_v1": receipt,
        "_autonomous_blocking_context_v1": {
            "schema_version": 1,
            "projection_restricted": True,
        },
    }
    event = NS(
        unified_msg_origin=scope["umo"],
        message_obj=NS(message_id="query-one"),
        get_messages=lambda: [Plain("coffee?")],
        get_message_str=lambda: "coffee?",
        get_message_type=lambda: MessageType.FRIEND_MESSAGE,
        get_sender_id=lambda: "user-1",
        get_sender_name=lambda: "Reader",
        get_platform_name=lambda: "fixture",
        get_platform_id=lambda: "fixture",
        get_self_id=lambda: "bot",
        get_extra=lambda key, default=None: extra.get(key, default),
        set_extra=lambda key, value: extra.update({key: value}),
    )
    blocker = NS(
        enabled=True,
        resolve_context=lambda **kw: {
            "ready": True,
            "umo": scope["umo"],
            "scope": scope,
            "owner": {
                "platform": "fixture",
                "platform_instance_id": "fixture",
                "self_id": "bot",
            },
        },
        filter_material=lambda context, bundle, purpose: bundle,
    )
    p.context.get_all_stars = lambda: [
        NS(name="astrbot_plugin_autonomous_blocking", activated=True, star_cls=blocker)
    ]
    p.context.get_platform_inst = lambda name: NS(
        meta=lambda: NS(id="fixture", name="fixture"),
        status=NS(value="running"),
        get_self_id=lambda: "bot",
    )
    p.event_handler = EventHandler(
        p.context,
        p.config_manager,
        engine,
        None,
        manager,
        evidence_filter=p._native_memory_evidence_filter,
        recall_runtime=p._recall_runtime,
        summary_guard=p._install_private_summary_guard,
    )
    req = NS(
        prompt="rewritten planner wrapper",
        system_prompt="",
        contexts=[{"role": "user", "content": row.content}],
        extra_user_content_parts=[],
    )
    try:
        await p.handle_memory_recall(event, req)
        assert engine.search_memories.await_args.kwargs["query"] == "coffee?"
        text = "\n".join(part.text for part in req.extra_user_content_parts)
        assert row.content in text and "unverified summary" not in text
        validator = getattr(req, NATIVE_MEMORY_VALIDATE_KEY)
        assert await validator()
        event.get_self_id = lambda: "different-bot"
        assert not await validator()
        event.get_self_id = lambda: "bot"
        assert await validator()
        metadata["status"] = "archived"
        assert not await validator()
        provider.text_chat.assert_not_awaited()
    finally:
        await p.event_handler.shutdown()
        await p._recall_runtime.close()
