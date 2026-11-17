"""Unit tests covering PluginHelpersMixin edge paths and helpers."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core import plugin_helpers_mixin
from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    conversation_row_hash,
)
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.core.plugin_helpers_mixin import (
    PluginHelpersMixin,
    _SCOPE_RECALL_TIMINGS,
)


def _scope(umo="qq:FriendMessage:u1"):
    return {
        "platform": "qq",
        "umo": umo,
        "persona_id": "p1",
        "instance_id": "p1",
    }


def _make_helper():
    helper = PluginHelpersMixin()
    helper.context = SimpleNamespace(
        get_all_stars=lambda: {},
        get_config=lambda _umo=None: {},
    )
    helper.config_manager = SimpleNamespace(
        get=lambda key, default=None: default,
        filtering_settings={"use_persona_filtering": True},
    )
    helper.initializer = SimpleNamespace(
        is_initialized=True,
        _providers_ready=True,
        memory_engine=None,
        conversation_manager=None,
        memory_processor=None,
    )
    helper._blocking_policy_gate = SimpleNamespace(
        observe_trusted_event=AsyncMock(return_value=True),
        check=AsyncMock(return_value=True),
    )
    helper.series_runtime_allowed = lambda: True
    return helper


def test_physical_original_input_status_paths(monkeypatch):
    ev_not_owned = SimpleNamespace()
    assert (
        PluginHelpersMixin._physical_original_input_status(ev_not_owned) == "not_owned"
    )

    ev_bad_owned = SimpleNamespace(_cd_owned_original_input_v1="bad")
    assert (
        PluginHelpersMixin._physical_original_input_status(ev_bad_owned)
        == "unavailable"
    )

    ev_legacy_only = SimpleNamespace(_chat_dynamics_owned_request=True)
    assert (
        PluginHelpersMixin._physical_original_input_status(ev_legacy_only)
        == "unavailable"
    )

    ev_no_resolver = SimpleNamespace(_cd_owned_original_input_v1=True)
    assert (
        PluginHelpersMixin._physical_original_input_status(ev_no_resolver)
        == "unavailable"
    )

    async def awaitable_resolver(_ev):
        return None

    ev_await = SimpleNamespace(
        _cd_owned_original_input_v1=True,
        _cd_original_input_resolver_v1=awaitable_resolver,
    )
    assert PluginHelpersMixin._physical_original_input_status(ev_await) == "unavailable"

    def raising_resolver(_ev):
        raise RuntimeError("boom")

    ev_err = SimpleNamespace(
        _cd_owned_original_input_v1=True,
        _cd_original_input_resolver_v1=raising_resolver,
    )
    assert PluginHelpersMixin._physical_original_input_status(ev_err) == "unavailable"

    monkeypatch.setattr(
        plugin_helpers_mixin,
        "validate_owned_original_input",
        lambda dto, **_kw: dto,
    )
    ev_valid = SimpleNamespace(
        _cd_owned_original_input_v1=True,
        _cd_original_input_resolver_v1=lambda _ev: {"text": "hi"},
        unified_msg_origin=" qq:FriendMessage:u1 ",
        get_platform_name=lambda: "QQ",
        get_self_id=lambda: " bot ",
        get_sender_id=lambda: " u1 ",
        message_obj=SimpleNamespace(message_id=42),
    )
    assert PluginHelpersMixin._physical_original_input_status(ev_valid) == "valid"


def test_worldbook_receipt_and_native_helper_stateless_methods():
    scope = _scope()
    assert PluginHelpersMixin._strict_worldbook_scope("bad") is None
    assert PluginHelpersMixin._strict_worldbook_scope({"platform": ""}) is None
    assert PluginHelpersMixin._strict_worldbook_scope(scope) == scope

    receipt = {
        "schema_version": 1,
        "provider": "astrbot_plugin_chat_dynamics",
        "status": "skipped",
        "reason": "cd_owned_request",
        "scope": scope,
        "purpose": "reply",
        "card_version": "c1",
    }
    assert PluginHelpersMixin._is_cd_owned_worldbook_receipt(receipt) is True
    assert (
        PluginHelpersMixin._is_cd_owned_worldbook_receipt(
            {**receipt, "schema_version": True}
        )
        is False
    )
    assert PluginHelpersMixin._cd_owned_receipt_contract_identity("bad") is None
    assert (
        PluginHelpersMixin._fresh_worldbook_receipt_matches(receipt, scope, "c1")
        is True
    )
    assert (
        PluginHelpersMixin._fresh_worldbook_receipt_matches(receipt, scope, "c2")
        is False
    )

    req = SimpleNamespace(
        _lm_native_memory_v1={"x": 1},
        _lm_native_memory_validate_v1=lambda: True,
    )
    PluginHelpersMixin._native_clear_request(req)
    assert not hasattr(req, "_lm_native_memory_v1")
    assert PluginHelpersMixin._native_receipt_identity("bad") is None
    assert PluginHelpersMixin._native_receipt_identity(receipt)["card_version"] == "c1"

    assert PluginHelpersMixin._native_projection_restricted(SimpleNamespace()) is False
    ev_restricted = SimpleNamespace(
        get_extra=lambda k, d=None: {"schema_version": 1, "projection_restricted": True}
    )
    assert PluginHelpersMixin._native_projection_restricted(ev_restricted) is True

    assert PluginHelpersMixin._native_scope_from_request({}, "") is None
    assert (
        PluginHelpersMixin._native_scope_from_request(scope, "qq:FriendMessage:u2")[
            "umo"
        ]
        == "qq:FriendMessage:u2"
    )

    assert PluginHelpersMixin._native_finite_timestamp("bad") is False
    assert PluginHelpersMixin._native_finite_timestamp(float("inf")) is False
    assert PluginHelpersMixin._native_finite_timestamp(float("nan")) is False
    assert PluginHelpersMixin._native_finite_timestamp(123.5) is True


def test_native_prune_merge_and_effective_rerank_status():
    scope = _scope()
    source_view = {
        "scope": scope,
        "purpose": "reply",
        "card_version": "v1",
        "evidence_view": {
            "correction_revision": 0,
            "atoms": [
                {
                    "atom_id": "a1",
                    "source_refs": [{"material_id": "m1", "source_event_id": "e1"}],
                },
                {"atom_id": "a2", "source_refs": []},
            ],
            "slots": {
                "recent": [{"atom_id": "a1"}, {"atom_id": "a2"}],
                "bad": "not-list",
            },
        },
        "native_materials": [
            {"material_id": "m1", "source_locator": {"kind": "conversation_row"}},
            {"material_id": "m2", "source_locator": {"kind": "conversation_row"}},
        ],
        "native_events": [
            {"event_id": "e1", "atom_id": "a1", "retrieval_hit": True},
            {"event_id": "e2", "atom_id": "a2"},
        ],
    }
    pruned = PluginHelpersMixin._native_prune_source_views(
        ["bad", {"evidence_view": "bad"}, source_view], {"a1"}
    )
    assert len(pruned) == 1
    assert [a["atom_id"] for a in pruned[0]["evidence_view"]["atoms"]] == ["a1"]
    assert len(pruned[0]["native_materials"]) == 1
    assert len(pruned[0]["native_events"]) == 1
    assert PluginHelpersMixin._native_prune_source_views([source_view], {"none"}) == []

    views = []
    assert PluginHelpersMixin._native_merge_source_view("bad", source_view) is False
    assert (
        PluginHelpersMixin._native_merge_source_view(views, {"scope": "bad"}) is False
    )
    assert PluginHelpersMixin._native_merge_source_view(views, source_view) is True
    dup = {
        "scope": scope,
        "purpose": "reply",
        "card_version": "v1",
        "evidence_view": {
            "correction_revision": 0,
            "atoms": [{"atom_id": "a3"}],
            "slots": {"recent": [{"atom_id": "a3"}]},
        },
        "native_materials": [{"material_id": "m1"}],
        "native_events": [{"event_id": "e1", "atom_id": "a1", "retrieval_hit": False}],
    }
    assert PluginHelpersMixin._native_merge_source_view(views, dup) is True
    assert len(views[0]["evidence_view"]["atoms"]) == 3
    mismatch = {**dup, "purpose": "other"}
    assert PluginHelpersMixin._native_merge_source_view(views, mismatch) is False

    helper = _make_helper()
    assert helper._effective_rerank_status()["effective"] == "disabled"
    helper.initializer.memory_engine = SimpleNamespace(
        config={"rerank_enabled": True, "rerank_provider_id": ""},
        rerank_provider_resolver=None,
    )
    assert helper._effective_rerank_status()["effective"] == "skipped_unconfigured"
    helper.initializer.memory_engine.config["rerank_provider_id"] = "r1"
    assert helper._effective_rerank_status()["effective"] == "resolver_unavailable"
    helper.initializer.memory_engine.rerank_provider_resolver = lambda: None
    assert helper._effective_rerank_status()["effective"] == "configured_not_probed"


def test_native_row_matching_and_install_private_injection_getter():
    helper = _make_helper()
    ev = SimpleNamespace()
    req = SimpleNamespace(
        extra_user_content_parts=[],
        _lm_private_segment_refs_v1={},
        _lm_private_hook_outcome_v1="ready",
    )
    helper._install_private_injection_getter(ev, req)
    getter = req._lm_private_injection_v1
    assert getter(SimpleNamespace(), ev) is None
    part = SimpleNamespace(text="mem")
    req.extra_user_content_parts.append(part)
    req._lm_private_segment_refs_v1[id(part)] = {
        "source_level": "legacy_summary",
        "item_count": 1,
    }
    res = getter(req, ev)
    assert res["hook_outcome"] == "ready"
    assert len(res["segments"]) == 1 and res["segments"][0]["char_count"] == 3

    row = {
        "id": 1,
        "session_id": "qq:FriendMessage:u1",
        "role": "user",
        "content": "hello",
        "sender_id": "u1",
        "platform": "qq",
        "timestamp": 100.0,
    }
    assert PluginHelpersMixin._native_row_matches_snapshot(row, dict(row)) is True
    assert (
        PluginHelpersMixin._native_row_matches_snapshot(
            row, {**row, "timestamp": "bad"}
        )
        is False
    )
    assert (
        PluginHelpersMixin._native_row_matches_snapshot(
            row, {**row, "content": "other"}
        )
        is False
    )
    assert PluginHelpersMixin._native_row_matches_snapshot(row, {"id": 1}) is False

    thin_ref = {
        "id": 1,
        "session_id": "qq:FriendMessage:u1",
        "role": "user",
        "sender_id": "u1",
        "platform": "qq",
        "timestamp": 100.0,
        "content_sha256": hashlib.sha256(b"hello").hexdigest(),
        "row_hash": conversation_row_hash(row),
    }
    assert PluginHelpersMixin._native_row_matches_thin_ref(row, thin_ref) is True
    assert (
        PluginHelpersMixin._native_row_matches_thin_ref(
            {**row, "content": 123}, thin_ref
        )
        is False
    )
    assert (
        PluginHelpersMixin._native_row_matches_thin_ref(
            {**row, "timestamp": "bad"}, thin_ref
        )
        is False
    )
    assert (
        PluginHelpersMixin._native_row_matches_thin_ref(
            {**row, "sender_id": "u2"}, thin_ref
        )
        is False
    )


@pytest.mark.asyncio
async def test_fresh_cd_owned_worldbook_receipt_and_scope_checks(monkeypatch):
    helper = _make_helper()
    scope = _scope()
    req = SimpleNamespace(_cd_memory_feedback_v1={"schema_version": 1})
    ev = SimpleNamespace(
        unified_msg_origin="qq:FriendMessage:u1",
        get_platform_id=lambda: "qq",
        get_extra=lambda k, d=None: None,
    )
    assert await helper._fresh_cd_owned_worldbook_receipt(ev, req) is None

    async def bad_stars():
        raise RuntimeError("stars fail")

    helper.context.get_all_stars = bad_stars
    assert await helper._fresh_cd_owned_worldbook_receipt(ev, req) is None

    async def prepare_sync(*_a, **_kw):
        return {"schema_version": 1}

    helper.context.get_all_stars = lambda: {
        "other": {"activated": False},
        "astrbot_plugin_chat_dynamics": {
            "activated": True,
            "name": "astrbot_plugin_chat_dynamics",
            "prepare_memory_request": lambda *_a, **_kw: "not-awaitable",
        },
    }
    assert await helper._fresh_cd_owned_worldbook_receipt(ev, req) is None

    def mutating_prepare(_event, candidate_req, **_kw):
        async def _inner():
            candidate_req._cd_memory_feedback_v1 = {"schema_version": 99}
            return {"schema_version": 1}

        return _inner()

    helper.context.get_all_stars = lambda: {
        "astrbot_plugin_chat_dynamics": {
            "activated": True,
            "name": "astrbot_plugin_chat_dynamics",
            "prepare_memory_request": mutating_prepare,
        }
    }
    assert await helper._fresh_cd_owned_worldbook_receipt(ev, req) is None

    helper.context.get_all_stars = lambda: {
        "astrbot_plugin_chat_dynamics": {
            "activated": True,
            "name": "astrbot_plugin_chat_dynamics",
            "star_cls": SimpleNamespace(prepare_memory_request=prepare_sync),
        }
    }
    req._cd_memory_feedback_v1 = {"schema_version": 1}
    assert await helper._fresh_cd_owned_worldbook_receipt(ev, req) == {
        "schema_version": 1
    }

    # _remember_blocking_owner & _narrative_scope_allowed
    monkeypatch.setattr(
        plugin_helpers_mixin, "get_persona_id", AsyncMock(return_value="p1")
    )
    assert await helper._remember_blocking_owner(ev, None) is True
    assert (
        await helper._remember_blocking_owner(
            SimpleNamespace(unified_msg_origin="bad"), None
        )
        is None
    )

    monkeypatch.setattr(
        plugin_helpers_mixin, "is_event_memory_allowed", lambda *_a: True
    )
    monkeypatch.setattr(
        plugin_helpers_mixin, "is_session_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        plugin_helpers_mixin,
        "is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        plugin_helpers_mixin.PassiveGroupCaptureFilter,
        "_passes_global_whitelist",
        staticmethod(lambda *_a: True),
    )
    token = _SCOPE_RECALL_TIMINGS.set({})
    try:
        assert await helper._narrative_scope_allowed(scope) is True
        assert "scope_memory_policy" in _SCOPE_RECALL_TIMINGS.get()
    finally:
        _SCOPE_RECALL_TIMINGS.reset(token)

    helper.series_runtime_allowed = lambda: False
    assert await helper._narrative_scope_allowed(scope) is False


@pytest.mark.asyncio
async def test_filter_native_frames_and_row_policy_allowed(monkeypatch):
    helper = _make_helper()
    scope = _scope()
    frames = [{"atom_id": "a1", "text": "hello", "author_uid": "u1"}]
    ev = SimpleNamespace(
        get_platform_id=lambda: "qq",
        get_self_id=lambda: "bot",
    )

    assert await helper._filter_native_frames(ev, scope, frames) is False

    # Disabled autonomous_blocking passes frames through
    helper.context.get_all_stars = lambda: {
        "astrbot_plugin_autonomous_blocking": {
            "activated": True,
            "name": "astrbot_plugin_autonomous_blocking",
            "star_cls": SimpleNamespace(enabled=False),
        }
    }
    assert await helper._filter_native_frames(ev, scope, frames) == frames

    # Active autonomous_blocking filters frames safely
    async def resolve_ctx(**_kw):
        return {
            "ready": True,
            "umo": scope["umo"],
            "owner": {"platform_instance_id": "qq", "self_id": "bot"},
        }

    async def filter_mat(_ctx, bundle, _purpose):
        return {"quotes": bundle["quotes"]}

    helper.context.get_all_stars = lambda: {
        "astrbot_plugin_autonomous_blocking": {
            "activated": True,
            "name": "astrbot_plugin_autonomous_blocking",
            "star_cls": SimpleNamespace(
                enabled=True, resolve_context=resolve_ctx, filter_material=filter_mat
            ),
        }
    }
    assert await helper._filter_native_frames(ev, scope, frames) == frames

    # Mismatched owner rejects
    ev_bad = SimpleNamespace(get_platform_id=lambda: "other", get_self_id=lambda: "bot")
    assert await helper._filter_native_frames(ev_bad, scope, frames) is False

    # _native_row_policy_allowed & _evidence_material_policy
    helper.series_runtime_allowed = lambda: True
    helper.config_manager = SimpleNamespace(
        get=lambda k, d=None: d,
        filtering_settings={"use_persona_filtering": True},
    )
    helper._narrative_scope_allowed = AsyncMock(return_value=True)
    monkeypatch.setattr(
        plugin_helpers_mixin, "get_persona_id", AsyncMock(return_value="p1")
    )
    helper._native_read_scopes = AsyncMock(return_value=[(scope, False, None)])
    desc_row = {
        "material_id": "cr1",
        "source_locator": {"kind": "conversation_row", "scope": scope},
    }
    mat_row = {
        "source_status": "conversation_row",
        "role": "user",
        "author_id": "u1",
        "row_hash": "rh",
        "text": "hello",
        "subject_ids": ["u2"],
    }
    assert await helper._native_row_policy_allowed(ev, scope, desc_row, mat_row) is True
    desc_ev = {
        "material_kind": "source_event",
        "scope_key": scope_key(scope),
        "source_locator": {"scope": scope},
    }
    assert await helper._evidence_material_policy(desc_ev, {"actor_id": "u1"}) is True
    assert (
        await helper._native_row_policy_allowed(ev, "bad", desc_row, mat_row) is False
    )


@pytest.mark.asyncio
async def test_derive_evidence_materials_success_and_error_paths():
    import json

    helper = _make_helper()
    helper.initializer.is_initialized = False
    with pytest.raises(RuntimeError, match="summarizer_unavailable"):
        await helper._derive_evidence_materials([{"material_id": "m1"}])

    helper.initializer.is_initialized = True
    helper.initializer.memory_processor = SimpleNamespace(
        _get_current_llm_provider=lambda: None
    )
    with pytest.raises(RuntimeError, match="summarizer_changed"):
        await helper._derive_evidence_materials([{"material_id": "m1"}])

    class _Provider:
        provider_config = {"id": "p-bg"}

        async def text_chat(self, *, prompt, system_prompt, request_max_retries=1):
            return SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "materials": [
                            {
                                "material_id": "m1",
                                "atoms": [
                                    {
                                        "kind": "reported_claim",
                                        "claim": "likes tea",
                                        "span_start": 0,
                                        "span_end": 9,
                                        "conditions": "",
                                        "polarity": "positive",
                                    }
                                ],
                            }
                        ]
                    }
                )
            )

    provider = _Provider()
    helper.initializer.memory_processor = SimpleNamespace(
        _get_current_llm_provider=lambda: provider
    )
    mat = {
        "material_id": "m1",
        "source_status": "host_history",
        "role": "user",
        "text": "likes tea",
    }
    res = await helper._derive_evidence_materials([mat])
    assert "m1" in res and len(res["m1"]) == 1

    # Timeout and schema error paths
    provider.text_chat = AsyncMock(side_effect=TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        await helper._derive_evidence_materials([mat])

    provider.text_chat = AsyncMock(
        return_value=SimpleNamespace(completion_text="not json")
    )
    with pytest.raises(RuntimeError, match="invalid_derivation_schema"):
        await helper._derive_evidence_materials([mat])

    # Provider changed mid-flight -> raises summarizer_changed
    provider.text_chat = AsyncMock(
        return_value=SimpleNamespace(
            completion_text=json.dumps(
                {"materials": [{"material_id": "m1", "atoms": []}]}
            )
        )
    )
    calls = 0

    def switch_provider():
        nonlocal calls
        calls += 1
        return provider if calls == 1 else _Provider()

    helper.initializer.memory_processor = SimpleNamespace(
        _get_current_llm_provider=switch_provider
    )
    with pytest.raises(RuntimeError, match="summarizer_changed"):
        await helper._derive_evidence_materials([mat])
