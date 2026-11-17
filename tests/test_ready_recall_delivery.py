"""Verified local recall survives enrichment failure without stale-scope reuse."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import threading
import time

import pytest

from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar
from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    normalize_event_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService

SCOPE = {
    "platform": "qq",
    "umo": "qq:GroupMessage:1",
    "persona_id": "anan",
    "instance_id": "one",
}


def ready_view():
    atom = {
        "atom_id": "atm1:fixture",
        "claim": "用户说过希望继续讨论那个问题",
        "kind": "utterance_fact",
        "status": "verified_utterance",
        "source_status": "exact_event",
        "scope": {"scope_key": scope_key(SCOPE)},
        "source_refs": [
            {
                "material_id": "ev1:fixture",
                "source_event_id": "message:old",
                "source_status": "exact_event",
                "author_kind": "user",
                "author_id": "u1",
            }
        ],
        "root_evidence_ids": ["ev1:fixture"],
        "root_material_ids": [],
    }
    return {
        "schema_version": 1,
        "available": True,
        "scope_key": scope_key(SCOPE),
        "correction_revision": 0,
        "revocations": [],
        "purpose": "reply",
        "card_version": "card",
        "atoms": [atom],
        "slots": {
            "current_evidence": [],
            "commitments_and_boundaries": [],
            "recent": [atom],
            "historical_claims": [],
        },
        "history_projection": {},
        "evidence_status": "ready",
        "diagnostics": {},
    }


def plugin_with_ready_baseline(*, mixed=True):
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin._terminating = False
    values = {
        "recall_engine.narrative_total_timeout_seconds": 0.4,
        "recall_engine.narrative_local_timeout_seconds": 0.05,
        "recall_engine.narrative_hybrid_timeout_seconds": 0.05,
    }
    plugin.config_manager = SimpleNamespace(
        get=lambda key, default=None: values.get(key, default)
    )
    event = {
        "source_id": "newer",
        "text": "用户发言：补充条件",
        "scope": SCOPE,
        "fact_kind": "observation",
    }
    plugin.initializer = SimpleNamespace(
        is_initialized=True,
        memory_engine=SimpleNamespace(
            search_narrative_history=AsyncMock(return_value=[]),
            search_narrative_semantic=AsyncMock(return_value=[]),
        ),
        conversation_manager=SimpleNamespace(
            store=SimpleNamespace(
                recall_narrative_events=AsyncMock(return_value=[event] if mixed else [])
            )
        ),
    )
    plugin._narrative_scope_allowed = AsyncMock(return_value=True)
    plugin.peek_evidence_revision = lambda scope: {
        "available": True,
        "scope_key": scope_key(scope),
        "correction_revision": 0,
        "revocations": [],
    }
    view = ready_view()
    plugin.evidence_service = SimpleNamespace()
    plugin._ready_narrative_baseline = AsyncMock(
        return_value={
            "source_refs": [{"source_kind": "event", "source_event_id": "message:old"}],
            "typed_items": [
                {"item_kind": "evidence_atom", "source_id": "atm1:fixture"}
            ],
            "diagnostic": "ready",
            "evidence_view": view,
        }
    )
    plugin._revalidate_narrative_candidates = AsyncMock(return_value=[])
    plugin._effective_rerank_status = lambda: {}

    async def stalled(*args, **kwargs):
        await asyncio.sleep(2)
        return ready_view()

    plugin.get_evidence_view = AsyncMock(side_effect=stalled)
    return plugin, view


async def close_pool(plugin):
    runtime = getattr(plugin, "_recall_runtime", None)
    if runtime:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "error"])
async def test_final_enrichment_timeout_preserves_verified_ready_memory(failure):
    plugin, original = plugin_with_ready_baseline()
    if failure == "error":
        plugin.get_evidence_view = AsyncMock(
            side_effect=RuntimeError("controlled enrichment failure")
        )
    try:
        result = await plugin.recall_context(SCOPE, "以前聊过的问题", "reply", "card")
        assert result["available"] is True
        assert result["evidence_status"] == "partial"
        assert result["items"][0]["source_id"] == "atm1:fixture"
        assert result["evidence_view"]["atoms"] == original["atoms"]
        assert result["diagnostics"]["baseline_preserved"] is True
        assert original["evidence_status"] == "ready"
    finally:
        await close_pool(plugin)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["revision", "scope", "card", "disabled"])
async def test_ready_fallback_rechecks_identity_revision_and_permission(changed):
    plugin, view = plugin_with_ready_baseline()
    if changed == "revision":
        plugin.peek_evidence_revision = lambda scope: {
            "available": True,
            "scope_key": scope_key(scope),
            "correction_revision": 1,
            "revocations": [],
        }
    elif changed == "scope":
        view["scope_key"] = scope_key({**SCOPE, "umo": "qq:GroupMessage:other"})
    elif changed == "card":
        view["card_version"] = "other-card"
    else:
        plugin._narrative_scope_allowed = AsyncMock(side_effect=[True, False])
    try:
        result = await plugin.recall_context(SCOPE, "以前", "reply", "card")
        assert result["available"] is False
        assert not result["items"]
    finally:
        await close_pool(plugin)


@pytest.mark.asyncio
async def test_no_enrichment_material_reuses_ready_view_without_a_second_render():
    plugin, _ = plugin_with_ready_baseline(mixed=False)
    try:
        result = await plugin.recall_context(SCOPE, "以前", "reply", "card")
        assert result["available"] is True
        assert result["items"]
        plugin.get_evidence_view.assert_not_awaited()
    finally:
        await close_pool(plugin)


@pytest.mark.asyncio
async def test_revision_snapshot_does_not_wait_for_ordinary_database_work(tmp_path):
    sidecar = EvidenceSidecar(tmp_path / "evidence.sqlite3")
    await sidecar.initialize()
    entered = threading.Event()

    def writer():
        with sidecar._lock:
            entered.set()
            time.sleep(0.25)

    thread = threading.Thread(target=writer)
    thread.start()
    await asyncio.to_thread(entered.wait, 1)
    started = time.monotonic()
    try:
        value = sidecar.peek_evidence_revision(scope_key(SCOPE))
        assert time.monotonic() - started < 0.1
        assert value["available"] is True and value["correction_revision"] == 0
    finally:
        await asyncio.to_thread(thread.join, 1)
        await sidecar.close()


@pytest.mark.asyncio
async def test_scope_auth_timings_are_separate_and_body_free(monkeypatch):
    from astrbot_plugin_dynamics_memory.core import plugin_helpers_mixin

    plugin, _ = plugin_with_ready_baseline(mixed=False)
    del plugin._narrative_scope_allowed
    plugin.context = SimpleNamespace(get_config=lambda _: {})
    monkeypatch.setattr(
        plugin_helpers_mixin, "is_event_memory_allowed", lambda *_: True
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
        lambda *_: True,
    )
    try:
        result = await plugin.recall_context(SCOPE, "以前", "reply", "card")
        timings = result["diagnostics"]["route_elapsed_ms"]
        assert {
            "revision_peek",
            "scope_session",
            "scope_plugin",
            "scope_config",
        } <= set(timings)
        assert all(type(v) is int and v >= 0 for v in timings.values())
    finally:
        await close_pool(plugin)


async def seeded_evidence(tmp_path):
    sidecar = EvidenceSidecar(tmp_path / "source-proof.sqlite3")
    await sidecar.initialize()
    archive = {
        "event_id": "message:old",
        "event_type": "message.received",
        "scope": dict(SCOPE),
        "text": "我喜欢喝清咖啡。",
        "actor_id": "u1",
        "occurred_at": 1000,
    }

    async def load_event(scope, event_id):
        return (
            deepcopy(archive)
            if scope == SCOPE and event_id == archive["event_id"]
            else None
        )

    service = EvidenceViewService(sidecar, scope_key, event_loader=load_event)
    descriptor, material = normalize_event_material(
        scope_key(SCOPE),
        SCOPE,
        {"source_kind": "event", **archive},
        card_version="card",
        correction_revision=0,
        purpose="reply",
    )
    atoms = service._validated_atoms(
        descriptor,
        material,
        [
            {
                "kind": "utterance_fact",
                "claim": archive["text"],
                "span_start": 0,
                "span_end": len(archive["text"]),
            }
        ],
    )
    assert atoms and await sidecar.store_atoms(descriptor, atoms)
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin._terminating = False
    plugin.initializer = SimpleNamespace(
        is_initialized=True, memory_engine=SimpleNamespace(text_processor=None)
    )
    plugin.evidence_service = service
    plugin._narrative_scope_allowed = AsyncMock(return_value=True)

    async def resolve(scope, refs):
        if scope != SCOPE or any(
            r.get("source_event_id") != archive["event_id"] for r in refs
        ):
            return [], len(refs)
        return [deepcopy(archive) for _ in refs], 0

    plugin._resolve_evidence_source_refs = AsyncMock(side_effect=resolve)
    return plugin, service, sidecar, archive, atoms


@pytest.mark.asyncio
async def test_ready_baseline_uses_real_canonical_source_and_atom_validators(tmp_path):
    plugin, _, sidecar, archive, _ = await seeded_evidence(tmp_path)
    try:
        baseline = await plugin._ready_narrative_baseline(
            SCOPE, "咖啡", "reply", "card", 0
        )
        view = baseline["evidence_view"]
        assert view["available"] is True
        assert view["atoms"][0]["claim"] == archive["text"]
        assert view["atoms"][0]["source_refs"][0]["author_id"] == "u1"
        assert view["atoms"][0]["status"] == "verified_utterance"
    finally:
        await sidecar.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["text", "author", "scope", "revoked"])
async def test_unproved_or_revoked_source_never_builds_a_ready_fallback(
    tmp_path, changed
):
    plugin, _, sidecar, archive, atoms = await seeded_evidence(tmp_path)
    try:
        if changed == "text":
            archive["text"] = "改过的原文"
        elif changed == "author":
            archive["actor_id"] = "different-member"
        elif changed == "scope":
            archive["scope"] = {**SCOPE, "umo": "qq:GroupMessage:other"}
        else:
            await sidecar.correct(
                atoms[0]["atom_id"], 0, "withdraw", reason="用户撤回了这一条"
            )
        revision = plugin.peek_evidence_revision(SCOPE)["correction_revision"]
        baseline = await plugin._ready_narrative_baseline(
            SCOPE, "咖啡", "reply", "card", revision
        )
        assert not baseline.get("evidence_view")
        assert not baseline["typed_items"]
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_correction_commit_cannot_expose_a_stale_revision(tmp_path, monkeypatch):
    _, _, sidecar, _, atoms = await seeded_evidence(tmp_path)
    original_connect = sidecar._connect
    commit_started, release = threading.Event(), threading.Event()

    class CommitGate:
        def __init__(self, db):
            self.db = db

        def __getattr__(self, name):
            return getattr(self.db, name)

        def commit(self):
            commit_started.set()
            release.wait(2)
            self.db.commit()

    monkeypatch.setattr(sidecar, "_connect", lambda: CommitGate(original_connect()))
    correction = asyncio.create_task(
        sidecar.correct(atoms[0]["atom_id"], 0, "withdraw", reason="撤回")
    )
    try:
        assert await asyncio.to_thread(commit_started.wait, 2)
        started = time.monotonic()
        assert sidecar.peek_evidence_revision(scope_key(SCOPE))["available"] is False
        assert time.monotonic() - started < 0.1
        release.set()
        await correction
        snapshot = sidecar.peek_evidence_revision(scope_key(SCOPE))
        assert snapshot["available"] is True and snapshot["correction_revision"] == 1
        assert snapshot["revocations"][0]["target_id"] == atoms[0]["atom_id"]
    finally:
        release.set()
        await correction
        await sidecar.close()
