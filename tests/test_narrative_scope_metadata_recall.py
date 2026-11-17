from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    atom_id,
    normalize_event_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


SCOPE = {
    "platform": "synthetic-platform",
    "umo": "synthetic:FriendMessage:user-1",
    "persona_id": "synthetic-persona",
    "instance_id": "synthetic-instance",
}
RAW_TEXT = "Synthetic user statement: I enjoy jasmine tea in the morning."
SUMMARY_CLAIM = "The user prefers jasmine tea."
ACTOR = "synthetic-user-1"


def _plugin(store: ConversationStore, service: EvidenceViewService | None = None):
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.initializer = SimpleNamespace(
        conversation_manager=SimpleNamespace(store=store),
        memory_engine=SimpleNamespace(text_processor=None),
        is_initialized=True,
    )
    plugin.evidence_service = service
    plugin._terminating = False

    async def allow_scope(_scope, _actor_id=""):
        return True

    plugin._narrative_scope_allowed = allow_scope
    return plugin


async def _store_event(store, event_id: str, archive_scope: dict):
    event = {
        "schema_version": 1,
        "type": "message.received",
        "event_id": event_id,
        "scope": dict(archive_scope),
        "text": RAW_TEXT,
        "actor_id": ACTOR,
        "occurred_at": 1_700_000_123,
    }
    assert await store.ingest_narrative_event(event) is True
    return event


async def _ready_atom(sidecar: EvidenceSidecar, event: dict, *, with_author=True):
    raw_event = {**event, "source_kind": "event", "event_type": event["type"]}
    descriptor, _material = normalize_event_material(
        scope_key(SCOPE),
        SCOPE,
        raw_event,
        card_version="synthetic-card",
        correction_revision=0,
        purpose="reply",
    )
    assert descriptor is not None
    atom_key = atom_id(
        scope_key(SCOPE),
        "utterance_fact",
        SUMMARY_CLAIM,
        [descriptor["material_id"]],
        0,
    )
    ref = {
        "material_id": descriptor["material_id"],
        "source_status": "exact_event",
        "text_hash": descriptor["text_hash"],
        "exact_span": [0, len(RAW_TEXT)],
        "author_kind": "user" if with_author else "unknown",
    }
    if with_author:
        ref.update({"source_event_id": event["event_id"], "author_id": ACTOR})
    atom = {
        "atom_id": atom_key,
        "kind": "utterance_fact",
        "status": "verified_utterance",
        "source_status": "exact_event",
        "claim": SUMMARY_CLAIM,
        "source_refs": [ref],
        "root_evidence_ids": [descriptor["material_id"]],
        "root_material_ids": [],
        "subject_ids": [ACTOR] if with_author else [],
    }
    assert await sidecar.store_atoms(descriptor, [atom]) is True
    return descriptor, atom


async def _rewrite_archived_scope(store: ConversationStore, event: dict, scope: dict):
    envelope = {**event, "scope": scope}
    key = f"{scope_key(SCOPE)}:{event['event_id']}"
    await store.connection.execute(
        "UPDATE narrative_events SET envelope=? WHERE event_key=?",
        (json.dumps(envelope, ensure_ascii=False), key),
    )
    await store.connection.commit()


@pytest.mark.asyncio
async def test_actual_resolver_and_evidence_view_accept_archive_alias_metadata(
    tmp_path,
):
    store = ConversationStore(str(tmp_path / "conversation.db"))
    await store.initialize()
    archive_scope = {**SCOPE, "persona_aliases": ["archive-alias"]}
    caller_scope = {**SCOPE, "persona_names": ["caller-alias"]}
    event = await _store_event(store, "synthetic-alias-event", archive_scope)
    sidecar = EvidenceSidecar(tmp_path / "sidecar.sqlite3")
    await sidecar.initialize()
    descriptor, atom = await _ready_atom(sidecar, event)
    plugin = _plugin(store)
    service = EvidenceViewService(
        sidecar,
        scope_key,
        event_loader=plugin._load_archived_evidence_event,
    )
    await service.initialize()
    plugin.evidence_service = service
    try:
        resolved, rejected = await plugin._resolve_evidence_source_refs(
            caller_scope,
            [{"source_kind": "event", "source_event_id": event["event_id"]}],
        )
        assert rejected == 0
        assert len(resolved) == 1
        assert resolved[0]["scope"] == SCOPE
        assert resolved[0]["scope"] != archive_scope

        baseline = await plugin._ready_narrative_baseline(
            caller_scope,
            "tea preference",
            "reply",
            "synthetic-card",
            0,
        )
        assert baseline["diagnostic"] == "ready", baseline
        assert baseline["typed_items"][0]["source_id"] == atom["atom_id"]
        view = await plugin.get_evidence_view(
            caller_scope,
            source_refs=baseline["source_refs"],
            typed_items=baseline["typed_items"],
            purpose="reply",
            card_version="synthetic-card",
            allow_derivation=False,
        )
        assert view["diagnostics"]["typed_items_accepted"] == 1, view["diagnostics"]
        assert view["diagnostics"]["typed_items_rejected"] == 0
        assert [row["atom_id"] for row in view["atoms"]] == [atom["atom_id"]]
        assert view["atoms"][0]["claim"] == SUMMARY_CLAIM
    finally:
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_actual_archived_event_callback_restores_legacy_missing_author_with_aliases(
    tmp_path,
):
    store = ConversationStore(str(tmp_path / "legacy-conversation.db"))
    await store.initialize()
    archive_scope = {**SCOPE, "persona_aliases": ["archive-only-alias"]}
    event = await _store_event(store, "synthetic-legacy-event", archive_scope)
    sidecar = EvidenceSidecar(tmp_path / "legacy-sidecar.sqlite3")
    await sidecar.initialize()
    descriptor, atom = await _ready_atom(sidecar, event, with_author=False)
    plugin = _plugin(store)
    service = EvidenceViewService(
        sidecar,
        scope_key,
        event_loader=plugin._load_archived_evidence_event,
    )
    await service.initialize()
    plugin.evidence_service = service
    typed = {
        "item_kind": "evidence_atom",
        "evidence_schema_version": 1,
        "source_id": atom["atom_id"],
        "text": SUMMARY_CLAIM,
        "scope": dict(SCOPE),
        "scope_key": scope_key(SCOPE),
        "correction_revision": 0,
        "source_refs": atom["source_refs"],
        "root_evidence_ids": atom["root_evidence_ids"],
        "root_material_ids": [],
    }
    try:
        view = await plugin.get_evidence_view(
            SCOPE,
            typed_items=[typed],
            purpose="reply",
            card_version="synthetic-card",
            allow_derivation=False,
        )
        assert view["diagnostics"]["typed_items_accepted"] == 1, view["diagnostics"]
        assert view["diagnostics"]["source_authors_restored"] == 1
        restored = view["atoms"][0]
        assert restored["source_refs"][0]["source_event_id"] == event["event_id"]
        assert restored["source_refs"][0]["author_id"] == ACTOR
        assert restored["subject_ids"] == [ACTOR]
        persisted = await sidecar.get_atoms_by_ids(scope_key(SCOPE), [atom["atom_id"]])
        assert (
            persisted[atom["atom_id"]]["atom"]["source_refs"][0].get("source_event_id")
            is None
        )
        assert (
            persisted[atom["atom_id"]]["atom"]["source_refs"][0].get("author_id")
            is None
        )
    finally:
        await service.close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_key", ["platform", "umo", "persona_id", "instance_id"]
)
@pytest.mark.parametrize("mutation", ["changed", "missing", "wrong_type"])
async def test_actual_conversation_loader_rejects_invalid_archived_identity_fields(
    tmp_path, identity_key, mutation
):
    store = ConversationStore(str(tmp_path / f"invalid-{identity_key}-{mutation}.db"))
    await store.initialize()
    event = await _store_event(
        store, "synthetic-invalid-identity", {**SCOPE, "persona_aliases": ["x"]}
    )
    bad_scope = {**SCOPE, "persona_aliases": ["matching-alias-does-not-rescue"]}
    if mutation == "changed":
        bad_scope[identity_key] = f"different-{identity_key}"
    elif mutation == "missing":
        bad_scope.pop(identity_key)
    else:
        bad_scope[identity_key] = 17
    await _rewrite_archived_scope(store, event, bad_scope)
    plugin = _plugin(store)
    try:
        resolved, rejected = await plugin._resolve_evidence_source_refs(
            SCOPE,
            [{"source_kind": "event", "source_event_id": event["event_id"]}],
        )
        assert resolved == []
        assert rejected == 1
        assert (
            await plugin._load_archived_evidence_event(SCOPE, event["event_id"]) is None
        )
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_key", ["platform", "umo", "persona_id", "instance_id"]
)
@pytest.mark.parametrize("mutation", ["changed", "missing", "wrong_type"])
async def test_actual_resolver_rejects_invalid_caller_identity_even_with_alias(
    tmp_path, identity_key, mutation
):
    store = ConversationStore(str(tmp_path / f"caller-{identity_key}-{mutation}.db"))
    await store.initialize()
    event = await _store_event(
        store, "synthetic-caller-identity", {**SCOPE, "persona_aliases": ["x"]}
    )
    caller = {**SCOPE, "persona_names": ["must-not-repair-identity"]}
    if mutation == "changed":
        caller[identity_key] = f"different-{identity_key}"
    elif mutation == "missing":
        caller.pop(identity_key)
    else:
        caller[identity_key] = 17
    plugin = _plugin(store)
    try:
        resolved, rejected = await plugin._resolve_evidence_source_refs(
            caller,
            [{"source_kind": "event", "source_event_id": event["event_id"]}],
        )
        assert resolved == []
        assert rejected == 1
        assert (
            await plugin._load_archived_evidence_event(caller, event["event_id"])
            is None
        )
    finally:
        await store.close()
