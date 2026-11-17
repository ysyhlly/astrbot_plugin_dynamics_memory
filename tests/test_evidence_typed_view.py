from __future__ import annotations

import asyncio

import pytest

from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    atom_id,
    normalize_event_material,
    normalize_host_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


SCOPE_A = {
    "platform": "fixture-platform",
    "umo": "fixture:room-a",
    "persona_id": "fixture-persona",
    "instance_id": "fixture-instance",
}
SCOPE_B = {**SCOPE_A, "umo": "fixture:room-b"}


def test_event_projection_preserves_archive_actor_across_normalization():
    event = {
        "source_kind": "event",
        "event_id": "fixture-event-1",
        "event_type": "message.received",
        "scope": dict(SCOPE_A),
        "text": "fixture event content",
        "actor_id": "fixture-user-1",
        "occurred_at": 1234,
    }
    first = normalize_event_material(scope_key(SCOPE_A), SCOPE_A, event)
    assert first is not None
    second = normalize_event_material(scope_key(SCOPE_A), SCOPE_A, first[1])
    assert second is not None
    assert first[1]["author_id"] == second[1]["author_id"] == "fixture-user-1"
    assert first[1]["occurred_at"] == second[1]["occurred_at"] == 1234


async def _service_with_source_backed_atom(tmp_path):
    sidecar = EvidenceSidecar(tmp_path / "typed-view.sqlite3")
    await sidecar.initialize()
    service = EvidenceViewService(sidecar, scope_key)
    await service.initialize()

    source = {
        "source_kind": "host_history",
        "conversation_id": "fixture-conversation",
        "snapshot_fingerprint": "fixture-snapshot-fingerprint",
        "position": 0,
        "role": "user",
        "text": "fixture source statement",
        "source_prefix_anchor": "fixture-anchor",
        "history_epoch": "fixture-epoch",
    }
    descriptor, _ = normalize_host_material(
        scope_key(SCOPE_A),
        source,
        conversation_id=source["conversation_id"],
        snapshot_fingerprint=source["snapshot_fingerprint"],
    )
    await sidecar.upsert_material(descriptor)
    claim = "fixture sourced claim"
    roots = [descriptor["material_id"]]
    aid = atom_id(scope_key(SCOPE_A), "reported_claim", claim, roots, 0)
    atom = {
        "atom_id": aid,
        "kind": "reported_claim",
        "claim": claim,
        "status": "reported_unverified",
        "confidence": 0.6,
        "source_refs": [
            {"source_kind": "host_history", "text_hash": descriptor["text_hash"]}
        ],
        "root_evidence_ids": [],
        "root_material_ids": roots,
    }
    await sidecar.store_atoms(descriptor, [atom])
    typed_item = {
        "item_kind": "evidence_atom",
        "evidence_schema_version": 1,
        "source_id": aid,
        "text": claim,
        "scope": dict(SCOPE_A),
        "scope_key": scope_key(SCOPE_A),
        "correction_revision": 0,
        "source_refs": atom["source_refs"],
        "root_evidence_ids": atom["root_evidence_ids"],
        "root_material_ids": atom["root_material_ids"],
    }
    return sidecar, service, typed_item, aid


@pytest.mark.asyncio
async def test_public_typed_view_accepts_only_current_source_backed_atom(tmp_path):
    (
        sidecar,
        service,
        typed_item,
        atom_id_value,
    ) = await _service_with_source_backed_atom(tmp_path)
    try:
        view = await service.get_evidence_view(SCOPE_A, typed_items=[typed_item])
        assert view["evidence_status"] == "ready"
        assert view["diagnostics"]["typed_items_accepted"] == 1, view["diagnostics"]
        assert view["diagnostics"]["typed_items_rejected"] == 0
        assert [atom["atom_id"] for atom in view["atoms"]] == [atom_id_value]
        assert view["atoms"][0]["claim"] == "fixture sourced claim"
        assert view["scope_key"] == typed_item["scope_key"]
        assert view["correction_revision"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_public_typed_view_rejects_unbacked_and_wrong_scope_items(tmp_path):
    _, service, typed_item, _ = await _service_with_source_backed_atom(tmp_path)
    try:
        unbacked = {**typed_item, "source_id": "atm1:missing-fixture-atom"}
        no_tag = {key: value for key, value in typed_item.items() if key != "item_kind"}
        wrong_scope = {
            **typed_item,
            "scope": dict(SCOPE_B),
            "scope_key": scope_key(SCOPE_B),
        }

        missing_view = await service.get_evidence_view(SCOPE_A, typed_items=[unbacked])
        untyped_view = await service.get_evidence_view(SCOPE_A, typed_items=[no_tag])
        wrong_scope_view = await service.get_evidence_view(
            SCOPE_A, typed_items=[wrong_scope]
        )

        for view in (missing_view, untyped_view, wrong_scope_view):
            assert view["atoms"] == []
            assert view["evidence_status"] == "invalid"
            assert view["diagnostics"]["typed_items_rejected"] == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_public_typed_view_does_not_revive_withdrawn_atom(tmp_path):
    (
        sidecar,
        service,
        typed_item,
        atom_id_value,
    ) = await _service_with_source_backed_atom(tmp_path)
    try:
        await sidecar.correct(
            atom_id_value,
            expected_revision=0,
            action="withdraw",
            reason="synthetic test revocation",
        )
        revoked = {**typed_item, "correction_revision": 1}
        view = await service.get_evidence_view(SCOPE_A, typed_items=[revoked])
        assert view["correction_revision"] == 1
        assert view["atoms"] == []
        assert view["evidence_status"] == "invalid"
        assert view["diagnostics"]["typed_items_rejected"] == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_ready_event_atom_repairs_missing_author_from_archive_without_derivation(
    tmp_path,
):
    event = {
        "source_kind": "event",
        "event_id": "fixture-ready-event-author",
        "event_type": "message.received",
        "scope": dict(SCOPE_A),
        "text": "fixture archived event",
        "actor_id": "fixture-author",
        "occurred_at": 5678,
    }
    descriptor, material = normalize_event_material(scope_key(SCOPE_A), SCOPE_A, event)
    sidecar = EvidenceSidecar(tmp_path / "old-ready-event.sqlite3")
    await sidecar.initialize()
    await sidecar.upsert_material(descriptor)
    roots = [descriptor["material_id"]]
    claim = "fixture old ready atom"
    atom_key = atom_id(scope_key(SCOPE_A), "reported_claim", claim, roots, 0)
    atom = {
        "atom_id": atom_key,
        "kind": "reported_claim",
        "status": "reported_unverified",
        "claim": claim,
        "confidence": 0.55,
        "scope": {"scope_key": scope_key(SCOPE_A)},
        "subject_ids": [],
        "source_refs": [
            {
                "material_id": descriptor["material_id"],
                "source_event_id": event["event_id"],
                "author_id": None,
                "source_status": "exact_event",
                "text_hash": descriptor["text_hash"],
                "exact_span": [0, len(event["text"])],
            }
        ],
        "root_evidence_ids": roots,
        "root_material_ids": [],
        "source_status": "exact_event",
        "valid_from": event["occurred_at"],
    }
    await sidecar.store_atoms(descriptor, [atom])

    async def event_loader(scope, event_id):
        return event if scope == SCOPE_A and event_id == event["event_id"] else None

    service = EvidenceViewService(sidecar, scope_key, event_loader=event_loader)
    await service.initialize()
    typed = {
        "item_kind": "evidence_atom",
        "evidence_schema_version": 1,
        "source_id": atom_key,
        "text": claim,
        "scope": dict(SCOPE_A),
        "scope_key": scope_key(SCOPE_A),
        "correction_revision": 0,
        "source_refs": atom["source_refs"],
        "root_evidence_ids": roots,
        "root_material_ids": [],
    }
    try:
        view = await service.get_evidence_view(SCOPE_A, typed_items=[typed])
        assert view["diagnostics"]["typed_items_accepted"] == 1, view["diagnostics"]
        assert view["diagnostics"]["source_authors_restored"] == 1
        restored = view["atoms"][0]
        assert restored["claim"] == claim
        assert restored["status"] == "reported_unverified"
        assert restored["source_refs"][0]["author_id"] == "fixture-author"
        assert restored["subject_ids"] == ["fixture-author"]
        assert restored["valid_from"] == 5678
        stored = await sidecar.get_atoms_by_ids(scope_key(SCOPE_A), [atom_key])
        assert stored[atom_key]["atom"]["source_refs"][0]["author_id"] is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_ready_event_atom_is_rejected_when_archive_provenance_is_missing(
    tmp_path,
):
    event = {
        "source_kind": "event",
        "event_id": "fixture-missing-archive-event",
        "event_type": "message.received",
        "scope": dict(SCOPE_A),
        "text": "fixture archived event",
        "actor_id": "fixture-author",
        "occurred_at": 5678,
    }
    descriptor, _ = normalize_event_material(scope_key(SCOPE_A), SCOPE_A, event)
    sidecar = EvidenceSidecar(tmp_path / "missing-event-archive.sqlite3")
    await sidecar.initialize()
    await sidecar.upsert_material(descriptor)
    roots = [descriptor["material_id"]]
    claim = "fixture event claim without archive"
    atom_key = atom_id(scope_key(SCOPE_A), "reported_claim", claim, roots, 0)
    atom = {
        "atom_id": atom_key,
        "kind": "reported_claim",
        "status": "reported_unverified",
        "claim": claim,
        "confidence": 0.55,
        "scope": {"scope_key": scope_key(SCOPE_A)},
        "subject_ids": [],
        "source_refs": [
            {
                "material_id": descriptor["material_id"],
                "source_event_id": event["event_id"],
                "author_id": None,
                "source_status": "exact_event",
                "text_hash": descriptor["text_hash"],
                "exact_span": [0, len(event["text"])],
            }
        ],
        "root_evidence_ids": roots,
        "root_material_ids": [],
        "source_status": "exact_event",
    }
    await sidecar.store_atoms(descriptor, [atom])
    service2 = EvidenceViewService(
        sidecar,
        scope_key,
        event_loader=lambda _scope, _event_id: asyncio.sleep(0, result=None),
    )
    await service2.initialize()
    typed = {
        "item_kind": "evidence_atom",
        "evidence_schema_version": 1,
        "source_id": atom_key,
        "text": claim,
        "scope": dict(SCOPE_A),
        "scope_key": scope_key(SCOPE_A),
        "correction_revision": 0,
        "source_refs": atom["source_refs"],
        "root_evidence_ids": roots,
        "root_material_ids": [],
    }
    try:
        view = await service2.get_evidence_view(SCOPE_A, typed_items=[typed])
        assert view["atoms"] == []
        assert view["diagnostics"]["typed_items_rejected"] == 1
        assert view["evidence_status"] == "invalid"
    finally:
        await service2.close()
