from __future__ import annotations

import json

import pytest

from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    normalize_event_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


SCOPE = {
    "platform": "fixture-platform",
    "umo": "fixture:GroupMessage:fixture-room",
    "persona_id": "fixture-persona",
    "instance_id": "fixture-instance",
}
EVENT = {
    "source_kind": "event",
    "event_id": "fixture-event-1001",
    "event_type": "message.received",
    "scope": dict(SCOPE),
    "text": "fixture exact event text",
    "actor_id": "fixture-user",
    "occurred_at": 1700000000.0,
}


async def _seed(tmp_path, *, loader=None, missing_event_id=True, with_root=True):
    event_archive = {
        "event_id": EVENT["event_id"],
        "type": EVENT["event_type"],
        "scope": dict(SCOPE),
        "text": EVENT["text"],
        "actor_id": EVENT["actor_id"],
        "occurred_at": EVENT["occurred_at"],
    }

    async def event_loader(scope, event_id):
        assert scope == SCOPE
        if loader:
            return await loader(scope, event_id)
        return dict(event_archive) if event_id == event_archive["event_id"] else None

    sidecar = EvidenceSidecar(tmp_path / "legacy-ready.sqlite3")
    service = EvidenceViewService(sidecar, scope_key, event_loader=event_loader)
    await service.initialize()
    prepared = normalize_event_material(scope_key(SCOPE), SCOPE, EVENT)
    assert prepared is not None
    descriptor, material = prepared
    await sidecar.upsert_material(descriptor)
    atom = service._validated_atoms(
        descriptor,
        material,
        [
            {
                "kind": "utterance_fact",
                "claim": EVENT["text"],
                "span_start": 0,
                "span_end": len(EVENT["text"]),
            }
        ],
    )[0]
    source_ref = dict(atom["source_refs"][0])
    if missing_event_id:
        source_ref.pop("source_event_id", None)
        source_ref.pop("author_id", None)
        source_ref.pop("actor_id", None)
        source_ref["author_kind"] = "unknown"
        atom["subject_ids"] = []
    else:
        source_ref["source_event_id"] = "fixture-event-wrong"
    atom["source_refs"] = [source_ref]
    if not with_root:
        atom["root_evidence_ids"] = []
        atom["root_material_ids"] = []
        atom["source_refs"][0]["material_id"] = ""
    assert await sidecar.store_atoms(descriptor, [atom])
    typed_item = {
        "item_kind": "evidence_atom",
        "evidence_schema_version": 1,
        "source_id": atom["atom_id"],
        "text": atom["claim"],
        "scope": dict(SCOPE),
        "scope_key": scope_key(SCOPE),
        "correction_revision": 0,
        "source_refs": json.loads(json.dumps(atom["source_refs"])),
        "root_evidence_ids": list(atom["root_evidence_ids"]),
        "root_material_ids": list(atom["root_material_ids"]),
    }
    return service, sidecar, descriptor, atom, typed_item


@pytest.mark.asyncio
async def test_scoped_material_descriptor_reconstructs_only_canonical_event_locator(
    tmp_path,
):
    service, sidecar, descriptor, _atom, _typed = await _seed(tmp_path)
    try:
        rows = await service.get_material_descriptors(
            SCOPE, [descriptor["material_id"]]
        )
        row = rows[descriptor["material_id"]]
        assert row["scope_key"] == scope_key(SCOPE)
        assert row["material_id"] == descriptor["material_id"]
        assert row["material_kind"] == "source_event"
        assert row["text_hash"] == descriptor["text_hash"]
        assert row["projection_key"] == descriptor["projection_key"]
        assert row["correction_revision"] == 0
        assert row["source_locator"] == {
            "kind": "event",
            "event_id": EVENT["event_id"],
            "scope_key": scope_key(SCOPE),
            "scope": dict(SCOPE),
        }
        assert "text" not in row
        other_scope = {**SCOPE, "umo": "fixture:GroupMessage:other-room"}
        assert (
            await service.get_material_descriptors(
                other_scope, [descriptor["material_id"]]
            )
            == {}
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_legacy_ready_atom_response_only_restores_event_id_and_author(tmp_path):
    service, sidecar, descriptor, atom, typed = await _seed(tmp_path)
    before = await sidecar.get_atoms_by_ids(scope_key(SCOPE), [atom["atom_id"]])
    try:
        view = await service.get_evidence_view(
            SCOPE, typed_items=[typed], allow_derivation=False
        )
        assert view["diagnostics"]["source_authors_restored"] == 1
        restored = next(
            item for item in view["atoms"] if item["atom_id"] == atom["atom_id"]
        )
        ref = restored["source_refs"][0]
        assert ref["source_event_id"] == EVENT["event_id"]
        assert ref["author_id"] == EVENT["actor_id"]
        assert ref["actor_id"] == EVENT["actor_id"]
        assert ref["author_kind"] == "user"
        assert restored["subject_ids"] == [EVENT["actor_id"]]
        assert restored["claim"] == atom["claim"]
        assert restored["status"] == atom["status"]
        after = await sidecar.get_atoms_by_ids(scope_key(SCOPE), [atom["atom_id"]])
        assert after == before  # response-only repair; stored JSON is unchanged
        assert "source_event_id" not in after[atom["atom_id"]]["atom"]["source_refs"][0]
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper",
    [
        "text_hash",
        "snapshot_fingerprint",
        "wrong_ref_event_id",
        "missing_root",
        "archive_text",
    ],
)
async def test_legacy_ready_recovery_fails_closed_without_complete_proof(
    tmp_path, tamper
):
    async def changed_archive(_scope, event_id):
        return {
            "event_id": event_id,
            "type": EVENT["event_type"],
            "scope": dict(SCOPE),
            "text": "fixture changed event text",
            "actor_id": EVENT["actor_id"],
            "occurred_at": EVENT["occurred_at"],
        }

    service, sidecar, descriptor, atom, typed = await _seed(
        tmp_path,
        loader=changed_archive if tamper == "archive_text" else None,
        missing_event_id=tamper != "wrong_ref_event_id",
        with_root=tamper != "missing_root",
    )
    try:
        if tamper in {"text_hash", "snapshot_fingerprint"}:
            column, value = (
                ("text_hash", "0" * 64)
                if tamper == "text_hash"
                else ("snapshot_fingerprint", "event:fixture-event-forged")
            )
            with sidecar._lock, sidecar._session() as db:
                db.execute(
                    f"UPDATE evidence_materials SET {column}=? WHERE scope_key=? AND material_id=?",
                    (value, scope_key(SCOPE), descriptor["material_id"]),
                )
                db.commit()
        view = await service.get_evidence_view(
            SCOPE, typed_items=[typed], allow_derivation=False
        )
        assert not any(item["atom_id"] == atom["atom_id"] for item in view["atoms"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_withdrawn_legacy_ready_atom_is_never_restored(tmp_path):
    service, sidecar, _descriptor, atom, typed = await _seed(tmp_path)
    try:
        correction = await sidecar.correct(
            atom["atom_id"],
            expected_revision=0,
            action="withdraw",
            reason="fixture withdrawal",
        )
        assert correction["correction_revision"] == 1
        typed["correction_revision"] = correction["correction_revision"]
        view = await service.get_evidence_view(
            SCOPE, typed_items=[typed], allow_derivation=False
        )
        assert not any(item["atom_id"] == atom["atom_id"] for item in view["atoms"])
    finally:
        await service.close()


@pytest.mark.parametrize("tamper", ["actor", "subject"])
@pytest.mark.asyncio
async def test_existing_actor_or_subject_conflict_is_not_overwritten(tmp_path, tamper):
    service, sidecar, descriptor, atom, typed = await _seed(tmp_path)
    try:
        stored = await sidecar.get_atoms_by_ids(scope_key(SCOPE), [atom["atom_id"]])
        candidate = stored[atom["atom_id"]]["atom"]
        if tamper == "actor":
            candidate["source_refs"][0]["author_id"] = "fixture-other-user"
        else:
            candidate["subject_ids"] = ["fixture-other-user"]
        assert await sidecar.store_atoms(
            descriptor, [{**candidate, "atom_id": atom["atom_id"]}]
        )
        view = await service.get_evidence_view(
            SCOPE, typed_items=[typed], allow_derivation=False
        )
        assert not any(item["atom_id"] == atom["atom_id"] for item in view["atoms"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_mixed_present_and_missing_event_ids_are_rejected(tmp_path):
    service, sidecar, descriptor, atom, typed = await _seed(tmp_path)
    try:
        stored = await sidecar.get_atoms_by_ids(scope_key(SCOPE), [atom["atom_id"]])
        candidate = stored[atom["atom_id"]]["atom"]
        candidate["source_refs"].append(
            {
                **candidate["source_refs"][0],
                "source_event_id": EVENT["event_id"],
            }
        )
        assert await sidecar.store_atoms(
            descriptor, [{**candidate, "atom_id": atom["atom_id"]}]
        )
        view = await service.get_evidence_view(
            SCOPE, typed_items=[typed], allow_derivation=False
        )
        assert not any(item["atom_id"] == atom["atom_id"] for item in view["atoms"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_unverified_existing_author_is_not_erased(tmp_path):
    service, sidecar, _descriptor, atom, typed = await _seed(tmp_path)
    try:
        stored = await sidecar.get_atoms_by_ids(scope_key(SCOPE), [atom["atom_id"]])
        candidate = stored[atom["atom_id"]]["atom"]
        candidate["source_refs"][0].update(
            {
                "author_id": "fixture-preexisting-author",
                "actor_id": "fixture-preexisting-author",
                "author_kind": "user",
            }
        )

        async def archive_without_actor(_scope, event_id):
            return {
                "event_id": event_id,
                "type": EVENT["event_type"],
                "scope": dict(SCOPE),
                "text": EVENT["text"],
                "actor_id": None,
                "occurred_at": EVENT["occurred_at"],
            }

        service.event_loader = archive_without_actor
        assert await sidecar.store_atoms(
            _descriptor, [{**candidate, "atom_id": atom["atom_id"]}]
        )
        view = await service.get_evidence_view(
            SCOPE, typed_items=[typed], allow_derivation=False
        )
        assert not any(item["atom_id"] == atom["atom_id"] for item in view["atoms"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_unrelated_later_revision_does_not_invalidate_matching_old_projection(
    tmp_path,
):
    service, sidecar, _descriptor, atom, typed = await _seed(tmp_path)
    try:
        other_event = {
            **EVENT,
            "event_id": "fixture-event-1002",
            "text": "fixture unrelated event",
        }
        other_prepared = normalize_event_material(scope_key(SCOPE), SCOPE, other_event)
        other_descriptor, other_material = other_prepared
        await sidecar.upsert_material(other_descriptor)
        other_atom = service._validated_atoms(
            other_descriptor,
            other_material,
            [
                {
                    "kind": "utterance_fact",
                    "claim": other_event["text"],
                    "span_start": 0,
                    "span_end": len(other_event["text"]),
                }
            ],
        )[0]
        assert await sidecar.store_atoms(other_descriptor, [other_atom])
        correction = await sidecar.correct(
            other_atom["atom_id"],
            expected_revision=0,
            action="withdraw",
            reason="unrelated fixture correction",
        )
        assert correction["correction_revision"] == 1
        typed["correction_revision"] = 1

        view = await service.get_evidence_view(
            SCOPE, typed_items=[typed], allow_derivation=False
        )
        restored = next(
            item for item in view["atoms"] if item["atom_id"] == atom["atom_id"]
        )
        assert restored["source_refs"][0]["source_event_id"] == EVENT["event_id"]
        assert restored["source_refs"][0]["author_id"] == EVENT["actor_id"]
        assert view["correction_revision"] == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_revision_change_during_archive_read_rejects_response_repair(tmp_path):
    service, sidecar, _descriptor, atom, typed = await _seed(tmp_path)
    try:
        other_event = {
            **EVENT,
            "event_id": "fixture-event-1003",
            "text": "fixture event changed during read",
        }
        other_descriptor, other_material = normalize_event_material(
            scope_key(SCOPE),
            SCOPE,
            other_event,
        )
        await sidecar.upsert_material(other_descriptor)
        other_atom = service._validated_atoms(
            other_descriptor,
            other_material,
            [
                {
                    "kind": "utterance_fact",
                    "claim": other_event["text"],
                    "span_start": 0,
                    "span_end": len(other_event["text"]),
                }
            ],
        )[0]
        assert await sidecar.store_atoms(other_descriptor, [other_atom])

        async def mutate_revision(_scope, event_id):
            await sidecar.correct(
                other_atom["atom_id"],
                expected_revision=0,
                action="withdraw",
                reason="revision changed during event read",
            )
            return {
                "event_id": event_id,
                "type": EVENT["event_type"],
                "scope": dict(SCOPE),
                "text": EVENT["text"],
                "actor_id": EVENT["actor_id"],
                "occurred_at": EVENT["occurred_at"],
            }

        service.event_loader = mutate_revision
        view = await service.get_evidence_view(
            SCOPE, typed_items=[typed], allow_derivation=False
        )
        assert not any(item["atom_id"] == atom["atom_id"] for item in view["atoms"])
        assert (
            sidecar.peek_evidence_revision(scope_key(SCOPE))["correction_revision"] == 1
        )
    finally:
        await service.close()
