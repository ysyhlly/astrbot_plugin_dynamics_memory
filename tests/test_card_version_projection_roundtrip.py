from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    normalize_event_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


@pytest.mark.asyncio
async def test_reply_card_change_keeps_old_ready_summary_claim_with_raw_event_reference(
    tmp_path,
):
    scope = {
        "platform": "synthetic-platform",
        "umo": "synthetic:FriendMessage:user",
        "persona_id": "synthetic-persona",
        "instance_id": "synthetic-instance",
    }
    raw_text = "Synthetic user statement: I prefer jasmine tea."
    event = {
        "source_kind": "event",
        "event_id": "synthetic-event-card-check",
        "event_type": "message.received",
        "scope": dict(scope),
        "text": raw_text,
        "actor_id": "synthetic-user",
        "occurred_at": 1_700_000_300,
    }

    async def load_event(request_scope, event_id):
        assert request_scope == scope
        return dict(event) if event_id == event["event_id"] else None

    sidecar = EvidenceSidecar(tmp_path / "card-version-roundtrip.sqlite3")
    service = EvidenceViewService(sidecar, scope_key, event_loader=load_event)
    await service.initialize()
    try:
        old_descriptor, old_material = normalize_event_material(
            scope_key(scope),
            scope,
            event,
            card_version="synthetic-card-old",
            correction_revision=0,
            purpose="reply",
        )
        assert await sidecar.upsert_material(old_descriptor) == "queued"
        atom = service._validated_atoms(
            old_descriptor,
            old_material,
            [
                {
                    "kind": "utterance_fact",
                    "claim": "The user prefers jasmine tea.",
                    "span_start": 0,
                    "span_end": len(raw_text),
                }
            ],
        )[0]
        assert atom["status"] == "verified_utterance"
        assert await sidecar.store_atoms(old_descriptor, [atom]) is True
        typed = {
            "item_kind": "evidence_atom",
            "evidence_schema_version": 1,
            "source_id": atom["atom_id"],
            "text": atom["claim"],
            "scope": dict(scope),
            "scope_key": scope_key(scope),
            "correction_revision": 0,
            "source_refs": atom["source_refs"],
            "root_evidence_ids": atom["root_evidence_ids"],
            "root_material_ids": atom["root_material_ids"],
        }

        typed_only = await service.get_evidence_view(
            scope,
            typed_items=[typed],
            purpose="reply",
            card_version="synthetic-card-new",
            allow_derivation=False,
        )
        assert len(typed_only["atoms"]) == 1
        assert typed_only["diagnostics"]["typed_items_accepted"] == 1
        assert typed_only["diagnostics"]["source_authors_restored"] == 1

        mixed = await service.get_evidence_view(
            scope,
            source_refs=[event],
            typed_items=[typed],
            purpose="reply",
            card_version="synthetic-card-new",
            allow_derivation=False,
        )
        assert len(mixed["atoms"]) == 1
        assert mixed["atoms"][0]["claim"] == atom["claim"]
        assert mixed["diagnostics"]["typed_items_accepted"] == 1
        assert mixed["diagnostics"]["materials_cached"] == 1
        assert mixed["diagnostics"]["materials_deferred"] == 0
        assert mixed["diagnostics"]["typed_items_rejected"] == 0
        persisted = (
            await service.get_material_descriptors(
                scope, [old_descriptor["material_id"]]
            )
        )[old_descriptor["material_id"]]
        assert persisted["card_version"] == "synthetic-card-old"
        assert mixed["card_version"] == "synthetic-card-new"
    finally:
        await service.close()
