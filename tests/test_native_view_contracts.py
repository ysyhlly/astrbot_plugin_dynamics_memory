"""Multiple retrieval hits must preserve one scope/version and exact provenance."""

import copy
import pytest
from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin

SCOPE = {
    "platform": "fixture",
    "umo": "fixture:FriendMessage:u1",
    "persona_id": "p",
    "instance_id": "p",
}


def view(identifier="a", scope=None, revision=0, card="card"):
    atom = {
        "atom_id": identifier,
        "claim": "source " + identifier,
        "source_refs": [
            {"material_id": "m-" + identifier, "source_event_id": "e-" + identifier}
        ],
    }
    return {
        "scope": scope or dict(SCOPE),
        "purpose": "reply",
        "card_version": card,
        "evidence_view": {
            "available": True,
            "correction_revision": revision,
            "atoms": [atom],
            "slots": {"recent": [atom]},
            "diagnostics": {},
        },
        "native_materials": [
            {
                "material_id": "m-" + identifier,
                "source_locator": {"kind": "conversation_row"},
                "retrieval_hit": True,
            }
        ],
        "native_events": [{"event_id": "e-" + identifier, "atom_id": identifier}],
    }


def test_merge_deduplicates_hits_without_losing_retrieval_priority():
    rows = []
    a = view()
    assert LivingMemoryPlugin._native_merge_source_view(rows, a)
    assert LivingMemoryPlugin._native_merge_source_view(rows, view("b"))
    duplicate = view()
    duplicate["native_materials"][0]["retrieval_hit"] = False
    assert LivingMemoryPlugin._native_merge_source_view(rows, duplicate)
    assert len(rows) == 1
    assert {a["atom_id"] for a in rows[0]["evidence_view"]["atoms"]} == {"a", "b"}
    assert len(rows[0]["native_events"]) == 2
    assert all(m["retrieval_hit"] for m in rows[0]["native_materials"])
    pruned = LivingMemoryPlugin._native_prune_source_views(rows, {"a"})
    assert len(pruned[0]["evidence_view"]["atoms"]) == 1
    assert [m["material_id"] for m in pruned[0]["native_materials"]] == ["m-a"]
    assert [e["event_id"] for e in pruned[0]["native_events"]] == ["e-a"]
    assert len(rows[0]["evidence_view"]["atoms"]) == 2


@pytest.mark.parametrize("changes", [{"revision": 1}, {"card": "other"}])
def test_merge_rejects_mixed_correction_or_character_versions(changes):
    rows = [view()]
    before = copy.deepcopy(rows)
    assert not LivingMemoryPlugin._native_merge_source_view(rows, view("b", **changes))
    assert rows == before


def test_different_scopes_stay_separate_and_invalid_shapes_fail_closed():
    rows = [view()]
    assert LivingMemoryPlugin._native_merge_source_view(
        rows, view("b", scope={**SCOPE, "umo": "fixture:FriendMessage:u2"})
    )
    assert len(rows) == 2
    for malformed in (
        None,
        {},
        {"scope": {}, "evidence_view": {}},
        {"scope": dict(SCOPE), "evidence_view": None},
    ):
        assert not LivingMemoryPlugin._native_merge_source_view(rows, malformed)
    assert not LivingMemoryPlugin._native_merge_source_view(None, view())
    assert LivingMemoryPlugin._native_prune_source_views([None, {}, *rows], set()) == []
