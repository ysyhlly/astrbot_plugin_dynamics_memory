from __future__ import annotations

import asyncio

import pytest

from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    normalize_conversation_row,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


SCOPE = {
    "platform": "fixture-platform",
    "umo": "fixture-private-session",
    "persona_id": "fixture-persona",
    "instance_id": "fixture-instance",
}
ROW = {
    "id": 41,
    "session_id": "fixture-private-session",
    "role": "user",
    "content": "fixture exact user statement",
    "sender_id": "fixture-user",
    "sender_name": "fixture name",
    "group_id": "",
    "platform": "fixture-platform",
    "timestamp": 1234.5,
    "metadata": "{}",
}


@pytest.mark.asyncio
async def test_exact_private_user_row_becomes_deterministic_ready_atom(tmp_path):
    loaded = dict(ROW)
    calls = 0

    async def loader(descriptor):
        nonlocal calls
        calls += 1
        locator = descriptor["source_locator"]
        assert locator["session_id"] == ROW["session_id"]
        assert locator["row_id"] == ROW["id"]
        return dict(loaded)

    sidecar = EvidenceSidecar(tmp_path / "conversation-row.sqlite3")
    service = EvidenceViewService(sidecar, scope_key, conversation_row_loader=loader)
    await service.initialize()
    try:
        expected, _ = normalize_conversation_row(
            scope_key(SCOPE),
            SCOPE,
            {**ROW, "source_memory_id": 9, "source_kind": "conversation_row"},
            source_memory_id=9,
        )
        result = await service.ensure_conversation_row_ready(SCOPE, expected)
        assert result["status"] == "ready"
        assert len(result["atom_ids"]) == 1
        assert calls == 2  # source checked before derivation and again before commit

        view = await service.get_evidence_view(
            SCOPE,
            materials=[
                {**ROW, "source_memory_id": 9, "source_kind": "conversation_row"}
            ],
            allow_derivation=False,
        )
        assert view["evidence_status"] == "ready"
        atom = next(
            atom for atom in view["atoms"] if atom["atom_id"] == result["atom_ids"][0]
        )
        assert atom["kind"] == "utterance_fact"
        assert atom["status"] == "verified_utterance"
        assert atom["claim"] == ROW["content"]
        ref = atom["source_refs"][0]
        assert ref["source_kind"] == "conversation_row"
        assert ref["row_id"] == ROW["id"]
        assert ref["author_id"] == ROW["sender_id"]
        assert ref["source_event_id"] is None
        assert atom["root_evidence_ids"] == []
        assert atom["root_material_ids"] == [expected["material_id"]]
        assert len(await service.list_ready_atoms(SCOPE)) >= 1
        assert await service.validate_ready_material(SCOPE, expected)
        # Idempotent repeated invocation returns the same canonical atom.
        again = await service.ensure_conversation_row_ready(SCOPE, expected)
        assert again["status"] == "ready"
        assert again["atom_ids"] == result["atom_ids"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_changed_row_is_rejected_before_ready_write(tmp_path):
    loaded = dict(ROW)

    async def loader(_descriptor):
        return dict(loaded)

    sidecar = EvidenceSidecar(tmp_path / "tampered-row.sqlite3")
    service = EvidenceViewService(sidecar, scope_key, conversation_row_loader=loader)
    await service.initialize()
    try:
        expected, _ = normalize_conversation_row(
            scope_key(SCOPE),
            SCOPE,
            {**ROW, "source_memory_id": 10, "source_kind": "conversation_row"},
            source_memory_id=10,
        )
        loaded["content"] = "fixture changed source statement"
        result = await service.ensure_conversation_row_ready(SCOPE, expected)
        assert result["status"] == "invalid"
        assert result["atom_ids"] == []
        assert await service.list_ready_atoms(SCOPE) == []
        assert not await service.validate_ready_material(SCOPE, expected)
    finally:
        await service.close()


def test_assistant_rows_are_not_conversation_facts():
    descriptor = normalize_conversation_row(
        scope_key(SCOPE),
        SCOPE,
        {
            **ROW,
            "role": "assistant",
            "source_memory_id": 11,
            "source_kind": "conversation_row",
        },
        source_memory_id=11,
    )
    assert descriptor is None


@pytest.mark.asyncio
async def test_per_call_policy_deny_and_precommit_deny_never_store_ready(tmp_path):
    async def loader(_descriptor):
        return dict(ROW)

    sidecar = EvidenceSidecar(tmp_path / "per-call-policy.sqlite3")
    service = EvidenceViewService(sidecar, scope_key, conversation_row_loader=loader)
    await service.initialize()
    try:
        descriptor, _ = normalize_conversation_row(
            scope_key(SCOPE),
            SCOPE,
            {**ROW, "source_memory_id": 12, "source_kind": "conversation_row"},
            source_memory_id=12,
        )
        calls = 0

        async def deny_u2(_descriptor, material):
            assert material["author_id"] == ROW["sender_id"]
            return False

        denied = await service.ensure_conversation_row_ready(
            SCOPE, descriptor, policy_check=deny_u2
        )
        assert denied["status"] == "invalid"
        assert await service.list_ready_atoms(SCOPE) == []

        async def allow_then_deny(_descriptor, _material):
            nonlocal calls
            calls += 1
            return calls == 1

        changed_during_derivation = await service.ensure_conversation_row_ready(
            SCOPE,
            descriptor,
            policy_check=allow_then_deny,
        )
        assert calls == 2
        assert changed_during_derivation["status"] == "invalid"
        assert await service.list_ready_atoms(SCOPE) == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_per_call_policy_callbacks_are_isolated_for_concurrent_sources(tmp_path):
    scope_a = dict(SCOPE)
    scope_b = {**SCOPE, "umo": "fixture-private-session-b"}
    rows = {
        scope_a["umo"]: {
            **ROW,
            "id": 51,
            "session_id": scope_a["umo"],
            "sender_id": "fixture-user-a",
        },
        scope_b["umo"]: {
            **ROW,
            "id": 52,
            "session_id": scope_b["umo"],
            "sender_id": "fixture-user-b",
        },
    }

    async def loader(descriptor):
        return dict(rows.get(descriptor["source_locator"]["session_id"], {})) or None

    sidecar = EvidenceSidecar(tmp_path / "concurrent-policy.sqlite3")
    service = EvidenceViewService(sidecar, scope_key, conversation_row_loader=loader)
    await service.initialize()
    try:
        desc_a, _ = normalize_conversation_row(
            scope_key(scope_a),
            scope_a,
            {
                **rows[scope_a["umo"]],
                "source_memory_id": 21,
                "source_kind": "conversation_row",
            },
            source_memory_id=21,
        )
        desc_b, _ = normalize_conversation_row(
            scope_key(scope_b),
            scope_b,
            {
                **rows[scope_b["umo"]],
                "source_memory_id": 22,
                "source_kind": "conversation_row",
            },
            source_memory_id=22,
        )
        seen_a = []
        seen_b = []

        async def allow_a(descriptor, material):
            seen_a.append((descriptor["scope_key"], material["author_id"]))
            return (
                descriptor["source_locator"]["session_id"] == scope_a["umo"]
                and material["author_id"] == "fixture-user-a"
            )

        async def deny_b(descriptor, material):
            seen_b.append((descriptor["scope_key"], material["author_id"]))
            assert descriptor["source_locator"]["session_id"] == scope_b["umo"]
            assert material["author_id"] == "fixture-user-b"
            return False

        async def deny_all(_descriptor, _material):
            return False

        result_a, result_b = await asyncio.gather(
            service.ensure_conversation_row_ready(
                scope_a, desc_a, policy_check=allow_a
            ),
            service.ensure_conversation_row_ready(scope_b, desc_b, policy_check=deny_b),
        )
        assert result_a["status"] == "ready"
        assert result_b["status"] == "invalid"
        assert await service.validate_ready_material(
            scope_a, desc_a, policy_check=allow_a
        )
        assert not await service.validate_ready_material(
            scope_a, desc_a, policy_check=deny_all
        )
        assert seen_a and all(scope_id == scope_key(scope_a) for scope_id, _ in seen_a)
        assert seen_b and all(scope_id == scope_key(scope_b) for scope_id, _ in seen_b)
        assert len(await service.list_ready_atoms(scope_a)) == 1
        assert await service.list_ready_atoms(scope_b) == []
    finally:
        await service.close()
