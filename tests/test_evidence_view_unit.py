"""Unit tests covering EvidenceViewService admission, wakeup, validation, and admin helpers."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


def _scope():
    return {
        "platform": "qq",
        "umo": "qq:FriendMessage:u1",
        "persona_id": "p1",
        "instance_id": "p1",
    }


@pytest.mark.asyncio
async def test_evidence_view_admissions_wakeups_and_admin(tmp_path):
    sidecar = EvidenceSidecar(str(tmp_path / "ev.db"))
    await sidecar.initialize()
    svc = EvidenceViewService(
        sidecar,
        scope_key_fn=scope_key,
        derive_callback=AsyncMock(return_value={}),
        conversation_row_loader=AsyncMock(return_value=None),
    )
    try:
        scope = _scope()
        # Before worker starts: _defer_admissions and defer_policy_wakeup report unavailable/False
        assert svc._defer_admissions([], scope, "reply", "c1")["available"] is False
        assert svc.defer_policy_wakeup(scope) is False
        assert svc.defer_policy_wakeup({}) is False
        assert await svc.wake_policy_jobs({}) == 0

        await svc.initialize()
        svc.wake_scheduler()

        assert svc.admission_diagnostics()["pending"] == 0
        assert svc.policy_wakeup_diagnostics()["pending"] == 0

        # Queue invalid material / purpose / empty materials
        assert (await svc.queue_material_derivation([], scope=scope, purpose="bad"))[
            "accepted"
        ] is False
        assert (await svc.queue_material_derivation([], scope=scope, purpose="reply"))[
            "available"
        ] is True
        res = await svc.queue_material_derivation(
            [{"source_kind": "bad"}], scope=scope, purpose="reply"
        )
        assert res["skipped"] == 1

        # Policy wakeup defer + coalescing
        assert svc.defer_policy_wakeup(scope) is True
        assert svc.defer_policy_wakeup(scope) is True
        await asyncio.sleep(0.05)
        assert svc.policy_wakeup_diagnostics()["processed"] >= 1

        # Defer admissions + coalescing
        prepared = svc._normalize(
            {
                "source_kind": "event",
                "event_id": "e1",
                "event_type": "message.received",
                "scope": scope,
                "text": "likes coffee",
                "actor_id": "u1",
                "occurred_at": 1000.0,
            },
            scope,
            "c1",
            "reply",
            0,
        )
        assert prepared is not None
        adm = svc._defer_admissions([prepared, prepared], scope, "reply", "c1")
        assert adm["admission_pending"] == 2
        await asyncio.sleep(0.05)
        assert svc.admission_diagnostics()["coalesced"] >= 1

        # Admin helpers and peek revision
        assert svc.peek_evidence_revision(scope)["available"] is True
        assert svc.peek_evidence_revision("bad")["available"] is False
        assert (await svc.list_admin_insights("bad"))["available"] is False
        assert (await svc.list_admin_insights(scope))["available"] is True
        assert (await svc.list_admin_insights_global())["available"] is True
        with pytest.raises(ValueError, match="insight or material not found"):
            await svc.correct_insight(
                "atm1:missing", 0, "withdraw", reason="test reason"
            )

        # get_material_descriptors & invalid guards
        assert await svc.get_material_descriptors("bad", []) == {}
        assert await svc.get_material_descriptors({"platform": "qq"}, []) == {}

        # get_evidence_view guard paths
        bad_p = await svc.get_evidence_view(scope, purpose="bad")
        assert bad_p["diagnostics"]["reason"] == "invalid_purpose"
        bad_s = await svc.get_evidence_view("bad", purpose="reply")
        assert bad_s["diagnostics"]["reason"] == "invalid_scope"

        # ensure_conversation_row_ready & validate_ready_material guard paths
        assert (await svc.ensure_conversation_row_ready("bad", {}))[
            "status"
        ] == "invalid"
        assert (await svc.ensure_conversation_row_ready(scope, {"source_locator": {}}))[
            "status"
        ] == "invalid"
        assert await svc.validate_ready_material("bad", {}) is False
        assert await svc.validate_ready_material(scope, {"source_locator": {}}) is False
        bad_locator = {
            "material_id": "cr1",
            "scope_key": scope_key(scope),
            "correction_revision": 0,
            "source_locator": {
                "kind": "conversation_row",
                "source_store": "livingmemory.conversations.messages",
                "scope": scope,
            },
        }
        assert (await svc.ensure_conversation_row_ready(scope, bad_locator))[
            "status"
        ] == "invalid"
        assert await svc.validate_ready_material(scope, bad_locator) is False
        bad_rev = {**bad_locator, "correction_revision": 999}
        assert (await svc.ensure_conversation_row_ready(scope, bad_rev))[
            "status"
        ] == "invalid"
        assert await svc.validate_ready_material(scope, bad_rev) is False
    finally:
        await svc.close()
