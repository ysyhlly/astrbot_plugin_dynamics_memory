"""Durable worker failures must never publish stale or unverified evidence."""

import asyncio
from unittest.mock import AsyncMock
import pytest
from tests.test_evidence_durable_queue import SCOPE_A, FakeClock, _source, _descriptor
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,outcome",
    [
        ("missing_source", "source_unavailable"),
        ("load_error", "failed"),
        ("changed_source", "source_unavailable"),
        ("missing_provider", "provider_unavailable"),
        ("empty_result", "invalid_schema"),
        ("bad_result", "invalid_schema"),
        ("bad_atoms", "invalid_schema"),
        ("changed_after", "source_unavailable"),
        ("policy_lost", "policy_unavailable"),
    ],
)
async def test_worker_failure_is_durable_and_does_not_publish(tmp_path, mode, outcome):
    descriptor, material = _descriptor(SCOPE_A, 0, "original immutable source")
    derived = False

    async def load(job):
        if mode == "missing_provider":
            service.derive_callback = None
        if mode == "load_error":
            raise OSError("source unavailable")
        if mode == "missing_source" or (mode == "changed_after" and derived):
            return None
        if mode == "changed_source":
            return {**material, "text": "changed source"}
        return material

    async def derive(materials):
        nonlocal derived
        derived = True
        key = materials[0]["material_id"]
        if mode == "empty_result":
            return []
        if mode == "bad_result":
            return {key: None}
        if mode == "bad_atoms":
            return {key: [{"claim": "invented"}]}
        return {key: []}

    policy = AsyncMock(
        side_effect=lambda *args: None if mode == "policy_lost" and derived else True
    )
    sidecar = EvidenceSidecar(tmp_path / "worker.db")
    service = EvidenceViewService(
        sidecar,
        scope_key,
        derive,
        material_loader=load,
        material_policy=policy,
        clock=FakeClock(),
        scheduler_poll_seconds=0.01,
    )
    await service.initialize()
    try:
        queued = await service.queue_material_derivation(
            [_source(SCOPE_A, 0, "original immutable source")], scope=SCOPE_A
        )
        assert queued["queued"] == 1
        for _ in range(200):
            state = await sidecar.material_retry_state(
                descriptor["material_id"], scope_key(SCOPE_A)
            )
            if state.get("outcome_code") not in (None, "", "pending"):
                break
            await asyncio.sleep(0.01)
        assert state["status"] != "ready", state
        if outcome == "failed":
            assert state["status"] == "failed", state
        else:
            assert state["outcome_code"] == outcome, state
        view = await service.get_evidence_view(
            SCOPE_A,
            source_refs=[_source(SCOPE_A, 0, "original immutable source")],
            allow_derivation=False,
        )
        assert view["atoms"] == []
        if mode in {
            "missing_source",
            "load_error",
            "changed_source",
            "missing_provider",
        }:
            assert not derived
    finally:
        await service.close()
