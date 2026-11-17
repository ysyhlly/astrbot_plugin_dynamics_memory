import asyncio
import threading
from types import SimpleNamespace as NS

import pytest
from astrbot_plugin_dynamics_memory.core.blocking_policy import BlockingPolicyGate
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar

SCOPE = {
    "platform": "qq",
    "umo": "qq:GroupMessage:test",
    "persona_id": "role",
    "instance_id": "role",
}


@pytest.mark.asyncio
async def test_trusted_owner_capture_does_not_wait_for_scheduler_write(tmp_path):
    async def derive(*args):
        return {}

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "evidence.sqlite3"), scope_key, derive
    )
    await service.initialize()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(scope):
        entered.set()
        await release.wait()
        return 0

    service.wake_policy_jobs = blocked
    event = NS(
        unified_msg_origin=SCOPE["umo"],
        get_platform_id=lambda: "qq",
        get_self_id=lambda: "bot",
    )
    platform = object()

    async def context(**kwargs):
        return dict(
            ready=True,
            umo=SCOPE["umo"],
            scope=SCOPE,
            owner={"self_id": "bot", "platform_instance_id": "qq"},
        )

    gate = BlockingPolicyGate(None)
    gate._discover = lambda: ("found", NS(enabled=True, resolve_context=context))
    # Use one stable service and one stable actual platform identity.
    owner_service = gate._discover()[1]
    gate._discover = lambda: ("found", owner_service)
    gate._platform = lambda *args: platform
    gate.wakeup = service.defer_policy_wakeup
    try:
        assert (
            await asyncio.wait_for(gate.observe_trusted_event(event, SCOPE), 0.3)
            is True
        )
        await asyncio.wait_for(entered.wait(), 0.3)
        assert not release.is_set()
        assert await gate.owner_ready(SCOPE["umo"]) is True
        assert service.defer_policy_wakeup(SCOPE) is True
        assert service.policy_wakeup_diagnostics()["coalesced"] == 1
    finally:
        release.set()
        await service.close()
    assert service.policy_wakeup_diagnostics()["pending"] == 0
    assert service._policy_wakeup_worker is None


@pytest.mark.asyncio
async def test_no_waiters_skips_writer_lock_even_during_contention(tmp_path):
    sidecar = EvidenceSidecar(tmp_path / "evidence.sqlite3")
    await sidecar.initialize()
    entered, release = threading.Event(), threading.Event()

    def hold():
        with sidecar._lock:
            entered.set()
            release.wait()

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        await asyncio.to_thread(entered.wait)
        assert (
            await asyncio.wait_for(sidecar.wake_policy_deferred(SCOPE["umo"], 100), 0.3)
            == 0
        )
        assert not release.is_set()
        assert sidecar.read_diagnostics()["_has_policy_waiters_sync"]["calls"] == 1
    finally:
        release.set()
        await asyncio.to_thread(thread.join)
        await sidecar.close()


@pytest.mark.asyncio
async def test_existing_waiters_are_woken_only_for_the_requested_umo(tmp_path):
    sidecar = EvidenceSidecar(tmp_path / "evidence.sqlite3")
    await sidecar.initialize()
    try:
        with sidecar._lock, sidecar._session() as db:
            for material_id, umo in [
                ("mine", SCOPE["umo"]),
                ("other", "qq:GroupMessage:other"),
            ]:
                db.execute(
                    """INSERT INTO evidence_derivation_queue
                    (material_id,scope_key,umo,source_locator_json,correction_revision,projection_key,text_hash,
                     status,available_at,created_at,updated_at) VALUES (?,?,?,'{}',0,'p','h','owner_wait',999,1,1)""",
                    (material_id, material_id, umo),
                )
            db.commit()
        assert await sidecar.wake_policy_deferred(SCOPE["umo"], 100) == 1
        with sidecar._read_session() as db:
            states = dict(
                db.execute("SELECT material_id,status FROM evidence_derivation_queue")
            )
        assert states == {"mine": "deferred", "other": "owner_wait"}
        assert await sidecar.wake_policy_deferred(SCOPE["umo"], 101) == 0
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_deferred_wakeups_settle_before_sidecar_close(tmp_path):
    async def derive(*args):
        return {}

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "evidence.sqlite3"), scope_key, derive
    )
    await service.initialize()
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(scope):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    service.wake_policy_jobs = blocked
    assert service.defer_policy_wakeup(SCOPE)
    await entered.wait()
    await service.close()
    assert cancelled.is_set()
    assert not service.defer_policy_wakeup(SCOPE)
