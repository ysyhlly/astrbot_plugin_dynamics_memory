from __future__ import annotations

import asyncio
import sqlite3

import pytest

from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    normalize_host_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    normalize_event_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


SCOPE_A = {
    "platform": "fixture",
    "umo": "fixture:scope-a",
    "persona_id": "p",
    "instance_id": "i",
}
SCOPE_B = {
    "platform": "fixture",
    "umo": "fixture:scope-b",
    "persona_id": "p",
    "instance_id": "i",
}


class FakeClock:
    def __init__(self, value: float = 10000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def _source(scope: dict, position: int, text: str):
    return {
        "source_kind": "host_history",
        "conversation_id": "fixture-conversation-" + scope["umo"],
        "snapshot_fingerprint": "fixture-snapshot-" + scope["umo"],
        "position": position,
        "role": "user",
        "text": text,
    }


def _descriptor(scope: dict, position: int, text: str):
    key = scope_key(scope)
    raw = _source(scope, position, text)
    descriptor, material = normalize_host_material(
        key,
        raw,
        conversation_id=raw["conversation_id"],
        snapshot_fingerprint=raw["snapshot_fingerprint"],
    )
    descriptor["correction_revision"] = 0
    locator = dict(descriptor["source_locator"])
    locator["scope"] = {
        name: scope[name] for name in ("platform", "umo", "persona_id", "instance_id")
    }
    descriptor["source_locator"] = locator
    material["source_locator"] = locator
    return descriptor, material


def _queue_status(path, material_id):
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT status,available_at,lease_id,lease_until FROM evidence_derivation_queue WHERE material_id=?",
            (material_id,),
        ).fetchone()
    return row


async def _wait_until(predicate, timeout: float = 2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached before timeout")


async def _wait_material_status(
    sidecar, material_id: str, scope: dict, status: str, timeout: float = 2.0
):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        state = await sidecar.material_retry_state(material_id, scope_key(scope))
        if state["status"] == status:
            return state
        await asyncio.sleep(0.01)
    raise AssertionError(f"material did not reach {status}")


@pytest.mark.asyncio
async def test_bounded_queue_deferred_source_runs_without_repeated_request(tmp_path):
    clock = FakeClock()
    path = tmp_path / "queue.sqlite3"
    sidecar = EvidenceSidecar(path)
    first_scope = dict(SCOPE_A)
    sources = {}
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_finished = asyncio.Event()
    calls = []

    first_descriptor, first_material = _descriptor(
        first_scope, 0, "fixture first source"
    )
    second_descriptor, second_material = _descriptor(
        first_scope, 1, "fixture deferred source"
    )
    sources[first_descriptor["material_id"]] = first_material
    sources[second_descriptor["material_id"]] = second_material

    async def load(job):
        return sources.get(job["material_id"])

    async def derive(materials):
        for item in materials:
            calls.append(item["text"])
        if len(calls) == 1:
            first_started.set()
            await release_first.wait()
        if len(calls) >= 2:
            second_finished.set()
        return {item["material_id"]: [] for item in materials}

    service = EvidenceViewService(
        sidecar,
        scope_key,
        derive,
        queue_size=1,
        per_scope_hourly_limit=24,
        material_loader=load,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        first = await service.queue_material_derivation(
            [_source(first_scope, 0, "fixture first source")], scope=first_scope
        )
        assert first["queued"] == 1
        await asyncio.wait_for(first_started.wait(), 1)
        second = await service.queue_material_derivation(
            [_source(first_scope, 1, "fixture deferred source")], scope=first_scope
        )
        assert second["deferred"] == 1
        assert _queue_status(path, second_descriptor["material_id"])[0] == "deferred"
        release_first.set()
        await asyncio.wait_for(second_finished.wait(), 1)
        state = await _wait_material_status(
            sidecar, second_descriptor["material_id"], first_scope, "ready"
        )
        assert state["status"] == "ready"
        assert calls == ["fixture first source", "fixture deferred source"]
    finally:
        release_first.set()
        await service.close()


@pytest.mark.asyncio
async def test_deferred_event_rehydrates_author_and_source_time_after_restart(tmp_path):
    clock = FakeClock(90000.0)
    path = tmp_path / "event-author-restart.sqlite3"
    events = {
        "fixture-event-a": {
            "source_kind": "event",
            "event_id": "fixture-event-a",
            "event_type": "message.received",
            "scope": dict(SCOPE_A),
            "text": "fixture first event",
            "actor_id": "fixture-user-a",
            "occurred_at": 1233,
        },
        "fixture-event-b": {
            "source_kind": "event",
            "event_id": "fixture-event-b",
            "event_type": "message.received",
            "scope": dict(SCOPE_A),
            "text": "fixture deferred event",
            "actor_id": "fixture-user-b",
            "occurred_at": 1234,
        },
    }
    descriptors = {
        key: normalize_event_material(scope_key(SCOPE_A), SCOPE_A, row)[0]
        for key, row in events.items()
    }
    first_started = asyncio.Event()

    async def load(job):
        event_id = job["source_locator"]["event_id"]
        return events.get(event_id)

    async def block_first(materials):
        first_started.set()
        await asyncio.Event().wait()

    sidecar1 = EvidenceSidecar(path)
    service1 = EvidenceViewService(
        sidecar1,
        scope_key,
        block_first,
        queue_size=1,
        material_loader=load,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service1.initialize()
    queued = await service1.queue_material_derivation(
        list(events.values()), scope=SCOPE_A
    )
    assert queued["queued"] == 1 and queued["deferred"] == 1
    await asyncio.wait_for(first_started.wait(), 1)
    await service1.close()

    async def derive(materials):
        output = {}
        for material in materials:
            text = material["text"]
            output[material["material_id"]] = [
                {
                    "kind": "reported_claim",
                    "claim": "fixture durable event claim",
                    "span_start": 0,
                    "span_end": len(text),
                }
            ]
        return output

    sidecar2 = EvidenceSidecar(path)
    service2 = EvidenceViewService(
        sidecar2,
        scope_key,
        derive,
        queue_size=1,
        material_loader=load,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service2.initialize()
    try:
        await _wait_material_status(
            sidecar2, descriptors["fixture-event-b"]["material_id"], SCOPE_A, "ready"
        )
        cached = await sidecar2.get_materials(
            scope_key(SCOPE_A), [descriptors["fixture-event-b"]["material_id"]]
        )
        atom = cached[descriptors["fixture-event-b"]["material_id"]]["atoms"][0]["atom"]
        assert atom["subject_ids"] == ["fixture-user-b"]
        assert atom["source_refs"][0]["author_id"] == "fixture-user-b"
        assert atom["source_refs"][0]["occurred_at"] == 1234
        assert atom["valid_from"] == 1234
    finally:
        await service2.close()


@pytest.mark.asyncio
async def test_verified_execution_receipt_survives_normal_cache_view(tmp_path):
    event = {
        "source_kind": "event",
        "event_id": "fixture-receipt-event",
        "event_type": "tool.execution_receipt",
        "scope": dict(SCOPE_A),
        "text": "runtime confirmed fixture operation",
        "actor_id": "fixture-bot",
        "occurred_at": 4321,
        "receipt": {
            "schema_version": 1,
            "source": "astrbot_runtime",
            "status": "success",
            "operation_id": "fixture-operation",
        },
    }
    descriptor, _ = normalize_event_material(scope_key(SCOPE_A), SCOPE_A, event)

    async def derive(materials):
        return {
            item["material_id"]: [
                {
                    "kind": "execution_receipt",
                    "claim": "fixture operation succeeded",
                    "span_start": 0,
                    "span_end": len(item["text"]),
                }
            ]
            for item in materials
        }

    sidecar = EvidenceSidecar(tmp_path / "verified-receipt.sqlite3")
    service = EvidenceViewService(
        sidecar, scope_key, derive, scheduler_poll_seconds=0.02
    )
    await service.initialize()
    try:
        initial = await service.get_evidence_view(SCOPE_A, source_refs=[event])
        assert initial["diagnostics"]["materials_admission_pending"] == 1
        await _wait_material_status(
            sidecar, descriptor["material_id"], SCOPE_A, "ready"
        )
        cached = await service.get_evidence_view(SCOPE_A, source_refs=[event])
        assert cached["diagnostics"]["materials_cached"] == 1
        assert len(cached["atoms"]) == 1
        assert cached["atoms"][0]["status"] == "verified"
        assert cached["atoms"][0]["kind"] == "execution_receipt"
        assert (
            cached["slots"]["commitments_and_boundaries"][0]["atom_id"]
            == cached["atoms"][0]["atom_id"]
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_volatile_unknown_host_source_is_rejected_by_current_policy(tmp_path):
    clock = FakeClock(44000.0)
    source = _source(SCOPE_A, 0, "fixture unknown host author")
    descriptor, _ = _descriptor(SCOPE_A, 0, source["text"])
    policy_calls = []
    derived = []

    async def policy(job, material):
        policy_calls.append((job["material_id"], material.get("author_id")))
        return bool(material.get("author_id"))

    async def derive(materials):
        derived.extend(materials)
        return {}

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "policy-volatile.sqlite3"),
        scope_key,
        derive,
        material_policy=policy,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        result = await service.queue_material_derivation([source], scope=SCOPE_A)
        assert result["queued"] == 1
        state = await _wait_material_status(
            service.sidecar, descriptor["material_id"], SCOPE_A, "failed"
        )
        assert state["outcome_code"] == "policy_refusal"
        assert policy_calls == [(descriptor["material_id"], None)]
        assert derived == []
        assert (
            await service.sidecar.material_retry_due(
                descriptor["material_id"], scope_key(SCOPE_A), now=clock()
            )
            is False
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_policy_is_rechecked_before_persisting_provider_result(tmp_path):
    clock = FakeClock(45000.0)
    source = _source(SCOPE_A, 0, "fixture policy changes during derivation")
    descriptor, _ = _descriptor(SCOPE_A, 0, source["text"])
    policy_calls = 0

    async def policy(job, material):
        nonlocal policy_calls
        policy_calls += 1
        return policy_calls == 1

    async def derive(materials):
        return {
            item["material_id"]: [
                {
                    "kind": "reported_claim",
                    "claim": "fixture candidate",
                    "span_start": 0,
                    "span_end": len(item["text"]),
                }
            ]
            for item in materials
        }

    sidecar = EvidenceSidecar(tmp_path / "policy-before-store.sqlite3")
    service = EvidenceViewService(
        sidecar,
        scope_key,
        derive,
        material_policy=policy,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        await service.queue_material_derivation([source], scope=SCOPE_A)
        state = await _wait_material_status(
            sidecar, descriptor["material_id"], SCOPE_A, "failed"
        )
        assert state["outcome_code"] == "policy_refusal"
        assert policy_calls == 2
        cached = await sidecar.get_materials(
            scope_key(SCOPE_A), [descriptor["material_id"]]
        )
        assert cached[descriptor["material_id"]]["status"] == "failed"
        assert cached[descriptor["material_id"]].get("atoms", []) == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_ordinary_completion_claim_does_not_become_verified_receipt(tmp_path):
    event = {
        "source_kind": "event",
        "event_id": "fixture-completion-claim",
        "event_type": "delivery.success",
        "scope": dict(SCOPE_A),
        "text": "I finished the fixture task",
        "actor_id": "fixture-bot",
        "occurred_at": 4322,
    }
    descriptor, _ = normalize_event_material(scope_key(SCOPE_A), SCOPE_A, event)

    async def derive(materials):
        return {
            item["material_id"]: [
                {
                    "kind": "execution_receipt",
                    "claim": "fixture task completed",
                    "span_start": 0,
                    "span_end": len(item["text"]),
                }
            ]
            for item in materials
        }

    sidecar = EvidenceSidecar(tmp_path / "unverified-completion.sqlite3")
    service = EvidenceViewService(
        sidecar, scope_key, derive, scheduler_poll_seconds=0.02
    )
    await service.initialize()
    try:
        await service.get_evidence_view(SCOPE_A, source_refs=[event])
        await _wait_material_status(
            sidecar, descriptor["material_id"], SCOPE_A, "ready"
        )
        cached = await service.get_evidence_view(SCOPE_A, source_refs=[event])
        assert len(cached["atoms"]) == 1
        assert cached["atoms"][0]["kind"] == "execution_claim"
        assert cached["atoms"][0]["status"] == "reported_unverified"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_scope_quota_survives_restart_and_fake_clock_wakes_deferred_job(tmp_path):
    clock = FakeClock(20000.0)
    path = tmp_path / "quota.sqlite3"
    sources = {}
    first_descriptor, first_material = _descriptor(SCOPE_A, 0, "fixture quota first")
    second_descriptor, second_material = _descriptor(SCOPE_A, 1, "fixture quota second")
    sources[first_descriptor["material_id"]] = first_material
    sources[second_descriptor["material_id"]] = second_material
    first_calls = []

    async def load(job):
        return sources.get(job["material_id"])

    async def derive_first(materials):
        first_calls.extend(item["text"] for item in materials)
        return {item["material_id"]: [] for item in materials}

    sidecar1 = EvidenceSidecar(path)
    service1 = EvidenceViewService(
        sidecar1,
        scope_key,
        derive_first,
        queue_size=8,
        per_scope_hourly_limit=1,
        material_loader=load,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service1.initialize()
    await service1.queue_material_derivation(
        [_source(SCOPE_A, 0, "fixture quota first")], scope=SCOPE_A
    )
    await _wait_until(lambda: "fixture quota first" in first_calls)
    await service1.queue_material_derivation(
        [_source(SCOPE_A, 1, "fixture quota second")], scope=SCOPE_A
    )
    await _wait_until(
        lambda: (
            (_queue_status(path, second_descriptor["material_id"]) or (None,))[0]
            == "deferred"
        )
    )
    pending = _queue_status(path, second_descriptor["material_id"])
    assert pending[1] == pytest.approx(23600.0)
    await service1.close()

    second_calls = []

    async def derive_second(materials):
        second_calls.extend(item["text"] for item in materials)
        return {item["material_id"]: [] for item in materials}

    sidecar2 = EvidenceSidecar(path)
    service2 = EvidenceViewService(
        sidecar2,
        scope_key,
        derive_second,
        queue_size=8,
        per_scope_hourly_limit=1,
        material_loader=load,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service2.initialize()
    try:
        await asyncio.sleep(0.06)
        assert second_calls == []
        clock.value += 3601
        service2.wake_scheduler()
        await _wait_until(lambda: "fixture quota second" in second_calls)
        # The derivation callback precedes the transaction that publishes its result.
        await _wait_material_status(
            sidecar2, second_descriptor["material_id"], SCOPE_A, "ready"
        )
        with sqlite3.connect(path) as db:
            count = db.execute(
                "SELECT COUNT(*) FROM evidence_queue_admissions WHERE scope_key=?",
                (scope_key(SCOPE_A),),
            ).fetchone()[0]
        assert count == 2
    finally:
        await service2.close()


@pytest.mark.asyncio
async def test_scope_cursor_prevents_backlog_starvation(tmp_path):
    clock = FakeClock(30000.0)
    path = tmp_path / "fairness.sqlite3"
    sources = {}
    cases = [
        (SCOPE_A, 0, "scope-a-first"),
        (SCOPE_B, 0, "scope-b-first"),
        (SCOPE_A, 1, "scope-a-second"),
    ]
    for scope, position, text in cases:
        descriptor, material = _descriptor(scope, position, "fixture " + text)
        sources[descriptor["material_id"]] = material
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def load(job):
        return sources.get(job["material_id"])

    async def derive(materials):
        calls.extend(item["text"] for item in materials)
        if len(calls) == 1:
            started.set()
            await release.wait()
        return {item["material_id"]: [] for item in materials}

    service = EvidenceViewService(
        EvidenceSidecar(path),
        scope_key,
        derive,
        queue_size=2,
        per_scope_hourly_limit=24,
        material_loader=load,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        await service.queue_material_derivation(
            [_source(SCOPE_A, 0, "fixture scope-a-first")], scope=SCOPE_A
        )
        await asyncio.wait_for(started.wait(), 1)
        await service.queue_material_derivation(
            [_source(SCOPE_B, 0, "fixture scope-b-first")], scope=SCOPE_B
        )
        deferred = await service.queue_material_derivation(
            [_source(SCOPE_A, 1, "fixture scope-a-second")], scope=SCOPE_A
        )
        assert deferred["deferred"] == 1
        release.set()
        await _wait_until(lambda: len(calls) == 3)
        assert calls[0] == "fixture scope-a-first"
        assert calls[1] == "fixture scope-b-first"
        assert calls[2] == "fixture scope-a-second"
    finally:
        release.set()
        await service.close()


@pytest.mark.asyncio
async def test_owner_wait_does_not_use_capacity_or_quota_and_wakes_all_personas(
    tmp_path,
):
    clock = FakeClock(35000.0)
    scope_a2 = {
        **SCOPE_A,
        "persona_id": "fixture-persona-2",
        "instance_id": "fixture-instance-2",
    }
    ready_umos = {SCOPE_A["umo"]: False, SCOPE_B["umo"]: True}
    cases = [
        (SCOPE_A, 0, "fixture owner wait a"),
        (scope_a2, 0, "fixture owner wait a persona two"),
        (SCOPE_B, 0, "fixture ready scope b"),
    ]
    sources = {}
    descriptors = {}
    descriptors_by_scope = {}
    for scope, position, text in cases:
        descriptor, material = _descriptor(scope, position, text)
        descriptors[descriptor["material_id"]] = descriptor
        descriptors_by_scope[scope_key(scope)] = descriptor
        sources[descriptor["material_id"]] = material

    async def load(job):
        return sources.get(job["material_id"])

    async def owner_ready(umo):
        return ready_umos.get(umo, True)

    async def derive(materials):
        return {item["material_id"]: [] for item in materials}

    path = tmp_path / "owner-wait-capacity.sqlite3"
    sidecar = EvidenceSidecar(path)
    service = EvidenceViewService(
        sidecar,
        scope_key,
        derive,
        queue_size=2,
        per_scope_hourly_limit=24,
        material_loader=load,
        owner_readiness=owner_ready,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        for scope, position, text in cases:
            await service.queue_material_derivation(
                [_source(scope, position, text)], scope=scope
            )

        await _wait_material_status(
            sidecar,
            descriptors_by_scope[scope_key(SCOPE_A)]["material_id"],
            SCOPE_A,
            "deferred",
        )
        await _wait_material_status(
            sidecar,
            descriptors_by_scope[scope_key(SCOPE_B)]["material_id"],
            SCOPE_B,
            "ready",
        )
        # Both personas for the ownerless UMO are paused out of the active cap.
        with sqlite3.connect(path) as db:
            owner_wait_rows = db.execute(
                "SELECT COUNT(*) FROM evidence_derivation_queue WHERE umo=? AND status='owner_wait'",
                (SCOPE_A["umo"],),
            ).fetchone()[0]
            owner_admissions = db.execute(
                "SELECT COUNT(*) FROM evidence_queue_admissions WHERE scope_key IN (?,?)",
                (scope_key(SCOPE_A), scope_key(scope_a2)),
            ).fetchone()[0]
        assert owner_wait_rows == 2
        assert owner_admissions == 0
        assert (
            await sidecar.material_retry_state(
                descriptors_by_scope[scope_key(SCOPE_B)]["material_id"],
                scope_key(SCOPE_B),
            )
        )["status"] == "ready"

        ready_umos[SCOPE_A["umo"]] = True
        assert await service.wake_policy_jobs(SCOPE_A) == 2
        for scope in (SCOPE_A, scope_a2):
            descriptor = descriptors_by_scope[scope_key(scope)]
            await _wait_material_status(
                sidecar, descriptor["material_id"], scope, "ready"
            )
        with sqlite3.connect(path) as db:
            owner_admissions = db.execute(
                "SELECT COUNT(*) FROM evidence_queue_admissions WHERE scope_key IN (?,?)",
                (scope_key(SCOPE_A), scope_key(scope_a2)),
            ).fetchone()[0]
        assert owner_admissions == 2
    finally:
        await service.close()


@pytest.mark.parametrize("capacity", [1, 128])
@pytest.mark.asyncio
async def test_queued_work_behind_deferred_head_runs_when_capacity_is_full(
    tmp_path, capacity
):
    path = tmp_path / f"head-of-line-{capacity}.sqlite3"
    sidecar = EvidenceSidecar(path)
    await sidecar.initialize()
    deferred, _ = _descriptor(SCOPE_A, 0, "fixture deferred head")
    queued = [
        _descriptor(SCOPE_A, index + 1, f"fixture queued {index}")[0]
        for index in range(capacity)
    ]
    try:
        await sidecar.upsert_material(deferred)
        assert (
            await sidecar.admit_material_job(deferred, now=1000.0, queue_size=capacity)
            == "queued"
        )
        lease = (
            await sidecar.claim_due_materials(
                1000.0,
                queue_size=capacity,
                per_scope_hourly_limit=24,
                lease_seconds=15,
                batch_size=1,
            )
        )[0]
        await sidecar.record_material_failure(
            deferred["material_id"],
            deferred["scope_key"],
            "timeout",
            lease["lease_id"],
            now=1000.0,
        )
        assert (
            await sidecar.material_retry_state(
                deferred["material_id"], deferred["scope_key"]
            )
        )["status"] == "failed"
        for descriptor in queued:
            await sidecar.upsert_material(descriptor)
            assert (
                await sidecar.admit_material_job(
                    descriptor, now=1001.0, queue_size=capacity
                )
                == "queued"
            )

        claimed = await sidecar.claim_due_materials(
            1301.0,
            queue_size=capacity,
            per_scope_hourly_limit=24,
            lease_seconds=15,
            batch_size=1,
        )
        assert len(claimed) == 1
        assert claimed[0]["material_id"] in {row["material_id"] for row in queued}
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_expired_lease_reclaims_idempotently_and_old_owner_cannot_store(tmp_path):
    path = tmp_path / "lease.sqlite3"
    sidecar = EvidenceSidecar(path)
    await sidecar.initialize()
    descriptor, _ = _descriptor(SCOPE_A, 0, "fixture lease source")
    try:
        await sidecar.upsert_material(descriptor)
        assert (
            await sidecar.admit_material_job(descriptor, now=1000.0, queue_size=8)
            == "queued"
        )
        first = await sidecar.claim_due_materials(
            1000.0,
            queue_size=8,
            per_scope_hourly_limit=24,
            lease_seconds=15,
            batch_size=1,
        )
        assert len(first) == 1
        assert (
            await sidecar.claim_due_materials(
                1014.0,
                queue_size=8,
                per_scope_hourly_limit=24,
                lease_seconds=15,
                batch_size=1,
            )
            == []
        )
        assert (
            await sidecar.renew_material_lease(
                descriptor["material_id"],
                scope_key(SCOPE_A),
                first[0]["lease_id"],
                1016.0,
                15,
            )
            is False
        )
        assert (
            await sidecar.store_atoms(descriptor, [], first[0]["lease_id"], now=1016.0)
            is False
        )
        second = await sidecar.claim_due_materials(
            1016.0,
            queue_size=8,
            per_scope_hourly_limit=24,
            lease_seconds=15,
            batch_size=1,
        )
        assert len(second) == 1
        assert second[0]["lease_id"] != first[0]["lease_id"]
        assert (
            await sidecar.renew_material_lease(
                descriptor["material_id"],
                scope_key(SCOPE_A),
                first[0]["lease_id"],
                1016.0,
                15,
            )
            is False
        )
        assert (
            await sidecar.store_atoms(descriptor, [], first[0]["lease_id"], now=1016.0)
            is False
        )
        assert (
            await sidecar.store_atoms(descriptor, [], second[0]["lease_id"], now=1016.0)
            is True
        )
        assert (
            await sidecar.claim_due_materials(
                2000.0,
                queue_size=8,
                per_scope_hourly_limit=24,
                lease_seconds=15,
                batch_size=1,
            )
            == []
        )
        with sqlite3.connect(path) as db:
            count = db.execute(
                "SELECT COUNT(*) FROM evidence_queue_admissions WHERE scope_key=?",
                (scope_key(SCOPE_A),),
            ).fetchone()[0]
        assert count == 2
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_shutdown_releases_lease_and_terminal_failures_do_not_retry(tmp_path):
    clock = FakeClock(40000.0)
    path = tmp_path / "terminal.sqlite3"
    descriptor, material = _descriptor(SCOPE_A, 0, "fixture shutdown source")
    started = asyncio.Event()

    async def load(job):
        return material

    async def blocked(materials):
        started.set()
        await asyncio.Event().wait()

    sidecar1 = EvidenceSidecar(path)
    service1 = EvidenceViewService(
        sidecar1,
        scope_key,
        blocked,
        material_loader=load,
        clock=clock,
        scheduler_poll_seconds=0.02,
    )
    await service1.initialize()
    await service1.queue_material_derivation(
        [_source(SCOPE_A, 0, "fixture shutdown source")], scope=SCOPE_A
    )
    await asyncio.wait_for(started.wait(), 1)
    await service1.close()

    sidecar2 = EvidenceSidecar(path)
    await sidecar2.initialize()
    try:
        state = await sidecar2.material_retry_state(
            descriptor["material_id"], scope_key(SCOPE_A)
        )
        assert state["attempt_count"] == 0
        reclaimed = await sidecar2.claim_due_materials(
            clock(),
            queue_size=128,
            per_scope_hourly_limit=24,
            lease_seconds=15,
            batch_size=1,
        )
        assert len(reclaimed) == 1
        await sidecar2.record_material_failure(
            descriptor["material_id"],
            scope_key(SCOPE_A),
            "policy_refusal",
            reclaimed[0]["lease_id"],
            now=clock(),
        )
        assert (
            await sidecar2.claim_due_materials(
                clock() + 99999,
                queue_size=128,
                per_scope_hourly_limit=24,
                lease_seconds=15,
                batch_size=1,
            )
            == []
        )
        assert (
            await sidecar2.material_retry_due(
                descriptor["material_id"], scope_key(SCOPE_A), now=clock() + 99999
            )
            is False
        )
        with sqlite3.connect(path) as db:
            row = db.execute(
                "SELECT outcome_code,attempt_count FROM evidence_materials WHERE material_id=?",
                (descriptor["material_id"],),
            ).fetchone()
        assert row == ("policy_refusal", 1)
    finally:
        await sidecar2.close()


@pytest.mark.asyncio
async def test_concurrent_close_joins_cancel_safe_lease_release(tmp_path):
    path = tmp_path / "close-barrier.sqlite3"
    descriptor, material = _descriptor(SCOPE_A, 0, "fixture close barrier")
    derive_started = asyncio.Event()
    release_started = asyncio.Event()
    allow_release = asyncio.Event()

    async def load(job):
        return material

    async def blocked(materials):
        derive_started.set()
        await asyncio.Event().wait()

    sidecar = EvidenceSidecar(path)
    original_release = sidecar.release_material_lease

    async def delayed_release(*args, **kwargs):
        release_started.set()
        await allow_release.wait()
        return await original_release(*args, **kwargs)

    sidecar.release_material_lease = delayed_release
    service = EvidenceViewService(
        sidecar, scope_key, blocked, material_loader=load, scheduler_poll_seconds=0.02
    )
    await service.initialize()
    await service.queue_material_derivation(
        [_source(SCOPE_A, 0, "fixture close barrier")], scope=SCOPE_A
    )
    await asyncio.wait_for(derive_started.wait(), 1)
    worker = service._worker
    close_one = asyncio.create_task(service.close())
    await asyncio.wait_for(release_started.wait(), 1)
    assert service._worker is worker
    close_two = asyncio.create_task(service.close())
    close_one.cancel()
    await asyncio.sleep(0.02)
    assert not close_one.done()
    close_one.cancel()
    await asyncio.sleep(0.02)
    assert not close_one.done()
    assert not close_two.done()
    allow_release.set()
    await asyncio.wait_for(close_two, 1)
    with pytest.raises(asyncio.CancelledError):
        await close_one
    assert service._worker is None
    assert sidecar._ready is False
    assert _queue_status(path, descriptor["material_id"])[0] == "queued"


@pytest.mark.asyncio
async def test_invalid_schema_retry_limit_and_queue_migration_are_idempotent(tmp_path):
    path = tmp_path / "migration.sqlite3"
    sidecar = EvidenceSidecar(path)
    await sidecar.initialize()
    descriptor, _ = _descriptor(SCOPE_A, 0, "fixture invalid schema")
    try:
        await sidecar.upsert_material(descriptor)
        assert (
            await sidecar.admit_material_job(descriptor, now=5000.0, queue_size=8)
            == "queued"
        )
        now = 5000.0
        for expected_attempt in (1, 2, 3):
            jobs = await sidecar.claim_due_materials(
                now,
                queue_size=8,
                per_scope_hourly_limit=24,
                lease_seconds=15,
                batch_size=1,
            )
            assert len(jobs) == 1
            await sidecar.record_material_failure(
                descriptor["material_id"],
                scope_key(SCOPE_A),
                "invalid_schema",
                jobs[0]["lease_id"],
                now=now,
            )
            state = await sidecar.material_retry_state(
                descriptor["material_id"], scope_key(SCOPE_A)
            )
            assert state["attempt_count"] == expected_attempt
            if expected_attempt < 3:
                now = state["next_attempt_at"]
        assert (
            await sidecar.material_retry_due(
                descriptor["material_id"], scope_key(SCOPE_A), now=now + 99999
            )
            is False
        )
        assert (
            await sidecar.claim_due_materials(
                now + 99999,
                queue_size=8,
                per_scope_hourly_limit=24,
                lease_seconds=15,
                batch_size=1,
            )
            == []
        )
        with sqlite3.connect(path) as db:
            assert (
                db.execute("SELECT COUNT(*) FROM evidence_derivation_queue").fetchone()[
                    0
                ]
                == 0
            )
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='evidence_queue_admissions'"
                ).fetchone()[0]
                == 1
            )
            version = db.execute(
                "SELECT schema_version FROM evidence_schema WHERE singleton=1"
            ).fetchone()[0]
        assert version == 3
    finally:
        await sidecar.close()

    reopened = EvidenceSidecar(path)
    await reopened.initialize()
    try:
        with sqlite3.connect(path) as db:
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM evidence_queue_cursor WHERE singleton=1"
                ).fetchone()[0]
                == 1
            )
            assert (
                db.execute(
                    "SELECT schema_version FROM evidence_schema WHERE singleton=1"
                ).fetchone()[0]
                == 3
            )
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_provider_unavailable_is_deferred_not_ready_without_spending_attempt(
    tmp_path,
):
    path = tmp_path / "provider-unavailable.sqlite3"
    sidecar = EvidenceSidecar(path)
    await sidecar.initialize()
    descriptor, _ = _descriptor(SCOPE_A, 0, "fixture unavailable provider")
    try:
        await sidecar.upsert_material(descriptor)
        assert (
            await sidecar.admit_material_job(descriptor, now=7000.0, queue_size=8)
            == "queued"
        )
        job = (
            await sidecar.claim_due_materials(
                7000.0, queue_size=8, per_scope_hourly_limit=24, lease_seconds=15
            )
        )[0]
        await sidecar.record_material_failure(
            descriptor["material_id"],
            scope_key(SCOPE_A),
            "provider_unavailable",
            job["lease_id"],
            now=7000.0,
        )
        state = await sidecar.material_retry_state(
            descriptor["material_id"], scope_key(SCOPE_A)
        )
        assert state["attempt_count"] == 0
        assert state["outcome_code"] == "provider_unavailable"
        assert _queue_status(path, descriptor["material_id"])[0] == "deferred"
        assert (
            await sidecar.claim_due_materials(
                7299.0, queue_size=8, per_scope_hourly_limit=24, lease_seconds=15
            )
            == []
        )
        due = await sidecar.claim_due_materials(
            7300.0, queue_size=8, per_scope_hourly_limit=24, lease_seconds=15
        )
        assert len(due) == 1
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_correction_during_lease_prevents_stale_atom_persistence(tmp_path):
    path = tmp_path / "revision-race.sqlite3"
    sidecar = EvidenceSidecar(path)
    await sidecar.initialize()
    descriptor, _ = _descriptor(SCOPE_A, 0, "fixture revision race")
    try:
        await sidecar.upsert_material(descriptor)
        await sidecar.admit_material_job(descriptor, now=8000.0, queue_size=8)
        job = (
            await sidecar.claim_due_materials(
                8000.0, queue_size=8, per_scope_hourly_limit=24, lease_seconds=15
            )
        )[0]
        correction = await sidecar.correct(
            descriptor["material_id"],
            expected_revision=0,
            action="withdraw",
            reason="synthetic revision race",
        )
        assert correction["correction_revision"] == 1
        atom = {
            "atom_id": "atm1:fixture-stale",
            "kind": "reported_claim",
            "claim": "fixture stale claim",
            "status": "reported_unverified",
            "root_evidence_ids": [],
            "root_material_ids": [descriptor["material_id"]],
            "source_refs": [],
        }
        assert await sidecar.store_atoms(descriptor, [atom], job["lease_id"]) is False
        assert (
            await sidecar.get_atoms_by_ids(scope_key(SCOPE_A), [atom["atom_id"]]) == {}
        )
        state = await sidecar.material_retry_state(
            descriptor["material_id"], scope_key(SCOPE_A)
        )
        assert state["status"] == "invalidated"
        assert _queue_status(path, descriptor["material_id"]) is None
    finally:
        await sidecar.close()
