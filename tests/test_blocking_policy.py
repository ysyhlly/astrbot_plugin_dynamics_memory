from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_memory.core.blocking_policy import BlockingPolicyGate
from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    normalize_event_material,
    normalize_host_material,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar


SCOPE = {
    "platform": "qq",
    "umo": "qq-bot-1:GroupMessage:123",
    "persona_id": "persona-A",
    "instance_id": "persona-A",
}
FILTER_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "autonomous_blocking_filters.py"
)
SPEC = importlib.util.spec_from_file_location(
    "actual_autonomous_blocking_filters", FILTER_PATH
)
assert SPEC and SPEC.loader
ACTUAL_FILTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTUAL_FILTER)


class FakeBlockingService:
    enabled = True

    def __init__(
        self, rules=(), *, ready=True, owner=None, owner_platform=None, context_umo=None
    ):
        self.rules = list(rules)
        self.ready = ready
        self.owner_override = owner
        self.owner_platform = owner_platform
        self.context_umo = context_umo
        self.resolve_calls = []

    def resolve_context(self, *, event, scope):
        # Mirror the public service contract: scope alone cannot manufacture an
        # owner, and valid ownership is read through the supplied event getters.
        self.resolve_calls.append((event, dict(scope)))
        if not self.ready or event is None:
            return {"ready": False, "owner": {"self_id": ""}}
        owner_id = event.get_self_id()
        if self.owner_override is not None:
            owner_id = self.owner_override
        platform_id = self.owner_platform or event.get_platform_id()
        return {
            "ready": True,
            "owner": {
                "self_id": owner_id,
                "platform": event.get_platform_name(),
                "platform_instance_id": platform_id,
            },
            "umo": self.context_umo or event.unified_msg_origin,
            "scope": dict(scope),
        }

    def policy_snapshot(self, context):
        return {"rules": list(self.rules)}

    def filter_material(self, context, bundle, purpose):
        return ACTUAL_FILTER.filter_material(self, context, bundle, purpose)


class FakeHostContext:
    def __init__(self, service=None):
        self.service = service
        self.platform_requests = []
        self.platform = SimpleNamespace(
            meta=lambda: SimpleNamespace(id="qq-bot-1", name="qq"),
            status=SimpleNamespace(value="running"),
            get_self_id=lambda: "bot-42",
        )
        self.platforms = {"qq-bot-1": self.platform}

    def get_all_stars(self):
        if self.service is None:
            return {}
        return {
            "astrbot_plugin_autonomous_blocking": {
                "activated": True,
                "star_cls": self.service,
            }
        }

    def get_platform_inst(self, platform_id):
        self.platform_requests.append(platform_id)
        return self.platforms.get(platform_id)


class FakeAstrMessageEvent:
    """Minimal shape of the trusted AstrMessageEvent public getters."""

    def __init__(
        self,
        *,
        platform_id="qq-bot-1",
        self_id="bot-42",
        platform_name="qq",
        umo=None,
        body="trusted event must not be cached",
    ):
        self._platform_id = platform_id
        self._self_id = self_id
        self._platform_name = platform_name
        self.unified_msg_origin = umo or SCOPE["umo"]
        self.message_body = body
        self.message_obj = SimpleNamespace(self_id=self_id)
        self.platform_meta = SimpleNamespace(id=platform_id, name=platform_name)

    def get_platform_id(self):
        return self._platform_id

    def get_self_id(self):
        return self._self_id

    def get_platform_name(self):
        return self._platform_name


def _event(actor="fixture-user", event_id="fixture-event", scope=SCOPE):
    return {
        "source_kind": "event",
        "event_id": event_id,
        "event_type": "message.received",
        "scope": dict(scope),
        "text": "fixture event source",
        "actor_id": actor,
        "occurred_at": 1234,
    }


_NO_EVENT = object()


def _observe(gate, scope=SCOPE, event=_NO_EVENT):
    return gate.observe_trusted_event(
        FakeAstrMessageEvent() if event is _NO_EVENT else event, scope
    )


def _scope(umo, *, persona="persona-A"):
    platform = "wx" if umo.startswith("wx-") else "qq"
    return {
        "platform": platform,
        "umo": umo,
        "persona_id": persona,
        "instance_id": persona,
    }


def _host_source():
    source = {
        "source_kind": "host_history",
        "conversation_id": "fixture-conversation",
        "snapshot_fingerprint": "fixture-fingerprint",
        "position": 0,
        "role": "user",
        "text": "fixture unknown speaker source",
        "source_prefix_anchor": "fixture-prefix",
        "history_epoch": "fixture-epoch",
    }
    return normalize_host_material(
        scope_key(SCOPE),
        source,
        conversation_id=source["conversation_id"],
        snapshot_fingerprint=source["snapshot_fingerprint"],
    )


@pytest.mark.asyncio
async def test_actual_blocking_filter_blocks_unknown_host_before_derivation(tmp_path):
    blocker = FakeBlockingService(
        [{"target_uid": "shielded-user", "mode": "shield", "active": True}]
    )
    host = FakeHostContext(blocker)
    gate = BlockingPolicyGate(host)
    assert await _observe(gate) is True
    descriptor, _ = _host_source()
    derived = []

    async def derive(materials):
        derived.extend(materials)
        return {}

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "blocked-host.sqlite3"),
        scope_key,
        derive,
        material_policy=gate.check,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        await service.queue_material_derivation([_host_source()[1]], scope=SCOPE)
        state = await _wait_state(service, descriptor["material_id"], "failed")
        assert state["outcome_code"] == "policy_refusal"
        assert derived == []
        assert host.platform_requests and set(host.platform_requests) == {"qq-bot-1"}
    finally:
        await service.close()


async def _wait_state(service, material_id, wanted, timeout=2.0, scope=SCOPE):
    loop = asyncio.get_running_loop()
    until = loop.time() + timeout
    while loop.time() < until:
        state = await service.sidecar.material_retry_state(
            material_id, scope_key(scope)
        )
        if state["status"] == wanted:
            return state
        await asyncio.sleep(0.01)
    raise AssertionError(f"material did not reach {wanted}")


def _admission_count(sidecar, scope, lower, upper):
    with sidecar._lock, sidecar._session() as db:
        return int(
            db.execute(
                "SELECT COUNT(*) FROM evidence_queue_admissions WHERE scope_key=? AND admitted_at>? AND admitted_at<=?",
                (scope_key(scope), lower, upper),
            ).fetchone()[0]
        )


def _stored_policy_state(sidecar, material_id, scope):
    with sidecar._lock, sidecar._session() as db:
        row = db.execute(
            "SELECT m.diagnostics_code,q.status FROM evidence_materials m "
            "JOIN evidence_derivation_queue q USING(material_id,scope_key) "
            "WHERE m.material_id=? AND m.scope_key=?",
            (material_id, scope_key(scope)),
        ).fetchone()
    return (str(row[0]), str(row[1])) if row else ("", "missing")


@pytest.mark.asyncio
async def test_actual_blocking_filter_allows_unshielded_archived_actor(tmp_path):
    blocker = FakeBlockingService(
        [{"target_uid": "other-user", "mode": "shield", "active": True}]
    )
    host = FakeHostContext(blocker)
    gate = BlockingPolicyGate(host)
    assert await _observe(gate) is True
    event = _event()
    descriptor, _ = normalize_event_material(scope_key(SCOPE), SCOPE, event)
    calls = []

    async def load(job):
        return event

    async def derive(materials):
        calls.extend(materials)
        return {
            item["material_id"]: [
                {
                    "kind": "reported_claim",
                    "claim": "fixture allowed claim",
                    "span_start": 0,
                    "span_end": len(item["text"]),
                }
            ]
            for item in materials
        }

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "allowed-event.sqlite3"),
        scope_key,
        derive,
        material_loader=load,
        material_policy=gate.check,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        await service.queue_material_derivation([event], scope=SCOPE)
        state = await _wait_state(service, descriptor["material_id"], "ready")
        assert state["status"] == "ready"
        assert len(calls) == 1
        cache = await service.sidecar.get_materials(
            scope_key(SCOPE), [descriptor["material_id"]]
        )
        atom = cache[descriptor["material_id"]]["atoms"][0]["atom"]
        assert atom["source_refs"][0]["author_id"] == "fixture-user"
        assert host.platform_requests and set(host.platform_requests) == {"qq-bot-1"}
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_blocking_policy_change_during_derivation_prevents_commit(tmp_path):
    blocker = FakeBlockingService([])
    host = FakeHostContext(blocker)
    gate = BlockingPolicyGate(host)
    assert await _observe(gate) is True
    event = _event()
    descriptor, _ = normalize_event_material(scope_key(SCOPE), SCOPE, event)
    started = asyncio.Event()

    async def load(job):
        return event

    async def derive(materials):
        started.set()
        blocker.rules = [
            {"target_uid": "fixture-user", "mode": "shield", "active": True}
        ]
        return {
            item["material_id"]: [
                {
                    "kind": "reported_claim",
                    "claim": "fixture should be blocked after derivation",
                    "span_start": 0,
                    "span_end": len(item["text"]),
                }
            ]
            for item in materials
        }

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "policy-change.sqlite3"),
        scope_key,
        derive,
        material_loader=load,
        material_policy=gate.check,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        await service.queue_material_derivation([event], scope=SCOPE)
        await asyncio.wait_for(started.wait(), 1)
        state = await _wait_state(service, descriptor["material_id"], "failed")
        assert state["outcome_code"] == "policy_refusal"
        cached = await service.sidecar.get_materials(
            scope_key(SCOPE), [descriptor["material_id"]]
        )
        assert cached[descriptor["material_id"]].get("atoms", []) == []
        assert host.platform_requests and set(host.platform_requests) == {"qq-bot-1"}
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_blocking_unavailable_states_defer_but_absent_or_disabled_is_optional(
    tmp_path,
):
    event = _event()
    descriptor, material = normalize_event_material(scope_key(SCOPE), SCOPE, event)
    not_ready = BlockingPolicyGate(FakeHostContext(FakeBlockingService(ready=False)))
    missing_owner = BlockingPolicyGate(FakeHostContext(FakeBlockingService(owner="")))
    absent = BlockingPolicyGate(FakeHostContext())
    disabled_service = FakeBlockingService()
    disabled_service.enabled = False
    disabled = BlockingPolicyGate(FakeHostContext(disabled_service))

    assert (
        FakeBlockingService().resolve_context(event=None, scope=SCOPE)["ready"] is False
    )
    assert (
        await _observe(
            BlockingPolicyGate(FakeHostContext(FakeBlockingService())), event=None
        )
        is None
    )
    assert await absent.check(descriptor, material) is True
    assert await disabled.check(descriptor, material) is True
    assert await not_ready.check(descriptor, material) is None
    assert await missing_owner.check(descriptor, material) is None

    calls = []

    async def derive(materials):
        calls.extend(materials)
        return {}

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "policy-unavailable.sqlite3"),
        scope_key,
        derive,
        material_policy=not_ready.check,
        scheduler_poll_seconds=0.02,
    )
    await service.initialize()
    try:
        await service.get_evidence_view(SCOPE, source_refs=[event])
        state = await _wait_state(service, descriptor["material_id"], "deferred")
        assert state["outcome_code"] == "policy_unavailable"
        assert state["attempt_count"] == 0
        assert calls == []
        view = await service.get_evidence_view(SCOPE, source_refs=[event])
        assert view["evidence_status"] in {"deferred", "pending"}
        assert view["diagnostics"]["policy_unavailable"] == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_restart_defers_until_real_event_then_wakes_and_resumes_without_requery(
    tmp_path,
):
    blocker = FakeBlockingService([])
    host = FakeHostContext(blocker)
    # A newly constructed gate represents process restart: owner context is volatile.
    gate = BlockingPolicyGate(host)
    event_a = _event("actor-A", "event-A")
    descriptor_a, _ = normalize_event_material(scope_key(SCOPE), SCOPE, event_a)
    scope_b = _scope("qq-bot-1:GroupMessage:456")
    event_b = _event("actor-B", "event-B", scope_b)
    descriptor_b, _ = normalize_event_material(scope_key(scope_b), scope_b, event_b)
    derive_calls = []
    fake_now = [1_000.0]
    events = {
        descriptor_a["material_id"]: event_a,
        descriptor_b["material_id"]: event_b,
    }

    async def load(job):
        return events.get(job.get("material_id"))

    async def derive(materials):
        derive_calls.extend(materials)
        return {
            item["material_id"]: [
                {
                    "kind": "reported_claim",
                    "claim": "fixture post-resume claim",
                    "span_start": 0,
                    "span_end": len(item["text"]),
                }
            ]
            for item in materials
        }

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "owner-resume.sqlite3"),
        scope_key,
        derive,
        queue_size=1,
        per_scope_hourly_limit=1,
        material_loader=load,
        material_policy=gate.check,
        owner_readiness=gate.owner_ready,
        clock=lambda: fake_now[0],
        scheduler_poll_seconds=0.05,
    )
    await service.initialize()
    gate.wakeup = service.wake_policy_jobs
    try:
        queued = await service.queue_material_derivation([event_a], scope=SCOPE)
        assert queued["accepted"] is True
        assert queued["queued"] + queued["deferred"] == 1
        waiting = await _wait_state(service, descriptor_a["material_id"], "deferred")
        assert _stored_policy_state(
            service.sidecar, descriptor_a["material_id"], SCOPE
        ) == (
            "policy_unavailable",
            "owner_wait",
        )
        assert waiting["attempt_count"] == 0
        assert derive_calls == []
        assert _admission_count(service.sidecar, SCOPE, 0, 3_000_000_000) == 0

        # With global queue capacity one, the owner-wait row must not prevent
        # a different scope that already has a trusted owner from progressing.
        event_b_trusted = FakeAstrMessageEvent(umo=scope_b["umo"])
        assert await gate.observe_trusted_event(event_b_trusted, scope_b) is True
        queued_b = await service.queue_material_derivation([event_b], scope=scope_b)
        assert queued_b["accepted"] is True
        assert queued_b["queued"] + queued_b["deferred"] == 1
        ready_b = await _wait_state(
            service, descriptor_b["material_id"], "ready", scope=scope_b
        )
        assert ready_b["status"] == "ready"
        assert len(derive_calls) == 1
        assert _admission_count(service.sidecar, SCOPE, 0, 3_000_000_000) == 0
        assert _admission_count(service.sidecar, scope_b, 0, 3_000_000_000) == 1

        trusted = FakeAstrMessageEvent(umo=SCOPE["umo"])
        assert await gate.observe_trusted_event(trusted, SCOPE) is True
        ready = await _wait_state(service, descriptor_a["material_id"], "ready")
        assert ready["status"] == "ready"
        assert len(derive_calls) == 2
        assert _admission_count(service.sidecar, SCOPE, 0, 3_000_000_000) == 1
        assert len(blocker.resolve_calls) == 2
        assert {call[0] for call in blocker.resolve_calls} == {event_b_trusted, trusted}
        assert len(host.platform_requests) >= 4
        assert set(host.platform_requests) == {"qq-bot-1"}
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_trusted_owner_cache_is_umo_scoped_and_contains_no_event_body(tmp_path):
    blocker = FakeBlockingService([])
    host = FakeHostContext(blocker)
    gate = BlockingPolicyGate(host)
    group_scope = _scope("qq-bot-1:GroupMessage:123")
    other_group = _scope("qq-bot-1:GroupMessage:456")
    private_scope = _scope("qq-bot-1:FriendMessage:user-7")
    other_persona_same_umo = _scope(group_scope["umo"], persona="persona-B")

    assert (
        await gate.observe_trusted_event(
            FakeAstrMessageEvent(umo=group_scope["umo"]), group_scope
        )
        is True
    )
    host_material_event = _event(scope=other_group)
    other_desc, other_material = normalize_event_material(
        scope_key(other_group), other_group, host_material_event
    )
    assert await gate.check(other_desc, other_material) is None
    assert (
        await gate.check(
            *normalize_event_material(
                scope_key(private_scope), private_scope, _event(scope=private_scope)
            )
        )
        is None
    )

    # Owner identity may be shared inside one UMO, while the evidence source
    # remains bound to a separate persona/scope key.
    same_umo_desc, same_umo_material = normalize_event_material(
        scope_key(other_persona_same_umo),
        other_persona_same_umo,
        _event(scope=other_persona_same_umo),
    )
    assert await gate.check(same_umo_desc, same_umo_material) is True

    assert (
        await gate.observe_trusted_event(
            FakeAstrMessageEvent(umo=other_group["umo"]), other_group
        )
        is True
    )
    assert (
        await gate.observe_trusted_event(
            FakeAstrMessageEvent(umo=private_scope["umo"]), private_scope
        )
        is True
    )
    for scope in (group_scope, other_group, private_scope):
        desc, material = normalize_event_material(
            scope_key(scope), scope, _event(scope=scope)
        )
        assert await gate.check(desc, material) is True

    assert len(gate._owner_contexts) == 3
    serialized_cache = repr(gate._owner_contexts)
    assert "trusted event must not be cached" not in serialized_cache
    assert "fixture event source" not in serialized_cache
    assert "fixture-user" not in serialized_cache
    assert {key[1] for key in gate._owner_contexts} == {
        group_scope["umo"],
        other_group["umo"],
        private_scope["umo"],
    }
    assert len({key[1] for key in gate._owner_contexts}) == 3

    # Platform object replacement invalidates the cached owner context until a
    # subsequent genuine event verifies the current platform instance.
    host.platform = SimpleNamespace(
        meta=lambda: SimpleNamespace(id="qq-bot-1", name="qq"),
        status=SimpleNamespace(value="running"),
        get_self_id=lambda: "bot-42",
    )
    host.platforms["qq-bot-1"] = host.platform
    assert await gate.owner_ready(group_scope["umo"]) is None


@pytest.mark.asyncio
async def test_trusted_event_rejects_platform_or_umo_mismatch_and_persona_is_not_platform():
    blocker = FakeBlockingService([])
    host = FakeHostContext(blocker)
    gate = BlockingPolicyGate(host)

    assert SCOPE["instance_id"] == "persona-A"
    assert SCOPE["umo"].split(":", 1)[0] == "qq-bot-1"
    assert (
        await gate.observe_trusted_event(
            FakeAstrMessageEvent(platform_id="persona-A", umo=SCOPE["umo"]), SCOPE
        )
        is None
    )
    assert (
        await gate.observe_trusted_event(
            FakeAstrMessageEvent(
                platform_id="qq-bot-1", umo="qq-bot-1:FriendMessage:other"
            ),
            SCOPE,
        )
        is None
    )
    assert await gate.observe_trusted_event(None, SCOPE) is None
    assert gate._owner_contexts == {}
    assert blocker.resolve_calls == []
    assert host.platform_requests == []

    mismatched_service = FakeBlockingService(
        [], context_umo="qq-bot-1:FriendMessage:other"
    )
    mismatched_gate = BlockingPolicyGate(FakeHostContext(mismatched_service))
    assert (
        await _observe(mismatched_gate, event=FakeAstrMessageEvent(umo=SCOPE["umo"]))
        is None
    )
    assert mismatched_gate._owner_contexts == {}

    wrong_owner_service = FakeBlockingService([], owner_platform="persona-A")
    wrong_owner_gate = BlockingPolicyGate(FakeHostContext(wrong_owner_service))
    assert await _observe(wrong_owner_gate) is None
    assert wrong_owner_gate._owner_contexts == {}


@pytest.mark.asyncio
async def test_bot_self_id_change_invalidates_other_umos_on_same_platform_only():
    blocker = FakeBlockingService([])
    host = FakeHostContext(blocker)
    host.platforms["wx-bot-1"] = SimpleNamespace(
        meta=lambda: SimpleNamespace(id="wx-bot-1", name="wx"),
        status=SimpleNamespace(value="running"),
        get_self_id=lambda: "wx-self-7",
    )
    gate = BlockingPolicyGate(host)
    scope_a = _scope("qq-bot-1:GroupMessage:A")
    scope_b = _scope("qq-bot-1:GroupMessage:B")
    scope_other_platform = _scope("wx-bot-1:GroupMessage:C")

    assert (
        await gate.observe_trusted_event(
            FakeAstrMessageEvent(umo=scope_a["umo"]), scope_a
        )
        is True
    )
    assert (
        await gate.observe_trusted_event(
            FakeAstrMessageEvent(umo=scope_b["umo"]), scope_b
        )
        is True
    )
    assert (
        await gate.observe_trusted_event(
            FakeAstrMessageEvent(
                platform_id="wx-bot-1",
                self_id="wx-self-7",
                platform_name="wx",
                umo=scope_other_platform["umo"],
            ),
            scope_other_platform,
        )
        is True
    )
    assert await gate.owner_ready(scope_b["umo"]) is True
    assert await gate.owner_ready(scope_other_platform["umo"]) is True

    changed = FakeAstrMessageEvent(umo=scope_a["umo"], self_id="bot-43")
    assert await gate.observe_trusted_event(changed, scope_a) is True
    assert await gate.owner_ready(scope_a["umo"]) is True
    assert await gate.owner_ready(scope_b["umo"]) is None
    assert await gate.owner_ready(scope_other_platform["umo"]) is True
