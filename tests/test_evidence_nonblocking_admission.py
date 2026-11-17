import asyncio
import threading

import pytest

from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key

SCOPE = {
    "platform": "qq",
    "umo": "qq:GroupMessage:test",
    "persona_id": "test",
    "instance_id": "test",
}


@pytest.mark.asyncio
async def test_cached_read_does_not_wait_for_admission_and_close_cancels_it(tmp_path):
    async def derive(*args):
        return {}

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "evidence.sqlite3"), scope_key, derive
    )
    await service.initialize()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def blocked(materials, **kwargs):
        entered.set()
        calls.append(materials)
        await release.wait()
        return {"queued": 1}

    service.queue_material_derivation = blocked
    raw = {
        "source_kind": "host_history",
        "role": "user",
        "text": "我之前说过想换显卡",
        "position": 0,
        "conversation_id": "conversation",
        "snapshot_fingerprint": "snapshot",
    }
    try:
        view = await asyncio.wait_for(
            service.get_evidence_view(SCOPE, [raw], card_version="card"), 0.5
        )
        assert view["available"]
        assert view["diagnostics"]["materials_admission_pending"] == 1
        await asyncio.wait_for(entered.wait(), 0.5)
        assert not release.is_set()
        again = await service.get_evidence_view(SCOPE, [raw], card_version="card")
        assert again["diagnostics"]["materials_admission_pending"] == 1
        assert len(calls) == 1
        assert service.admission_diagnostics()["coalesced"] == 1
    finally:
        await service.close()
    assert service._admission_worker is None and not service._admissions


@pytest.mark.asyncio
async def test_read_only_mode_never_admits(tmp_path):
    async def derive(*args):
        return {}

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "evidence.sqlite3"), scope_key, derive
    )
    await service.initialize()
    raw = {
        "source_kind": "host_history",
        "role": "user",
        "text": "历史资料",
        "position": 0,
        "conversation_id": "conversation",
        "snapshot_fingerprint": "snapshot",
    }
    try:
        view = await service.get_evidence_view(
            SCOPE, [raw], card_version="card", allow_derivation=False
        )
        assert view["diagnostics"]["materials_admission_pending"] == 0
        assert not service._admissions
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_writer_contention_does_not_block_evidence_reads(tmp_path):
    async def derive(*args):
        return {}

    sidecar = EvidenceSidecar(tmp_path / "evidence.sqlite3")
    service = EvidenceViewService(sidecar, scope_key, derive)
    await service.initialize()
    entered, release = threading.Event(), threading.Event()

    def writer():
        with sidecar._lock:
            entered.set()
            release.wait()

    thread = threading.Thread(target=writer)
    thread.start()
    raw = {
        "source_kind": "host_history",
        "role": "user",
        "text": "我之前说过想换显卡",
        "position": 0,
        "conversation_id": "conversation",
        "snapshot_fingerprint": "snapshot",
    }
    try:
        await asyncio.to_thread(entered.wait)
        view = await asyncio.wait_for(
            service.get_evidence_view(SCOPE, [raw], card_version="card"), 0.5
        )
        assert view["available"] and not release.is_set()
        assert view["diagnostics"]["materials_admission_pending"] == 1
    finally:
        release.set()
        await asyncio.to_thread(thread.join)
        await service.close()


@pytest.mark.asyncio
async def test_ready_source_atoms_are_still_returned(tmp_path):
    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "evidence.sqlite3"), scope_key
    )
    await service.initialize()
    raw = {
        "source_kind": "host_history",
        "role": "user",
        "text": "我想换一张显卡",
        "position": 0,
        "conversation_id": "conversation",
        "snapshot_fingerprint": "snapshot",
    }
    try:
        desc, material = service._normalize(raw, SCOPE, "card", "reply")
        await service.sidecar.upsert_material(desc)
        atom = {
            "atom_id": "a1",
            "kind": "claim",
            "claim": "用户说想换显卡",
            "status": "reported_unverified",
            "root_evidence_ids": [desc["material_id"]],
            "root_material_ids": [],
            "source_refs": [],
            "subject_ids": [],
            "source_status": "host_record_only",
        }
        assert await service.sidecar.store_atoms(desc, [atom])
        view = await service.get_evidence_view(SCOPE, [raw], card_version="card")
        assert [a["atom_id"] for a in view["atoms"]] == ["a1"]
        assert view["diagnostics"]["materials_admission_pending"] == 0
        assert not service._admissions
        row = (await service.sidecar.get_atoms_by_ids(scope_key(SCOPE), ["a1"]))["a1"]
        await service.sidecar.correct("a1", row["revision"], "withdraw", reason="test")
        corrected = await service.get_evidence_view(SCOPE, [raw], card_version="card")
        assert not corrected["atoms"], "fast reads must still honor corrections"
        other = await service.get_evidence_view(
            dict(SCOPE, umo="qq:GroupMessage:other"), [raw], card_version="card"
        )
        assert not other["atoms"], "ready atoms cannot cross scope"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_different_card_versions_do_not_share_pending_admission(tmp_path):
    async def derive(*args):
        return {}

    service = EvidenceViewService(
        EvidenceSidecar(tmp_path / "evidence.sqlite3"), scope_key, derive
    )
    await service.initialize()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        return {"queued": 0}

    service.queue_material_derivation = blocked
    raw = {
        "source_kind": "host_history",
        "role": "user",
        "text": "同一条历史消息",
        "position": 0,
        "conversation_id": "conversation",
        "snapshot_fingerprint": "snapshot",
    }
    try:
        await service.get_evidence_view(SCOPE, [raw], card_version="card1")
        await entered.wait()
        await service.get_evidence_view(SCOPE, [raw], card_version="card2")
        assert len(service._admissions) == 2
    finally:
        await service.close()
