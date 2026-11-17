from types import SimpleNamespace
import copy

import pytest

from astrbot_plugin_dynamics_memory.core.native_recall import (
    append_quote_surface,
    quote_frames,
    relevant_frames,
)


def _view(*, claim, status="verified_utterance", source_status="conversation_row"):
    ref = {
        "material_id": "mat1:private-row",
        "source_status": source_status,
        "source_kind": source_status,
        "row_hash": "rowhash" if source_status == "conversation_row" else None,
        "source_event_id": "evt-user-1" if source_status == "exact_event" else None,
        "text_hash": "sourcehash",
        "author_kind": "user",
        "author_id": "u1",
        "occurred_at": 1_700_000_000,
        "exact_span": [0, len(claim)],
    }
    return {
        "available": True,
        "atoms": [
            {
                "atom_id": "atm1:quote",
                "kind": "utterance_fact",
                "status": status,
                "source_status": source_status,
                "claim": claim,
                "source_refs": [ref],
            }
        ],
    }


def test_quote_frames_require_verified_author_and_source_locator():
    bundle = {
        "schema_version": 1,
        "source_views": [
            {
                "scope": {"umo": "qq:FriendMessage:u1"},
                "native_materials": [{"material_id": "mat1:private-row"}],
                "evidence_view": _view(claim="我喜欢桂花乌龙茶"),
            }
        ],
    }
    assert quote_frames(bundle)[0]["author_uid"] == "u1"
    assert quote_frames(bundle)[0]["scope_umo"] == "qq:FriendMessage:u1"

    bundle["source_views"][0]["evidence_view"] = _view(
        claim="我喜欢桂花乌龙茶", status="reported_unverified"
    )
    assert quote_frames(bundle) == []
    bundle["source_views"][0]["evidence_view"] = _view(claim="我喜欢桂花乌龙茶")
    bundle["source_views"][0]["evidence_view"]["atoms"][0]["source_refs"][0][
        "author_id"
    ] = ""
    assert quote_frames(bundle) == []


def test_canonical_private_quotes_match_plain_chinese_questions():
    frames = [{"atom_id": "a1", "text": "我喜欢桂花乌龙茶"}]
    assert relevant_frames(frames, "我喜欢什么", ["喜欢"]) == frames
    assert relevant_frames(frames, "我喜欢什么") == frames
    semantic_hit = [
        {
            "atom_id": "a2",
            "text": "我会弹钢琴",
            "retrieval_hit": True,
            "retrieval_rank": 0,
        }
    ]
    assert relevant_frames(semantic_hit, "有什么爱好", ["爱好"]) == semantic_hit


def test_exact_event_quote_requires_a_matching_archive_span_proof():
    bundle = {
        "schema_version": 1,
        "source_views": [
            {
                "scope": {"umo": "qq:GroupMessage:g1"},
                "native_materials": [],
                "evidence_view": _view(
                    claim="The user likes tea", source_status="exact_event"
                ),
                "native_events": [],
            }
        ],
    }
    assert quote_frames(bundle) == []
    source_ref = bundle["source_views"][0]["evidence_view"]["atoms"][0]["source_refs"][
        0
    ]
    bundle["source_views"][0]["native_events"] = [
        {
            "kind": "event",
            "event_id": source_ref["source_event_id"],
            "actor_id": "u1",
            "text_hash": source_ref["text_hash"],
            "exact_span": [0, 17],
            "atom_id": "atm1:quote",
        }
    ]
    assert quote_frames(bundle) == []
    bundle["source_views"][0]["native_events"][0]["exact_span"] = [
        0,
        len("The user likes tea"),
    ]
    assert quote_frames(bundle)[0]["text"] == "The user likes tea"


def test_quote_surface_is_temporary_and_keeps_source_identity():
    req = SimpleNamespace(extra_user_content_parts=[])
    frames = [
        {
            "scope_umo": "qq:GroupMessage:g1",
            "author_uid": "u1",
            "occurred_at": 1_700_000_000,
            "source_kind": "conversation_row",
            "text": "群里提过这家茶店",
        }
    ]
    assert append_quote_surface(req, frames) is True
    assert len(req.extra_user_content_parts) == 1
    part = req.extra_user_content_parts[0]
    assert part._no_save is True
    assert "qq:GroupMessage:g1" in part.text
    assert "u1" in part.text
    assert "conversation_row" in part.text
    assert "群里提过这家茶店" in part.text


@pytest.mark.asyncio
async def test_default_row_material_policy_still_checks_the_blocking_gate():
    from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
    from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin

    scope = {
        "platform": "qq",
        "umo": "qq:FriendMessage:u1",
        "persona_id": "anan",
        "instance_id": "anan",
    }
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    checked = []

    class Gate:
        async def check(self, descriptor, material):
            checked.append((descriptor, material))
            return False

    async def allow_source_scope(_scope, _actor=""):
        return True

    plugin._blocking_policy_gate = Gate()
    plugin._narrative_scope_allowed = allow_source_scope
    descriptor = {
        "scope_key": scope_key(scope),
        "material_kind": "conversation_row",
        "source_locator": {
            "scope": scope,
            "kind": "conversation_row",
            "row_hash": "hash",
        },
    }
    material = {"source_status": "conversation_row", "role": "user", "author_id": "u2"}
    assert await plugin._evidence_material_policy(descriptor, material) is False
    assert checked == [(descriptor, material)]


def _thin_ref(row):
    import hashlib

    from astrbot_plugin_dynamics_memory.core.evidence_projection import (
        conversation_row_hash,
    )

    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "role": row["role"],
        "sender_id": row["sender_id"],
        "platform": row["platform"],
        "timestamp": float(row["timestamp"]),
        "content_sha256": hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
        "row_hash": conversation_row_hash(row),
    }


def _native_source_loader(raw_row, *, metadata_extra=None, snapshots=None):
    from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin

    session_id = "qq:FriendMessage:u1"
    scope = {
        "platform": "qq",
        "umo": session_id,
        "persona_id": "anan",
        "instance_id": "anan",
    }
    metadata = {
        "status": "active",
        "session_id": session_id,
        "source_session_id": session_id,
        "interaction_type": "private_chat",
        "persona_id": "anan",
    }
    metadata.update(metadata_extra or {})

    class Engine:
        async def get_memory(self, _memory_id):
            return {"id": 101, "text": "source summary", "metadata": dict(metadata)}

        async def get_memory_source(self, _memory_id):
            return copy.deepcopy(snapshots or [])

    class Store:
        async def get_message_by_id(self, _session_id, _row_id):
            return copy.deepcopy(raw_row)

    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.initializer = SimpleNamespace(
        memory_engine=Engine(),
        conversation_manager=SimpleNamespace(store=Store()),
    )
    plugin.config_manager = SimpleNamespace(
        filtering_settings={"use_persona_filtering": True}
    )

    async def allow_scope(_scope, _actor=""):
        return True

    plugin._narrative_scope_allowed = allow_scope
    return plugin, scope


def _source_row():
    return {
        "id": 17,
        "session_id": "qq:FriendMessage:u1",
        "role": "user",
        "content": "我喜欢桂花乌龙茶",
        "sender_id": "u1",
        "platform": "qq",
        "timestamp": 1_700_000_100.0,
    }


@pytest.mark.asyncio
async def test_thin_source_refs_read_exact_row_when_full_snapshot_is_absent():
    row = _source_row()
    plugin, scope = _native_source_loader(
        row, metadata_extra={"source_row_refs_v1": [_thin_ref(row)]}, snapshots=[]
    )
    assert (
        await plugin._native_verified_source_row(
            scope, 101, row["id"], persona_id="anan", persona_filtering=True
        )
        == row
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["content", "sender_id", "timestamp", "deleted"])
async def test_thin_source_refs_reject_changed_or_missing_raw_rows(tamper):
    row = _source_row()
    changed = copy.deepcopy(row)
    if tamper == "content":
        changed["content"] = "我喜欢别的饮料"
    elif tamper == "sender_id":
        changed["sender_id"] = "u2"
    elif tamper == "timestamp":
        changed["timestamp"] += 1
    elif tamper == "deleted":
        changed = None
    plugin, scope = _native_source_loader(
        changed, metadata_extra={"source_row_refs_v1": [_thin_ref(row)]}, snapshots=[]
    )
    assert (
        await plugin._native_verified_source_row(
            scope, 101, row["id"], persona_id="anan", persona_filtering=True
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("hash_field", ["content_sha256", "row_hash"])
async def test_thin_source_refs_reject_wrong_hashes(hash_field):
    row = _source_row()
    reference = _thin_ref(row)
    reference[hash_field] = (
        "0" * 64 if hash_field == "content_sha256" else "row1:" + ("0" * 64)
    )
    plugin, scope = _native_source_loader(
        row, metadata_extra={"source_row_refs_v1": [reference]}, snapshots=[]
    )
    assert (
        await plugin._native_verified_source_row(
            scope, 101, row["id"], persona_id="anan", persona_filtering=True
        )
        is None
    )


@pytest.mark.asyncio
async def test_bad_full_snapshot_never_falls_back_to_valid_thin_refs():
    row = _source_row()
    plugin, scope = _native_source_loader(
        row,
        metadata_extra={
            "has_source": True,
            "source_message_count": 1,
            "source_row_refs_v1": [_thin_ref(row)],
        },
        snapshots=[],
    )
    assert (
        await plugin._native_verified_source_row(
            scope, 101, row["id"], persona_id="anan", persona_filtering=True
        )
        is None
    )


@pytest.mark.asyncio
async def test_full_snapshot_missing_requested_row_never_falls_back_to_thin_ref():
    row = _source_row()
    other = dict(row, id=18)
    plugin, scope = _native_source_loader(
        row,
        metadata_extra={
            "has_source": True,
            "source_message_count": 1,
            "source_row_refs_v1": [_thin_ref(row)],
        },
        snapshots=[other],
    )
    assert (
        await plugin._native_verified_source_row(
            scope, 101, row["id"], persona_id="anan", persona_filtering=True
        )
        is None
    )
