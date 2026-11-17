"""Deferred host sources are bound to one immutable history snapshot."""

import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from tests.integration.test_public_evidence_pipeline import pipeline as pipeline, SCOPE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "exact source",
        [
            {"type": "text", "text": "exact source"},
            {"type": "image_url", "image_url": {"url": "not text"}},
        ],
    ],
)
async def test_host_snapshot_reload_requires_exact_fingerprint_and_position(
    pipeline, content
):
    p, _, _ = pipeline
    history = json.dumps([{"role": "user", "content": content}])
    conversation = NS(history=history)
    p.initializer.conversation_manager.get_conversation = AsyncMock(
        return_value=conversation
    )
    raw = {
        "source_kind": "host_history",
        "conversation_id": "c",
        "snapshot_fingerprint": hashlib.sha256(history.encode()).hexdigest(),
        "position": 0,
        "role": "user",
        "text": "exact source",
    }
    descriptor, _ = p.evidence_service._normalize(raw, SCOPE, "", "reply", 0)
    loaded = await p._load_evidence_material(descriptor)
    assert loaded["text"] == "exact source"
    conversation.history = json.dumps([{"role": "user", "content": "changed"}])
    assert await p._load_evidence_material(descriptor) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"position": True},
        {"position": -1},
        {"position": 99},
        {"conversation_id": ""},
        {"snapshot_fingerprint": ""},
        {"role": "assistant"},
    ],
)
async def test_host_source_rejects_malformed_locator(pipeline, change):
    p, _, _ = pipeline
    history = json.dumps([{"role": "user", "content": "source"}])
    p.initializer.conversation_manager.get_conversation = AsyncMock(
        return_value=NS(history=history)
    )
    raw = {
        "source_kind": "host_history",
        "conversation_id": "c",
        "snapshot_fingerprint": hashlib.sha256(history.encode()).hexdigest(),
        "position": 0,
        "role": "user",
        "text": "source",
    }
    descriptor, _ = p.evidence_service._normalize(raw, SCOPE, "", "reply", 0)
    descriptor["source_locator"].update(change)
    assert await p._load_evidence_material(descriptor) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "history",
    [
        "not-json",
        "{}",
        "[{}]",
        '[{"role":"user","content":"source","tool_calls":[]}]',
        '[{"role":"user","content":1}]',
    ],
)
async def test_host_history_structural_drift_never_becomes_source(pipeline, history):
    p, _, _ = pipeline
    p.initializer.conversation_manager.get_conversation = AsyncMock(
        return_value=NS(history=history)
    )
    raw = {
        "source_kind": "host_history",
        "conversation_id": "c",
        "snapshot_fingerprint": hashlib.sha256(history.encode()).hexdigest(),
        "position": 0,
        "role": "user",
        "text": "source",
    }
    descriptor, _ = p.evidence_service._normalize(raw, SCOPE, "", "reply", 0)
    assert await p._load_evidence_material(descriptor) is None
