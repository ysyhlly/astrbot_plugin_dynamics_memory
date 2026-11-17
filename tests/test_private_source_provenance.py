from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection import (
    MemoryReflection,
)
from astrbot_plugin_dynamics_memory.core.evidence_projection import (
    conversation_row_hash,
)
from astrbot_plugin_dynamics_memory.core.memory_source import build_source_row_refs_v1
from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message


SESSION = "fixture:FriendMessage:user-1"


def _message(message_id, role, content, sender_id, *, session_id=SESSION):
    return Message(
        id=message_id,
        session_id=session_id,
        role=role,
        content=content,
        sender_id=sender_id,
        sender_name="fixture sender",
        group_id=None,
        platform="fixture-platform",
        timestamp=1700000000.0 + message_id,
        metadata={},
    )


def test_source_row_refs_are_body_free_exact_private_user_rows():
    user = _message(1, "user", "fixture private body", "fixture-user")
    messages = [
        user,
        _message(2, "assistant", "fixture assistant body", "fixture-bot"),
        _message(3, "user", "session fallback body", SESSION),
        _message(4, "user", "unknown body", "unknown"),
        _message(
            5,
            "user",
            "other session body",
            "fixture-user",
            session_id="other:FriendMessage:user-1",
        ),
    ]

    refs = build_source_row_refs_v1(messages, SESSION)
    assert len(refs) == 1
    ref = refs[0]
    assert set(ref) == {
        "id",
        "session_id",
        "role",
        "sender_id",
        "platform",
        "timestamp",
        "content_sha256",
        "row_hash",
    }
    assert ref["id"] == user.id
    assert ref["session_id"] == SESSION
    assert ref["role"] == "user"
    assert ref["sender_id"] == "fixture-user"
    assert ref["platform"] == "fixture-platform"
    assert (
        ref["content_sha256"]
        == hashlib.sha256(user.content.encode("utf-8")).hexdigest()
    )
    assert ref["row_hash"] == conversation_row_hash(
        {
            "id": user.id,
            "session_id": SESSION,
            "role": "user",
            "sender_id": "fixture-user",
            "platform": "fixture-platform",
            "timestamp": user.timestamp,
            "content": user.content,
        }
    )
    assert "fixture private body" not in repr(refs)


def _reflection_and_capture(importance: float):
    captured = {}

    class Processor:
        async def process_conversation(self, **_kwargs):
            return (
                "fixture summary",
                {"source_row_refs_v1": [{"forged": True}]},
                importance,
            )

        def classify_atoms_from_metadata(self, **_kwargs):
            return []

    class Engine:
        async def add_memory(self, **kwargs):
            captured.update(kwargs)
            return 7

    class Store:
        def __init__(self):
            self.rows = {}

        async def get_message_by_id(self, session_id, message_id):
            return self.rows.get((session_id, message_id))

    class ConversationManager:
        def __init__(self):
            self.store = Store()

        async def get_session_metadata(self, _session_id, key, default=None):
            return 0 if key == "last_summarized_index" else default

        async def update_session_metadata(self, *_args):
            return None

    reflection = MemoryReflection(
        context=None,
        config_manager=SimpleNamespace(get=lambda _key, default=None: default),
        memory_engine=Engine(),
        memory_processor=Processor(),
        conversation_manager=ConversationManager(),
        message_utils=None,
        storage_tasks=set(),
        storage_sessions_inflight=set(),
        storage_state_lock=asyncio.Lock(),
    )
    return reflection, captured


def _seed_exact_rows(reflection, messages):
    rows = {}
    for message in messages:
        rows[(message.session_id, message.id)] = {
            "id": message.id,
            "session_id": message.session_id,
            "role": message.role,
            "content": message.content,
            "sender_id": message.sender_id,
            "sender_name": message.sender_name,
            "group_id": message.group_id,
            "platform": message.platform,
            "timestamp": message.timestamp,
            "metadata": message.metadata,
        }
    reflection.conversation_manager.store.rows.update(rows)


@pytest.mark.asyncio
async def test_low_importance_private_summary_persists_only_row_refs_in_same_add_call():
    reflection, captured = _reflection_and_capture(0.79)
    messages = [
        _message(11, "user", "fixture private body", "fixture-user"),
        _message(12, "assistant", "fixture assistant body", "fixture-bot"),
    ]
    _seed_exact_rows(reflection, messages)
    await reflection._storage_task(SESSION, messages, "fixture-persona", 0, 2, 0)

    metadata = captured["metadata"]
    assert captured["content"] == "fixture summary"
    assert captured["source_messages"] is None
    assert "has_source" not in metadata
    assert "source_message_count" not in metadata
    assert len(metadata["source_row_refs_v1"]) == 1
    assert metadata["source_row_refs_v1"][0]["id"] == 11
    assert "forged" not in repr(metadata["source_row_refs_v1"])
    assert "fixture private body" not in repr(metadata["source_row_refs_v1"])


@pytest.mark.asyncio
async def test_high_importance_private_summary_keeps_full_source_and_adds_row_refs():
    reflection, captured = _reflection_and_capture(0.9)
    messages = [
        _message(21, "user", "fixture private body", "fixture-user"),
        _message(22, "assistant", "fixture assistant body", "fixture-bot"),
    ]
    _seed_exact_rows(reflection, messages)
    await reflection._storage_task(SESSION, messages, "fixture-persona", 0, 2, 0)

    metadata = captured["metadata"]
    assert [item["content"] for item in captured["source_messages"]] == [
        "fixture private body",
        "fixture assistant body",
    ]
    assert len(metadata["source_row_refs_v1"]) == 1
    assert metadata["source_row_refs_v1"][0]["id"] == 21
    assert (
        "has_source" not in metadata
    )  # Engine sets this only when persisting source_messages.
    assert "source_message_count" not in metadata
