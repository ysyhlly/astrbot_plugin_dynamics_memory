from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection import (
    MemoryReflection,
)
from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
    ConversationManager,
)


def _message():
    return Message(
        id=31,
        session_id="synthetic:FriendMessage:user",
        role="user",
        content="synthetic original text",
        sender_id="synthetic-user",
        sender_name="Synthetic",
        group_id=None,
        platform="synthetic-adapter",
        timestamp=1_700_000_000.0,
        metadata={
            "schema_version": 1,
            "source_event_id": "synthetic-event",
            "scope": {
                "platform": "synthetic",
                "umo": "synthetic:FriendMessage:user",
                "persona_id": "persona",
                "instance_id": "persona",
            },
            "fact_kind": "observation",
            "is_bot_message": False,
        },
    )


@pytest.fixture
def reflection():
    value = object.__new__(MemoryReflection)
    value.conversation_manager = SimpleNamespace(store=None)
    return value


@pytest.mark.asyncio
async def test_reflection_source_check_accepts_exact_persisted_row(reflection):
    message = _message()
    row = {
        "id": message.id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "sender_id": message.sender_id,
        "sender_name": message.sender_name,
        "group_id": message.group_id,
        "platform": message.platform,
        "timestamp": message.timestamp,
        "metadata": json.dumps(message.metadata),
    }

    class Store:
        async def get_message_by_id(self, session_id, message_id):
            assert session_id == message.session_id
            assert message_id == message.id
            return row

    reflection.conversation_manager.store = Store()
    assert await reflection._history_rows_current(message.session_id, [message]) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "edited original"),
        ("sender_id", "other-user"),
        ("timestamp", 1_700_000_001.0),
        ("source_event_id", "different-event"),
    ],
)
async def test_reflection_source_check_rejects_changed_or_reattributed_row(
    reflection, field, value
):
    message = _message()
    row = {
        "id": message.id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "sender_id": message.sender_id,
        "sender_name": message.sender_name,
        "group_id": message.group_id,
        "platform": message.platform,
        "timestamp": message.timestamp,
        "metadata": json.dumps(message.metadata),
    }
    if field == "source_event_id":
        changed = dict(message.metadata)
        changed[field] = value
        row["metadata"] = json.dumps(changed)
    else:
        row[field] = value

    class Store:
        async def get_message_by_id(self, _session_id, _message_id):
            return row

    reflection.conversation_manager.store = Store()
    assert (
        await reflection._history_rows_current(message.session_id, [message]) is False
    )


@pytest.mark.asyncio
async def test_reflection_source_check_rejects_missing_exact_row(reflection):
    message = _message()

    class Store:
        async def get_message_by_id(self, _session_id, _message_id):
            return None

    reflection.conversation_manager.store = Store()
    assert (
        await reflection._history_rows_current(message.session_id, [message]) is False
    )


@pytest.mark.asyncio
async def test_memory_engine_rejects_source_change_before_write_journal():
    engine = object.__new__(MemoryEngine)
    writes = []

    async def source_changed():
        return False

    async def start_write(*_args, **_kwargs):
        writes.append("journal")
        return 1

    engine._start_write_op = start_write
    with pytest.raises(ValueError, match="source_snapshot_changed"):
        await MemoryEngine.add_memory(
            engine,
            content="synthetic derived memory",
            session_id="synthetic-session",
            persona_id="synthetic-persona",
            source_validator=source_changed,
            source_publish_lease=lambda: None,
        )
    assert writes == []


@pytest.mark.asyncio
async def test_source_publication_lease_serializes_final_check_with_message_writers():
    lock = __import__("asyncio").Lock()
    manager = object.__new__(ConversationManager)
    manager.store = SimpleNamespace(_write_lock=lock)
    inside_lease = __import__("asyncio").Event()
    release_lease = __import__("asyncio").Event()
    writer_entered = __import__("asyncio").Event()
    validator_calls = 0

    async def validator():
        nonlocal validator_calls
        validator_calls += 1
        return True

    async def publish():
        async with manager.source_publication_lease(validator) as valid:
            assert valid is True
            inside_lease.set()
            await release_lease.wait()

    async def writer():
        async with manager.store._write_lock:
            writer_entered.set()

    publish_task = __import__("asyncio").create_task(publish())
    await inside_lease.wait()
    writer_task = __import__("asyncio").create_task(writer())
    await __import__("asyncio").sleep(0)
    assert not writer_entered.is_set()
    release_lease.set()
    await publish_task
    await writer_task
    assert writer_entered.is_set()
    assert validator_calls == 1
