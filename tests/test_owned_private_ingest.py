from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
    ConversationManager,
)
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key


@pytest.mark.asyncio
async def test_owned_private_ingest_archives_canonically_and_writes_one_physical_umo_row(
    tmp_path,
):
    store = ConversationStore(str(tmp_path / "owned-private.sqlite3"))
    await store.initialize()
    scope = {
        "platform": "instance-9",
        "umo": "instance-9:FriendMessage:user-7",
        "persona_id": "synthetic-persona",
        "instance_id": "synthetic-persona",
    }
    event = {
        "schema_version": 1,
        "type": "message.received",
        "event_id": "cd-original-input-v1:internal-current-event-key",
        "scope": scope,
        "actor_id": "user-7",
        "occurred_at": 1_700_000_000.25,
        "text": " exact original text ",
        "original_parts": [
            {"kind": "text", "text": " exact original text "},
            {"kind": "quote", "text": "quoted source", "author_id": "quoted-user"},
            {"kind": "image", "opaque_type": "image"},
        ],
        "platform_message_id": "platform-message-42",
    }

    assert (
        await store.ingest_owned_private_original_event(
            event, source_platform="aiocqhttp"
        )
        is True
    )
    raw_row = await store.get_message_by_id(scope["umo"], 1)
    assert raw_row["session_id"] == scope["umo"]
    assert raw_row["role"] == "user"
    assert raw_row["content"] == " exact original text "
    assert raw_row["sender_id"] == "user-7"
    assert raw_row["platform"] == "aiocqhttp"

    archive = await store.load_narrative_event(scope, event["event_id"])
    assert archive["scope"] == scope
    assert archive["original_parts"][1]["kind"] == "quote"
    assert archive["original_parts"][2]["kind"] == "image"
    async with store.connection.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (scope_key(scope),)
    ) as cursor:
        assert (await cursor.fetchone())[0] == 0

    # A replay under another persona/card still describes the same physical
    # event and must not create a second UMO projection or replace its archive.
    other_scope = {
        **scope,
        "persona_id": "another-persona",
        "instance_id": "another-persona",
    }
    other_persona_event = {**event, "scope": other_scope}
    assert (
        await store.ingest_owned_private_original_event(
            other_persona_event, source_platform="aiocqhttp"
        )
        is False
    )
    assert await store.load_narrative_event(other_scope, event["event_id"]) is None
    async with store.connection.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (scope["umo"],)
    ) as cursor:
        assert (await cursor.fetchone())[0] == 1

    # Archive idempotency blocks replay before writing a replacement row, even
    # if normal retention already removed the UMO message.
    await store.connection.execute(
        "DELETE FROM messages WHERE session_id=?", (scope["umo"],)
    )
    await store.connection.commit()
    assert (
        await store.ingest_owned_private_original_event(
            other_persona_event, source_platform="aiocqhttp"
        )
        is False
    )
    async with store.connection.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (scope["umo"],)
    ) as cursor:
        assert (await cursor.fetchone())[0] == 0

    await store.close()


@pytest.mark.asyncio
async def test_owned_private_ingest_rejects_group_scope_and_invalid_identity(tmp_path):
    store = ConversationStore(str(tmp_path / "owned-private-reject.sqlite3"))
    await store.initialize()
    scope = {
        "platform": "instance-9",
        "umo": "instance-9:GroupMessage:group-1",
        "persona_id": "synthetic-persona",
        "instance_id": "synthetic-persona",
    }
    event = {
        "schema_version": 1,
        "type": "message.received",
        "event_id": "cd-original-input-v1:internal-group-event",
        "scope": scope,
        "actor_id": "user-7",
        "occurred_at": 1_700_000_000,
        "text": "synthetic text",
    }
    assert (
        await store.ingest_owned_private_original_event(
            event, source_platform="aiocqhttp"
        )
        is False
    )
    assert (
        await store.ingest_owned_private_original_event(
            {
                **event,
                "scope": {**scope, "umo": "instance-9:FriendMessage:user-7"},
                "event_id": "unknown:event",
            },
            source_platform="aiocqhttp",
        )
        is False
    )
    await store.close()


@pytest.mark.asyncio
async def test_manager_owned_ingest_invalidates_the_physical_umo_history_cache(
    tmp_path,
):
    store = ConversationStore(str(tmp_path / "owned-cache.sqlite3"))
    await store.initialize()
    manager = object.__new__(ConversationManager)
    manager.store = store
    manager._cache_lock = __import__("asyncio").Lock()
    session_id = "instance-9:FriendMessage:user-7"
    manager._cache = {session_id: ["stale cached history"]}
    scope = {
        "platform": "instance-9",
        "umo": session_id,
        "persona_id": "synthetic-persona",
        "instance_id": "synthetic-persona",
    }
    event = {
        "schema_version": 1,
        "type": "message.received",
        "event_id": "cd-original-input-v1:cache-event",
        "scope": scope,
        "actor_id": "user-7",
        "occurred_at": 1_700_000_010,
        "text": "synthetic cache invalidation text",
        "original_parts": [],
    }
    assert (
        await manager.ingest_owned_private_original_event(
            event, source_platform="aiocqhttp"
        )
        is True
    )
    assert session_id not in manager._cache
    await store.close()
