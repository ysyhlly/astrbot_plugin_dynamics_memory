"""Shared fixtures and helpers for the event handler test modules."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler
from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import HybridResult
from astrbot_plugin_dynamics_memory.core.summary_source import text_sha256

from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.message.components import Plain


@pytest.fixture
def memory_engine():
    engine = Mock()
    engine.search_memories = AsyncMock(return_value=[])
    engine.add_memory = AsyncMock(return_value=1)
    engine.find_similar_memory = AsyncMock(return_value=None)
    engine._create_tracked_task = None
    return engine


@pytest.fixture
def memory_processor():
    processor = Mock()
    processor.process_conversation = AsyncMock(
        return_value=("summary", {"topics": ["t1"]}, 0.6)
    )
    processor.classify_atoms_from_metadata = Mock(return_value=[])
    return processor


@pytest.fixture
def conversation_manager():
    manager = Mock()
    manager.add_message_from_event = AsyncMock(return_value=Mock(id=1, metadata={}))
    manager.get_session_info = AsyncMock(return_value=Mock(message_count=12))
    session_metadata = {"last_summarized_index": 0, "pending_summary": None}
    messages = [
        Message(
            id=index + 1,
            session_id="test:private:sid-1",
            role="user" if index % 2 == 0 else "assistant",
            content=f"fixture message {index + 1}",
            sender_id="user-1" if index % 2 == 0 else "bot-1",
            timestamp=1_760_000_000 + index,
        )
        for index in range(12)
    ]

    async def _get_session_metadata(session_id, key, default=None):
        return session_metadata.get(key, default)

    async def _update_session_metadata(session_id, key, value):
        session_metadata[key] = value

    manager.get_session_metadata = AsyncMock(side_effect=_get_session_metadata)
    async def _get_messages_range(*, session_id, start_index, end_index):
        return [message for message in messages if message.session_id == session_id][
            start_index:end_index
        ]

    manager.get_messages_range = AsyncMock(side_effect=_get_messages_range)
    manager.update_session_metadata = AsyncMock(side_effect=_update_session_metadata)

    async def _commit_summary_cursor(session_id, last_message, expected_revision, *, window_messages=None):
        position = next(
            i for i, message in enumerate(messages, 1) if message is last_message
        )
        await manager.update_session_metadata(
            session_id, "last_summarized_index", position
        )
        await manager.update_session_metadata(session_id, "pending_summary", None)
        return True

    manager.commit_summary_cursor = AsyncMock(side_effect=_commit_summary_cursor)

    async def _get_summary_window(session_id, limit, *, exact_pending=False):
        start = await manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        )
        pending = await manager.get_session_metadata(
            session_id, "pending_summary", None
        )
        if pending:
            pending_start = pending.get("start_index", start)
            if exact_pending and pending_start <= start:
                limit = min(1000, pending["end_index"] - pending_start)
            start = min(start, pending_start)
        end = min(await manager.store.get_message_count(session_id), start + limit)
        rows = await manager.get_messages_range(
            session_id=session_id, start_index=start, end_index=end
        )
        return start, rows

    manager.get_summary_window = AsyncMock(side_effect=_get_summary_window)

    manager.invalidate_cache = AsyncMock()
    manager.reconcile_session_tail = AsyncMock(return_value=0)
    manager.get_last_message = AsyncMock(return_value=None)
    manager.store = Mock()
    manager.store.get_message_count = AsyncMock(return_value=12)
    manager.store.get_last_message = AsyncMock(return_value=None)
    manager.store.get_session = AsyncMock(return_value=SimpleNamespace(metadata=session_metadata))
    manager.store.native_summary_window_start = AsyncMock(return_value=0)
    manager.store.update_message_metadata = AsyncMock()
    manager.store.connection = Mock()
    manager.store.connection.execute = AsyncMock(return_value=Mock(rowcount=1))
    manager.store.connection.commit = AsyncMock()
    manager.get_session_revision = AsyncMock(return_value=None)

    async def _get_message_by_id(session_id, message_id):
        for message in messages:
            if message.session_id == session_id and message.id == message_id:
                return message.to_dict()
        return None

    manager.store.get_message_by_id = AsyncMock(side_effect=_get_message_by_id)
    return manager


@pytest.fixture
def handler(memory_engine, memory_processor, conversation_manager):
    return EventHandler(
        context=Mock(),
        config_manager=ConfigManager(
            {
                "recall_engine": {"top_k": 3, "injection_method": "extra_user_content"},
                "reflection_engine": {"summary_trigger_rounds": 1},
                "session_manager": {"max_messages_per_session": 100},
            }
        ),
        memory_engine=memory_engine,
        memory_processor=memory_processor,
        conversation_manager=conversation_manager,
        summary_guard=_summary_guard,
    )


def _recalled(content: str, *, doc_id: int = 99, importance: float = 0.9):
    return HybridResult(
        doc_id=doc_id,
        final_score=0.8,
        rrf_score=0.8,
        bm25_score=0.8,
        vector_score=None,
        content=content,
        metadata={"importance": importance, "session_id": "test:private:sid-1"},
    )


async def _summary_guard(event, req, source_records, *, segment_text, item_count, **_):
    """Model the guard receipt boundary for request mutation tests."""
    if len(source_records) != item_count or not all(
        type(source["document_id"]) is int
        and source["document_id"] > 0
        and source["source_session_id"] == source["source_scope"]["umo"] == event.unified_msg_origin
        and len(source["document_text_sha256"]) == 64
        and len(source["summary_metadata_sha256"]) == 64
        for source in source_records
    ):
        return False
    receipt = {
        "segment_sha256": text_sha256(segment_text),
        "segment_item_count": item_count,
        "source_documents": source_records,
    }
    req._lm_private_summary_sources_v1 = receipt

    async def validate():
        refs = req._lm_private_segment_refs_v1
        return req._lm_private_summary_sources_v1 is receipt and any(
            getattr(part, "text", None) == segment_text
            and refs.get(id(part)) == {"source_level": "legacy_summary", "item_count": item_count}
            for part in req.extra_user_content_parts
        )

    req._lm_private_summary_validate_v1 = validate
    return True


def _make_req(prompt: str = "hello") -> ProviderRequest:
    return ProviderRequest(prompt=prompt)


def _make_resp(text: str = "assistant reply") -> LLMResponse:
    return LLMResponse(role="assistant", completion_text=text)


def _make_event(group: bool = False, checkpoint: str | None = None, *, text: str = "hello"):
    event = Mock()
    event._cd_owned_original_input_v1 = None
    event._chat_dynamics_owned_request = False
    event._cd_owned_memory_sources_validate_v1 = None
    event._cd_memory_source_validate = None
    event.unified_msg_origin = "test:private:sid-1"
    event.get_message_type = Mock(
        return_value=MessageType.GROUP_MESSAGE if group else MessageType.FRIEND_MESSAGE
    )
    event.get_sender_id = Mock(return_value="user-1")
    event.get_self_id = Mock(return_value="bot-1")
    event.get_sender_name = Mock(return_value="Tester")
    event.get_message_str = Mock(return_value=text)
    event.get_messages = Mock(
        side_effect=lambda: [Plain(event.get_message_str.return_value)]
        if isinstance(event.get_message_str.return_value, str) and event.get_message_str.return_value
        else []
    )
    event.get_platform_name = Mock(return_value="test")
    event.get_platform_id = Mock(return_value="test")
    event.get_extra = Mock(
        side_effect=lambda key, default=None: checkpoint if key == "llm_checkpoint_id" else default
    )
    return event


async def _make_real_handler(tmp_path, memory_engine=None):
    """构建使用真实 SQLite 的 EventHandler（memory_engine 用 mock）。"""
    from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
        ConversationManager,
    )
    from astrbot_plugin_dynamics_memory.storage.conversation_store import (
        ConversationStore,
    )

    store = ConversationStore(str(tmp_path / "conv-history.db"))
    await store.initialize()
    manager = ConversationManager(
        store=store, max_cache_size=10, context_window_size=20, session_ttl=3600
    )

    if memory_engine is None:
        engine = Mock()
        engine.search_memories = AsyncMock(return_value=[])
        engine.add_memory = AsyncMock(return_value=1)
        engine._create_tracked_task = None
        memory_engine = engine

    handler = EventHandler(
        context=Mock(),
        config_manager=ConfigManager(
            {
                "recall_engine": {
                    "top_k": 3,
                    "injection_method": "extra_user_content",
                },
                "reflection_engine": {"summary_trigger_rounds": 100},
                "session_manager": {"max_messages_per_session": 100},
                "filtering_settings": {"use_persona_filtering": False},
            }
        ),
        memory_engine=memory_engine,
        memory_processor=Mock(),
        conversation_manager=manager,
    )
    return handler, manager, store


async def _seed_turns(manager, sid: str, checkpoints: tuple[str, ...]) -> None:
    """写入若干完整轮次（user + assistant，均带 checkpoint）。"""
    for checkpoint in checkpoints:
        await manager.add_message(
            session_id=sid,
            role="user",
            content=f"q-{checkpoint}",
            sender_id="u1",
            metadata={"llm_checkpoint_id": checkpoint},
        )
        await manager.add_message(
            session_id=sid,
            role="assistant",
            content=f"a-{checkpoint}",
            sender_id="bot1",
            metadata={"llm_checkpoint_id": checkpoint},
        )
