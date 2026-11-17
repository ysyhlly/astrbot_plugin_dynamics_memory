"""Reliability-focused tests for EventHandler."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler
from tests.event_handler_support import (
    _make_event,
    _make_req,
    _make_resp,
    _make_real_handler,
)


def _make_handler() -> EventHandler:
    memory_engine = Mock()
    memory_engine.search_memories = AsyncMock(return_value=[])
    memory_engine.add_memory = AsyncMock(return_value=1)
    memory_engine._create_tracked_task = None

    memory_processor = Mock()
    memory_processor.process_conversation = AsyncMock(return_value=("summary", {}, 0.5))

    conversation_manager = Mock()
    conversation_manager.add_message_from_event = AsyncMock(return_value=Mock(id=1))
    conversation_manager.get_session_info = AsyncMock(
        return_value=Mock(message_count=2)
    )
    conversation_manager.get_session_metadata = AsyncMock(return_value=0)
    conversation_manager.update_session_metadata = AsyncMock()
    conversation_manager.get_messages_range = AsyncMock(return_value=[])
    conversation_manager.store = Mock()
    conversation_manager.store.get_message_count = AsyncMock(return_value=2)

    handler = EventHandler(
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
    )
    return handler


@pytest.mark.asyncio
async def test_handle_all_group_messages_reraises_cancelled_error() -> None:
    handler = _make_handler()
    handler.conversation_manager.add_message_from_event = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    with pytest.raises(asyncio.CancelledError):
        await handler.handle_all_group_messages(_make_event(group=True))


@pytest.mark.asyncio
async def test_handle_memory_recall_reraises_caller_cancellation() -> None:
    handler = _make_handler()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def search(**kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    handler.memory_engine.search_memories = AsyncMock(side_effect=search)
    req = _make_req("provider prompt")
    task = asyncio.create_task(handler.handle_memory_recall(_make_event(), req))
    try:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(stopped.wait(), 1)

        assert req._lm_private_hook_outcome_v1 == "cancelled"
        assert req.extra_user_content_parts == []
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await handler.shutdown()


@pytest.mark.asyncio
async def test_cancelled_recall_scope_returns_partial() -> None:
    handler = _make_handler()
    handler.memory_engine.search_memories = AsyncMock(
        side_effect=asyncio.CancelledError()
    )
    req = _make_req()

    await handler.handle_memory_recall(_make_event(), req)
    await handler.shutdown()

    handler.memory_engine.search_memories.assert_awaited_once()
    assert req._lm_private_hook_outcome_v1 == "partial"
    assert req.extra_user_content_parts == []


@pytest.mark.asyncio
async def test_handle_memory_reflection_reraises_cancelled_error() -> None:
    handler = _make_handler()
    handler.conversation_manager.add_message_from_event = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    with pytest.raises(asyncio.CancelledError):
        await handler.handle_memory_reflection(_make_event(), _make_resp())
    handler.conversation_manager.add_message_from_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_memory_recall_awaits_async_component_getter() -> None:
    from astrbot.core.message.components import Plain

    handler = _make_handler()
    event = _make_event()
    event.get_messages = AsyncMock(return_value=[Plain("async getter text")])
    event.get_message_str = AsyncMock(return_value="flattened text")

    await handler.handle_memory_recall(event, _make_req("query"))
    await handler.shutdown()

    event.get_messages.assert_awaited_once()
    event.get_message_str.assert_not_called()
    handler.memory_engine.search_memories.assert_awaited_once()
    assert (
        handler.memory_engine.search_memories.await_args.kwargs["query"]
        == "async getter text"
    )


@pytest.mark.asyncio
async def test_handle_memory_recall_awaits_async_text_getter_for_legacy_adapter() -> (
    None
):
    handler = _make_handler()
    event = _make_event()
    event.get_messages = None
    event.get_message_str = AsyncMock(return_value="legacy adapter text")

    await handler.handle_memory_recall(event, _make_req("query"))
    await handler.shutdown()

    event.get_message_str.assert_awaited_once()
    handler.memory_engine.search_memories.assert_awaited_once()
    assert (
        handler.memory_engine.search_memories.await_args.kwargs["query"]
        == "legacy adapter text"
    )


@pytest.mark.asyncio
async def test_extract_message_content_skips_unknown_components() -> None:
    from astrbot.core.message.components import Plain

    handler = _make_handler()
    event = _make_event()
    unknown = Mock(type="sticker")
    event.get_messages = Mock(return_value=[Plain("hello"), unknown])

    content = await handler._message_utils.extract_message_content(event)

    assert content == "hello"


@pytest.mark.asyncio
async def test_shutdown_cancels_storage_tasks_after_timeout(monkeypatch) -> None:
    """卸载时存储任务超过超时时间应被取消，不能无限挂起。"""
    handler = _make_handler()
    monkeypatch.setattr(EventHandler, "_SHUTDOWN_TIMEOUT_SECONDS", 0.05)

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hang() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(hang())
    handler._storage_tasks.add(task)
    await started.wait()

    await asyncio.wait_for(handler.shutdown(), timeout=2.0)

    assert cancelled.is_set()
    assert not handler._storage_tasks
    assert task.done()


@pytest.mark.asyncio
async def test_provider_policy_refusal_is_not_retried():
    from astrbot_plugin_dynamics_memory.core.processors.memory_processor import (
        MemoryProcessor,
    )

    processor = MemoryProcessor.__new__(MemoryProcessor)
    provider = Mock(
        text_chat=AsyncMock(
            side_effect=RuntimeError("Gemini prompt blocked: PROHIBITED_CONTENT")
        )
    )
    processor._get_current_llm_provider = lambda: provider
    with pytest.raises(RuntimeError):
        await processor._call_llm_with_retry("synthetic", "system")
    provider.text_chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_stored_summary_does_not_call_model_again(tmp_path):
    from tests.test_event_handler_recall import _seed_summary_window

    handler, manager, store = await _make_real_handler(tmp_path)
    handler._memory_reflection.memory_processor.process_conversation = AsyncMock()
    handler.memory_engine.stored_summary_window = AsyncMock(return_value=123)
    session_id = "test:private:sid-1"
    try:
        messages = await _seed_summary_window(manager, session_id)
        committed = await handler._memory_reflection._storage_task(
            session_id, messages, "p1", 0, 2, 1
        )

        assert committed is True
        handler.memory_engine.stored_summary_window.assert_awaited_once()
        handler.memory_processor.process_conversation.assert_not_awaited()
        handler.memory_engine.add_memory.assert_not_awaited()
        assert (
            await manager.get_session_metadata(session_id, "last_summarized_index", 0)
            == 2
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_managed_receipt_does_not_replay_rejected_window(tmp_path):
    from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
    from tests.event_handler_support import _make_real_handler
    from tests.test_narrative_api import SCOPE, envelope

    handler, manager, store = await _make_real_handler(tmp_path)
    handler.config_manager._config["reflection_engine"]["summary_trigger_rounds"] = 1
    handler.memory_processor.process_conversation = AsyncMock()
    session = scope_key(SCOPE)
    try:
        assert await store.ingest_narrative_event(
            envelope("input", text="first original source")
        )
        assert await store.ingest_narrative_event(
            envelope("delivery", "delivery.success", "second original source")
        )
        await manager.update_session_metadata(
            session,
            "pending_summary",
            {
                "failure_stage": "provider_terminal",
                "retry_count": 1,
                "start_index": 0,
                "end_index": 2,
            },
        )
        await handler.reflect_recorded_scope(SCOPE)
        handler.memory_processor.process_conversation.assert_not_awaited()
        assert not handler._storage_tasks
        assert await manager.get_session_metadata(session, "last_summarized_index") == 0
        _, rows, refusal = await store.get_managed_summary_window(
            session, 2, 2, retry_refused=True
        )
        assert [row.content for row in rows] == [
            "first original source",
            "second original source",
        ]
        assert refusal["failure_stage"] == "provider_terminal"
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_memory_model_total_deadline_prevents_nested_retry_storm():
    from astrbot_plugin_dynamics_memory.core.processors.memory_processor import (
        MemoryProcessor,
    )

    processor = MemoryProcessor.__new__(MemoryProcessor)
    processor.config = {
        "llm_attempt_timeout_seconds": 0.005,
        "llm_total_timeout_seconds": 0.02,
    }

    async def slow(**kwargs):
        await asyncio.sleep(1)

    provider = Mock(text_chat=AsyncMock(side_effect=slow))
    processor._get_current_llm_provider = lambda: provider
    with pytest.raises(TimeoutError):
        await processor._call_llm_with_retry("synthetic", "system")
    assert provider.text_chat.await_count == 1
    assert provider.text_chat.call_args.kwargs["request_max_retries"] == 1
