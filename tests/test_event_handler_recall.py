"""
Event handler tests: message capture, recall injection and storage basics.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from astrbot.core.agent.message import TextPart
from astrbot.core.message.components import Image, Plain
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.base.constants import (
    MEMORY_INJECTION_FOOTER,
    MEMORY_INJECTION_HEADER,
)
from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler
from astrbot_plugin_dynamics_memory.core.summary_source import text_sha256

from tests.event_handler_support import (
    _make_event,
    _make_real_handler,
    _make_req,
    _make_resp,
    _recalled,
    _summary_guard,
    conversation_manager,
    handler,
    memory_engine,
    memory_processor,
)
from tests.integration.test_real_db_end_to_end import _TestEvent, real_db_stack


@pytest.mark.asyncio
async def test_message_dedup_cache_works(handler):
    key = "id:123"
    assert await handler._message_utils.is_duplicate_message(key) is False
    await handler._message_utils.mark_message_processed(key)
    assert await handler._message_utils.is_duplicate_message(key) is True


@pytest.mark.asyncio
async def test_handle_memory_recall_injects_extra_user_content(handler, memory_engine):
    """extra_user_content 注入方式：记忆应追加到 extra_user_content_parts 并标记为临时消息。"""
    event = _make_event(group=False)
    req = _make_req("query text")
    recalled = _recalled("mem1")
    memory_engine.search_memories = AsyncMock(return_value=[recalled])

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_1"
        await handler.handle_memory_recall(event, req)

    memory_engine.search_memories.assert_awaited_once()
    assert len(req.extra_user_content_parts) == 1
    text_part = req.extra_user_content_parts[0]
    assert "<RAG-Faiss-Memory>" in text_part.text
    assert getattr(text_part, "_no_save", False) is True
    # system_prompt 不应被修改
    assert req.system_prompt == ""


@pytest.mark.asyncio
async def test_handle_memory_recall_stores_private_user_message(
    handler, conversation_manager
):
    event = _make_event(group=False)
    req = _make_req("user input")

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_1"
        await handler.handle_memory_recall(event, req)

    conversation_manager.add_message_from_event.assert_awaited()


@pytest.mark.asyncio
async def test_handle_memory_reflection_triggers_storage_task(
    handler, conversation_manager, memory_engine
):
    event = _make_event(group=False)
    resp = _make_resp("assistant answer")

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_1"
        await handler.handle_memory_reflection(event, resp)
        # Wait for background storage task.
        await handler.shutdown()

    assert conversation_manager.get_messages_range.await_count >= 1
    assert memory_engine.add_memory.await_count >= 1


@pytest.mark.asyncio
async def test_handle_all_group_messages_and_limit_cleanup(
    handler, conversation_manager
):
    event = _make_event(group=True)
    handler.config_manager._config["session_manager"]["max_messages_per_session"] = 10
    conversation_manager.store.get_message_count = AsyncMock(return_value=12)
    conversation_manager.get_session_metadata = AsyncMock(return_value=5)
    conversation_manager.trim_session_messages = AsyncMock(return_value=5)

    await handler.handle_all_group_messages(event)

    conversation_manager.add_message_from_event.assert_awaited()
    conversation_manager.trim_session_messages.assert_awaited_once_with(event.unified_msg_origin, 5)


@pytest.mark.asyncio
async def test_enforce_message_limit_uses_cleanup_batch_size(
    memory_engine, memory_processor, conversation_manager
):
    handler = EventHandler(
        context=Mock(),
        config_manager=ConfigManager(
            {
                "recall_engine": {"top_k": 3, "injection_method": "extra_user_content"},
                "reflection_engine": {"summary_trigger_rounds": 1},
                "session_manager": {
                    "max_messages_per_session": 100,
                    "cleanup_batch_size": 20,
                },
            }
        ),
        memory_engine=memory_engine,
        memory_processor=memory_processor,
        conversation_manager=conversation_manager,
    )

    conversation_manager.store.get_message_count = AsyncMock(return_value=101)
    conversation_manager.get_session_metadata = AsyncMock(return_value=80)
    conversation_manager.trim_session_messages = AsyncMock(return_value=20)

    await handler._message_utils.enforce_message_limit("test:private:sid-1")

    conversation_manager.trim_session_messages.assert_awaited_once_with(
        "test:private:sid-1", 20
    )
    assert [call.args for call in conversation_manager.get_session_metadata.await_args_list] == [
        ("test:private:sid-1", "last_summarized_index", 0),
        ("test:private:sid-1", "last_summarized_index", 60),
    ]
    conversation_manager.update_session_metadata.assert_not_awaited()
    conversation_manager.invalidate_cache.assert_awaited_once_with("test:private:sid-1")


@pytest.mark.asyncio
async def test_handle_all_group_messages_skips_bot_own_messages(
    handler, conversation_manager
):
    """Bot 自己的消息应被跳过，由 handle_memory_reflection 负责写入。"""
    event = _make_event(group=True)
    # sender_id == self_id → bot's own message
    event.get_sender_id = Mock(return_value="bot-1")
    event.get_self_id = Mock(return_value="bot-1")

    await handler.handle_all_group_messages(event)

    # should NOT store bot's own message
    conversation_manager.add_message_from_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_waits_for_storage_tasks(handler):
    finished = asyncio.Event()

    async def store_summary():
        await asyncio.sleep(0)
        finished.set()

    handler._storage_tasks.add(asyncio.create_task(store_summary()))

    await handler.shutdown()

    assert finished.is_set()
    assert not handler._storage_tasks
    assert handler._shutting_down is True


# ── EventHandler 边界条件与 source_window 测试 ────────────────────────────────


@pytest.mark.asyncio
async def test_handle_memory_recall_skips_when_original_text_empty(handler, memory_engine):
    """Provider-only text is neither stored nor searched without a native Plain part."""
    event = _make_event(text="")
    req = _make_req("generated envelope")

    await handler.handle_memory_recall(event, req)

    assert req._lm_private_hook_outcome_v1 == "empty"
    assert req.prompt == "generated envelope"
    assert req.extra_user_content_parts == []
    memory_engine.search_memories.assert_not_awaited()
    handler.conversation_manager.add_message_from_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_remove", [True, False])
async def test_native_image_only_request_applies_legacy_cleanup(handler, memory_engine, auto_remove):
    handler.config_manager._config["recall_engine"]["auto_remove_injected"] = auto_remove
    event = _make_event(text="")
    event.get_messages.return_value = [Image(file="synthetic.png")]
    event.get_messages.side_effect = None
    req = _make_req("generated image caption")
    stale = f"{MEMORY_INJECTION_HEADER}old memory{MEMORY_INJECTION_FOOTER}"
    old_part = TextPart(text=stale).mark_as_temp()
    caption = TextPart(text="generated image caption").mark_as_temp()
    req.system_prompt = f"persona\n{stale}"
    req.extra_user_content_parts = [old_part, caption]
    req._lm_private_summary_sources_v1 = {"segment_sha256": text_sha256(stale)}
    req._lm_private_summary_validate_v1 = AsyncMock(return_value=True)
    req._lm_private_segment_refs_v1 = {id(old_part): {"source_level": "legacy_summary"}}
    visible_user = {"role": "user", "content": "original conversation"}
    req.contexts = [
        visible_user,
        {"role": "assistant", "tool_calls": [{"id": "fake_recall_old"}]},
        {"role": "tool", "tool_call_id": "fake_recall_old", "content": "old memory"},
    ]

    await handler.handle_memory_recall(event, req)

    assert req._lm_private_hook_outcome_v1 == "empty"
    memory_engine.search_memories.assert_not_awaited()
    if auto_remove:
        assert req.system_prompt == "persona"
        assert req.extra_user_content_parts == [caption]
        assert req.contexts == [visible_user]
        assert not hasattr(req, "_lm_private_summary_sources_v1")
        assert not hasattr(req, "_lm_private_summary_validate_v1")
        assert req._lm_private_segment_refs_v1 == {}
    else:
        assert stale in req.system_prompt
        assert req.extra_user_content_parts == [old_part, caption]
        assert len(req.contexts) == 3


@pytest.mark.asyncio
async def test_guarded_recall_with_persona_filtering_disabled(real_db_stack):
    plugin = real_db_stack["plugin"]
    plugin.config_manager._config["filtering_settings"]["use_persona_filtering"] = False
    engine = real_db_stack["memory_engine"]
    session_id = "test:FriendMessage:current-session"
    saved_id = await engine.add_memory(
        content="synthetic running memory from this conversation",
        session_id=session_id, persona_id="persona-real", importance=0.95,
    )
    other_id = await engine.add_memory(
        content="synthetic running memory from another conversation",
        session_id="test:FriendMessage:foreign-session", persona_id="persona-real", importance=0.95,
    )
    assert saved_id and other_id
    req = _make_req("provider envelope")

    await real_db_stack["event_handler"].handle_memory_recall(
        _TestEvent(session_id, "what do you remember about running?"), req
    )

    assert req._lm_private_hook_outcome_v1 == "ready"
    assert len(req.extra_user_content_parts) == 1
    assert "from this conversation" in req.extra_user_content_parts[0].text
    assert "another conversation" not in req.extra_user_content_parts[0].text
    assert await req._lm_private_summary_validate_v1() is True


@pytest.mark.asyncio
async def test_handle_memory_recall_injection_user_message_before(
    handler, memory_engine
):
    """injection_method=user_message_before 已废弃，应回退为 extra_user_content。"""
    from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
    from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler

    h = EventHandler(
        context=Mock(),
        config_manager=ConfigManager(
            {
                "recall_engine": {
                    "top_k": 3,
                    "injection_method": "user_message_before",
                },
                "reflection_engine": {"summary_trigger_rounds": 1},
                "session_manager": {"max_messages_per_session": 100},
            }
        ),
        memory_engine=memory_engine,
        memory_processor=Mock(),
        conversation_manager=Mock(),
        summary_guard=_summary_guard,
    )
    h.conversation_manager.add_message_from_event = AsyncMock()

    recalled = _recalled("mem_before")
    memory_engine.search_memories = AsyncMock(return_value=[recalled])

    event = _make_event(group=False)
    req = _make_req("user question")

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "p1"
        await h.handle_memory_recall(event, req)

    # 废弃模式不应改写 prompt（改写会被 AstrBot 持久化进对话历史）
    assert req.prompt == "user question"

    # 记忆应以临时片段追加到用户消息末尾，且不落库
    assert len(req.extra_user_content_parts) == 1
    text_part = req.extra_user_content_parts[0]
    assert "mem_before" in text_part.text
    assert getattr(text_part, "_no_save", False) is True


@pytest.mark.asyncio
async def test_handle_memory_recall_injection_user_message_after(
    handler, memory_engine
):
    """injection_method=user_message_after 已废弃，应回退为 extra_user_content。"""
    from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
    from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler

    h = EventHandler(
        context=Mock(),
        config_manager=ConfigManager(
            {
                "recall_engine": {
                    "top_k": 3,
                    "injection_method": "user_message_after",
                },
                "reflection_engine": {"summary_trigger_rounds": 1},
                "session_manager": {"max_messages_per_session": 100},
            }
        ),
        memory_engine=memory_engine,
        memory_processor=Mock(),
        conversation_manager=Mock(),
        summary_guard=_summary_guard,
    )
    h.conversation_manager.add_message_from_event = AsyncMock()

    recalled = _recalled("mem_after")
    memory_engine.search_memories = AsyncMock(return_value=[recalled])

    event = _make_event(group=False)
    req = _make_req("user question")

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "p1"
        await h.handle_memory_recall(event, req)

    # 废弃模式不应改写 prompt
    assert req.prompt == "user question"

    # 记忆应以临时片段追加到用户消息末尾，且不落库
    assert len(req.extra_user_content_parts) == 1
    text_part = req.extra_user_content_parts[0]
    assert "mem_after" in text_part.text
    assert getattr(text_part, "_no_save", False) is True


async def _seed_summary_window(manager, session_id):
    await manager.add_message(session_id=session_id, role="user", content="hello", sender_id="u1")
    await manager.add_message(session_id=session_id, role="assistant", content="hi", sender_id="bot")
    return await manager.get_messages_range(session_id=session_id, start_index=0, end_index=2)


@pytest.mark.asyncio
async def test_storage_task_writes_source_window(tmp_path, memory_engine, memory_processor):
    handler, manager, store = await _make_real_handler(tmp_path, memory_engine)
    handler._memory_reflection.memory_processor = memory_processor
    memory_processor.process_conversation.return_value = ("summary", {"topics": ["t1"]}, 0.9)
    session_id = "test:private:sid-1"
    scope = "livingmemory:user:test:u1"
    try:
        messages = await _seed_summary_window(manager, session_id)
        await handler._memory_reflection._storage_task(
            session_id, messages, "p1", 0, 2, 0, memory_scope=scope
        )

        memory_engine.add_memory.assert_awaited_once()
        stored = memory_engine.add_memory.await_args.kwargs
        assert stored["session_id"] == scope
        assert [item["content"] for item in stored["source_messages"]] == ["hello", "hi"]
        window = stored["metadata"]["source_window"]
        assert window["session_id"] == session_id
        assert (window["start_index"], window["end_index"], window["message_count"]) == (0, 2, 2)
        assert window["end_message_id"] == messages[-1].id
        assert await manager.get_session_metadata(session_id, "last_summarized_index", 0) == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_storage_task_does_not_retain_source_below_threshold(
    tmp_path, memory_engine, memory_processor
):
    handler, manager, store = await _make_real_handler(tmp_path, memory_engine)
    handler._memory_reflection.memory_processor = memory_processor
    memory_processor.process_conversation.return_value = ("summary", {"topics": ["t1"]}, 0.79)
    session_id = "test:private:sid-1"
    try:
        messages = await _seed_summary_window(manager, session_id)
        await handler._memory_reflection._storage_task(session_id, messages, "p1", 0, 2, 0)

        memory_engine.add_memory.assert_awaited_once()
        assert memory_engine.add_memory.await_args.kwargs["source_messages"] is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_storage_task_skips_when_already_summarized(
    handler, conversation_manager, memory_engine
):
    """当 last_summarized_index >= end_index 时，_storage_task 应直接跳过。"""
    from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message

    # 模拟已经总结到 end_index=5
    conversation_manager.get_session_metadata = AsyncMock(return_value=5)

    messages = [
        Message(
            id=1,
            session_id="s1",
            role="user",
            content="msg",
            sender_id="u1",
            sender_name="U",
            group_id=None,
            platform="test",
            metadata={},
        )
    ]

    await handler._memory_reflection._storage_task(
        session_id="s1",
        history_messages=messages,
        persona_id=None,
        start_index=0,
        end_index=5,  # end_index == current_summarized → 过期任务
        retry_count=0,
    )

    # 过期任务不应调用 add_memory
    memory_engine.add_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_memory_recall_prefers_direct_plain_over_flattened_text(
    handler, memory_engine
):
    event = _make_event(text="direct Plain")
    event.get_messages = Mock(return_value=[Plain("direct Plain")])
    event.message_str = Mock()
    event.get_message_str = Mock(return_value="untrusted flattened quote")
    req = _make_req("generated envelope")

    await handler.handle_memory_recall(event, req)
    await handler.shutdown()

    memory_engine.search_memories.assert_awaited_once()
    assert memory_engine.search_memories.await_args.kwargs["query"] == "direct Plain"
    event.get_message_str.assert_not_called()


@pytest.mark.asyncio
async def test_handle_memory_recall_uses_original_text_when_prompt_empty(
    handler, memory_engine
):
    event = _make_event(text="describe image")
    req = _make_req(prompt="")
    req.extra_user_content_parts = [TextPart(text="<image_caption>cat</image_caption>")]

    await handler.handle_memory_recall(event, req)
    await handler.shutdown()

    memory_engine.search_memories.assert_awaited_once()
    assert memory_engine.search_memories.await_args.kwargs["query"] == "describe image"
    assert len(req.extra_user_content_parts) == 1
    assert req.extra_user_content_parts[0].text == "<image_caption>cat</image_caption>"
    handler.conversation_manager.add_message_from_event.assert_awaited_once_with(
        event=event, role="user", content="describe image"
    )


@pytest.mark.asyncio
async def test_handle_memory_reflection_skips_error_response(
    handler, conversation_manager, memory_engine
):
    """包含错误指示词的响应应被跳过，不触发记忆存储。"""
    event = _make_event(group=False)
    resp = _make_resp("api error: rate limit exceeded")

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "p1"
        await handler.handle_memory_reflection(event, resp)

    # 错误响应不应触发任何存储
    memory_engine.add_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_memory_reflection_skips_empty_response(
    handler, conversation_manager, memory_engine
):
    """空响应应被跳过，不触发记忆存储。"""
    event = _make_event(group=False)
    resp = _make_resp("")

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "p1"
        await handler.handle_memory_reflection(event, resp)

    memory_engine.add_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_memory_reflection_pending_retry_enters_cooldown(
    tmp_path, memory_engine, memory_processor
):
    handler, manager, store = await _make_real_handler(tmp_path, memory_engine)
    handler._memory_reflection.memory_processor = memory_processor
    handler.config_manager._config["reflection_engine"]["summary_trigger_rounds"] = 1
    event = _make_event()
    session_id = event.unified_msg_origin
    try:
        await _seed_summary_window(manager, session_id)
        await manager.update_session_metadata(
            session_id, "pending_summary", {"start_index": 0, "end_index": 2, "retry_count": 3}
        )

        await handler.handle_memory_reflection(event, _make_resp())
        await handler.shutdown()

        pending = await manager.get_session_metadata(session_id, "pending_summary", None)
        assert pending["failure_stage"] == "retry_exhausted"
        assert pending["retry_count"] == 0
        assert pending["exhausted_retry_count"] == 3
        assert pending["retry_after"] > time.time()
        assert (pending["start_index"], pending["end_index"]) == (0, 2)
        assert await manager.get_session_metadata(session_id, "last_summarized_index", 0) == 0
        memory_engine.add_memory.assert_not_awaited()
        memory_processor.process_conversation.assert_not_awaited()
    finally:
        await store.close()
