"""
Event handler tests: prompt fallback, original input and summary windows.
"""

import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler

from tests.event_handler_support import (
    _make_event,
    _make_real_handler,
    _make_req,
    _make_resp,
    _recalled,
    _seed_turns,
)

# ==================== system_prompt 自动回退测试 ====================


@pytest.mark.asyncio
async def test_system_prompt_auto_falls_back_to_extra_user_content(
    handler, memory_engine
):
    config = ConfigManager({"recall_engine": {"injection_method": "system_prompt"}})
    handler._memory_recall.config_manager = config
    memory_engine.search_memories.return_value = [_recalled("mem_fallback")]
    req = _make_req("query text")
    req.system_prompt = "persona text"

    await handler.handle_memory_recall(_make_event(text="query text"), req)
    await handler.shutdown()

    assert req.system_prompt == "persona text"
    assert len(req.extra_user_content_parts) == 1
    assert "mem_fallback" in req.extra_user_content_parts[0].text
    assert req.extra_user_content_parts[0]._no_save is True


# ==================== 原始输入测试 ====================


@pytest.mark.asyncio
async def test_recall_does_not_expand_query_with_recent_assistant_text(
    handler, memory_engine, conversation_manager
):
    """Accept the old config field without bringing assistant text into retrieval."""
    handler._memory_recall.config_manager = ConfigManager(
        {"recall_engine": {"top_k": 3, "inject_with_recent_context": True}}
    )
    conversation_manager.get_context = AsyncMock(
        return_value=[
            {"content": "当前用户消息", "timestamp": time.time()},
            {"content": "Bot 的上一条回复", "timestamp": time.time() - 10},
            {"content": "用户之前说的事情", "timestamp": time.time() - 20},
        ]
    )
    req = _make_req("generated envelope")
    req.contexts = [{"role": "assistant", "content": "Current request history"}]

    await handler.handle_memory_recall(_make_event(text="当前用户消息"), req)
    await handler.shutdown()

    memory_engine.search_memories.assert_awaited_once()
    assert memory_engine.search_memories.await_args.kwargs["query"] == "当前用户消息"
    conversation_manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_ignores_recent_stale_and_undated_history(
    handler, memory_engine, conversation_manager
):
    now = time.time()
    handler._memory_recall.config_manager = ConfigManager(
        {
            "recall_engine": {
                "top_k": 3,
                "inject_with_recent_context": True,
                "recent_context_max_age_seconds": 7200,
            }
        }
    )
    conversation_manager.get_context = AsyncMock(
        return_value=[
            {"content": "当前消息", "timestamp": now},
            {"content": "一小时内", "timestamp": now - 1800},
            {"content": "三小时前", "timestamp": now - 10800},
            {"content": "没有时间"},
        ]
    )

    await handler.handle_memory_recall(
        _make_event(text="当前消息"), _make_req("前轮重写请求")
    )
    await handler.shutdown()

    memory_engine.search_memories.assert_awaited_once()
    assert memory_engine.search_memories.await_args.kwargs["query"] == "当前消息"
    conversation_manager.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_with_no_history_uses_original_text(
    handler, memory_engine, conversation_manager
):
    conversation_manager.get_context = AsyncMock(return_value=[])

    await handler.handle_memory_recall(
        _make_event(text="唯一一条消息"), _make_req("generated envelope")
    )
    await handler.shutdown()

    memory_engine.search_memories.assert_awaited_once()
    assert memory_engine.search_memories.await_args.kwargs["query"] == "唯一一条消息"
    conversation_manager.get_context.assert_not_awaited()


# ==================== 总结重试测试 ====================


@pytest.mark.asyncio
async def test_exhausted_legacy_window_with_no_matching_start_does_not_advance_cursor(
    tmp_path, memory_engine
):
    handler, manager, store = await _make_real_handler(tmp_path, memory_engine)
    handler.config_manager._config["reflection_engine"]["summary_trigger_rounds"] = 1
    event = _make_event()
    session_id = event.unified_msg_origin
    try:
        await _seed_turns(manager, session_id, ("C1", "C2", "C3"))
        await manager.update_session_metadata(
            session_id,
            "pending_summary",
            {"start_index": 2, "end_index": 6, "retry_count": 3},
        )

        await handler.handle_memory_reflection(event, _make_resp())
        await handler.shutdown()

        assert (
            await manager.get_session_metadata(session_id, "last_summarized_index", 0)
            == 0
        )
        pending = await manager.get_session_metadata(
            session_id, "pending_summary", None
        )
        assert pending["failure_stage"] == "retry_exhausted"
        assert pending["retry_after"] > time.time()
        assert pending["start_index"] == 0
        assert await store.get_message_count(session_id) == 7
        memory_engine.add_memory.assert_not_awaited()
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_without_end", [False, True])
async def test_pending_summary_retry_keeps_exact_range_or_uses_bounded_legacy_fallback(
    handler,
    conversation_manager,
    legacy_without_end,
):
    """Retry exact known bounds; a legacy start-only range uses one bounded window."""
    pending = {"start_index": 2, "retry_count": 1}
    if not legacy_without_end:
        pending["end_index"] = 6
    conversation_manager.get_session_metadata = AsyncMock(
        side_effect=lambda sid, key, default=None: {
            "last_summarized_index": 5,
            "pending_summary": pending,
        }.get(key, default)
    )
    conversation_manager.store.get_message_count = AsyncMock(return_value=12)
    end_index = 12 if legacy_without_end else 6
    msgs = [Mock(id=i + 1, group_id=None) for i in range(end_index - 2)]
    conversation_manager.get_messages_range = AsyncMock(return_value=msgs)
    handler._memory_reflection._schedule_reflection = AsyncMock()

    event = _make_event(group=False)
    resp = _make_resp("assistant reply")

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_1"
        await handler.handle_memory_reflection(event, resp)
    conversation_manager.get_messages_range.assert_awaited_once()
    call_kwargs = conversation_manager.get_messages_range.await_args.kwargs
    assert call_kwargs["start_index"] == 2
    assert call_kwargs["end_index"] == end_index
    plan = handler._memory_reflection._schedule_reflection.await_args.args[0]
    assert (plan.start_index, plan.end_index, plan.retry_count) == (2, end_index, 1)
    assert plan.history_messages == msgs


# ==================== #255 总结窗口按天/空闲间隔切分 ====================


def _window_handler(
    memory_engine, memory_processor, conversation_manager, **reflection_cfg
):
    """构建 reflection_engine 可定制的 EventHandler（其余依赖沿用 mock）。"""
    return EventHandler(
        context=Mock(),
        config_manager=ConfigManager(
            {
                "recall_engine": {"top_k": 3, "injection_method": "extra_user_content"},
                "reflection_engine": {"summary_trigger_rounds": 1, **reflection_cfg},
                "session_manager": {"max_messages_per_session": 100},
            }
        ),
        memory_engine=memory_engine,
        memory_processor=memory_processor,
        conversation_manager=conversation_manager,
    )


def _ts(day: int, hour: int, minute: int) -> float:
    return datetime(2025, 3, day, hour, minute).timestamp()


def _day_split_timestamps() -> list[float]:
    """4 条 03-01 晚间 + 8 条 03-02 上午：第一个边界在第 4 条之后。"""
    return [_ts(1, 22, i) for i in range(4)] + [_ts(2, 9, i) for i in range(8)]


def _gap_split_timestamps() -> list[float]:
    """6 条 09:00-09:05 + 6 条 12:06-12:11：同一天，间隔 181 分钟。"""
    return [_ts(1, 9, i) for i in range(6)] + [_ts(1, 12, 6 + i) for i in range(6)]


def _same_day_timestamps() -> list[float]:
    """12 条同一天、间隔 1 分钟的消息：没有边界，保持整段总结。"""
    return [_ts(1, 9, i) for i in range(12)]


@pytest.mark.parametrize(
    ("reflection_cfg", "timestamps", "expected_end"),
    [
        ({}, _day_split_timestamps(), 4),
        ({}, _gap_split_timestamps(), 6),
        ({}, _same_day_timestamps(), 12),
        ({"window_split_gap_minutes": 0}, _gap_split_timestamps(), 12),
        (
            {"window_split_by_day": False, "window_split_gap_minutes": 0},
            _day_split_timestamps(),
            12,
        ),
    ],
    ids=["day-split", "gap-split", "no-boundary", "gap-disabled", "split-disabled"],
)
@pytest.mark.asyncio
async def test_reflection_window_split_chooses_oldest_segment(
    memory_engine,
    memory_processor,
    conversation_manager,
    reflection_cfg,
    timestamps,
    expected_end,
):
    """#255：触发总结时 end_index 收敛到第一个切分边界，只总结最老片段。"""
    conversation_manager.get_messages_range = AsyncMock(
        return_value=[SimpleNamespace(timestamp=ts) for ts in timestamps]
    )
    conversation_manager.store.get_message_count = AsyncMock(return_value=12)
    handler = _window_handler(
        memory_engine, memory_processor, conversation_manager, **reflection_cfg
    )
    handler._memory_reflection._storage_task = AsyncMock()

    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_1"
        await handler.handle_memory_reflection(_make_event(), _make_resp())
        await handler.shutdown()

    # 切分扫描与总结输入复用同一次范围读取
    conversation_manager.get_messages_range.assert_awaited_once()
    call_kwargs = conversation_manager.get_messages_range.await_args.kwargs
    assert call_kwargs["start_index"] == 0
    assert call_kwargs["end_index"] == 12

    storage_call = handler._memory_reflection._storage_task.await_args
    assert storage_call is not None
    assert storage_call.args[3] == 0  # start_index 不变
    assert storage_call.args[4] == expected_end
    assert len(storage_call.args[1]) == expected_end
