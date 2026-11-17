"""Shared fixture exports and the deprecated-mode logging regression."""

import pytest
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager

from tests.event_handler_support import (
    _make_event,
    _make_req,
    _recalled,
    conversation_manager,
    handler,
    memory_engine,
    memory_processor,
)


@pytest.mark.asyncio
async def test_handle_memory_recall_fake_tool_call_fallback_logs_once(handler, memory_engine, caplog):
    config = ConfigManager({"recall_engine": {"injection_method": "fake_tool_call"}})
    handler._memory_recall.config_manager = config
    memory_engine.search_memories.return_value = [_recalled("记忆内容")]

    for text in ("今天吃什么", "昨天吃什么"):
        req = _make_req(text)
        await handler.handle_memory_recall(_make_event(text=text), req)
        assert req._lm_private_hook_outcome_v1 == "ready"
        assert len(req.extra_user_content_parts) == 1

    await handler.shutdown()
    warnings = [
        record.message for record in caplog.records
        if record.levelname == "WARNING" and "fake_tool_call" in record.message
    ]
    assert len(warnings) == 1
    assert "已废弃" in warnings[0]
    assert "extra_user_content" in warnings[0]
