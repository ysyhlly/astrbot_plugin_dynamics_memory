"""Budget boundaries for the long-term-memory agent tool."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.recall_runtime import (
    BoundedRecallTaskPool,
    RecallDeadline,
)
from astrbot_plugin_dynamics_memory.core.tools.memory_search_tool import (
    MemorySearchTool,
)


def _context():
    event = Mock()
    event.unified_msg_origin = "test:private:budget"
    event._cd_tool_memory_sources_register_v1 = None
    run_context = Mock()
    run_context.context.event = event
    return run_context


@pytest.mark.asyncio
async def test_tool_returns_within_one_absolute_budget_when_engine_stalls():
    engine = Mock()
    started = asyncio.Event()

    async def stalled_search(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    engine.search_memories = AsyncMock(side_effect=stalled_search)
    tool = MemorySearchTool(
        context=Mock(),
        config_manager=ConfigManager(),
        memory_engine=engine,
    )
    real_start = RecallDeadline.start

    def short_deadline(*_args, **_kwargs):
        return real_start(total_seconds=0.24, final_reserve_seconds=0.08)

    with (
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.RecallDeadline.start",
            side_effect=short_deadline,
        ),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
            new=AsyncMock(return_value="persona-a"),
        ),
    ):
        began = time.monotonic()
        task = asyncio.create_task(tool.call(_context(), query="bounded"))
        await asyncio.wait_for(started.wait(), timeout=0.2)
        result = json.loads(await asyncio.wait_for(task, timeout=0.4))
        elapsed = time.monotonic() - began

    assert elapsed < 0.4
    assert result["status"] == "deadline_exceeded"
    assert result["results"] == []
    engine.search_memories.assert_awaited_once()
    kwargs = engine.search_memories.await_args.kwargs
    assert kwargs["touch_access"] is False
    assert kwargs["cache_results"] is False
    assert kwargs["allow_legacy_session_migration"] is False
    assert kwargs["execution_context"].budget.final_reserve_seconds == 0.08


@pytest.mark.asyncio
async def test_closed_search_pool_is_unavailable_not_busy():
    engine = Mock()
    pool = BoundedRecallTaskPool()
    await pool.close()

    async def closed_search(**kwargs):
        outcome = await kwargs["execution_context"].run(
            "closed-route",
            lambda: [],
        )
        return outcome.get("value") or []

    engine.search_memories = AsyncMock(side_effect=closed_search)
    tool = MemorySearchTool(
        context=Mock(),
        config_manager=ConfigManager(),
        memory_engine=engine,
        recall_runtime=pool,
    )

    with patch(
        "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
        new=AsyncMock(return_value="persona-a"),
    ):
        result = json.loads(await tool.call(_context(), query="closed"))

    assert result["status"] == "unavailable"
    assert result["results"] == []
    assert "closed" in result["diagnostics"]["route_states"].values()
    engine.search_memories.assert_awaited_once()
