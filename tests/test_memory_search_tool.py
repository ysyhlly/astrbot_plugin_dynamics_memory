"""Tests for the active long-term memory search tool."""

import asyncio
import contextlib
import json
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.memory_scope import GLOBAL_MEMORY_SCOPE
from astrbot_plugin_dynamics_memory.core.tools.memory_search_tool import (
    MemorySearchTool,
)


SESSION_ID = "test:private:session-1"


@pytest.fixture
def memory_engine():
    engine = Mock()
    engine.search_memories = AsyncMock(return_value=[])
    return engine


@pytest.fixture
def astr_context():
    return Mock()


def _make_run_context():
    event = Mock()
    event.unified_msg_origin = "test:private:session-1"
    # 真实事件不带 CD 的 owned 标记、也没有 tool 源注册回调；Mock 会自动生成任意
    # 属性，而产品把非法 owned 标记/非 True 的注册结果视为不可用，故显式置 None。
    event._chat_dynamics_owned_request = None
    event._cd_owned_original_input_v1 = None
    event._cd_tool_memory_sources_register_v1 = None
    event.get_extra = Mock(return_value=None)

    run_context = Mock()
    run_context.context = Mock()
    run_context.context.event = event
    return run_context


def _fake_plugin(memory_engine):
    """产品在返回结果前会做 source 校验，需要一个"活着的插件实例"外形。"""
    plugin = Mock()
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = memory_engine
    plugin.peek_evidence_revision = Mock(
        return_value={"available": True, "correction_revision": None}
    )
    plugin._native_read_scopes = AsyncMock(return_value=[])
    plugin._native_verified_source_row = AsyncMock(return_value=None)
    plugin._native_memory_source_row_ids = AsyncMock(return_value=[])
    # 产品用插件侧的 scope 归一化把"读取来源 umo"转成四字段 scope；
    # 替身必须如实返回 dict（Mock 默认返回值会被判为非 dict 而跳过整条读取路径）。
    plugin._native_scope_from_request = Mock(
        side_effect=lambda request_scope, source_umo: {
            "platform": (request_scope or {}).get("platform", "test"),
            "umo": source_umo,
            "persona_id": (request_scope or {}).get("persona_id", ""),
            "instance_id": (request_scope or {}).get("instance_id", ""),
        }
    )
    return plugin


@contextlib.contextmanager
def _tool_seams(memory_engine):
    """把"插件实例 + 源校验"两个外部依赖替换为可控替身。

    校验器本身（make_memory_source_validator）有独立覆盖；这里只让被校验的
    文档快照通过，从而专注断言工具的序列化结果。
    """

    async def _valid():
        return True

    with (
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_active_plugin",
            return_value=_fake_plugin(memory_engine),
        ),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.make_memory_source_validator",
            return_value=_valid,
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_memory_search_tool_uses_filtering_settings(memory_engine, astr_context):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(
            {
                "recall_engine": {"top_k": 3, "max_k": 8},
                "filtering_settings": {
                    "use_session_filtering": True,
                    "use_persona_filtering": True,
                },
            }
        ),
        memory_engine=memory_engine,
    )
    memory_engine.search_memories = AsyncMock(return_value=[])

    with patch(
        "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_a"
        raw_result = await tool.call(_make_run_context(), query="喜欢的游戏", k=6)

    result = json.loads(raw_result)
    assert result["query"] == "喜欢的游戏"
    assert result["applied_filters"] == {
        "session_filtered": True,
        "persona_filtered": True,
    }
    memory_engine.search_memories.assert_awaited_once_with(
        query="喜欢的游戏",
        k=6,
        session_id="test:private:session-1",
        persona_id="persona_a",
        touch_access=False,
        cache_results=False,
        allow_legacy_session_migration=False,
        execution_context=ANY,
    )


@pytest.mark.asyncio
async def test_memory_search_tool_disables_filters_when_config_disabled(
    memory_engine, astr_context
):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(
            {
                "filtering_settings": {
                    "use_session_filtering": False,
                    "use_persona_filtering": False,
                }
            }
        ),
        memory_engine=memory_engine,
    )

    with patch(
        "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_a"
        await tool.call(_make_run_context(), query="项目约定")

    memory_engine.search_memories.assert_awaited_once_with(
        query="项目约定",
        k=5,
        # 作用域解析失败时回退到原始会话（与写入侧语义一致）
        session_id=_make_run_context().context.event.unified_msg_origin,
        persona_id=None,
        touch_access=False,
        cache_results=False,
        allow_legacy_session_migration=False,
        execution_context=ANY,
    )
    get_persona.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_search_tool_uses_user_scope(memory_engine, astr_context):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(
            {
                "filtering_settings": {
                    "memory_scope_mode": "user",
                    "use_persona_filtering": False,
                }
            }
        ),
        memory_engine=memory_engine,
    )
    run_context = _make_run_context()
    event = run_context.context.event
    event.get_platform_name = Mock(return_value="test")
    event.get_sender_id = Mock(return_value="user-1")

    await tool.call(run_context, query="跨会话项目")

    memory_engine.search_memories.assert_awaited_once_with(
        query="跨会话项目",
        k=5,
        session_id="livingmemory:user:test:user-1",
        persona_id=None,
        touch_access=False,
        cache_results=False,
        allow_legacy_session_migration=False,
        execution_context=ANY,
    )


@pytest.mark.parametrize(
    ("filtering_settings", "expected_session_id", "expected_session_filtered"),
    [
        ({"use_session_filtering": True}, SESSION_ID, True),
        ({"use_session_filtering": False}, SESSION_ID, True),
        ({"memory_scope_mode": "session"}, SESSION_ID, True),
        ({"memory_scope_mode": "global"}, GLOBAL_MEMORY_SCOPE, False),
        ({"memory_scope_mode": "user"}, "livingmemory:user:test:user-1", False),
        # isolated_sessions 始终强制会话级作用域，即使配置为 global
        (
            {"memory_scope_mode": "global", "isolated_sessions": SESSION_ID},
            SESSION_ID,
            True,
        ),
    ],
    ids=[
        "legacy-session-filtering",
        "legacy-no-filtering",
        "session",
        "global",
        "user",
        "isolated-session-overrides-global",
    ],
)
@pytest.mark.asyncio
async def test_memory_search_tool_reports_actual_session_filtering(
    memory_engine,
    astr_context,
    filtering_settings,
    expected_session_id,
    expected_session_filtered,
):
    """Distinguish an exact origin constraint from a named storage pool."""
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(
            {
                "filtering_settings": {
                    **filtering_settings,
                    "use_persona_filtering": False,
                }
            }
        ),
        memory_engine=memory_engine,
    )
    run_context = _make_run_context()
    event = run_context.context.event
    event.get_platform_name = Mock(return_value="test")
    event.get_sender_id = Mock(return_value="user-1")

    result = json.loads(await tool.call(run_context, query="scope report"))

    assert result["applied_filters"] == {
        "session_filtered": expected_session_filtered,
        "persona_filtered": False,
    }
    assert memory_engine.search_memories.await_args.kwargs["session_id"] == (
        expected_session_id
    )


@pytest.mark.asyncio
async def test_memory_search_tool_denies_unlisted_user(memory_engine, astr_context):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(
            {
                "access_control": {
                    "whitelist_enabled": True,
                    "allowed_ids": "another-user",
                }
            }
        ),
        memory_engine=memory_engine,
    )
    run_context = _make_run_context()
    run_context.context.event.get_sender_id = Mock(return_value="user-1")

    result = json.loads(await tool.call(run_context, query="private memory"))

    assert result["error"] == "memory access is not allowed"
    memory_engine.search_memories.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_search_tool_serializes_results(memory_engine, astr_context):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(),
        memory_engine=memory_engine,
    )
    memory_engine.search_memories = AsyncMock(
        return_value=[
            Mock(
                doc_id=7,
                content="用户喜欢高难度动作游戏",
                final_score=0.91,
                metadata={
                    "importance": 0.8,
                    "session_id": "test:private:session-1",
                    "persona_id": "persona_a",
                    "create_time": 100.0,
                    "last_access_time": 200.0,
                },
            )
        ]
    )
    memory_engine.get_memory_source = AsyncMock()

    with (
        _tool_seams(memory_engine),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
            new_callable=AsyncMock,
        ) as get_persona,
    ):
        get_persona.return_value = "persona_a"
        raw_result = await tool.call(_make_run_context(), query="游戏偏好")

    result = json.loads(raw_result)
    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["results"][0] == {
        "id": 7,
        "content": "用户喜欢高难度动作游戏",
        "score": 0.91,
        "importance": 0.8,
        "session_id": "test:private:session-1",
        "persona_id": "persona_a",
        "create_time": 100.0,
        "last_access_time": 200.0,
    }
    memory_engine.get_memory_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_search_tool_includes_retained_source_on_request(
    memory_engine, astr_context
):
    memory_engine.search_memories = AsyncMock(
        return_value=[
            Mock(
                doc_id=9,
                content="summary",
                final_score=0.9,
                metadata={
                    "has_source": True,
                    "source_session_id": _make_run_context().context.event.unified_msg_origin,
                },
            )
        ]
    )
    memory_engine.get_memory_source = AsyncMock(
        return_value=[
            {
                "role": "user",
                "content": "exact detail",
                "session_id": _make_run_context().context.event.unified_msg_origin,
            }
        ]
    )
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(),
        memory_engine=memory_engine,
    )

    plugin = _fake_plugin(memory_engine)
    plugin._native_memory_source_row_ids = AsyncMock(return_value=[5])
    plugin._native_verified_source_row = AsyncMock(
        return_value={
            "role": "user",
            "content": "exact detail",
            "session_id": _make_run_context().context.event.unified_msg_origin,
            "sender_id": "user-1",
            "sender_name": "Tester",
            "platform": "test",
            "timestamp": 100.0,
        }
    )
    with (
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_active_plugin",
            return_value=plugin,
        ),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.make_memory_source_validator",
            return_value=AsyncMock(return_value=True),
        ),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
            new=AsyncMock(return_value="persona_a"),
        ),
    ):
        result = json.loads(
            await tool.call(_make_run_context(), query="details", include_source=True)
        )

    assert result["results"][0]["source_messages"][0]["content"] == "exact detail"
    # 产品改为经插件侧的"已校验源行"接口取原文（不再走 engine.get_memory_source）
    plugin._native_verified_source_row.assert_awaited_once()
    memory_engine.get_memory_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_search_tool_serializes_non_dict_metadata(
    memory_engine, astr_context
):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(),
        memory_engine=memory_engine,
    )
    memory_engine.search_memories = AsyncMock(
        return_value=[
            Mock(
                doc_id=8,
                content="用户提到过想学摄影",
                final_score=0.55,
                metadata=None,
            )
        ]
    )

    with (
        _tool_seams(memory_engine),
        patch(
            "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
            new_callable=AsyncMock,
        ) as get_persona,
    ):
        get_persona.return_value = "persona_a"
        raw_result = await tool.call(_make_run_context(), query="摄影")

    result = json.loads(raw_result)
    # 新契约：无法定位来源 scope 的记忆不会崩溃，而是被标记为 partial 并丢弃
    # （工具必须先通过"来源已校验"才会返回结果）。断言"不崩溃 + 明确状态"。
    assert result["status"] == "unavailable"
    assert result["count"] == 0
    assert result["results"] == []


@pytest.mark.asyncio
async def test_memory_search_tool_limits_k_by_config(memory_engine, astr_context):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager({"recall_engine": {"top_k": 5, "max_k": 4}}),
        memory_engine=memory_engine,
    )

    with patch(
        "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_a"
        await tool.call(_make_run_context(), query="偏好", k=9)

    memory_engine.search_memories.assert_awaited_once_with(
        query="偏好",
        k=4,
        session_id="test:private:session-1",
        persona_id="persona_a",
        touch_access=False,
        cache_results=False,
        allow_legacy_session_migration=False,
        execution_context=ANY,
    )


@pytest.mark.asyncio
async def test_memory_search_tool_clamps_low_k_to_one(memory_engine, astr_context):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(
            {
                "recall_engine": {"top_k": 3, "max_k": 8},
                "filtering_settings": {
                    "use_session_filtering": True,
                    "use_persona_filtering": True,
                },
            }
        ),
        memory_engine=memory_engine,
    )
    memory_engine.search_memories = AsyncMock(return_value=[])

    with patch(
        "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_a"

        await tool.call(_make_run_context(), query="test query", k=0)
        called_k_zero = memory_engine.search_memories.await_args.kwargs["k"]
        assert called_k_zero == 1

        memory_engine.search_memories.reset_mock()
        await tool.call(_make_run_context(), query="test query", k=-3)
        called_k_negative = memory_engine.search_memories.await_args.kwargs["k"]
        assert called_k_negative == 1


@pytest.mark.asyncio
async def test_memory_search_tool_falls_back_to_default_k_for_invalid_input(
    memory_engine, astr_context
):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager({"recall_engine": {"top_k": 3, "max_k": 8}}),
        memory_engine=memory_engine,
    )
    memory_engine.search_memories = AsyncMock(return_value=[])

    with patch(
        "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_a"
        await tool.call(_make_run_context(), query="test query", k="bad")

    memory_engine.search_memories.assert_awaited_once_with(
        query="test query",
        k=3,
        session_id="test:private:session-1",
        persona_id="persona_a",
        touch_access=False,
        cache_results=False,
        allow_legacy_session_migration=False,
        execution_context=ANY,
    )


@pytest.mark.asyncio
async def test_memory_search_tool_returns_structured_error_for_empty_query(
    memory_engine, astr_context
):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(),
        memory_engine=memory_engine,
    )

    raw_result = await tool.call(_make_run_context(), query="   ")
    result = json.loads(raw_result)

    assert result["results"] == []
    assert result["error"] == "query is empty"
    memory_engine.search_memories.assert_not_called()


@pytest.mark.asyncio
async def test_memory_search_tool_returns_not_initialized_error(memory_engine):
    tool = MemorySearchTool(
        context=None,
        config_manager=None,
        memory_engine=memory_engine,
    )

    raw_result = await tool.call(_make_run_context(), query="测试")
    result = json.loads(raw_result)

    # 新的工具契约：固定状态 + 固定错误串 + 结构化 diagnostics（不含内部细节）
    assert result["status"] == "unavailable"
    assert result["query"] == "测试"
    assert result["count"] == 0
    assert result["results"] == []
    assert result["error"] == "memory search tool is not initialized"
    assert set(result["diagnostics"]) == {
        "elapsed_ms",
        "route_states",
        "inflight_count",
    }
    memory_engine.search_memories.assert_not_called()


@pytest.mark.asyncio
async def test_memory_search_tool_hides_internal_exception_details(
    memory_engine, astr_context
):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(),
        memory_engine=memory_engine,
    )
    memory_engine.search_memories = AsyncMock(
        side_effect=RuntimeError("secret db path")
    )

    with patch(
        "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_a"
        raw_result = await tool.call(_make_run_context(), query="异常")

    result = json.loads(raw_result)
    # 检索路由内部异常会被记录为 route 状态，工具对外只给固定状态；
    # 关键断言（不泄漏内部异常文本）保持不变。
    assert result["status"] == "unavailable"
    assert result["count"] == 0
    assert result["results"] == []
    assert result["diagnostics"]["route_states"].get("scope_1") == "error"
    assert "secret db path" not in raw_result
    assert "RuntimeError" not in raw_result


@pytest.mark.asyncio
async def test_memory_search_tool_propagates_cancellation(memory_engine, astr_context):
    tool = MemorySearchTool(
        context=astr_context,
        config_manager=ConfigManager(),
        memory_engine=memory_engine,
    )
    memory_engine.search_memories = AsyncMock(side_effect=asyncio.CancelledError())

    with patch(
        "astrbot_plugin_dynamics_memory.core.tools.memory_search_tool.get_persona_id",
        new_callable=AsyncMock,
    ) as get_persona:
        get_persona.return_value = "persona_a"
        with pytest.raises(asyncio.CancelledError):
            await tool.call(_make_run_context(), query="取消")
