"""Beta 自主回忆规则注入的测试。"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler

from tests.event_handler_support import _make_event as _make_native_event
from tests.event_handler_support import _make_req as _make_provider_request
from tests.event_handler_support import _recalled

POLICY_HEADER = "<Agent-Recall-Policy>"
POLICY_FOOTER = "</Agent-Recall-Policy>"


def _make_handler(agent_tools, **recall_engine):
    recall_cfg = {"top_k": 3, "injection_method": "extra_user_content"}
    recall_cfg.update(recall_engine)
    config_manager = ConfigManager(
        {
            "recall_engine": recall_cfg,
            "reflection_engine": {"summary_trigger_rounds": 1},
            "session_manager": {"max_messages_per_session": 100},
            "agent_tools": agent_tools,
        }
    )
    memory_engine = Mock()
    memory_engine.search_memories = AsyncMock(return_value=[])
    memory_engine._create_tracked_task = None
    memory_processor = Mock()
    conversation_manager = Mock()
    conversation_manager.add_message_from_event = AsyncMock(
        return_value=Mock(id=1, metadata={})
    )
    conversation_manager.store = Mock()
    conversation_manager.store.get_message_count = AsyncMock(return_value=12)
    conversation_manager.store.connection = Mock()
    conversation_manager.store.connection.execute = AsyncMock(
        return_value=Mock(rowcount=1)
    )
    conversation_manager.store.connection.commit = AsyncMock()
    handler = EventHandler(
        context=Mock(),
        config_manager=config_manager,
        memory_engine=memory_engine,
        memory_processor=memory_processor,
        conversation_manager=conversation_manager,
    )
    return handler, memory_engine, conversation_manager


def _make_event():
    return _make_native_event(text="还记得我吗")


def _make_req(tool_names=None):
    req = _make_provider_request("还记得我吗")
    if tool_names is not None:
        req.func_tool = Mock(names=Mock(return_value=list(tool_names)))
    return req


def _policy_parts(req):
    return [
        part
        for part in req.extra_user_content_parts
        if POLICY_HEADER in getattr(part, "text", "")
    ]


async def _run(handler, event, req):
    with patch(
        "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall.get_persona_id",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await handler.handle_memory_recall(event, req)


class TestPolicyInjection:
    @pytest.mark.asyncio
    async def test_beta_on_injects_policy_with_available_tools(self):
        handler, engine, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        event = _make_event()
        req = _make_req(["recall_long_term_memory", "read_memory_evidence"])
        await _run(handler, event, req)

        policies = _policy_parts(req)
        assert len(policies) == 1
        text = policies[0].text
        assert POLICY_FOOTER in text
        assert "{available_tools}" not in text
        assert "recall_long_term_memory" in text
        assert "read_memory_evidence" in text
        assert getattr(policies[0], "_no_save", False) is True
        # 不改写 prompt/system_prompt/history
        assert req.prompt == "还记得我吗"
        assert req.system_prompt == ""
        assert req.contexts == []
        # 自动召回仍正常运行（Beta 不改变自动流程），但 query 不得混入规则
        engine.search_memories.assert_awaited_once()
        assert POLICY_HEADER not in engine.search_memories.await_args.kwargs["query"]

    @pytest.mark.asyncio
    async def test_beta_on_without_read_tool_omits_read_capability(self):
        handler, _, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        req = _make_req(["recall_long_term_memory"])
        await _run(handler, _make_event(), req)
        policies = _policy_parts(req)
        assert len(policies) == 1
        assert "read_memory_evidence" not in policies[0].text

    @pytest.mark.asyncio
    async def test_beta_off_injects_nothing(self):
        handler, _, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": False}
        )
        req = _make_req(["recall_long_term_memory", "read_memory_evidence"])
        await _run(handler, _make_event(), req)
        assert _policy_parts(req) == []

    @pytest.mark.asyncio
    async def test_beta_defaults_off_when_config_missing(self):
        handler, _, _ = _make_handler({"enable_recall_tool": True})
        req = _make_req(["recall_long_term_memory"])
        await _run(handler, _make_event(), req)
        assert _policy_parts(req) == []

    @pytest.mark.asyncio
    async def test_recall_off_blocks_beta(self):
        """Beta 不得越过主动回忆主开关。"""
        handler, _, _ = _make_handler(
            {"enable_recall_tool": False, "enable_agentic_recall_beta": True}
        )
        req = _make_req(["recall_long_term_memory", "read_memory_evidence"])
        await _run(handler, _make_event(), req)
        assert _policy_parts(req) == []

    @pytest.mark.asyncio
    async def test_top_k_zero_still_injects_policy(self):
        handler, engine, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}, top_k=0
        )
        req = _make_req(["recall_long_term_memory", "read_memory_evidence"])
        await _run(handler, _make_event(), req)
        assert len(_policy_parts(req)) == 1
        engine.search_memories.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_auto_hit_still_injects_policy(self):
        handler, engine, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        req = _make_req(["recall_long_term_memory"])
        await _run(handler, _make_event(), req)
        assert len(_policy_parts(req)) == 1

    @pytest.mark.asyncio
    async def test_tool_not_in_request_skips_policy(self):
        handler, _, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        req = _make_req(["some_other_tool"])
        await _run(handler, _make_event(), req)
        assert _policy_parts(req) == []

    @pytest.mark.asyncio
    async def test_unresolvable_func_tool_skips_policy(self):
        handler, _, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        req = _make_req(None)
        await _run(handler, _make_event(), req)
        assert _policy_parts(req) == []

    @pytest.mark.asyncio
    async def test_func_tool_names_error_skips_policy(self):
        handler, _, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        req = _make_req(["recall_long_term_memory"])
        req.func_tool.names = Mock(side_effect=RuntimeError("boom"))
        await _run(handler, _make_event(), req)
        assert _policy_parts(req) == []

    @pytest.mark.asyncio
    async def test_duplicate_hook_invocation_dedupes(self):
        handler, _, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        req = _make_req(["recall_long_term_memory"])
        await _run(handler, _make_event(), req)
        await _run(handler, _make_event(), req)
        assert len(_policy_parts(req)) == 1

    @pytest.mark.asyncio
    async def test_policy_not_stored_and_not_used_as_query(self):
        handler, engine, conversation_manager = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        engine.search_memories = AsyncMock(return_value=[_recalled("记忆内容")])
        event = _make_event()
        req = _make_req(["recall_long_term_memory"])
        await _run(handler, event, req)

        assert len(_policy_parts(req)) == 1
        stored = conversation_manager.add_message_from_event.await_args
        assert stored.kwargs["content"] == "还记得我吗"
        assert POLICY_HEADER not in stored.kwargs["content"]
        searched = engine.search_memories.await_args
        assert searched.kwargs["query"] == "还记得我吗"
        assert POLICY_HEADER not in searched.kwargs["query"]

    @pytest.mark.asyncio
    async def test_whitelist_denied_skips_everything(self):
        handler, _, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        handler._memory_recall.config_manager = ConfigManager(
            {
                "recall_engine": {"top_k": 3, "injection_method": "extra_user_content"},
                "reflection_engine": {"summary_trigger_rounds": 1},
                "session_manager": {"max_messages_per_session": 100},
                "agent_tools": {
                    "enable_recall_tool": True,
                    "enable_agentic_recall_beta": True,
                },
                "access_control": {"whitelist_enabled": True, "allowed_ids": ""},
            }
        )
        req = _make_req(["recall_long_term_memory"])
        await _run(handler, _make_event(), req)
        assert _policy_parts(req) == []

    @pytest.mark.asyncio
    async def test_malformed_owned_event_skips_policy_and_never_falls_back_to_plain(
        self,
    ):
        handler, engine, conversation_manager = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True}
        )
        event = _make_event()
        event._chat_dynamics_owned_request = True
        event._cd_owned_original_input_v1 = True
        event._cd_original_input_resolver_v1 = Mock(return_value=None)
        req = _make_req(["recall_long_term_memory"])

        await _run(handler, event, req)

        assert req._lm_private_hook_outcome_v1 == "original_input_unavailable"
        assert _policy_parts(req) == []
        engine.search_memories.assert_not_awaited()
        conversation_manager.add_message_from_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_uses_the_actual_deep_read_cursor_schema():
    from astrbot_plugin_dynamics_memory.core.tools.memory_read_tool import (
        MemoryReadTool,
    )

    handler, _, _ = _make_handler(
        {"enable_recall_tool": True, "enable_agentic_recall_beta": True}, top_k=0
    )
    req = _make_req(["recall_long_term_memory", "read_memory_evidence"])
    await handler.handle_memory_recall(_make_event(), req)
    text = _policy_parts(req)[0].text
    schema = MemoryReadTool().parameters["properties"]
    assert "cursor" in schema and "offset" not in schema and "limit" not in schema
    assert "offset/limit" not in text
    assert "next_cursor unchanged as cursor" in text
    assert "include_source=true" in text


class TestPolicyDedupIndependentOfLegacyCleanup:
    @pytest.mark.asyncio
    async def test_policy_deduplicates_when_legacy_cleanup_is_disabled(self):
        """R12：关掉旧的 auto_remove_injected 后，规则去重仍然生效。"""
        handler, _, _ = _make_handler(
            {"enable_recall_tool": True, "enable_agentic_recall_beta": True},
            auto_remove_injected=False,
        )
        req = _make_req(["recall_long_term_memory"])
        await _run(handler, _make_event(), req)
        await _run(handler, _make_event(), req)
        assert len(_policy_parts(req)) == 1


class TestPromptCatalogBetaVisibility:
    """R11：Beta 关闭时提示词管理页不展示新模板与分类。"""

    def _plugin(self, recall_on, beta_on):
        plugin = Mock()
        plugin.config_manager = ConfigManager(
            {
                "agent_tools": {
                    "enable_recall_tool": recall_on,
                    "enable_agentic_recall_beta": beta_on,
                }
            }
        )
        return plugin

    @pytest.mark.asyncio
    async def test_beta_off_prompt_catalog_has_no_beta_editor(self, tmp_path):
        from astrbot_plugin_dynamics_memory.core.page_api_modules.prompt_handler import (
            PromptHandler,
        )
        from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import (
            PageApiUtils,
        )
        from astrbot_plugin_dynamics_memory.core.prompts.prompt_manager import (
            init_prompt_manager,
        )

        init_prompt_manager(str(tmp_path))
        handler = PromptHandler(PageApiUtils(), self._plugin(True, False))
        result = await handler.list_prompts()
        assert result["status"] == "ok"
        data = result["data"]
        prompt_ids = {p["id"] for p in data["prompts"]}
        category_ids = {c["id"] for c in data["categories"]}
        assert "agent_recall_policy" not in prompt_ids
        assert "agent_recall" not in category_ids
        # 既有模板不受影响
        assert "memory_injection_header" in prompt_ids

    @pytest.mark.asyncio
    async def test_beta_on_prompt_catalog_shows_beta_editor(self, tmp_path):
        from astrbot_plugin_dynamics_memory.core.page_api_modules.prompt_handler import (
            PromptHandler,
        )
        from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import (
            PageApiUtils,
        )
        from astrbot_plugin_dynamics_memory.core.prompts.prompt_manager import (
            init_prompt_manager,
        )

        init_prompt_manager(str(tmp_path))
        handler = PromptHandler(PageApiUtils(), self._plugin(True, True))
        result = await handler.list_prompts()
        data = result["data"]
        prompt_ids = {p["id"] for p in data["prompts"]}
        category_ids = {c["id"] for c in data["categories"]}
        assert "agent_recall_policy" in prompt_ids
        assert "agent_recall" in category_ids
