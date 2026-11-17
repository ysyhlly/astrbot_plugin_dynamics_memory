"""Integration gates for upstream 2.8.0 with the deployed source guards."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from tests.event_handler_support import _make_real_handler, _seed_turns


@pytest.mark.parametrize("beta", [False, True])
def test_tool_registration_keeps_runtime_on_the_legacy_search_tool(beta):
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin._llm_tools_registered = False
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_agentic_recall_beta": beta}}
    )
    plugin.context = Mock()
    runtime = object()
    plugin._recall_runtime = runtime
    plugin.initializer = SimpleNamespace(
        memory_engine=object(), memory_processor=object(), conversation_manager=object()
    )
    plugin._register_agent_tools_if_needed()
    registered = plugin.context.add_llm_tools.call_args.args
    assert len(registered) == (2 if beta else 1)
    if not beta:
        assert registered[0].recall_runtime is runtime
        assert (
            registered[0].conversation_manager
            is plugin.initializer.conversation_manager
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True])
async def test_reflection_exact_source_snapshot_and_publication_lease(tmp_path, stale):
    engine = Mock()
    engine.stored_summary_window = AsyncMock(return_value=None)
    engine.delete_memory = AsyncMock(return_value=True)
    written = []

    async def write(**kwargs):
        assert await kwargs["source_validator"]() is True
        async with kwargs["source_publish_lease"]() as valid:
            assert valid is True
            written.append(kwargs)
        return 123

    engine.add_memory = AsyncMock(side_effect=write)
    handler, manager, store = await _make_real_handler(tmp_path, memory_engine=engine)
    sid = "test:private:upgrade"
    try:
        await _seed_turns(manager, sid, ("C1",))
        rows = await manager.get_messages_range(sid, 0, 2)
        if stale:
            rows[0].content = "invented source"
        processor = handler._memory_reflection.memory_processor
        processor.process_conversation = AsyncMock(
            return_value=("summary", {"topics": []}, 0.9)
        )
        processor.classify_atoms_from_metadata = Mock(return_value=[])
        result = await handler._memory_reflection._storage_task(
            sid, rows, "p1", 0, 2, 0, expected_revision=0
        )
        if stale:
            assert not result
            engine.add_memory.assert_not_awaited()
            processor.process_conversation.assert_not_awaited()
        else:
            assert result is True
            assert len(written) == 1
            assert (
                await manager.get_session_metadata(sid, "last_summarized_index", 0) == 2
            )
            engine.delete_memory.assert_not_awaited()
    finally:
        await handler.shutdown()
        await store.close()
