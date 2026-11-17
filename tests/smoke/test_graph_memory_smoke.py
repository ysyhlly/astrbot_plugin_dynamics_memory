"""Smoke tests for graph memory critical user flows."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest
from astrbot.core.message.components import Plain
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.command_handler import CommandHandler
from astrbot_plugin_dynamics_memory.core.event_handler import EventHandler
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.plugin_helpers_mixin import PluginHelpersMixin

from tests.test_agentic_engine_retrieval import _SharedDbFaiss
from tests.test_graph_memory import _FakeFaissDB

SESSION_ID = "test:FriendMessage:smoke-session"
PERSONA_ID = "persona_smoke"
MEMORY_CONTENT = "project sync record"
MEMORY_METADATA = {
    "topics": ["project sync"],
    "participants": ["alice"],
    "key_facts": ["meeting starts at 3pm tomorrow"],
    "canonical_summary": "project sync record",
}


async def _create_engine(tmp_path: Path) -> MemoryEngine:
    holder = {}
    engine = MemoryEngine(
        db_path=str(tmp_path / "smoke_memory.db"),
        faiss_db=_SharedDbFaiss(holder),
        graph_vector_db=_FakeFaissDB(),
        config={
            "fallback_enabled": True,
            "graph_memory_enabled": True,
            "document_route_weight": 0.6,
            "graph_route_weight": 0.4,
            "cross_route_bonus": 0.08,
        },
    )
    holder["engine"] = engine
    await engine.initialize()
    return engine


async def _seed_memory(engine: MemoryEngine) -> int:
    return await engine.add_memory(
        content=MEMORY_CONTENT,
        session_id=SESSION_ID,
        persona_id=PERSONA_ID,
        importance=0.9,
        metadata=dict(MEMORY_METADATA),
    )


async def _settle_background_tasks() -> None:
    await asyncio.sleep(0.05)


def _make_event(query: str):
    event = Mock()
    event._cd_owned_original_input_v1 = None
    event._chat_dynamics_owned_request = False
    event.unified_msg_origin = SESSION_ID
    event.get_message_type = Mock(return_value=MessageType.FRIEND_MESSAGE)
    event.get_sender_id = Mock(return_value="user-1")
    event.get_self_id = Mock(return_value="bot-1")
    event.get_sender_name = Mock(return_value="SmokeUser")
    event.message_str = query
    event.get_message_str = Mock(return_value=query)
    event.get_messages = Mock(return_value=[Plain(query)])
    event.get_extra = Mock(return_value=None)
    event.get_platform_id = Mock(return_value="test")
    event.get_platform_name = Mock(return_value="test")
    event.plain_result = lambda message: message
    return event


@pytest.mark.asyncio
async def test_smoke_search_by_participant_hits_graph_route(tmp_path: Path):
    engine = await _create_engine(tmp_path)
    try:
        memory_id = await _seed_memory(engine)
        results = await engine.search_memories(
            query="alice",
            k=3,
            session_id=SESSION_ID,
            persona_id=PERSONA_ID,
        )

        assert results
        assert results[0].doc_id == memory_id
        assert (results[0].score_breakdown or {}).get("graph_keyword_score", 0.0) > 0
    finally:
        await _settle_background_tasks()
        await engine.close()


@pytest.mark.asyncio
async def test_smoke_search_by_summary_includes_four_mode_breakdown(tmp_path: Path):
    engine = await _create_engine(tmp_path)
    try:
        memory_id = await _seed_memory(engine)
        results = await engine.search_memories(
            query=MEMORY_CONTENT,
            k=3,
            session_id=SESSION_ID,
            persona_id=PERSONA_ID,
        )

        assert results
        assert results[0].doc_id == memory_id
        breakdown = results[0].score_breakdown
        for channel in (
            "document_keyword",
            "document_vector",
            "graph_keyword",
            "graph_vector",
        ):
            assert breakdown[f"{channel}_score"] > 0
        assert breakdown["graph_route_score"] > 0
    finally:
        await _settle_background_tasks()
        await engine.close()


@pytest.mark.asyncio
async def test_smoke_rebuild_graph_command_reports_success(tmp_path: Path):
    engine = await _create_engine(tmp_path)
    try:
        memory_id = await _seed_memory(engine)
        assert engine.graph_memory_manager is not None
        await engine.graph_memory_manager.delete_memory(memory_id)

        handler = CommandHandler(
            context=Mock(),
            config_manager=ConfigManager(),
            memory_engine=engine,
            conversation_manager=None,
            index_validator=None,
        )
        messages = [
            message
            async for message in handler.handle_rebuild_graph(
                _make_event("/lmem rebuild-graph")
            )
        ]

        assert len(messages) == 2
        assert messages[0].endswith("...")
        assert any(char.isdigit() for char in messages[1])
    finally:
        await _settle_background_tasks()
        await engine.close()


@pytest.mark.asyncio
async def test_smoke_event_recall_injects_memory_into_prompt(tmp_path: Path):
    engine = await _create_engine(tmp_path)
    try:
        memory_id = await _seed_memory(engine)
        await engine.add_memory(
            "unrelated private session record",
            session_id="test:FriendMessage:other-session",
            persona_id=PERSONA_ID,
            metadata=dict(MEMORY_METADATA),
        )
        conversation_manager = Mock()
        conversation_manager.add_message_from_event = AsyncMock()
        conversation_manager.store = None
        context = SimpleNamespace(
            get_all_stars=lambda: {
                "autonomous_blocking": {"plugin": SimpleNamespace(enabled=False)}
            }
        )
        config = ConfigManager(
            {
                "recall_engine": {"top_k": 3, "injection_method": "extra_user_content"},
                "session_manager": {"max_messages_per_session": 100},
            }
        )
        plugin = PluginHelpersMixin()
        plugin.context = context
        plugin.config_manager = config
        plugin.initializer = SimpleNamespace(
            memory_engine=engine, conversation_manager=conversation_manager
        )
        plugin.peek_evidence_revision = lambda _scope: {
            "available": True,
            "correction_revision": 0,
        }
        plugin._narrative_scope_allowed = AsyncMock(
            side_effect=lambda scope: scope["umo"] == SESSION_ID
        )

        handler = EventHandler(
            context=context,
            config_manager=config,
            memory_engine=engine,
            memory_processor=Mock(),
            conversation_manager=conversation_manager,
            summary_guard=plugin._install_private_summary_guard,
        )
        prompt = "a rewritten prompt that does not mention the participant"
        req = ProviderRequest(prompt=prompt)
        event = _make_event("alice")
        with (
            patch(
                "astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall.get_persona_id",
                new=AsyncMock(return_value=PERSONA_ID),
            ),
            patch(
                "astrbot_plugin_dynamics_memory.core.plugin_helpers_mixin.get_persona_id",
                new=AsyncMock(return_value=PERSONA_ID),
            ),
        ):
            await handler.handle_memory_recall(event, req)
            assert req._lm_private_hook_outcome_v1 == "ready"
            assert len(req.extra_user_content_parts) == 1
            text = req.extra_user_content_parts[0].text
            assert "<RAG-Faiss-Memory>" in text
            assert MEMORY_CONTENT in text
            assert "unrelated private session record" not in text
            assert req.prompt == prompt
            source_rows = req._lm_private_summary_sources_v1["source_documents"]
            assert [source["document_id"] for source in source_rows] == [memory_id]
            assert await req._lm_private_summary_validate_v1() is True
        conversation_manager.add_message_from_event.assert_awaited_once_with(
            event=event, role="user", content="alice"
        )
    finally:
        await _settle_background_tasks()
        await engine.close()
