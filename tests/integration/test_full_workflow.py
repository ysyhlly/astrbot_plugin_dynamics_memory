"""Recall, reflection, search, and reset against an isolated memory stack."""

import pytest
from astrbot.api.provider import LLMResponse, ProviderRequest
from tests.integration.test_real_db_end_to_end import _TestEvent


@pytest.mark.asyncio
async def test_recall_reflection_and_search_workflow(real_db_stack):
    event_handler = real_db_stack["event_handler"]
    command_handler = real_db_stack["command_handler"]
    conversation_manager = real_db_stack["conversation_manager"]
    session_id = "test:FriendMessage:full-workflow"
    event = _TestEvent(session_id, "I went running yesterday.")
    no_memory_req = ProviderRequest(prompt="Synthetic request envelope")
    await event_handler.handle_memory_recall(event, no_memory_req)
    assert no_memory_req._lm_private_hook_outcome_v1 == "empty"
    assert not no_memory_req.extra_user_content_parts

    response = LLMResponse(role="assistant", completion_text="Great habit, keep going!")
    await event_handler.handle_memory_reflection(event, response)
    await event_handler.shutdown()
    assert (
        await conversation_manager.get_session_metadata(
            session_id, "last_summarized_index"
        )
        == 2
    )

    search_messages = [
        msg async for msg in command_handler.handle_search(event, query="running", k=5)
    ]
    assert len(search_messages) == 1
    assert "running" in search_messages[0]

    reset_messages = [msg async for msg in command_handler.handle_reset(event)]
    assert "已重置" in reset_messages[0]
    assert await conversation_manager.store.get_message_count(session_id) == 0
