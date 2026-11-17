"""Upgrading to 2.10 must not discard source history while summary work is pending."""

import time
import pytest
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
    ConversationManager,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.message_utils import (
    MessageUtils,
)
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore


@pytest.mark.asyncio
@pytest.mark.parametrize("summarized,expected", [(0, 120), (40, 80)])
async def test_capacity_cleanup_preserves_unsummarized_source_rows(
    tmp_path, summarized, expected
):
    store = ConversationStore(str(tmp_path / "history.db"))
    await store.initialize()
    manager = ConversationManager(
        store=store, max_cache_size=10, context_window_size=20
    )
    session = "test:GroupMessage:retention"
    try:
        for index in range(120):
            await manager.add_message(
                session_id=session,
                role="user",
                content=f"source-{index}",
                sender_id="member",
                timestamp=time.time() + index,
            )
        before = await store.get_session_messages_asc(session)
        await manager.update_session_metadata(
            session, "last_summarized_index", summarized
        )
        utility = MessageUtils(
            ConfigManager(
                {
                    "session_manager": {
                        "max_messages_per_session": 100,
                        "cleanup_batch_size": 50,
                    }
                }
            ),
            manager,
        )
        await utility.enforce_message_limit(session)
        after = await store.get_session_messages_asc(session)
        assert len(after) == expected
        assert {row.id for row in before[summarized:]} <= {row.id for row in after}
    finally:
        await store.close()
