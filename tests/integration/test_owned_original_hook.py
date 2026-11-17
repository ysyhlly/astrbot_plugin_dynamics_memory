"""Native original-input hook stores direct human text once with typed provenance."""

from types import SimpleNamespace as NS
import pytest
from tests.test_private_original_input import _dto, _validate, PHYSICAL_SCOPE
from tests.integration.test_public_evidence_pipeline import pipeline as pipeline
from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin
from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
    ConversationManager,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall import (
    MemoryRecall,
)
from astrbot_plugin_dynamics_memory.core.recall_runtime import RecallDeadline


def event():
    return NS(
        _cd_owned_original_input_v1=True,
        _chat_dynamics_owned_request=True,
        unified_msg_origin=PHYSICAL_SCOPE["umo"],
        get_platform_name=lambda: "synthetic",
        get_self_id=lambda: "synthetic-bot",
        get_sender_id=lambda: "synthetic-current-user",
        message_obj=NS(message_id="platform-current"),
        _cd_original_input_resolver_v1=lambda event: _dto(),
        get_extra=lambda *a: None,
    )


@pytest.mark.asyncio
async def test_owned_input_hook_records_direct_sources_once(pipeline):
    _, _, store = pipeline
    owner = MemoryRecall.__new__(MemoryRecall)
    owner.conversation_manager = ConversationManager(store)
    source = _validate(_dto())
    scope = {
        "umo": PHYSICAL_SCOPE["umo"],
        "platform": "synthetic",
        "persona_id": "p",
        "instance_id": "p",
    }
    req = NS(_cd_memory_feedback_v1={"scope": scope})
    host_event = event()
    assert LivingMemoryPlugin._physical_original_input_status(host_event) == "valid"
    for _ in range(2):
        assert await owner._store_owned_original_messages(
            host_event, req, source, RecallDeadline.start()
        )
    rows = await store.get_messages(scope["umo"], limit=20)
    assert len(rows) == 2
    assert {r.content for r in rows} == {"older direct text", "current greeting"}
    archive = await store.load_narrative_event(
        scope, "cd-original-input-v1:internal:event-context"
    )
    assert archive["original_parts"][1]["kind"] == "quote"
    assert archive["actor_id"] == "synthetic-other-user"
    assert not await owner._store_owned_original_messages(
        host_event, NS(), source, RecallDeadline.start()
    )


@pytest.mark.parametrize(
    "damage",
    ["legacy_only", "missing_resolver", "wrong_message", "wrong_bot", "async_resolver"],
)
def test_physical_input_status_rejects_stale_or_invalid_owner(damage):
    e = event()
    if damage == "legacy_only":
        e._cd_owned_original_input_v1 = False
    elif damage == "missing_resolver":
        e._cd_original_input_resolver_v1 = None
    elif damage == "wrong_message":
        e.message_obj.message_id = "wrong"
    elif damage == "wrong_bot":
        e.get_self_id = lambda: "different"
    else:

        async def resolver(event):
            return _dto()

        e._cd_original_input_resolver_v1 = resolver
    assert LivingMemoryPlugin._physical_original_input_status(e) == "unavailable"
