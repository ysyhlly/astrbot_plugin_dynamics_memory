"""Exact receipt reflection windows survive refusals, late receipts, and reloads."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from tests.event_handler_support import _make_real_handler
from tests.test_narrative_api import SCOPE, envelope
from tests.test_reflection_backlog import _configure_processor


SESSION = scope_key(SCOPE)
BASE_TIME = 1_790_000_000


async def add_receipts(store, *sequence):
    for index in sequence:
        event = envelope(
            f"receipt-{index}",
            "delivery.success" if index % 2 else "message.received",
            f"synthetic {index}",
        )
        event["occurred_at"] = BASE_TIME + index
        assert await store.ingest_narrative_event(event)


async def configured_handler(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    handler.config_manager._config["reflection_engine"].update(
        summary_trigger_rounds=1,
        max_messages_per_segment=2,
        max_segments_per_run=3,
    )
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=(None, None))
    _configure_processor(handler)
    return handler, manager, store


async def run_receipt(handler):
    await handler.reflect_recorded_scope(SCOPE)
    await asyncio.gather(*handler._storage_tasks)


def model_inputs(handler):
    return [
        [message.content for message in call.kwargs["messages"]]
        for call in handler.memory_processor.process_conversation.await_args_list
    ]


@pytest.mark.asyncio
async def test_legacy_refusal_stays_exact_and_later_windows_are_saved(tmp_path):
    handler, manager, store = await configured_handler(tmp_path)
    try:
        await add_receipts(store, 0, 1, 2, 3)
        await manager.update_session_metadata(
            SESSION,
            "pending_summary",
            {
                "start_index": 0,
                "end_index": 2,
                "retry_count": 1,
                "failure_stage": "provider_terminal",
                "error_code": "provider_content_blocked",
            },
        )
        await run_receipt(handler)
        assert model_inputs(handler) == [["synthetic 2", "synthetic 3"]]
        assert await manager.get_session_metadata(SESSION, "last_summarized_index") == 0
        assert await manager.get_session_metadata(SESSION, "pending_summary") is None
    finally:
        await handler.shutdown()
        await store.close()

    handler, manager, store = await configured_handler(tmp_path)
    try:
        await add_receipts(store, -1)
        assert await handler.retry_refused_scope(SCOPE)
        await asyncio.gather(*handler._storage_tasks)
        assert model_inputs(handler) == [["synthetic 0", "synthetic 1"]]
        assert await manager.get_session_metadata(SESSION, "last_summarized_index") == 0
        assert await store.get_message_count(SESSION) == 5
        assert not await handler.retry_refused_scope(SCOPE)
        await add_receipts(store, 4)
        await run_receipt(handler)
        assert model_inputs(handler)[1:] == [["synthetic -1"], ["synthetic 4"]]
        assert await manager.get_session_metadata(SESSION, "last_summarized_index") == 6
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_provider_refusal_continues_with_later_budgeted_window(tmp_path):
    handler, manager, store = await configured_handler(tmp_path)
    try:
        await add_receipts(store, 0, 1, 2, 3)
        handler.memory_processor.process_conversation.side_effect = [
            RuntimeError("Gemini prompt blocked: PROHIBITED_CONTENT"),
            ("safe separate summary", {}, 0.8),
        ]
        await run_receipt(handler)
        assert model_inputs(handler) == [
            ["synthetic 0", "synthetic 1"],
            ["synthetic 2", "synthetic 3"],
        ]
        assert handler.memory_engine.add_memory.await_count == 1
        assert await manager.get_session_metadata(SESSION, "last_summarized_index") == 0
        _, messages, refusal = await store.get_managed_summary_window(
            SESSION, 2, 2, retry_refused=True
        )
        assert len(messages) == 2
        assert refusal["failure_stage"] == "provider_terminal"
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_three_errors_cool_down_and_retry_same_window_after_restart(
    tmp_path, monkeypatch
):
    handler, manager, store = await configured_handler(tmp_path)
    try:
        await add_receipts(store, 0, 1)
        handler.memory_processor.process_conversation.side_effect = RuntimeError(
            "temporary provider error"
        )
        for attempt in range(3):
            if attempt > 0:
                cur_pending = await manager.get_session_metadata(
                    SESSION, "pending_summary"
                )
                assert cur_pending["retry_after"] > time.time()
                # Advance past exponential retry_after backoff (60s -> 300s) for next attempt
                await manager.update_session_metadata(
                    SESSION,
                    "pending_summary",
                    {**cur_pending, "retry_after": 0},
                )
            await run_receipt(handler)
        pending = await manager.get_session_metadata(SESSION, "pending_summary")
        assert pending["failure_stage"] == "retry_exhausted"
        assert pending["exhausted_retry_count"] == 3
        assert pending["retry_after"] > time.time()
        assert await manager.get_session_metadata(SESSION, "last_summarized_index") == 0
        assert model_inputs(handler) == [["synthetic 0", "synthetic 1"]] * 3
    finally:
        await handler.shutdown()
        await store.close()

    handler, manager, store = await configured_handler(tmp_path)
    try:
        await add_receipts(store, -1, 2, 3)
        await run_receipt(handler)
        handler.memory_processor.process_conversation.assert_not_awaited()
        assert await manager.get_session_metadata(SESSION, "pending_summary") == pending
        from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin

        plugin = SimpleNamespace(
            event_handler=handler,
            _narrative_scope_allowed=AsyncMock(return_value=False),
        )
        now = pending["retry_after"] + 1
        monkeypatch.setattr(
            "astrbot_plugin_dynamics_memory.main.time",
            SimpleNamespace(time=lambda: now),
        )
        assert (
            await LivingMemoryPlugin._recover_managed_reflection_page(plugin, store, 0)
            == 0
        )
        plugin._narrative_scope_allowed.assert_awaited_once_with(SCOPE, "u1")
        handler.memory_processor.process_conversation.assert_not_awaited()
        monkeypatch.setattr(
            "astrbot_plugin_dynamics_memory.storage.conversation_summary_progress.time",
            type(
                "Clock", (), {"time": staticmethod(lambda: pending["retry_after"] + 1)}
            ),
        )
        plugin._narrative_scope_allowed.return_value = True
        assert (
            await LivingMemoryPlugin._recover_managed_reflection_page(plugin, store, 0)
            == 0
        )
        await asyncio.gather(*handler._storage_tasks)
        assert model_inputs(handler) == [
            ["synthetic 0", "synthetic 1"],
            ["synthetic -1"],
            ["synthetic 2", "synthetic 3"],
        ]
        assert await manager.get_session_metadata(SESSION, "last_summarized_index") == 5
        assert await store.get_message_count(SESSION) == 5
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_delivery_during_inflight_is_coalesced_without_second_summary_of_same_window(
    tmp_path,
):
    handler, manager, store = await configured_handler(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def summary(**kwargs):
        entered.set()
        await release.wait()
        return "summary", {}, 0.8

    try:
        await add_receipts(store, 0, 1)
        handler.memory_processor.process_conversation.side_effect = summary
        await handler.reflect_recorded_scope(SCOPE)
        await asyncio.wait_for(entered.wait(), 2)
        await add_receipts(store, 2, 3)
        await asyncio.gather(*(handler.reflect_recorded_scope(SCOPE) for _ in range(5)))
        assert len(handler._storage_tasks) == 1
        release.set()
        await asyncio.gather(*handler._storage_tasks)
        assert model_inputs(handler) == [
            ["synthetic 0", "synthetic 1"],
            ["synthetic 2", "synthetic 3"],
        ]
        assert await manager.get_session_metadata(SESSION, "last_summarized_index") == 4
    finally:
        release.set()
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("world_state", ["ready", "unavailable", "native", "not_admin"])
async def test_admin_retry_resolves_authoritative_character_scope(
    monkeypatch, world_state
):
    from astrbot_plugin_dynamics_memory import main

    world_scope = {**SCOPE, "persona_id": "world-card", "instance_id": "world-instance"}
    world = SimpleNamespace(
        get_character_context=AsyncMock(
            return_value={"schema_version": 1, "scope": world_scope}
        ),
        get_dynamics_capabilities=lambda: {
            "schema_version": 1,
            "enabled": True,
            "ready": world_state == "ready",
            "capabilities": {"character": 1},
        },
    )
    stars = (
        [{"name": "world", "star_cls": world}]
        if world_state in {"ready", "unavailable"}
        else []
    )
    plugin = SimpleNamespace(
        context=SimpleNamespace(get_all_stars=lambda: stars),
        event_handler=SimpleNamespace(retry_refused_scope=AsyncMock(return_value=True)),
        _native_scope_from_request=main.LivingMemoryPlugin._native_scope_from_request,
        _narrative_scope_allowed=AsyncMock(return_value=True),
    )
    event = SimpleNamespace(
        is_admin=lambda: world_state != "not_admin",
        unified_msg_origin=SCOPE["umo"],
        get_sender_id=lambda: "admin",
        plain_result=lambda text: text,
    )
    monkeypatch.setattr(main, "_get_native_persona_id", AsyncMock(return_value="host"))
    results = [
        item async for item in main.LivingMemoryPlugin.summarize_retry(plugin, event)
    ]
    if world_state == "ready":
        plugin.event_handler.retry_refused_scope.assert_awaited_once_with(world_scope)
    elif world_state == "native":
        plugin.event_handler.retry_refused_scope.assert_awaited_once_with(
            {**SCOPE, "persona_id": "host", "instance_id": "host"},
        )
    else:
        plugin.event_handler.retry_refused_scope.assert_not_awaited()
        assert len(results) == (0 if world_state == "not_admin" else 1)
