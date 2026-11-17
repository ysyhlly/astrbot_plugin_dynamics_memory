from __future__ import annotations

import asyncio
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from astrbot.api.platform import MessageType

from astrbot_plugin_dynamics_memory.core.event_handler_modules import memory_reflection
from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection import (
    MemoryReflection,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.message_utils import (
    MessageUtils,
)
from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
    ConversationManager,
)
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELPERS_SOURCE = PLUGIN_ROOT / "core" / "plugin_helpers_mixin.py"
SESSION = "synthetic:FriendMessage:user-7"
LINEAGE_KEY = "_lm_owned_model_output_lineage_v1"
VALIDATOR_KEY = "_cd_owned_memory_sources_validate_v1"


@pytest.mark.asyncio
async def test_owned_input_resolver_coroutine_is_closed_without_await_or_warning():
    async def resolver(_event):
        raise AssertionError("async resolver must not be awaited")

    event = SimpleNamespace(
        _cd_owned_original_input_v1=True,
        _chat_dynamics_owned_request=True,
        _cd_original_input_resolver_v1=resolver,
        unified_msg_origin=SESSION,
        get_platform_name=lambda: "synthetic",
        get_self_id=lambda: "bot-7",
        get_sender_id=lambda: "user-7",
        message_obj=SimpleNamespace(message_id="platform-1"),
    )
    # Capture the actual accidental coroutine returned by the resolver so the
    # consumer's rejection can be verified without relying on GC timing.
    created = []

    def async_capability(_event):
        pending = resolver(event)
        created.append(pending)
        return pending

    event._cd_original_input_resolver_v1 = async_capability
    from astrbot_plugin_dynamics_memory.core.event_handler_modules.message_utils import (
        MessageUtils,
    )

    resolved = await MessageUtils(None, None).resolve_original_input(event)
    assert resolved.status == "unavailable"
    assert len(created) == 1
    assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED


def _extract_physical_input_observer():
    tree = ast.parse(
        HELPERS_SOURCE.read_text(encoding="utf-8"), filename=str(HELPERS_SOURCE)
    )
    plugin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PluginHelpersMixin"
    )
    method = next(
        node
        for node in plugin.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_physical_original_input_status"
    )
    method.decorator_list = []
    wrapper = ast.ClassDef(
        name="Observer", bases=[], keywords=[], body=[method], decorator_list=[]
    )
    module = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(module)
    import inspect as inspect_module
    from astrbot_plugin_dynamics_memory.core.event_handler_modules.message_utils import (
        discard_unexpected_awaitable,
        validate_owned_original_input,
    )

    namespace = {
        "inspect": inspect_module,
        "validate_owned_original_input": validate_owned_original_input,
        "discard_unexpected_awaitable": discard_unexpected_awaitable,
    }
    exec(compile(module, str(HELPERS_SOURCE), "exec"), namespace)
    return namespace["Observer"]._physical_original_input_status


def test_main_input_observer_closes_async_resolver_coroutine():
    async def resolver(_event):
        return {"schema_version": 1}

    created = []
    event = SimpleNamespace(
        _cd_owned_original_input_v1=True,
        _chat_dynamics_owned_request=True,
        unified_msg_origin=SESSION,
        get_platform_name=lambda: "synthetic",
        get_self_id=lambda: "bot-7",
        get_sender_id=lambda: "user-7",
        message_obj=SimpleNamespace(message_id="platform-1"),
    )

    def capability(candidate):
        pending = resolver(candidate)
        created.append(pending)
        return pending

    event._cd_original_input_resolver_v1 = capability
    assert _extract_physical_input_observer()(event) == "unavailable"
    assert len(created) == 1
    assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED


class _Config:
    def __init__(self, *, trigger_rounds=1):
        self.trigger_rounds = trigger_rounds

    def get(self, key, default=None):
        if key == "reflection_engine.summary_trigger_rounds":
            return self.trigger_rounds
        if key == "reflection_engine.source_retention_importance_threshold":
            return 0.8
        return default


class _Processor:
    def __init__(self, on_process=None):
        self.on_process = on_process
        self.calls = []

    async def process_conversation(self, *, messages, **_kwargs):
        self.calls.append(tuple(message.content for message in messages))
        if self.on_process is not None:
            self.on_process()
        return "synthetic derived summary", {"topics": ["synthetic"]}, 0.9

    def classify_atoms_from_metadata(self, **_kwargs):
        return []


class _Engine:
    def __init__(self, *, on_embedding=None):
        self.on_embedding = on_embedding
        self.active = []
        self.calls = 0

    async def add_memory(
        self, *, content, source_validator, source_publish_lease, **_kwargs
    ):
        self.calls += 1
        if await source_validator() is not True:
            return
        if self.on_embedding is not None:
            self.on_embedding()
            await asyncio.sleep(0)
        if await source_validator() is not True:
            return
        async with source_publish_lease() as valid:
            if valid is True:
                self.active.append(content)
                return len(self.active)
        return None


def _owned_event(validator, *, suffix="1", input_row=None):
    source_event_id = None
    source_text = ""
    occurred_at = None
    if isinstance(input_row, dict):
        metadata = input_row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        source_event_id = (
            metadata.get("source_event_id") if isinstance(metadata, dict) else None
        )
        source_text = input_row.get("content", "")
        occurred_at = input_row.get("timestamp")
    event_key = (
        source_event_id[len("cd-original-input-v1:") :]
        if isinstance(source_event_id, str)
        and source_event_id.startswith("cd-original-input-v1:")
        else f"event-{suffix}"
    )
    dto = {
        "schema_version": 1,
        "scope": {"umo": SESSION, "platform": "synthetic", "self_id": "bot-7"},
        "current_event_key": event_key,
        "messages": (
            {
                "event_key": event_key,
                "platform_message_id": f"platform-{suffix}",
                "author_id": "user-7",
                "occurred_at": occurred_at,
                "parts": ({"kind": "text", "text": source_text},),
            },
        ),
    }
    event = SimpleNamespace(
        _cd_owned_original_input_v1=True,
        _chat_dynamics_owned_request=True,
        _cd_owned_memory_sources_validate_v1=validator,
        unified_msg_origin=SESSION,
        get_message_type=lambda: MessageType.FRIEND_MESSAGE,
        get_sender_id=lambda: "user-7",
        get_sender_name=lambda: "Synthetic user",
        get_self_id=lambda: "bot-7",
        get_platform_name=lambda: "synthetic",
        message_obj=SimpleNamespace(message_id=f"platform-{suffix}", sender=None),
    )
    event._cd_original_input_resolver_v1 = lambda candidate: (
        dto if candidate is event else None
    )
    return event


def _response(text="synthetic assistant output"):
    return SimpleNamespace(
        role="assistant",
        tools_call_name=None,
        tools_call_extra_content=None,
        completion_text=text,
    )


async def _make_reflection(
    tmp_path,
    monkeypatch,
    *,
    trigger_rounds=1,
    processor=None,
    engine=None,
    manager=None,
):
    if manager is None:
        store = ConversationStore(str(tmp_path / "reflection-lineage.sqlite3"))
        await store.initialize()
        manager = ConversationManager(store)
    else:
        store = manager.store
    storage_tasks = set()
    message_utils = MessageUtils(None, manager)
    message_utils.enforce_message_limit = lambda _session: asyncio.sleep(0)
    reflection = MemoryReflection(
        context=None,
        config_manager=_Config(trigger_rounds=trigger_rounds),
        memory_engine=engine or _Engine(),
        memory_processor=processor or _Processor(),
        conversation_manager=manager,
        message_utils=message_utils,
        storage_tasks=storage_tasks,
        storage_sessions_inflight=set(),
        storage_state_lock=asyncio.Lock(),
    )
    monkeypatch.setattr(
        memory_reflection, "is_event_memory_allowed", lambda *_args: True
    )
    monkeypatch.setattr(
        memory_reflection,
        "get_persona_id",
        lambda *_args: asyncio.sleep(0, result="synthetic-persona"),
    )
    monkeypatch.setattr(
        memory_reflection, "resolve_memory_write_scope", lambda _config, _event: SESSION
    )
    return reflection, manager, store


async def _append_user(manager, content, message_id):
    source_event_id = f"cd-original-input-v1:event-{message_id}"
    message = await manager.add_message(
        SESSION,
        "user",
        content,
        sender_id="user-7",
        sender_name="Synthetic user",
        platform="synthetic",
        metadata={
            "source_event_id": source_event_id,
            "scope": {
                "platform": "synthetic",
                "umo": SESSION,
                "persona_id": "synthetic-persona",
                "instance_id": "synthetic-persona",
            },
            "fact_kind": "observation",
            "is_bot_message": False,
        },
    )
    return await manager.store.get_message_by_id(SESSION, message.id)


async def _await_storage_tasks(reflection):
    tasks = tuple(reflection._storage_tasks)
    if tasks:
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("cap_state", ["missing", "denied", "error"])
async def test_owned_private_on_response_requires_live_cap_before_assistant_append(
    tmp_path, monkeypatch, cap_state
):
    reflection, manager, store = await _make_reflection(tmp_path, monkeypatch)
    calls = []

    async def denied(_candidate):
        if cap_state == "error":
            raise RuntimeError("synthetic failure")
        return False

    event = _owned_event(None if cap_state == "missing" else denied)
    await reflection.handle_memory_reflection(event, _response())
    assert await store.get_message_count(SESSION) == 0
    assert calls == []
    await store.close()


@pytest.mark.asyncio
async def test_owned_private_unchanged_cap_publishes_and_persists_only_opaque_lineage(
    tmp_path, monkeypatch
):
    processor = _Processor()
    engine = _Engine()
    reflection, manager, store = await _make_reflection(
        tmp_path, monkeypatch, processor=processor, engine=engine
    )
    input_row = await _append_user(manager, "synthetic authentic user input", 1)
    validations = []

    async def source_cap(candidate):
        validations.append(candidate)
        return True

    event = _owned_event(source_cap, input_row=input_row)
    await reflection.handle_memory_reflection(event, _response())
    await _await_storage_tasks(reflection)

    rows = await store.get_messages(SESSION, limit=10)
    assistant = next(message for message in rows if message.role == "assistant")
    token = assistant.metadata.get(LINEAGE_KEY)
    assert isinstance(token, str) and len(token) == 48
    assert "synthetic assistant output" not in repr(assistant.metadata)
    assert token in reflection._owned_source_lineages
    assert validations and all(candidate is event for candidate in validations)
    assert processor.calls
    assert engine.active == ["synthetic derived summary"]
    await store.close()


@pytest.mark.asyncio
async def test_owned_private_exact_current_row_mismatch_blocks_assistant_append(
    tmp_path, monkeypatch
):
    reflection, manager, store = await _make_reflection(tmp_path, monkeypatch)
    input_row = await _append_user(manager, "synthetic persisted input", 1)
    validations = []

    async def source_cap(_candidate):
        validations.append(True)
        return True

    event = _owned_event(source_cap, input_row=input_row)
    dto = event._cd_original_input_resolver_v1(event)
    altered_message = dict(dto["messages"][0])
    altered_message["parts"] = ({"kind": "text", "text": "synthetic mismatched DTO"},)
    altered_dto = {**dto, "messages": (altered_message,)}
    event._cd_original_input_resolver_v1 = lambda candidate: (
        altered_dto if candidate is event else None
    )

    await reflection.handle_memory_reflection(event, _response())
    assert validations == [True]
    assert await store.get_message_count(SESSION) == 1
    rows = await store.get_messages(SESSION, limit=5)
    assert len(rows) == 1 and rows[0].role == "user"
    await store.close()


@pytest.mark.asyncio
async def test_owned_private_validator_cancellation_propagates_before_append(
    tmp_path, monkeypatch
):
    reflection, manager, store = await _make_reflection(tmp_path, monkeypatch)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def source_cap(_candidate):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    event = _owned_event(source_cap)
    task = asyncio.create_task(reflection.handle_memory_reflection(event, _response()))
    await asyncio.wait_for(started.wait(), timeout=0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert await store.get_message_count(SESSION) == 0
    await store.close()


@pytest.mark.asyncio
async def test_unowned_private_with_cd_source_cap_also_binds_draft_to_that_cap(
    tmp_path, monkeypatch
):
    processor = _Processor()
    engine = _Engine()
    reflection, manager, store = await _make_reflection(
        tmp_path, monkeypatch, processor=processor, engine=engine
    )
    input_row = await _append_user(manager, "synthetic authentic user input", 1)

    async def source_cap(_candidate):
        return True

    event = _owned_event(source_cap, input_row=input_row)
    event._cd_owned_original_input_v1 = False
    event._chat_dynamics_owned_request = False
    await reflection.handle_memory_reflection(event, _response())
    await _await_storage_tasks(reflection)

    rows = await store.get_messages(SESSION, limit=10)
    assistant = next(message for message in rows if message.role == "assistant")
    token = assistant.metadata.get(LINEAGE_KEY)
    assert isinstance(token, str) and token in reflection._owned_source_lineages
    assert processor.calls
    assert engine.active == ["synthetic derived summary"]
    await store.close()


@pytest.mark.asyncio
async def test_revocation_during_processing_stops_before_engine_publish(
    tmp_path, monkeypatch
):
    active = {"allowed": True}
    processor = _Processor(on_process=lambda: active.update(allowed=False))
    engine = _Engine()
    reflection, manager, store = await _make_reflection(
        tmp_path, monkeypatch, processor=processor, engine=engine
    )
    input_row = await _append_user(manager, "synthetic authentic user input", 1)

    async def source_cap(_candidate):
        return active["allowed"]

    await reflection.handle_memory_reflection(
        _owned_event(source_cap, input_row=input_row), _response()
    )
    await _await_storage_tasks(reflection)
    assert processor.calls
    assert engine.calls == 0
    assert engine.active == []
    await store.close()


@pytest.mark.asyncio
async def test_revocation_during_embedding_stops_before_active_publication(
    tmp_path, monkeypatch
):
    active = {"allowed": True}
    engine = _Engine(on_embedding=lambda: active.update(allowed=False))
    reflection, manager, store = await _make_reflection(
        tmp_path, monkeypatch, engine=engine
    )
    input_row = await _append_user(manager, "synthetic authentic user input", 1)

    async def source_cap(_candidate):
        return active["allowed"]

    await reflection.handle_memory_reflection(
        _owned_event(source_cap, input_row=input_row), _response()
    )
    await _await_storage_tasks(reflection)
    assert engine.calls == 1
    assert engine.active == []
    await store.close()


@pytest.mark.asyncio
async def test_later_request_cannot_reauthorize_prior_draft_and_reload_missing_map_excludes_it(
    tmp_path, monkeypatch
):
    reflection, manager, store = await _make_reflection(
        tmp_path, monkeypatch, trigger_rounds=10
    )
    old_input_row = await _append_user(manager, "synthetic earlier human input", 1)
    old_cap_state = {"allowed": True}

    async def old_cap(_candidate):
        return old_cap_state["allowed"]

    old_event = _owned_event(old_cap, suffix="1", input_row=old_input_row)
    await reflection.handle_memory_reflection(
        old_event, _response("old generated draft")
    )
    old_rows = await store.get_messages(SESSION, limit=10)
    old_draft = next(row for row in old_rows if row.role == "assistant")
    assert old_draft.metadata.get(LINEAGE_KEY) in reflection._owned_source_lineages

    # A later valid request must not replace the original capability attached
    # to the old draft. The old draft remains in the same reflection window,
    # but its source has been withdrawn before the new response arrives.
    old_cap_state["allowed"] = False
    reflection.config_manager.trigger_rounds = 1
    new_input_row = await _append_user(manager, "synthetic later human input", 2)

    async def new_cap(_candidate):
        return True

    new_event = _owned_event(new_cap, suffix="2", input_row=new_input_row)
    await reflection.handle_memory_reflection(
        new_event, _response("new generated draft")
    )
    await _await_storage_tasks(reflection)
    assert reflection.memory_processor.calls == []
    assert reflection.memory_engine.calls == 0

    # The failed original capability is evicted. A later complete window can
    # summarize newer authentic rows without reusing the withdrawn draft.
    later_input_row = await _append_user(manager, "synthetic later independent fact", 3)

    async def later_cap(_candidate):
        return True

    await reflection.handle_memory_reflection(
        _owned_event(later_cap, suffix="3", input_row=later_input_row),
        _response("later generated draft"),
    )
    await _await_storage_tasks(reflection)
    assert reflection.memory_processor.calls
    assert any(
        "synthetic later independent fact" in batch
        for batch in reflection.memory_processor.calls
    )
    assert all(
        "old generated draft" not in batch
        for batch in reflection.memory_processor.calls
    )
    assert reflection.memory_engine.active == [
        "synthetic derived summary",
        "synthetic derived summary",
    ]
    still_stored = await store.get_message_by_id(SESSION, old_draft.id)
    assert still_stored is not None
    assert still_stored["content"] == "old generated draft"
    await store.close()


@pytest.mark.asyncio
async def test_reload_without_lineage_handle_filters_draft_without_deleting_history(
    tmp_path, monkeypatch
):
    reflection, manager, store = await _make_reflection(
        tmp_path, monkeypatch, trigger_rounds=10
    )
    input_row = await _append_user(manager, "synthetic human input", 1)

    async def source_cap(_candidate):
        return True

    await reflection.handle_memory_reflection(
        _owned_event(source_cap, input_row=input_row),
        _response("persistent synthetic draft"),
    )
    rows = await store.get_messages(SESSION, limit=10)
    draft = next(row for row in rows if row.role == "assistant")
    restarted = await _make_reflection(
        tmp_path,
        monkeypatch,
        trigger_rounds=1,
        processor=_Processor(),
        engine=_Engine(),
        manager=manager,
    )
    new_reflection, _, _ = restarted
    assert new_reflection._eligible_reflection_messages(rows) == [
        row for row in rows if row.id != draft.id
    ]
    stored = await store.get_message_by_id(SESSION, draft.id)
    assert stored is not None and stored["content"] == "persistent synthetic draft"
    await store.close()


def test_lru_eviction_makes_old_draft_ineligible_without_deleting_its_row():
    from collections import OrderedDict
    from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection import (
        _OWNED_LINEAGE_LIMIT,
    )

    reflection = object.__new__(MemoryReflection)
    reflection._owned_source_lineages = OrderedDict()
    first_event = SimpleNamespace(
        **{VALIDATOR_KEY: lambda _event: asyncio.sleep(0, result=True)}
    )
    first_token = "first-lineage-token"
    first_message = SimpleNamespace(
        id=1,
        session_id=SESSION,
        role="assistant",
        content="first draft",
    )
    reflection._remember_owned_source_lineage(
        first_token, first_event, getattr(first_event, VALIDATOR_KEY), first_message
    )
    for index in range(_OWNED_LINEAGE_LIMIT):
        event = SimpleNamespace(
            **{VALIDATOR_KEY: lambda _event: asyncio.sleep(0, result=True)}
        )
        message = SimpleNamespace(
            id=index + 2,
            session_id=SESSION,
            role="assistant",
            content=f"draft {index}",
        )
        reflection._remember_owned_source_lineage(
            f"lineage-{index}", event, getattr(event, VALIDATOR_KEY), message
        )
    old_draft = SimpleNamespace(
        id=1,
        session_id=SESSION,
        role="assistant",
        metadata={LINEAGE_KEY: first_token},
        content="first draft",
    )
    assert len(reflection._owned_source_lineages) == _OWNED_LINEAGE_LIMIT
    assert first_token not in reflection._owned_source_lineages
    assert reflection._eligible_reflection_messages([old_draft]) == []
    assert old_draft.content == "first draft"


def test_lineage_token_cannot_be_transplanted_to_another_assistant_row():
    from collections import OrderedDict

    reflection = object.__new__(MemoryReflection)
    reflection._owned_source_lineages = OrderedDict()
    messages = []
    for index, token in enumerate(("token-one", "token-two"), start=1):
        event = SimpleNamespace(
            **{VALIDATOR_KEY: lambda _event: asyncio.sleep(0, result=True)}
        )
        message = SimpleNamespace(
            id=index,
            session_id=SESSION,
            role="assistant",
            content=f"draft-{index}",
        )
        reflection._remember_owned_source_lineage(
            token, event, getattr(event, VALIDATOR_KEY), message
        )
        message.metadata = {LINEAGE_KEY: token}
        messages.append(message)
    messages[0].metadata[LINEAGE_KEY] = "token-two"

    assert reflection._eligible_reflection_messages(messages) == [messages[1]]
