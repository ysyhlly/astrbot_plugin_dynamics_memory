from __future__ import annotations

from types import MappingProxyType
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_memory.core.event_handler_modules.message_utils import (
    MessageUtils,
    validate_owned_original_input,
)


PHYSICAL_SCOPE = {
    "umo": "synthetic:FriendMessage:current-user",
    "platform": "synthetic",
    "self_id": "synthetic-bot",
}


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _dto(messages=None, **overrides):
    raw = {
        "schema_version": 1,
        "scope": dict(PHYSICAL_SCOPE),
        "current_event_key": "internal:event-current",
        "messages": messages
        or (
            {
                "event_key": "internal:event-context",
                "platform_message_id": "platform-context",
                "author_id": "synthetic-other-user",
                "occurred_at": 1_700_000_000.0,
                "parts": (
                    {"kind": "text", "text": "older direct text"},
                    {
                        "kind": "quote",
                        "text": "quoted words",
                        "author_id": "quoted-author",
                    },
                    {"kind": "image", "opaque_type": "image"},
                ),
            },
            {
                "event_key": "internal:event-current",
                "platform_message_id": "platform-current",
                "author_id": "synthetic-current-user",
                "occurred_at": 1_700_000_001.0,
                "parts": (
                    {"kind": "text", "text": "current greeting"},
                    {"kind": "mention", "text": "mentioned user"},
                    {"kind": "video", "opaque_type": "video"},
                ),
            },
            {
                "event_key": "internal:event-bot",
                "platform_message_id": None,
                "author_id": "synthetic-bot",
                "occurred_at": 1_700_000_002.0,
                "parts": ({"kind": "text", "text": "bot text is not user query"},),
            },
        ),
    }
    raw.update(overrides)
    return _freeze(raw)


def _validate(dto, **overrides):
    args = {
        "expected_scope": PHYSICAL_SCOPE,
        "trigger_author_id": "synthetic-current-user",
        "platform_message_id": "platform-current",
        "self_id": "synthetic-bot",
    }
    args.update(overrides)
    return validate_owned_original_input(dto, **args)


def test_owned_input_uses_current_plain_text_first_and_keeps_typed_parts_separate():
    result = _validate(_dto())

    assert result is not None
    assert result.status == "valid"
    assert result.current_event_key == "internal:event-current"
    assert result.query_text == "current greeting older direct text"
    assert "quoted words" not in result.query_text
    assert "mentioned user" not in result.query_text
    assert "bot text" not in result.query_text
    assert result.messages[0].parts[1]["kind"] == "quote"
    assert result.messages[1].parts[2]["kind"] == "video"
    assert result.messages[0].occurred_at == 1_700_000_000.0


@pytest.mark.parametrize(
    "case",
    [
        "wrong_physical_scope",
        "wrong_trigger_author",
        "wrong_platform_message_id",
        "current_key_missing",
        "duplicate_event_key",
        "bad_schema_version",
        "non_tuple_messages",
        "too_many_messages",
        "too_many_parts",
        "oversized_part_text",
        "oversized_total_text",
        "unknown_part_kind",
        "nonfinite_timestamp",
        "missing_self_id",
    ],
)
def test_owned_input_rejects_invalid_identity_or_overlimit_without_partial_truncation(
    case,
):
    dto = _dto()
    kwargs = {}
    if case == "wrong_physical_scope":
        kwargs["expected_scope"] = {**PHYSICAL_SCOPE, "umo": "other:FriendMessage:user"}
    elif case == "wrong_trigger_author":
        kwargs["trigger_author_id"] = "other-user"
    elif case == "wrong_platform_message_id":
        kwargs["platform_message_id"] = "different-platform-id"
    elif case == "current_key_missing":
        dto = _freeze({**dict(dto), "current_event_key": "missing"})
    elif case == "duplicate_event_key":
        msgs = list(dto["messages"])
        msgs[-1] = {**dict(msgs[-1]), "event_key": msgs[0]["event_key"]}
        dto = _freeze({**dict(dto), "messages": tuple(msgs)})
    elif case == "bad_schema_version":
        dto = _freeze({**dict(dto), "schema_version": True})
    elif case == "non_tuple_messages":
        raw = dict(dto)
        raw["messages"] = list(dto["messages"])
        dto = MappingProxyType(raw)
    elif case == "too_many_messages":
        msg = dict(dto["messages"][0])
        msgs = tuple({**msg, "event_key": f"event-{index}"} for index in range(33))
        dto = _freeze({**dict(dto), "current_event_key": "event-0", "messages": msgs})
        kwargs["trigger_author_id"] = msg["author_id"]
        kwargs["platform_message_id"] = msg["platform_message_id"]
    elif case == "too_many_parts":
        current = dict(dto["messages"][1])
        current["parts"] = tuple({"kind": "text", "text": "x"} for _ in range(129))
        msgs = (dto["messages"][0], current, dto["messages"][2])
        dto = _freeze({**dict(dto), "messages": msgs})
    elif case == "oversized_part_text":
        current = dict(dto["messages"][1])
        current["parts"] = ({"kind": "text", "text": "x" * 64_001},)
        msgs = (dto["messages"][0], current, dto["messages"][2])
        dto = _freeze({**dict(dto), "messages": msgs})
    elif case == "oversized_total_text":
        current = dict(dto["messages"][1])
        current["parts"] = ({"kind": "text", "text": "x" * 64_000},) * 4
        msgs = (dto["messages"][0], current, dto["messages"][2])
        dto = _freeze({**dict(dto), "messages": msgs})
    elif case == "unknown_part_kind":
        current = dict(dto["messages"][1])
        current["parts"] = ({"kind": "tool_result", "text": "x"},)
        msgs = (dto["messages"][0], current, dto["messages"][2])
        dto = _freeze({**dict(dto), "messages": msgs})
    elif case == "nonfinite_timestamp":
        current = dict(dto["messages"][1])
        current["occurred_at"] = float("inf")
        msgs = (dto["messages"][0], current, dto["messages"][2])
        dto = _freeze({**dict(dto), "messages": msgs})
    elif case == "missing_self_id":
        expected = dict(PHYSICAL_SCOPE)
        expected["self_id"] = ""
        kwargs["expected_scope"] = expected

    assert _validate(dto, **kwargs) is None


def test_owned_query_is_bounded_without_mutating_or_shortening_original_parts():
    text = "current " + ("x" * 3_000)
    current = {
        "event_key": "internal:event-current",
        "platform_message_id": "platform-current",
        "author_id": "synthetic-current-user",
        "occurred_at": 1_700_000_001.0,
        "parts": ({"kind": "text", "text": text},),
    }
    dto = _dto(messages=(current,))
    result = _validate(dto)

    assert result is not None
    assert len(result.query_text) == 2_000
    assert result.query_truncated is True
    assert result.messages[0].plain_text == text


def test_owned_input_allows_128_parts_per_message_across_multiple_messages():
    messages = []
    for index, author in enumerate(("synthetic-current-user", "synthetic-other-user")):
        event_key = "event-current" if index == 0 else "event-context"
        messages.append(
            {
                "event_key": event_key,
                "platform_message_id": None,
                "author_id": author,
                "occurred_at": 1_700_000_000.0 + index,
                "parts": tuple({"kind": "text", "text": "x"} for _ in range(65)),
            }
        )
    dto = _freeze(
        {
            "schema_version": 1,
            "scope": dict(PHYSICAL_SCOPE),
            "current_event_key": "event-current",
            "messages": tuple(messages),
        }
    )
    result = _validate(dto, platform_message_id=None)
    assert result is not None
    assert result.query_text.startswith("x")


def test_owned_input_accepts_text_size_boundaries_without_mutating_source():
    messages = []
    for index in range(4):
        event_key = f"event-{index}"
        messages.append(
            {
                "event_key": event_key,
                "platform_message_id": None,
                "author_id": "synthetic-current-user"
                if index == 0
                else "synthetic-other-user",
                "occurred_at": 1_700_000_000.0 + index,
                "parts": ({"kind": "text", "text": "x" * 64_000},),
            }
        )
    dto = _freeze(
        {
            "schema_version": 1,
            "scope": dict(PHYSICAL_SCOPE),
            "current_event_key": "event-0",
            "messages": tuple(messages),
        }
    )
    result = _validate(dto, platform_message_id=None)
    assert result is not None
    assert len(result.messages[0].plain_text) == 64_000
    assert len(result.messages[1].plain_text) == 64_000


@pytest.mark.asyncio
async def test_real_adapter_platform_name_is_bound_separately_from_umo_instance_prefix():
    scope = {
        "umo": "self-instance:FriendMessage:current-user",
        "platform": "aiocqhttp",
        "self_id": "bot-1",
    }
    dto = _freeze(
        {
            "schema_version": 1,
            "scope": scope,
            "current_event_key": "host-internal-key",
            "messages": (
                {
                    "event_key": "host-internal-key",
                    "platform_message_id": "real-platform-message",
                    "author_id": "current-user",
                    "occurred_at": 1_700_000_000.0,
                    "parts": ({"kind": "text", "text": "hello"},),
                },
            ),
        }
    )
    event = SimpleNamespace(
        _cd_owned_original_input_v1=True,
        _cd_original_input_resolver_v1=lambda candidate: (
            dto if candidate is event else None
        ),
        unified_msg_origin=f"  {scope['umo']}  ",
        message_obj=SimpleNamespace(message_id=" real-platform-message "),
        get_platform_name=lambda: " AIocqhttp ",
        get_self_id=lambda: " bot-1 ",
        get_sender_id=lambda: " current-user ",
    )
    result = await MessageUtils(None, None).resolve_original_input(event)

    assert result.status == "valid"
    assert result.query_text == "hello"
    assert result.scope["umo"] == scope["umo"]


@pytest.mark.asyncio
async def test_legacy_chat_dynamics_owned_marker_with_missing_cap_fails_closed():
    event = SimpleNamespace(
        _chat_dynamics_owned_request=True,
        unified_msg_origin=PHYSICAL_SCOPE["umo"],
        get_messages=lambda: [],
        message_str="rewritten planner JSON must not be used",
    )
    result = await MessageUtils(None, None).resolve_original_input(event)

    assert result.status == "unavailable"
    assert result.query_text == ""
