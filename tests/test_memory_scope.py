"""Memory access, identity mapping, and scope resolution tests."""

import pytest
from astrbot.api.platform import (
    AstrBotMessage,
    AstrMessageEvent,
    MessageMember,
    MessageType,
    PlatformMetadata,
)

from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
    ConversationManager,
)
from astrbot_plugin_dynamics_memory.core.memory_scope import (
    GLOBAL_MEMORY_SCOPE,
    classify_umo,
    is_event_memory_allowed,
    is_suspicious_session_id,
    parse_identity_aliases,
    parse_value_list,
    resolve_event_identity,
    resolve_memory_scope,
    resolve_memory_access_policy,
    resolve_verified_group_sessions,
    resolve_sender_alias,
)
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore


class _Event:
    def __init__(
        self,
        session_id: str = "test:private:session-1",
        sender_id: str = "user-1",
        sender_name: str = "Original Name",
        platform: str = "test",
        platform_id: str | None = None,
        group_id: str = "",
        self_id: str = "bot-1",
    ) -> None:
        self.unified_msg_origin = session_id
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.platform = platform
        self.platform_id = platform_id or platform
        self.group_id = group_id
        self.self_id = self_id

    def get_sender_id(self):
        return self.sender_id

    def get_sender_name(self):
        return self.sender_name

    def get_platform_name(self):
        return self.platform

    def get_platform_id(self):
        return self.platform_id

    def get_self_id(self):
        return self.self_id

    def get_group_id(self):
        return self.group_id


def _config(**overrides):
    config = {
        "access_control": {
            "whitelist_enabled": False,
            "allowed_ids": "",
            "identity_aliases": "",
        },
        "filtering_settings": {
            "memory_scope_mode": "legacy",
            "use_session_filtering": True,
            "isolated_sessions": "",
            "enable_group_to_private_recall": False,
            "private_shared_group_sources": [],
            "private_shared_group_instance_id": "",
        },
    }
    for section, values in overrides.items():
        config[section].update(values)
    return config


def test_parse_lists_and_aliases():
    assert parse_value_list("one, two;three\none") == ["one", "two", "three"]
    assert parse_identity_aliases("test:user-1=Alex\nOriginal Name=Alice") == {
        "test:user-1": "Alex",
        "original name": "Alice",
    }


@pytest.mark.parametrize(
    "allowed_ids",
    [
        "user-1",
        "test:user-1",
        "test:private:session-1",
        "group-1",
        "Canonical Name",
    ],
)
def test_whitelist_accepts_supported_identifiers(allowed_ids):
    config = _config(
        access_control={
            "whitelist_enabled": True,
            "allowed_ids": allowed_ids,
            "identity_aliases": "test:user-1=Canonical Name",
        }
    )
    assert is_event_memory_allowed(config, _Event(group_id="group-1")) is True


def test_enabled_empty_whitelist_denies_access():
    config = _config(access_control={"whitelist_enabled": True, "allowed_ids": ""})
    assert is_event_memory_allowed(config, _Event()) is False


def test_alias_priority_and_sender_mapping():
    config = _config(
        access_control={
            "identity_aliases": (
                "Original Name=By Name\nuser-1=By ID\ntest:user-1=By Platform"
            )
        }
    )
    assert resolve_event_identity(config, _Event()) == "By Platform"
    assert (
        resolve_sender_alias(
            config["access_control"]["identity_aliases"],
            "test",
            "user-1",
            "Original Name",
        )
        == "By Platform"
    )


def test_scope_modes_and_isolated_override():
    first = _Event(session_id="test:private:first")
    second = _Event(session_id="test:group:second")
    user_config = _config(filtering_settings={"memory_scope_mode": "user"})
    assert resolve_memory_scope(user_config, first) == resolve_memory_scope(
        user_config, second
    )

    assert (
        resolve_memory_scope(
            _config(filtering_settings={"memory_scope_mode": "global"}), first
        )
        == GLOBAL_MEMORY_SCOPE
    )
    assert (
        resolve_memory_scope(
            _config(filtering_settings={"memory_scope_mode": "session"}), first
        )
        == first.unified_msg_origin
    )
    assert (
        resolve_memory_scope(
            _config(
                filtering_settings={
                    "memory_scope_mode": "global",
                    "isolated_sessions": first.unified_msg_origin,
                }
            ),
            first,
        )
        == first.unified_msg_origin
    )


def test_legacy_scope_compatibility():
    event = _Event()
    assert resolve_memory_scope(_config(), event) == event.unified_msg_origin
    assert (
        resolve_memory_scope(
            _config(filtering_settings={"use_session_filtering": False}), event
        )
        is None
    )


def test_legacy_shared_pool_excludes_configured_isolated_sessions():
    shared = _Event(session_id="test:private:shared")
    isolated = _Event(session_id="test:private:isolated")
    config = _config(
        filtering_settings={
            "use_session_filtering": False,
            "isolated_sessions": isolated.unified_msg_origin,
        }
    )

    assert resolve_memory_scope(config, shared) == GLOBAL_MEMORY_SCOPE
    assert resolve_memory_scope(config, isolated) == isolated.unified_msg_origin


def test_classify_umo_fails_closed_for_unknown_message_types():
    assert classify_umo("test:GroupMessage:group-1") == "group"
    assert classify_umo("test:FriendMessage:user-1") == "private"
    assert classify_umo("test:PrivateMessage:user-1") == "private"
    assert classify_umo("test:SomethingElse:session-1") == "unknown"
    assert classify_umo("") == "unknown"


def test_private_policy_keeps_private_write_and_adds_only_verified_group_reads():
    private = "test-instance:FriendMessage:user-a"
    group = "test-instance:GroupMessage:group-b"
    other_platform = "other-instance:GroupMessage:group-c"
    config = _config(
        filtering_settings={
            "use_session_filtering": False,
            "enable_group_to_private_recall": True,
            "private_shared_group_sources": [
                group,
                other_platform,
                "test:FriendMessage:user-z",
                "test:GroupMessage:not-captured",
            ],
            "private_shared_group_instance_id": "persona-a",
        }
    )

    policy = resolve_memory_access_policy(
        config,
        _Event(session_id=private, platform="aiocqhttp", platform_id="test-instance"),
        persona_id="persona-a",
        verified_group_sessions=[group, other_platform],
    )

    assert policy.message_kind == "private"
    assert policy.write_scope == private
    assert policy.write_interaction_type == "private_chat"
    assert policy.write_visibility == "private"
    assert policy.read_scopes[0].session_ids == (private,)
    assert len(policy.read_scopes) == 2
    assert policy.read_scopes[1].session_ids is None
    assert policy.read_scopes[1].source_session_id == group
    assert policy.read_scopes[1].interaction_types == ("group_chat",)
    assert policy.read_scopes[1].require_persona is True
    assert policy.read_scopes[1].source_platform == "test-instance"


@pytest.mark.asyncio
async def test_verified_group_scope_uses_platform_id_and_adapter_name_separately():
    group = "test-instance:GroupMessage:group-b"
    private = "test-instance:FriendMessage:user-a"
    config = _config(
        filtering_settings={
            "enable_group_to_private_recall": True,
            "private_shared_group_sources": [group],
            "private_shared_group_instance_id": "persona-a",
        }
    )

    class _ConversationManager:
        async def get_session_info(self, session_id):
            assert session_id == group
            return type("Session", (), {"platform": "aiocqhttp"})()

    event = _Event(
        session_id=private,
        platform="aiocqhttp",
        platform_id="test-instance",
    )
    verified = await resolve_verified_group_sessions(
        config, event, _ConversationManager()
    )
    policy = resolve_memory_access_policy(
        config,
        event,
        persona_id="persona-a",
        verified_group_sessions=verified,
    )

    assert verified == (group,)
    assert policy.read_scopes[1].source_session_id == group
    assert policy.read_scopes[1].source_platform == "test-instance"


def test_private_policy_is_disabled_by_default_and_without_persona():
    private = "test-instance:FriendMessage:user-a"
    group = "test-instance:GroupMessage:group-b"
    config = _config(filtering_settings={"private_shared_group_sources": [group]})
    disabled = resolve_memory_access_policy(
        config,
        _Event(session_id=private, platform="aiocqhttp", platform_id="test-instance"),
        persona_id="persona-a",
        verified_group_sessions=[group],
    )
    no_persona = resolve_memory_access_policy(
        _config(
            filtering_settings={
                "enable_group_to_private_recall": True,
                "private_shared_group_sources": [group],
                "private_shared_group_instance_id": "persona-a",
            }
        ),
        _Event(session_id=private, platform="aiocqhttp", platform_id="test-instance"),
        persona_id=None,
        verified_group_sessions=[group],
    )

    assert len(disabled.read_scopes) == 1
    assert len(no_persona.read_scopes) == 1
    assert disabled.write_scope == no_persona.write_scope == private


def test_group_policy_reads_group_only_and_keeps_unknown_in_own_scope():
    group = "test:GroupMessage:group-a"
    policy = resolve_memory_access_policy(
        _config(filtering_settings={"use_session_filtering": False}),
        _Event(session_id=group),
        persona_id="persona-a",
    )

    assert policy.message_kind == "group"
    assert policy.write_scope == group
    assert policy.write_interaction_type == "group_chat"
    assert policy.write_visibility == "group"
    assert policy.read_scopes[0].session_ids is None
    assert policy.read_scopes[0].interaction_types == ("group_chat",)
    assert policy.read_scopes[1].session_ids == (group,)
    assert policy.read_scopes[1].interaction_types == ("group_chat",)
    assert policy.read_scopes[2].session_ids == (group,)
    assert policy.read_scopes[2].interaction_types == ("unknown",)


def test_unknown_origin_is_private_only_and_cannot_import_group_memory():
    unknown = "test:UnknownEvent:session-a"
    group = "test:GroupMessage:group-a"
    policy = resolve_memory_access_policy(
        _config(
            filtering_settings={
                "enable_group_to_private_recall": True,
                "private_shared_group_sources": [group],
                "private_shared_group_instance_id": "persona-a",
            }
        ),
        _Event(session_id=unknown),
        persona_id="persona-a",
        verified_group_sessions=[group],
    )

    assert policy.message_kind == "unknown"
    assert policy.write_scope == unknown
    assert policy.write_visibility == "private"
    assert len(policy.read_scopes) == 1
    assert policy.read_scopes[0].session_ids == (unknown,)


@pytest.mark.asyncio
async def test_conversation_manager_applies_user_alias_only_to_user_messages(tmp_path):
    store = ConversationStore(str(tmp_path / "aliases.db"))
    await store.initialize()
    manager = ConversationManager(
        store=store,
        identity_aliases="test:user-1=Canonical Name",
    )
    event = _Event()

    user_message = await manager.add_message_from_event(event, "user", "hello")
    assistant_message = await manager.add_message_from_event(event, "assistant", "hi")

    assert user_message.sender_name == "Canonical Name"
    assert assistant_message.sender_id == "bot-1"
    assert assistant_message.sender_name == "bot-1"
    await store.close()


def test_is_suspicious_session_id_detects_leaked_errors():
    assert is_suspicious_session_id("aiocqhttp:Error: connection refused") is True
    assert is_suspicious_session_id("webrtc:ERROR:timeout") is True
    assert is_suspicious_session_id("test:private:session-1") is False
    assert is_suspicious_session_id("") is False
    assert is_suspicious_session_id(None) is False


def _adapter_event(adapter_id, platform_name):
    message = AstrBotMessage()
    message.type = MessageType.FRIEND_MESSAGE
    message.sender = MessageMember("user-1")
    message.message = []
    return AstrMessageEvent(
        "", message, PlatformMetadata(platform_name, "test", adapter_id), "user-1"
    )


def test_shared_adapters_and_isolated_instance_scope():
    from astrbot_plugin_dynamics_memory.core.memory_scope import (
        adapter_access,
        adapter_sql,
        current_adapter_access,
    )

    config = _config(
        filtering_settings={"shared_adapter_ids": ["qq-instance", "wechat-instance"]}
    )
    with adapter_access(config, _adapter_event("qq-instance", "qq")):
        access = current_adapter_access.get()
        assert access.adapter_id == "qq-instance"
        assert access.adapter_ids == ("qq-instance", "wechat-instance")
        sql, params = adapter_sql("metadata")
        assert "IN (?,?)" in sql
        assert params == ["qq-instance", "wechat-instance"]
    with adapter_access(config, _adapter_event("telegram-instance", "telegram")):
        assert current_adapter_access.get().adapter_ids == ("telegram-instance",)
    assert current_adapter_access.get() is None


def test_adapter_scope_preserves_existing_session_and_persona_settings():
    from astrbot_plugin_dynamics_memory.core.memory_scope import (
        adapter_access,
        adapter_sql,
    )

    event = _adapter_event("test-instance", "test")
    legacy = _config()
    with adapter_access(legacy, event):
        assert adapter_sql("metadata") == ("1", [])
        assert resolve_memory_scope(legacy, event) == event.unified_msg_origin
    config = _config(
        filtering_settings={"shared_adapter_ids": [], "memory_scope_mode": "global"}
    )
    with adapter_access(config, event):
        assert adapter_sql("metadata")[1] == ["test-instance"]
        assert resolve_memory_scope(config, event) == GLOBAL_MEMORY_SCOPE


@pytest.mark.asyncio
async def test_adapter_access_is_task_local():
    import asyncio
    from astrbot_plugin_dynamics_memory.core.memory_scope import (
        adapter_access,
        current_adapter_access,
    )

    config = _config(filtering_settings={"shared_adapter_ids": []})

    async def read(platform):
        with adapter_access(config, _adapter_event(f"{platform}-instance", platform)):
            await asyncio.sleep(0)
            return current_adapter_access.get().adapter_ids

    assert await asyncio.gather(read("qq"), read("wechat")) == [
        ("qq-instance",),
        ("wechat-instance",),
    ]
