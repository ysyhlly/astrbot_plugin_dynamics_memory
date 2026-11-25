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
    memory_metadata_matches_policy,
    normalize_interaction_types,
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
    assert len(policy.read_scopes) == 2
    assert policy.read_scopes[0].session_ids is None
    assert policy.read_scopes[0].interaction_types == ("group_chat",)
    assert policy.read_scopes[0].source_platform == "test"
    assert policy.read_scopes[1].session_ids == (group,)
    assert policy.read_scopes[1].interaction_types == ("unknown",)


def test_group_policy_respects_default_session_filtering_without_fanout():
    group = "test:GroupMessage:group-a"
    policy = resolve_memory_access_policy(
        _config(),
        _Event(session_id=group),
        persona_id="persona-a",
    )

    assert policy.message_kind == "group"
    assert policy.write_scope == group
    assert len(policy.read_scopes) == 1
    assert policy.read_scopes[0].session_ids == (group,)
    assert policy.read_scopes[0].interaction_types == ("group_chat", "unknown")


def test_group_policy_respects_session_mode_even_when_legacy_flag_false():
    group = "test:GroupMessage:group-a"
    policy = resolve_memory_access_policy(
        _config(
            filtering_settings={
                "memory_scope_mode": "session",
                "use_session_filtering": False,
            }
        ),
        _Event(session_id=group),
        persona_id="persona-a",
    )

    assert len(policy.read_scopes) == 1
    assert policy.read_scopes[0].session_ids == (group,)
    assert policy.read_scopes[0].interaction_types == ("group_chat", "unknown")


def test_group_policy_isolates_configured_isolated_sessions_both_directions():
    isolated = "test:GroupMessage:secret-group"
    other = "test:GroupMessage:public-group"
    config = _config(
        filtering_settings={
            "memory_scope_mode": "global",
            "use_session_filtering": False,
            "isolated_sessions": isolated,
        }
    )

    isolated_policy = resolve_memory_access_policy(
        config,
        _Event(session_id=isolated),
        persona_id="persona-a",
    )
    assert isolated_policy.write_scope == isolated
    assert len(isolated_policy.read_scopes) == 1
    assert isolated_policy.read_scopes[0].session_ids == (isolated,)
    assert isolated_policy.read_scopes[0].interaction_types == ("group_chat", "unknown")

    other_policy = resolve_memory_access_policy(
        config,
        _Event(session_id=other),
        persona_id="persona-a",
    )
    assert len(other_policy.read_scopes) == 2
    assert other_policy.read_scopes[0].session_ids is None
    assert other_policy.read_scopes[0].interaction_types == ("group_chat",)
    assert other_policy.read_scopes[0].excluded_session_ids == (isolated,)
    assert other_policy.read_scopes[1].session_ids == (other,)
    assert other_policy.read_scopes[1].interaction_types == ("unknown",)


def test_group_policy_user_mode_stays_non_private_and_single_scope():
    group = "test:GroupMessage:group-a"
    policy = resolve_memory_access_policy(
        _config(filtering_settings={"memory_scope_mode": "user"}),
        _Event(session_id=group, sender_id="user-1", platform="test"),
        persona_id="persona-a",
    )
    assert len(policy.read_scopes) == 1
    assert policy.read_scopes[0].session_ids == ("livingmemory:user:test:user-1",)
    assert policy.read_scopes[0].interaction_types == ("group_chat", "unknown")


def test_memory_metadata_matches_policy_tuple_types_and_exclusions():
    assert normalize_interaction_types("group_chat") == ("group_chat",)
    assert normalize_interaction_types(("group_chat", "unknown")) == (
        "group_chat",
        "unknown",
    )
    assert normalize_interaction_types(None) is None

    group_meta = {
        "session_id": "test:GroupMessage:g1",
        "source_session_id": "test:GroupMessage:g1",
        "source_platform": "test",
        "interaction_type": "group_chat",
        "memory_visibility": "group",
    }
    legacy_meta = {
        "session_id": "test:GroupMessage:g1",
        "source_session_id": "test:GroupMessage:g1",
        "interaction_type": "unknown",
    }
    private_meta = {
        "session_id": "test:GroupMessage:g1",
        "source_session_id": "test:FriendMessage:u1",
        "interaction_type": "unknown",
        "memory_visibility": "private",
    }
    private_chat_meta = {
        "session_id": "test:GroupMessage:g1",
        "source_session_id": "test:FriendMessage:u1",
        "interaction_type": "private_chat",
        "memory_visibility": "private",
    }

    assert memory_metadata_matches_policy(
        group_meta, interaction_type=("group_chat", "unknown")
    ) is True
    assert memory_metadata_matches_policy(
        legacy_meta, interaction_type=("group_chat", "unknown")
    ) is True
    assert memory_metadata_matches_policy(
        private_meta, interaction_type=("group_chat", "unknown")
    ) is False
    assert memory_metadata_matches_policy(
        private_chat_meta, interaction_type=("group_chat", "unknown")
    ) is False
    assert memory_metadata_matches_policy(
        {}, interaction_type=("group_chat", "unknown")
    ) is False
    assert memory_metadata_matches_policy(
        group_meta,
        interaction_type="group_chat",
        source_platform="test",
        excluded_session_ids=("test:GroupMessage:g1",),
    ) is False
    assert memory_metadata_matches_policy(
        group_meta,
        interaction_type="group_chat",
        source_platform="test",
        excluded_session_ids=("test:GroupMessage:other",),
    ) is True


@pytest.mark.asyncio
async def test_group_bm25_and_atom_search_respect_tuple_types_and_excluded_sessions(
    tmp_path,
):
    import json
    import time
    import aiosqlite
    from astrbot_plugin_dynamics_memory.core.models.memory_atom import (
        AtomType,
        DecayType,
        MemoryAtom,
    )
    from astrbot_plugin_dynamics_memory.core.processors.text_processor import (
        TextProcessor,
    )
    from astrbot_plugin_dynamics_memory.core.retrieval.bm25_retriever import (
        BM25Retriever,
    )
    from astrbot_plugin_dynamics_memory.storage.atom_store import AtomStore

    db_path = str(tmp_path / "scope_isolation.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
        )
        rows = [
            (
                1,
                "alpha coffee club",
                {
                    "session_id": "qq:GroupMessage:g1",
                    "source_session_id": "qq:GroupMessage:g1",
                    "source_platform": "qq",
                    "interaction_type": "group_chat",
                    "memory_visibility": "group",
                    "status": "active",
                },
            ),
            (
                2,
                "alpha legacy coffee note",
                {
                    "session_id": "qq:GroupMessage:g1",
                    "source_session_id": "qq:GroupMessage:g1",
                    "interaction_type": "unknown",
                    "status": "active",
                },
            ),
            (
                3,
                "alpha private coffee secret",
                {
                    "session_id": "qq:GroupMessage:g1",
                    "source_session_id": "qq:FriendMessage:u1",
                    "interaction_type": "private_chat",
                    "memory_visibility": "private",
                    "status": "active",
                },
            ),
            (
                4,
                "alpha secret group coffee",
                {
                    "session_id": "qq:GroupMessage:secret",
                    "source_session_id": "qq:GroupMessage:secret",
                    "source_platform": "qq",
                    "interaction_type": "group_chat",
                    "memory_visibility": "group",
                    "status": "active",
                },
            ),
        ]
        for doc_id, text, meta in rows:
            await db.execute(
                "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
                (doc_id, text, json.dumps(meta)),
            )
        await db.commit()

    tp = TextProcessor()
    bm25 = BM25Retriever(db_path, tp)
    await bm25.initialize()
    for doc_id, text, _ in rows:
        await bm25.add_document(doc_id, text)

    # Session-isolated group search returns group_chat + legacy unknown, excludes private & other groups
    own_hits = await bm25.search(
        "alpha coffee",
        limit=10,
        session_id="qq:GroupMessage:g1",
        interaction_type=("group_chat", "unknown"),
    )
    assert {hit.doc_id for hit in own_hits} == {1, 2}

    # Cross-group search excludes isolated_sessions
    cross_hits = await bm25.search(
        "alpha coffee",
        limit=10,
        session_id=None,
        interaction_type="group_chat",
        source_platform="qq",
        excluded_session_ids=("qq:GroupMessage:secret",),
    )
    assert {hit.doc_id for hit in cross_hits} == {1}

    # AtomStore respects the exact same tuple interaction_type and excluded_session_ids
    atom_store = AtomStore(db_path, text_processor=tp)
    await atom_store.initialize()
    now = time.time()
    for doc_id, text, meta in rows:
        await atom_store.insert_many(
            [
                MemoryAtom(
                    parent_memory_id=doc_id,
                    content=text,
                    atom_type=AtomType.FACTUAL,
                    importance=0.8,
                    confidence=0.9,
                    ttl_days=30.0,
                    decay_type=DecayType.EXPONENTIAL,
                    created_at=now,
                    last_accessed_at=now,
                    expires_at=now + 86400 * 30,
                    session_id=meta["session_id"],
                    persona_id="p1",
                )
            ]
        )

    own_atoms = await atom_store.search_fts(
        "alpha coffee",
        limit=10,
        session_id="qq:GroupMessage:g1",
        interaction_type=("group_chat", "unknown"),
    )
    assert {atom.parent_memory_id for atom in own_atoms} == {1, 2}

    cross_atoms = await atom_store.search_fts(
        "alpha coffee",
        limit=10,
        session_id=None,
        interaction_type="group_chat",
        source_platform="qq",
        excluded_session_ids=("qq:GroupMessage:secret",),
    )
    assert {atom.parent_memory_id for atom in cross_atoms} == {1}

    # GraphStore read filters respect tuple interaction_type and excluded_session_ids
    from astrbot_plugin_dynamics_memory.core.models.graph_models import (
        GraphEntry,
        GraphNode,
    )
    from astrbot_plugin_dynamics_memory.storage.graph_store import GraphStore

    graph_store = GraphStore(db_path)
    await graph_store.initialize()
    node_map = await graph_store.upsert_nodes([GraphNode("topic", "coffee", "coffee")])
    await graph_store.add_entries(
        [
            GraphEntry(
                entry_key=f"entry-{doc_id}",
                source_memory_id=doc_id,
                session_id=meta["session_id"],
                persona_id="p1",
                entry_type="summary",
                content=text,
                metadata={"importance": 0.8},
                node_keys=["topic:coffee"],
            )
            for doc_id, text, meta in rows
        ],
        node_map,
        {},
    )

    own_graph = await graph_store.search_entries_by_bm25(
        '"coffee"',
        limit=10,
        session_id="qq:GroupMessage:g1",
        interaction_type=("group_chat", "unknown"),
    )
    assert {row["source_memory_id"] for row in own_graph} == {1, 2}

    cross_graph = await graph_store.search_entries_by_bm25(
        '"coffee"',
        limit=10,
        session_id=None,
        interaction_type="group_chat",
        source_platform="qq",
        excluded_session_ids=("qq:GroupMessage:secret",),
    )
    assert {row["source_memory_id"] for row in cross_graph} == {1}

    node_id = node_map["topic:coffee"]
    own_hop = await graph_store.get_entries_for_node_ids(
        node_ids=[node_id],
        limit=10,
        session_id="qq:GroupMessage:g1",
        interaction_type=("group_chat", "unknown"),
    )
    assert {row["source_memory_id"] for row in own_hop} == {1, 2}

    cross_hop = await graph_store.get_entries_for_node_ids(
        node_ids=[node_id],
        limit=10,
        session_id=None,
        interaction_type="group_chat",
        source_platform="qq",
        excluded_session_ids=("qq:GroupMessage:secret",),
    )
    assert {row["source_memory_id"] for row in cross_hop} == {1}
    await graph_store.close()


def test_vector_metadata_filter_supports_tuple_interaction_types_and_exclusions():
    from astrbot_plugin_dynamics_memory.core.retrieval.vector_search import (
        _metadata_matches_filters,
    )

    group_meta = {
        "session_id": "qq:GroupMessage:g1",
        "source_session_id": "qq:GroupMessage:g1",
        "interaction_type": "group_chat",
        "memory_visibility": "group",
    }
    legacy_meta = {
        "session_id": "qq:GroupMessage:g1",
        "source_session_id": "qq:GroupMessage:g1",
        "interaction_type": "unknown",
    }
    private_meta = {
        "session_id": "qq:GroupMessage:g1",
        "source_session_id": "qq:FriendMessage:u1",
        "interaction_type": "private_chat",
        "memory_visibility": "private",
    }
    secret_meta = {
        "session_id": "qq:GroupMessage:secret",
        "source_session_id": "qq:GroupMessage:secret",
        "interaction_type": "group_chat",
        "memory_visibility": "group",
    }

    own_filters = {
        "session_id": "qq:GroupMessage:g1",
        "__interaction_types__": ("group_chat", "unknown"),
        "__exclude_private_visibility__": True,
    }
    assert _metadata_matches_filters(group_meta, own_filters) is True
    assert _metadata_matches_filters(legacy_meta, own_filters) is True
    assert _metadata_matches_filters(private_meta, own_filters) is False

    cross_filters = {
        "interaction_type": "group_chat",
        "__source_platform__": "qq",
        "__exclude_private_visibility__": True,
        "__excluded_session_ids__": ("qq:GroupMessage:secret",),
    }
    assert _metadata_matches_filters(group_meta, cross_filters) is True
    assert _metadata_matches_filters(secret_meta, cross_filters) is False


@pytest.mark.asyncio
async def test_group_recall_fanout_degrades_from_3x_to_1x_isolated_and_2x_shared():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall import (
        MemoryRecall,
    )

    async def _run_group_recall(filtering_overrides: dict) -> list[dict]:
        calls: list[dict] = []

        async def _fake_search(**kwargs):
            calls.append(kwargs)
            return []

        config_data = _config(filtering_settings=filtering_overrides)
        config_data["recall_engine"] = {"enabled": True, "top_k": 5}
        config_manager = SimpleNamespace(
            get=lambda key, default=None: _config_lookup(config_data, key, default)
        )
        engine = SimpleNamespace(search_memories=_fake_search)
        conv_mgr = SimpleNamespace(
            get_recent_messages=AsyncMock(return_value=[]),
            get_session_info=AsyncMock(return_value=None),
            store=SimpleNamespace(get_message_count=AsyncMock(return_value=0)),
        )
        context = SimpleNamespace(
            get_all_stars=lambda: [],
            conversation_manager=SimpleNamespace(
                get_curr_conversation_id=AsyncMock(return_value="conv-1"),
                get_conversation=AsyncMock(
                    return_value=SimpleNamespace(persona_id="persona-a")
                ),
            ),
            persona_manager=SimpleNamespace(
                get_default_persona_v3=AsyncMock(return_value={"name": "persona-a"})
            ),
        )
        msg_utils = SimpleNamespace(
            resolve_original_input=AsyncMock(
                return_value=SimpleNamespace(
                    status="not_owned",
                    query_text="remember our coffee plan",
                    scope={"platform": "qq", "umo": "qq:GroupMessage:group-a"},
                    is_owned=False,
                )
            ),
            clean_message_text=lambda text: text,
        )
        injection_adapter = SimpleNamespace(
            inject=lambda req, content: None,
        )
        mgr = MemoryRecall(
            context=context,
            config_manager=config_manager,
            memory_engine=engine,
            conversation_manager=conv_mgr,
            message_utils=msg_utils,
            injection_adapter=injection_adapter,
        )
        event = _Event(
            session_id="qq:GroupMessage:group-a",
            platform="aiocqhttp",
            platform_id="qq",
            group_id="group-a",
        )
        event.message_str = "remember our coffee plan"
        event.get_message_type = lambda: MessageType.GROUP_MESSAGE
        event.get_extra = lambda _k, default=None: default
        req = SimpleNamespace(
            prompt="remember our coffee plan",
            contexts=[],
            extra_user_content_parts=[],
            system_prompt="",
        )
        await mgr.handle_memory_recall(event, req)
        return calls

    def _config_lookup(data, key, default=None):
        cur = data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    # 1. Default config (use_session_filtering=True): 1x single-scope search, no cross-group call
    default_calls = await _run_group_recall({})
    assert len(default_calls) == 1
    assert default_calls[0]["session_id"] == "qq:GroupMessage:group-a"
    assert default_calls[0]["interaction_type"] == ("group_chat", "unknown")
    assert "excluded_session_ids" not in default_calls[0]

    # 2. Cross-group enabled with isolated_sessions: 2x (NOT 3x), excluding isolated group
    shared_calls = await _run_group_recall(
        {
            "use_session_filtering": False,
            "isolated_sessions": "qq:GroupMessage:secret",
        }
    )
    assert len(shared_calls) == 2
    assert shared_calls[0]["session_id"] is None
    assert shared_calls[0]["interaction_type"] == "group_chat"
    assert shared_calls[0]["source_platform"] == "qq"
    assert shared_calls[0]["excluded_session_ids"] == ("qq:GroupMessage:secret",)
    assert shared_calls[1]["session_id"] == "qq:GroupMessage:group-a"
    assert shared_calls[1]["interaction_type"] == "unknown"


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
