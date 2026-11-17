"""Adapter filtering with real SQL, FAISS, graph and concurrent requests."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import aiosqlite
import pytest_asyncio
from astrbot_plugin_dynamics_memory.core.models.memory_atom import MemoryAtom

import pytest

from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB
from astrbot.api.provider import ProviderRequest
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.memory_scope import (
    adapter_access,
    GLOBAL_MEMORY_SCOPE,
)
from astrbot_plugin_dynamics_memory.core.passive_group_capture import (
    get_active_plugin,
    set_active_plugin,
)
from astrbot_plugin_dynamics_memory.core.tools.memory_search_tool import (
    MemorySearchTool,
)
from tests.integration.test_real_db_end_to_end import (
    _DeterministicEmbeddingProvider,
    _TestEvent,
)


@pytest.mark.asyncio
async def test_adapter_filter_intersects_persona_and_prevents_cache_and_id_leaks(
    tmp_path,
):
    db = FaissVecDB(
        str(tmp_path / "main.db"),
        str(tmp_path / "main.index"),
        _DeterministicEmbeddingProvider(),
    )
    await db.initialize()
    engine = MemoryEngine(
        str(tmp_path / "main.db"), db, config={"recent_memory_count": 0}
    )
    await engine.initialize()
    ids = {}
    try:
        for owner, persona in [
            ("qq-a", "p"),
            ("wechat-a", "p"),
            ("telegram-a", "p"),
            ("qq-a", "other"),
        ]:
            ids[owner, persona] = await engine.add_memory(
                "memory coffee preference " + owner,
                session_id=GLOBAL_MEMORY_SCOPE,
                persona_id=persona,
                metadata={"adapter_id": owner},
            )
        config = {"filtering_settings": {"shared_adapter_ids": ["qq-a", "wechat-a"]}}
        with adapter_access(config, "qq-a"):
            results = await engine.search_memories(
                "coffee", 10, GLOBAL_MEMORY_SCOPE, "p"
            )
            assert {item.doc_id for item in results} == {
                ids["qq-a", "p"],
                ids["wechat-a", "p"],
            }
            assert await engine.get_memory(ids["telegram-a", "p"]) is None
            assert await engine.delete_memory(ids["telegram-a", "p"]) is False
        with adapter_access(config, "telegram-a"):
            results = await engine.search_memories(
                "coffee", 10, GLOBAL_MEMORY_SCOPE, "p"
            )
            assert {item.doc_id for item in results} == {ids["telegram-a", "p"]}
        assert await engine.get_memory(ids["telegram-a", "p"]) is not None
    finally:
        await engine.close()
        await db.close()


@pytest_asyncio.fixture
async def adapter_engine(tmp_path):
    provider = _DeterministicEmbeddingProvider()
    db = FaissVecDB(str(tmp_path / "main.db"), str(tmp_path / "main.index"), provider)
    graph = FaissVecDB(
        str(tmp_path / "graph.db"), str(tmp_path / "graph.index"), provider
    )
    await db.initialize()
    await graph.initialize()
    engine = MemoryEngine(
        str(tmp_path / "main.db"),
        db,
        graph_vector_db=graph,
        config={"recent_memory_count": 0, "graph_memory_enabled": True},
    )
    await engine.initialize()
    try:
        yield engine
    finally:
        await engine.close()
        await db.close()


@pytest.mark.asyncio
async def test_graph_atoms_recent_and_agent_evidence_share_adapter_boundary(
    adapter_engine,
):
    engine = adapter_engine
    ids = {}
    for owner in ("qq-a", "wechat-a", "telegram-a"):
        ids[owner] = await engine.add_memory(
            "shared coffee fact " + owner,
            session_id=GLOBAL_MEMORY_SCOPE,
            persona_id="p",
            metadata={
                "adapter_id": owner,
                "topics": ["coffee"],
                "key_facts": ["coffee fact " + owner],
            },
            atoms=[
                MemoryAtom(
                    parent_memory_id=0,
                    content="coffee fact " + owner,
                    expires_at=time.time() + 86400,
                )
            ],
        )
    config = {"filtering_settings": {"shared_adapter_ids": ["qq-a", "wechat-a"]}}
    with adapter_access(config, "qq-a"):
        allowed = {ids["qq-a"], ids["wechat-a"]}
        assert {
            hit.parent_memory_id
            for hit in await engine.atom_retriever.search(
                "coffee", 10, GLOBAL_MEMORY_SCOPE, "p"
            )
        } == allowed
        assert {
            hit.doc_id
            for hit in await engine._get_recent_memory_results(
                10, GLOBAL_MEMORY_SCOPE, "p"
            )
        } == allowed
        snapshot = await engine.graph_store.get_full_graph_snapshot(
            GLOBAL_MEMORY_SCOPE, "p"
        )
        assert {row["memory_id"] for row in snapshot["memories"]} == allowed
        assert await engine.get_memory_for_recall(ids["telegram-a"]) is None
        hits = await engine.graph_vector_retriever.search(
            "coffee", 10, GLOBAL_MEMORY_SCOPE, "p"
        )
        assert {hit.doc_id for hit in hits} == allowed
        stats = await engine.get_statistics()
        assert stats["total_memories"] == 2


@pytest.mark.asyncio
async def test_sharing_does_not_remove_session_isolation_and_writes_keep_their_owner(
    adapter_engine,
):
    engine = adapter_engine
    config = {"filtering_settings": {"shared_adapter_ids": ["qq-a", "wechat-a"]}}
    for owner, scope in [
        ("qq-a", "qq-a:FriendMessage:123"),
        ("wechat-a", "wechat-a:FriendMessage:abc"),
    ]:
        with adapter_access(config, owner, write=True):
            await engine.add_memory(
                "coffee preference", session_id=scope, persona_id="p"
            )
    with adapter_access(config, "qq-a"):
        hits = await engine.search_memories("coffee", 10, "qq-a:FriendMessage:123", "p")
        assert len(hits) == 1
        assert hits[0].metadata["adapter_id"] == "qq-a"


@pytest.mark.asyncio
async def test_adapter_predicate_precedes_keyword_limit_and_concurrent_caches(
    adapter_engine,
):
    engine = adapter_engine
    for i in range(12):
        await engine.add_memory(
            "coffee",
            session_id=GLOBAL_MEMORY_SCOPE,
            metadata={"adapter_id": "telegram-a"},
        )
    own = await engine.add_memory(
        "coffee", session_id=GLOBAL_MEMORY_SCOPE, metadata={"adapter_id": "qq-a"}
    )
    config = {"filtering_settings": {"shared_adapter_ids": []}}
    with adapter_access(config, "qq-a"):
        assert [
            hit.doc_id
            for hit in await engine.bm25_retriever.search(
                "coffee", 1, GLOBAL_MEMORY_SCOPE
            )
        ] == [own]

    async def search(owner):
        with adapter_access(config, owner):
            await asyncio.sleep(0)
            return await engine.search_memories("coffee", 3, GLOBAL_MEMORY_SCOPE)

    own_hits, other_hits = await asyncio.gather(search("qq-a"), search("telegram-a"))
    assert {hit.doc_id for hit in own_hits} == {own}
    assert own not in {hit.doc_id for hit in other_hits}


@pytest.mark.asyncio
async def test_consolidation_candidates_and_statistics_follow_the_adapter_filter(
    adapter_engine,
):
    engine = adapter_engine
    for owner in ("qq-a", "telegram-a"):
        await engine.add_memory(
            "coffee",
            session_id=GLOBAL_MEMORY_SCOPE,
            importance=0.1,
            metadata={"adapter_id": owner, "create_time": time.time() - 86400},
            preserve_create_time=True,
        )
    with adapter_access({"filtering_settings": {"shared_adapter_ids": []}}, "qq-a"):
        candidates = await engine.get_consolidation_candidates(0.5, time.time())
        assert len(candidates) == 1
        assert candidates[0]["metadata"]["adapter_id"] == "qq-a"
        assert (await engine.get_statistics())["total_memories"] == 1


@pytest.mark.asyncio
async def test_vector_adapter_candidates_are_selected_before_the_global_expansion_cap(
    adapter_engine,
):
    import json
    import numpy as np

    engine = adapter_engine
    total = 8300
    query = np.asarray(
        await engine.faiss_db.embedding_provider.get_embedding("coffee"),
        dtype=np.float32,
    )
    vectors = np.tile(query, (total, 1))
    vectors[-1] *= 0.9
    ids = np.arange(1, total + 1, dtype=np.int64)
    engine.faiss_db.embedding_storage.index.add_with_ids(vectors, ids)
    rows = [
        (
            int(i),
            str(i),
            "coffee",
            json.dumps(
                {"adapter_id": "qq-a" if i == total else "other", "status": "active"}
            ),
        )
        for i in ids
    ]
    await engine.db_connection.executemany(
        "INSERT INTO documents(id, doc_id, text, metadata) VALUES(?,?,?,?)", rows
    )
    await engine.db_connection.commit()
    from astrbot_plugin_dynamics_memory.core.faiss_async_persist import (
        install_async_persist,
    )

    persister = install_async_persist(engine.faiss_db.embedding_storage)
    try:
        with adapter_access({"filtering_settings": {"shared_adapter_ids": []}}, "qq-a"):
            hits = await engine.vector_retriever.search("coffee", 1)
            assert [hit.doc_id for hit in hits] == [total]
    finally:
        await persister.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_persona_filtering", [True, False])
async def test_search_tool_preserves_private_origin_with_shared_adapters(
    real_db_stack,
    use_persona_filtering,  # noqa: F811
):
    engine = real_db_stack["memory_engine"]
    plugin = real_db_stack["plugin"]
    own_scope = "qq-a:FriendMessage:123"
    ids = []
    for owner, scope, persona in [
        ("qq-a", own_scope, "persona-real"),
        ("qq-a", own_scope, "different-persona"),
        ("qq-a", "qq-a:FriendMessage:other", "persona-real"),
        ("wechat-a", "wechat-a:FriendMessage:abc", "persona-real"),
        ("qq-b", own_scope, "persona-real"),
    ]:
        ids.append(
            await engine.add_memory(
                "coffee preference",
                session_id=scope,
                persona_id=persona,
                metadata={"adapter_id": owner},
            )
        )
    config = ConfigManager(
        {
            "filtering_settings": {
                "shared_adapter_ids": ["qq-a", "wechat-a"],
                "use_session_filtering": False,
                "use_persona_filtering": use_persona_filtering,
            }
        }
    )
    plugin.config_manager = config
    event = _TestEvent(own_scope, "coffee")
    tool = MemorySearchTool(
        context=plugin.context, config_manager=config, memory_engine=engine
    )
    context = SimpleNamespace(context=SimpleNamespace(event=event))
    validators = []

    def register_validator(validator, *, tool):
        validators.append(validator)
        return True

    event._cd_tool_memory_sources_register_v1 = register_validator
    previous_plugin = get_active_plugin()
    set_active_plugin(plugin)
    try:
        output = json.loads(await tool.call(context, query="coffee"))
        assert output["status"] == "ok", output
        expected = set(ids[:1] if use_persona_filtering else ids[:2])
        assert {row["id"] for row in output["results"]} == expected
        assert output["applied_filters"] == {
            "persona_filtered": use_persona_filtering,
            "session_filtered": True,
        }
        assert len(validators) == 1
        assert await validators[0]() is True

        if not use_persona_filtering:
            filtering = config._config["filtering_settings"]
            filtering["use_persona_filtering"] = True
            assert await validators[0]() is False
            filtering["use_persona_filtering"] = False

        async with aiosqlite.connect(real_db_stack["memory_db_path"]) as db:
            await db.execute(
                "UPDATE documents SET metadata = json_set(metadata, '$.adapter_id', ?) WHERE id = ?",
                ("qq-b", ids[0]),
            )
            await db.commit()
            assert await validators[0]() is False
            await db.execute(
                "UPDATE documents SET metadata = json_set(metadata, '$.adapter_id', ?) WHERE id = ?",
                ("qq-a", ids[0]),
            )
            await db.commit()

        plugin.context.stars.clear()
        blocked = json.loads(await tool.call(context, query="coffee"))
        assert blocked["status"] == "unavailable"
        assert blocked["error"] == "source validation failed"
        assert blocked["results"] == []
    finally:
        set_active_plugin(previous_plugin)
        await tool.recall_runtime.close()


@pytest.mark.asyncio
async def test_search_tool_accepts_verified_group_only_while_sharing_is_enabled(
    real_db_stack,
):  # noqa: F811
    engine = real_db_stack["memory_engine"]
    plugin = real_db_stack["plugin"]
    manager = real_db_stack["conversation_manager"]
    own_group = "qq-a:GroupMessage:captured"
    wrong_persona = "qq-a:GroupMessage:other-persona"
    uncaptured = "qq-a:GroupMessage:uncaptured"
    other_adapter = "wechat-a:GroupMessage:captured"
    for group in (own_group, wrong_persona, other_adapter):
        await manager.add_message(
            group,
            "user",
            "coffee group message",
            sender_id="user-2",
            group_id="captured",
            platform="test",
        )
    ids = []
    for group in (own_group, wrong_persona, uncaptured, other_adapter):
        ids.append(
            await engine.add_memory(
                "coffee group memory",
                session_id=group,
                persona_id="different-persona"
                if group == wrong_persona
                else "persona-real",
            )
        )

    config = ConfigManager(
        {
            "filtering_settings": {
                "enable_group_to_private_recall": True,
                "private_shared_group_instance_id": "persona-real",
                "private_shared_group_sources": "\n".join(
                    (own_group, wrong_persona, uncaptured, other_adapter)
                ),
                "shared_adapter_ids": ["qq-a", "wechat-a"],
                "use_persona_filtering": False,
            }
        }
    )
    plugin.config_manager = config
    event = _TestEvent("qq-a:FriendMessage:share", "coffee")
    tool = MemorySearchTool(
        context=plugin.context,
        config_manager=config,
        memory_engine=engine,
        conversation_manager=manager,
    )
    validators = []

    def register_validator(validator, *, tool):
        validators.append(validator)
        return True

    event._cd_tool_memory_sources_register_v1 = register_validator
    previous_plugin = get_active_plugin()
    set_active_plugin(plugin)
    try:
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        output = json.loads(await tool.call(context, query="coffee"))
        assert output["status"] == "ok", output
        assert [result["id"] for result in output["results"]] == ids[:1]
        assert len(validators) == 1
        assert await validators[0]() is True

        original_read = engine.get_summary_source_snapshots
        calls = 0

        async def revoke_during_final_read(self, ids):
            nonlocal calls
            rows = await original_read(ids)
            calls += 1
            if calls == 2:
                config.filtering_settings["enable_group_to_private_recall"] = False
            return rows

        with patch.object(
            type(engine), "get_summary_source_snapshots", revoke_during_final_read
        ):
            assert await validators[0]() is False
        assert calls == 2
        assert await validators[0]() is False
    finally:
        set_active_plugin(previous_plugin)
        await tool.recall_runtime.close()


@pytest.mark.asyncio
async def test_search_tool_validates_active_atom_projection_at_effect_boundary(
    real_db_stack,
):  # noqa: F811
    engine = real_db_stack["memory_engine"]
    plugin = real_db_stack["plugin"]
    session_id = "qq-a:FriendMessage:atoms"
    content = "coffee active fact\ncoffee expired fact"
    memory_id = await engine.add_memory(
        content,
        session_id=session_id,
        persona_id="persona-real",
        atoms=[
            MemoryAtom(
                0,
                content="coffee active fact",
                importance=0.9,
                expires_at=time.time() + 3600,
            ),
            MemoryAtom(
                0,
                content="coffee expired fact",
                importance=0.9,
                expires_at=time.time() + 3600,
            ),
        ],
    )
    event = _TestEvent(session_id, "coffee")
    async with aiosqlite.connect(real_db_stack["memory_db_path"]) as db:
        await db.execute(
            "UPDATE memory_atoms SET expires_at = 0 WHERE parent_memory_id = ? AND content = ?",
            (memory_id, "coffee expired fact"),
        )
        await db.commit()

        tool = MemorySearchTool(
            context=plugin.context,
            config_manager=plugin.config_manager,
            memory_engine=engine,
        )
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        validators = []

        def register_validator(validator, *, tool):
            validators.append(validator)
            return True

        event._cd_tool_memory_sources_register_v1 = register_validator
        previous_plugin = get_active_plugin()
        set_active_plugin(plugin)
        try:
            output = json.loads(await tool.call(context, query="coffee"))
            assert output["status"] == "ok", output
            assert [(row["id"], row["content"]) for row in output["results"]] == [
                (memory_id, "coffee active fact"),
            ]
            assert len(validators) == 1
            assert await validators[0]() is True

            for sql, value, unchanged in [
                (
                    "UPDATE documents SET text = ? WHERE id = ?",
                    "changed parent",
                    content,
                ),
                (
                    "UPDATE memory_atoms SET content = ? WHERE parent_memory_id = ? AND expires_at > 0",
                    "changed fact",
                    "coffee active fact",
                ),
                (
                    "UPDATE memory_atoms SET status = ? WHERE parent_memory_id = ? AND expires_at > 0",
                    "forgotten",
                    "active",
                ),
            ]:
                await db.execute(sql, (value, memory_id))
                await db.commit()
                assert await validators[0]() is False
                await db.execute(sql, (unchanged, memory_id))
                await db.commit()
                assert await validators[0]() is True
        finally:
            set_active_plugin(previous_plugin)
            await tool.recall_runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "card",
        "instance",
        "event-receipt",
        "adapter-during-read",
        "owner",
        "target-allowlist",
    ],
)
async def test_search_tool_revalidates_owned_request_at_effect_boundary(
    real_db_stack, change
):  # noqa: F811
    engine = real_db_stack["memory_engine"]
    plugin = real_db_stack["plugin"]
    umo = "qq-a:FriendMessage:owned"
    event = _TestEvent(umo, "coffee")
    event._chat_dynamics_owned_request = True
    event._cd_owned_original_input_v1 = True
    req = ProviderRequest(prompt="coffee")
    receipt = {
        "schema_version": 1,
        "provider": "astrbot_plugin_chat_dynamics",
        "status": "skipped",
        "reason": "cd_owned_request",
        "purpose": "reply",
        "card_version": "card-1",
        "scope": {
            "umo": umo,
            "platform": "qq-a",
            "persona_id": "worldbook",
            "instance_id": "instance-1",
        },
    }

    async def prepare_memory_request(
        current_event, current_req, materials=None, purpose="reply"
    ):
        assert current_event is event and current_req is req
        assert materials is None and purpose == "reply"
        return {**receipt, "scope": dict(receipt["scope"])}

    plugin.context.stars["astrbot_plugin_chat_dynamics"] = {
        "star_cls": SimpleNamespace(prepare_memory_request=prepare_memory_request),
    }
    assert await plugin._prepare_native_memory_projection(event, req) == receipt
    plugin.config_manager.filtering_settings["shared_adapter_ids"] = [
        "qq-a",
        "wechat-a",
    ]
    await engine.add_memory(
        "coffee preference",
        session_id=umo,
        persona_id="persona-real",
        metadata={"adapter_id": "wechat-a"},
    )
    tool = MemorySearchTool(
        context=plugin.context,
        config_manager=plugin.config_manager,
        memory_engine=engine,
    )
    validators = []

    def register_validator(validator, *, tool):
        validators.append(validator)
        return True

    event._cd_tool_memory_sources_register_v1 = register_validator
    previous_plugin = get_active_plugin()
    set_active_plugin(plugin)
    try:
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        output = json.loads(await tool.call(context, query="coffee"))
        assert output["status"] == "ok", output
        (guard,) = validators
        assert await guard() is True
        if change == "card":
            receipt["card_version"] = "card-2"
        elif change == "instance":
            receipt["scope"]["instance_id"] = "instance-2"
        elif change == "event-receipt":
            event.set_extra(
                "_cd_memory_feedback_v1", {**receipt, "card_version": "card-2"}
            )
        elif change == "owner":
            set_active_plugin(None)
        elif change == "target-allowlist":
            plugin.config_manager.get_section("access_control").update(
                whitelist_enabled=True,
                allowed_ids="some-other-user",
            )
        else:
            original_read = engine.get_summary_source_snapshots
            calls = 0

            async def revoke_during_final_read(self, ids):
                nonlocal calls
                rows = await original_read(ids)
                calls += 1
                if calls == 2:
                    plugin.config_manager.filtering_settings["shared_adapter_ids"] = []
                return rows

            with patch.object(
                type(engine), "get_summary_source_snapshots", revoke_during_final_read
            ):
                assert await guard() is False
            assert calls == 2
            return
        assert await guard() is False
    finally:
        set_active_plugin(previous_plugin)
        await tool.recall_runtime.close()


@pytest.mark.asyncio
async def test_vector_search_preserves_hits_with_adapter_access_and_platform_filters(
    adapter_engine,
):
    """P0 regression: adapter access + platform/group filters must pass storage_filters to retrieve_documents."""
    engine = adapter_engine
    group_scope = "qq:GroupMessage:room-1"
    target_id = await engine.add_memory(
        "coffee",
        session_id=group_scope,
        metadata={"adapter_id": "qq-a"},
    )
    await engine.add_memory(
        "coffee",
        session_id="qq:FriendMessage:user-1",
        metadata={"adapter_id": "qq-a"},
    )
    await engine.add_memory(
        "coffee",
        session_id=group_scope,
        metadata={"adapter_id": "telegram-a"},
    )

    with adapter_access({"filtering_settings": {"shared_adapter_ids": []}}, "qq-a"):
        hits = await engine.vector_retriever.search(
            "coffee",
            5,
            session_id=group_scope,
            interaction_type="group_chat",
            source_platform="qq",
        )
    assert [hit.doc_id for hit in hits] == [target_id]
