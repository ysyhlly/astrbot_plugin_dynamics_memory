import asyncio
from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin
from astrbot_plugin_dynamics_memory.core.recall_runtime import (
    BoundedRecallTaskPool,
    BoundedRecallSearchContext,
    RecallDeadline,
)
from astrbot_plugin_dynamics_memory.core.retrieval.vector_search import (
    QueryEmbeddingCache,
    retrieve_vectors,
    query_embeddings,
)
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from .test_ready_recall_delivery import (
    plugin_with_ready_baseline,
    close_pool,
    ready_view,
    seeded_evidence,
    SCOPE,
)
from .test_narrative_api import envelope


@pytest.mark.asyncio
async def test_empty_successful_enrichment_keeps_verified_baseline():
    plugin, original = plugin_with_ready_baseline()
    empty = deepcopy(original)
    empty.update(atoms=[], slots={}, evidence_status="pending")
    plugin.get_evidence_view = AsyncMock(return_value=empty)
    try:
        result = await plugin.recall_context(SCOPE, "coffee", "reply", "card")
        assert result["evidence_view"]["atoms"] == original["atoms"]
        assert result["diagnostics"]["baseline_preserved"]
    finally:
        await close_pool(plugin)


@pytest.mark.asyncio
async def test_empty_enrichment_cannot_restore_revoked_baseline():
    plugin, original = plugin_with_ready_baseline()
    empty = dict(original, atoms=[], slots={}, evidence_status="pending")

    async def enriched(*args, **kwargs):
        plugin.peek_evidence_revision = lambda scope: {
            "available": True,
            "scope_key": scope_key(scope),
            "correction_revision": 1,
            "revocations": [],
        }
        return empty

    plugin.get_evidence_view = AsyncMock(side_effect=enriched)
    try:
        result = await plugin.recall_context(SCOPE, "coffee", "reply", "card")
        assert not result["items"]
        assert not result["diagnostics"]["baseline_preserved"]
    finally:
        await close_pool(plugin)


@pytest.mark.asyncio
async def test_empty_enrichment_cannot_restore_newly_blocked_member():
    plugin, original = plugin_with_ready_baseline()

    async def allowed(scope, actor_id=""):
        return actor_id != "u1"

    plugin._narrative_scope_allowed = AsyncMock(side_effect=allowed)
    plugin.get_evidence_view = AsyncMock(
        return_value=dict(original, atoms=[], slots={})
    )
    try:
        result = await plugin.recall_context(SCOPE, "coffee", "reply", "card")
        assert not result["items"]
        assert not result["diagnostics"]["baseline_preserved"]
    finally:
        await close_pool(plugin)


@pytest.mark.asyncio
async def test_ready_source_proof_precedes_deduplicated_material_expansion():
    plugin, _ = plugin_with_ready_baseline()
    evidence = [
        {
            "source_id": str(i),
            "text": "用户发言：extra",
            "scope": SCOPE,
            "fact_kind": "observation",
        }
        for i in range(90)
    ]
    evidence.insert(0, dict(evidence[0]))
    plugin.initializer.conversation_manager.store.recall_narrative_events = AsyncMock(
        return_value=evidence
    )

    async def view(*args, **kwargs):
        refs = kwargs["source_refs"]
        assert len(refs) == 64
        assert refs[0]["source_event_id"] == "message:old"
        assert len({r.get("event_id", r.get("source_event_id")) for r in refs}) == 64
        return ready_view()

    plugin.get_evidence_view = AsyncMock(side_effect=view)
    try:
        result = await plugin.recall_context(SCOPE, "coffee", "reply", "card")
        assert result["items"]
        plugin.get_evidence_view.assert_awaited_once()
    finally:
        await close_pool(plugin)


@pytest.mark.asyncio
async def test_display_alias_does_not_change_event_identity(tmp_path):
    plugin, service, sidecar, archive, _ = await seeded_evidence(tmp_path)
    try:
        baseline = await plugin._ready_narrative_baseline(
            SCOPE, "coffee", "reply", "card", 0
        )
        service.event_loader = AsyncMock(return_value=deepcopy(archive))
        view = await service.get_evidence_view(
            dict(SCOPE, persona_aliases=["display"]),
            typed_items=baseline["typed_items"],
            card_version="card",
            allow_derivation=False,
        )
        assert len(view["atoms"]) == 1
        service.event_loader.return_value["scope"] = dict(
            SCOPE, umo="qq:GroupMessage:other"
        )
        view = await service.get_evidence_view(
            dict(SCOPE, persona_aliases=["display"]),
            typed_items=baseline["typed_items"],
            card_version="card",
            allow_derivation=False,
        )
        assert not view["atoms"]
    finally:
        await sidecar.close()


def context(pool, deadline, cache=None, origin="room", parallel=None):
    return BoundedRecallSearchContext(
        pool,
        RecallDeadline(asyncio.get_running_loop().time(), deadline, 0),
        scope=SCOPE,
        query="coffee",
        purpose="reply",
        card_version="card",
        correction_revision=0,
        persona_ids=("anan",),
        origin=origin,
        hard_deadline=True,
        embedding_cache=cache,
        _parallel=parallel,
    )


@pytest.mark.asyncio
async def test_budget_cancels_provider_leaf_and_starts_no_late_work():
    pool = BoundedRecallTaskPool()
    cancelled = asyncio.Event()

    async def slow():
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.set()

    ctx = context(pool, asyncio.get_running_loop().time() + 0.08)
    try:
        result = await ctx.run("slow", slow)
        assert result["state"] == "timeout"
        await asyncio.wait_for(cancelled.wait(), 0.5)
        late = AsyncMock(return_value=["unexpected"])
        assert (await ctx.run("late", late))["state"] == "timeout"
        late.assert_not_awaited()
        await asyncio.sleep(0.01)
        assert pool.stats()["inflight_count"] == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pure_cache_is_carried_without_request_context():
    pool = BoundedRecallTaskPool()
    marker = ContextVar("test_request", default=None)
    token = marker.set("private request")
    deadline = asyncio.get_running_loop().time() + 1
    cache = QueryEmbeddingCache(deadline)

    async def read():
        assert marker.get() is None
        assert query_embeddings.get() is cache
        return ["ok"]

    try:
        assert (await context(pool, deadline, cache).run("leaf", read))["value"] == [
            "ok"
        ]
    finally:
        marker.reset(token)
        await pool.close()


def vector_db(provider, origin, doc_id):
    document = {
        "id": doc_id,
        "text": "scoped fact",
        "metadata": {"session_id": origin, "status": "active"},
    }
    return NS(
        embedding_provider=provider,
        embedding_storage=NS(
            index=NS(ntotal=1),
            dimension=2,
            search=AsyncMock(return_value=([[1.0]], [[doc_id]])),
        ),
        document_storage=NS(get_documents=AsyncMock(return_value=[document])),
    )


@pytest.mark.asyncio
async def test_two_scopes_share_embedding_but_not_documents():
    pool = BoundedRecallTaskPool()
    provider = NS(get_embedding=AsyncMock(return_value=[1.0, 0.0]))
    deadline = asyncio.get_running_loop().time() + 1
    cache, parallel = QueryEmbeddingCache(deadline), asyncio.Semaphore(4)

    async def search(db, origin):
        return await retrieve_vectors(db, "coffee", 1, {"session_id": origin})

    try:
        results = await asyncio.gather(
            *[
                context(pool, deadline, cache, origin, parallel).run(
                    "vector", search, vector_db(provider, origin, i), origin
                )
                for i, origin in [(1, "legacy"), (2, "managed")]
            ]
        )
        assert provider.get_embedding.await_count == 1
        assert [r["value"][0].data["id"] for r in results] == [1, 2]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_shielded_embedding_also_obeys_owner_deadline():
    pool = BoundedRecallTaskPool()
    cancelled = asyncio.Event()

    async def embed(query):
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.set()

    deadline = asyncio.get_running_loop().time() + 0.08
    cache = QueryEmbeddingCache(deadline)
    try:
        await context(pool, deadline, cache).run(
            "vector",
            retrieve_vectors,
            vector_db(NS(get_embedding=embed), "room", 1),
            "coffee",
            1,
            {"session_id": "room"},
        )
        await asyncio.wait_for(cancelled.wait(), 0.5)
        assert all(task.done() for task in cache.values())
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_batch_across_summaries_preserves_order_and_rechecks_sources(tmp_path):
    store = ConversationStore(str(tmp_path / "events.db"))
    await store.initialize()
    try:
        for i in range(205):
            await store.ingest_narrative_event(
                envelope(str(i), text=str(i), scope=SCOPE)
            )

        def metadata(ids):
            return {
                "source_window": {"session_id": scope_key(SCOPE)},
                "narrative_provenance": [
                    {"source_id": str(i), "scope": SCOPE, "fact_kind": "observation"}
                    for i in ids
                ],
            }

        queries = []
        await store.connection.set_trace_callback(queries.append)
        async with store._read_session() as read_conn:
            if read_conn is not store.connection:
                await read_conn.set_trace_callback(queries.append)
        plans = [metadata(range(205)), metadata(reversed(range(205)))]
        result = await store.load_narrative_sources_many(plans)
        assert (
            len(
                [
                    q
                    for q in queries
                    if q.startswith("SELECT event_key,session_id,envelope")
                ]
            )
            == 2
        )
        assert [r["content"] for r in result[0]] == list(map(str, range(205)))
        assert result[1][0]["content"] == "204"
        await store.connection.execute(
            "DELETE FROM narrative_events WHERE event_key=?", (scope_key(SCOPE) + ":0",)
        )
        await store.connection.commit()
        assert await store.load_narrative_sources_many(plans) == [None, None]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_source_resolver_batches_and_still_checks_claimed_author(tmp_path):
    store = ConversationStore(str(tmp_path / "events.db"))
    await store.initialize()
    try:
        await store.ingest_narrative_event(envelope("a", scope=SCOPE))
        plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
        plugin.initializer = NS(conversation_manager=NS(store=store))
        queries = []
        await store.connection.set_trace_callback(queries.append)
        async with store._read_session() as read_conn:
            if read_conn is not store.connection:
                await read_conn.set_trace_callback(queries.append)
        refs = [
            {"source_kind": "event", "source_event_id": "a"},
            {"source_kind": "event", "source_event_id": "a", "actor_id": "forged"},
        ]
        rows, rejected = await plugin._resolve_evidence_source_refs(SCOPE, refs)
        assert len(rows) == 1 and rejected == 1
        assert (
            len(
                [
                    q
                    for q in queries
                    if q.startswith("SELECT event_key,session_id,envelope")
                ]
            )
            == 1
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_recall_context_binds_both_semantic_scopes_to_one_query_budget():
    plugin, _ = plugin_with_ready_baseline(mixed=False)
    seen = []

    async def semantic(query, k, origin, personas, *, execution_context=None):
        seen.append(execution_context)
        assert execution_context.hard_deadline
        assert execution_context.origin == origin
        return []

    plugin.initializer.memory_engine.search_narrative_semantic = semantic
    try:
        result = await plugin.recall_context(SCOPE, "coffee", "reply", "card")
        assert result["items"]
        assert len(seen) == 2
        assert seen[0].budget.deadline == seen[1].budget.deadline
        assert seen[0].embedding_cache is seen[1].embedding_cache
    finally:
        await close_pool(plugin)


@pytest.mark.asyncio
async def test_semantic_engine_forwards_budget_and_keeps_atom_policy():
    from astrbot_plugin_dynamics_memory.core.managers.memory_engine_crud import (
        MemoryEngineCrudMixin,
    )
    from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
        HybridResult,
    )

    row = HybridResult(
        1, 1.0, 0, 1.0, None, "fact", {"session_id": "room", "persona_id": "anan"}, {}
    )
    retriever = NS(search=AsyncMock(return_value=[row]))

    async def run(route, method, *args, **kwargs):
        assert route == "semantic_atom_policy"
        return {"state": "completed", "value": await method(*args)}

    ctx = NS(run=run)
    engine = NS(
        dual_route_retriever=retriever,
        hybrid_retriever=None,
        _filter_by_retrieval_policy=lambda rows: rows,
        _apply_atom_policy=AsyncMock(return_value=[]),
    )
    assert (
        await MemoryEngineCrudMixin.search_narrative_semantic(
            engine, "coffee", 5, "room", ("anan",), execution_context=ctx
        )
        == []
    )
    assert retriever.search.call_args.kwargs["execution_context"] is ctx
    engine._apply_atom_policy.assert_awaited_once()
