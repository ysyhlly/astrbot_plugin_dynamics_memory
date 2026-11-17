"""Native recall keeps validated candidates within the reply budget."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.memory_scope import current_adapter_access
from astrbot_plugin_dynamics_memory.core.recall_runtime import (
    BoundedRecallTaskPool,
    RecallDeadline,
)
from astrbot_plugin_dynamics_memory.core.retrieval.bm25_retriever import BM25Result
from astrbot_plugin_dynamics_memory.core.retrieval.dual_route_retriever import (
    DualRouteRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.graph_retriever import GraphRetriever
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.rrf_fusion import RRFFusion
from astrbot_plugin_dynamics_memory.core.retrieval.vector_search import (
    QueryEmbeddingCache,
)

from .test_private_summary_projection import _PERSONA, _SESSION, _recall


@pytest.mark.asyncio
async def test_slow_vector_and_optional_recent_preserve_live_atoms(
    private_hook, source_db
):
    recall = private_hook.handler._memory_recall
    recall.recall_runtime = BoundedRecallTaskPool()
    engine = source_db.engine
    del engine.search_memories  # Exercise real fusion, atom and recent paths.
    engine.config = {"recent_memory_count": 2}
    engine._search_cache_generation = 0
    engine.atom_retriever = None
    parent = source_db.candidates[0]
    cleanup = asyncio.Event()
    vector_cancelled = asyncio.Event()

    async def delayed_vector(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            vector_cancelled.set()
            await cleanup.wait()
        return [BM25Result(2, 1.0, "Late remote result", parent.metadata)]

    async def slow_recent(*_args, **_kwargs):
        await asyncio.Event().wait()

    engine._get_recent_memory_results = slow_recent
    hybrid = HybridRetriever(
        SimpleNamespace(
            search=AsyncMock(
                return_value=[
                    BM25Result(parent.doc_id, 1.0, parent.content, parent.metadata),
                ]
            )
        ),
        SimpleNamespace(search=delayed_vector),
        RRFFusion(),
    )
    graph = GraphRetriever(
        SimpleNamespace(search=AsyncMock(return_value=[])),
        SimpleNamespace(search=delayed_vector),
        RRFFusion(),
    )
    engine.dual_route_retriever = DualRouteRetriever(hybrid, graph, engine.get_memory)
    private_hook.plugin.initializer.memory_engine = engine
    recall.memory_engine = engine
    deadline = RecallDeadline.start(0.85, 0.35)
    try:
        await private_hook.handler.handle_memory_recall(
            private_hook.event, private_hook.req, budget=deadline
        )
        req = private_hook.req
        assert time.monotonic() < deadline.deadline
        assert req._lm_private_hook_outcome_v1 == "partial"
        assert len(req.extra_user_content_parts) == 1
        assert "LIVE_FACT SECOND_LIVE_FACT" in req.extra_user_content_parts[0].text
        assert "REMOVED_FACT" not in req.extra_user_content_parts[0].text
        assert await req._lm_private_summary_validate_v1() is True
        assert vector_cancelled.is_set()
        assert recall.recall_runtime.stats()["inflight_count"] >= 1
        cleanup.set()
    finally:
        cleanup.set()
        await recall.recall_runtime.close()
    assert len(req.extra_user_content_parts) == 1
    assert "Late remote result" not in req.extra_user_content_parts[0].text


@pytest.mark.asyncio
async def test_slow_shared_scope_keeps_completed_self_scope_and_reserves_validation(
    private_hook, source_db
):
    recall = private_hook.handler._memory_recall
    recall.recall_runtime = BoundedRecallTaskPool()
    engine = source_db.engine
    original_search = engine.search_memories
    groups = tuple(f"test:GroupMessage:{index}" for index in range(1, 6))
    filtering = recall.config_manager._config["filtering_settings"]
    filtering.update(
        {
            "enable_group_to_private_recall": True,
            "private_shared_group_sources": list(groups),
            "private_shared_group_instance_id": _PERSONA,
            "shared_adapter_ids": ["test"],
        }
    )
    recall._verified_shared_group_sessions = AsyncMock(return_value=groups)
    cancel_started = asyncio.Event()
    cleanup = asyncio.Event()
    slow_returned = asyncio.Event()
    visited = set()

    async def scoped_search(**kwargs):
        access = current_adapter_access.get()
        assert access is not None and "test" in access.adapter_ids
        source = kwargs["source_session_id"]
        visited.add(source)
        if not source:
            assert kwargs["session_id"] == _SESSION
            return await original_search(**kwargs)
        assert kwargs["interaction_type"] == "group_chat"
        assert kwargs["persona_id"] == _PERSONA
        assert kwargs["source_platform"] == "test"
        assert kwargs["touch_access"] is False
        assert kwargs["cache_results"] is False
        if source == groups[0]:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_started.set()
                await cleanup.wait()
            slow_returned.set()
            return [source_db.candidates[0]]
        return []

    engine.search_memories = scoped_search
    deadline = RecallDeadline.start(1.2, 0.3)
    install_guard = recall.summary_guard

    async def slow_guard(*args, **kwargs):
        await asyncio.sleep(0.10)
        return await install_guard(*args, **kwargs)

    recall.summary_guard = slow_guard
    try:
        await _recall(private_hook, source_db, budget=deadline)
        req = private_hook.req
        assert time.monotonic() < deadline.deadline
        assert deadline.search_remaining() == 0
        assert req._lm_private_hook_outcome_v1 == "partial"
        assert len(req.extra_user_content_parts) == 1
        assert "REMOVED_FACT" not in req.extra_user_content_parts[0].text
        assert await req._lm_private_summary_validate_v1() is True
        assert visited == {None, *groups}
        assert cancel_started.is_set()
        assert recall.recall_runtime.stats()["scope_inflight_count"] == 1
        sources_before = req._lm_private_summary_sources_v1
        cleanup.set()
        await asyncio.wait_for(slow_returned.wait(), timeout=1)
        assert len(req.extra_user_content_parts) == 1
        assert req._lm_private_summary_sources_v1 is sources_before
    finally:
        cleanup.set()
        await recall.recall_runtime.close()
    assert recall.recall_runtime.stats()["scope_inflight_count"] == 0


@pytest.mark.asyncio
async def test_shared_embedding_cancellation_keeps_pool_capacity_until_cleanup():
    pool = BoundedRecallTaskPool(max_inflight=1)
    loop = asyncio.get_running_loop()
    cache = QueryEmbeddingCache(loop.time() + 0.2, owner=pool)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def embed(_query):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return [1.0]

    provider = SimpleNamespace(get_embedding=AsyncMock(side_effect=embed))
    task = cache.start(provider, "coffee")
    waiter = asyncio.shield(task)
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not task.done()
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        assert pool.submit("other", embed, cache_result=False)["state"] == "busy"
        release.set()
        with pytest.raises(TimeoutError):
            await task
        assert pool.stats()["inflight_count"] == 0
        provider.get_embedding.assert_awaited_once_with("coffee")
    finally:
        release.set()
        await pool.close()
