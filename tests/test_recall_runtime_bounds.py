from __future__ import annotations

import asyncio
import gc
import weakref
from contextvars import ContextVar

import pytest

from astrbot_plugin_dynamics_memory.core.recall_runtime import (
    BoundedRecallSearchContext,
    BoundedRecallTaskPool,
    RecallDeadline,
    recall_task_key,
)
from astrbot_plugin_dynamics_memory.core.retrieval.bm25_retriever import BM25Result
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.dual_route_retriever import (
    DualRouteRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.rrf_fusion import RRFFusion


@pytest.mark.asyncio
async def test_detached_route_uses_clean_context_and_returns_real_candidate():
    request_context = ContextVar("test_request_context", default=None)
    request_context.set({"must_not_leak": True})
    pool = BoundedRecallTaskPool(max_inflight=16)
    budget = RecallDeadline.start(total_seconds=0.5, final_reserve_seconds=0.1)
    context = BoundedRecallSearchContext(
        pool,
        budget,
        scope={"umo": "synthetic:FriendMessage:user"},
        query="synthetic query",
        purpose="reply",
        card_version="card-a",
        correction_revision=3,
        persona_ids=("synthetic-persona",),
        origin="native",
    )

    async def leaf(query):
        assert request_context.get() is None
        return [query]

    outcome = await context.run("bm25", leaf, "synthetic query")
    assert outcome == {
        "state": "completed",
        "value": ["synthetic query"],
        "cached": False,
    }
    await pool.close()


@pytest.mark.asyncio
async def test_late_route_done_callback_does_not_retain_waiter_contextvars():
    request_context = ContextVar("late_route_request_context", default=None)
    pool = BoundedRecallTaskPool(max_inflight=2)
    context = BoundedRecallSearchContext(
        pool,
        RecallDeadline.start(total_seconds=0.12, final_reserve_seconds=0.03),
        scope={"umo": "synthetic:FriendMessage:user"},
        query="q",
        purpose="reply",
        card_version="card",
        correction_revision=1,
        persona_ids=("persona",),
        origin="test",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    class Sentinel:
        pass

    sentinel = Sentinel()
    reference = weakref.ref(sentinel)
    token = request_context.set(sentinel)

    async def slow_leaf():
        started.set()
        await release.wait()
        return "late"

    outcome = await context.run("slow", slow_leaf)
    assert outcome["state"] == "timeout"
    await started.wait()
    request_context.reset(token)
    del sentinel
    gc.collect()
    assert reference() is None
    release.set()
    await asyncio.sleep(0)
    await pool.close()


@pytest.mark.asyncio
async def test_one_request_never_owns_more_than_four_pending_or_running_leaves():
    pool = BoundedRecallTaskPool(max_inflight=16)
    context = BoundedRecallSearchContext(
        pool,
        RecallDeadline.start(total_seconds=0.8, final_reserve_seconds=0.1),
        scope={"umo": "synthetic:FriendMessage:user"},
        query="q",
        purpose="reply",
        card_version="card",
        correction_revision=1,
        persona_ids=("p",),
        origin="test",
    )
    active = 0
    max_active = 0
    at_capacity = asyncio.Event()
    release = asyncio.Event()

    async def leaf(index):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 4:
            at_capacity.set()
        await release.wait()
        active -= 1
        return index

    waiters = [
        asyncio.create_task(context.run(f"route-{i}", leaf, i)) for i in range(7)
    ]
    await asyncio.wait_for(at_capacity.wait(), timeout=0.4)
    assert max_active == 4
    assert pool.stats()["inflight_count"] == 4
    release.set()
    results = await asyncio.gather(*waiters)
    assert all(result["state"] == "completed" for result in results)
    assert max_active == 4
    await pool.close()


@pytest.mark.asyncio
async def test_completed_bm25_survives_slow_vector_deadline_as_partial_result():
    pool = BoundedRecallTaskPool(max_inflight=16)
    budget = RecallDeadline.start(total_seconds=0.18, final_reserve_seconds=0.03)
    context = BoundedRecallSearchContext(
        pool,
        budget,
        scope={"umo": "synthetic:FriendMessage:user"},
        query="synthetic query",
        purpose="reply",
        card_version="card",
        correction_revision=1,
        persona_ids=("persona",),
        origin="native",
    )
    release = asyncio.Event()
    bm25_result = BM25Result(
        doc_id=17,
        score=0.9,
        content="synthetic local result",
        metadata={"importance": 0.7, "create_time": 1_700_000_000},
    )

    class BM25:
        async def search(self, *_args, **_kwargs):
            return [bm25_result]

    class Vector:
        async def search(self, *_args, **_kwargs):
            await release.wait()
            return []

    retriever = HybridRetriever(
        BM25(), Vector(), RRFFusion(), config={"fallback_enabled": True}
    )
    result = await retriever.search("synthetic query", 3, execution_context=context)
    assert [item.doc_id for item in result] == [17]
    assert context.route_failures.get("document_vector") == "timeout"
    assert len(result) == 1
    release.set()
    await asyncio.sleep(0)
    await pool.close()


@pytest.mark.asyncio
async def test_nested_dual_hybrid_deadlines_preserve_completed_bm25_partial_result():
    for attempt in range(12):
        pool = BoundedRecallTaskPool(max_inflight=16)
        budget = RecallDeadline.start(total_seconds=0.12, final_reserve_seconds=0.025)
        context = BoundedRecallSearchContext(
            pool,
            budget,
            scope={"umo": f"synthetic:FriendMessage:user-{attempt}"},
            query="synthetic query",
            purpose="reply",
            card_version="card",
            correction_revision=1,
            persona_ids=("persona",),
            origin="native",
        )
        release = asyncio.Event()
        candidate = BM25Result(
            doc_id=23,
            score=0.8,
            content="synthetic local hit",
            metadata={"importance": 0.7, "create_time": 1_700_000_000},
        )

        class BM25:
            async def search(self, *_args, **_kwargs):
                return [candidate]

        class Vector:
            async def search(self, *_args, **_kwargs):
                await release.wait()
                return []

        class Graph:
            async def search(self, *_args, **_kwargs):
                return []

        document = HybridRetriever(
            BM25(), Vector(), RRFFusion(), config={"fallback_enabled": True}
        )
        dual = DualRouteRetriever(
            document,
            Graph(),
            memory_loader=lambda _doc_id: asyncio.sleep(0, result=None),
        )
        result = await dual.search("synthetic query", 3, execution_context=context)
        assert [item.doc_id for item in result] == [23]
        assert context.route_failures.get("document_vector") == "timeout"
        release.set()
        await pool.close()


def test_coalescing_key_changes_on_scope_card_or_revision():
    base = {
        "route": "vector",
        "scope": {"umo": "synthetic:FriendMessage:u1"},
        "query": "synthetic query",
        "purpose": "reply",
        "card_version": "c1",
        "correction_revision": 2,
        "persona_ids": ("p1",),
        "origin": "native",
    }
    key = recall_task_key(**base)
    assert (
        recall_task_key(**{**base, "scope": {"umo": "synthetic:FriendMessage:u2"}})
        != key
    )
    assert recall_task_key(**{**base, "card_version": "c2"}) != key
    assert recall_task_key(**{**base, "correction_revision": 3}) != key
