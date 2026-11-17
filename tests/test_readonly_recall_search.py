from __future__ import annotations

import asyncio

import pytest
from types import SimpleNamespace

from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall import (
    MemoryRecall,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.recall_runtime import RecallDeadline


@pytest.mark.asyncio
async def test_native_search_disables_cache_migration_and_access_writes_without_changing_filters():
    engine = object.__new__(MemoryEngine)
    result = SimpleNamespace(
        doc_id=71,
        metadata={
            "interaction_type": "private_chat",
            "source_platform": "synthetic-adapter",
            "source_session_id": "synthetic:FriendMessage:user",
        },
    )
    calls = {}

    class Hybrid:
        async def search(self, query, k, session_id, persona_id, **kwargs):
            calls["route"] = (query, k, session_id, persona_id, kwargs)
            return [result]

    engine._search_cache_key = lambda *_args, **_kwargs: "private-digest"
    engine._get_cached_search_results = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("readonly lookup must skip cache")
    )
    engine._set_cached_search_results = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("readonly lookup must skip cache write")
    )
    engine._create_tracked_task = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("readonly lookup must skip access/migration writes")
    )
    engine.dual_route_retriever = None
    engine.hybrid_retriever = Hybrid()
    engine.atom_retriever = None

    async def keep_atoms(items, *, skip_doc_ids=None):
        return items

    async def merge_recent(items, *_args, **_kwargs):
        return items

    engine._apply_atom_policy = keep_atoms
    engine._filter_by_retrieval_policy = lambda items: items
    engine._merge_recent_memories = merge_recent

    execution_context = object()
    found = await MemoryEngine.search_memories(
        engine,
        query="synthetic original query",
        k=5,
        session_id="synthetic:FriendMessage:user",
        persona_id="synthetic-persona",
        interaction_type="private_chat",
        source_platform="synthetic-adapter",
        source_session_id="synthetic:FriendMessage:user",
        touch_access=False,
        cache_results=False,
        allow_legacy_session_migration=False,
        execution_context=execution_context,
    )

    assert found == [result]
    query, k, session_id, persona_id, kwargs = calls["route"]
    assert (query, k, session_id, persona_id) == (
        "synthetic original query",
        5,
        "synthetic:FriendMessage:user",
        "synthetic-persona",
    )
    assert kwargs["execution_context"] is execution_context
    assert kwargs["interaction_type"] == "private_chat"
    assert kwargs["source_platform"] == "synthetic-adapter"
    assert kwargs["source_session_id"] == "synthetic:FriendMessage:user"


@pytest.mark.asyncio
async def test_readonly_engine_postprocessing_is_bounded_by_shared_deadline():
    """The outer read-only engine call also bounds parent hydration/filter work."""
    engine = object.__new__(MemoryEngine)
    result = SimpleNamespace(
        doc_id=72,
        metadata={
            "interaction_type": "private_chat",
            "source_platform": "synthetic-adapter",
            "source_session_id": "synthetic:FriendMessage:user",
        },
    )
    merge_started = asyncio.Event()
    merge_cancelled = asyncio.Event()

    class Hybrid:
        async def search(self, *_args, **_kwargs):
            return [result]

    engine._search_cache_key = lambda *_args, **_kwargs: "deadline-test"
    engine._get_cached_search_results = lambda *_args, **_kwargs: None
    engine._set_cached_search_results = lambda *_args, **_kwargs: None
    engine._create_tracked_task = lambda *_args, **_kwargs: None
    engine.dual_route_retriever = None
    engine.hybrid_retriever = Hybrid()
    engine.atom_retriever = None

    async def keep(items, *, skip_doc_ids=None):
        return items

    async def slow_parent_hydration(items, *_args, **_kwargs):
        merge_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            merge_cancelled.set()
            raise

    engine._apply_atom_policy = keep
    engine._filter_by_retrieval_policy = lambda items: items
    engine._merge_recent_memories = slow_parent_hydration

    deadline = RecallDeadline.start(total_seconds=0.12, final_reserve_seconds=0)
    recall = object.__new__(MemoryRecall)
    recall.memory_engine = engine
    recall.recall_runtime = None
    recall._scope_tasks = set()
    recall._scope_revision = lambda _identity: 0
    scope = SimpleNamespace(
        source_platform="synthetic-adapter",
        source_session_id="synthetic:FriendMessage:user",
    )
    jobs = [
        (scope, "synthetic:FriendMessage:user", "synthetic-persona", "private_chat")
    ]
    started = asyncio.get_running_loop().time()
    outcomes = await recall._run_scope_searches(
        jobs,
        deadline,
        "synthetic original query",
        5,
        None,
        SimpleNamespace(),
        "synthetic:FriendMessage:user",
        "synthetic-persona",
    )
    await asyncio.sleep(0)
    elapsed = asyncio.get_running_loop().time() - started

    assert outcomes == [{"state": "timeout", "value": []}]
    assert merge_started.is_set()
    assert merge_cancelled.is_set()
    assert elapsed < 0.25
