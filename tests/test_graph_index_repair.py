"""Offline safety tests for graph-index repair planning and application."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import aiosqlite

from astrbot_plugin_dynamics_memory.core.maintenance.graph_index_repair import (
    GraphIndexDocument,
    PersistedGraphVector,
    RepairClass,
    apply_cached_graph_vector_repair,
    apply_stale_graph_vector_prune,
    content_digest,
    plan_graph_index_repair,
)
from astrbot_plugin_dynamics_memory.core.managers.graph_memory_manager import (
    GraphMemoryManager,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.retrieval.graph_retriever import GraphRetriever
from astrbot_plugin_dynamics_memory.core.retrieval.graph_vector_retriever import (
    GraphVectorRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.rrf_fusion import RRFFusion
from astrbot_plugin_dynamics_memory.core.maintenance.source_state_guard import (
    SourceStateGuard,
)


MODEL = "provider/model-v1"
DIMENSION = 3


def _document(doc_id: int = 10, *, active: bool = True) -> GraphIndexDocument:
    return GraphIndexDocument(
        document_id=doc_id,
        source_memory_id=100 + doc_id,
        source_is_active=active,
        content_sha256=content_digest(f"private content {doc_id}"),
    )


def _vector(
    document: GraphIndexDocument, values=(0.1, 0.2, 0.3)
) -> PersistedGraphVector:
    return PersistedGraphVector(
        document_id=document.document_id,
        source_memory_id=document.source_memory_id,
        content_sha256=document.content_sha256,
        model_id=MODEL,
        values=tuple(values),
    )


class _MemoryIndex:
    def __init__(self):
        self.vectors: dict[int, tuple[float, ...]] = {}
        self.bindings: dict[int, tuple[int, str, bool]] = {}
        self.persist_count = 0
        self.fail_persist = False

    def snapshot(self):
        return dict(self.vectors), dict(self.bindings)

    def add_with_ids(self, vectors):
        self.vectors.update({key: tuple(value) for key, value in vectors.items()})

    def restore(self, snapshot):
        vectors, bindings = snapshot
        self.vectors = dict(vectors)
        self.bindings = dict(bindings)

    def persist_atomic(self):
        if self.fail_persist:
            raise OSError("simulated atomic persistence failure")
        self.persist_count += 1

    def existing_vectors(self, document_ids):
        return {key: self.vectors[key] for key in document_ids if key in self.vectors}

    def existing_document_bindings(self, document_ids):
        return {key: self.bindings[key] for key in document_ids if key in self.bindings}

    def delete_ids(self, document_ids):
        for key in document_ids:
            self.vectors.pop(key, None)
            self.bindings.pop(key, None)


def _plan(document, vector, **kwargs):
    return plan_graph_index_repair(
        [document],
        indexed_ids=set(),
        persisted_vectors={document.document_id: vector} if vector else {},
        expected_model_id=MODEL,
        index_dimension=DIMENSION,
        **kwargs,
    )


def test_compatible_persisted_vector_repairs_once_and_repeat_is_noop():
    document = _document()
    vector = _vector(document)
    plan = _plan(document, vector)
    index = _MemoryIndex()

    assert plan.apply_ids == (document.document_id,)
    assert apply_cached_graph_vector_repair(
        plan, {document.document_id: vector}, index
    ) == (10,)
    assert index.vectors == {10: vector.values}
    assert index.persist_count == 1

    assert apply_cached_graph_vector_repair(
        plan, {document.document_id: vector}, index
    ) == (10,)
    assert index.vectors == {10: vector.values}
    assert index.persist_count == 1


@pytest.mark.parametrize(
    ("vector", "classification"),
    [
        (None, RepairClass.REQUIRES_EMBEDDING),
        (_vector(_document(), (0.1, 0.2)), RepairClass.CACHE_DIMENSION_MISMATCH),
        (
            PersistedGraphVector(
                10, 110, content_digest("different"), MODEL, (0.1, 0.2, 0.3)
            ),
            RepairClass.CACHE_IDENTITY_MISMATCH,
        ),
        (
            PersistedGraphVector(
                10,
                110,
                content_digest("private content 10"),
                "other/model",
                (0.1, 0.2, 0.3),
            ),
            RepairClass.CACHE_MODEL_MISMATCH,
        ),
    ],
)
def test_non_compatible_sources_are_not_approved(vector, classification):
    document = _document()
    plan = _plan(document, vector)
    assert dict(plan.classifications)[document.document_id] is classification
    assert plan.apply_ids == ()


def test_archived_unindexed_source_is_never_repaired_even_with_cached_vector():
    document = _document(active=False)
    plan = _plan(document, _vector(document))
    assert (
        dict(plan.classifications)[document.document_id]
        is RepairClass.ARCHIVED_UNINDEXED_SOURCE
    )
    assert plan.apply_ids == ()
    assert plan.stale_index_ids == ()


def test_live_audit_counts_plan_as_1616_active_88_stale_215_archived_unindexed():
    documents = [
        GraphIndexDocument(
            document_id=doc_id,
            source_memory_id=doc_id + 10000,
            source_is_active=doc_id <= 1616,
            content_sha256=content_digest(f"fixture {doc_id}"),
        )
        for doc_id in range(1, 1920)
    ]
    indexed_ids = set(range(1, 1705))
    plan = plan_graph_index_repair(
        documents,
        indexed_ids=indexed_ids,
        persisted_vectors={},
        expected_model_id=None,
        index_dimension=1024,
    )
    counts = plan.counts()
    assert counts[RepairClass.ALREADY_INDEXED.value] == 1616
    assert counts[RepairClass.STALE_INDEXED_SOURCE.value] == 88
    assert counts[RepairClass.ARCHIVED_UNINDEXED_SOURCE.value] == 215
    assert plan.missing_active_ids == ()
    assert plan.requires_embedding_ids == ()
    assert plan.stale_index_ids == tuple(range(1617, 1705))


def test_archived_indexed_source_is_planned_for_vector_only_prune_and_repeat_is_noop():
    document = _document(active=False)
    plan = plan_graph_index_repair(
        [document],
        indexed_ids={document.document_id},
        persisted_vectors={},
        expected_model_id=MODEL,
        index_dimension=DIMENSION,
    )
    index = _MemoryIndex()
    index.vectors[10] = (0.1, 0.2, 0.3)
    index.bindings[10] = (document.source_memory_id, document.content_sha256, False)

    assert dict(plan.classifications)[10] is RepairClass.STALE_INDEXED_SOURCE
    assert plan.stale_index_ids == (10,)
    assert plan.missing_active_ids == ()
    assert apply_stale_graph_vector_prune(plan, index) == (10,)
    assert index.vectors == {}
    assert index.persist_count == 1
    assert apply_stale_graph_vector_prune(plan, index) == (10,)
    assert index.persist_count == 1


def test_stale_prune_rechecks_archive_state_and_rolls_back_on_persist_failure():
    document = _document(active=False)
    plan = plan_graph_index_repair(
        [document],
        indexed_ids={10},
        persisted_vectors={},
        expected_model_id=MODEL,
        index_dimension=DIMENSION,
    )
    index = _MemoryIndex()
    index.vectors[10] = (0.1, 0.2, 0.3)
    index.bindings[10] = (document.source_memory_id, document.content_sha256, True)
    with pytest.raises(ValueError, match="binding changed"):
        apply_stale_graph_vector_prune(plan, index)
    assert index.vectors[10] == (0.1, 0.2, 0.3)

    index.bindings[10] = (document.source_memory_id, document.content_sha256, False)
    index.fail_persist = True
    with pytest.raises(OSError, match="simulated"):
        apply_stale_graph_vector_prune(plan, index)
    assert index.vectors[10] == (0.1, 0.2, 0.3)
    assert index.bindings[10] == (
        document.source_memory_id,
        document.content_sha256,
        False,
    )


def test_requested_scope_cannot_expand_past_inventory():
    with pytest.raises(ValueError, match="outside inventory"):
        plan_graph_index_repair(
            [_document()],
            indexed_ids=set(),
            persisted_vectors={},
            expected_model_id=MODEL,
            index_dimension=DIMENSION,
            requested_ids=[10, 11],
        )


def test_apply_rejects_extra_ids_or_changed_source_binding():
    document = _document()
    vector = _vector(document)
    plan = _plan(document, vector)
    index = _MemoryIndex()
    with pytest.raises(ValueError, match="exactly match"):
        apply_cached_graph_vector_repair(
            plan,
            {10: vector, 11: _vector(_document(11))},
            index,
        )

    changed = PersistedGraphVector(10, 999, vector.content_sha256, MODEL, vector.values)
    with pytest.raises(ValueError, match="source binding changed"):
        apply_cached_graph_vector_repair(plan, {10: changed}, index)
    assert index.vectors == {}
    assert index.persist_count == 0


def test_persistence_failure_rolls_back_in_memory_index():
    document = _document()
    vector = _vector(document)
    plan = _plan(document, vector)
    index = _MemoryIndex()
    index.vectors[99] = (9.0, 9.0, 9.0)
    index.fail_persist = True

    with pytest.raises(OSError, match="simulated"):
        apply_cached_graph_vector_repair(plan, {10: vector}, index)

    assert index.vectors == {99: (9.0, 9.0, 9.0)}


def test_cancellation_during_persistence_rolls_back_in_memory_index():
    document = _document()
    vector = _vector(document)
    plan = _plan(document, vector)
    index = _MemoryIndex()
    index.fail_persist = True

    def cancel_on_persist():
        raise asyncio.CancelledError

    index.persist_atomic = cancel_on_persist
    with pytest.raises(asyncio.CancelledError):
        apply_cached_graph_vector_repair(plan, {10: vector}, index)
    assert index.vectors == {}


def test_existing_identical_vector_is_idempotent_but_conflict_is_rejected():
    document = _document()
    vector = _vector(document)
    plan = _plan(document, vector)
    index = _MemoryIndex()
    index.vectors[10] = vector.values
    assert apply_cached_graph_vector_repair(plan, {10: vector}, index) == (10,)
    assert index.persist_count == 0

    index.vectors[10] = (8.0, 8.0, 8.0)
    with pytest.raises(ValueError, match="different vector"):
        apply_cached_graph_vector_repair(plan, {10: vector}, index)


@pytest.mark.asyncio
async def test_query_rechecks_source_state_and_excludes_newly_archived_source():
    class _Route:
        def __init__(self, ids):
            self.ids = ids

        async def search(self, *args, **kwargs):
            return [
                SimpleNamespace(
                    doc_id=doc_id,
                    score=1.0,
                    content=f"candidate-{doc_id}",
                    metadata={"importance": 0.8, "graph_confidence": 0.9},
                )
                for doc_id in self.ids
            ]

    state = {10, 11}
    calls = []

    async def active_sources(source_ids):
        calls.append(tuple(source_ids))
        return set(state) & set(source_ids)

    retriever = GraphRetriever(
        _Route([10, 11]),
        _Route([10, 11]),
        RRFFusion(60),
        {},
        source_validity_checker=active_sources,
    )
    before_archive = await retriever.search("query", k=5)
    assert {result.doc_id for result in before_archive} == {10, 11}

    state.remove(10)
    after_archive = await retriever.search("query", k=5)
    assert [result.doc_id for result in after_archive] == [11]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_normal_query_finishes_while_write_side_restore_guard_is_held():
    guard = SourceStateGuard()
    state = {10}
    restore_entered = asyncio.Event()
    allow_restore_commit = asyncio.Event()

    async def hold_restore_guard():
        async with guard.hold():
            restore_entered.set()
            await allow_restore_commit.wait()

    async def active_sources(source_ids):
        return state & set(source_ids)

    retriever = GraphRetriever(
        _RouteForQueryTest([10]),
        _RouteForQueryTest([10]),
        RRFFusion(60),
        {},
        source_validity_checker=active_sources,
    )
    restore = asyncio.create_task(hold_restore_guard())
    await restore_entered.wait()
    try:
        results = await asyncio.wait_for(retriever.search("query", k=5), timeout=0.2)
        assert [result.doc_id for result in results] == [10]
    finally:
        allow_restore_commit.set()
        await restore


@pytest.mark.asyncio
async def test_query_fails_closed_when_source_state_cannot_be_verified():
    async def unavailable(_source_ids):
        raise RuntimeError("database unavailable")

    retriever = GraphRetriever(
        _RouteForQueryTest([10]),
        _RouteForQueryTest([10]),
        RRFFusion(60),
        {},
        source_validity_checker=unavailable,
    )
    assert await retriever.search("query", k=5) == []


class _RouteForQueryTest:
    def __init__(self, ids):
        self.ids = ids

    async def search(self, *args, **kwargs):
        return [
            SimpleNamespace(
                doc_id=doc_id,
                score=1.0,
                content=f"candidate-{doc_id}",
                metadata={"importance": 0.8, "graph_confidence": 0.9},
            )
            for doc_id in self.ids
        ]


class _FakeIndex:
    def __init__(self, ids):
        self.ids = list(ids)
        self.id_map = list(ids)

    def remove_ids(self, ids):
        removed = set(int(value) for value in ids)
        self.ids = [value for value in self.ids if value not in removed]
        self.id_map = list(self.ids)


class _FakeEmbeddingStorage:
    def __init__(self, ids, persister):
        self.index = _FakeIndex(ids)
        self.path = "graph.index"
        self._async_index_persister = persister


@pytest.mark.asyncio
async def test_vector_only_prune_preserves_docs_and_is_idempotent(monkeypatch):
    import sys

    import astrbot_plugin_dynamics_memory.core.faiss_async_persist as persistence

    class _Persister:
        def __init__(self):
            self._mutation_lock = asyncio.Lock()
            self.flush_now = AsyncMock()

    class _DocumentStorage:
        def __init__(self):
            self.rows = {1, 2}

    persister = _Persister()
    storage = _FakeEmbeddingStorage([1, 2], persister)
    document_storage = _DocumentStorage()
    fake_db = SimpleNamespace(
        embedding_storage=storage,
        document_storage=document_storage,
    )
    fake_faiss = SimpleNamespace(
        downcast_index=lambda index: index,
        vector_to_array=lambda id_map: id_map,
        clone_index=lambda index: _FakeIndex(index.ids),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda ids, dtype=None: ids, int64=object()),
    )
    persisted = []
    monkeypatch.setattr(
        persistence,
        "_write_index_atomic",
        lambda index, path: persisted.append(tuple(index.ids)),
    )

    retriever = GraphVectorRetriever(fake_db)
    assert await retriever.delete_vector_ids_atomically([2]) == (2,)
    assert storage.index.ids == [1]
    assert document_storage.rows == {1, 2}
    assert persisted == [(1,)]
    assert await retriever.delete_vector_ids_atomically([2]) == ()
    assert storage.index.ids == [1]
    assert document_storage.rows == {1, 2}
    assert persisted == [(1,)]


@pytest.mark.asyncio
async def test_vector_only_prune_restores_index_after_atomic_persist_failure(
    monkeypatch,
):
    import sys

    import astrbot_plugin_dynamics_memory.core.faiss_async_persist as persistence

    class _Persister:
        def __init__(self):
            self._mutation_lock = asyncio.Lock()
            self.flush_now = AsyncMock()

    persister = _Persister()
    storage = _FakeEmbeddingStorage([1, 2], persister)
    fake_db = SimpleNamespace(
        embedding_storage=storage,
        document_storage=SimpleNamespace(),
    )
    fake_faiss = SimpleNamespace(
        downcast_index=lambda index: index,
        vector_to_array=lambda id_map: id_map,
        clone_index=lambda index: _FakeIndex(index.ids),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda ids, dtype=None: ids, int64=object()),
    )
    writes = []

    def fail_first_write(index, _path):
        writes.append(tuple(index.ids))
        if len(writes) == 1:
            raise OSError("disk write failed")

    monkeypatch.setattr(persistence, "_write_index_atomic", fail_first_write)
    retriever = GraphVectorRetriever(fake_db)
    with pytest.raises(OSError, match="disk write failed"):
        await retriever.delete_vector_ids_atomically([2])

    assert storage.index.ids == [1, 2]
    assert writes == [(1,), (1, 2)]


@pytest.mark.asyncio
async def test_vector_prune_settles_double_cancelled_writer_and_rollback_under_lock(
    monkeypatch,
):
    import sys

    import astrbot_plugin_dynamics_memory.core.faiss_async_persist as persistence

    class _Persister:
        def __init__(self):
            self._mutation_lock = asyncio.Lock()
            self.flush_now = AsyncMock()

    persister = _Persister()
    storage = _FakeEmbeddingStorage([1, 2], persister)
    fake_db = SimpleNamespace(
        embedding_storage=storage,
        document_storage=SimpleNamespace(),
    )
    fake_faiss = SimpleNamespace(
        downcast_index=lambda index: index,
        vector_to_array=lambda id_map: id_map,
        clone_index=lambda index: _FakeIndex(index.ids),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda ids, dtype=None: ids, int64=object()),
    )

    first_started = threading.Event()
    first_release = threading.Event()
    rollback_started = threading.Event()
    rollback_release = threading.Event()
    durable_ids = []
    write_count = 0

    def barrier_writer(index, _path):
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            first_started.set()
            first_release.wait()
        elif write_count == 2:
            rollback_started.set()
            rollback_release.wait()
        durable_ids[:] = index.ids

    monkeypatch.setattr(persistence, "_write_index_atomic", barrier_writer)
    retriever = GraphVectorRetriever(fake_db)
    task = asyncio.create_task(retriever.delete_vector_ids_atomically([2]))

    assert await asyncio.to_thread(first_started.wait, 3)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert persister._mutation_lock.locked()

    first_release.set()
    assert await asyncio.to_thread(rollback_started.wait, 3)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert persister._mutation_lock.locked()

    rollback_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert write_count == 2
    assert durable_ids == [1, 2]
    assert storage.index.ids == [1, 2]
    assert not persister._mutation_lock.locked()


@pytest.mark.asyncio
async def test_graph_manager_builds_exact_stale_vector_prune_plan_and_revalidates():
    class _VectorRetriever:
        def __init__(self):
            self.rows = [
                {"id": 1, "text": "active", "metadata": '{"source_memory_id": 101}'},
                {"id": 2, "text": "archived", "metadata": '{"source_memory_id": 102}'},
                {
                    "id": 3,
                    "text": "archive without vector",
                    "metadata": '{"source_memory_id": 103}',
                },
            ]
            self.deleted = []
            self.faiss_db = SimpleNamespace(
                embedding_storage=SimpleNamespace(dimension=DIMENSION)
            )

        async def get_index_document_inventory(self):
            return list(self.rows), {1, 2}

        async def get_documents_by_ids(self, ids):
            return [row for row in self.rows if row["id"] in ids]

        async def delete_vector_ids_atomically(self, ids):
            pending = tuple(value for value in ids if value not in self.deleted)
            self.deleted.extend(pending)
            return pending

    active_ids = {101}
    vector_retriever = _VectorRetriever()

    async def check_sources(ids):
        return active_ids & set(ids)

    manager = GraphMemoryManager(
        graph_store=SimpleNamespace(),
        graph_vector_retriever=vector_retriever,
        graph_extractor=SimpleNamespace(),
        source_validity_checker=check_sources,
    )
    plan = await manager.plan_stale_vector_prune()
    assert plan.stale_index_ids == (2,)
    assert plan.missing_active_ids == ()
    assert dict(plan.classifications)[3] is RepairClass.ARCHIVED_UNINDEXED_SOURCE

    active_ids.add(102)
    with pytest.raises(ValueError, match="became active"):
        await manager.apply_stale_vector_prune(plan)
    assert vector_retriever.deleted == []

    active_ids.remove(102)
    assert await manager.apply_stale_vector_prune(plan) == (2,)
    assert await manager.apply_stale_vector_prune(plan) == ()
    assert vector_retriever.deleted == [2]
    assert len(vector_retriever.rows) == 3


@pytest.mark.asyncio
async def test_prune_holds_source_guard_through_vector_flush_window():
    class _VectorRetriever:
        def __init__(self):
            self.rows = [
                {"id": 2, "text": "archived", "metadata": '{"source_memory_id": 102}'},
            ]
            self.faiss_db = SimpleNamespace(
                embedding_storage=SimpleNamespace(dimension=DIMENSION)
            )
            self.delete_started = asyncio.Event()
            self.allow_delete = asyncio.Event()
            self.vector_present = True

        async def get_index_document_inventory(self):
            return list(self.rows), {2}

        async def get_documents_by_ids(self, ids):
            return [row for row in self.rows if row["id"] in ids]

        async def delete_vector_ids_atomically(self, ids):
            self.delete_started.set()
            await self.allow_delete.wait()
            if self.vector_present and 2 in ids:
                self.vector_present = False
                return (2,)
            return ()

    source_state = set()
    guard = SourceStateGuard()
    vectors = _VectorRetriever()
    ordering = []

    async def check_sources(ids):
        return source_state & set(ids)

    manager = GraphMemoryManager(
        graph_store=SimpleNamespace(),
        graph_vector_retriever=vectors,
        graph_extractor=SimpleNamespace(),
        source_validity_checker=check_sources,
        source_state_guard=guard,
    )
    plan = await manager.plan_stale_vector_prune()

    prune_task = asyncio.create_task(manager.apply_stale_vector_prune(plan))
    await vectors.delete_started.wait()

    async def reactivate_after_prune():
        async with guard.hold():
            # The caller must rebuild derived graph state before publishing active.
            ordering.append("reindex")
            vectors.vector_present = True
            source_state.add(102)
            ordering.append("active")

    reactivate_task = asyncio.create_task(reactivate_after_prune())
    await asyncio.sleep(0)
    assert not reactivate_task.done()
    assert 102 not in source_state

    vectors.allow_delete.set()
    assert await prune_task == (2,)
    await reactivate_task
    assert ordering == ["reindex", "active"]
    assert vectors.vector_present


@pytest.mark.asyncio
async def test_metadata_reactivation_reindexes_before_committing_active_status(
    monkeypatch,
):
    import sys

    guard = SourceStateGuard()
    state = {"status": "archived", "graph_vector_present": False}
    ordering = []

    class _GraphManager:
        async def index_memory(self, memory_id, content, metadata):
            assert state["status"] == "archived"
            state["graph_vector_present"] = True
            ordering.append("graph_reindexed")
            return True

        async def source_graph_index_complete(self, memory_id):
            return state["graph_vector_present"]

    class _HybridRetriever:
        async def update_metadata(self, memory_id, updates):
            assert ordering == ["graph_reindexed"]
            state.update(updates)
            ordering.append("active_committed")
            return True

    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = guard
    engine.faiss_db = SimpleNamespace(
        embedding_storage=SimpleNamespace(index=SimpleNamespace(id_map=[55]))
    )
    engine.graph_memory_manager = _GraphManager()
    engine.hybrid_retriever = _HybridRetriever()
    engine.get_memory = AsyncMock(
        return_value={"text": "source content", "metadata": {"status": "archived"}}
    )
    engine._invalidate_search_cache = lambda: None
    engine._mark_graph_repair_needed_locked = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(
            downcast_index=lambda index: index,
            vector_to_array=lambda id_map: id_map,
        ),
    )

    assert await engine._update_memory_status_guarded(55, {"status": "active"})
    assert ordering == ["graph_reindexed", "active_committed"]
    assert state["status"] == "active"
    assert state["needs_graph_repair"] is False
    engine._mark_graph_repair_needed_locked.assert_not_awaited()


@pytest.mark.asyncio
async def test_metadata_reactivation_with_missing_main_vector_stays_archived(
    monkeypatch,
):
    import sys

    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = SourceStateGuard()
    engine.faiss_db = SimpleNamespace(
        embedding_storage=SimpleNamespace(index=SimpleNamespace(id_map=[]))
    )
    engine.graph_memory_manager = SimpleNamespace(
        index_memory=AsyncMock(return_value=True),
        source_graph_index_complete=AsyncMock(return_value=True),
    )
    engine.hybrid_retriever = SimpleNamespace(update_metadata=AsyncMock())
    engine.get_memory = AsyncMock(
        return_value={"text": "source", "metadata": {"status": "archived"}}
    )
    engine._invalidate_search_cache = lambda: None
    engine._mark_graph_repair_needed_locked = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(
            downcast_index=lambda index: index,
            vector_to_array=lambda id_map: id_map,
        ),
    )

    assert not await engine._update_memory_status_guarded(55, {"status": "active"})
    engine.hybrid_retriever.update_metadata.assert_not_awaited()
    engine._mark_graph_repair_needed_locked.assert_awaited_once_with(
        55, "main_vector_missing_at_reactivation_commit"
    )


@pytest.mark.asyncio
async def test_reactivation_index_failure_keeps_source_archived_and_records_repair():
    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = SourceStateGuard()
    engine.graph_memory_manager = SimpleNamespace(
        index_memory=AsyncMock(side_effect=RuntimeError("fixture failure"))
    )
    engine.hybrid_retriever = SimpleNamespace(update_metadata=AsyncMock())
    engine.get_memory = AsyncMock(
        return_value={"text": "source content", "metadata": {"status": "archived"}}
    )
    engine._mark_graph_repair_needed_locked = AsyncMock()

    assert not await engine._update_memory_status_guarded(55, {"status": "active"})
    engine.hybrid_retriever.update_metadata.assert_not_awaited()
    engine._mark_graph_repair_needed_locked.assert_awaited_once_with(
        55, "reactivation_prepare_failed"
    )


@pytest.mark.asyncio
async def test_graph_repair_marker_is_durable_and_does_not_unarchive_source():
    connection = await aiosqlite.connect(":memory:", isolation_level=None)
    await connection.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, metadata TEXT)"
    )
    await connection.execute(
        "INSERT INTO documents (id, metadata) VALUES (?, ?)",
        (55, '{"status":"archived"}'),
    )
    engine = object.__new__(MemoryEngine)
    engine._db_write_lock = asyncio.Lock()
    engine.db_path = ":memory:"
    connection.row_factory = aiosqlite.Row
    engine.db_connection = connection
    engine.source_state_guard = SourceStateGuard()
    async with engine.source_state_guard.hold():
        await engine._mark_graph_repair_needed_locked(
            55, "graph_vector_missing_at_reactivation_commit"
        )
    cursor = await connection.execute(
        "SELECT metadata FROM documents WHERE id = ?", (55,)
    )
    metadata = json.loads((await cursor.fetchone())[0])
    assert metadata["status"] == "archived"
    assert metadata["needs_graph_repair"] is True
    assert (
        metadata["graph_repair_reason"] == "graph_vector_missing_at_reactivation_commit"
    )
    await connection.close()


@pytest.mark.asyncio
async def test_reactivation_deferred_and_cancelled_never_publish_active():
    for outcome, reason in (
        (False, "reactivation_prepare_failed"),
        (asyncio.CancelledError(), "reactivation_prepare_cancelled"),
    ):
        engine = object.__new__(MemoryEngine)
        engine.source_state_guard = SourceStateGuard()
        engine.graph_memory_manager = SimpleNamespace(
            index_memory=AsyncMock(return_value=outcome)
        )
        engine.hybrid_retriever = SimpleNamespace(update_metadata=AsyncMock())
        engine.get_memory = AsyncMock(
            return_value={"text": "source", "metadata": {"status": "archived"}}
        )
        engine._mark_graph_repair_needed_locked = AsyncMock()

        if isinstance(outcome, asyncio.CancelledError):
            engine.graph_memory_manager.index_memory.side_effect = outcome
            with pytest.raises(asyncio.CancelledError):
                await engine._update_memory_status_guarded(55, {"status": "active"})
        else:
            assert not await engine._update_memory_status_guarded(
                55, {"status": "active"}
            )

        engine.hybrid_retriever.update_metadata.assert_not_awaited()
        engine._mark_graph_repair_needed_locked.assert_awaited_once_with(55, reason)


@pytest.mark.asyncio
async def test_second_cancel_waits_for_repair_marker_sql_task():
    guard = SourceStateGuard()
    marker_started = asyncio.Event()
    allow_marker_commit = asyncio.Event()
    marker_persisted = False

    async def persist_marker(_memory_id, _reason):
        nonlocal marker_persisted
        marker_started.set()
        await allow_marker_commit.wait()
        marker_persisted = True

    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = guard
    engine.graph_memory_manager = SimpleNamespace(
        index_memory=AsyncMock(side_effect=asyncio.CancelledError())
    )
    engine.get_memory = AsyncMock(
        return_value={"text": "source", "metadata": {"status": "archived"}}
    )
    engine._mark_graph_repair_needed_locked = persist_marker

    task = asyncio.create_task(
        engine._update_memory_status_guarded(55, {"status": "active"})
    )
    await marker_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert guard._owner is task

    allow_marker_commit.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert marker_persisted
    assert guard._owner is None


@pytest.mark.asyncio
async def test_batch_restore_cancel_keeps_archived_source_and_persists_repair(
    monkeypatch,
):
    import sys

    embedding_started = asyncio.Event()

    async def blocked_embedding(_content):
        embedding_started.set()
        await asyncio.Event().wait()

    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda vector, dtype=None: vector, float32=object()),
    )
    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = SourceStateGuard()
    engine.db_connection = object()
    engine.get_memory = AsyncMock(
        return_value={"text": "source", "metadata": {"status": "archived"}}
    )
    engine.faiss_db = SimpleNamespace(
        embedding_storage=SimpleNamespace(
            insert=AsyncMock(),
            delete=AsyncMock(),
        ),
        embedding_provider=SimpleNamespace(get_embedding=blocked_embedding),
    )
    engine.bm25_retriever = SimpleNamespace(add_document=AsyncMock())
    engine.graph_memory_manager = None
    engine.atom_store = None
    engine._mark_graph_repair_needed_locked = AsyncMock()

    restore = asyncio.create_task(engine.restore_memory(55))
    await embedding_started.wait()
    restore.cancel()
    restore.cancel()
    with pytest.raises(asyncio.CancelledError):
        await restore
    engine._mark_graph_repair_needed_locked.assert_awaited_once_with(
        55, "restore_reindex_failed"
    )


@pytest.mark.asyncio
async def test_batch_restore_after_prune_rebuilds_vectors_before_active_commit(
    monkeypatch,
):
    import sys

    state = {"metadata": '{"status":"archived","needs_graph_repair":true}'}

    class _MainVectorIndex:
        def __init__(self):
            self.id_map = []

    class _EmbeddingStorage:
        def __init__(self):
            self.index = _MainVectorIndex()

        async def insert(self, _vector, document_id):
            self.index.id_map.append(document_id)

        async def delete(self, ids):
            self.index.id_map = [
                value for value in self.index.id_map if value not in ids
            ]

    class _Database:
        async def execute(self, query, params):
            if query.startswith("UPDATE documents SET metadata"):
                state["metadata"] = params[0]

    @asynccontextmanager
    async def write_db():
        yield _Database()

    class _GraphManager:
        def __init__(self):
            self.vector_present = False

        async def index_memory(self, *_args):
            self.vector_present = True

        async def source_graph_index_complete(self, _source_id):
            return self.vector_present

        async def delete_memory(self, _source_id):
            self.vector_present = False

    embedding_storage = _EmbeddingStorage()
    graph = _GraphManager()
    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = SourceStateGuard()
    engine.db_connection = object()
    engine.get_memory = AsyncMock(
        side_effect=lambda _id: {
            "text": "source",
            "metadata": json.loads(state["metadata"]),
        }
    )
    engine.faiss_db = SimpleNamespace(
        embedding_storage=embedding_storage,
        embedding_provider=SimpleNamespace(
            get_embedding=AsyncMock(return_value=[0.1, 0.2])
        ),
    )
    engine.bm25_retriever = SimpleNamespace(add_document=AsyncMock())
    engine.graph_memory_manager = graph
    engine.atom_store = None
    engine._write_db = write_db
    engine._invalidate_search_cache = lambda: None
    engine._mark_graph_repair_needed_locked = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda vector, dtype=None: vector, float32=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(
            downcast_index=lambda index: index,
            vector_to_array=lambda id_map: id_map,
        ),
    )

    assert await engine.restore_memory(55)
    assert embedding_storage.index.id_map == [55]
    assert graph.vector_present
    restored_metadata = json.loads(state["metadata"])
    assert restored_metadata["status"] == "active"
    assert restored_metadata["needs_graph_repair"] is False
    engine._mark_graph_repair_needed_locked.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_after_restore_sql_commit_preserves_vectors(monkeypatch):
    import sys

    state = {"metadata": '{"status":"archived"}'}
    sql_committed = asyncio.Event()
    allow_task_return = asyncio.Event()

    class _Index:
        def __init__(self):
            self.id_map = []

    class _EmbeddingStorage:
        def __init__(self):
            self.index = _Index()

        async def insert(self, _vector, document_id):
            self.index.id_map.append(document_id)

        async def delete(self, ids):
            self.index.id_map = [
                value for value in self.index.id_map if value not in ids
            ]

    class _Database:
        def __init__(self):
            self.pending_metadata = None

        async def execute(self, _query, params):
            self.pending_metadata = params[0]

    @asynccontextmanager
    async def write_db():
        db = _Database()
        yield db
        state["metadata"] = db.pending_metadata
        sql_committed.set()
        await allow_task_return.wait()

    class _GraphManager:
        def __init__(self):
            self.vector_present = False

        async def index_memory(self, *_args):
            self.vector_present = True

        async def source_graph_index_complete(self, _source_id):
            return self.vector_present

        async def delete_memory(self, _source_id):
            self.vector_present = False

    embedding_storage = _EmbeddingStorage()
    graph = _GraphManager()
    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = SourceStateGuard()
    engine.db_connection = object()
    engine.get_memory = AsyncMock(
        return_value={"text": "source", "metadata": {"status": "archived"}}
    )
    engine.faiss_db = SimpleNamespace(
        embedding_storage=embedding_storage,
        embedding_provider=SimpleNamespace(
            get_embedding=AsyncMock(return_value=[0.1, 0.2])
        ),
    )
    engine.bm25_retriever = SimpleNamespace(add_document=AsyncMock())
    engine.graph_memory_manager = graph
    engine.atom_store = None
    engine._write_db = write_db
    engine._invalidate_search_cache = lambda: None
    engine._mark_graph_repair_needed_locked = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda vector, dtype=None: vector, float32=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(
            downcast_index=lambda index: index,
            vector_to_array=lambda id_map: id_map,
        ),
    )

    restore = asyncio.create_task(engine.restore_memory(55))
    await sql_committed.wait()
    restore.cancel()
    restore.cancel()
    await asyncio.sleep(0)
    assert not restore.done()
    assert embedding_storage.index.id_map == [55]
    assert graph.vector_present

    allow_task_return.set()
    with pytest.raises(asyncio.CancelledError):
        await restore
    assert json.loads(state["metadata"])["status"] == "active"
    assert embedding_storage.index.id_map == [55]
    assert graph.vector_present
    engine._mark_graph_repair_needed_locked.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", ["verify", "coverage"])
async def test_commit_error_verification_cancellation_never_deletes_vectors(
    monkeypatch, cancel_phase
):
    import sys

    state = {"metadata": '{"status":"archived"}'}
    pending = {}
    barrier_started = asyncio.Event()
    allow_barrier = asyncio.Event()
    coverage_calls = 0

    class _Index:
        def __init__(self):
            self.id_map = []

    class _EmbeddingStorage:
        def __init__(self):
            self.index = _Index()
            self.deleted = []

        async def insert(self, _vector, document_id):
            self.index.id_map.append(document_id)

        async def delete(self, ids):
            self.deleted.extend(ids)
            self.index.id_map = [
                value for value in self.index.id_map if value not in ids
            ]

    class _Database:
        async def execute(self, _query, params):
            pending["metadata"] = params[0]

    class _Cursor:
        async def fetchone(self):
            return {"metadata": state["metadata"]}

    class _Reader:
        async def execute(self, _query, _params):
            return _Cursor()

    @asynccontextmanager
    async def write_db():
        yield _Database()
        # Simulate SQLite's commit completing before a later cleanup error.
        state["metadata"] = pending["metadata"]
        raise OSError("post-commit connection cleanup error")

    class _GraphManager:
        def __init__(self):
            self.vector_present = False
            self.deleted = False

        async def index_memory(self, *_args):
            self.vector_present = True

        async def source_graph_index_complete(self, _source_id):
            return self.vector_present

        async def delete_memory(self, _source_id):
            self.deleted = True
            self.vector_present = False

    embedding_storage = _EmbeddingStorage()
    graph = _GraphManager()
    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = SourceStateGuard()
    engine._db_write_lock = asyncio.Lock()
    engine.db_path = ":memory:"
    engine.db_connection = _Reader()
    engine.get_memory = AsyncMock(
        return_value={"text": "source", "metadata": {"status": "archived"}}
    )
    engine.faiss_db = SimpleNamespace(
        embedding_storage=embedding_storage,
        embedding_provider=SimpleNamespace(
            get_embedding=AsyncMock(return_value=[0.1, 0.2])
        ),
    )
    engine.bm25_retriever = SimpleNamespace(add_document=AsyncMock())
    engine.graph_memory_manager = graph
    engine.atom_store = None
    engine._write_db = write_db
    engine._invalidate_search_cache = lambda: None
    engine._mark_graph_repair_needed_locked = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda vector, dtype=None: vector, float32=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(
            downcast_index=lambda index: index,
            vector_to_array=lambda id_map: id_map,
        ),
    )

    real_verify = engine._verify_restore_commit
    real_coverage = engine._restore_index_coverage_complete

    async def barrier_verify(memory_id, expected_metadata):
        if cancel_phase == "verify":
            barrier_started.set()
            await allow_barrier.wait()
        return await real_verify(memory_id, expected_metadata)

    async def barrier_coverage(memory_id):
        nonlocal coverage_calls
        coverage_calls += 1
        if cancel_phase == "coverage" and coverage_calls == 2:
            barrier_started.set()
            await allow_barrier.wait()
        return await real_coverage(memory_id)

    engine._verify_restore_commit = barrier_verify
    engine._restore_index_coverage_complete = barrier_coverage
    restore = asyncio.create_task(engine.restore_memory(55))
    await barrier_started.wait()
    restore.cancel()
    restore.cancel()
    await asyncio.sleep(0)
    assert not restore.done()

    allow_barrier.set()
    with pytest.raises(asyncio.CancelledError):
        await restore
    assert json.loads(state["metadata"])["status"] == "active"
    assert embedding_storage.index.id_map == [55]
    assert embedding_storage.deleted == []
    assert graph.vector_present
    assert not graph.deleted


@pytest.mark.asyncio
async def test_batch_restore_active_early_return_checks_coverage(monkeypatch):
    import sys

    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = SourceStateGuard()
    engine.db_connection = object()
    engine.get_memory = AsyncMock(
        return_value={"text": "source", "metadata": {"status": "active"}}
    )
    engine.faiss_db = SimpleNamespace(
        embedding_storage=SimpleNamespace(index=SimpleNamespace(id_map=[]))
    )
    engine.graph_memory_manager = SimpleNamespace(
        source_graph_index_complete=AsyncMock(return_value=True),
        index_memory=AsyncMock(),
    )
    engine._mark_graph_repair_needed_locked = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(
            downcast_index=lambda index: index,
            vector_to_array=lambda id_map: id_map,
        ),
    )

    assert not await engine.restore_memory(55)
    engine.graph_memory_manager.index_memory.assert_not_awaited()
    engine._mark_graph_repair_needed_locked.assert_awaited_once_with(
        55, "active_restore_coverage_missing"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "index_outcome", [False, RuntimeError("fixture graph failure")]
)
async def test_batch_restore_graph_index_failure_keeps_source_archived(
    monkeypatch, index_outcome
):
    import sys

    state = {"metadata": '{"status":"archived"}'}

    class _Index:
        def __init__(self):
            self.id_map = []

    class _EmbeddingStorage:
        def __init__(self):
            self.index = _Index()

        async def insert(self, _vector, document_id):
            self.index.id_map.append(document_id)

        async def delete(self, ids):
            self.index.id_map = [
                value for value in self.index.id_map if value not in ids
            ]

    class _GraphManager:
        async def index_memory(self, *_args):
            if isinstance(index_outcome, BaseException):
                raise index_outcome
            return index_outcome

        async def delete_memory(self, _source_id):
            return True

    class _Database:
        async def execute(self, _query, _params):
            return None

    @asynccontextmanager
    async def write_db():
        yield _Database()

    embedding_storage = _EmbeddingStorage()
    engine = object.__new__(MemoryEngine)
    engine.source_state_guard = SourceStateGuard()
    engine.db_connection = object()
    engine.get_memory = AsyncMock(
        return_value={"text": "source", "metadata": json.loads(state["metadata"])}
    )
    engine.faiss_db = SimpleNamespace(
        embedding_storage=embedding_storage,
        embedding_provider=SimpleNamespace(
            get_embedding=AsyncMock(return_value=[0.1, 0.2])
        ),
    )
    engine.bm25_retriever = SimpleNamespace(add_document=AsyncMock())
    engine.graph_memory_manager = _GraphManager()
    engine.atom_store = None
    engine._write_db = write_db
    engine._mark_graph_repair_needed_locked = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda vector, dtype=None: vector, float32=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(
            downcast_index=lambda index: index,
            vector_to_array=lambda id_map: id_map,
        ),
    )

    with pytest.raises((RuntimeError,)):
        await engine.restore_memory(55)
    assert json.loads(state["metadata"])["status"] == "archived"
    assert embedding_storage.index.id_map == []
    engine._mark_graph_repair_needed_locked.assert_awaited_once_with(
        55, "restore_reindex_failed"
    )


@pytest.mark.asyncio
async def test_source_state_guard_nested_owner_and_child_waiter_cancellation():
    guard = SourceStateGuard()
    child_entered = asyncio.Event()
    child_started = asyncio.Event()

    async with guard.hold():
        async with guard.hold():
            assert guard._owner is asyncio.current_task()

        async def child():
            child_started.set()
            async with guard.hold():
                child_entered.set()

        waiter = asyncio.create_task(child())
        await child_started.wait()
        await asyncio.sleep(0)
        assert not child_entered.is_set()
        waiter.cancel()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert guard._owner is asyncio.current_task()
        async with guard.hold():
            assert guard._owner is asyncio.current_task()

    assert guard._owner is None
    async with guard.hold():
        assert guard._owner is asyncio.current_task()
    assert guard._owner is None


@pytest.mark.asyncio
async def test_source_state_guard_releases_after_owner_exception_and_double_cancel():
    guard = SourceStateGuard()
    started = asyncio.Event()
    never_release = asyncio.Event()

    async def cancelled_owner():
        async with guard.hold():
            started.set()
            await never_release.wait()

    owner = asyncio.create_task(cancelled_owner())
    await started.wait()
    owner.cancel()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert guard._owner is None

    with pytest.raises(RuntimeError, match="fixture owner failure"):
        async with guard.hold():
            raise RuntimeError("fixture owner failure")
    assert guard._owner is None

    acquired = asyncio.Event()

    async def next_owner():
        async with guard.hold():
            acquired.set()

    await asyncio.wait_for(asyncio.create_task(next_owner()), timeout=0.2)
    assert acquired.is_set()
