from __future__ import annotations

import pytest
import sys
from types import SimpleNamespace

from astrbot_plugin_dynamics_memory.core.retrieval.vector_retriever import (
    VectorRetriever,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine


class _FakeVector:
    ndim = 1
    shape = (2,)

    def __iter__(self):
        return iter((0.25, 0.75))


class _FakeNumpy:
    float32 = "float32"

    @staticmethod
    def asarray(_value, dtype=None):
        return _FakeVector()

    @staticmethod
    def isfinite(_value):
        return SimpleNamespace(all=lambda: True)


@pytest.mark.asyncio
async def test_guarded_reflection_rechecks_source_after_embedding_before_document_commit(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "numpy", _FakeNumpy)
    calls = []

    class Embeddings:
        async def get_embedding(self, _content):
            calls.append("embedding")
            return [0.25, 0.75]

    class Storage:
        dimension = 2

        async def insert(self, vector, doc_id):
            calls.append(("vector", doc_id, tuple(vector)))

    class Documents:
        async def insert_document(self, _uuid, _content, _metadata):
            calls.append("document_commit")
            return 99

    class Faiss:
        embedding_provider = Embeddings()
        embedding_storage = Storage()
        document_storage = Documents()

        async def delete(self, _uuid):
            calls.append("rollback")

    retriever = object.__new__(VectorRetriever)
    retriever.faiss_db = Faiss()
    retriever._id_cache = {}

    async def source_changed_after_model():
        return False

    with pytest.raises(ValueError, match="source_snapshot_changed"):
        await VectorRetriever.add_document(
            retriever,
            "synthetic derived summary",
            {"status": "pending_source_validation"},
            source_validator=source_changed_after_model,
        )

    assert calls == ["embedding"]


@pytest.mark.asyncio
async def test_guarded_reflection_commits_only_after_post_embedding_source_check(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "numpy", _FakeNumpy)
    calls = []

    class Embeddings:
        async def get_embedding(self, _content):
            calls.append("embedding")
            return [0.25, 0.75]

    class Storage:
        dimension = 2

        async def insert(self, vector, doc_id):
            calls.append(("vector", doc_id, tuple(vector)))

    class Documents:
        async def insert_document(self, _uuid, _content, _metadata):
            calls.append("document_commit")
            return 99

    class Faiss:
        embedding_provider = Embeddings()
        embedding_storage = Storage()
        document_storage = Documents()

    retriever = object.__new__(VectorRetriever)
    retriever.faiss_db = Faiss()
    retriever._id_cache = {}
    validations = 0

    async def source_is_current():
        nonlocal validations
        validations += 1
        return True

    result = await VectorRetriever.add_document(
        retriever,
        "synthetic derived summary",
        {"status": "pending_source_validation"},
        source_validator=source_is_current,
    )
    assert result == 99
    assert validations == 2
    assert calls[0] == "embedding"
    assert calls[1] == "document_commit"
    assert calls[2][0] == "vector"


@pytest.mark.asyncio
async def test_pending_source_parent_is_hidden_from_vector_candidates(monkeypatch):
    from astrbot_plugin_dynamics_memory.core.retrieval import vector_retriever as module

    async def candidates(*_args, **_kwargs):
        return [
            SimpleNamespace(
                similarity=0.9,
                data={
                    "id": 1,
                    "text": "pending",
                    "metadata": {
                        "status": "pending_source_validation",
                    },
                },
            ),
            SimpleNamespace(
                similarity=0.8,
                data={
                    "id": 2,
                    "text": "active",
                    "metadata": {
                        "status": "active",
                    },
                },
            ),
        ]

    monkeypatch.setattr(module, "retrieve_vectors", candidates)
    retriever = object.__new__(VectorRetriever)
    retriever.faiss_db = SimpleNamespace()
    retriever.config = {}
    result = await VectorRetriever.search(retriever, "synthetic query", 5)
    assert [candidate.doc_id for candidate in result] == [2]


@pytest.mark.asyncio
async def test_pending_source_parent_is_hidden_from_atom_candidate_merge():
    engine = object.__new__(MemoryEngine)
    pending = SimpleNamespace(parent_memory_id=1, final_score=0.99)
    active = SimpleNamespace(parent_memory_id=2, final_score=0.8)

    class Hybrid:
        async def search(self, *_args, **_kwargs):
            return []

    class Atoms:
        async def search(self, *_args, **_kwargs):
            return [pending, active]

    class Documents:
        async def get_documents(self, **_kwargs):
            return [
                {
                    "id": 1,
                    "text": "pending",
                    "metadata": {
                        "status": "pending_source_validation",
                    },
                },
                {"id": 2, "text": "active", "metadata": {"status": "active"}},
            ]

    async def keep(items, *_args, **_kwargs):
        return items

    engine._search_cache_key = lambda *_args, **_kwargs: "no-cache"
    engine.dual_route_retriever = None
    engine.hybrid_retriever = Hybrid()
    engine.atom_retriever = Atoms()
    engine.faiss_db = SimpleNamespace(document_storage=Documents())
    engine._apply_atom_policy = keep
    engine._filter_by_retrieval_policy = lambda items: items
    engine._merge_recent_memories = keep
    results = await MemoryEngine.search_memories(
        engine,
        "synthetic query",
        k=5,
        touch_access=False,
        cache_results=False,
        allow_legacy_session_migration=False,
    )
    assert [candidate.doc_id for candidate in results] == [2]


@pytest.mark.asyncio
async def test_pending_source_parent_is_hidden_from_graph_source_validation():
    import aiosqlite

    connection = await aiosqlite.connect(":memory:")
    await connection.execute(
        "CREATE TABLE documents(id INTEGER PRIMARY KEY, metadata TEXT)"
    )
    await connection.execute(
        'INSERT INTO documents VALUES (1, \'{"status":"pending_source_validation"}\')'
    )
    await connection.execute(
        'INSERT INTO documents VALUES (2, \'{"status":"active"}\')'
    )
    await connection.commit()
    engine = object.__new__(MemoryEngine)
    engine.db_connection = connection
    result = await MemoryEngine.get_active_graph_source_ids(engine, (1, 2))
    assert result == {2}
    await connection.close()


@pytest.mark.asyncio
async def test_pending_source_parent_is_hidden_from_bm25_candidates(tmp_path):
    import aiosqlite
    from astrbot_plugin_dynamics_memory.core.retrieval.bm25_retriever import (
        BM25Retriever,
    )

    db_path = str(tmp_path / "pending-bm25.sqlite3")
    connection = await aiosqlite.connect(db_path)
    await connection.execute(
        "CREATE TABLE documents(id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
    )
    await connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?)",
        (1, "synthetic query pending", '{"status":"pending_source_validation"}'),
    )
    await connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?)",
        (2, "synthetic query active", '{"status":"active"}'),
    )
    await connection.commit()
    await connection.close()

    class Tokenizer:
        async def tokenize_async(self, _text, remove_stopwords=True):
            return ["synthetic", "query"]

    retriever = BM25Retriever(db_path, Tokenizer())
    await retriever.initialize()
    await retriever.add_document(1, "synthetic query pending")
    await retriever.add_document(2, "synthetic query active")
    candidates = await retriever.search("synthetic query", limit=10)
    assert [candidate.doc_id for candidate in candidates] == [2]
