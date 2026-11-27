"""Focused unit tests for 2GB RAM performance & memory optimizations."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from astrbot_plugin_dynamics_memory import series_panels
from astrbot_plugin_dynamics_memory.core import (
    faiss_async_persist,
    plugin_initializer_faiss,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.group_capture import (
    GroupCapture,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection import (
    _RECORDED_SCOPE_WAKES_LIMIT,
    _BoundedWakeSet,
)
from astrbot_plugin_dynamics_memory.core.evidence_view import EvidenceViewService
from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import (
    ConversationManager,
)
from astrbot_plugin_dynamics_memory.core.models.memory_atom import (
    AtomStatus,
    MemoryAtom,
)
from astrbot_plugin_dynamics_memory.core.processors.text_processor import TextProcessor
from astrbot_plugin_dynamics_memory.core.recall_runtime import BoundedRecallTaskPool
from astrbot_plugin_dynamics_memory.core.retrieval.atom_vector_retriever import (
    AtomVectorRetriever,
    _apply_faiss_omp_threads as _apply_atom_faiss_threads,
)
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
    is_local_rerank_provider,
    resolve_effective_rerank_candidates,
    resolve_rerank_cache_max_size,
)
from astrbot_plugin_dynamics_memory.storage import sqlite_connection
from astrbot_plugin_dynamics_memory.storage.atom_store import AtomStore
from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import EvidenceSidecar
from astrbot_plugin_dynamics_memory.storage.graph_store import GraphStore


def _set_profile(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    monkeypatch.setenv("ASTRBOT_RESOURCE_PROFILE", profile)


def test_faiss_omp_thread_cap_follows_resource_profile(monkeypatch: pytest.MonkeyPatch):
    recorded: list[int] = []
    fake_faiss = SimpleNamespace(omp_set_num_threads=lambda n: recorded.append(n))

    _set_profile(monkeypatch, "low_memory")
    assert plugin_initializer_faiss._apply_faiss_omp_threads(fake_faiss) == 1
    assert _apply_atom_faiss_threads(fake_faiss) == 1

    _set_profile(monkeypatch, "standard")
    assert plugin_initializer_faiss._apply_faiss_omp_threads(fake_faiss) == 2
    assert recorded == [1, 1, 2]


def test_shared_faiss_persister_executor_refcount_and_debounce(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    async def _run():
        _set_profile(monkeypatch, "low_memory")
        s1 = SimpleNamespace(path=str(tmp_path / "a.index"), index=None)
        s2 = SimpleNamespace(path=str(tmp_path / "b.index"), index=None)

        p1 = faiss_async_persist.AsyncIndexPersister(s1)
        p2 = faiss_async_persist.AsyncIndexPersister(s2)
        try:
            assert p1._executor is p2._executor
            assert p1._debounce_seconds == 20.0

            await p1.aclose()
            # Closing first persister must NOT shut down the shared executor while p2 is live.
            assert faiss_async_persist._SHARED_EXECUTOR is p2._executor
            assert faiss_async_persist._SHARED_EXECUTOR_REFS == 1
        finally:
            await p2.aclose()

        assert faiss_async_persist._SHARED_EXECUTOR is None
        assert faiss_async_persist._SHARED_EXECUTOR_REFS == 0

        _set_profile(monkeypatch, "low_memory_aggressive")
        p3 = faiss_async_persist.AsyncIndexPersister(s1)
        try:
            assert p3._debounce_seconds == 30.0
        finally:
            await p3.aclose()

    asyncio.run(_run())


def test_atom_vector_rebuild_uses_heavy_slot_and_malloc_trim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    events: list[str] = []

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_slot(context, purpose, *, plugin_id="", timeout=None):
        events.append(f"enter:{purpose}:{plugin_id}")
        try:
            yield {}
        finally:
            events.append("exit")

    monkeypatch.setattr(series_panels, "heavy_subprocess_slot", fake_slot)
    monkeypatch.setattr(
        series_panels,
        "try_malloc_trim",
        lambda **kwargs: events.append("malloc_trim") or True,
    )

    class _FakeIndex:
        def __init__(self, d: int):
            self.d = d
            self.ntotal = 0

        def add_with_ids(self, vecs, ids):
            self.ntotal += len(ids)

    import astrbot_plugin_dynamics_memory.core.retrieval.atom_vector_retriever as avr_mod

    monkeypatch.setattr(
        avr_mod, "_create_empty_idmap", lambda dim, context=None: _FakeIndex(dim)
    )

    fake_store = SimpleNamespace(
        list_active_atoms=AsyncMock(
            return_value=[
                MemoryAtom(atom_id=1, parent_memory_id=1, content="hello world")
            ]
        )
    )

    async def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32).tolist()

    fake_provider = SimpleNamespace(get_dim=lambda: 4, get_embeddings=fake_embed)
    retriever = AtomVectorRetriever(fake_store, fake_provider, dimension=4)
    result = asyncio.run(retriever.rebuild_from_store())

    assert result == {"total": 1, "indexed": 1, "failed": 0}
    assert events == [
        "enter:atom_vector_rebuild:astrbot_plugin_dynamics_memory",
        "malloc_trim",
        "exit",
    ]


def test_evidence_view_defaults_and_bounded_volatile_materials(tmp_path: Path):
    sidecar = EvidenceSidecar(tmp_path / "sidecar.sqlite3")
    svc_with_loader = EvidenceViewService(
        sidecar,
        scope_key_fn=lambda s: "scope:1",
        material_loader=AsyncMock(return_value=None),
        queue_size=16,
    )
    assert svc_with_loader.scheduler_poll_seconds == 30.0
    svc_with_loader._remember_volatile_material("m1", {"text": "abc"})
    assert len(svc_with_loader._volatile_materials) == 0

    svc_no_loader = EvidenceViewService(
        sidecar,
        scope_key_fn=lambda s: "scope:1",
        queue_size=16,
    )
    cap = max(64, 16 * 2)
    for i in range(cap + 25):
        svc_no_loader._remember_volatile_material(f"m{i}", {"text": f"v{i}"})
    assert len(svc_no_loader._volatile_materials) == cap
    assert "m0" not in svc_no_loader._volatile_materials
    assert f"m{cap + 24}" in svc_no_loader._volatile_materials
    asyncio.run(sidecar.close())


def test_evidence_sidecar_idle_claim_skips_write_tx_and_sets_pragmas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _set_profile(monkeypatch, "low_memory_aggressive")
    sidecar = EvidenceSidecar(tmp_path / "sidecar_idle.sqlite3")
    assert sidecar._read_executor._max_workers == 1

    async def _run():
        await sidecar.initialize()
        with sidecar._session() as db:
            assert db.execute("PRAGMA temp_store").fetchone()[0] == 1  # FILE
            assert db.execute("PRAGMA mmap_size").fetchone()[0] == 0
            assert db.execute("PRAGMA cache_size").fetchone()[0] < 0

        # First idle claim initializes _last_admission_prune_ts without BEGIN IMMEDIATE
        executed_sql: list[str] = []
        orig_connect = sidecar._connect

        class _SpyConn:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *args):
                executed_sql.append(sql.strip().upper())
                return self._inner.execute(sql, *args)

            def commit(self):
                return self._inner.commit()

            def close(self):
                return self._inner.close()

        monkeypatch.setattr(sidecar, "_connect", lambda: _SpyConn(orig_connect()))
        assert sidecar._claim_due_materials_sync(1000.0, 128, 24, 120.0, 4) == []
        assert not any(sql.startswith("BEGIN IMMEDIATE") for sql in executed_sql)
        assert not any(
            sql.startswith("DELETE FROM EVIDENCE_QUEUE_ADMISSIONS")
            for sql in executed_sql
        )

        # Second idle claim within 300s also avoids both BEGIN IMMEDIATE and admission check
        executed_sql.clear()
        assert sidecar._claim_due_materials_sync(1010.0, 128, 24, 120.0, 4) == []
        assert not any(sql.startswith("BEGIN IMMEDIATE") for sql in executed_sql)
        await sidecar.close()

    asyncio.run(_run())


def test_graph_store_pool_size_and_recall_pool_workers_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _set_profile(monkeypatch, "low_memory_aggressive")
    store = GraphStore(str(tmp_path / "graph.db"), cache_reads=True)
    assert store._READ_POOL_SIZE == 1

    pool = BoundedRecallTaskPool()
    assert pool._copy_executor._max_workers == 1
    asyncio.run(pool.close())

    _set_profile(monkeypatch, "low_memory")
    store2 = GraphStore(str(tmp_path / "graph2.db"), cache_reads=True)
    assert store2._READ_POOL_SIZE == 2


def test_conversation_manager_ttl_default_cap_and_no_debug_get_session(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_profile(monkeypatch, "low_memory_aggressive")
    fake_store = SimpleNamespace(
        add_message=AsyncMock(return_value=42),
        get_session=AsyncMock(side_effect=AssertionError("should not be called")),
    )
    mgr = ConversationManager(store=fake_store, session_ttl=10)
    assert mgr.max_cache_size == 16

    async def _run():
        msg = await mgr.add_message(session_id="s1", role="user", content="hi")
        assert msg.id == 42
        assert fake_store.get_session.await_count == 0

        # Manually insert an expired entry and verify _get_from_cache evicts it
        mgr._cache["s1"] = ([msg], 1.0)
        assert await mgr._get_from_cache("s1") is None
        assert "s1" not in mgr._cache

    asyncio.run(_run())


def test_group_capture_throttles_enforce_message_limit_every_20_messages():
    cfg = SimpleNamespace(
        get=lambda key, default=None: (
            True if key == "session_manager.enable_full_group_capture" else default
        )
    )
    conv_mgr = SimpleNamespace(add_message_from_event=AsyncMock())
    msg_utils = SimpleNamespace(
        extract_message_content=AsyncMock(return_value="hello"),
        build_dedup_key=AsyncMock(return_value=None),
        is_duplicate_message=AsyncMock(return_value=False),
        mark_message_processed=AsyncMock(),
        enforce_message_limit=AsyncMock(),
    )
    cap = GroupCapture(cfg, conv_mgr, msg_utils)

    from astrbot.api.platform import MessageType

    event = SimpleNamespace(
        unified_msg_origin="aiocqhttp:GroupMessage:123",
        get_message_type=lambda: MessageType.GROUP_MESSAGE,
        get_sender_id=lambda: "u1",
        get_self_id=lambda: "bot",
        get_sender_name=lambda: "Alice",
        get_platform_name=lambda: "aiocqhttp",
        get_group_id=lambda: "123",
    )

    async def _run():
        for _ in range(19):
            await cap.handle_all_group_messages(event)
        assert msg_utils.enforce_message_limit.await_count == 0

        await cap.handle_all_group_messages(event)
        assert msg_utils.enforce_message_limit.await_count == 1

        for _ in range(20):
            await cap.handle_all_group_messages(event)
        assert msg_utils.enforce_message_limit.await_count == 2

    asyncio.run(_run())


def test_bounded_recorded_scope_wakes_cap_and_set_api():
    wakes = _BoundedWakeSet()
    for i in range(_RECORDED_SCOPE_WAKES_LIMIT + 50):
        wakes.add(f"sess:{i}")
    assert len(wakes) == _RECORDED_SCOPE_WAKES_LIMIT
    assert "sess:0" not in wakes
    assert f"sess:{_RECORDED_SCOPE_WAKES_LIMIT + 49}" in wakes
    wakes.discard(f"sess:{_RECORDED_SCOPE_WAKES_LIMIT + 49}")
    assert f"sess:{_RECORDED_SCOPE_WAKES_LIMIT + 49}" not in wakes


def test_rerank_cloud_cap_and_cache_size_preserve_standard_and_local_sidecar(
    monkeypatch: pytest.MonkeyPatch,
):
    cloud_provider = SimpleNamespace(
        id="dashscope_rerank",
        provider_config={"api_base": "https://dashscope.aliyuncs.com/api/v1"},
    )
    local_provider = SimpleNamespace(
        id="local_rerank_sidecar",
        provider_config={"api_base": "http://127.0.0.1:8090/v1/rerank"},
    )

    assert is_local_rerank_provider(local_provider) is True
    assert is_local_rerank_provider(cloud_provider) is False

    _set_profile(monkeypatch, "standard")
    assert resolve_rerank_cache_max_size() == 128
    assert resolve_effective_rerank_candidates(40, provider=cloud_provider) == 40
    assert resolve_effective_rerank_candidates(40, provider=local_provider) == 40

    _set_profile(monkeypatch, "low_memory")
    assert resolve_rerank_cache_max_size() == 48
    assert resolve_effective_rerank_candidates(40, provider=cloud_provider) == 20
    assert resolve_effective_rerank_candidates(40, provider=local_provider) == 40

    _set_profile(monkeypatch, "low_memory_aggressive")
    assert resolve_rerank_cache_max_size() == 24
    assert resolve_effective_rerank_candidates(40, provider=cloud_provider) == 15
    assert resolve_effective_rerank_candidates(40, provider=local_provider) == 40


def test_atom_store_fts_resyncs_on_tokenizer_switch(tmp_path: Path):
    if not hasattr(sqlite_connection.aiosqlite, "Row"):
        pytest.skip("aiosqlite not installed in local interpreter")

    async def _run():
        db_path = str(tmp_path / "atom_tok_switch.db")
        store_jieba = AtomStore(
            db_path, text_processor=TextProcessor(tokenizer="jieba")
        )
        await store_jieba.initialize()
        await store_jieba.insert(
            MemoryAtom(
                parent_memory_id=1, content="我喜欢吃火锅", status=AtomStatus.ACTIVE
            )
        )
        await store_jieba.close()

        conn = sqlite3.connect(db_path)
        fmt1 = conn.execute(
            "SELECT value FROM atom_store_fts_state WHERE key='fts_format'"
        ).fetchone()[0]
        conn.close()
        assert fmt1 == "tokenized:jieba"

        store_builtin = AtomStore(
            db_path, text_processor=TextProcessor(tokenizer="builtin")
        )
        await store_builtin.initialize()
        await store_builtin.close()

        conn = sqlite3.connect(db_path)
        fmt2 = conn.execute(
            "SELECT value FROM atom_store_fts_state WHERE key='fts_format'"
        ).fetchone()[0]
        conn.close()
        assert fmt2 == "tokenized:builtin"

    asyncio.run(_run())
