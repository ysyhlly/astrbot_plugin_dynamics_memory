"""Batch 3 regression tests: P0-4, P1-#5/#6/#7/#10/#13, P2-#14/#17, startup decoupling."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core import i18n_backend
from astrbot_plugin_dynamics_memory.core.base.config_validator import LivingMemoryConfig
from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall import (
    MemoryRecall,
)
from astrbot_plugin_dynamics_memory.core.managers.consolidation_manager import (
    MemoryConsolidationManager,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.memory_scope import (
    adapter_access,
    adapter_sql,
)
from astrbot_plugin_dynamics_memory.core.models.memory_atom import (
    AtomType,
    MemoryAtom,
)
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.core.page_api_modules import (
    settings_handler as settings_mod,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.settings_handler import (
    SettingsHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import PageApiUtils
from astrbot_plugin_dynamics_memory.core.plugin_helpers_mixin import PluginHelpersMixin
from astrbot_plugin_dynamics_memory.core.processors.text_processor import TextProcessor
from astrbot_plugin_dynamics_memory.core.retrieval.bm25_retriever import (
    BM25Result,
    BM25Retriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.rrf_fusion import (
    BM25Result as RRFBM25Result,
    FusedResult,
    RRFFusion,
    VectorResult as RRFVectorResult,
)
from astrbot_plugin_dynamics_memory.core.retrieval.vector_retriever import (
    VectorResult,
    VectorRetriever,
)
from astrbot_plugin_dynamics_memory.storage.atom_store import AtomStore
from astrbot_plugin_dynamics_memory.storage.conversation_summary_progress import (
    TERMINAL_SUMMARY_STAGES,
)
from tests.event_handler_support import _make_real_handler
from tests.test_narrative_api import SCOPE, envelope
from tests.test_reflection_backlog import _configure_processor


class _StubFaissDB:
    async def insert(self, content: str, metadata: dict, id: int | None = None):
        return id or 1

    async def delete_documents(self, doc_ids):
        return len(list(doc_ids))


# ---------------------------------------------------------------------------
# P0-4: Managed scope poison-window circuit breaker (next_exhausted_cycles >= 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p04_managed_scope_poison_window_circuit_breaks_after_two_cycles(
    tmp_path: Path,
):
    session_id = scope_key(SCOPE)
    handler, manager, store = await _make_real_handler(tmp_path)
    try:
        handler.config_manager._config["reflection_engine"].update(
            summary_trigger_rounds=1,
            max_messages_per_segment=2,
            max_segments_per_run=3,
        )
        handler.memory_engine.find_similar_memory = AsyncMock(return_value=(None, None))
        _configure_processor(handler)

        base_ts = 1_790_000_000
        for idx in range(4):
            ev = envelope(
                f"r-{idx}",
                "delivery.success" if idx % 2 else "message.received",
                f"managed msg {idx}",
            )
            ev["occurred_at"] = base_ts + idx
            assert await store.ingest_narrative_event(ev)

        # First window [0:2] fails persistently for 2 full exhaustion cycles (6 attempts).
        handler.memory_processor.process_conversation.side_effect = RuntimeError(
            "persistent poison failure"
        )

        # Cycle 1: 3 attempts -> retry_exhausted, exhausted_cycles=1, pending_summary kept.
        for attempt in range(3):
            if attempt > 0:
                cur = await manager.get_session_metadata(session_id, "pending_summary")
                await manager.update_session_metadata(
                    session_id, "pending_summary", {**cur, "retry_after": 0}
                )
            await handler.reflect_recorded_scope(SCOPE)
            await asyncio.gather(*handler._storage_tasks)

        p_cycle1 = await manager.get_session_metadata(session_id, "pending_summary")
        assert p_cycle1 is not None
        assert p_cycle1["failure_stage"] == "retry_exhausted"
        assert p_cycle1["exhausted_cycles"] == 1

        # Cycle 2: expire cooldown, fail 3 more attempts -> circuit breaks to poisoned_window!
        for _ in range(3):
            cur = await manager.get_session_metadata(session_id, "pending_summary")
            assert cur is not None
            await manager.update_session_metadata(
                session_id, "pending_summary", {**cur, "retry_after": 0}
            )
            await handler.reflect_recorded_scope(SCOPE)
            await asyncio.gather(*handler._storage_tasks)

        # pending_summary MUST be cleared so managed recovery stops looping every 15m!
        assert await manager.get_session_metadata(session_id, "pending_summary") is None
        due = await store.managed_summary_recovery_candidates(0, time.time() + 10**6, 10)
        assert due == []

        # Quarantined window is preserved in managed_summary_refusals as poisoned_window.
        assert "poisoned_window" in TERMINAL_SUMMARY_STAGES
        _, refused_msgs, refusal = await store.get_managed_summary_window(
            session_id, 2, 2, retry_refused=True
        )
        assert len(refused_msgs) == 2
        assert refusal["failure_stage"] == "poisoned_window"
        assert refusal["exhausted_cycles"] == 2

        # Subsequent healthy window [2:4] can now advance normally!
        handler.memory_processor.process_conversation.side_effect = None
        handler.memory_processor.process_conversation.return_value = (
            "healthy second window",
            {},
            0.7,
        )
        await handler.reflect_recorded_scope(SCOPE)
        await asyncio.gather(*handler._storage_tasks)
        assert handler.memory_engine.add_memory.await_count == 1
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_p04_preexisting_exhausted_managed_pending_is_quarantined_on_read(
    tmp_path: Path,
):
    session_id = scope_key(SCOPE)
    handler, manager, store = await _make_real_handler(tmp_path)
    try:
        base_ts = 1_790_000_000
        for idx in range(2):
            ev = envelope(
                f"pre-{idx}",
                "delivery.success" if idx % 2 else "message.received",
                f"legacy poisoned {idx}",
            )
            ev["occurred_at"] = base_ts + idx
            assert await store.ingest_narrative_event(ev)

        _, msgs, _ = await store.get_managed_summary_window(session_id, 2, 2)
        await store.write_managed_summary_progress(
            session_id,
            msgs,
            {
                "start_index": 0,
                "end_index": 2,
                "retry_count": 0,
                "exhausted_cycles": 2,
                "failure_stage": "retry_exhausted",
                "retry_after": 0,
                "recovery_after": 0,
            },
        )
        # Reading window immediately quarantines legacy exhausted_cycles>=2 without LLM call.
        _, next_msgs, pending = await store.get_managed_summary_window(session_id, 2, 2)
        assert next_msgs == []
        assert pending is None
        assert await manager.get_session_metadata(session_id, "pending_summary") is None
    finally:
        await handler.shutdown()
        await store.close()


# ---------------------------------------------------------------------------
# P1-#5: _apply_mmr receives oversampled candidate pool (> k)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p15_hybrid_retriever_oversamples_so_mmr_deduplicates():
    now = time.time()
    meta = {"importance": 0.5, "create_time": now}
    # Doc 1 & Doc 2 are nearly identical; Doc 3 is distinct and slightly lower scored.
    bm25_hits = [
        BM25Result(1, 0.95, "Alice loves hiking in Kyoto mountains every autumn", meta),
        BM25Result(2, 0.94, "Alice loves hiking in Kyoto mountains every autumn!", meta),
        BM25Result(3, 0.80, "Bob built a Rust WebAssembly audio synthesizer", meta),
        BM25Result(4, 0.70, "Charlie prefers dark roast coffee in the morning", meta),
    ]
    vec_hits = [
        VectorResult(1, 0.95, "Alice loves hiking in Kyoto mountains every autumn", meta),
        VectorResult(2, 0.94, "Alice loves hiking in Kyoto mountains every autumn!", meta),
        VectorResult(3, 0.80, "Bob built a Rust WebAssembly audio synthesizer", meta),
        VectorResult(4, 0.70, "Charlie prefers dark roast coffee in the morning", meta),
    ]

    bm25 = SimpleNamespace(search=AsyncMock(return_value=bm25_hits))
    vec = SimpleNamespace(search=AsyncMock(return_value=vec_hits))
    retriever = HybridRetriever(
        bm25,
        vec,
        RRFFusion(k=60),
        config={"mmr_lambda": 0.5, "fusion_mode": "score"},
    )
    results = await retriever.search("Alice Kyoto Bob Rust", k=2)
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == 2
    assert 1 in doc_ids
    # Because MMR ran on >k candidates, near-duplicate doc 2 was displaced by distinct doc 3!
    assert 2 not in doc_ids
    assert 3 in doc_ids


# ---------------------------------------------------------------------------
# P1-#6: skipped_* and policy_denied recall outcomes are preserved
# ---------------------------------------------------------------------------


def test_p16_recall_outcome_preserves_skipped_and_rejects_forged():
    req = SimpleNamespace()
    for valid in (
        "skipped_at_other",
        "skipped_command_prefix",
        "skipped_short_noise",
        "policy_denied",
        "ready",
    ):
        MemoryRecall._set_recall_outcome(req, valid)
        assert req._lm_private_hook_outcome_v1 == valid

    MemoryRecall._set_recall_outcome(req, "forged outcome")
    assert req._lm_private_hook_outcome_v1 == "error"

    # Also verify PluginHelpersMixin._install_private_injection_getter preserves skipped_*.
    plugin = PluginHelpersMixin()
    req2 = SimpleNamespace(_lm_private_hook_outcome_v1="skipped_at_other")
    ev = SimpleNamespace()
    plugin._install_private_injection_getter(ev, req2)
    getter = req2._lm_private_injection_v1
    assert getter(req2, ev)["hook_outcome"] == "skipped_at_other"

    req2._lm_private_hook_outcome_v1 = "forged outcome"
    assert getter(req2, ev)["hook_outcome"] == "error"


# ---------------------------------------------------------------------------
# P1-#7: Single-hit min-max normalization does NOT saturate to 1.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p17_single_hit_normalization_does_not_saturate_to_one(tmp_path: Path):
    # 1. AtomStore single hit (FTS5 & LIKE fallback) < 1.0
    atom_db = str(tmp_path / "atom_p17.db")
    atom_store = AtomStore(atom_db, text_processor=TextProcessor())
    await atom_store.initialize()
    await atom_store.insert(
        MemoryAtom(
            parent_memory_id=1,
            atom_type=AtomType.PREFERENCE,
            content="likes matcha latte",
            session_id="s1",
        )
    )
    atoms = await atom_store.search_fts("matcha", session_id="s1", limit=5)
    assert len(atoms) == 1
    assert 0.0 < atoms[0].metadata["bm25_score"] < 0.86

    # 2. BM25Retriever single hit < 1.0
    bm25_db = str(tmp_path / "bm25_p17.db")
    async with aiosqlite.connect(bm25_db) as db:
        await db.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, doc_id TEXT, text TEXT, metadata TEXT)"
        )
        await db.execute(
            "INSERT INTO documents VALUES (1, 'd1', 'quantum computing note', '{\"status\":\"active\"}')"
        )
        await db.commit()
    bm25 = BM25Retriever(bm25_db, TextProcessor())
    await bm25.initialize()
    await bm25.add_document(1, "quantum computing note")
    bm25_res = await bm25.search("quantum", limit=5)
    assert len(bm25_res) == 1
    assert 0.0 < bm25_res[0].score < 0.86

    # 3. RRFFusion.fuse_scores single weak hit stays weak (not inflated to 1.0)
    fused = RRFFusion(k=60).fuse_scores(
        [RRFBM25Result(1, 0.35, "weak single hit", {})],
        [],
        top_k=5,
        bm25_weight=0.5,
    )
    assert len(fused) == 1
    assert abs(fused[0].rrf_score - 0.35) < 1e-6

    # 4. HybridRetriever._apply_rerank single low-relevance hit preserves ~0.12 (not 1.0)
    #    and unscored candidates get 0.0 (not 1.0).
    fake_rerank = SimpleNamespace(
        rerank=AsyncMock(return_value=[SimpleNamespace(index=0, relevance_score=0.12)])
    )
    hr = HybridRetriever(
        SimpleNamespace(),
        SimpleNamespace(),
        RRFFusion(k=60),
        config={"rerank_enabled": True},
        rerank_provider_resolver=lambda: fake_rerank,
    )
    candidates = [
        FusedResult(1, 0.9, 0.9, 0.9, "doc one", {}),
        FusedResult(2, 0.8, 0.8, 0.8, "doc two", {}),
    ]
    reranked = await hr._apply_rerank("query", candidates, k=2)
    by_id = {r.doc_id: r.rerank_score for r in reranked}
    assert abs(by_id[1] - 0.12) < 1e-6
    assert by_id[2] == 0.0


# ---------------------------------------------------------------------------
# P1-#10: Consolidation lock, prompt cap, atoms, dead code & streaming derivation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p110_consolidation_reentrancy_prompt_cap_atoms_and_scope_isolation():
    assert not hasattr(MemoryConsolidationManager, "_group_by_session")

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_merge(group):
        started.set()
        await release.wait()
        return {
            "summary": "merged summary",
            "key_facts": ["Alice prefers oolong tea"],
            "topics": ["tea"],
            "importance": 0.6,
        }

    candidates = [
        {
            "id": i,
            "content": f"session A memory {i} " + ("x" * 2500),
            "metadata": {
                "session_id": "sA",
                "persona_id": "p1",
                "adapter_id": "a1",
                "interaction_type": "private",
            },
        }
        for i in range(1, 9)
    ] + [
        {
            "id": 100 + i,
            "content": f"session B memory {i}",
            "metadata": {
                "session_id": "sB",
                "persona_id": "p1",
                "adapter_id": "a1",
                "interaction_type": "private",
            },
        }
        for i in range(1, 4)
    ]

    engine = SimpleNamespace(
        get_consolidation_candidates=AsyncMock(return_value=candidates),
        add_memory=AsyncMock(return_value=999),
        archive_memories=AsyncMock(side_effect=lambda ids: len(ids)),
    )
    fake_atom = MemoryAtom(
        parent_memory_id=0,
        atom_type=AtomType.PREFERENCE,
        content="Alice prefers oolong tea",
        session_id="sA",
    )
    processor = SimpleNamespace(
        merge_memories=AsyncMock(side_effect=slow_merge),
        classify_atoms_from_metadata=lambda meta, **kw: [fake_atom],
    )
    cfg_mgr = SimpleNamespace(
        get_section=lambda name: {
            "enabled": True,
            "granularity": "session",
            "min_memories_per_group": 2,
            "max_groups_per_run": 10,
            "keep_original": "archive",
        }
    )

    mgr = MemoryConsolidationManager(engine, processor, cfg_mgr)

    # Oversized session A group must be chunked into prompt-bounded sub-groups,
    # while never mixing session A and session B!
    groups = await mgr._build_groups(candidates, cfg_mgr.get_section("memory_consolidation"))
    assert len(groups) >= 2
    for grp in groups:
        sessions = {m["metadata"]["session_id"] for m in grp}
        assert len(sessions) == 1

    # Re-entrancy lock prevents concurrent overlap.
    t1 = asyncio.create_task(mgr.run_consolidation(force=True))
    await asyncio.wait_for(started.wait(), 2.0)
    overlap = await mgr.run_consolidation(force=True)
    assert overlap == {"skipped": True, "reason": "already_running"}
    release.set()
    res = await t1
    assert res["groups"] >= 1
    # Verify atoms were passed into add_memory!
    assert engine.add_memory.await_args_list[0].kwargs.get("atoms") == [fake_atom]


@pytest.mark.asyncio
async def test_p110_derive_evidence_materials_streaming_and_batch_slice_consistency():
    async def fake_stream(**kwargs):
        yield SimpleNamespace(is_chunk=True, completion_text='{"materials":[')
        yield SimpleNamespace(
            is_chunk=False,
            completion_text=json.dumps(
                {
                    "materials": [
                        {"material_id": f"m{i}", "atoms": []}
                        for i in range(1, 5)
                    ]
                }
            ),
        )

    stream_only_provider = SimpleNamespace(
        provider_config={"id": "stream-prov"},
        text_chat_stream=fake_stream,
    )
    plugin = PluginHelpersMixin()
    plugin.context = None
    plugin.initializer = SimpleNamespace(
        is_initialized=True,
        _providers_ready=True,
        memory_processor=SimpleNamespace(
            config={"llm_stream_mode": "auto"},
            _get_current_llm_provider=lambda: stream_only_provider,
        ),
    )

    materials = [
        {
            "material_id": f"m{i}",
            "source_status": "verified",
            "role": "user",
            "text": f"text {i}",
        }
        for i in range(1, 6)
    ]
    res = await plugin._derive_evidence_materials(materials)
    assert res == {"m1": [], "m2": [], "m3": [], "m4": []}

    # If the LLM hallucinates material_id "m5" (outside materials[:4]), it MUST be rejected!
    async def hallucinating_stream(**kwargs):
        yield SimpleNamespace(
            is_chunk=False,
            completion_text=json.dumps({"materials": [{"material_id": "m5", "atoms": []}]}),
        )

    stream_only_provider.text_chat_stream = hallucinating_stream
    with pytest.raises(RuntimeError, match="invalid_derivation_schema"):
        await plugin._derive_evidence_materials(materials)


# ---------------------------------------------------------------------------
# P1-#13: SettingsHandler.update_config hot-refreshes MemoryEngine & cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p113_settings_update_config_hot_refreshes_memory_engine(
    tmp_path: Path, monkeypatch
):
    db_path = tmp_path / "hot_reload.db"
    engine = MemoryEngine(
        db_path=str(db_path),
        faiss_db=_StubFaissDB(),
        config={"min_similarity_for_retrieval": 0.45, "rerank_enabled": False},
    )
    await engine.initialize()
    try:
        # Seed search cache so we can verify it is cleared on save.
        engine._search_cache[("k",)] = (time.time() + 60, [])
        assert len(engine._search_cache) == 1

        cfg_data = {
            "recall_engine": {
                "min_similarity_for_retrieval": 0.45,
                "rerank_enabled": False,
            }
        }

        class _FakeCfgMgr:
            def get(self, path, default=None):
                cur = cfg_data
                for part in path.split("."):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        return default
                return cur

            def get_section(self, name):
                return dict(cfg_data.get(name, {}))

            def apply_updates(self, updates):
                for path, val in updates.items():
                    sec, key = path.split(".", 1)
                    cfg_data.setdefault(sec, {})[key] = val
                return cfg_data

        plugin = SimpleNamespace(
            config_manager=_FakeCfgMgr(),
            initializer=SimpleNamespace(memory_engine=engine, memory_processor=None),
        )
        handler = SettingsHandler(PageApiUtils(), plugin)
        monkeypatch.setattr(
            settings_mod,
            "request",
            SimpleNamespace(
                get_json=AsyncMock(
                    return_value={
                        "values": {
                            "recall_engine.min_similarity_for_retrieval": 0.72,
                            "recall_engine.rerank_enabled": True,
                        }
                    }
                )
            ),
        )
        res = await handler.update_config()
        assert res["status"] == "ok"
        assert engine.config["min_similarity_for_retrieval"] == 0.72
        assert engine.hybrid_retriever.rerank_enabled is True
        assert len(engine._search_cache) == 0
    finally:
        await engine.close()


# ---------------------------------------------------------------------------
# P2-#14 / #17 & startup decoupling: offload, adapter_id index, schema, i18n
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p214_find_similar_pairs_offloads_off_event_loop(monkeypatch):
    import sys

    worker_threads: list[int] = []
    main_thread = threading.get_ident()

    class _FakeArray:
        def __init__(self, rows):
            self._rows = rows
            self.shape = (len(rows), len(rows[0]) if rows else 0)

        def astype(self, _dtype):
            return self

        def __getitem__(self, idx):
            return self._rows[idx]

        def __truediv__(self, denom):
            return _FakeArray([[v / denom for v in row] for row in self._rows])

        def __rsub__(self, lhs):
            return _FakeArray([[lhs - v for v in row] for row in self._rows])

    fake_np = SimpleNamespace(stack=lambda vecs: _FakeArray(vecs))
    monkeypatch.setitem(sys.modules, "numpy", fake_np)

    class _FakeIndex:
        def reconstruct(self, doc_id):
            worker_threads.append(threading.get_ident())
            return [float(doc_id), 0.0]

        def search(self, matrix, k):
            worker_threads.append(threading.get_ident())
            return _FakeArray([[0.0, 0.1], [0.0, 0.1]]), _FakeArray([[1, 2], [2, 1]])

    fake_faiss = SimpleNamespace(embedding_storage=SimpleNamespace(index=_FakeIndex()))
    vr = VectorRetriever(fake_faiss, config={})
    pairs = await vr.find_similar_pairs([1, 2], threshold=0.8)
    assert len(pairs) == 1
    assert pairs[0][:2] == (1, 2)
    assert worker_threads and all(tid != main_thread for tid in worker_threads)


@pytest.mark.asyncio
async def test_p217_adapter_id_index_schema_completeness_and_lazy_i18n(tmp_path: Path):
    # 1. idx_doc_adapter exists on freshly initialized livingmemory.db & is used by SQLite!
    db_path = tmp_path / "idx_adapter.db"
    engine = MemoryEngine(db_path=str(db_path), faiss_db=_StubFaissDB(), config={})
    await engine.initialize()
    try:
        idx_row = await (
            await engine.db_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_doc_adapter'"
            )
        ).fetchone()
        assert idx_row is not None

        cfg = SimpleNamespace(
            get=lambda p, d=None: [] if p == "filtering_settings.shared_adapter_ids" else d
        )
        ev = SimpleNamespace(
            get_platform_id=lambda: "qq-main",
            get_platform_name=lambda: "qq",
        )
        with adapter_access(cfg, ev):
            sql_pred, sql_params = adapter_sql("metadata")
            plan = await (
                await engine.db_connection.execute(
                    f"EXPLAIN QUERY PLAN SELECT id FROM documents WHERE {sql_pred}",
                    sql_params,
                )
            ).fetchall()
            plan_text = " ".join(str(r[3]) for r in plan)
            assert "idx_doc_adapter" in plan_text
    finally:
        await engine.close()

    # 2. Schema completeness: no zombie importance_weight, 3 missing fields added,
    #    granularity hint fixed.
    schema_path = Path(__file__).resolve().parent.parent / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "importance_weight" not in schema["recall_engine"]["items"]
    assert "hard_max_messages_per_session" in schema["session_manager"]["items"]
    assert "archived_retention_days" in schema["forgetting_agent"]["items"]
    assert "graph_route_timeout_seconds" in schema["graph_memory"]["items"]
    assert "跨会话按语义" not in schema["memory_consolidation"]["items"]["granularity"]["hint"]

    # Every validator field has a corresponding schema entry.
    schema_paths: set[str] = set()

    def _walk(prefix, items):
        for k, v in items.items():
            if not isinstance(v, dict):
                continue
            p = f"{prefix}.{k}" if prefix else k
            if v.get("type") == "object" and isinstance(v.get("items"), dict):
                _walk(p, v["items"])
            else:
                schema_paths.add(p)

    _walk("", schema)
    model_paths: set[str] = set()

    def _walk_model(prefix, cls):
        for name, field in cls.model_fields.items():
            p = f"{prefix}.{name}" if prefix else name
            ann = field.annotation
            if hasattr(ann, "model_fields"):
                _walk_model(p, ann)
            else:
                model_paths.add(p)

    _walk_model("", LivingMemoryConfig)
    assert model_paths - schema_paths == set()

    # 3. Lazy i18n resolution decouples startup order from master control plugin.
    master_state = {"ui_language": ""}
    i18n_backend.init(
        "zh",
        resolver=lambda: master_state["ui_language"] or "zh",
    )
    assert i18n_backend.get_language() == "zh"
    # Master control loads AFTER dynamics_memory and sets ui_language="en":
    master_state["ui_language"] = "en"
    assert i18n_backend.get_language() == "en"
    # Reset back to zh without resolver for clean global state.
    i18n_backend.init("zh")
