"""Comprehensive unit + integration tests for Recall V2 (Modules B, C, D, E)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

# Ensure staging root is importable as a package named `astrbot_plugin_dynamics_memory`
STAGING_ROOT = Path(__file__).resolve().parents[1]
PKG_NAME = "astrbot_plugin_dynamics_memory"
if PKG_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PKG_NAME,
        STAGING_ROOT / "__init__.py"
        if (STAGING_ROOT / "__init__.py").exists()
        else None,
        submodule_search_locations=[str(STAGING_ROOT)],
    )
    import types

    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(STAGING_ROOT)]
    sys.modules[PKG_NAME] = pkg


from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.base.config_validator import (
    LivingMemoryConfig,
    validate_config,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall import (
    MemoryRecall,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.message_utils import (
    OriginalInput,
)
from astrbot_plugin_dynamics_memory.core.models.memory_atom import (
    AtomStatus,
    AtomType,
    MemoryAtom,
)
from astrbot_plugin_dynamics_memory.core.recall_runtime import (
    RecallDeadline,
    trace_recall_hook,
)
from astrbot_plugin_dynamics_memory.core.retrieval.atom_retriever import (
    AtomRetrievalResult,
    AtomRetriever,
    aggregate_atoms_by_parent,
)
from astrbot_plugin_dynamics_memory.core.retrieval.atom_vector_retriever import (
    AtomVectorRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.dual_route_retriever import (
    DualRouteRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
    HybridResult,
    HybridRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.query_rewriter import (
    QueryRewriter,
    RewriteResult,
    detect_temporal_intent,
    extract_reply_quote,
    should_rewrite_query,
)
from astrbot_plugin_dynamics_memory.core.retrieval.rerank_text import (
    build_rerank_document,
)
from astrbot_plugin_dynamics_memory.core.retrieval.rrf_fusion import (
    BM25Result,
    FusedResult,
    RRFFusion,
    VectorResult,
)
from astrbot_plugin_dynamics_memory.core.retrieval.vector_search import (
    QueryEmbeddingCache,
    get_cached_embedding,
    query_embeddings,
)
from astrbot_plugin_dynamics_memory.core.summary_source import (
    AtomProjectionProof,
    summary_metadata_sha256,
)
from astrbot_plugin_dynamics_memory.core.utils.formatting import (
    format_memories_for_injection,
)
from astrbot_plugin_dynamics_memory.core.utils.injection_adapter import InjectionAdapter
from astrbot_plugin_dynamics_memory.storage.atom_store import AtomStore


# =============================================================================
# Helpers / Fakes
# =============================================================================


class FakeEmbeddingProvider:
    """Deterministic 1024-dim (or custom dim) embedding provider tracking API calls."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def get_dim(self) -> int:
        return self.dim

    def _encode(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        # Deterministic topic-sensitive hash so semantically identical keywords cluster
        for idx, ch in enumerate(text):
            vec[(ord(ch) + idx * 7) % self.dim] += 1.0
        if "抹茶" in text or "奶茶" in text or "饮料" in text:
            vec[0] += 10.0
        if "拉面" in text or "日料" in text or "火锅" in text:
            vec[1] += 10.0
        if "游戏" in text or "黑神话" in text or "联机" in text:
            vec[2] += 10.0
        if "猫咪" in text or "宠物" in text or "橘猫" in text:
            vec[3] += 10.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def get_embedding(self, text: str) -> list[float]:
        self.single_calls.append(text)
        return self._encode(text)

    async def get_embeddings(self, text: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(text))
        return [self._encode(t) for t in text]


class FakeRerankProvider:
    """Reranker returning explicit score mapping (calibrated probabilities in [0, 1])."""

    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, documents: list[str]):
        self.calls.append((query, list(documents)))
        out = []
        for idx, doc in enumerate(documents):
            score = float(self.score_fn(query, doc, idx))
            out.append(SimpleNamespace(index=idx, relevance_score=score))
        out.sort(key=lambda r: r.relevance_score, reverse=True)
        return out


class FakeLLMProvider:
    def __init__(self, response_text: str, delay: float = 0.0):
        self.response_text = response_text
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    async def text_chat(self, prompt: str = "", system_prompt: str = "", **kwargs):
        self.calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "kwargs": kwargs}
        )
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(completion_text=self.response_text)


# =============================================================================
# Module B Tests: Conditional Rewrite, 550ms Fallback, Tool Bypass, Batch Embed
# =============================================================================


def test_should_rewrite_triggers_and_skips():
    # 1. Clear long standalone question without pronouns/temporal/quote -> NO trigger
    trig, reason, temp, _ = should_rewrite_query(
        "夏目安安最喜欢的抹茶拿铁和甜点具体是什么口味",
        recent_messages=[{"role": "user", "content": "大家好"}],
    )
    assert trig is False
    assert reason == "none"
    assert temp == "none"

    # 2. Short query (<15 chars) WITH recent context -> triggers short_with_context
    trig, reason, _, _ = should_rewrite_query(
        "叫啥名字",
        recent_messages=[{"role": "user", "content": "我在玩一款新出的动作游戏"}],
    )
    assert trig is True
    assert reason == "short_with_context"

    # 3. Short query WITHOUT recent context and no pronoun/temporal -> NO trigger
    trig, reason, _, _ = should_rewrite_query("安安喜欢喝什么饮料", recent_messages=[])
    assert trig is False
    assert reason == "none"

    # 4. Coreference pronoun -> triggers coreference
    trig, reason, _, _ = should_rewrite_query(
        "他说的那家拉面馆地址在哪里啊", recent_messages=[]
    )
    assert trig is True
    assert reason == "coreference"

    # 5. Reply quote -> triggers reply_quote
    trig, reason, _, _ = should_rewrite_query(
        "真的很好吃吗推荐一下具体的招牌菜吧",
        reply_quote="老王: 昨天去的秋叶原那家拉面太棒了",
    )
    assert trig is True
    assert reason == "reply_quote"

    # 6. Temporal expression -> triggers temporal + temporal_intent="recent"
    trig, reason, temp, hint_days = should_rewrite_query(
        "昨天聊的那个游戏发售日是什么时候"
    )
    assert trig is True
    assert reason == "temporal"
    assert temp == "recent"
    assert hint_days == 2


@pytest.mark.asyncio
async def test_query_rewriter_success_and_thinking_disabled():
    llm = FakeLLMProvider(
        json.dumps(
            {
                "standalone_query": "老王昨天推荐的秋叶原拉面店名字是什么",
                "sub_queries": ["秋叶原拉面店", "老王推荐餐厅"],
                "temporal_intent": "recent",
                "time_hint_days": 2,
            },
            ensure_ascii=False,
        )
    )
    rewriter = QueryRewriter(
        provider_resolver=lambda: llm, config={"query_rewrite_timeout_seconds": 0.55}
    )
    res = await rewriter.rewrite(
        "他昨天说的那家店叫啥",
        recent_messages=[{"sender_name": "老王", "content": "秋叶原有家豚骨拉面绝了"}],
        sender_name="小明",
    )
    assert res.triggered is True
    assert res.rewritten is True
    assert res.standalone_query == "老王昨天推荐的秋叶原拉面店名字是什么"
    assert res.sub_queries == ["秋叶原拉面店", "老王推荐餐厅"]
    assert res.temporal_intent == "recent"
    assert res.time_hint_days == 2
    assert len(llm.calls) == 1
    extra_body = llm.calls[0]["kwargs"].get("extra_body", {})
    assert extra_body.get("enable_thinking") is False
    assert extra_body.get("thinking") == {"type": "disabled"}


@pytest.mark.asyncio
async def test_query_rewriter_550ms_timeout_parallel_raw_fallback():
    # Slow LLM takes 2.0s (> 0.15s timeout used for fast test; default config is 0.55s)
    slow_llm = FakeLLMProvider('{"standalone_query":"never"}', delay=2.0)
    rewriter = QueryRewriter(
        provider_resolver=lambda: slow_llm,
        config={"query_rewrite_timeout_seconds": 0.15},
    )

    raw_called = 0

    async def raw_search():
        nonlocal raw_called
        raw_called += 1
        await asyncio.sleep(0.05)
        return ["raw_hit_doc_42"]

    started = time.monotonic()
    result, raw_output = await rewriter.rewrite_with_raw_fallback(
        "他昨天说的那家店",
        raw_search,
        recent_messages=[{"sender_name": "老王", "content": "拉面很好吃"}],
    )
    elapsed = time.monotonic() - started

    assert result.triggered is True
    assert result.rewritten is False
    assert result.fallback_reason == "timeout"
    assert raw_called == 1
    assert raw_output == ["raw_hit_doc_42"]
    # Zero tail-latency penalty: raw search already finished while waiting for timeout!
    assert elapsed < 0.45


@pytest.mark.asyncio
async def test_query_rewriter_cancels_raw_fallback_on_fast_success():
    fast_llm = FakeLLMProvider(
        '{"standalone_query":"老王推荐的拉面店","sub_queries":["拉面店"],"temporal_intent":"none","time_hint_days":null}',
        delay=0.02,
    )
    rewriter = QueryRewriter(
        provider_resolver=lambda: fast_llm,
        config={"query_rewrite_timeout_seconds": 0.55},
    )
    raw_cancelled = False

    async def slow_raw():
        nonlocal raw_cancelled
        try:
            await asyncio.sleep(2.0)
            return ["raw"]
        except asyncio.CancelledError:
            raw_cancelled = True
            raise

    result, raw_output = await rewriter.rewrite_with_raw_fallback(
        "他说的那个",
        slow_raw,
        recent_messages=[{"sender_name": "老王", "content": "拉面店"}],
    )
    assert result.rewritten is True
    assert result.standalone_query == "老王推荐的拉面店"
    assert raw_output is None
    assert raw_cancelled is True


def test_memory_search_tool_never_imports_or_invokes_query_rewriter():
    tool_src = (STAGING_ROOT / "core" / "tools" / "memory_search_tool.py").read_text(
        encoding="utf-8"
    )
    assert "QueryRewriter" not in tool_src
    assert "query_rewriter" not in tool_src


@pytest.mark.asyncio
async def test_batch_embedding_cache_single_api_call():
    provider = FakeEmbeddingProvider(dim=16)
    loop = asyncio.get_running_loop()
    cache = QueryEmbeddingCache(loop.time() + 10.0)
    token = query_embeddings.set(cache)
    try:
        queries = ["老王推荐的秋叶原拉面店", "秋叶原拉面", "老王餐厅"]
        vectors = await cache.get_embeddings_batch(provider, queries)
        assert len(vectors) == 3
        assert len(provider.batch_calls) == 1
        assert provider.batch_calls[0] == queries
        assert len(provider.single_calls) == 0

        # Subsequent single lookups across all routes must hit cache with 0 API calls
        for q, expected in zip(queries, vectors):
            cached = await get_cached_embedding(provider, q)
            assert cached == expected
        assert len(provider.batch_calls) == 1
        assert len(provider.single_calls) == 0
    finally:
        query_embeddings.reset(token)


# =============================================================================
# Module C Tests: AtomVectorRetriever, Lifecycle Sync, Hybrid AtomRetriever
# =============================================================================


@pytest.mark.asyncio
async def test_atom_vector_index_incremental_lifecycle_and_hybrid_aggregation(tmp_path):
    db_path = str(tmp_path / "livingmemory.db")
    atom_index_path = str(tmp_path / "livingmemory_atoms.index")

    atom_store = AtomStore(db_path, cache_reads=True)
    await atom_store.initialize()

    emb_provider = FakeEmbeddingProvider(dim=16)
    atom_vec = AtomVectorRetriever(
        atom_store, emb_provider, index_path=atom_index_path, dimension=16
    )
    await atom_vec.initialize()
    atom_store.set_vector_sync(atom_vec)

    # Insert active atoms across 2 parent memories
    a1 = MemoryAtom(
        parent_memory_id=101,
        atom_type=AtomType.PREFERENCE,
        content="夏目安安非常喜欢喝无糖抹茶拿铁",
        entities=["夏目安安", "抹茶拿铁"],
        importance=0.8,
        session_id="qq:group:1",
        persona_id="anan",
    )
    a2 = MemoryAtom(
        parent_memory_id=101,
        atom_type=AtomType.FACTUAL,
        content="夏目安安说她每天下午都会来一杯抹茶饮料",
        entities=["夏目安安", "抹茶"],
        importance=0.7,
        session_id="qq:group:1",
        persona_id="anan",
    )
    a3 = MemoryAtom(
        parent_memory_id=202,
        atom_type=AtomType.EPISODIC,
        content="小明昨天晚上通宵玩黑神话悟空联机",
        entities=["小明", "黑神话"],
        importance=0.6,
        session_id="qq:group:1",
        persona_id="anan",
    )
    ids = await atom_store.insert_many([a1, a2, a3])
    assert len(ids) == 3

    # Incremental add to FAISS
    added = await atom_vec.add_atoms([a1, a2, a3])
    assert added == 3
    assert atom_vec.ntotal == 3

    # Verify disk persistence + reload
    await atom_vec.flush_now()
    assert Path(atom_index_path).exists()

    reloaded = AtomVectorRetriever(
        atom_store, emb_provider, index_path=atom_index_path, dimension=16
    )
    await reloaded.initialize()
    assert reloaded.ntotal == 3
    await reloaded.close()

    # Hybrid AtomRetriever shares QueryEmbeddingCache -> 0 extra embedding API calls!
    loop = asyncio.get_running_loop()
    cache = QueryEmbeddingCache(loop.time() + 10.0)
    token = query_embeddings.set(cache)
    try:
        await cache.get_embeddings_batch(emb_provider, ["抹茶饮料偏好"])
        calls_before = len(emb_provider.single_calls) + len(emb_provider.batch_calls)

        retriever = AtomRetriever(
            atom_store,
            config={"atom_vector_enabled": True},
            atom_vector_retriever=atom_vec,
        )
        parents = await retriever.search_parents(
            "抹茶饮料偏好",
            k=5,
            session_id="qq:group:1",
            persona_id="anan",
        )
        calls_after = len(emb_provider.single_calls) + len(emb_provider.batch_calls)
        assert calls_after == calls_before  # Reused QueryEmbeddingCache!

        assert len(parents) >= 1
        assert parents[0].parent_memory_id == 101
        # Both matching atoms aggregated into parent 101 with gentle second-atom boost
        assert len(parents[0].matched_atoms) == 2
        assert set(parents[0].retrieved_atom_ids) == {a1.atom_id, a2.atom_id}
        top_score = parents[0].matched_atoms[0]["score"]
        second_score = parents[0].matched_atoms[1]["score"]
        assert math.isclose(
            parents[0].score, top_score + 0.10 * second_score, abs_tol=1e-3
        )
    finally:
        query_embeddings.reset(token)

    # Incremental delete sync via AtomStore.delete_by_parent
    deleted = await atom_store.delete_by_parent(202)
    assert deleted == 1
    assert atom_vec.ntotal == 2

    # Batch rebuild_from_store rebuilds active atoms cleanly
    stats = await atom_vec.rebuild_from_store(batch_size=2)
    assert stats["total"] == 2
    assert stats["indexed"] == 2
    assert stats["failed"] == 0
    assert atom_vec.ntotal == 2

    await atom_vec.close()
    await atom_store.close()


# =============================================================================
# Module D Tests: Unified 3-Route Post-Fusion Rerank, Absolute Gating, Affinity,
#                 Temporal Intent, No Min-Max, No Graph Floor, Rerank Text & Format
# =============================================================================


class FakeBM25:
    def __init__(self, rows: list[BM25Result]):
        self.rows = rows

    async def search(self, query, k, session_id=None, persona_id=None, **kwargs):
        return self.rows[:k]


class FakeVector:
    def __init__(self, rows: list[VectorResult]):
        self.rows = rows

    async def search(self, query, k, session_id=None, persona_id=None, **kwargs):
        return self.rows[:k]


class FakeGraph:
    def __init__(self, rows: list[Any]):
        self.rows = rows

    async def search(self, query, k, *args, **kwargs):
        return self.rows[:k]


class FakeAtomRoute:
    def __init__(self, parents: list[Any]):
        self.parents = parents

    async def search_parents(self, query, k=25, **kwargs):
        return self.parents[:k]


@pytest.mark.asyncio
async def test_no_minmax_stretching_and_absolute_rejection():
    """Irrelevant candidates (0.12, 0.08, 0.05) MUST be rejected -> empty list."""
    bm25 = FakeBM25(
        [
            BM25Result(1, 0.9, "今天群里聊天气不错", {"importance": 0.5}),
            BM25Result(2, 0.7, "大家发了一堆表情包", {"importance": 0.5}),
            BM25Result(3, 0.5, "晚上九点有人打卡签到", {"importance": 0.5}),
        ]
    )
    vec = FakeVector([])
    scores_by_text = {
        "今天群里聊天气不错": 0.12,
        "大家发了一堆表情包": 0.08,
        "晚上九点有人打卡签到": 0.05,
    }
    reranker = FakeRerankProvider(lambda q, doc, i: scores_by_text.get(doc, 0.05))
    hybrid = HybridRetriever(
        bm25,
        vec,
        RRFFusion(),
        config={
            "rerank_enabled": True,
            "rerank_min_score": 0.35,
            "rerank_high_confidence_score": 0.70,
        },
        rerank_provider_resolver=lambda: reranker,
    )

    diag: dict[str, Any] = {}
    results = await hybrid.search("量子力学薛定谔方程", k=5, diagnostics=diag)
    assert results == []
    assert diag.get("rerank_rejected_all") is True

    # Single irrelevant candidate must ALSO be reranked and rejected
    single_hybrid = HybridRetriever(
        FakeBM25([BM25Result(9, 0.95, "毫无关系的闲聊", {"importance": 0.5})]),
        FakeVector([]),
        RRFFusion(),
        config={"rerank_enabled": True, "rerank_min_score": 0.35},
        rerank_provider_resolver=lambda: FakeRerankProvider(lambda q, d, i: 0.15),
    )
    assert await single_hybrid.search("完全无关的问题", k=5) == []


@pytest.mark.asyncio
async def test_hybrid_skip_rerank_flag_and_3route_post_fusion_unified_rerank():
    """Document + Atom + Graph run concurrently, skip doc rerank, run 1 unified rerank."""
    assert not hasattr(DualRouteRetriever, "_apply_graph_presence_floor")

    now = time.time()
    docs_store = {
        1: {
            "id": 1,
            "text": "群聊摘要：大家聊了天气，也顺便提了一句安安喜欢无糖抹茶拿铁。",
            "metadata": {
                "status": "active",
                "importance": 0.5,
                "create_time": now - 86400 * 10,
                "session_id": "qq:group:1",
            },
        },
        2: {
            "id": 2,
            "text": "群聊摘要：昨天老王分享了秋叶原一家拉面店，小明说改天去吃。",
            "metadata": {
                "status": "active",
                "importance": 0.5,
                "create_time": now - 86400 * 1,
                "session_id": "qq:group:2",
            },
        },
        3: {
            "id": 3,
            "text": "图谱命中：毫不相干的旧闲聊记录。",
            "metadata": {
                "status": "active",
                "importance": 0.5,
                "create_time": now - 86400 * 2,
                "session_id": "qq:group:1",
            },
        },
    }
    bm25 = FakeBM25(
        [BM25Result(1, 0.8, docs_store[1]["text"], docs_store[1]["metadata"])]
    )
    vec = FakeVector([])

    # Reranker gives high probability to atom-matched doc 1, medium to doc 2, and rejects graph-only doc 3 (0.10 < 0.35)!
    def score_fn(q, doc, idx):
        if "无糖抹茶拿铁" in doc:
            return 0.91
        if "拉面店" in doc:
            return 0.52
        return 0.10

    reranker = FakeRerankProvider(score_fn)
    hybrid = HybridRetriever(
        bm25,
        vec,
        RRFFusion(),
        config={
            "rerank_enabled": True,
            "rerank_min_score": 0.35,
            "rerank_high_confidence_score": 0.70,
            "rerank_candidates": 40,
        },
        rerank_provider_resolver=lambda: reranker,
    )

    atom_parents = aggregate_atoms_by_parent(
        [
            AtomRetrievalResult(
                atom_id=501,
                parent_memory_id=1,
                content="安安喜欢喝无糖抹茶拿铁",
                base_score=0.92,
                temporal_score=1.0,
                final_score=0.92,
                atom_type="preference",
                importance=0.8,
                confidence=0.9,
                ttl_days=60.0,
                decay_type="exponential",
                metadata={},
                vector_score=0.92,
                entities=["安安", "抹茶拿铁"],
                created_at=now - 86400 * 10,
            ),
            AtomRetrievalResult(
                atom_id=502,
                parent_memory_id=2,
                content="老王推荐了秋叶原拉面店",
                base_score=0.75,
                temporal_score=1.0,
                final_score=0.75,
                atom_type="episodic",
                importance=0.6,
                confidence=0.8,
                ttl_days=14.0,
                decay_type="exponential",
                metadata={},
                vector_score=0.75,
                entities=["老王", "拉面店"],
                created_at=now - 86400 * 1,
            ),
        ]
    )
    graph_rows = [
        SimpleNamespace(
            doc_id=3,
            final_score=0.99,
            rrf_score=0.5,
            keyword_score=0.9,
            vector_score=0.8,
            score_breakdown={},
        )
    ]

    async def batch_loader(metadata_filters, ids, limit):
        return [docs_store[i] for i in ids if i in docs_store]

    dual = DualRouteRetriever(
        hybrid,
        FakeGraph(graph_rows),
        memory_loader=lambda doc_id: asyncio.sleep(0, result=docs_store.get(doc_id)),
        config={
            "rerank_post_fusion": True,
            "rerank_candidates": 40,
            "rerank_min_score": 0.35,
            "rerank_high_confidence_score": 0.70,
        },
        memory_batch_loader=batch_loader,
        atom_retriever=FakeAtomRoute(atom_parents),
    )

    timings: dict[str, float] = {}
    results = await dual.search(
        "安安喜欢喝什么",
        k=5,
        current_session_id="qq:group:1",
        phase_timings=timings,
    )

    # Exactly ONE unified post-fusion rerank call across all 3 candidates!
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) == 3
    # Atom-matched doc 1 received [核心事实] focused rerank text
    assert any(
        "[核心事实]" in text and "安安喜欢喝无糖抹茶拿铁" in text
        for text in reranker.calls[0][1]
    )
    # Low-scoring graph-only doc 3 (0.10) is REJECTED (no graph presence floor!)
    assert [r.doc_id for r in results] == [1, 2]
    assert results[0].metadata.get("matched_atoms")
    assert results[0].metadata.get("retrieved_atom_ids") == [501]
    # Phase timings captured for all 3 routes + rerank
    for phase in ("doc_route", "atom_route", "graph_route", "rerank"):
        assert phase in timings


@pytest.mark.asyncio
async def test_confidence_cap_control_medium_vs_high():
    """Medium-confidence hits [0.35, 0.70) capped at 3; high-confidence fills up to k=5."""
    rows = [
        BM25Result(
            i,
            0.9 - i * 0.05,
            f"候选内容编号{i}主题完全不同{i * 111}",
            {"importance": 0.5},
        )
        for i in range(1, 6)
    ]

    # Case 1: All 5 candidates in medium confidence [0.40..0.56] -> capped at 3
    med_reranker = FakeRerankProvider(lambda q, d, idx: 0.56 - idx * 0.03)
    med_hybrid = HybridRetriever(
        FakeBM25(rows),
        FakeVector([]),
        RRFFusion(),
        config={
            "rerank_enabled": True,
            "rerank_min_score": 0.35,
            "rerank_high_confidence_score": 0.70,
            "rerank_medium_confidence_cap": 3,
        },
        rerank_provider_resolver=lambda: med_reranker,
    )
    med_res = await med_hybrid.search("测试查询", k=5)
    assert len(med_res) == 3

    # Case 2: All 5 candidates in high confidence [0.75..0.95] -> returns full k=5
    hi_reranker = FakeRerankProvider(lambda q, d, idx: 0.95 - idx * 0.04)
    hi_hybrid = HybridRetriever(
        FakeBM25(rows),
        FakeVector([]),
        RRFFusion(),
        config={
            "rerank_enabled": True,
            "rerank_min_score": 0.35,
            "rerank_high_confidence_score": 0.70,
            "rerank_medium_confidence_cap": 3,
        },
        rerank_provider_resolver=lambda: hi_reranker,
    )
    hi_res = await hi_hybrid.search("测试查询", k=5)
    assert len(hi_res) == 5


@pytest.mark.asyncio
async def test_post_rerank_affinity_and_intent_aware_temporal_weighting():
    """Verify +0.04 session, +0.03 speaker, +0.03 multi-route, and recent-only temporal boost."""
    now = time.time()
    # Two docs with IDENTICAL rerank probability (0.80) and importance (0.5):
    # Doc 10: old (300 days ago), same session + same speaker + multi-route
    # Doc 20: recent (1 day ago), different session + different speaker + single-route
    doc10_meta = {
        "status": "active",
        "importance": 0.5,
        "create_time": now - 86400 * 300,
        "session_id": "qq:group:current",
        "sender_id": "u_alice",
        "participants": ["Alice"],
    }
    doc20_meta = {
        "status": "active",
        "importance": 0.5,
        "create_time": now - 86400 * 1,
        "session_id": "qq:group:other",
        "sender_id": "u_bob",
        "participants": ["Bob"],
    }
    bm25 = FakeBM25(
        [
            BM25Result(10, 0.8, "Alice 喜欢喝美式咖啡不加糖", doc10_meta),
            BM25Result(20, 0.8, "Bob 昨天喝了一杯拿铁咖啡", doc20_meta),
        ]
    )
    reranker = FakeRerankProvider(lambda q, d, i: 0.80)
    hybrid = HybridRetriever(
        bm25,
        FakeVector([]),
        RRFFusion(),
        config={
            "rerank_enabled": True,
            "rerank_min_score": 0.35,
            "rerank_high_confidence_score": 0.70,
        },
        rerank_provider_resolver=lambda: reranker,
    )
    # Give Doc 10 an additional graph hit for multi-route consensus (+0.03)
    graph = FakeGraph(
        [
            SimpleNamespace(
                doc_id=10,
                final_score=0.7,
                rrf_score=0.3,
                keyword_score=0.7,
                vector_score=0.0,
                score_breakdown={},
            )
        ]
    )
    dual = DualRouteRetriever(
        hybrid,
        graph,
        memory_loader=lambda i: asyncio.sleep(0, result=None),
        config={
            "rerank_post_fusion": True,
            "affinity_session_bonus": 0.04,
            "affinity_speaker_bonus": 0.03,
            "affinity_multi_route_bonus": 0.03,
            "temporal_recent_bonus": 0.10,
        },
    )

    # Case A: Timeless query (temporal_intent="none") -> ZERO temporal penalty on 300-day-old doc10!
    res_timeless = await dual.search(
        "喜欢喝什么咖啡",
        k=5,
        current_session_id="qq:group:current",
        current_sender_id="u_alice",
        current_sender_name="Alice",
        temporal_intent="none",
    )
    by_id_timeless = {r.doc_id: r for r in res_timeless}
    assert by_id_timeless[10].score_breakdown["temporal_bonus"] == 0.0
    assert by_id_timeless[20].score_breakdown["temporal_bonus"] == 0.0
    assert by_id_timeless[10].score_breakdown["affinity_session_bonus"] == 0.04
    assert by_id_timeless[10].score_breakdown["affinity_speaker_bonus"] == 0.03
    assert by_id_timeless[10].score_breakdown["affinity_multi_route_bonus"] == 0.03
    assert res_timeless[0].doc_id == 10  # Affinity wins cleanly, old fact NOT penalized

    # Case B: Recent temporal query (temporal_intent="recent") on equal affinity -> recent doc wins!
    res_recent = await dual.search(
        "昨天喝了什么咖啡",
        k=5,
        current_session_id="qq:group:unrelated",
        current_sender_id="u_charlie",
        temporal_intent="recent",
    )
    by_id_recent = {r.doc_id: r for r in res_recent}
    expected_recent_boost = 0.10 * math.exp(-0.08 * 1.0)
    assert math.isclose(
        by_id_recent[20].score_breakdown["temporal_bonus"],
        expected_recent_boost,
        abs_tol=1e-3,
    )
    assert by_id_recent[10].score_breakdown["temporal_bonus"] < 1e-4
    assert res_recent[0].doc_id == 20


def test_rerank_text_and_atom_highlighted_injection_and_evidence_compat():
    long_summary = (
        "今天群里从早上聊到晚上，大家先讨论了天气和上班通勤塞车的问题。" * 12
        + "\n后来夏目安安明确更正说她不再喝含糖奶茶，现在只喝无糖抹茶拿铁。\n"
        + "晚上又聊了游戏服务器维护和周边团购的物流信息。" * 10
    )
    matched_atoms = [
        {
            "atom_id": 77,
            "content": "夏目安安不再喝含糖奶茶，只喝无糖抹茶拿铁",
            "created_at": 1730000000.0,
        }
    ]
    rerank_doc = build_rerank_document(
        "安安喝奶茶吗",
        long_summary,
        matched_atoms=matched_atoms,
    )
    assert len(rerank_doc) <= 350
    assert "[核心事实]" in rerank_doc
    assert "无糖抹茶拿铁" in rerank_doc
    assert "[上下文摘录]" in rerank_doc
    assert "不再喝含糖奶茶" in rerank_doc

    # Doc-only fallback also obeys default 350 char budget and preserves qualifier line
    doc_only_rerank = build_rerank_document("安安喝奶茶吗", long_summary)
    assert len(doc_only_rerank) <= 350
    assert "不再喝含糖奶茶" in doc_only_rerank

    # Injection formatting highlights matched atoms + concise parent context
    meta = {
        "importance": 0.85,
        "create_time": 1730000000.0,
        "topics": ["饮品偏好"],
        "matched_atoms": matched_atoms,
        "retrieved_atom_ids": [77],
    }
    # Verify summary_metadata_sha256 is unaffected by matched_atoms/retrieved_atom_ids
    meta_without_v2 = {
        k: v
        for k, v in meta.items()
        if k not in {"matched_atoms", "retrieved_atom_ids"}
    }
    assert summary_metadata_sha256(meta) == summary_metadata_sha256(meta_without_v2)

    formatted = format_memories_for_injection(
        [
            {
                "id": 101,
                "content": long_summary,
                "score": 0.91,
                "metadata": meta,
                "timestamp": 1730000000.0,
            }
        ]
    )
    assert "【精准命中事实】" in formatted
    assert "无糖抹茶拿铁" in formatted
    assert "【所属背景摘要】" in formatted


# =============================================================================
# Module E Tests: Telemetry Outcomes, Phases, Evidence Filter Compat, Schema
# =============================================================================


@pytest.mark.asyncio
async def test_memory_recall_telemetry_phases_outcomes_and_evidence_filter():
    cfg = ConfigManager(
        {
            "recall_engine": {
                "top_k": 5,
                "query_rewrite_enabled": True,
                "query_rewrite_timeout_seconds": 0.55,
            },
            "filtering_settings": {
                "use_persona_filtering": False,
                "use_session_filtering": True,
            },
        }
    )

    emb_provider = FakeEmbeddingProvider(dim=16)

    class StubEngine:
        def __init__(self):
            self.faiss_db = SimpleNamespace(embedding_provider=emb_provider)
            self.calls = []

        async def search_memories(self, query, k=5, **kwargs):
            self.calls.append((query, kwargs))
            timings = kwargs.get("phase_timings")
            if isinstance(timings, dict):
                timings["doc_route"] = 5.2
                timings["atom_route"] = 3.1
                timings["graph_route"] = 4.4
                timings["rerank"] = 18.5
            return [
                HybridResult(
                    doc_id=42,
                    final_score=0.89,
                    rrf_score=0.5,
                    bm25_score=0.7,
                    vector_score=0.85,
                    content="夏目安安喜欢喝无糖抹茶拿铁",
                    metadata={
                        "importance": 0.8,
                        "create_time": 1730000000.0,
                        "session_id": "qq:FriendMessage:10001",
                        "matched_atoms": [
                            {"atom_id": 9, "content": "夏目安安喜欢喝无糖抹茶拿铁"}
                        ],
                        "retrieved_atom_ids": [9],
                    },
                    score_breakdown={"rerank": 0.92},
                )
            ]

    class StubConvManager:
        async def get_messages(self, session_id, limit=6, use_cache=True):
            return [
                SimpleNamespace(
                    role="user", sender_name="小明", content="安安平常爱喝啥"
                )
            ]

        async def add_message_from_event(self, **kwargs):
            return None

    class StubMsgUtils:
        async def resolve_original_input(self, event):
            return OriginalInput("not_owned", None, None, (), "她喜欢喝啥")

        async def build_dedup_key(self, event, session_id, content):
            return "k1"

        async def is_duplicate_message(self, key):
            return True

        async def mark_message_processed(self, key):
            return None

    evidence_seen = []

    async def fake_evidence_filter(event, req, memory_list, actual_query, persona_id):
        evidence_seen.append((actual_query, memory_list))
        assert memory_list[0]["id"] == 42
        assert memory_list[0]["metadata"]["retrieved_atom_ids"] == [9]
        return memory_list

    def fake_summary_guard(event, req, source_records, **kwargs):
        async def _valid():
            return True

        req._lm_private_summary_validate_v1 = _valid
        return True

    fast_llm = FakeLLMProvider(
        '{"standalone_query":"夏目安安平时喜欢喝什么饮料","sub_queries":["安安饮料偏好"],"temporal_intent":"none","time_hint_days":null}'
    )
    rewriter = QueryRewriter(provider_resolver=lambda: fast_llm, config_manager=cfg)
    engine = StubEngine()

    recall = MemoryRecall(
        context=None,
        config_manager=cfg,
        memory_engine=engine,
        conversation_manager=StubConvManager(),
        message_utils=StubMsgUtils(),
        injection_adapter=InjectionAdapter(),
        evidence_filter=fake_evidence_filter,
        summary_guard=fake_summary_guard,
        query_rewriter=rewriter,
    )

    from astrbot.api.platform import MessageType

    event = SimpleNamespace(
        unified_msg_origin="qq:FriendMessage:10001",
        get_message_type=lambda: MessageType.FRIEND_MESSAGE,
        get_sender_id=lambda: "10001",
        get_sender_name=lambda: "小明",
        get_messages=lambda: [],
        get_extra=lambda k, d=None: d,
    )
    req = SimpleNamespace(
        extra_user_content_parts=[],
        contexts=[],
        system_prompt="",
        _lm_private_phase_ms_v1={},
    )

    await recall.handle_memory_recall(event, req, budget=RecallDeadline.start())

    # Explicit outcome (never "unknown")
    assert getattr(req, "_lm_private_hook_outcome_v1") == "ready"
    # All required phases recorded
    phases = req._lm_private_phase_ms_v1
    for required_phase in (
        "rewrite",
        "embed",
        "doc_route",
        "atom_route",
        "graph_route",
        "rerank",
        "evidence",
    ):
        assert required_phase in phases, f"missing phase {required_phase} in {phases}"
    # Rewritten query + sub_queries + batch embedding used
    assert engine.calls[0][0] == "夏目安安平时喜欢喝什么饮料"
    assert engine.calls[0][1]["sub_queries"] == ("安安饮料偏好",)
    assert len(emb_provider.batch_calls) == 1
    assert len(evidence_seen) == 1
    # Injected text includes atom highlight
    assert len(req.extra_user_content_parts) == 1
    assert "【精准命中事实】" in req.extra_user_content_parts[0].text

    # Policy-denied path must set explicit outcome="policy_denied" (never "unknown")
    denied_cfg = ConfigManager(
        {"access_control": {"whitelist_enabled": True, "allowed_ids": ""}}
    )
    denied_recall = MemoryRecall(
        context=None,
        config_manager=denied_cfg,
        memory_engine=engine,
        conversation_manager=StubConvManager(),
        message_utils=StubMsgUtils(),
        injection_adapter=InjectionAdapter(),
    )
    denied_req = SimpleNamespace(
        extra_user_content_parts=[], _lm_private_phase_ms_v1={}
    )
    await denied_recall.handle_memory_recall(event, denied_req)
    assert getattr(denied_req, "_lm_private_hook_outcome_v1") == "policy_denied"


def test_conf_schema_and_validator_feature_flags():
    schema_path = STAGING_ROOT / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    recall_items = schema["recall_engine"]["items"]
    expected_keys = [
        "rerank_candidates",
        "rerank_excerpt_max_chars",
        "rerank_post_fusion",
        "rerank_min_score",
        "rerank_high_confidence_score",
        "rerank_medium_confidence_cap",
        "rerank_timeout_seconds",
        "query_rewrite_enabled",
        "query_rewrite_provider_id",
        "query_rewrite_timeout_seconds",
        "atom_vector_enabled",
        "atom_route_weight",
        "affinity_session_bonus",
        "affinity_speaker_bonus",
        "affinity_multi_route_bonus",
        "temporal_recent_bonus",
    ]
    for key in expected_keys:
        assert key in recall_items, f"Missing {key} in _conf_schema.json"

    # Default validation succeeds with Stage 0 calibrated Pareto defaults
    defaults = validate_config({}).model_dump()
    assert defaults["recall_engine"]["rerank_candidates"] == 20
    assert defaults["recall_engine"]["rerank_excerpt_max_chars"] == 350
    assert defaults["recall_engine"]["rerank_post_fusion"] is True
    assert defaults["recall_engine"]["rerank_min_score"] == 0.60
    assert defaults["recall_engine"]["rerank_high_confidence_score"] == 0.85
    assert defaults["recall_engine"]["query_rewrite_enabled"] is True
    assert defaults["recall_engine"]["query_rewrite_timeout_seconds"] == 0.55
    assert defaults["recall_engine"]["atom_vector_enabled"] is True

    # Rollback feature flags validate cleanly
    rollback = validate_config(
        {
            "recall_engine": {
                "query_rewrite_enabled": False,
                "atom_vector_enabled": False,
                "rerank_post_fusion": False,
                "rerank_min_score": 0.0,
            }
        }
    ).model_dump()
    assert rollback["recall_engine"]["query_rewrite_enabled"] is False
    assert rollback["recall_engine"]["atom_vector_enabled"] is False
    assert rollback["recall_engine"]["rerank_post_fusion"] is False


@pytest.mark.asyncio
async def test_atom_expire_sync_and_atom_policy_filters_expired_matched_atoms(tmp_path):
    """Expired atoms are removed from FAISS on expire_stale_atoms and stripped from matched_atoms."""
    import aiosqlite

    db_path = str(tmp_path / "expire_test.db")
    atom_index_path = str(tmp_path / "livingmemory_atoms.index")

    atom_store = AtomStore(db_path, cache_reads=True)
    await atom_store.initialize()
    emb_provider = FakeEmbeddingProvider(dim=16)
    atom_vec = AtomVectorRetriever(
        atom_store, emb_provider, index_path=atom_index_path, dimension=16
    )
    await atom_vec.initialize()
    atom_store.set_vector_sync(atom_vec)

    live_atom = MemoryAtom(
        parent_memory_id=301,
        atom_type=AtomType.FACTUAL,
        content="夏目安安养了一只橘猫",
        importance=0.8,
        session_id="qq:group:1",
        persona_id="anan",
    )
    stale_atom = MemoryAtom(
        parent_memory_id=301,
        atom_type=AtomType.PLANNED,
        content="明天去买猫粮",
        importance=0.5,
        session_id="qq:group:1",
        persona_id="anan",
    )
    await atom_store.insert_many([live_atom, stale_atom])
    await atom_vec.add_atoms([live_atom, stale_atom])
    assert atom_vec.ntotal == 2

    # Force stale_atom past expires_at in SQLite and trigger expire_stale_atoms
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "UPDATE memory_atoms SET expires_at = ? WHERE id = ?",
            (time.time() - 100.0, stale_atom.atom_id),
        )
        await conn.commit()

    expired_count = await atom_store.expire_stale_atoms()
    assert expired_count == 1
    # Vector index automatically pruned expired atom!
    assert atom_vec.ntotal == 1

    # Verify MemoryEngineRecallMixin._apply_atom_policy strips expired matched_atoms
    from astrbot_plugin_dynamics_memory.core.managers.memory_engine_recall import (
        MemoryEngineRecallMixin,
    )

    class PolicyHarness(MemoryEngineRecallMixin):
        def __init__(self, store):
            self.atom_enabled = True
            self.atom_store = store

        async def _agentic_atom_rows(self, doc_ids):
            now = time.time()
            return {
                301: [
                    (live_atom.atom_id, 301, live_atom.content, "active", now + 3600),
                    (stale_atom.atom_id, 301, stale_atom.content, "expired", now - 100),
                ]
            }

        def _agentic_live_rows(self, rows):
            now = time.time()
            return [r for r in rows if r[3] == "active" and r[4] > now]

    harness = PolicyHarness(atom_store)
    cand = HybridResult(
        doc_id=301,
        final_score=0.9,
        rrf_score=0.5,
        bm25_score=0.8,
        vector_score=0.8,
        content="原始摘要文本",
        metadata={
            "matched_atoms": [
                {"atom_id": live_atom.atom_id, "content": live_atom.content},
                {"atom_id": stale_atom.atom_id, "content": stale_atom.content},
            ],
            "retrieved_atom_ids": [live_atom.atom_id, stale_atom.atom_id],
        },
    )
    projected = await harness._apply_atom_policy([cand])
    assert len(projected) == 1
    assert projected[0].metadata["retrieved_atom_ids"] == [live_atom.atom_id]
    assert [a["atom_id"] for a in projected[0].metadata["matched_atoms"]] == [
        live_atom.atom_id
    ]
    assert isinstance(projected[0].atom_projection, AtomProjectionProof)

    await atom_vec.close()
    await atom_store.close()


# =============================================================================
# Verifier Regression Tests (#7)
# =============================================================================


@pytest.mark.asyncio
async def test_bounded_pool_rewrite_and_two_sub_queries_no_key_collision():
    """Rewritten standalone_query + 2 sub_queries must NEVER coalesce onto raw or each other."""
    from astrbot_plugin_dynamics_memory.core.recall_runtime import (
        BoundedRecallSearchContext,
        BoundedRecallTaskPool,
    )

    pool = BoundedRecallTaskPool(max_inflight=16, max_cache=32, cache_ttl_seconds=5.0)
    try:
        seen_bm25_queries: list[str] = []

        class RecordingBM25:
            async def search(
                self, query, k, session_id=None, persona_id=None, **kwargs
            ):
                seen_bm25_queries.append(query)
                # Keep raw query slightly slow so it stays in-flight in the pool
                if query == "他昨天说的那个":
                    await asyncio.sleep(0.25)
                    return [
                        BM25Result(
                            999,
                            0.9,
                            "原始噪声文档",
                            {"importance": 0.5, "status": "active"},
                        )
                    ]
                doc_map = {
                    "老王昨天推荐的秋叶原拉面店": BM25Result(
                        10,
                        0.95,
                        "老王推荐秋叶原豚骨拉面店",
                        {"importance": 0.7, "status": "active"},
                    ),
                    "秋叶原拉面": BM25Result(
                        11,
                        0.85,
                        "秋叶原拉面攻略清单",
                        {"importance": 0.6, "status": "active"},
                    ),
                    "老王餐厅": BM25Result(
                        12,
                        0.80,
                        "老王收藏的东京餐厅",
                        {"importance": 0.6, "status": "active"},
                    ),
                }
                row = doc_map.get(query)
                return [row] if row else []

        class EmptyVec:
            async def search(self, query, k, *args, **kwargs):
                return []

        class EmptyGraph:
            async def search(self, query, k, *args, **kwargs):
                return []

        hybrid = HybridRetriever(
            RecordingBM25(), EmptyVec(), RRFFusion(), config={"rerank_enabled": False}
        )
        dual = DualRouteRetriever(
            hybrid,
            EmptyGraph(),
            memory_loader=lambda i: asyncio.sleep(0, result=None),
            config={"rerank_post_fusion": False},
        )

        cfg = ConfigManager(
            {
                "recall_engine": {
                    "top_k": 5,
                    "query_rewrite_enabled": True,
                    "query_rewrite_timeout_seconds": 0.55,
                },
                "filtering_settings": {
                    "use_persona_filtering": False,
                    "use_session_filtering": False,
                },
            }
        )

        class PoolEngine:
            def __init__(self):
                self.faiss_db = SimpleNamespace(embedding_provider=None)

            async def search_memories(self, query, k=5, **kwargs):
                return await dual.search(
                    query,
                    k=k,
                    execution_context=kwargs.get("execution_context"),
                    sub_queries=kwargs.get("sub_queries"),
                )

        class StubConvManager:
            async def get_messages(self, session_id, limit=6, use_cache=True):
                return [
                    SimpleNamespace(
                        role="user", sender_name="老王", content="昨天吃了秋叶原拉面"
                    )
                ]

            async def add_message_from_event(self, **kwargs):
                return None

        class StubMsgUtils:
            async def resolve_original_input(self, event):
                return OriginalInput("not_owned", None, None, (), "他昨天说的那个")

            async def build_dedup_key(self, event, session_id, content):
                return "k_pool"

            async def is_duplicate_message(self, key):
                return True

            async def mark_message_processed(self, key):
                return None

        captured_records: list[int] = []

        def fake_summary_guard(event, req, source_records, **kwargs):
            captured_records.extend(int(r["document_id"]) for r in source_records)

            async def _ok():
                return True

            req._lm_private_summary_validate_v1 = _ok
            return True

        fast_llm = FakeLLMProvider(
            json.dumps(
                {
                    "standalone_query": "老王昨天推荐的秋叶原拉面店",
                    "sub_queries": ["秋叶原拉面", "老王餐厅"],
                    "temporal_intent": "recent",
                    "time_hint_days": 2,
                },
                ensure_ascii=False,
            ),
            delay=0.02,
        )
        recall = MemoryRecall(
            context=None,
            config_manager=cfg,
            memory_engine=PoolEngine(),
            conversation_manager=StubConvManager(),
            message_utils=StubMsgUtils(),
            injection_adapter=InjectionAdapter(),
            recall_runtime=pool,
            summary_guard=fake_summary_guard,
            query_rewriter=QueryRewriter(
                provider_resolver=lambda: fast_llm, config_manager=cfg
            ),
        )

        from astrbot.api.platform import MessageType

        event = SimpleNamespace(
            unified_msg_origin="qq:FriendMessage:20002",
            get_message_type=lambda: MessageType.FRIEND_MESSAGE,
            get_sender_id=lambda: "20002",
            get_sender_name=lambda: "小明",
            get_messages=lambda: [],
            get_extra=lambda k, d=None: d,
        )
        req = SimpleNamespace(
            extra_user_content_parts=[],
            contexts=[],
            system_prompt="",
            _lm_private_phase_ms_v1={},
        )

        await recall.handle_memory_recall(event, req, budget=RecallDeadline.start())

        assert getattr(req, "_lm_private_hook_outcome_v1") == "ready"
        # All 3 distinct rewritten queries must have executed independently (no collision!)
        assert "老王昨天推荐的秋叶原拉面店" in seen_bm25_queries
        assert "秋叶原拉面" in seen_bm25_queries
        assert "老王餐厅" in seen_bm25_queries
        assert set(captured_records) == {10, 11, 12}
        assert 999 not in captured_records
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_batch_embedding_reused_across_submit_scope_search_and_all_3_routes(
    tmp_path,
):
    """Pre-warmed batch QueryEmbeddingCache must survive _submit_scope_search into Doc/Atom/Graph."""
    from astrbot_plugin_dynamics_memory.core.recall_runtime import (
        BoundedRecallTaskPool,
    )

    pool = BoundedRecallTaskPool(max_inflight=16, max_cache=32, cache_ttl_seconds=5.0)
    db_path = str(tmp_path / "embed_reuse.db")
    atom_store = AtomStore(db_path, cache_reads=True)
    await atom_store.initialize()
    emb_provider = FakeEmbeddingProvider(dim=16)
    atom_vec = AtomVectorRetriever(
        atom_store, emb_provider, index_path=str(tmp_path / "atoms.index"), dimension=16
    )
    await atom_vec.initialize()
    try:
        atom = MemoryAtom(
            parent_memory_id=50,
            atom_type=AtomType.PREFERENCE,
            content="安安最喜欢无糖抹茶拿铁",
            importance=0.8,
            session_id="qq:FriendMessage:30003",
            persona_id=None,
        )
        await atom_store.insert_many([atom])
        await atom_vec.add_atoms([atom])
        # Reset counters after index build
        emb_provider.single_calls.clear()
        emb_provider.batch_calls.clear()

        atom_retriever = AtomRetriever(
            atom_store,
            config={"atom_vector_enabled": True},
            atom_vector_retriever=atom_vec,
        )

        class CacheUsingVector:
            async def search(self, query, k, *args, **kwargs):
                await get_cached_embedding(emb_provider, query)
                return [
                    VectorResult(
                        50,
                        0.9,
                        "安安最喜欢无糖抹茶拿铁",
                        {
                            "importance": 0.8,
                            "status": "active",
                            "session_id": "qq:FriendMessage:30003",
                        },
                    )
                ]

        class CacheUsingGraph:
            async def search(self, query, k, *args, **kwargs):
                await get_cached_embedding(emb_provider, query)
                return []

        class EmptyBM25:
            async def search(self, query, k, *args, **kwargs):
                return []

        hybrid = HybridRetriever(
            EmptyBM25(),
            CacheUsingVector(),
            RRFFusion(),
            config={"rerank_enabled": False},
        )
        dual = DualRouteRetriever(
            hybrid,
            CacheUsingGraph(),
            memory_loader=lambda doc_id: asyncio.sleep(
                0,
                result={
                    "id": 50,
                    "text": "安安最喜欢无糖抹茶拿铁",
                    "metadata": {
                        "importance": 0.8,
                        "status": "active",
                        "session_id": "qq:FriendMessage:30003",
                    },
                },
            ),
            config={"rerank_post_fusion": False},
            atom_retriever=atom_retriever,
        )

        cfg = ConfigManager(
            {
                "recall_engine": {
                    "top_k": 5,
                    "query_rewrite_enabled": False,
                },
                "filtering_settings": {
                    "use_persona_filtering": False,
                    "use_session_filtering": False,
                },
            }
        )

        class ReuseEngine:
            def __init__(self):
                self.faiss_db = SimpleNamespace(embedding_provider=emb_provider)

            async def search_memories(self, query, k=5, **kwargs):
                return await dual.search(
                    query,
                    k=k,
                    execution_context=kwargs.get("execution_context"),
                    sub_queries=kwargs.get("sub_queries"),
                )

        class StubConvManager:
            async def get_messages(self, session_id, limit=6, use_cache=True):
                return []

            async def add_message_from_event(self, **kwargs):
                return None

        class StubMsgUtils:
            async def resolve_original_input(self, event):
                return OriginalInput(
                    "not_owned", None, None, (), "夏目安安最喜欢喝什么抹茶饮料"
                )

            async def build_dedup_key(self, event, session_id, content):
                return "k_emb"

            async def is_duplicate_message(self, key):
                return True

            async def mark_message_processed(self, key):
                return None

        def fake_summary_guard(event, req, source_records, **kwargs):
            async def _ok():
                return True

            req._lm_private_summary_validate_v1 = _ok
            return True

        recall = MemoryRecall(
            context=None,
            config_manager=cfg,
            memory_engine=ReuseEngine(),
            conversation_manager=StubConvManager(),
            message_utils=StubMsgUtils(),
            injection_adapter=InjectionAdapter(),
            recall_runtime=pool,
            summary_guard=fake_summary_guard,
        )

        from astrbot.api.platform import MessageType

        event = SimpleNamespace(
            unified_msg_origin="qq:FriendMessage:30003",
            get_message_type=lambda: MessageType.FRIEND_MESSAGE,
            get_sender_id=lambda: "30003",
            get_sender_name=lambda: "小明",
            get_messages=lambda: [],
            get_extra=lambda k, d=None: d,
        )
        req = SimpleNamespace(
            extra_user_content_parts=[],
            contexts=[],
            system_prompt="",
            _lm_private_phase_ms_v1={},
        )

        await recall.handle_memory_recall(event, req, budget=RecallDeadline.start())

        assert getattr(req, "_lm_private_hook_outcome_v1") == "ready"
        # Exactly ONE single embedding call total across prewarm + Doc + AtomVector + Graph!
        total_api_calls = len(emb_provider.single_calls) + len(emb_provider.batch_calls)
        assert total_api_calls == 1, (
            f"Expected 1 total embedding call, got single={emb_provider.single_calls} batch={emb_provider.batch_calls}"
        )
    finally:
        await atom_vec.close()
        await atom_store.close()
        await pool.close()


@pytest.mark.asyncio
async def test_rerank_rejected_all_not_resurrected_in_narrative_agentic_or_rollback():
    """Hard rejection must stay empty in search_narrative_semantic, agentic, and rerank_post_fusion=False."""
    from astrbot_plugin_dynamics_memory.core.managers.memory_engine_agentic import (
        MemoryEngineAgenticMixin,
    )
    from astrbot_plugin_dynamics_memory.core.managers.memory_engine_crud import (
        MemoryEngineCrudMixin,
    )
    from astrbot_plugin_dynamics_memory.core.managers.memory_engine_recall import (
        MemoryEngineRecallMixin,
    )

    low_reranker = FakeRerankProvider(lambda q, d, i: 0.08)
    doc_row = BM25Result(
        77,
        0.9,
        "毫不相关的群聊噪音",
        {
            "status": "active",
            "importance": 0.6,
            "session_id": "qq:group:1",
            "persona_id": "anan",
        },
    )
    hybrid = HybridRetriever(
        FakeBM25([doc_row]),
        FakeVector([]),
        RRFFusion(),
        config={"rerank_enabled": True, "rerank_min_score": 0.35},
        rerank_provider_resolver=lambda: low_reranker,
    )

    # (A) Rollback mode rerank_post_fusion=False must still propagate rerank_rejected_all
    #     and prevent MemoryEngineRecallMixin._merge_recent_memories from injecting noise!
    dual_rollback = DualRouteRetriever(
        hybrid,
        FakeGraph([]),
        memory_loader=lambda i: asyncio.sleep(0, result=None),
        config={"rerank_post_fusion": False},
    )

    recent_called = False

    class RollbackEngine(MemoryEngineRecallMixin):
        def __init__(self):
            self.config = {"recent_memory_count": 3, "rerank_post_fusion": False}
            self.dual_route_retriever = dual_rollback
            self.hybrid_retriever = hybrid
            self.atom_retriever = None
            self.atom_enabled = False
            self.atom_store = None
            self._search_cache_enabled = False
            self._search_cache_ttl = 0.0
            self._search_cache_max_size = 0
            self._search_cache_generation = 0
            self._search_cache = {}

        async def _merge_recent_memories(self, *args, **kwargs):
            nonlocal recent_called
            recent_called = True
            return [
                HybridResult(
                    999, 0.6, 0.0, None, None, "无关近期记忆", {"status": "active"}
                )
            ]

    rb_engine = RollbackEngine()
    rb_res = await rb_engine.search_memories(
        "完全无关的问题", k=5, touch_access=False, cache_results=False
    )
    assert rb_res == []
    assert recent_called is False

    # (B) search_narrative_semantic must NOT resurrect plan.document_fallback on rerank_rejected_all
    class NarrativeEngine(MemoryEngineCrudMixin, MemoryEngineRecallMixin):
        def __init__(self):
            self.config = {}
            self.dual_route_retriever = DualRouteRetriever(
                hybrid,
                FakeGraph([]),
                memory_loader=lambda i: asyncio.sleep(0, result=None),
                config={"rerank_post_fusion": True},
            )
            self.hybrid_retriever = hybrid
            self.atom_enabled = False
            self.atom_store = None

    nar_engine = NarrativeEngine()
    nar_res = await nar_engine.search_narrative_semantic(
        "完全无关的问题",
        k=5,
        origin="qq:group:1",
        personas=("anan",),
        diagnostics=None,
    )
    assert nar_res == []

    # (C) _agentic_search_conversation must NOT call _attach_agentic_atoms when rerank_rejected_all
    attach_called = False

    class AgenticEngine(MemoryEngineAgenticMixin, MemoryEngineRecallMixin):
        def __init__(self):
            self.config = {}
            self.dual_route_retriever = DualRouteRetriever(
                hybrid,
                FakeGraph([]),
                memory_loader=lambda i: asyncio.sleep(0, result=None),
                config={"rerank_post_fusion": True},
            )
            self.hybrid_retriever = hybrid
            self.atom_retriever = object()
            self.atom_enabled = False
            self.atom_store = None

        async def _attach_agentic_atoms(self, *args, **kwargs):
            nonlocal attach_called
            attach_called = True
            return args[0]

        async def _apply_agentic_atom_policy(self, rows):
            return rows

    ag_engine = AgenticEngine()
    ag_outcome = await ag_engine._agentic_search_conversation(
        query="完全无关的问题",
        k=5,
        session_id="qq:group:1",
        persona_id="anan",
        start_ts=None,
        end_ts=None,
        exclude_ids=None,
    )
    assert ag_outcome.results == []
    assert attach_called is False


@pytest.mark.asyncio
async def test_exhausted_budget_skips_rerank_without_cooldown_and_cd_outcome_whitelist():
    """Exhausted budget must skip rerank (no 30s cooldown), and outcomes must pass CD whitelist."""
    from astrbot_plugin_dynamics_memory.core.recall_runtime import (
        BoundedRecallSearchContext,
        BoundedRecallTaskPool,
    )

    reranker = FakeRerankProvider(lambda q, d, i: 0.9)
    hybrid = HybridRetriever(
        FakeBM25(
            [BM25Result(1, 0.9, "内容一", {"status": "active", "importance": 0.5})]
        ),
        FakeVector([]),
        RRFFusion(),
        config={"rerank_enabled": True, "rerank_min_score": 0.35},
        rerank_provider_resolver=lambda: reranker,
    )
    dual = DualRouteRetriever(
        hybrid,
        FakeGraph([]),
        memory_loader=lambda i: asyncio.sleep(0, result=None),
        config={"rerank_post_fusion": True},
    )

    # Create an execution_context whose remote_deadline is already in the past
    pool = BoundedRecallTaskPool(max_inflight=4, max_cache=4)
    try:
        now_m = time.monotonic()
        expired_ctx = BoundedRecallSearchContext(
            pool,
            RecallDeadline(now_m - 2.0, now_m + 5.0, 0.0),
            scope={
                "platform": "qq",
                "umo": "qq:group:1",
                "persona_id": "p",
                "instance_id": "p",
            },
            query="测试",
            purpose="reply",
            card_version="",
            correction_revision=None,
            persona_ids=("p",),
            origin="test",
            remote_deadline=now_m - 0.1,
        )
        diag: dict[str, Any] = {}
        res = await dual.search(
            "测试", k=5, execution_context=expired_ctx, diagnostics=diag
        )
        # Rerank was cleanly skipped without calling provider or triggering cooldown
        assert len(reranker.calls) == 0
        assert getattr(hybrid, "_rerank_retry_at", 0.0) == 0.0
        assert diag.get("rerank", {}).get("budget_exhausted") is True
        assert len(res) == 1
    finally:
        await pool.close()

    # Verify main.py outcomes match plugin_helpers_mixin.py whitelist (skipped_*)
    main_src = (STAGING_ROOT / "main.py").read_text(encoding="utf-8")
    assert '"skipped_not_ready"' in main_src
    assert '"skipped_managed"' in main_src
    assert '"not_ready"' not in main_src
    assert '"managed_skipped"' not in main_src


@pytest.mark.asyncio
async def test_calibrated_defaults_excerpt_max_chars_350_and_threshold_060():
    """HybridRetriever uses calibrated defaults: candidates=20, excerpt<=350, min=0.60, high=0.85."""
    long_body = (
        "今天群里从早聊到晚聊了天气交通和午饭菜单。" * 20
        + "\n后来夏目安安明确更正说她不再喝含糖奶茶，改喝无糖抹茶拿铁。\n"
        + "晚上又讨论了周末聚餐和桌游安排。" * 20
    )
    bm25 = FakeBM25(
        [
            BM25Result(1, 0.9, long_body, {"status": "active", "importance": 0.6}),
            BM25Result(
                2,
                0.8,
                "另一条普通候选内容长度适中",
                {"status": "active", "importance": 0.5},
            ),
        ]
    )
    # Default min_score=0.60 must pass 0.65 and reject 0.55
    reranker = FakeRerankProvider(
        lambda q, doc, idx: 0.65 if "无糖抹茶拿铁" in doc else 0.55
    )
    hybrid = HybridRetriever(
        bm25,
        FakeVector([]),
        RRFFusion(),
        config={"rerank_enabled": True},
        rerank_provider_resolver=lambda: reranker,
    )
    assert hybrid.rerank_candidates == 20
    assert hybrid.rerank_excerpt_max_chars == 350
    assert hybrid.rerank_min_score == 0.60
    assert hybrid.rerank_high_confidence_score == 0.85

    results = await hybrid.search("安安喝奶茶吗", k=5)
    assert [r.doc_id for r in results] == [1]
    assert len(reranker.calls) == 1
    sent_docs = reranker.calls[0][1]
    assert all(len(d) <= 350 for d in sent_docs)
    assert any("不再喝含糖奶茶" in d for d in sent_docs)

    # Custom rerank_excerpt_max_chars override is respected
    reranker_custom = FakeRerankProvider(lambda q, doc, idx: 0.90)
    hybrid_custom = HybridRetriever(
        FakeBM25(
            [BM25Result(1, 0.9, long_body, {"status": "active", "importance": 0.6})]
        ),
        FakeVector([]),
        RRFFusion(),
        config={"rerank_enabled": True, "rerank_excerpt_max_chars": 200},
        rerank_provider_resolver=lambda: reranker_custom,
    )
    await hybrid_custom.search("安安喝奶茶吗", k=5)
    assert all(len(d) <= 200 for d in reranker_custom.calls[0][1])
