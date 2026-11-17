"""Hybrid retrieval inside the graph-memory route."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from astrbot.api import logger


import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

from ..models.memory_atom import compute_decay_score
from ..utils.json_utils import safe_json_dict
from ..utils.number_utils import clamp_float, safe_float
from .graph_keyword_retriever import GraphKeywordRetriever
from .graph_vector_retriever import GraphVectorRetriever
from .recall_diagnostics import record_route
from .route_execution import search_route
from .rrf_fusion import BM25Result, RRFFusion, VectorResult


@dataclass(slots=True)
class GraphResult:
    """Combined graph-route result mapped to one memory document."""

    doc_id: int
    final_score: float
    rrf_score: float
    keyword_score: float | None
    vector_score: float | None
    content: str
    metadata: dict[str, Any]
    score_breakdown: dict[str, float] | None = None


class GraphRetriever:
    """Fuse graph keyword and graph vector retrieval results."""

    def __init__(
        self,
        keyword_retriever: GraphKeywordRetriever,
        vector_retriever: GraphVectorRetriever,
        rrf_fusion: RRFFusion,
        config: dict[str, Any] | None = None,
        source_validity_checker: Callable[[Sequence[int]], Awaitable[set[int]]]
        | None = None,
    ):
        self.keyword_retriever = keyword_retriever
        self.vector_retriever = vector_retriever
        self.rrf_fusion = rrf_fusion
        self.config = config or {}
        self.source_validity_checker = source_validity_checker
        self.decay_rate = float(self.config.get("decay_rate", 0.01))
        self.score_alpha = float(self.config.get("graph_score_alpha", 0.55))
        self.score_beta = float(self.config.get("graph_score_beta", 0.2))
        self.score_gamma = float(self.config.get("graph_score_gamma", 0.15))
        self.score_delta = float(self.config.get("graph_score_delta", 0.1))

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        persona_id: str | None = None,
        source_platform: str | None = None,
        interaction_type: str | None = None,
        source_session_id: str | None = None,
        execution_context=None,
        diagnostics: dict[str, Any] | None = None,
        relevance_only: bool = False,
    ) -> list[GraphResult]:
        """Run graph keyword and vector retrieval in parallel."""
        if not query or not query.strip():
            return []

        timeout = self.config.get("retrieval_timeout_seconds", 10.0)
        retrieval_context = execution_context
        if getattr(execution_context, "remote_deadline", None) is not None:
            budget = execution_context.budget
            reserve = min(0.15, budget.search_remaining() * 0.20)
            retrieval_context = execution_context.with_deadline(
                budget.search_deadline - reserve
            )
        keyword_filters = {}
        if source_platform is not None:
            keyword_filters["source_platform"] = source_platform
        if interaction_type is not None:
            keyword_filters["interaction_type"] = interaction_type
        if source_session_id is not None:
            keyword_filters["source_session_id"] = source_session_id
        vector_filters = (
            {"source_platform": source_platform} if source_platform is not None else {}
        )

        async def run_route(route_name, method, args, kwargs):
            if execution_context is None:
                return await search_route(route_name, method(*args, **kwargs), timeout)
            outcome = await retrieval_context.run(
                route_name.replace(" ", "_"),
                method,
                *args,
                remote=route_name == "graph vector",
                discriminator={
                    "session_id": session_id,
                    "persona_id": persona_id,
                    "candidate_limit": args[1],
                    "source_platform": source_platform,
                    "interaction_type": interaction_type,
                    "source_session_id": source_session_id,
                },
                **kwargs,
            )
            if outcome.get("state") == "completed":
                return outcome.get("value", []), None
            return [], RuntimeError(
                "retrieval_route_" + str(outcome.get("state", "error"))
            )

        keyword_task = run_route(
            "graph keyword",
            self.keyword_retriever.search,
            (query, k, session_id, persona_id),
            keyword_filters,
        )
        vector_task = (
            run_route(
                "graph vector",
                self.vector_retriever.search,
                (query, k, session_id, persona_id),
                vector_filters,
            )
            if source_session_id is None
            and getattr(execution_context, "local_only", False) is not True
            else asyncio.sleep(0, result=([], None))
        )
        (
            (keyword_results, keyword_error),
            (vector_results, vector_error),
        ) = await asyncio.gather(
            keyword_task,
            vector_task,
        )
        record_route(diagnostics, "keyword", keyword_error)
        record_route(diagnostics, "vector", vector_error)
        if diagnostics is not None:
            diagnostics["candidate_limited"] = (
                len(keyword_results) >= k
                or len(vector_results) >= k
                or len(
                    {r.doc_id for r in keyword_results}
                    | {r.doc_id for r in vector_results}
                )
                > k
            )

        if not keyword_results and not vector_results:
            return []

        if self.source_validity_checker is not None:
            source_ids = {
                int(item.doc_id) for item in (*keyword_results, *vector_results)
            }
            try:
                if execution_context is None:
                    active_source_ids = await self.source_validity_checker(
                        tuple(sorted(source_ids))
                    )
                else:
                    outcome = await execution_context.run(
                        "graph_source_validity",
                        self.source_validity_checker,
                        tuple(sorted(source_ids)),
                        discriminator={"source_ids": sorted(source_ids)},
                    )
                    if outcome.get("state") != "completed":
                        return []
                    active_source_ids = outcome.get("value", set())
            except Exception:
                logger.warning(
                    "图谱检索无法确认来源当前状态，已按失败关闭处理",
                    exc_info=True,
                )
                return []
            keyword_results = [
                item
                for item in keyword_results
                if int(item.doc_id) in active_source_ids
            ]
            vector_results = [
                item for item in vector_results if int(item.doc_id) in active_source_ids
            ]

        if not keyword_results and not vector_results:
            return []

        fused = self.rrf_fusion.fuse(
            [
                BM25Result(
                    doc_id=item.doc_id,
                    score=item.score,
                    content=item.content,
                    metadata=item.metadata,
                )
                for item in keyword_results
            ],
            [
                VectorResult(
                    doc_id=item.doc_id,
                    score=item.score,
                    content=item.content,
                    metadata=item.metadata,
                )
                for item in vector_results
            ],
            top_k=k,
        )
        if not fused:
            return []

        keyword_score_map = {item.doc_id: item.score for item in keyword_results}
        vector_score_map = {item.doc_id: item.score for item in vector_results}

        max_rrf = max(item.rrf_score for item in fused) or 1.0
        current_time = time.time()
        results: list[GraphResult] = []

        for item in fused:
            metadata = safe_json_dict(item.metadata)

            importance = clamp_float(metadata.get("importance"), default=0.5)
            create_time = safe_float(metadata.get("create_time"), current_time)
            last_access_time = safe_float(metadata.get("last_access_time"), create_time)
            if last_access_time <= 0:
                last_access_time = create_time
            reference_time = max(create_time, last_access_time)
            days_old = max(0.0, (current_time - reference_time) / 86400)
            recency_weight = math.exp(-self.decay_rate * days_old)
            graph_confidence = clamp_float(
                metadata.get("graph_confidence"), default=0.7
            )
            rrf_normalized = item.rrf_score / max_rrf

            # Temporal decay: use atom-level TTL when available
            atom_ttl = safe_float(metadata.get("ttl_days"), 0.0)
            temporal_factor = 1.0
            decay_type = str(metadata.get("decay_type", ""))
            if atom_ttl > 0:
                days_since_access = max(
                    0.0, (current_time - last_access_time) / 86400.0
                )
                temporal_factor = compute_decay_score(
                    decay_type,
                    atom_ttl,
                    days_since_access,
                )

            final_score = (
                self.score_alpha * rrf_normalized
                + self.score_beta * importance
                + self.score_gamma * recency_weight
                + self.score_delta * graph_confidence
            ) * temporal_factor
            if relevance_only:
                final_score = rrf_normalized

            results.append(
                GraphResult(
                    doc_id=item.doc_id,
                    final_score=final_score,
                    rrf_score=item.rrf_score,
                    keyword_score=keyword_score_map.get(item.doc_id),
                    vector_score=vector_score_map.get(item.doc_id),
                    content=item.content,
                    metadata=metadata,
                    score_breakdown={
                        "graph_rrf_normalized": round(rrf_normalized, 4),
                        "graph_importance": round(importance, 4),
                        "graph_recency_weight": round(recency_weight, 4),
                        "graph_confidence": round(graph_confidence, 4),
                        "graph_temporal_factor": round(temporal_factor, 4),
                        "graph_final_score": round(final_score, 4),
                        **(
                            {"agentic_relevance": final_score} if relevance_only else {}
                        ),
                    },
                )
            )

        results.sort(key=lambda item: item.final_score, reverse=True)
        return results[:k]


__all__ = ["GraphRetriever", "GraphResult"]
