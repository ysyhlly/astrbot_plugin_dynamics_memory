"""Fuse document, atom, and graph retrieval routes with unified post-fusion rerank."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

from ..memory_scope import memory_metadata_matches_policy
from ..utils.json_utils import safe_json_dict
from ..utils.number_utils import clamp_float, safe_float
from .atom_retriever import AggregatedAtomParent
from .graph_retriever import GraphRetriever
from .hybrid_retriever import (
    HybridResult,
    HybridRetriever,
    resolve_effective_rerank_candidates,
)
from .query_rewriter import detect_temporal_intent
from .recall_diagnostics import merge_route
from .route_execution import search_route
from .vector_search import query_embeddings


class DualRouteRetriever:
    """Coordinate 3-route concurrent retrieval (Document + Atom + Graph) and unified rerank."""

    def __init__(
        self,
        document_retriever: HybridRetriever,
        graph_retriever: GraphRetriever,
        memory_loader: Callable[[int], Awaitable[dict[str, Any] | None]],
        config: dict[str, Any] | None = None,
        memory_batch_loader=None,
        atom_retriever: Any = None,
    ):
        self.document_retriever = document_retriever
        self.graph_retriever = graph_retriever
        self.atom_retriever = atom_retriever
        self.memory_loader = memory_loader
        self.memory_batch_loader = memory_batch_loader
        self.config = config or {}
        self.document_route_weight = float(
            self.config.get("document_route_weight", 0.65)
        )
        self.graph_route_weight = float(self.config.get("graph_route_weight", 0.35))
        self.atom_route_weight = float(self.config.get("atom_route_weight", 0.35))
        self.cross_route_bonus = float(self.config.get("cross_route_bonus", 0.08))
        self.dynamic_route_weighting = bool(
            self.config.get("dynamic_route_weighting", True)
        )
        self.graph_route_timeout = float(
            self.config.get("graph_route_timeout_seconds", 3.5)
        )
        self.rerank_post_fusion = bool(self.config.get("rerank_post_fusion", True))
        self.rerank_candidates = max(
            2, min(100, int(self.config.get("rerank_candidates", 20) or 20))
        )
        self.affinity_session_bonus = float(
            self.config.get("affinity_session_bonus", 0.04)
        )
        self.affinity_speaker_bonus = float(
            self.config.get("affinity_speaker_bonus", 0.03)
        )
        self.affinity_multi_route_bonus = float(
            self.config.get("affinity_multi_route_bonus", 0.03)
        )
        self.temporal_recent_bonus = float(
            self.config.get("temporal_recent_bonus", 0.10)
        )
        self.doc_route_top_n = max(
            10, int(self.config.get("doc_route_top_n", 25) or 25)
        )
        self.atom_route_top_n = max(
            10, int(self.config.get("atom_route_top_n", 25) or 25)
        )
        self.graph_route_top_n = max(
            10, int(self.config.get("graph_route_top_n", 20) or 20)
        )

    @staticmethod
    def _record_phase(
        phase_timings: dict[str, float] | None, name: str, started: float
    ) -> None:
        if phase_timings is None:
            return
        elapsed = round((time.monotonic() - started) * 1000, 1)
        phase_timings[name] = round(phase_timings.get(name, 0.0) + elapsed, 1)

    def _can_post_fusion_rerank(self, local_only: bool) -> bool:
        return (
            self.rerank_post_fusion
            and not local_only
            and bool(getattr(self.document_retriever, "rerank_enabled", False))
            and getattr(self.document_retriever, "rerank_provider_resolver", None)
            is not None
        )

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        persona_id: str | None = None,
        interaction_type: str | tuple[str, ...] | None = None,
        source_platform: str | None = None,
        source_session_id: str | None = None,
        execution_context=None,
        diagnostics: dict[str, Any] | None = None,
        relevance_only: bool = False,
        narrative_plan=None,
        excluded_session_ids: tuple[str, ...] | None = None,
        *,
        sub_queries: Sequence[str] | None = None,
        current_session_id: str | None = None,
        current_sender_id: str | None = None,
        current_sender_name: str | None = None,
        temporal_intent: str | None = None,
        time_hint_days: int | None = None,
        phase_timings: dict[str, float] | None = None,
    ) -> list[HybridResult]:
        """Run Document + Atom + Graph routes concurrently, merge, and post-fusion rerank."""
        local_only = getattr(execution_context, "local_only", False) is True
        post_fusion_rerank = self._can_post_fusion_rerank(local_only)

        timeout = (
            execution_context.budget.remaining()
            if execution_context is not None
            else self.config.get("retrieval_timeout_seconds", 10.0) * 2 + 1
        )
        graph_timeout = (
            min(timeout, self.graph_route_timeout)
            if self.graph_route_timeout > 0
            else timeout
        )
        graph_context = execution_context
        if getattr(execution_context, "remote_deadline", None) is not None:
            budget = execution_context.budget
            reserve = min(0.2, budget.search_remaining() * 0.20)
            capped_deadline = min(
                budget.search_deadline - reserve,
                asyncio.get_running_loop().time() + graph_timeout
                if hasattr(budget, "search_deadline")
                else budget.search_deadline - reserve,
            )
            graph_context = execution_context.with_deadline(capped_deadline)

        ambient_cache = query_embeddings.get()
        context_cache = (
            getattr(execution_context, "embedding_cache", None)
            if execution_context is not None
            else None
        )
        outer_cache = ambient_cache if ambient_cache is not None else context_cache
        owns_cache = outer_cache is None
        cache = outer_cache if outer_cache is not None else {}
        token = query_embeddings.set(cache) if ambient_cache is None else None

        doc_diagnostics: dict[str, Any] = {}
        atom_diagnostics: dict[str, Any] = {}
        graph_diagnostics: dict[str, Any] = {}

        doc_options: dict[str, Any] = (
            {"diagnostics": doc_diagnostics} if diagnostics is not None else {}
        )
        graph_options: dict[str, Any] = (
            {"diagnostics": graph_diagnostics} if diagnostics is not None else {}
        )
        if relevance_only:
            doc_options["relevance_only"] = True
            graph_options["relevance_only"] = True
        if narrative_plan is not None:
            doc_options["narrative_plan"] = narrative_plan
        if excluded_session_ids:
            doc_options["excluded_session_ids"] = tuple(excluded_session_ids)
            graph_options["excluded_session_ids"] = tuple(excluded_session_ids)

        clean_sub_queries = [
            str(sq).strip()
            for sq in (sub_queries or ())
            if str(sq or "").strip() and str(sq).strip() != query.strip()
        ][:2]

        doc_limit = (
            narrative_plan.initial
            if narrative_plan is not None
            else max(self.doc_route_top_n if post_fusion_rerank else max(k * 2, k), k)
        )
        atom_limit = max(
            self.atom_route_top_n if post_fusion_rerank else max(k * 2, k), k
        )
        graph_limit_init = (
            narrative_plan.initial
            if narrative_plan is not None
            else max(self.graph_route_top_n if post_fusion_rerank else max(k * 2, k), k)
        )

        try:

            async def _run_doc_query(q: str, limit: int) -> list[HybridResult]:
                call_opts = dict(doc_options)
                if post_fusion_rerank:
                    call_opts["skip_rerank"] = True
                return await self.document_retriever.search(
                    q,
                    limit,
                    session_id,
                    persona_id,
                    interaction_type=interaction_type,
                    source_platform=source_platform,
                    source_session_id=source_session_id,
                    execution_context=execution_context,
                    **call_opts,
                )

            async def document_candidates() -> list[HybridResult]:
                started = time.monotonic()
                try:
                    if not clean_sub_queries:
                        return await _run_doc_query(query, doc_limit)
                    sub_limit = max(10, doc_limit // 2)
                    batches = await asyncio.gather(
                        _run_doc_query(query, doc_limit),
                        *(_run_doc_query(sq, sub_limit) for sq in clean_sub_queries),
                        return_exceptions=True,
                    )
                    return self._merge_multi_query_docs(batches, doc_limit)
                finally:
                    self._record_phase(phase_timings, "doc_route", started)

            def document_route():
                return search_route(
                    "document",
                    document_candidates(),
                    timeout,
                    safe_errors=execution_context is not None,
                )

            async def atom_candidates() -> list[AggregatedAtomParent]:
                if self.atom_retriever is None:
                    return []
                started = time.monotonic()
                try:
                    search_parents = getattr(
                        self.atom_retriever, "search_parents", None
                    )
                    if callable(search_parents):
                        return await search_parents(
                            query,
                            k=atom_limit,
                            atom_k=max(30, atom_limit),
                            session_id=session_id,
                            persona_id=persona_id,
                            interaction_type=interaction_type,
                            source_platform=source_platform,
                            source_session_id=source_session_id,
                            relevance_only=True,
                            diagnostics=atom_diagnostics
                            if diagnostics is not None
                            else None,
                            excluded_session_ids=excluded_session_ids,
                            local_only=local_only,
                            extra_queries=clean_sub_queries or None,
                        )
                    # Backwards compatibility for custom AtomRetriever doubles
                    raw_atoms = await self.atom_retriever.search(
                        query,
                        max(30, atom_limit),
                        session_id,
                        persona_id,
                        interaction_type=interaction_type,
                        source_platform=source_platform,
                        source_session_id=source_session_id,
                        relevance_only=True,
                        **(
                            {"excluded_session_ids": tuple(excluded_session_ids)}
                            if excluded_session_ids
                            else {}
                        ),
                    )
                    from .atom_retriever import aggregate_atoms_by_parent

                    return aggregate_atoms_by_parent(raw_atoms, top_k=atom_limit)
                finally:
                    self._record_phase(phase_timings, "atom_route", started)

            def atom_route():
                if self.atom_retriever is None:
                    return asyncio.sleep(0, result=([], None))
                return search_route(
                    "atom",
                    atom_candidates(),
                    timeout,
                    safe_errors=True,
                )

            async def graph_candidates():
                started = time.monotonic()
                try:
                    limit = graph_limit_init
                    while True:
                        rows = await self.graph_retriever.search(
                            query,
                            limit,
                            session_id or source_session_id,
                            persona_id,
                            source_platform=source_platform,
                            interaction_type=interaction_type,
                            source_session_id=source_session_id,
                            execution_context=graph_context,
                            **graph_options,
                        )
                        if narrative_plan is None or not rows:
                            return rows
                        ids = list(dict.fromkeys(r.doc_id for r in rows))

                        async def load_parents():
                            if self.memory_batch_loader is not None:
                                return await self.memory_batch_loader(
                                    metadata_filters={}, ids=ids, limit=len(ids)
                                )
                            return await asyncio.gather(
                                *(self.memory_loader(i) for i in ids)
                            )

                        if execution_context is None:
                            parents = await load_parents()
                        else:
                            outcome = await execution_context.run(
                                "graph_parents",
                                load_parents,
                                discriminator={"ids": ids},
                            )
                            parents = (
                                outcome.get("value", [])
                                if outcome.get("state") == "completed"
                                else []
                            )
                        if self.memory_batch_loader is not None:
                            canonical = {
                                int(r["id"]): r for r in parents if isinstance(r, dict)
                            }
                        else:
                            canonical = {
                                i: r
                                for i, r in zip(ids, parents)
                                if isinstance(r, dict)
                            }
                        candidates = [
                            HybridResult(
                                r.doc_id,
                                r.final_score,
                                r.rrf_score,
                                r.keyword_score,
                                r.vector_score,
                                str(canonical[r.doc_id].get("text") or ""),
                                safe_json_dict(canonical[r.doc_id].get("metadata")),
                                r.score_breakdown,
                            )
                            for r in rows
                            if r.doc_id in canonical
                        ]
                        valid = await narrative_plan.validate(
                            candidates, execution_context, "graph_policy"
                        )
                        if valid:
                            narrative_plan.graph_fallback = list(valid)
                        valid_map = {r.doc_id: r for r in valid}
                        checked = [
                            replace(
                                r,
                                content=valid_map[r.doc_id].content,
                                metadata=valid_map[r.doc_id].metadata,
                            )
                            for r in rows
                            if r.doc_id in valid_map
                        ]
                        if (
                            len(checked) >= narrative_plan.target_k
                            or len(rows) < limit
                            or limit >= narrative_plan.graph_ceiling
                            or not narrative_plan.can_expand(execution_context)
                        ):
                            return checked
                        limit = min(limit * 2, narrative_plan.graph_ceiling)
                finally:
                    self._record_phase(phase_timings, "graph_route", started)

            def graph_route():
                sub_timeout = (
                    min(timeout, self.graph_route_timeout)
                    if self.graph_route_timeout > 0
                    else timeout
                )
                return search_route(
                    "graph",
                    graph_candidates(),
                    sub_timeout,
                    safe_errors=True,
                )

            # Always run all 3 routes concurrently (Document + Atom + Graph)
            (
                (doc_results, doc_error),
                (atom_parents, atom_error),
                (graph_results, graph_error),
            ) = await asyncio.gather(document_route(), atom_route(), graph_route())

            if diagnostics is not None:
                diagnostics["graph_selection"] = "parallel_3route"
                diagnostics["post_fusion_rerank"] = post_fusion_rerank
            merge_route(diagnostics, "document", doc_diagnostics, doc_error)
            if self.atom_retriever is not None:
                merge_route(diagnostics, "atom", atom_diagnostics, atom_error)
            merge_route(diagnostics, "graph", graph_diagnostics, graph_error)
            if (
                diagnostics is not None
                and not post_fusion_rerank
                and doc_diagnostics.get("rerank_rejected_all")
                and (atom_parents or graph_results)
            ):
                diagnostics.pop("rerank_rejected_all", None)
        finally:
            if token is not None:
                query_embeddings.reset(token)
            if owns_cache:
                for task in cache.values():
                    if not task.done():
                        task.cancel()
                if cache:
                    await asyncio.gather(*cache.values(), return_exceptions=True)

        # Hydrate canonical parent documents for atom-only and graph-only hits in one batch
        doc_map: dict[int, HybridResult] = {
            int(item.doc_id): item for item in (doc_results or [])
        }
        atom_map: dict[int, AggregatedAtomParent] = {
            int(parent.parent_memory_id): parent for parent in (atom_parents or [])
        }
        graph_map: dict[int, Any] = {
            int(item.doc_id): item for item in (graph_results or [])
        }

        all_doc_ids = set(doc_map) | set(atom_map) | set(graph_map)
        if not all_doc_ids:
            return []

        policy_active = bool(
            interaction_type is not None
            or source_platform is not None
            or source_session_id is not None
            or excluded_session_ids
            or session_id is not None
            or persona_id is not None
        )

        # Hydrate any doc_id missing full text/metadata OR coming only from graph/atom
        need_hydration = [
            doc_id
            for doc_id in all_doc_ids
            if (
                doc_id not in doc_map
                or not doc_map[doc_id].content
                or not doc_map[doc_id].metadata
            )
            or (policy_active and doc_id in graph_map and doc_id not in doc_map)
        ]
        loaded: dict[int, dict[str, Any]] = {}
        if narrative_plan is not None:
            for r in graph_results or []:
                if getattr(r, "content", None) and getattr(r, "metadata", None):
                    loaded[int(r.doc_id)] = {
                        "id": int(r.doc_id),
                        "text": r.content,
                        "metadata": dict(r.metadata),
                    }
        missing_ids = [doc_id for doc_id in need_hydration if doc_id not in loaded]
        if missing_ids:
            loaded.update(await self._hydrate_documents(missing_ids, execution_context))

        # Validate hydrated docs against active status & scope policy
        def _is_valid_parent(doc_id: int, metadata: dict[str, Any]) -> bool:
            if str(metadata.get("status") or "active") != "active":
                return False
            if doc_id not in doc_map:
                if session_id is not None and metadata.get("session_id") != session_id:
                    return False
                if persona_id is not None and metadata.get("persona_id") != persona_id:
                    return False
                if (
                    interaction_type is not None
                    or source_platform is not None
                    or source_session_id is not None
                    or excluded_session_ids
                ) and not memory_metadata_matches_policy(
                    metadata,
                    interaction_type=interaction_type,
                    source_platform=source_platform,
                    source_session_id=source_session_id,
                    excluded_session_ids=excluded_session_ids,
                ):
                    return False
            return True

        # Cross-route RRF pre-ranking across Document + Atom + Graph
        rrf_k = int(self.config.get("rrf_k", 60) or 60)
        doc_rank = {
            int(item.doc_id): rank for rank, item in enumerate(doc_results or [])
        }
        atom_rank = {
            int(item.parent_memory_id): rank
            for rank, item in enumerate(atom_parents or [])
        }
        graph_rank = {
            int(item.doc_id): rank for rank, item in enumerate(graph_results or [])
        }

        document_weight, graph_weight, intent = self._route_weights_for_query(query)
        atom_weight = self.atom_route_weight
        document_max = (
            max((item.final_score for item in doc_results), default=1.0) or 1.0
        )
        atom_max = max((item.score for item in atom_parents), default=1.0) or 1.0
        graph_max = (
            max((item.final_score for item in graph_results), default=1.0) or 1.0
        )

        merged_candidates: list[HybridResult] = []
        routes_hit_by_id: dict[int, set[str]] = {}

        for doc_id in all_doc_ids:
            doc_res = doc_map.get(doc_id)
            atom_res = atom_map.get(doc_id)
            graph_res = graph_map.get(doc_id)

            memory_content = doc_res.content if doc_res is not None else ""
            memory_metadata = (
                dict(doc_res.metadata)
                if doc_res is not None and isinstance(doc_res.metadata, dict)
                else {}
            )
            if not memory_content or not memory_metadata:
                hydrated = loaded.get(doc_id)
                if not hydrated:
                    continue
                memory_content = str(hydrated.get("text") or memory_content)
                raw_meta = safe_json_dict(hydrated.get("metadata"))
                if raw_meta:
                    memory_metadata = raw_meta
            if not memory_content:
                continue
            if not _is_valid_parent(doc_id, memory_metadata):
                continue

            routes_hit: set[str] = set()
            pre_rrf = 0.0
            if doc_res is not None:
                routes_hit.add("document")
                pre_rrf += 1.0 / (rrf_k + doc_rank[doc_id] + 1)
            if atom_res is not None:
                routes_hit.add("atom")
                pre_rrf += 1.0 / (rrf_k + atom_rank[doc_id] + 1)
                memory_metadata["matched_atoms"] = list(atom_res.matched_atoms)
                memory_metadata["retrieved_atom_ids"] = list(
                    atom_res.retrieved_atom_ids
                )
            if graph_res is not None:
                routes_hit.add("graph")
                pre_rrf += 1.0 / (rrf_k + graph_rank[doc_id] + 1)
            routes_hit_by_id[doc_id] = routes_hit

            doc_signal = (
                doc_res.final_score / document_max if doc_res is not None else 0.0
            )
            atom_signal = atom_res.score / atom_max if atom_res is not None else 0.0
            graph_signal = (
                graph_res.final_score / graph_max if graph_res is not None else 0.0
            )
            multi_bonus = self.cross_route_bonus if len(routes_hit) >= 2 else 0.0
            total_w = (
                document_weight
                + (atom_weight if atom_map else 0.0)
                + (graph_weight if graph_map else 0.0)
            )
            fused_fallback_score = min(
                1.0,
                (
                    document_weight * doc_signal
                    + (atom_weight * atom_signal if atom_map else 0.0)
                    + (graph_weight * graph_signal if graph_map else 0.0)
                )
                / max(total_w, 1e-6)
                + multi_bonus,
            )

            score_breakdown: dict[str, Any] = {}
            if doc_res and doc_res.score_breakdown:
                score_breakdown.update(doc_res.score_breakdown)
            if graph_res and graph_res.score_breakdown:
                score_breakdown.update(graph_res.score_breakdown)
            score_breakdown.update(
                {
                    "document_route_score": round(doc_signal, 4),
                    "document_route_weight": round(document_weight, 4),
                    "atom_route_score": round(atom_signal, 4),
                    "graph_route_score": round(graph_signal, 4),
                    "graph_route_weight": round(graph_weight, 4),
                    "pre_fusion_rrf": round(pre_rrf, 6),
                    "routes_hit_count": len(routes_hit),
                    "dual_route_final_score": round(fused_fallback_score, 4),
                }
            )
            if intent:
                score_breakdown["query_intent"] = intent
            if relevance_only:
                score_breakdown["agentic_relevance"] = fused_fallback_score
            if doc_res is not None:
                score_breakdown["document_keyword_score"] = round(
                    float(doc_res.bm25_score or 0.0), 4
                )
                score_breakdown["document_vector_score"] = round(
                    float(doc_res.vector_score or 0.0), 4
                )
            if atom_res is not None and atom_res.best_vector_score is not None:
                score_breakdown["atom_vector_score"] = round(
                    float(atom_res.best_vector_score), 4
                )
            if graph_res is not None:
                score_breakdown["graph_keyword_score"] = round(
                    float(graph_res.keyword_score or 0.0), 4
                )
                score_breakdown["graph_vector_score"] = round(
                    float(graph_res.vector_score or 0.0), 4
                )

            merged_candidates.append(
                HybridResult(
                    doc_id=doc_id,
                    final_score=fused_fallback_score,
                    rrf_score=pre_rrf,
                    bm25_score=doc_res.bm25_score if doc_res is not None else None,
                    vector_score=(
                        doc_res.vector_score
                        if doc_res is not None
                        else (
                            atom_res.best_vector_score if atom_res is not None else None
                        )
                    ),
                    content=memory_content,
                    metadata=memory_metadata,
                    score_breakdown=score_breakdown,
                )
            )

        if not merged_candidates:
            return []

        # Order by cross-route RRF (then fused fallback score) before cutting pool
        merged_candidates.sort(
            key=lambda item: (item.rrf_score, item.final_score, -item.doc_id),
            reverse=True,
        )

        # Resolve temporal intent (explicit from rewriter or lightweight fallback regex)
        effective_temporal_intent = temporal_intent
        effective_time_hint_days = time_hint_days
        if effective_temporal_intent is None:
            effective_temporal_intent, detected_days = detect_temporal_intent(query)
            if effective_time_hint_days is None:
                effective_time_hint_days = detected_days

        effective_session_id = current_session_id or session_id or source_session_id

        if post_fusion_rerank:
            rerank_timeout = self.document_retriever.rerank_timeout_seconds
            budget_exhausted = False
            if execution_context is not None:
                remaining_budget = execution_context.budget.search_remaining()
                remote_dl = getattr(execution_context, "remote_deadline", None)
                if remote_dl is not None:
                    remaining_budget = min(
                        remaining_budget, float(remote_dl) - time.monotonic()
                    )
                if remaining_budget <= 0:
                    budget_exhausted = True
                else:
                    rerank_timeout = min(rerank_timeout, remaining_budget)

            if not budget_exhausted:
                resolver = getattr(
                    self.document_retriever, "rerank_provider_resolver", None
                )
                resolved_provider = None
                if callable(resolver):
                    try:
                        resolved_provider = resolver()
                    except Exception:
                        resolved_provider = None
                provider_id = str(
                    self.config.get("rerank_provider_id")
                    or getattr(self.document_retriever, "config", {}).get(
                        "rerank_provider_id", ""
                    )
                    or ""
                )
                eff_rerank_candidates = resolve_effective_rerank_candidates(
                    self.rerank_candidates,
                    provider=resolved_provider,
                    provider_id=provider_id,
                    context=getattr(
                        self,
                        "context",
                        getattr(self.document_retriever, "context", None),
                    ),
                )
                pool_size = max(k, min(eff_rerank_candidates, len(merged_candidates)))
                rerank_pool = merged_candidates[:pool_size]
                rerank_started = time.monotonic()
                rerank_timings = (
                    execution_context.route_elapsed_ms
                    if execution_context is not None
                    else None
                )
                (
                    reranked,
                    rerank_meta,
                ) = await self.document_retriever.score_rerank_candidates(
                    query,
                    rerank_pool,
                    pool_size,
                    timings=rerank_timings,
                    compact=True,
                    apply_threshold=True,
                    confidence_target_k=k,
                    timeout_seconds=rerank_timeout,
                )
                self._record_phase(phase_timings, "rerank", rerank_started)
            else:
                reranked, rerank_meta = (
                    [],
                    {"reranked": False, "budget_exhausted": True},
                )
            if diagnostics is not None:
                diagnostics["rerank"] = dict(rerank_meta)
                if rerank_meta.get("rejected_all"):
                    diagnostics["rerank_rejected_all"] = True

            if rerank_meta.get("reranked"):
                if not reranked:
                    # Absolute threshold hard-rejected all candidates -> zero noise!
                    return []

                if relevance_only:
                    for item in reranked:
                        prob = float(
                            (item.score_breakdown or {}).get("rerank", item.final_score)
                        )
                        item.final_score = prob
                        if isinstance(item.score_breakdown, dict):
                            item.score_breakdown["agentic_relevance"] = prob
                    reranked.sort(
                        key=lambda item: (item.final_score, -item.doc_id), reverse=True
                    )
                    effective_k = min(k, int(rerank_meta.get("effective_k", k)))
                    return reranked[:effective_k]

                # Post-rerank gentle tie-break: importance + affinity + intent-aware temporal
                now = time.time()
                self._apply_post_rerank_scoring(
                    reranked,
                    routes_hit_by_id=routes_hit_by_id,
                    current_session_id=effective_session_id,
                    current_sender_id=current_sender_id,
                    current_sender_name=current_sender_name,
                    temporal_intent=effective_temporal_intent,
                    time_hint_days=effective_time_hint_days,
                    now=now,
                )
                reranked.sort(
                    key=lambda item: (item.final_score, -item.doc_id), reverse=True
                )
                effective_k = min(k, int(rerank_meta.get("effective_k", k)))
                if effective_k <= 0:
                    return []
                mmr_fn = getattr(self.document_retriever, "_apply_mmr", None)
                if len(reranked) > 1 and callable(mmr_fn):
                    final_results = await asyncio.to_thread(
                        mmr_fn, reranked, effective_k
                    )
                else:
                    final_results = reranked[:effective_k]
                if diagnostics is not None and len(merged_candidates) > effective_k:
                    diagnostics["candidate_limited"] = True
                return final_results

        # Fallback path when rerank is disabled or unavailable (no graph presence floor!)
        if not relevance_only:
            now = time.time()
            self._apply_post_rerank_scoring(
                merged_candidates,
                routes_hit_by_id=routes_hit_by_id,
                current_session_id=effective_session_id,
                current_sender_id=current_sender_id,
                current_sender_name=current_sender_name,
                temporal_intent=effective_temporal_intent,
                time_hint_days=effective_time_hint_days,
                now=now,
                use_rerank_base=False,
            )
        merged_candidates.sort(
            key=lambda item: (item.final_score, -item.doc_id), reverse=True
        )
        mmr_fn = getattr(self.document_retriever, "_apply_mmr", None)
        if not relevance_only and len(merged_candidates) > 1 and callable(mmr_fn):
            final_results = await asyncio.to_thread(mmr_fn, merged_candidates, k)
        else:
            final_results = merged_candidates[:k]
        if diagnostics is not None and len(merged_candidates) > k:
            diagnostics["candidate_limited"] = True
        return final_results

    def _apply_post_rerank_scoring(
        self,
        candidates: list[HybridResult],
        *,
        routes_hit_by_id: dict[int, set[str]],
        current_session_id: str | None,
        current_sender_id: str | None,
        current_sender_name: str | None,
        temporal_intent: str,
        time_hint_days: int | None,
        now: float,
        use_rerank_base: bool = True,
    ) -> None:
        """Apply post-rerank gentle affinity and intent-aware temporal tie-breaks.

        Formula (per approved design plan):
            S_final = 0.85 * p_rerank + 0.05 * importance + B_affinity + B_temporal
        where:
            B_affinity = same_session(+0.04) + same_speaker(+0.03) + multi_route(+0.03)
            B_temporal = +0.10 * exp(-0.08 * days_old) ONLY when temporal_intent == "recent"
                         (0.0 otherwise — NEVER penalizes timeless facts!).
        """
        norm_session = str(current_session_id or "").strip()
        norm_sender_id = str(current_sender_id or "").strip()
        norm_sender_name = str(current_sender_name or "").strip().casefold()

        for item in candidates:
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            breakdown = (
                item.score_breakdown if isinstance(item.score_breakdown, dict) else {}
            )
            base_prob = (
                float(breakdown.get("rerank", item.final_score))
                if use_rerank_base
                else float(item.final_score)
            )
            importance = clamp_float(metadata.get("importance"), default=0.5)

            # 1. Session affinity (+0.04)
            session_bonus = 0.0
            if norm_session and (
                str(metadata.get("session_id") or "").strip() == norm_session
                or str(metadata.get("source_session_id") or "").strip() == norm_session
            ):
                session_bonus = self.affinity_session_bonus

            # 2. Speaker affinity (+0.03)
            speaker_bonus = 0.0
            if (norm_sender_id or norm_sender_name) and self._matches_speaker(
                metadata, norm_sender_id, norm_sender_name
            ):
                speaker_bonus = self.affinity_speaker_bonus

            # 3. Multi-route consensus (+0.03)
            routes_hit = routes_hit_by_id.get(item.doc_id, set())
            multi_route_bonus = (
                self.affinity_multi_route_bonus if len(routes_hit) >= 2 else 0.0
            )

            affinity_bonus = session_bonus + speaker_bonus + multi_route_bonus

            # 4. Intent-aware temporal boost (ONLY for temporal_intent == "recent")
            temporal_bonus = 0.0
            days_old = self._candidate_days_old(metadata, now)
            if temporal_intent == "recent" and days_old is not None:
                temporal_bonus = self.temporal_recent_bonus * math.exp(
                    -0.08 * max(0.0, days_old)
                )

            final_score = (
                0.85 * base_prob + 0.05 * importance + affinity_bonus + temporal_bonus
            )
            item.final_score = round(final_score, 6)
            breakdown.update(
                {
                    "importance": round(importance, 4),
                    "affinity_session_bonus": round(session_bonus, 4),
                    "affinity_speaker_bonus": round(speaker_bonus, 4),
                    "affinity_multi_route_bonus": round(multi_route_bonus, 4),
                    "temporal_bonus": round(temporal_bonus, 4),
                    "temporal_intent": temporal_intent,
                    "final_score": round(final_score, 4),
                }
            )
            if days_old is not None:
                breakdown["days_old"] = round(days_old, 2)
            item.score_breakdown = breakdown

    @staticmethod
    def _matches_speaker(
        metadata: dict[str, Any], sender_id: str, sender_name_lower: str
    ) -> bool:
        if sender_id:
            if str(metadata.get("sender_id") or "").strip() == sender_id:
                return True
            identities = metadata.get("participant_identities")
            if isinstance(identities, list):
                for ident in identities:
                    if (
                        isinstance(ident, dict)
                        and str(ident.get("sender_id") or "").strip() == sender_id
                    ):
                        return True
        if sender_name_lower:
            participants = metadata.get("participants")
            if isinstance(participants, list):
                for p in participants:
                    if str(p or "").strip().casefold() == sender_name_lower:
                        return True
            identities = metadata.get("participant_identities")
            if isinstance(identities, list):
                for ident in identities:
                    if isinstance(ident, dict) and (
                        str(ident.get("display_name") or "").strip().casefold()
                        == sender_name_lower
                    ):
                        return True
            matched_atoms = metadata.get("matched_atoms")
            if isinstance(matched_atoms, list):
                for atom in matched_atoms:
                    if not isinstance(atom, dict):
                        continue
                    entities = atom.get("entities") or []
                    if any(
                        str(e or "").strip().casefold() == sender_name_lower
                        for e in entities
                    ):
                        return True
                    if sender_name_lower in str(atom.get("content") or "").casefold():
                        return True
        return False

    @staticmethod
    def _candidate_days_old(metadata: dict[str, Any], now: float) -> float | None:
        timestamps: list[float] = []
        matched_atoms = metadata.get("matched_atoms")
        if isinstance(matched_atoms, list):
            for atom in matched_atoms:
                if isinstance(atom, dict):
                    for key in ("event_time", "created_at"):
                        ts = safe_float(atom.get(key), 0.0)
                        if ts > 0:
                            timestamps.append(ts)
        for key in ("create_time", "updated_at"):
            ts = safe_float(metadata.get(key), 0.0)
            if ts > 0:
                timestamps.append(ts)
        if not timestamps:
            return None
        reference = max(timestamps)
        return max(0.0, (now - reference) / 86400.0)

    def _merge_multi_query_docs(
        self, batches: Sequence[Any], limit: int
    ) -> list[HybridResult]:
        """Merge primary + sub-query document results via weighted RRF."""
        rrf_k = int(self.config.get("rrf_k", 60) or 60)
        by_id: dict[int, HybridResult] = {}
        rrf_scores: dict[int, float] = {}
        for q_idx, batch in enumerate(batches):
            if isinstance(batch, BaseException) or not batch:
                continue
            q_weight = 1.0 if q_idx == 0 else 0.7
            for rank, item in enumerate(batch):
                doc_id = int(item.doc_id)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + q_weight / (
                    rrf_k + rank + 1
                )
                prev = by_id.get(doc_id)
                if prev is None or item.final_score > prev.final_score:
                    by_id[doc_id] = item
        ordered_ids = sorted(
            by_id,
            key=lambda did: (rrf_scores[did], by_id[did].final_score, -did),
            reverse=True,
        )
        return [by_id[did] for did in ordered_ids[:limit]]

    async def _hydrate_documents(
        self, missing_ids: list[int], execution_context: Any
    ) -> dict[int, dict[str, Any]]:
        if not missing_ids:
            return {}
        if self.memory_batch_loader is not None:
            if execution_context is None:
                documents = await self.memory_batch_loader(
                    metadata_filters={}, ids=missing_ids, limit=len(missing_ids)
                )
            else:
                outcome = await execution_context.run(
                    "document_hydration",
                    self.memory_batch_loader,
                    metadata_filters={},
                    ids=missing_ids,
                    limit=len(missing_ids),
                    discriminator={"document_ids": missing_ids},
                )
                documents = (
                    outcome.get("value", [])
                    if outcome.get("state") == "completed"
                    else []
                )
            return {
                int(document["id"]): document
                for document in (documents or ())
                if isinstance(document, dict) and document.get("id") is not None
            }

        async def load_missing():
            return await asyncio.gather(
                *(self.memory_loader(doc_id) for doc_id in missing_ids)
            )

        if execution_context is None:
            documents = await load_missing()
        else:
            outcome = await execution_context.run(
                "document_hydration",
                load_missing,
                discriminator={"document_ids": missing_ids},
            )
            documents = (
                outcome.get("value", []) if outcome.get("state") == "completed" else []
            )
        return {
            int(doc_id): doc
            for doc_id, doc in zip(missing_ids, documents or ())
            if isinstance(doc, dict)
        }

    def _route_weights_for_query(self, query: str) -> tuple[float, float, str]:
        """Adjust document/graph weights with lightweight query intent rules."""
        base_document = self.document_route_weight
        base_graph = self.graph_route_weight
        if not self.dynamic_route_weighting:
            return base_document, base_graph, "fixed"

        normalized = query.casefold()
        relation_terms = (
            "谁",
            "和谁",
            "关系",
            "认识",
            "朋友",
            "同事",
            "家人",
            "父母",
            "妈妈",
            "爸爸",
            "老师",
            "同学",
            "partner",
            "friend",
            "relationship",
            "with whom",
        )
        temporal_terms = (
            "上次",
            "昨天",
            "前天",
            "刚才",
            "之前",
            "什么时候",
            "哪天",
            "最近",
            "last time",
            "yesterday",
            "recently",
            "when",
        )
        factual_terms = (
            "是什么",
            "什么是",
            "解释",
            "定义",
            "怎么",
            "如何",
            "why",
            "what is",
            "explain",
            "define",
            "how to",
        )

        relation_hit = any(term in normalized for term in relation_terms)
        temporal_hit = any(term in normalized for term in temporal_terms)
        factual_hit = any(term in normalized for term in factual_terms)

        document_weight = base_document
        graph_weight = base_graph
        intent = "default"

        if relation_hit:
            graph_weight += 0.2
            document_weight -= 0.2
            intent = "relationship"
        if temporal_hit:
            graph_weight += 0.1
            document_weight -= 0.1
            intent = "temporal" if intent == "default" else f"{intent}+temporal"
        if factual_hit and not relation_hit:
            document_weight += 0.15
            graph_weight -= 0.15
            intent = "factual" if intent == "default" else f"{intent}+factual"

        document_weight = max(0.15, min(0.9, document_weight))
        graph_weight = max(0.1, min(0.85, graph_weight))
        total = document_weight + graph_weight
        if total <= 0:
            return base_document, base_graph, "fixed"
        return document_weight / total, graph_weight / total, intent


__all__ = ["DualRouteRetriever"]
