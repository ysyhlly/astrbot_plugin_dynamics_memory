"""Fuse document-route and graph-route retrieval into one result list."""

from __future__ import annotations

import re
from dataclasses import replace
from ..memory_scope import memory_metadata_matches_policy
from ..utils.json_utils import safe_json_dict


import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .graph_retriever import GraphRetriever
from .hybrid_retriever import HybridResult, HybridRetriever
from .recall_diagnostics import merge_route
from .route_execution import search_route
from .vector_search import query_embeddings


class DualRouteRetriever:
    """Coordinate document and graph retrieval routes."""

    def __init__(
        self,
        document_retriever: HybridRetriever,
        graph_retriever: GraphRetriever,
        memory_loader: Callable[[int], Awaitable[dict[str, Any] | None]],
        config: dict[str, Any] | None = None,
        memory_batch_loader=None,
    ):
        self.document_retriever = document_retriever
        self.graph_retriever = graph_retriever
        self.memory_loader = memory_loader
        self.memory_batch_loader = memory_batch_loader
        self.config = config or {}
        self.document_route_weight = float(
            self.config.get("document_route_weight", 0.65)
        )
        self.graph_route_weight = float(self.config.get("graph_route_weight", 0.35))
        self.cross_route_bonus = float(self.config.get("cross_route_bonus", 0.08))
        self.dynamic_route_weighting = bool(
            self.config.get("dynamic_route_weighting", True)
        )

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        persona_id: str | None = None,
        interaction_type: str | None = None,
        source_platform: str | None = None,
        source_session_id: str | None = None,
        execution_context=None,
        diagnostics: dict[str, Any] | None = None,
        relevance_only: bool = False,
        narrative_plan=None,
    ) -> list[HybridResult]:
        """Run both retrieval routes and merge their memory candidates."""
        timeout = (
            execution_context.budget.remaining()
            if execution_context is not None
            else self.config.get("retrieval_timeout_seconds", 10.0) * 2 + 1
        )
        graph_context = execution_context
        if getattr(execution_context, "remote_deadline", None) is not None:
            budget = execution_context.budget
            reserve = min(0.2, budget.search_remaining() * 0.20)
            graph_context = execution_context.with_deadline(
                budget.search_deadline - reserve
            )
        outer_cache = (
            query_embeddings.get()
            if relevance_only or execution_context is not None
            else None
        )
        cache = outer_cache if outer_cache is not None else {}
        token = query_embeddings.set(cache) if outer_cache is None else None
        doc_diagnostics: dict[str, Any] = {}
        graph_diagnostics: dict[str, Any] = {}
        # 旧调用不增加下游关键字参数，保留宿主和已有替身的调用契约。
        doc_options = (
            {"diagnostics": doc_diagnostics} if diagnostics is not None else {}
        )
        graph_options = (
            {"diagnostics": graph_diagnostics} if diagnostics is not None else {}
        )
        if relevance_only:
            doc_options["relevance_only"] = True
            graph_options["relevance_only"] = True
        if narrative_plan is not None:
            doc_options["narrative_plan"] = narrative_plan
        try:

            def document_route():
                return search_route(
                    "document",
                    self.document_retriever.search(
                        query,
                        narrative_plan.initial
                        if narrative_plan is not None
                        else max(k * 2, k),
                        session_id,
                        persona_id,
                        interaction_type=interaction_type,
                        source_platform=source_platform,
                        source_session_id=source_session_id,
                        execution_context=execution_context,
                        **doc_options,
                    ),
                    timeout,
                    safe_errors=execution_context is not None,
                )

            async def graph_candidates():
                limit = (
                    narrative_plan.initial
                    if narrative_plan is not None
                    else max(k * 2, k)
                )
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
                            "graph_parents", load_parents, discriminator={"ids": ids}
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
                            i: r for i, r in zip(ids, parents) if isinstance(r, dict)
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

            def graph_route():
                return search_route(
                    "graph",
                    graph_candidates(),
                    timeout,
                    safe_errors=execution_context is not None,
                )

            graph_skipped = False
            if self._document_first_query(query):
                started = asyncio.get_running_loop().time()
                overall_timeout = timeout
                document_task = asyncio.create_task(document_route())
                ready_task = (
                    asyncio.create_task(narrative_plan.documents_ready.wait())
                    if narrative_plan is not None
                    else None
                )
                try:
                    waiting = (
                        {document_task, ready_task}
                        if ready_task is not None
                        else {document_task}
                    )
                    done, _ = await asyncio.wait(
                        waiting,
                        timeout=overall_timeout * 0.65,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if (
                        narrative_plan is not None
                        and narrative_plan.documents_ready.is_set()
                    ):
                        graph_skipped = narrative_plan.documents_sufficient()
                    elif document_task in done:
                        doc_results, doc_error = document_task.result()
                        graph_skipped = (
                            narrative_plan is None
                            and doc_error is None
                            and len(doc_results) >= k > 0
                            and all(
                                item.final_score > 0
                                and item.content
                                and float(item.bm25_score or 0) > 0
                                for item in doc_results[:k]
                            )
                        )
                    if graph_skipped:
                        doc_results, doc_error = await document_task
                        graph_results, graph_error = [], None
                        # An error after the early signal must still allow rescue.
                        if doc_error is not None or not doc_results:
                            graph_skipped = False
                            timeout = max(
                                0.0,
                                overall_timeout
                                - (asyncio.get_running_loop().time() - started),
                            )
                            graph_results, graph_error = await graph_route()
                    else:
                        timeout = max(
                            0.0,
                            overall_timeout
                            - (asyncio.get_running_loop().time() - started),
                        )
                        (
                            (doc_results, doc_error),
                            (graph_results, graph_error),
                        ) = await asyncio.gather(document_task, graph_route())
                finally:
                    owned = [t for t in (document_task, ready_task) if t is not None]
                    for task in owned:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*owned, return_exceptions=True)
            else:
                (
                    (doc_results, doc_error),
                    (graph_results, graph_error),
                ) = await asyncio.gather(document_route(), graph_route())
            if diagnostics is not None:
                diagnostics["graph_selection"] = (
                    "skipped_simple_preference"
                    if graph_skipped
                    else "fallback"
                    if self._document_first_query(query)
                    else "parallel"
                )
            merge_route(diagnostics, "document", doc_diagnostics, doc_error)
            if not graph_skipped:
                merge_route(diagnostics, "graph", graph_diagnostics, graph_error)
        finally:
            if token is not None:
                query_embeddings.reset(token)
                for task in cache.values():
                    if not task.done():
                        task.cancel()
                if cache:
                    await asyncio.gather(*cache.values(), return_exceptions=True)

        preloaded_graph_docs: dict[int, dict[str, Any]] = {}
        if (
            interaction_type is not None
            or source_platform is not None
            or source_session_id is not None
        ) and graph_results:
            checked = []
            # Graph index metadata is derived and may be stale. Only the
            # canonical parent document can grant cross-scope visibility.
            if callable(self.memory_batch_loader):
                graph_ids = list(
                    dict.fromkeys(int(result.doc_id) for result in graph_results)
                )
                if execution_context is None:
                    loaded_docs = await self.memory_batch_loader(
                        metadata_filters={}, ids=graph_ids, limit=len(graph_ids)
                    )
                else:
                    outcome = await execution_context.run(
                        "graph_policy_hydration",
                        self.memory_batch_loader,
                        metadata_filters={},
                        ids=graph_ids,
                        limit=len(graph_ids),
                        discriminator={"document_ids": graph_ids},
                    )
                    loaded_docs = (
                        outcome.get("value", [])
                        if outcome.get("state") == "completed"
                        else []
                    )
                memories_by_id: dict[int, Any] = {}
                for doc in loaded_docs or ():
                    doc_key = (
                        doc.get("id")
                        if isinstance(doc, dict)
                        else getattr(doc, "id", None)
                    )
                    if doc_key is not None:
                        memories_by_id[int(doc_key)] = doc
            else:
                memories_by_id = {}
                for result in graph_results:
                    if execution_context is None:
                        memory = await self.memory_loader(result.doc_id)
                    else:
                        outcome = await execution_context.run(
                            "graph_policy_hydration",
                            self.memory_loader,
                            result.doc_id,
                            discriminator={"document_id": int(result.doc_id)},
                        )
                        memory = (
                            outcome.get("value")
                            if outcome.get("state") == "completed"
                            else None
                        )
                    memories_by_id[int(result.doc_id)] = memory

            for result in graph_results:
                doc_id = int(result.doc_id)
                memory = memories_by_id.get(doc_id)
                if isinstance(memory, dict):
                    metadata = safe_json_dict(memory.get("metadata"))
                    text_val = memory.get("text")
                    if text_val is not None:
                        preloaded_graph_docs[doc_id] = {
                            "id": doc_id,
                            "text": str(text_val),
                            "metadata": dict(metadata),
                        }
                elif memory is not None:
                    metadata = safe_json_dict(getattr(memory, "metadata", None))
                else:
                    metadata = {}
                if memory_metadata_matches_policy(
                    metadata,
                    interaction_type=interaction_type,
                    source_platform=source_platform,
                    source_session_id=source_session_id,
                ):
                    result.metadata = dict(metadata)
                    checked.append(result)
            graph_results = checked

        if not graph_results:
            if diagnostics is not None and len(doc_results) > k:
                diagnostics["candidate_limited"] = True
            return doc_results[:k]

        document_weight, graph_weight, intent = self._route_weights_for_query(query)

        document_max = (
            max((item.final_score for item in doc_results), default=1.0) or 1.0
        )
        graph_max = (
            max((item.final_score for item in graph_results), default=1.0) or 1.0
        )

        doc_map = {item.doc_id: item for item in doc_results}
        graph_map = {item.doc_id: item for item in graph_results}
        all_doc_ids = set(doc_map) | set(graph_map)

        preloaded = dict(preloaded_graph_docs)
        if narrative_plan is not None:
            preloaded.update(
                {
                    r.doc_id: {
                        "id": r.doc_id,
                        "text": r.content,
                        "metadata": r.metadata,
                    }
                    for r in graph_results
                }
            )
        missing_ids = [
            doc_id
            for doc_id in all_doc_ids
            if doc_id not in preloaded
            and (
                doc_id not in doc_map
                or not doc_map[doc_id].content
                or not doc_map[doc_id].metadata
            )
        ]
        loaded = dict(preloaded)
        if missing_ids:
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
                loaded.update({int(document["id"]): document for document in documents})
            else:

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
                        outcome.get("value", [])
                        if outcome.get("state") == "completed"
                        else []
                    )
                loaded.update(dict(zip(missing_ids, documents)))

        merged_results: list[HybridResult] = []
        for doc_id in all_doc_ids:
            doc_result = doc_map.get(doc_id)
            graph_result = graph_map.get(doc_id)

            doc_signal = (
                doc_result.final_score / document_max if doc_result is not None else 0.0
            )
            graph_signal = (
                graph_result.final_score / graph_max
                if graph_result is not None
                else 0.0
            )
            route_bonus = (
                self.cross_route_bonus
                if doc_result is not None and graph_result is not None
                else 0.0
            )

            memory_content = doc_result.content if doc_result is not None else ""
            memory_metadata = (
                dict(doc_result.metadata)
                if doc_result is not None and isinstance(doc_result.metadata, dict)
                else {}
            )

            if not memory_content or not memory_metadata:
                memory = loaded.get(doc_id)
                if not memory:
                    continue
                memory_content = str(memory.get("text") or memory_content)
                raw_metadata = memory.get("metadata") or memory_metadata
                memory_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

            final_score = min(
                1.0,
                document_weight * doc_signal
                + graph_weight * graph_signal
                + route_bonus,
            )

            score_breakdown: dict[str, float] = {}
            if doc_result and doc_result.score_breakdown:
                score_breakdown.update(doc_result.score_breakdown)
            if graph_result and graph_result.score_breakdown:
                score_breakdown.update(graph_result.score_breakdown)
            score_breakdown.update(
                {
                    "document_route_score": round(doc_signal, 4),
                    "graph_route_score": round(graph_signal, 4),
                    "document_route_weight": round(document_weight, 4),
                    "graph_route_weight": round(graph_weight, 4),
                    "cross_route_bonus": round(route_bonus, 4),
                    "dual_route_final_score": round(final_score, 4),
                }
            )
            if intent:
                score_breakdown["query_intent"] = intent
            if relevance_only:
                score_breakdown["agentic_relevance"] = final_score
            if doc_result is not None:
                score_breakdown["document_keyword_score"] = round(
                    float(doc_result.bm25_score or 0.0),
                    4,
                )
                score_breakdown["document_vector_score"] = round(
                    float(doc_result.vector_score or 0.0),
                    4,
                )
            if graph_result is not None:
                score_breakdown["graph_keyword_score"] = round(
                    float(graph_result.keyword_score or 0.0),
                    4,
                )
                score_breakdown["graph_vector_score"] = round(
                    float(graph_result.vector_score or 0.0),
                    4,
                )

            merged_results.append(
                HybridResult(
                    doc_id=doc_id,
                    final_score=final_score,
                    rrf_score=max(
                        doc_result.rrf_score if doc_result is not None else 0.0,
                        graph_result.rrf_score if graph_result is not None else 0.0,
                    ),
                    bm25_score=doc_result.bm25_score
                    if doc_result is not None
                    else None,
                    vector_score=(
                        doc_result.vector_score if doc_result is not None else None
                    ),
                    content=memory_content,
                    metadata=memory_metadata,
                    score_breakdown=score_breakdown,
                )
            )

        merged_results.sort(
            key=lambda item: (item.final_score, -item.doc_id), reverse=True
        )
        final_results = self._apply_graph_presence_floor(
            merged_results,
            doc_map=doc_map,
            graph_map=graph_map,
            k=k,
            diagnostics=diagnostics,
        )
        if diagnostics is not None and len(merged_results) > k:
            diagnostics["candidate_limited"] = True
        return final_results

    # 图路权重上限是 graph_route_weight（默认 0.35），文档路任一条
    # signal > 0.65 即可压过所有 graph-only 结果；k>=5 时保底保留图路配额。
    _GRAPH_PRESENCE_MIN_K = 5

    def _apply_graph_presence_floor(
        self,
        merged_results: list[HybridResult],
        *,
        doc_map: dict[int, HybridResult],
        graph_map: dict[int, Any],
        k: int,
        diagnostics: dict[str, Any] | None,
    ) -> list[HybridResult]:
        """Keep a floor of graph-exclusive results in the fused top-k.

        Document-route signals are normalized by the document-route maximum, so a
        single document hit with signal above ``1 - graph_weight`` outranks every
        graph-only hit (graph route max contributes at most ``graph_weight``).
        When ``k >= 5`` at least ``max(1, k // 5)`` graph-exclusive docs are kept
        by replacing fused tail entries: the head keeps its fused order, injected
        results stay at the tail in their own score order.

        Args:
            merged_results: Fused results sorted by final score (descending).
            doc_map: Document-route results keyed by doc id.
            graph_map: Graph-route results keyed by doc id.
            k: Number of results the caller asked for.
            diagnostics: Optional per-request diagnostics dict.

        Returns:
            The fused list trimmed to ``k`` results with the graph floor applied.
        """
        if k < self._GRAPH_PRESENCE_MIN_K:
            return merged_results[:k]
        graph_only_ids = {doc_id for doc_id in graph_map if doc_id not in doc_map}
        if not graph_only_ids:
            return merged_results[:k]

        budget = max(1, k // 5)
        head = merged_results[: max(0, k - budget)]
        head_ids = {result.doc_id for result in head}
        kept_graph_only = sum(1 for result in head if result.doc_id in graph_only_ids)
        missing = budget - kept_graph_only
        if missing <= 0:
            return merged_results[:k]

        injected = [
            result
            for result in merged_results
            if result.doc_id in graph_only_ids and result.doc_id not in head_ids
        ][:missing]
        if not injected:
            return merged_results[:k]

        final = head + injected
        if len(final) < k:
            used = {result.doc_id for result in final}
            final.extend(
                result for result in merged_results if result.doc_id not in used
            )
            final = final[:k]
        if diagnostics is not None:
            diagnostics["graph_presence_injected"] = len(injected)
            if len(merged_results) > k:
                diagnostics["candidate_limited"] = True
        return final

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

    def _document_first_query(self, query: str) -> bool:
        """Narrow allowlist; ambiguous, relational and event queries stay dual-route."""
        normalized = query.strip().casefold().rstrip("？?。！!")
        _, _, intent = self._route_weights_for_query(normalized)
        if "relationship" in intent or "temporal" in intent:
            return False
        return bool(
            re.fullmatch(
                r"(?:我|你|我的|你的)?(?:饮料|饮食|口味|音乐|电影|游戏|颜色|作息)(?:偏好|喜好|习惯)"
                r"|(?:我|你)(?:喜欢|爱)(?:喝|吃|听|玩)什么",
                normalized,
            )
        )


__all__ = ["DualRouteRetriever"]
