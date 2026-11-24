"""Keyword retrieval for the graph-memory route."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ...storage.graph_store import GraphStore
from ..processors.text_processor import TextProcessor

_DEFAULT_MAX_GRAPH_TOKENS = 12


@dataclass(slots=True)
class GraphKeywordResult:
    """Keyword match aggregated to one source memory."""

    doc_id: int
    score: float
    content: str
    metadata: dict[str, Any]


class GraphKeywordRetriever:
    """Retrieve graph-memory candidates with FTS and one-hop expansion."""

    def __init__(
        self,
        graph_store: GraphStore,
        text_processor: TextProcessor,
        config: dict[str, Any] | None = None,
    ):
        self.graph_store = graph_store
        self.text_processor = text_processor
        self.config = config or {}
        self.expansion_limit = int(self.config.get("graph_expansion_limit", 24))
        self.expansion_hops = max(
            1,
            min(2, int(self.config.get("graph_expansion_hops", 1))),
        )
        self.second_hop_weight = float(self.config.get("graph_second_hop_weight", 0.4))
        self.max_query_tokens = max(
            1,
            int(self.config.get("graph_max_query_tokens", _DEFAULT_MAX_GRAPH_TOKENS)),
        )

    @staticmethod
    def _select_node_tokens(
        tokens: list[str], max_tokens: int, *, query: str = ""
    ) -> list[str]:
        """Prioritize longer, higher-signal tokens and cap count to bound query cost."""
        q_clean = str(query or "").strip()
        raw_items = list(tokens)
        if 2 <= len(q_clean) <= 12 and not any(ch.isspace() for ch in q_clean):
            raw_items.insert(0, q_clean)
        deduped = list(dict.fromkeys(str(t).strip() for t in raw_items if str(t).strip()))
        if not deduped:
            return []
        multi = [t for t in deduped if len(t) >= 2]
        pool = multi if multi else deduped
        # Stable sort: longer tokens first (more selective / trigram-eligible), preserving order.
        ranked = sorted(pool, key=lambda t: (-len(t),))
        return ranked[: max(1, int(max_tokens))]

    async def search(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
        persona_id: str | None = None,
        source_platform: str | None = None,
        interaction_type: str | None = None,
        source_session_id: str | None = None,
    ) -> list[GraphKeywordResult]:
        """Search the graph route with keyword matching."""
        if not query or not query.strip():
            return []

        tokens = await self.text_processor.tokenize_async(query, remove_stopwords=True)
        if not tokens:
            return []

        node_tokens = self._select_node_tokens(
            tokens, self.max_query_tokens, query=query
        )
        fts_tokens = list(dict.fromkeys([*node_tokens, *tokens]))[
            : self.max_query_tokens * 2
        ]
        escaped_tokens = [
            '"' + token.replace('"', '""') + '"' for token in fts_tokens
        ]
        fts_query = " OR ".join(escaped_tokens)

        direct_hits, matched_nodes = await asyncio.gather(
            self.graph_store.search_entries_by_bm25(
                fts_query=fts_query,
                limit=max(limit * 3, 12),
                session_id=session_id,
                persona_id=persona_id,
                source_platform=source_platform,
                interaction_type=interaction_type,
                source_session_id=source_session_id,
            ),
            self.graph_store.search_nodes_by_tokens(
                tokens=node_tokens,
                limit=max(limit * 3, 12),
            ),
        )
        matched_node_ids = [item["id"] for item in matched_nodes]
        expansion_hits: list[dict[str, Any]] = []
        edge_neighbor_hits: list[dict[str, Any]] = []
        second_hop_hits: list[dict[str, Any]] = []
        if matched_node_ids:
            expansion_hits, first_hop_raw = await asyncio.gather(
                self.graph_store.get_entries_for_node_ids(
                    node_ids=matched_node_ids,
                    limit=max(self.expansion_limit, limit * 3),
                    session_id=session_id,
                    persona_id=persona_id,
                    source_platform=source_platform,
                    interaction_type=interaction_type,
                    source_session_id=source_session_id,
                ),
                self.graph_store.get_neighbor_node_ids(
                    node_ids=matched_node_ids,
                    limit=max(self.expansion_limit, limit * 3),
                ),
            )
            matched_node_set = set(matched_node_ids)
            first_hop_node_ids = [
                node_id
                for node_id in first_hop_raw
                if node_id not in matched_node_set
            ]
            if first_hop_node_ids:
                if self.expansion_hops >= 2:
                    edge_neighbor_hits, second_hop_raw = await asyncio.gather(
                        self.graph_store.get_entries_for_node_ids(
                            node_ids=first_hop_node_ids,
                            limit=max(self.expansion_limit, limit * 3),
                            session_id=session_id,
                            persona_id=persona_id,
                            source_platform=source_platform,
                            interaction_type=interaction_type,
                            source_session_id=source_session_id,
                        ),
                        self.graph_store.get_neighbor_node_ids(
                            node_ids=first_hop_node_ids,
                            limit=max(self.expansion_limit, limit * 3),
                        ),
                    )
                    excluded_node_ids = matched_node_set | set(first_hop_node_ids)
                    second_hop_node_ids = [
                        node_id
                        for node_id in second_hop_raw
                        if node_id not in excluded_node_ids
                    ]
                    if second_hop_node_ids:
                        second_hop_hits = await self.graph_store.get_entries_for_node_ids(
                            node_ids=second_hop_node_ids,
                            limit=max(self.expansion_limit, limit * 3),
                            session_id=session_id,
                            persona_id=persona_id,
                            source_platform=source_platform,
                            interaction_type=interaction_type,
                            source_session_id=source_session_id,
                        )
                else:
                    edge_neighbor_hits = await self.graph_store.get_entries_for_node_ids(
                        node_ids=first_hop_node_ids,
                        limit=max(self.expansion_limit, limit * 3),
                        session_id=session_id,
                        persona_id=persona_id,
                        source_platform=source_platform,
                        interaction_type=interaction_type,
                        source_session_id=source_session_id,
                    )

        aggregated: dict[int, GraphKeywordResult] = {}

        def merge_hit(hit: dict[str, Any], weight: float, match_source: str) -> None:
            doc_id = int(hit["source_memory_id"])
            weighted_score = max(0.0, min(1.0, float(hit["score"]) * weight))
            hit_metadata = dict(hit.get("metadata") or {})
            hit_metadata["graph_match_source"] = match_source
            hit_metadata["graph_entry_type"] = hit.get("entry_type")
            hit_metadata["graph_relation_type"] = hit.get("relation_type")
            current = aggregated.get(doc_id)
            if current is None or weighted_score > current.score:
                aggregated[doc_id] = GraphKeywordResult(
                    doc_id=doc_id,
                    score=weighted_score,
                    content=str(hit.get("content") or ""),
                    metadata=hit_metadata,
                )
                return
            current.score = min(1.0, current.score + weighted_score * 0.35)
            if "graph_match_source" in current.metadata:
                current.metadata["graph_match_source"] = (
                    f"{current.metadata['graph_match_source']}+{match_source}"
                )

        for hit in direct_hits:
            merge_hit(hit, weight=1.0, match_source="graph_keyword")

        for hit in expansion_hits:
            merge_hit(hit, weight=0.7, match_source="graph_neighbor")

        for hit in edge_neighbor_hits:
            merge_hit(hit, weight=0.7, match_source="graph_edge_neighbor")

        for hit in second_hop_hits:
            merge_hit(
                hit,
                weight=max(0.0, min(1.0, self.second_hop_weight)),
                match_source="graph_second_hop",
            )

        results = sorted(aggregated.values(), key=lambda item: item.score, reverse=True)
        return results[:limit]


__all__ = ["GraphKeywordRetriever", "GraphKeywordResult"]
