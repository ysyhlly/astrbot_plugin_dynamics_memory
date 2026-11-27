"""Hybrid (Vector + FTS5) time-aware atom-level retriever with Small-to-Big aggregation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger

from ...storage.atom_store import AtomStore
from .atom_vector_retriever import AtomVectorResult, AtomVectorRetriever


@dataclass(slots=True)
class AtomRetrievalResult:
    """A single atom retrieval result with hybrid relevance and temporal scoring."""

    atom_id: int
    parent_memory_id: int
    content: str
    base_score: float  # fused FTS5 + vector similarity
    temporal_score: float  # decay multiplier
    final_score: float  # base_score (or base_score * temporal_score)
    atom_type: str
    importance: float
    confidence: float
    ttl_days: float
    decay_type: str
    metadata: dict[str, Any]
    bm25_score: float | None = None
    vector_score: float | None = None
    entities: list[str] = field(default_factory=list)
    event_time: float | None = None
    created_at: float | None = None


@dataclass(slots=True)
class AggregatedAtomParent:
    """Small-to-Big parent candidate aggregated from top matching atoms."""

    parent_memory_id: int
    score: float
    matched_atoms: list[dict[str, Any]]
    retrieved_atom_ids: list[int]
    best_vector_score: float | None = None
    best_bm25_score: float | None = None


def aggregate_atoms_by_parent(
    atom_results: Sequence[AtomRetrievalResult],
    *,
    max_atoms_per_parent: int = 3,
    second_atom_boost: float = 0.10,
    top_k: int | None = None,
) -> list[AggregatedAtomParent]:
    """Group atom hits by ``parent_memory_id`` carrying top 2-3 matched atoms.

    Uses ``max_atom_score + second_atom_boost * second_atom_score`` so multiple
    corroborating atoms provide a gentle boost without letting long summaries
    spam the candidate pool.
    """
    if not atom_results:
        return []

    by_parent: dict[int, list[AtomRetrievalResult]] = defaultdict(list)
    for item in atom_results:
        if item.parent_memory_id > 0:
            by_parent[int(item.parent_memory_id)].append(item)

    parents: list[AggregatedAtomParent] = []
    cap = max(1, int(max_atoms_per_parent))
    for parent_id, items in by_parent.items():
        # Deduplicate by atom_id keeping highest score
        best_by_atom: dict[int, AtomRetrievalResult] = {}
        for item in items:
            prev = best_by_atom.get(item.atom_id)
            if prev is None or item.final_score > prev.final_score:
                best_by_atom[item.atom_id] = item
        ordered = sorted(
            best_by_atom.values(),
            key=lambda r: (r.final_score, r.base_score, -r.atom_id),
            reverse=True,
        )
        top_atoms = ordered[:cap]
        agg_score = float(top_atoms[0].final_score)
        if len(top_atoms) >= 2 and second_atom_boost > 0:
            agg_score += float(second_atom_boost) * float(top_atoms[1].final_score)

        matched_atoms = [
            {
                "atom_id": int(a.atom_id),
                "content": a.content,
                "atom_type": a.atom_type,
                "score": round(float(a.final_score), 4),
                "vector_score": (
                    round(float(a.vector_score), 4)
                    if a.vector_score is not None
                    else None
                ),
                "bm25_score": (
                    round(float(a.bm25_score), 4) if a.bm25_score is not None else None
                ),
                "entities": list(a.entities),
                "event_time": a.event_time,
                "created_at": a.created_at,
            }
            for a in top_atoms
        ]
        vector_scores = [
            float(a.vector_score) for a in top_atoms if a.vector_score is not None
        ]
        bm25_scores = [
            float(a.bm25_score) for a in top_atoms if a.bm25_score is not None
        ]
        parents.append(
            AggregatedAtomParent(
                parent_memory_id=parent_id,
                score=round(agg_score, 4),
                matched_atoms=matched_atoms,
                retrieved_atom_ids=[int(a.atom_id) for a in top_atoms],
                best_vector_score=max(vector_scores) if vector_scores else None,
                best_bm25_score=max(bm25_scores) if bm25_scores else None,
            )
        )

    parents.sort(key=lambda p: (p.score, -p.parent_memory_id), reverse=True)
    if top_k is not None and top_k >= 0:
        return parents[:top_k]
    return parents


class AtomRetriever:
    """Retrieve memory atoms via Hybrid (Atom Vector + Atom FTS5) search."""

    def __init__(
        self,
        atom_store: AtomStore,
        config: dict[str, Any] | None = None,
        *,
        atom_vector_retriever: AtomVectorRetriever | None = None,
    ):
        self.atom_store = atom_store
        self.config = config or {}
        self.atom_vector_retriever = atom_vector_retriever

    async def search(
        self,
        query: str,
        k: int = 30,
        session_id: str | None = None,
        persona_id: str | None = None,
        interaction_type: str | tuple[str, ...] | None = None,
        source_platform: str | None = None,
        source_session_id: str | None = None,
        relevance_only: bool = False,
        diagnostics: dict[str, Any] | None = None,
        excluded_session_ids: tuple[str, ...] | None = None,
        *,
        local_only: bool = False,
        extra_queries: Sequence[str] | None = None,
    ) -> list[AtomRetrievalResult]:
        """Search atoms using Atom FTS5 + Atom Vector (when enabled) and fuse."""
        queries: list[str] = []
        seen_q: set[str] = set()
        for q in (query, *(extra_queries or ())):
            cleaned = str(q or "").strip()
            if cleaned and cleaned not in seen_q:
                seen_q.add(cleaned)
                queries.append(cleaned)
        if not queries or k <= 0:
            return []

        use_vector = (
            self.atom_vector_retriever is not None
            and bool(self.config.get("atom_vector_enabled", True))
            and not local_only
            and self.atom_vector_retriever.ntotal > 0
        )

        fetch_limit = max(k * 2, k)
        extra_kwargs = (
            {"excluded_session_ids": tuple(excluded_session_ids)}
            if excluded_session_ids
            else {}
        )

        async def _run_fts(q: str):
            try:
                return await self.atom_store.search_fts(
                    query=q,
                    limit=fetch_limit,
                    session_id=session_id,
                    persona_id=persona_id,
                    interaction_type=interaction_type,
                    source_platform=source_platform,
                    source_session_id=source_session_id,
                    **extra_kwargs,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("[AtomRetriever] FTS search error: %s", type(exc).__name__)
                return []

        async def _run_vec(q: str) -> list[AtomVectorResult]:
            if not use_vector or self.atom_vector_retriever is None:
                return []
            try:
                return await self.atom_vector_retriever.search(
                    query=q,
                    k=fetch_limit,
                    session_id=session_id,
                    persona_id=persona_id,
                    interaction_type=interaction_type,
                    source_platform=source_platform,
                    source_session_id=source_session_id,
                    **extra_kwargs,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(
                    "[AtomRetriever] Vector search error: %s", type(exc).__name__
                )
                return []

        fts_tasks = [_run_fts(q) for q in queries]
        vec_tasks = [_run_vec(q) for q in queries] if use_vector else []
        gathered = await asyncio.gather(*fts_tasks, *vec_tasks)
        fts_lists = gathered[: len(fts_tasks)]
        vec_lists = gathered[len(fts_tasks) :]

        # Merge across queries and routes (FTS5 + Vector) by atom_id
        rrf_k = int(self.config.get("rrf_k", 60) or 60)
        atoms_by_id: dict[int, Any] = {}
        bm25_by_id: dict[int, float] = {}
        vec_by_id: dict[int, float] = {}
        rrf_by_id: dict[int, float] = defaultdict(float)

        for q_idx, fts_atoms in enumerate(fts_lists):
            q_weight = 1.0 if q_idx == 0 else 0.75
            for rank, atom in enumerate(fts_atoms):
                aid = int(atom.atom_id)
                atoms_by_id.setdefault(aid, atom)
                score = float(atom.metadata.get("bm25_score", 0.5))
                bm25_by_id[aid] = max(bm25_by_id.get(aid, 0.0), score)
                rrf_by_id[aid] += q_weight / (rrf_k + rank + 1)

        for q_idx, vec_hits in enumerate(vec_lists):
            q_weight = 1.0 if q_idx == 0 else 0.75
            for rank, hit in enumerate(vec_hits):
                atom = hit.atom
                aid = int(atom.atom_id)
                atoms_by_id.setdefault(aid, atom)
                sim = float(hit.similarity)
                vec_by_id[aid] = max(vec_by_id.get(aid, 0.0), sim)
                rrf_by_id[aid] += q_weight / (rrf_k + rank + 1)

        if diagnostics is not None:
            diagnostics["atom_vector_used"] = use_vector
            diagnostics["atom_fts_hits"] = sum(len(lst) for lst in fts_lists)
            diagnostics["atom_vector_hits"] = sum(len(lst) for lst in vec_lists)
            diagnostics["candidate_limited"] = len(atoms_by_id) > k

        if not atoms_by_id:
            return []

        max_rrf = max(rrf_by_id.values(), default=1.0) or 1.0
        results: list[AtomRetrievalResult] = []
        for aid, atom in atoms_by_id.items():
            bm25_s = bm25_by_id.get(aid)
            vec_s = vec_by_id.get(aid)
            rrf_norm = rrf_by_id[aid] / max_rrf

            if bm25_s is not None and vec_s is not None:
                # Hybrid agreement: combine calibrated vector + BM25 + consensus bonus
                base_score = min(1.0, 0.60 * vec_s + 0.30 * bm25_s + 0.15)
            elif vec_s is not None:
                base_score = vec_s
            elif bm25_s is not None:
                base_score = bm25_s
            else:
                base_score = rrf_norm

            temporal_score = float(atom.metadata.get("temporal_score", 1.0))
            final_score = base_score if relevance_only else base_score * temporal_score
            meta = dict(atom.metadata)
            if bm25_s is not None:
                meta["bm25_score"] = round(bm25_s, 4)
            if vec_s is not None:
                meta["vector_score"] = round(vec_s, 4)
            meta["atom_rrf"] = round(rrf_by_id[aid], 6)

            results.append(
                AtomRetrievalResult(
                    atom_id=atom.atom_id,
                    parent_memory_id=atom.parent_memory_id,
                    content=atom.content,
                    base_score=round(base_score, 4),
                    temporal_score=round(temporal_score, 4),
                    final_score=round(final_score, 4),
                    atom_type=atom.atom_type.value,
                    importance=round(atom.importance, 4),
                    confidence=round(atom.confidence, 4),
                    ttl_days=round(atom.ttl_days, 2),
                    decay_type=atom.decay_type.value,
                    metadata=meta,
                    bm25_score=round(bm25_s, 4) if bm25_s is not None else None,
                    vector_score=round(vec_s, 4) if vec_s is not None else None,
                    entities=list(getattr(atom, "entities", ()) or ()),
                    event_time=getattr(atom, "event_time", None),
                    created_at=getattr(atom, "created_at", None),
                )
            )

        results.sort(
            key=lambda r: (r.final_score, r.base_score, -r.atom_id), reverse=True
        )
        return results[:k]

    async def search_parents(
        self,
        query: str,
        k: int = 25,
        *,
        atom_k: int = 30,
        max_atoms_per_parent: int = 3,
        second_atom_boost: float = 0.10,
        session_id: str | None = None,
        persona_id: str | None = None,
        interaction_type: str | tuple[str, ...] | None = None,
        source_platform: str | None = None,
        source_session_id: str | None = None,
        relevance_only: bool = True,
        diagnostics: dict[str, Any] | None = None,
        excluded_session_ids: tuple[str, ...] | None = None,
        local_only: bool = False,
        extra_queries: Sequence[str] | None = None,
    ) -> list[AggregatedAtomParent]:
        """Run hybrid atom retrieval and aggregate Small-to-Big by ``parent_memory_id``."""
        atoms = await self.search(
            query=query,
            k=max(atom_k, k),
            session_id=session_id,
            persona_id=persona_id,
            interaction_type=interaction_type,
            source_platform=source_platform,
            source_session_id=source_session_id,
            relevance_only=relevance_only,
            diagnostics=diagnostics,
            excluded_session_ids=excluded_session_ids,
            local_only=local_only,
            extra_queries=extra_queries,
        )
        return aggregate_atoms_by_parent(
            atoms,
            max_atoms_per_parent=max_atoms_per_parent,
            second_atom_boost=second_atom_boost,
            top_k=k,
        )

    async def touch(self, atom_id: int) -> None:
        """Update access time for an atom."""
        await self.atom_store.touch(atom_id)


__all__ = [
    "AggregatedAtomParent",
    "AtomRetrievalResult",
    "AtomRetriever",
    "aggregate_atoms_by_parent",
]
