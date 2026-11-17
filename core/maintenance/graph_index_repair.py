"""Plan and apply graph-index repairs using already-persisted vectors only.

This module deliberately has no embedding-provider or network dependency. A
caller must provide a persisted vector whose document, source, text digest,
model, and dimension all match the current active document before it can be
added to an index.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Protocol, Sequence


class RepairClass(StrEnum):
    ALREADY_INDEXED = "already_indexed"
    ARCHIVED_UNINDEXED_SOURCE = "archived_unindexed_source"
    STALE_INDEXED_SOURCE = "stale_indexed_source"
    CACHE_IDENTITY_MISMATCH = "cache_identity_mismatch"
    CACHE_MODEL_MISMATCH = "cache_model_mismatch"
    CACHE_DIMENSION_MISMATCH = "cache_dimension_mismatch"
    REQUIRES_EMBEDDING = "requires_embedding"
    REPAIRABLE_CACHED_VECTOR = "repairable_cached_vector"


@dataclass(frozen=True, slots=True)
class GraphIndexDocument:
    """Non-sensitive facts needed to validate one graph vector document."""

    document_id: int
    source_memory_id: int
    source_is_active: bool
    content_sha256: str


@dataclass(frozen=True, slots=True)
class PersistedGraphVector:
    """A vector recovered from a durable local index/cache, never generated here."""

    document_id: int
    source_memory_id: int
    content_sha256: str
    model_id: str
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RepairPlan:
    """An immutable, scope-limited repair decision suitable for review."""

    index_dimension: int
    expected_model_id: str | None
    requested_ids: tuple[int, ...]
    classifications: tuple[tuple[int, RepairClass], ...]
    apply_ids: tuple[int, ...]
    apply_bindings: tuple[tuple[int, int, str], ...]
    missing_active_ids: tuple[int, ...]
    requires_embedding_ids: tuple[int, ...]
    stale_index_ids: tuple[int, ...]
    prune_bindings: tuple[tuple[int, int, str], ...]
    orphan_index_ids: tuple[int, ...]

    def counts(self) -> dict[str, int]:
        counts = {item.value: 0 for item in RepairClass}
        for _, classification in self.classifications:
            counts[classification.value] += 1
        return counts


class GraphIndexRepairAdapter(Protocol):
    """Narrow mutation contract implemented by the FAISS integration layer.

    ``persist_atomic`` must write a complete index to a sibling temporary file
    and atomically replace the target. If it raises or is cancelled, the prior
    durable index must remain intact. The caller must serialize writes to the
    live index while this transaction is in progress.
    """

    def snapshot(self) -> object: ...

    def add_with_ids(self, vectors: Mapping[int, Sequence[float]]) -> None: ...

    def restore(self, snapshot: object) -> None: ...

    def persist_atomic(self) -> None: ...

    def existing_vectors(
        self, document_ids: Sequence[int]
    ) -> Mapping[int, Sequence[float]]: ...

    def existing_document_bindings(
        self, document_ids: Sequence[int]
    ) -> Mapping[int, tuple[int, str, bool]]: ...

    def delete_ids(self, document_ids: Sequence[int]) -> None: ...


def content_digest(text: str) -> str:
    """Return the stable UTF-8 SHA-256 digest used to bind vectors to content."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plan_graph_index_repair(
    documents: Sequence[GraphIndexDocument],
    *,
    indexed_ids: set[int] | frozenset[int],
    persisted_vectors: Mapping[int, PersistedGraphVector],
    expected_model_id: str | None,
    index_dimension: int,
    requested_ids: Sequence[int] | None = None,
) -> RepairPlan:
    """Classify an explicit ID scope without generating or fetching vectors.

    ``requested_ids=None`` means all supplied documents. If IDs are supplied,
    every ID must exist in ``documents``; this prevents accidentally widening a
    maintenance command to records outside the reviewed scope.
    """

    if index_dimension <= 0:
        raise ValueError("index_dimension must be positive")
    if expected_model_id is not None and not expected_model_id.strip():
        raise ValueError("expected_model_id must be non-empty when provided")

    by_id: dict[int, GraphIndexDocument] = {}
    for document in documents:
        doc_id = int(document.document_id)
        if doc_id in by_id:
            raise ValueError(f"duplicate graph document id: {doc_id}")
        by_id[doc_id] = document

    scope = tuple(
        sorted(by_id if requested_ids is None else {int(i) for i in requested_ids})
    )
    unknown = set(scope) - by_id.keys()
    if unknown:
        raise ValueError(
            f"requested graph document ids are outside inventory: {sorted(unknown)}"
        )

    classifications: list[tuple[int, RepairClass]] = []
    apply_ids: list[int] = []
    apply_bindings: list[tuple[int, int, str]] = []
    missing_active_ids: list[int] = []
    requires_embedding_ids: list[int] = []
    stale_index_ids: list[int] = []
    prune_bindings: list[tuple[int, int, str]] = []
    for doc_id in scope:
        document = by_id[doc_id]
        vector = persisted_vectors.get(doc_id)
        if not document.source_is_active:
            if doc_id in indexed_ids:
                classification = RepairClass.STALE_INDEXED_SOURCE
                stale_index_ids.append(doc_id)
                prune_bindings.append(
                    (doc_id, document.source_memory_id, document.content_sha256)
                )
            else:
                classification = RepairClass.ARCHIVED_UNINDEXED_SOURCE
        elif doc_id in indexed_ids:
            classification = RepairClass.ALREADY_INDEXED
        elif vector is not None and (
            vector.document_id != doc_id
            or vector.source_memory_id != document.source_memory_id
            or vector.content_sha256 != document.content_sha256
        ):
            # A stale/misattributed cached vector is not a safe embedding source.
            classification = RepairClass.CACHE_IDENTITY_MISMATCH
            missing_active_ids.append(doc_id)
        elif vector is None:
            classification = RepairClass.REQUIRES_EMBEDDING
            missing_active_ids.append(doc_id)
            requires_embedding_ids.append(doc_id)
        elif expected_model_id is None or vector.model_id != expected_model_id:
            classification = RepairClass.CACHE_MODEL_MISMATCH
            missing_active_ids.append(doc_id)
        elif len(vector.values) != index_dimension:
            classification = RepairClass.CACHE_DIMENSION_MISMATCH
            missing_active_ids.append(doc_id)
        elif not all(math.isfinite(float(value)) for value in vector.values):
            classification = RepairClass.CACHE_DIMENSION_MISMATCH
            missing_active_ids.append(doc_id)
        else:
            classification = RepairClass.REPAIRABLE_CACHED_VECTOR
            apply_ids.append(doc_id)
            missing_active_ids.append(doc_id)
            apply_bindings.append(
                (doc_id, document.source_memory_id, document.content_sha256)
            )
        classifications.append((doc_id, classification))

    return RepairPlan(
        index_dimension=int(index_dimension),
        expected_model_id=expected_model_id,
        requested_ids=scope,
        classifications=tuple(classifications),
        apply_ids=tuple(apply_ids),
        apply_bindings=tuple(apply_bindings),
        missing_active_ids=tuple(missing_active_ids),
        requires_embedding_ids=tuple(requires_embedding_ids),
        stale_index_ids=tuple(stale_index_ids),
        prune_bindings=tuple(prune_bindings),
        orphan_index_ids=tuple(sorted(set(indexed_ids) - by_id.keys())),
    )


def apply_cached_graph_vector_repair(
    plan: RepairPlan,
    persisted_vectors: Mapping[int, PersistedGraphVector],
    adapter: GraphIndexRepairAdapter,
) -> tuple[int, ...]:
    """Apply only the exact compatible IDs in ``plan`` with rollback on failure.

    The function performs no provider lookup. It revalidates every supplied
    vector at the mutation boundary and rejects extra/missing IDs. On any
    ``BaseException`` (including cancellation), it restores the in-memory
    snapshot and re-raises; durable atomicity is provided by ``persist_atomic``.
    """

    approved = tuple(plan.apply_ids)
    if not approved:
        return ()
    if len(approved) != len(set(approved)):
        raise ValueError("repair plan contains duplicate apply IDs")
    if set(persisted_vectors) != set(approved):
        raise ValueError(
            "persisted vector IDs must exactly match the approved repair scope"
        )

    vectors: dict[int, tuple[float, ...]] = {}
    for doc_id in approved:
        vector = persisted_vectors[doc_id]
        classification = dict(plan.classifications).get(doc_id)
        if classification is not RepairClass.REPAIRABLE_CACHED_VECTOR:
            raise ValueError(f"graph document id {doc_id} is not approved for repair")
        if (
            vector.document_id != doc_id
            or plan.expected_model_id is None
            or vector.model_id != plan.expected_model_id
        ):
            raise ValueError(
                f"graph document id {doc_id} cached vector identity changed"
            )
        expected_binding = next(
            (binding for binding in plan.apply_bindings if binding[0] == doc_id),
            None,
        )
        if expected_binding != (
            doc_id,
            vector.source_memory_id,
            vector.content_sha256,
        ):
            raise ValueError(
                f"graph document id {doc_id} cached vector source binding changed"
            )
        if len(vector.values) != plan.index_dimension:
            raise ValueError(
                f"graph document id {doc_id} cached vector dimension changed"
            )
        normalized = tuple(float(value) for value in vector.values)
        if not all(math.isfinite(value) for value in normalized):
            raise ValueError(f"graph document id {doc_id} cached vector is non-finite")
        vectors[doc_id] = normalized

    existing = adapter.existing_vectors(approved)
    for doc_id, current in existing.items():
        if doc_id not in vectors:
            raise ValueError(
                f"index adapter returned an unrequested graph document id: {doc_id}"
            )
        if tuple(float(value) for value in current) != vectors[doc_id]:
            raise ValueError(
                f"graph document id {doc_id} already has a different vector"
            )
    pending = {
        doc_id: value for doc_id, value in vectors.items() if doc_id not in existing
    }
    if not pending:
        return approved

    snapshot = adapter.snapshot()
    try:
        adapter.add_with_ids(pending)
        adapter.persist_atomic()
    except BaseException:
        adapter.restore(snapshot)
        raise
    return approved


def apply_stale_graph_vector_prune(
    plan: RepairPlan,
    adapter: GraphIndexRepairAdapter,
) -> tuple[int, ...]:
    """Delete only planned vectors whose source/content is still archived.

    Graph-document rows and source memories are intentionally untouched. A
    caller must obtain a fresh dry-run plan after source state changes. Repeated
    application is a no-op once the exact vector IDs are absent.
    """

    approved = tuple(plan.stale_index_ids)
    if not approved:
        return ()
    expected = {
        doc_id: (source_id, digest) for doc_id, source_id, digest in plan.prune_bindings
    }
    if set(expected) != set(approved) or len(expected) != len(plan.prune_bindings):
        raise ValueError(
            "stale-vector plan bindings must exactly match its prune scope"
        )

    current = adapter.existing_document_bindings(approved)
    for doc_id, (source_id, digest, source_is_active) in current.items():
        if doc_id not in expected:
            raise ValueError(
                f"adapter returned an unrequested graph document id: {doc_id}"
            )
        if source_is_active or (source_id, digest) != expected[doc_id]:
            raise ValueError(
                f"graph document id {doc_id} source binding changed since planning"
            )
    pending = tuple(doc_id for doc_id in approved if doc_id in current)
    if not pending:
        return approved

    snapshot = adapter.snapshot()
    try:
        adapter.delete_ids(pending)
        adapter.persist_atomic()
    except BaseException:
        adapter.restore(snapshot)
        raise
    return approved


__all__ = [
    "GraphIndexDocument",
    "GraphIndexRepairAdapter",
    "PersistedGraphVector",
    "RepairClass",
    "RepairPlan",
    "apply_cached_graph_vector_repair",
    "apply_stale_graph_vector_prune",
    "content_digest",
    "plan_graph_index_repair",
]
