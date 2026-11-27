"""Dedicated FAISS IndexIDMap vector retriever for active memory_atoms."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from astrbot.api import logger

from ..faiss_async_persist import (
    AsyncIndexPersister,
    get_async_persister,
    install_async_persist,
)
from ..models.memory_atom import AtomStatus, MemoryAtom
from ..utils.faiss_path_bridge import needs_bridge, read_index_safe
from .vector_search import (
    call_embeddings_batch_with_telemetry,
    get_cached_embedding,
)

try:
    from ... import series_panels as _series_panels
except Exception:  # pragma: no cover - isolated import fallback
    try:
        import series_panels as _series_panels  # type: ignore
    except Exception:
        _series_panels = None


def _apply_faiss_omp_threads(faiss_mod: Any, context: Any = None) -> int:
    """Cap FAISS OpenMP threads according to the active series resource profile."""
    threads = min(2, max(1, os.cpu_count() or 1))
    if _series_panels is not None:
        getter = getattr(_series_panels, "get_series_resource_limits", None)
        if callable(getter):
            try:
                limits = getter(context)
                if isinstance(limits, dict) and limits.get("native_threads"):
                    threads = max(1, int(limits["native_threads"]))
            except Exception:
                pass
    setter = getattr(faiss_mod, "omp_set_num_threads", None)
    if callable(setter):
        try:
            setter(threads)
        except Exception:
            pass
    return threads


@asynccontextmanager
async def _heavy_rebuild_slot(context: Any = None):
    slot_fn = (
        getattr(_series_panels, "heavy_subprocess_slot", None)
        if _series_panels is not None
        else None
    )
    if callable(slot_fn):
        async with slot_fn(
            context,
            "atom_vector_rebuild",
            plugin_id="astrbot_plugin_dynamics_memory",
        ):
            yield
    else:
        yield


def _try_malloc_trim() -> None:
    trim_fn = (
        getattr(_series_panels, "try_malloc_trim", None)
        if _series_panels is not None
        else None
    )
    if callable(trim_fn):
        try:
            trim_fn()
        except Exception:
            pass


@dataclass(slots=True)
class AtomVectorResult:
    """One active memory_atom matched by FAISS vector similarity."""

    atom: MemoryAtom
    similarity: float


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D float32 matrix row-wise so L2 distance maps to cosine."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


def _create_empty_idmap(dimension: int, *, context: Any = None):
    import faiss

    _apply_faiss_omp_threads(faiss, context)
    base = faiss.IndexFlatL2(int(dimension))
    return faiss.IndexIDMap(base)


def _load_or_create_index(
    path: str | None, dimension: int, *, context: Any = None
) -> tuple[Any, bool]:
    """Load an existing IndexIDMap from disk or create an empty one.

    Returns ``(index, recreated)``.
    """
    import faiss

    _apply_faiss_omp_threads(faiss, context)
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            loaded = (
                read_index_safe(path) if needs_bridge(path) else faiss.read_index(path)
            )
            if int(getattr(loaded, "d", 0)) == int(dimension):
                return loaded, False
            logger.warning(
                "[AtomVector] 索引维度不匹配 (index=%s, expected=%s)，将创建空索引等待重建",
                getattr(loaded, "d", 0),
                dimension,
            )
        except Exception as exc:
            logger.warning("[AtomVector] 加载原子向量索引失败，将创建空索引: %s", exc)
    return _create_empty_idmap(dimension, context=context), True


class _AtomFaissStorage:
    """Minimal storage wrapper compatible with ``core/faiss_async_persist.py``."""

    def __init__(self, path: str | None, dimension: int):
        self.path = path
        self.dimension = int(dimension)
        self.index = _create_empty_idmap(self.dimension)

    async def insert(self, vector: np.ndarray, atom_id: int) -> None:
        await self.insert_batch(vector.reshape(1, -1), [int(atom_id)])

    async def insert_batch(self, vectors: np.ndarray, atom_ids: Sequence[int]) -> None:
        if len(atom_ids) == 0:
            return
        ids_arr = np.asarray([int(i) for i in atom_ids], dtype=np.int64)
        vecs_arr = _l2_normalize_rows(vectors)
        if vecs_arr.shape[1] != self.dimension:
            raise ValueError(
                f"Atom embedding dimension mismatch: got {vecs_arr.shape[1]}, expected {self.dimension}"
            )
        # Keep updates idempotent: remove any previous vector with the same atom_id.
        if self.index.ntotal > 0:
            self.index.remove_ids(ids_arr)
        self.index.add_with_ids(vecs_arr, ids_arr)

    async def delete(self, atom_ids: Sequence[int]) -> int:
        if not atom_ids or self.index.ntotal == 0:
            return 0
        ids_arr = np.asarray(sorted({int(i) for i in atom_ids}), dtype=np.int64)
        removed = int(self.index.remove_ids(ids_arr))
        return removed

    async def save_index(self) -> None:
        # Replaced at runtime by install_async_persist when self.path is set.
        if not self.path:
            return
        from ..faiss_async_persist import _write_index_atomic

        await asyncio.to_thread(_write_index_atomic, self.index, self.path)


class AtomVectorRetriever:
    """Dedicated FAISS IndexIDMap vector retriever for active ``memory_atoms``.

    - Stores active atom embeddings keyed directly by ``memory_atoms.id`` (1024-dim).
    - Reuses ``QueryEmbeddingCache`` on query path (zero extra embedding calls).
    - Uses ``core/faiss_async_persist.py`` for lock-safe debounced async persistence.
    """

    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        atom_store: Any,
        embedding_provider: Any,
        index_path: str | None = None,
        *,
        dimension: int | None = None,
        config: dict[str, Any] | None = None,
        context: Any = None,
    ):
        self.atom_store = atom_store
        self.embedding_provider = embedding_provider
        self.index_path = str(index_path) if index_path else None
        self.config = config or {}
        self.context = context
        self.dimension = self._resolve_dimension(dimension, embedding_provider)
        self._storage = _AtomFaissStorage(self.index_path, self.dimension)
        self._persister: AsyncIndexPersister | None = None
        self._initialized = False

    @classmethod
    def _resolve_dimension(cls, explicit_dim: int | None, provider: Any) -> int:
        if explicit_dim is not None and int(explicit_dim) > 0:
            return int(explicit_dim)
        get_dim = getattr(provider, "get_dim", None)
        if callable(get_dim):
            try:
                val = int(get_dim())
                if val > 0:
                    return val
            except Exception:
                pass
        return cls.DEFAULT_DIMENSION

    @property
    def ntotal(self) -> int:
        index = getattr(self._storage, "index", None)
        return int(getattr(index, "ntotal", 0) or 0)

    async def initialize(self) -> bool:
        """Load persisted index off the event loop and attach async persister."""
        if self._initialized:
            return True
        if self.index_path:
            index, _recreated = await asyncio.to_thread(
                _load_or_create_index,
                self.index_path,
                self.dimension,
                context=self.context,
            )
            self._storage.index = index
            self._persister = install_async_persist(self._storage, context=self.context)
        self._initialized = True
        logger.info(
            "[AtomVector] 初始化完成: ntotal=%d dim=%d path=%s",
            self.ntotal,
            self.dimension,
            self.index_path or "<memory>",
        )
        return True

    async def close(self) -> None:
        """Flush pending writes and close the async persister."""
        persister = self._persister or get_async_persister(self._storage)
        if persister is not None:
            await persister.aclose()
        self._persister = None

    async def flush_now(self) -> None:
        persister = self._persister or get_async_persister(self._storage)
        if persister is not None:
            await persister.flush_now()
        elif self.index_path:
            await self._storage.save_index()

    async def add_atoms(
        self,
        atoms: Sequence[MemoryAtom],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
        batch_size: int = 32,
    ) -> int:
        """Incrementally embed and add active atoms into the FAISS index."""
        if not atoms:
            return 0
        valid: list[tuple[int, MemoryAtom]] = []
        for idx, atom in enumerate(atoms):
            atom_id = int(getattr(atom, "atom_id", 0) or 0)
            content = str(getattr(atom, "content", "") or "").strip()
            status = getattr(atom, "status", AtomStatus.ACTIVE)
            status_val = getattr(status, "value", str(status))
            if atom_id > 0 and content and status_val == AtomStatus.ACTIVE.value:
                valid.append((idx, atom))
        if not valid:
            return 0

        if embeddings is not None:
            vecs = np.asarray([embeddings[idx] for idx, _ in valid], dtype=np.float32)
            ids = [int(atom.atom_id) for _, atom in valid]
            if not np.isfinite(vecs).all():
                raise ValueError("Invalid atom embeddings")
            await self._storage.insert_batch(vecs, ids)
            await self._storage.save_index()
            return len(ids)

        if self.embedding_provider is None:
            return 0

        indexed = 0
        chunk_size = max(1, int(batch_size))
        for start in range(0, len(valid), chunk_size):
            chunk = valid[start : start + chunk_size]
            texts = [str(atom.content).strip()[:2000] for _, atom in chunk]
            raw_embeddings = await call_embeddings_batch_with_telemetry(
                self.embedding_provider, texts
            )
            if len(raw_embeddings) != len(chunk):
                raise ValueError(
                    f"Atom batch embedding count mismatch: got {len(raw_embeddings)}, expected {len(chunk)}"
                )
            vecs = np.asarray(raw_embeddings, dtype=np.float32)
            if (
                vecs.ndim != 2
                or vecs.shape[1] != self.dimension
                or not np.isfinite(vecs).all()
            ):
                raise ValueError("Invalid atom batch embeddings")
            ids = [int(atom.atom_id) for _, atom in chunk]
            await self._storage.insert_batch(vecs, ids)
            indexed += len(ids)

        if indexed > 0:
            await self._storage.save_index()
        return indexed

    async def delete_atom_ids(self, atom_ids: Sequence[int]) -> int:
        """Incrementally remove expired/forgotten/deleted atom IDs from FAISS."""
        ids = [
            int(i)
            for i in atom_ids
            if isinstance(i, int) and not isinstance(i, bool) and i > 0
        ]
        if not ids or self.ntotal == 0:
            return 0
        removed = await self._storage.delete(ids)
        if removed > 0:
            await self._storage.save_index()
        return removed

    async def rebuild_from_store(
        self,
        *,
        batch_size: int = 32,
        request_delay: float = 0.0,
        progress_callback: Callable[[int, int, str], Any] | None = None,
    ) -> dict[str, int]:
        """Batch-build the FAISS index from all active non-expired atoms in AtomStore.

        Designed for offline/online bootstrap of the ~6,506 active atoms using
        DashScope Batch Embedding. Builds into a fresh ``IndexIDMap`` and swaps
        atomically under the persister write lock.
        """
        if self.atom_store is None or self.embedding_provider is None:
            return {"total": 0, "indexed": 0, "failed": 0}

        list_active = getattr(self.atom_store, "list_active_atoms", None)
        if not callable(list_active):
            raise RuntimeError("AtomStore does not implement list_active_atoms()")

        async with _heavy_rebuild_slot(self.context):
            try:
                atoms = await list_active(include_expired=False)
                total = len(atoms)
                if total == 0:
                    fresh = _create_empty_idmap(self.dimension, context=self.context)
                    await self._swap_index(fresh)
                    return {"total": 0, "indexed": 0, "failed": 0}

                fresh_index = _create_empty_idmap(self.dimension, context=self.context)
                indexed = 0
                failed = 0
                chunk_size = max(1, int(batch_size))

                for start in range(0, total, chunk_size):
                    chunk = atoms[start : start + chunk_size]
                    texts = [str(atom.content).strip()[:2000] for atom in chunk]
                    ids = [int(atom.atom_id) for atom in chunk]
                    try:
                        raw_embeddings = await call_embeddings_batch_with_telemetry(
                            self.embedding_provider, texts
                        )
                        vecs = _l2_normalize_rows(
                            np.asarray(raw_embeddings, dtype=np.float32)
                        )
                        if (
                            vecs.shape[0] != len(chunk)
                            or vecs.shape[1] != self.dimension
                            or not np.isfinite(vecs).all()
                        ):
                            raise ValueError("Invalid batch embedding shape or values")
                        fresh_index.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))
                        indexed += len(chunk)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        failed += len(chunk)
                        logger.warning(
                            "[AtomVector] 批量重建批次失败 (offset=%d, size=%d): %s",
                            start,
                            len(chunk),
                            exc,
                        )
                    if callable(progress_callback):
                        maybe = progress_callback(
                            min(start + len(chunk), total),
                            total,
                            f"原子向量重建进度 {min(start + len(chunk), total)}/{total}",
                        )
                        if asyncio.iscoroutine(maybe) or hasattr(maybe, "__await__"):
                            await maybe
                    if request_delay > 0 and start + chunk_size < total:
                        await asyncio.sleep(request_delay)

                await self._swap_index(fresh_index)
                logger.info(
                    "[AtomVector] 原子向量索引重建完成: total=%d indexed=%d failed=%d",
                    total,
                    indexed,
                    failed,
                )
                return {"total": total, "indexed": indexed, "failed": failed}
            finally:
                _try_malloc_trim()

    async def _swap_index(self, new_index: Any) -> None:
        persister = self._persister or get_async_persister(self._storage)
        if persister is not None:
            async with persister._index_lock.write():
                self._storage.index = new_index
            persister._dirty = True
            await persister.flush_now()
        else:
            self._storage.index = new_index
            if self.index_path:
                await self._storage.save_index()

    async def search(
        self,
        query: str,
        k: int = 30,
        *,
        session_id: str | None = None,
        persona_id: str | None = None,
        interaction_type: str | tuple[str, ...] | None = None,
        source_platform: str | None = None,
        source_session_id: str | None = None,
        excluded_session_ids: tuple[str, ...] | None = None,
        query_vector: Sequence[float] | np.ndarray | None = None,
    ) -> list[AtomVectorResult]:
        """Search active atoms by vector similarity, sharing QueryEmbeddingCache."""
        if k <= 0 or not query or not query.strip():
            return []
        total = self.ntotal
        if total <= 0:
            return []

        if query_vector is not None:
            vec = np.asarray(query_vector, dtype=np.float32).ravel()
        else:
            if self.embedding_provider is None:
                return []
            embedding = await get_cached_embedding(self.embedding_provider, query)
            vec = np.asarray(embedding, dtype=np.float32).ravel()

        if vec.size != self.dimension or not np.isfinite(vec).all():
            raise ValueError(
                f"Invalid atom query embedding: expected dim {self.dimension}, got {vec.size}"
            )
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = (vec / norm).astype(np.float32, copy=False)

        has_filters = any(
            v is not None
            for v in (
                session_id,
                persona_id,
                interaction_type,
                source_platform,
                source_session_id,
                excluded_session_ids,
            )
        )
        cap = min(total, 8192)
        fetch_k = min(cap, max(k * (4 if has_filters else 2), k))
        persister = self._persister or get_async_persister(self._storage)

        fetched_atoms: dict[int, MemoryAtom] = {}
        while True:
            if persister is not None:
                distances, indices = await persister.search(vec, fetch_k)
            else:
                distances, indices = await asyncio.to_thread(
                    self._storage.index.search, vec.reshape(1, -1), fetch_k
                )
            ids = [int(i) for i in indices[0] if int(i) > 0]
            if not ids:
                return []

            new_ids = [aid for aid in ids if aid not in fetched_atoms]
            if new_ids:
                matched = await self.atom_store.get_by_ids_filtered(
                    new_ids,
                    session_id=session_id,
                    persona_id=persona_id,
                    include_expired=False,
                    interaction_type=interaction_type,
                    source_platform=source_platform,
                    source_session_id=source_session_id,
                    excluded_session_ids=excluded_session_ids,
                )
                for atom in matched:
                    fetched_atoms[int(atom.atom_id)] = atom

            results: list[AtomVectorResult] = []
            seen: set[int] = set()
            now = time.time()
            for dist, raw_id in zip(distances[0], indices[0]):
                atom_id = int(raw_id)
                if atom_id <= 0 or atom_id in seen:
                    continue
                atom = fetched_atoms.get(atom_id)
                if atom is None:
                    continue
                seen.add(atom_id)
                sim = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
                atom.metadata["vector_score"] = sim
                atom.metadata["temporal_score"] = atom.compute_temporal_score(now)
                results.append(AtomVectorResult(atom=atom, similarity=sim))
                if len(results) >= k:
                    return results

            if len(ids) < fetch_k or fetch_k >= cap:
                return results
            fetch_k = min(cap, fetch_k * 2)


__all__ = ["AtomVectorResult", "AtomVectorRetriever"]
