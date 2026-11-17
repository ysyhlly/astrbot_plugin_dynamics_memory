"""Manage graph-memory indexing and synchronization."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from ..maintenance.graph_index_repair import (
    GraphIndexDocument,
    RepairPlan,
    content_digest,
    plan_graph_index_repair,
)
from ..maintenance.source_state_guard import SourceStateGuard


import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any

from astrbot.api import logger

from ...storage.graph_store import GraphStore
from ..models.graph_models import GraphEntry
from ..processors.graph_extractor import GraphExtractor
from ..retrieval.graph_vector_retriever import GraphVectorRetriever


class GraphMemoryManager:
    """Synchronize graph-memory artifacts with the document memory store."""

    def __init__(
        self,
        graph_store: GraphStore,
        graph_vector_retriever: GraphVectorRetriever,
        graph_extractor: GraphExtractor,
        source_validity_checker: Callable[[Sequence[int]], Awaitable[set[int]]]
        | None = None,
        source_state_guard: SourceStateGuard | None = None,
    ):
        self.graph_store = graph_store
        self.graph_vector_retriever = graph_vector_retriever
        self.graph_extractor = graph_extractor
        self.source_validity_checker = source_validity_checker
        self.source_state_guard = source_state_guard or SourceStateGuard()
        self._rebuild_gate = asyncio.Lock()
        self._rebuild_active = False
        self._rebuild_delta: dict[
            int, tuple[str, dict[str, Any] | None, list | None] | None
        ] = {}

    async def index_memory(
        self,
        source_memory_id: int,
        content: str,
        metadata: dict[str, Any] | None,
        atoms: list | None = None,
    ) -> bool:
        """Rebuild graph artifacts for one source memory.

        When atoms are provided, each atom independently contributes
        nodes/edges/entries with per-atom confidence scores.
        """
        async with self._rebuild_gate:
            if self._rebuild_active:
                self._rebuild_delta[int(source_memory_id)] = (
                    content,
                    metadata,
                    atoms,
                )
                return False
            await self._index_memory_now(source_memory_id, content, metadata, atoms)
            return True

    async def _index_memory_now(
        self,
        source_memory_id: int,
        content: str,
        metadata: dict[str, Any] | None,
        atoms: list | None = None,
    ) -> None:
        await self._delete_memory_now(source_memory_id)

        entries, entry_ids = await self._store_graph_structure(
            source_memory_id,
            content,
            metadata,
            atoms,
        )
        if not entries:
            return

        vector_doc_id = await self.graph_vector_retriever.add_memory_entries(
            [(entry.content, dict(entry.metadata)) for entry in entries]
        )
        await self.graph_store.update_entry_vector_doc_ids(
            {entry_ids[0]: vector_doc_id}
        )

    async def _store_graph_structure(
        self,
        source_memory_id: int,
        content: str,
        metadata: dict[str, Any] | None,
        atoms: list | None = None,
    ) -> tuple[list[GraphEntry], list[int]]:
        """Persist graph structure without touching the vector index."""
        extracted = self.graph_extractor.extract(
            source_memory_id, content, metadata, atoms
        )
        if not extracted.entries:
            return [], []

        node_key_to_id = await self.graph_store.upsert_nodes(extracted.nodes)

        edge_key_to_id = await self.graph_store.add_edges(
            extracted.edges,
            node_key_to_id,
        )

        entry_ids = await self.graph_store.add_entries(
            extracted.entries,
            node_key_to_id,
            edge_key_to_id,
        )
        if len(entry_ids) != len(extracted.entries):
            raise RuntimeError(
                "graph entry id count mismatch: "
                f"ids={len(entry_ids)}, entries={len(extracted.entries)}"
            )
        return extracted.entries, entry_ids

    async def delete_memory(self, source_memory_id: int) -> bool:
        """Delete graph artifacts belonging to one source memory."""
        async with self._rebuild_gate:
            if self._rebuild_active:
                self._rebuild_delta[int(source_memory_id)] = None
                return False
            await self._delete_memory_now(source_memory_id)
            return True

    async def _delete_memory_now(self, source_memory_id: int) -> None:
        vector_doc_ids = await self.graph_store.delete_memory(source_memory_id)
        await self.graph_vector_retriever.delete_entries(
            source_memory_id, vector_doc_ids
        )

    async def batch_delete_memories(self, source_memory_ids: list[int]) -> bool:
        """Delete graph artifacts in one FAISS bulk operation when supported."""
        if not source_memory_ids:
            return True
        async with self._rebuild_gate:
            if self._rebuild_active:
                for source_memory_id in source_memory_ids:
                    self._rebuild_delta[int(source_memory_id)] = None
                return False
            memory_vec_map = await self.graph_store.batch_delete_memories(
                source_memory_ids
            )
            await self.graph_vector_retriever.delete_entries_batch(memory_vec_map)
            return True

    async def rebuild_memory_batches(
        self,
        memory_batches: AsyncIterable[list[tuple[int, str, dict[str, Any]]]],
        *,
        vector_batch_size: int = 100,
    ) -> dict[str, int]:
        """Build graph data in shadow storage and atomically switch when complete."""
        async with self._rebuild_gate:
            if self._rebuild_active:
                raise RuntimeError("graph rebuild already in progress")
            self._rebuild_active = True
            self._rebuild_delta.clear()

        temp_dir = Path(tempfile.mkdtemp(prefix="livingmemory_graph_rebuild_"))
        shadow_store = GraphStore(str(temp_dir / "graph.db"))
        shadow_manager = GraphMemoryManager(
            shadow_store,
            self.graph_vector_retriever,
            self.graph_extractor,
        )
        new_vector_doc_ids: dict[int, set[int]] = {}
        old_vector_doc_ids = await self.graph_store.list_vector_doc_ids_by_source()
        rebuilt = 0
        skipped = 0
        switched = False
        replay_delta = {}

        async def remove_shadow_vectors(
            source_memory_id: int, vector_doc_ids: list[int]
        ) -> None:
            known_ids = new_vector_doc_ids.get(int(source_memory_id), set())
            ids = [
                int(vector_doc_id)
                for vector_doc_id in vector_doc_ids
                if int(vector_doc_id) in known_ids
            ]
            if not ids:
                return
            await self.graph_vector_retriever.delete_entries_batch(
                {int(source_memory_id): ids}
            )
            known_ids.difference_update(ids)
            if not known_ids:
                new_vector_doc_ids.pop(int(source_memory_id), None)

        async def apply_shadow_delta(
            delta: dict[int, tuple[str, dict[str, Any] | None, list | None] | None],
        ) -> None:
            for source_memory_id, payload in delta.items():
                replaced_vector_ids = await shadow_store.delete_memory(source_memory_id)
                await remove_shadow_vectors(source_memory_id, replaced_vector_ids)
                if payload is None:
                    continue
                content, metadata, atoms = payload
                if not content.strip():
                    continue
                entries, entry_ids = await shadow_manager._store_graph_structure(
                    source_memory_id,
                    content,
                    metadata,
                    atoms,
                )
                if not entries:
                    continue
                vector_doc_id = (
                    await self.graph_vector_retriever.add_memory_entries_batch(
                        [[(entry.content, dict(entry.metadata)) for entry in entries]]
                    )
                )[0]
                new_vector_doc_ids.setdefault(int(source_memory_id), set()).add(
                    int(vector_doc_id)
                )
                await shadow_store.update_entry_vector_doc_ids(
                    {entry_ids[0]: int(vector_doc_id)}
                )

        try:
            await shadow_store.initialize()
            async for memories in memory_batches:
                for source_memory_id, content, metadata in memories:
                    if not content.strip():
                        skipped += 1
                        continue
                    entries, _entry_ids = await shadow_manager._store_graph_structure(
                        source_memory_id,
                        content,
                        metadata,
                    )
                    if entries:
                        rebuilt += 1
                    else:
                        skipped += 1

            async for groups in shadow_store.iter_memory_entry_groups(
                max(1, int(vector_batch_size))
            ):
                entry_groups = [entries for _, _, entries in groups]
                vector_doc_ids = (
                    await self.graph_vector_retriever.add_memory_entries_batch(
                        entry_groups
                    )
                )
                if len(vector_doc_ids) != len(groups):
                    raise RuntimeError(
                        "graph vector id count mismatch: "
                        f"ids={len(vector_doc_ids)}, memories={len(groups)}"
                    )
                for (source_memory_id, _, _), vector_doc_id in zip(
                    groups, vector_doc_ids, strict=True
                ):
                    new_vector_doc_ids.setdefault(int(source_memory_id), set()).add(
                        int(vector_doc_id)
                    )
                await shadow_store.update_entry_vector_doc_ids(
                    {
                        representative_entry_id: int(vector_doc_id)
                        for (_, representative_entry_id, _), vector_doc_id in zip(
                            groups, vector_doc_ids, strict=True
                        )
                    }
                )

            while True:
                async with self._rebuild_gate:
                    if self._rebuild_delta:
                        pending_delta = dict(self._rebuild_delta)
                        self._rebuild_delta.clear()
                        replay_delta.update(pending_delta)
                    else:
                        switch_task = asyncio.create_task(
                            self.graph_store.replace_all_from(
                                str(temp_dir / "graph.db")
                            )
                        )
                        try:
                            await asyncio.shield(switch_task)
                        except asyncio.CancelledError:
                            # Learn whether the transaction committed before
                            # propagating cancellation; never roll back the new
                            # vectors after a successful table switch.
                            await switch_task
                            switched = True
                            self._rebuild_active = False
                            raise
                        switched = True
                        self._rebuild_active = False
                        break
                await apply_shadow_delta(pending_delta)

            if old_vector_doc_ids:
                try:
                    await self.graph_vector_retriever.delete_entries_batch(
                        old_vector_doc_ids
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # The graph tables already reference the new generation. Old
                    # vectors are redundant but harmless and can be cleaned later.
                    logger.warning(
                        "图谱已切换，但旧图向量清理失败；保留新一代索引",
                        exc_info=True,
                    )
            return {"rebuilt": rebuilt, "skipped": skipped}
        except BaseException:
            if not switched and new_vector_doc_ids:
                try:
                    await self.graph_vector_retriever.delete_entries_batch(
                        {
                            source_memory_id: sorted(vector_doc_ids)
                            for source_memory_id, vector_doc_ids in new_vector_doc_ids.items()
                        }
                    )
                except Exception:
                    pass
            if not switched:
                async with self._rebuild_gate:
                    pending_delta = {**replay_delta, **self._rebuild_delta}
                    self._rebuild_delta.clear()
                    try:
                        for source_memory_id, payload in pending_delta.items():
                            try:
                                if payload is None:
                                    await self._delete_memory_now(source_memory_id)
                                else:
                                    await self._index_memory_now(
                                        source_memory_id, *payload
                                    )
                            except Exception:
                                logger.exception(
                                    "Graph rollback replay failed for memory %s; write log retains repair state",
                                    source_memory_id,
                                )
                    finally:
                        self._rebuild_active = False
            raise
        finally:
            # B10：shadow_store 现在持有可复用的空闲连接（连接池），必须先关闭再删
            # 目录；否则 Windows 上文件仍被占用会让 rmtree 静默失败、临时目录泄漏。
            try:
                await shadow_store.close()
            except Exception:
                logger.warning(
                    "关闭图谱重建影子库连接失败，临时目录可能残留", exc_info=True
                )
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def plan_stale_vector_prune(self) -> RepairPlan:
        """Classify graph documents against the live source-validity rule."""
        async with self.source_state_guard.hold():
            async with self._rebuild_gate:
                if self._rebuild_active:
                    raise RuntimeError(
                        "graph index rebuild is active; retry dry-run later"
                    )
            if self.source_validity_checker is None:
                raise RuntimeError("graph source validity checker is unavailable")

            (
                rows,
                indexed_ids,
            ) = await self.graph_vector_retriever.get_index_document_inventory()
            decoded: list[tuple[int, int, str, str]] = []
            source_ids: set[int] = set()
            for row in rows:
                metadata = row.get("metadata")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                try:
                    document_id = int(row["id"])
                    source_memory_id = int(metadata["source_memory_id"])
                except (KeyError, TypeError, ValueError):
                    # Malformed/missing source metadata has no active-source proof.
                    document_id = int(row["id"])
                    source_memory_id = -1
                text = str(row.get("text") or "")
                decoded.append(
                    (document_id, source_memory_id, content_digest(text), text)
                )
                if source_memory_id >= 0:
                    source_ids.add(source_memory_id)

            active_source_ids = await self.source_validity_checker(
                tuple(sorted(source_ids))
            )
            documents = [
                GraphIndexDocument(
                    document_id=document_id,
                    source_memory_id=source_memory_id,
                    source_is_active=source_memory_id in active_source_ids,
                    content_sha256=digest,
                )
                for document_id, source_memory_id, digest, _ in decoded
            ]
            return plan_graph_index_repair(
                documents,
                indexed_ids=indexed_ids,
                persisted_vectors={},
                expected_model_id=None,
                index_dimension=int(
                    getattr(
                        getattr(
                            self.graph_vector_retriever.faiss_db,
                            "embedding_storage",
                            None,
                        ),
                        "dimension",
                        0,
                    )
                    or 0
                ),
            )

    async def apply_stale_vector_prune(self, plan: RepairPlan) -> tuple[int, ...]:
        """Serialize exact stale-vector cleanup against graph writes/rebuilds."""
        async with self.source_state_guard.hold():
            async with self._rebuild_gate:
                if self._rebuild_active:
                    raise RuntimeError(
                        "graph index rebuild is active; retry prune later"
                    )
                return await self._apply_stale_vector_prune_locked(plan)

    async def source_graph_index_complete(self, source_memory_id: int) -> bool:
        """Check that graph-store vector references for a source still exist."""
        async with self._rebuild_gate:
            if self._rebuild_active:
                return False
            referenced = await self.graph_store.list_vector_doc_ids_by_source()
            required_ids = set(referenced.get(int(source_memory_id), []))
            if not required_ids:
                return True
            indexed_ids = self.graph_vector_retriever._indexed_ids()
            return required_ids <= indexed_ids

    async def _apply_stale_vector_prune_locked(
        self, plan: RepairPlan
    ) -> tuple[int, ...]:
        """Revalidate then remove only the plan's stale FAISS IDs.

        The document-storage rows and historical graph entries remain intact.
        Source state is queried again immediately before the index mutation.
        """
        if self.source_validity_checker is None:
            raise RuntimeError("graph source validity checker is unavailable")
        stale_ids = tuple(plan.stale_index_ids)
        if not stale_ids:
            return ()
        documents = await self.graph_vector_retriever.get_documents_by_ids(
            list(stale_ids)
        )
        by_id = {int(row["id"]): row for row in documents}
        expected = {
            document_id: (source_id, digest)
            for document_id, source_id, digest in plan.prune_bindings
        }
        if set(expected) != set(stale_ids):
            raise ValueError(
                "stale-vector plan bindings do not match its exact ID scope"
            )
        current_source_ids: set[int] = set()
        for document_id in stale_ids:
            row = by_id.get(document_id)
            if row is None:
                raise ValueError("stale graph document disappeared after planning")
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            source_memory_id = int(metadata.get("source_memory_id", -1))
            digest = content_digest(str(row.get("text") or ""))
            if expected[document_id] != (source_memory_id, digest):
                raise ValueError(
                    "stale graph document source binding changed after planning"
                )
            current_source_ids.add(source_memory_id)

        active_source_ids = await self.source_validity_checker(
            tuple(sorted(current_source_ids))
        )
        if current_source_ids & active_source_ids:
            raise ValueError(
                "a planned stale source became active; create a fresh plan"
            )
        return await self.graph_vector_retriever.delete_vector_ids_atomically(
            list(stale_ids)
        )


__all__ = ["GraphMemoryManager"]
