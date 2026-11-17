"""记忆写入、更新、删除与来源存储。"""

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any

import aiosqlite

from astrbot.api import logger

from ...storage.atom_store import AtomStore
from ..memory_revision import MemoryRevisionConflict, memory_revision
from ..memory_scope import (
    classify_umo,
    current_adapter_access,
    infer_adapter_id,
)
from ..processors.atom_classifier import classify_atoms
from ..retrieval.hybrid_retriever import HybridResult
from ..utils.json_utils import safe_json_dict
from ..utils.number_utils import clamp_float, safe_float
from .agentic_recall_types import AgenticRecallOutcome as AgenticRecallOutcome

# 合并后的正文上限：两段摘要拼接后的长度上限，超过则只保留更完整的一段，
# 避免同一记忆在多轮合并后无限膨胀（会放大 embedding/FTS 写入成本）。
_MERGED_CONTENT_MAX_CHARS = 1200
# 句子级切分：中英文句末标点与换行；保留标点由正则后行断言完成。
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;\n])")


class MemoryEngineCrudMixin:
    """MemoryEngine 的 MemoryEngineCrudMixin 分组。

    宿主共享状态由 ``MemoryEngine`` 独占持有，本模块只通过 self 访问。
    """

    # 宿主共享状态契约（由 MemoryEngine.__init__ / initialize 赋值）
    db_connection: aiosqlite.Connection | None
    config: dict[str, Any]
    db_path: str
    faiss_db: Any
    atom_enabled: bool
    atom_store: AtomStore | None
    atom_retriever: Any | None
    vector_retriever: Any | None
    bm25_retriever: Any | None
    hybrid_retriever: Any | None
    index_writer: Any | None
    dual_route_retriever: Any | None
    graph_memory_manager: Any | None
    graph_store: Any | None
    _search_cache_enabled: bool
    _search_cache_ttl: float
    _search_cache_max_size: int
    _search_cache_generation: int
    _search_cache: OrderedDict[tuple[Any, ...], tuple[float, list[HybridResult]]]
    _access_update_pending: set[int]
    _access_atom_update_pending: set[int]
    _access_flush_task: asyncio.Task | None
    index_maintenance_status: dict[str, Any]

    async def add_memory(
        self,
        content: str,
        session_id: str | None = None,
        persona_id: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        atoms: list | None = None,
        preserve_create_time: bool = False,
        source_messages: list[dict[str, Any]] | None = None,
        source_validator=None,
        source_publish_lease=None,
        embedding: list[float] | None = None,
        *,
        _publish_candidate=None,
        source_session_id: str | None = None,
    ) -> int:
        """
        添加新记忆

        Args:
            content: 记忆内容
            session_id: 会话ID(支持多种格式,自动提取UUID)
            persona_id: 人格ID(支持多种格式,自动提取UUID)
            importance: 重要性(0-1)
            metadata: 额外元数据

        Returns:
            int: 记忆ID(doc_id)
        """
        if not content or not content.strip():
            raise ValueError("记忆内容不能为空")
        if source_validator is not None:
            if not callable(source_publish_lease):
                raise ValueError("source_publication_lease_required")
            valid = source_validator()
            if asyncio.iscoroutine(valid) or hasattr(valid, "__await__"):
                valid = await valid
            if valid is not True:
                raise ValueError("source_snapshot_changed")

        op_id = await self._start_write_op(
            "add",
            {
                "content_preview": content[:500],
                "content": content,
                "source_messages": source_messages,
                "session_id": session_id,
                "persona_id": persona_id,
                "importance": importance,
                "metadata": metadata or {},
                "source_validation_required": source_validator is not None,
                "atoms": [
                    self._serialize_atom_for_repair(atom) for atom in (atoms or [])
                ],
            },
        )

        # 准备完整元数据 - 保存完整的 unified_msg_origin，不提取UUID
        # 只在查询/过滤时才提取UUID进行匹配，存储时保留完整信息
        current_time = time.time()
        full_metadata = {
            "session_id": session_id,  # 保存完整的 unified_msg_origin
            "persona_id": persona_id,  # 保存完整的 persona_id
            "importance": max(0.0, min(1.0, importance)),  # 限制在0-1范围
            "create_time": current_time,
            "last_access_time": current_time,
            "status": "active",
        }

        # 合并用户提供的额外元数据
        # 注意：先合并外部metadata，再确保时间字段不被覆盖
        if metadata:
            full_metadata.update(metadata)
        full_metadata.pop("source_publication_verified_v1", None)
        access = current_adapter_access.get()
        full_metadata["adapter_id"] = full_metadata.get("adapter_id") or (
            access.adapter_id
            if access is not None
            else infer_adapter_id(
                full_metadata,
                source_messages or [],
                self.config.get("adapter_instances"),
            )
        )
        if source_validator is not None:
            # Keep the parent invisible through embedding and all index prep.
            full_metadata["status"] = "pending_source_validation"
        # Conversation scope is trusted from the caller's original UMO, never
        # from processor/model metadata. This also keeps a private summary
        # private when it cites group-source facts.
        source_origin = (
            source_session_id
            if source_validator is not None and source_session_id
            else session_id
        )
        source_kind = classify_umo(source_origin)
        full_metadata["session_id"] = session_id
        full_metadata["persona_id"] = persona_id
        full_metadata["source_session_id"] = source_origin
        full_metadata["interaction_type"] = {
            "group": "group_chat",
            "private": "private_chat",
        }.get(source_kind, "unknown")
        full_metadata["memory_visibility"] = (
            "group" if source_kind == "group" else "private"
        )
        full_metadata["source_platform"] = (
            source_origin.split(":", 1)[0]
            if source_kind == "group"
            and isinstance(source_origin, str)
            and ":" in source_origin
            else ""
        )
        if source_messages:
            full_metadata["has_source"] = True
            full_metadata["source_message_count"] = len(source_messages)
        if atoms:
            full_metadata["atom_types"] = sorted(
                {
                    getattr(getattr(atom, "atom_type", None), "value", "unknown")
                    for atom in atoms
                }
            )

        # 普通新增使用当前时间；物理替换保留原记忆的时间轴位置。
        preserved_create_time = None
        if preserve_create_time and metadata:
            try:
                preserved_create_time = float(metadata.get("create_time"))
            except (TypeError, ValueError):
                preserved_create_time = None
        full_metadata["create_time"] = (
            preserved_create_time if preserved_create_time is not None else current_time
        )
        full_metadata["last_access_time"] = current_time

        if op_id is not None:
            full_metadata["write_op_id"] = op_id

        # 通过混合检索器添加(会同时添加到BM25和向量索引)
        if self.hybrid_retriever is None:
            raise RuntimeError("混合检索器未初始化")
        try:
            if source_validator is not None:
                valid = source_validator()
                if asyncio.iscoroutine(valid) or hasattr(valid, "__await__"):
                    valid = await valid
                if valid is not True:
                    await self._advance_write_op(
                        op_id,
                        "source_changed",
                        status="failed",
                        error="source snapshot changed before index write",
                    )
                    raise ValueError("source_snapshot_changed")
            doc_id = await self.hybrid_retriever.add_memory(
                content,
                full_metadata,
                source_validator=source_validator,
                **({"embedding": embedding} if embedding is not None else {}),
            )
            await self._advance_write_op(
                op_id,
                "document_indexed",
                memory_id=doc_id,
                payload_patch={"memory_id": doc_id},
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._advance_write_op(
                op_id,
                "document_failed",
                status="failed",
                error=str(e),
            )
            raise

        # 写入记忆原子
        atom_write_failed = False
        if atoms and self.atom_store is not None and self.atom_enabled:
            prepared_atoms = []
            for atom in atoms:
                atom.session_id = atom.session_id or session_id
                atom.persona_id = atom.persona_id or persona_id
                atom.parent_memory_id = doc_id
                atom.metadata["adapter_id"] = full_metadata.get("adapter_id")
                prepared_atoms.append(atom)
            try:
                await self.atom_store.insert_many(prepared_atoms)
                await self._advance_write_op(
                    op_id,
                    "atoms_indexed",
                    memory_id=doc_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._advance_write_op(
                    op_id,
                    "atoms_failed",
                    status="failed",
                    memory_id=doc_id,
                    error="atom insert failed",
                )
                await self.delete_memory(doc_id)
                raise
        else:
            await self._advance_write_op(op_id, "atoms_skipped", memory_id=doc_id)

        needs_repair = atom_write_failed
        if self.graph_memory_manager is not None:
            try:
                graph_metadata = full_metadata
                if source_validator is not None:
                    # Graph candidates are also prepared before publication;
                    # the parent document remains pending and the graph source
                    # checker excludes it until the source lease activates it.
                    graph_metadata = {**full_metadata, "status": "active"}
                indexed = await self.graph_memory_manager.index_memory(
                    doc_id, content, graph_metadata, atoms
                )
                if indexed is False:
                    raise RuntimeError(
                        "Graph indexing deferred until rebuild completes"
                    )
                await self._advance_write_op(
                    op_id,
                    "graph_indexed",
                    status="needs_repair" if needs_repair else "pending",
                    memory_id=doc_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._advance_write_op(
                    op_id,
                    "graph_failed",
                    status="needs_repair",
                    memory_id=doc_id,
                    error=str(e),
                )
                needs_repair = True
                logger.error(
                    f"[MemoryEngine] 图记忆索引失败，已标记待修复 (memory_id={doc_id})",
                    exc_info=True,
                )
        else:
            await self._advance_write_op(
                op_id,
                "graph_skipped",
                status="needs_repair" if needs_repair else "pending",
                memory_id=doc_id,
            )

        if source_messages:
            try:
                await self.save_memory_source(doc_id, source_messages)
            except asyncio.CancelledError:
                await asyncio.shield(self.delete_memory(doc_id))
                await asyncio.shield(
                    self._advance_write_op(
                        op_id,
                        "source_failed",
                        status="failed",
                        memory_id=doc_id,
                        error="source write cancelled",
                    )
                )
                raise
            except Exception as exc:
                await self._advance_write_op(
                    op_id,
                    "source_failed",
                    status="failed",
                    memory_id=doc_id,
                    error=str(exc),
                )
                if not await self.delete_memory(doc_id):
                    logger.error(
                        f"[MemoryEngine] 原文写入失败且记忆回滚失败 (memory_id={doc_id})"
                    )
                raise
        if source_validator is not None:
            source_current = False
            published = False

            async def publish_current_source():
                if _publish_candidate is not None:
                    return await _publish_candidate(doc_id, op_id, needs_repair)
                async with (
                    self.source_state_guard.hold(),
                    self._write_transaction() as db,
                ):
                    return await self._publish_verified_summary(
                        db, doc_id, op_id, needs_repair
                    )

            try:
                async with source_publish_lease() as source_current:
                    if source_current is True and self.hybrid_retriever is not None:
                        from .memory_engine_batch import _await_task_to_completion

                        publication = asyncio.create_task(publish_current_source())
                        cancelled, error = await _await_task_to_completion(publication)
                        published = error is None and publication.result()
                        if cancelled:
                            # A durable publication stays visible. The caller
                            # can recover its receipt from the original snapshot.
                            if published:
                                self._invalidate_search_cache()
                            cancellation = asyncio.CancelledError()
                            if published:
                                cancellation.published_memory_id = doc_id
                            raise cancellation
                        if error is not None:
                            raise error
            except asyncio.CancelledError as cancellation:
                if published:
                    self._invalidate_search_cache()
                    cancellation.published_memory_id = doc_id
                raise
            except Exception:
                if not published:
                    source_current = False
            if source_current is not True or published is not True:
                removed = await self.delete_memory(doc_id)
                await self._advance_write_op(
                    op_id,
                    "source_publication_rejected",
                    status=("failed" if removed else "needs_repair"),
                    memory_id=doc_id,
                    error="source snapshot changed or active publication failed",
                )
                raise ValueError("source_snapshot_changed")
        if not needs_repair and source_validator is None:
            await self._advance_write_op(
                op_id,
                "completed",
                status="completed",
                memory_id=doc_id,
            )
        superseded_ids = full_metadata.get("superseded_memory_ids")
        if superseded_ids:
            await self.mark_memories_superseded(superseded_ids, doc_id)
        self._invalidate_search_cache()
        return doc_id

    async def _publish_verified_summary(self, db, doc_id, op_id, needs_repair) -> bool:
        """Atomically publish the document and its restart-safe write receipt."""
        if type(op_id) is not int or op_id <= 0:
            return False
        async with db.execute(
            "SELECT metadata FROM documents WHERE id=?", (doc_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return False
        metadata = safe_json_dict(row[0])
        if (
            metadata.get("status") != "pending_source_validation"
            or metadata.get("write_op_id") != op_id
        ):
            return False
        step = "source_published_graph_pending" if needs_repair else "source_published"
        receipt = await db.execute(
            """
            UPDATE memory_write_ops SET status='completed', step=?, updated_at=?
            WHERE id=? AND memory_id=? AND op_type='add'
        """,
            (step, time.time(), op_id, doc_id),
        )
        if receipt.rowcount != 1:
            return False
        metadata.update(
            status="active",
            source_publication_verified_v1={
                "write_op_id": op_id,
                "memory_id": doc_id,
            },
        )
        await db.execute(
            "UPDATE documents SET metadata=? WHERE id=?",
            (json.dumps(metadata, ensure_ascii=False), doc_id),
        )
        return True

    def _build_memory_metadata(
        self,
        *,
        session_id: str | None,
        persona_id: str | None,
        importance: float,
        metadata: dict[str, Any] | None,
        atoms: list | None,
        preserve_create_time: bool,
        source_messages: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """合并调用方元数据与时间/原子类型字段，返回待写入的完整元数据。"""
        current_time = time.time()
        full_metadata: dict[str, Any] = {
            "session_id": session_id,  # 保存完整的 unified_msg_origin
            "persona_id": persona_id,  # 保存完整的 persona_id
            "importance": max(0.0, min(1.0, importance)),  # 限制在0-1范围
            "create_time": current_time,
            "last_access_time": current_time,
            "status": "active",
        }

        # 先合并外部 metadata，再确保时间字段不被覆盖
        if metadata:
            full_metadata.update(metadata)
        if source_messages:
            full_metadata["has_source"] = True
            full_metadata["source_message_count"] = len(source_messages)
        if atoms:
            full_metadata["atom_types"] = sorted(
                {
                    getattr(getattr(atom, "atom_type", None), "value", "unknown")
                    for atom in atoms
                }
            )

        # 普通新增使用当前时间；物理替换保留原记忆的时间轴位置。
        preserved_create_time = None
        if preserve_create_time and metadata:
            try:
                preserved_create_time = float(metadata.get("create_time"))
            except (TypeError, ValueError):
                preserved_create_time = None
        full_metadata["create_time"] = (
            preserved_create_time if preserved_create_time is not None else current_time
        )
        full_metadata["last_access_time"] = current_time
        return full_metadata

    async def _write_memory_atoms(
        self,
        doc_id: int,
        session_id: str | None,
        persona_id: str | None,
        atoms: list | None,
        adapter_id: str | None = None,
    ) -> None:
        """Insert owned atoms once and propagate storage failures to the caller."""
        if not atoms or self.atom_store is None or not self.atom_enabled:
            return
        prepared_atoms = []
        for atom in atoms:
            atom.session_id = atom.session_id or session_id
            atom.persona_id = atom.persona_id or persona_id
            atom.metadata["adapter_id"] = adapter_id
            atom.parent_memory_id = doc_id
            prepared_atoms.append(atom)
        await self.atom_store.insert_many(prepared_atoms)

    async def _index_memory_graph(
        self,
        doc_id: int,
        content: str,
        full_metadata: dict[str, Any],
        atoms: list | None,
    ) -> None:
        """建立图记忆索引；失败只记录日志，由重建对账。"""
        if self.graph_memory_manager is None:
            return
        try:
            indexed = await self.graph_memory_manager.index_memory(
                doc_id, content, full_metadata, atoms
            )
            if indexed is False:
                raise RuntimeError("Graph indexing deferred until rebuild completes")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                f"[MemoryEngine] 图记忆索引失败 (memory_id={doc_id})",
                exc_info=True,
            )

    async def _persist_memory_source(
        self,
        doc_id: int,
        source_messages: list[dict[str, Any]] | None,
    ) -> None:
        """保存原文消息；失败时回滚刚写入的记忆。"""
        if not source_messages:
            return
        try:
            await self.save_memory_source(doc_id, source_messages)
        except asyncio.CancelledError:
            await asyncio.shield(self.delete_memory(doc_id))
            raise
        except Exception:
            if not await self.delete_memory(doc_id):
                logger.error(
                    f"[MemoryEngine] 原文写入失败且记忆回滚失败 (memory_id={doc_id})"
                )
            raise

    async def save_memory_source(
        self, memory_id: int, source_messages: list[dict[str, Any]]
    ) -> None:
        """Persist source messages outside retrieval metadata and indexes."""
        now = time.time()
        async with self._write_db() as db:
            await db.execute(
                """
                INSERT INTO memory_sources(memory_id, source_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    source_json = excluded.source_json,
                    updated_at = excluded.updated_at
                """,
                (
                    int(memory_id),
                    json.dumps(source_messages, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    async def update_memory(
        self,
        memory_id: int,
        updates: dict[str, Any],
        *,
        expected_revision: str | None = None,
    ) -> bool:
        """
        更新记忆（确保多数据库同步）

        支持更新内容、重要性、元数据等。采用不同策略：
        - 内容更新：先创建后删除（避免数据丢失）+ 全库同步
        - 元数据更新：三库同步更新

        Args:
            memory_id: 记忆ID
            updates: 更新字典,可包含:
                - content: 新内容 (触发完整重建)
                - importance: 新重要性
                - metadata: 元数据更新

        Returns:
            bool: 是否更新成功
        """
        if "content" in updates:
            return await self._update_memory(memory_id, updates, expected_revision)
        async with self.source_state_guard.hold():
            return await self._update_memory(memory_id, updates, expected_revision)

    async def _update_memory(self, memory_id, updates, expected_revision):
        if expected_revision is not None or "content" in updates:
            expected_revision = await self._check_memory_revision(
                memory_id, expected_revision
            )
        memory = await self.get_memory(memory_id)
        if not memory:
            logger.error(f"[更新] 记忆不存在 (memory_id={memory_id})")
            return False

        # 解析 metadata（可能是JSON字符串）
        current_metadata = safe_json_dict(memory.get("metadata", {}))

        # 处理内容更新 (需要重建所有索引)
        if "content" in updates:
            new_content = updates["content"]
            if not new_content or not new_content.strip():
                return False

            try:
                importance = clamp_float(
                    updates.get("importance", current_metadata.get("importance", 0.5)),
                    default=0.5,
                )

                # 构建新元数据
                new_metadata = {}
                metadata_patch = updates.get("metadata")
                if isinstance(metadata_patch, dict):
                    new_metadata.update(metadata_patch)
                new_metadata["updated_at"] = time.time()
                new_metadata["previous_id"] = memory_id  # 记录旧ID
                new_metadata["importance"] = importance

                # 【改进】先创建新记忆，再删除旧记忆（避免数据丢失）
                logger.info(f"[更新] 开始内容更新流程 (old_id={memory_id})")

                new_memory_id = await self.replace_memory(
                    memory_id,
                    content=new_content,
                    importance=importance,
                    metadata=new_metadata,
                    expected_revision=expected_revision,
                )

                logger.info(
                    f"[更新] 内容更新完成 (old_id={memory_id} → new_id={new_memory_id})"
                )
                self._invalidate_search_cache()
                return True

            except asyncio.CancelledError:
                raise
            except MemoryRevisionConflict:
                raise
            except Exception as e:
                logger.error(
                    f"[更新] 内容更新失败 (memory_id={memory_id}): {e}", exc_info=True
                )
                return False

        # 处理非内容的元数据更新（不需要重建索引）
        metadata_updates = {}

        if "importance" in updates:
            metadata_updates["importance"] = clamp_float(
                updates["importance"], default=0.5
            )

        if "metadata" in updates:
            metadata_updates.update(updates["metadata"])

        if metadata_updates:
            if "status" in metadata_updates:
                return await self._update_memory_status_guarded(
                    memory_id,
                    metadata_updates,
                )

            # 确保 current_metadata 是字典（再次检查）
            current_metadata = safe_json_dict(current_metadata)

            # 合并元数据
            current_metadata.update(metadata_updates)
            current_metadata["updated_at"] = time.time()
            metadata_updates.setdefault("updated_at", current_metadata["updated_at"])

            # 使用写侧索引维护器确保多存储元数据同步
            if self.index_writer is None:
                logger.error("索引写入器未初始化")
                return False
            success = await self.index_writer.update_metadata(
                memory_id, metadata_updates
            )

            if success:
                logger.info(f"[更新] 元数据更新成功 (memory_id={memory_id})")
                # 仅当更新触及图谱提取依赖的字段时才重建图谱索引；
                # 状态/类型等纯元数据编辑重建整图（删节点+FAISS 落盘）纯属浪费
                graph_affecting = metadata_updates.keys() & {
                    "participants",
                    "participant_identities",
                    "topics",
                    "key_facts",
                    "canonical_summary",
                    "session_id",
                    "persona_id",
                    "importance",
                    "create_time",
                    "summary_schema_version",
                    "source_window",
                }
                if graph_affecting and self.graph_memory_manager is not None:
                    await self.graph_memory_manager.index_memory(
                        memory_id,
                        memory["text"],
                        current_metadata,
                    )
                self._invalidate_search_cache()
            else:
                logger.error(f"[更新] 元数据更新失败 (memory_id={memory_id})")

            return success

        return True

    async def replace_memory(
        self,
        memory_id: int,
        *,
        content: str,
        metadata: dict[str, Any],
        importance: float,
        expected_revision: str | None = None,
        expected_source: list[dict[str, Any]] | None = None,
    ) -> int:
        """Replace one memory and rebuild all derived data with a new ID."""
        expected_revision = await self._check_memory_revision(
            memory_id, expected_revision
        )
        current = await self.get_memory(memory_id)
        if not current:
            raise ValueError(f"记忆不存在 (memory_id={memory_id})")
        if not content or not content.strip():
            raise ValueError("记忆内容不能为空")

        current_metadata = safe_json_dict(current.get("metadata"))
        source_messages = await self.get_memory_source(memory_id)
        if expected_source is not None and source_messages != expected_source:
            raise MemoryRevisionConflict()
        replacement_metadata = current_metadata.copy()
        replacement_metadata.update(metadata or {})
        replacement_metadata["previous_id"] = memory_id
        replacement_metadata["updated_at"] = time.time()
        replacement_metadata["create_time"] = current_metadata.get("create_time")
        normalized_importance = clamp_float(importance, default=0.5)
        replacement_metadata["importance"] = normalized_importance

        session_id = replacement_metadata.get("session_id")
        persona_id = replacement_metadata.get("persona_id")
        raw_key_facts = replacement_metadata.get("key_facts")
        raw_topics = replacement_metadata.get("topics")
        raw_participants = replacement_metadata.get("participants")
        key_facts = raw_key_facts if isinstance(raw_key_facts, list) else []
        topics = raw_topics if isinstance(raw_topics, list) else []
        participants = raw_participants if isinstance(raw_participants, list) else []
        atoms = []
        if self.atom_enabled:
            atoms = classify_atoms(
                key_facts=key_facts,
                topics=topics,
                participants=participants,
                parent_importance=normalized_importance,
                session_id=session_id,
                persona_id=persona_id,
            )

        new_memory_id: int | None = None
        preserve_replacement = False
        add_task = self._create_tracked_task(
            self.add_memory(
                content=content,
                session_id=session_id,
                persona_id=persona_id,
                importance=normalized_importance,
                metadata=replacement_metadata,
                atoms=atoms,
                preserve_create_time=True,
                source_messages=source_messages or None,
            )
        )
        from .memory_engine_batch import _await_task_to_completion

        try:
            new_memory_id = await asyncio.shield(add_task)
            if new_memory_id is None:
                raise RuntimeError("新记忆创建失败")
            async with self.source_state_guard.hold():
                if expected_revision is not None:
                    await self._check_memory_revision(memory_id, expected_revision)
                if not await self.get_memory(memory_id):
                    raise MemoryRevisionConflict()
                if await self.get_memory_source(memory_id) != source_messages:
                    raise MemoryRevisionConflict()
                try:
                    if not await self.delete_memory(memory_id):
                        raise RuntimeError("旧记忆删除失败")
                except (asyncio.CancelledError, Exception) as exc:
                    # The host may delete the primary row before auxiliary
                    # cleanup returns. Reconcile while still holding the fence.
                    probe = asyncio.create_task(self.get_memory(memory_id))
                    probe_cancelled, probe_error = await _await_task_to_completion(
                        probe
                    )
                    preserve_replacement = (
                        probe_error is not None or probe.result() is None
                    )
                    if preserve_replacement:
                        repair = asyncio.create_task(
                            self._mark_index_rebuild_needed(
                                "replacement_delete_interrupted", memory_id
                            )
                        )
                        repair_cancelled, _ = await _await_task_to_completion(repair)
                        probe_cancelled |= repair_cancelled
                    if probe_cancelled:
                        raise asyncio.CancelledError
                    if (
                        isinstance(exc, asyncio.CancelledError)
                        or probe_error is not None
                        or not preserve_replacement
                    ):
                        raise
                    logger.warning(
                        "[更新] 旧记忆主记录已删除，需复核其衍生索引 (memory_id=%s)",
                        memory_id,
                    )
            self._invalidate_search_cache()
            return new_memory_id
        except asyncio.CancelledError:
            if new_memory_id is None:
                _, add_error = await _await_task_to_completion(add_task)
                if add_error is None:
                    new_memory_id = add_task.result()
            if new_memory_id is not None and not preserve_replacement:
                await self._rollback_replacement(new_memory_id)
            self._invalidate_search_cache()
            raise
        except Exception:
            if new_memory_id is not None and not preserve_replacement:
                await self._rollback_replacement(new_memory_id)
            self._invalidate_search_cache()
            raise

    async def _rollback_replacement(self, memory_id):
        from .memory_engine_batch import _await_task_to_completion

        cleanup = self._create_tracked_task(self.delete_memory(memory_id))
        cancelled, error = await _await_task_to_completion(cleanup)
        if error is not None or not cleanup.result():
            logger.error("[更新] 回滚新记忆失败 (memory_id=%s)", memory_id)
        if cancelled:
            raise asyncio.CancelledError

    async def _check_memory_revision(self, memory_id, expected_revision):
        current = await self.get_memory_record(memory_id)
        revision = memory_revision(current) if current is not None else None
        if expected_revision is not None and revision != expected_revision:
            raise MemoryRevisionConflict()
        return revision

    async def delete_memory(self, memory_id: int) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否删除成功
        """

        async with self.source_state_guard.hold():
            return await self._delete_memory(memory_id)

    async def _delete_memory(self, memory_id: int) -> bool:
        if (
            current_adapter_access.get() is not None
            and await self.get_memory(memory_id) is None
        ):
            return False

        # index_writer.delete_memory() 内部已按顺序删除 BM25、向量索引和 documents 表
        if self.index_writer is None:
            logger.error("索引写入器未初始化")
            return False
        success = await self.index_writer.delete_memory(memory_id)
        if not success:
            return False

        # Searches must not reuse the deleted document while derived data is
        # removed or if cancellation interrupts that removal.
        self._invalidate_search_cache()
        if self.db_connection is not None:
            async with self._write_db() as db:
                await db.execute(
                    "DELETE FROM memory_sources WHERE memory_id = ?", (memory_id,)
                )

        try:
            if self.graph_memory_manager is not None:
                deleted = await self.graph_memory_manager.delete_memory(memory_id)
                if deleted is False:
                    raise RuntimeError(
                        "Graph deletion deferred until rebuild completes"
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                f"[MemoryEngine] 图记忆删除失败 (memory_id={memory_id})",
                exc_info=True,
            )

        try:
            if self.atom_store is not None:
                await self.atom_store.delete_by_parent(memory_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                f"[MemoryEngine] 记忆原子删除失败 (memory_id={memory_id})",
                exc_info=True,
            )

        return success

    async def mark_memories_superseded(
        self, memory_ids: list[int], by_memory_id: int
    ) -> int:
        """把被取代的旧记忆标记为 superseded（不删除、不归档）。

        仅在 metadata 写入 ``superseded_by`` / ``superseded_at``：旧记忆保留在
        存储与索引中，由召回侧的 ``_filter_by_retrieval_policy`` 过滤，便于审计
        或将来恢复。标记失败只记录日志，不影响新记忆已完成的写入。

        Args:
            memory_ids: 被取代的记忆 id 列表。
            by_memory_id: 取代它们的新记忆 id。

        Returns:
            int: 实际标记成功的记忆条数。
        """
        if self.db_connection is None:
            return 0
        marked = 0
        now = time.time()
        for memory_id in memory_ids:
            if memory_id == by_memory_id:
                continue
            try:
                updated = await self.update_memory(
                    int(memory_id),
                    {
                        "metadata": {
                            "superseded_by": int(by_memory_id),
                            "superseded_at": now,
                        }
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    f"[MemoryEngine] 标记被取代记忆失败 (memory_id={memory_id})",
                    exc_info=True,
                )
                continue
            if updated:
                marked += 1
        if marked:
            logger.info(
                f"[MemoryEngine] 已标记 {marked} 条被取代记忆 (superseded_by={by_memory_id})"
            )
        return marked

    async def merge_guarded_summary_memory(
        self,
        memory_id: int,
        *,
        session_id: str | None,
        persona_id: str | None,
        content: str,
        importance: float,
        metadata: dict,
        atoms: list,
        source_messages: list[dict] | None,
        source_validator,
        source_publish_lease,
    ) -> int | None:
        """Stage merged content, then publish it with the source and target fenced."""
        if not callable(source_publish_lease):
            raise ValueError("source_publication_lease_required")
        old = await self.get_memory_record(memory_id)
        if old is None:
            return None
        previous = safe_json_dict(old["metadata"])
        if (
            previous.get("status", "active") != "active"
            or previous.get("superseded_by")
            or previous.get("session_id") != session_id
            or previous.get("persona_id") != persona_id
            or previous.get("source_session_id") != metadata.get("source_session_id")
            or previous.get("memory_visibility") != metadata.get("memory_visibility")
        ):
            return None
        revision = memory_revision(old)
        previous_sources = await self.get_memory_source(memory_id)
        merged = {
            **previous,
            "dedup_merge_count": int(previous.get("dedup_merge_count") or 0) + 1,
            "dedup_last_merged_at": time.time(),
        }
        for key in (
            "topics",
            "key_facts",
            "participants",
            "participant_identities",
            "source_row_refs_v1",
            "narrative_provenance",
        ):
            if metadata.get(key):
                merged[key] = self._merge_unique_values(
                    previous.get(key), metadata[key]
                )
        if isinstance(metadata.get("source_window"), dict):
            merged["source_windows"] = [
                *(previous.get("source_windows") or []),
                metadata["source_window"],
            ][-20:]
        merged_content = self._merge_memory_text(str(old["text"] or ""), content)
        merged["canonical_summary"] = merged_content
        merged["previous_id"] = memory_id
        merged["guarded_merge_snapshot_v1"] = (metadata.get("source_window") or {}).get(
            "snapshot_fingerprint"
        )
        merged["updated_at"] = time.time()
        merged_importance = max(
            clamp_float(previous.get("importance"), default=0.5),
            clamp_float(importance, default=0.5),
        )
        dedup_config = self.config.get("memory_dedup") or {}
        merged_sources = previous_sources
        if source_messages and dedup_config.get("merge_sources", True):
            maximum = max(1, int(dedup_config.get("max_source_messages", 50) or 50))
            merged_sources = self._merge_unique_values(
                previous_sources, source_messages
            )[-maximum:]
        if merged_sources:
            merged["source_message_count"] = len(merged_sources)
        merged_atoms = (
            classify_atoms(
                key_facts=merged.get("key_facts") or [],
                topics=merged.get("topics") or [],
                participants=merged.get("participants") or [],
                parent_importance=merged_importance,
                session_id=session_id,
                persona_id=persona_id,
            )
            if self.atom_enabled
            else []
        )
        if self.atom_enabled and atoms:
            seen = {atom.content for atom in merged_atoms}
            merged_atoms.extend(atom for atom in atoms if atom.content not in seen)

        async def publish(candidate_id, op_id, needs_repair):
            # This callback runs inside the conversation source lease. The
            # target snapshot and the two document states commit together.
            async with self.source_state_guard.hold(), self._write_transaction() as db:
                async with db.execute(
                    "SELECT id, text, metadata FROM documents WHERE id IN (?, ?)",
                    (memory_id, candidate_id),
                ) as cursor:
                    docs = {row["id"]: dict(row) for row in await cursor.fetchall()}
                latest = docs.get(memory_id)
                candidate = docs.get(candidate_id)
                if (
                    latest is None
                    or candidate is None
                    or memory_revision(latest) != revision
                ):
                    return False
                async with db.execute(
                    "SELECT source_json FROM memory_sources WHERE memory_id=?",
                    (memory_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if (json.loads(row[0]) if row else []) != previous_sources:
                    return False
                inactive = safe_json_dict(latest["metadata"])
                if not await self._publish_verified_summary(
                    db, candidate_id, op_id, needs_repair
                ):
                    return False
                inactive.update(
                    status="archived",
                    superseded_by=candidate_id,
                    superseded_at=time.time(),
                )
                await db.execute(
                    "UPDATE documents SET metadata=? WHERE id=?",
                    (json.dumps(inactive, ensure_ascii=False), memory_id),
                )
                return True

        # A post-publication cancellation must not hide the already committed
        # replacement and leave its predecessor archived. Run the existing
        # durable staging workflow to its outcome before delivering cancellation.
        from .memory_engine_batch import _await_task_to_completion

        task = self._create_tracked_task(
            self.add_memory(
                content=merged_content,
                session_id=session_id,
                persona_id=persona_id,
                importance=merged_importance,
                metadata=merged,
                atoms=merged_atoms,
                source_messages=merged_sources or None,
                preserve_create_time=True,
                source_validator=source_validator,
                source_publish_lease=source_publish_lease,
                source_session_id=previous["source_session_id"],
                _publish_candidate=publish,
            )
        )
        cancelled, error = await _await_task_to_completion(task)
        if cancelled:
            cancellation = asyncio.CancelledError()
            if error is None:
                cancellation.published_memory_id = task.result()
                cancellation.published_as_merge = True
            elif getattr(error, "published_memory_id", None) is not None:
                cancellation.published_memory_id = error.published_memory_id
                cancellation.published_as_merge = True
            raise cancellation
        if isinstance(error, ValueError) and str(error) == "source_snapshot_changed":
            return None
        if error is not None:
            raise error
        return task.result()

    async def rollback_guarded_summary_merge(
        self, memory_id: int, fingerprint: str
    ) -> bool:
        """Withdraw a just-published replacement and restore its fenced predecessor."""
        async with self.source_state_guard.hold(), self._write_transaction() as db:
            async with db.execute(
                "SELECT metadata FROM documents WHERE id=?", (memory_id,)
            ) as cursor:
                row = await cursor.fetchone()
            merged = safe_json_dict(row[0]) if row else {}
            predecessor_id = merged.get("previous_id")
            if (
                merged.get("guarded_merge_snapshot_v1") != fingerprint
                or merged.get("status") != "active"
                or type(predecessor_id) is not int
            ):
                return False
            async with db.execute(
                "SELECT metadata FROM documents WHERE id=?", (predecessor_id,)
            ) as cursor:
                row = await cursor.fetchone()
            previous = safe_json_dict(row[0]) if row else {}
            if (
                previous.get("status") != "archived"
                or previous.get("superseded_by") != memory_id
            ):
                return False
            previous["status"] = "active"
            previous.pop("superseded_by", None)
            previous.pop("superseded_at", None)
            merged["status"] = "pending_source_validation"
            await db.executemany(
                "UPDATE documents SET metadata=? WHERE id=?",
                [
                    (json.dumps(previous, ensure_ascii=False), predecessor_id),
                    (json.dumps(merged, ensure_ascii=False), memory_id),
                ],
            )
        self._invalidate_search_cache()
        await self.delete_memory(memory_id)
        return True

    async def merge_duplicate_memory(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        importance: float,
        metadata: dict[str, Any] | None = None,
        atoms: list | None = None,
        source_messages: list[dict[str, Any]] | None = None,
    ) -> int | None:
        """Merge a new summary into an existing near-duplicate memory.

        The merged body keeps the more complete variant (or the de-duplicated
        concatenation of both). When the body changes it is rebuilt through
        :meth:`replace_memory`, so the embedding, BM25/FTS index, graph entries
        and atoms all describe the merged text instead of the stale old one.
        Metadata-only merges keep the current document id and sync the affected
        indexes in place.

        Args:
            memory_id: Existing memory document id.
            content: Incoming summary text; ``None`` keeps the existing body.
            importance: Importance of the incoming summary.
            metadata: Incoming summary metadata.
            atoms: Newly classified memory atoms.
            source_messages: Optional retained source messages.

        Returns:
            The surviving memory id (``memory_id`` or its replacement), or
            ``None`` when the merge could not be persisted.

        Raises:
            asyncio.CancelledError: Re-raised so rollback compensation still runs.
        """
        if self.db_connection is None:
            return None
        memory = await self.get_memory(memory_id)
        if not memory:
            return None
        metadata = metadata or {}
        current_metadata = safe_json_dict(memory.get("metadata"))
        current_content = str(memory.get("text") or "")
        merged_content = self._merge_memory_text(current_content, content or "")
        patch: dict[str, Any] = {}

        for key in ("topics", "key_facts", "participants", "participant_identities"):
            incoming = metadata.get(key)
            if not incoming:
                continue
            merged = self._merge_unique_values(current_metadata.get(key), incoming)
            if merged:
                patch[key] = merged

        now = time.time()
        source_window = metadata.get("source_window")
        if isinstance(source_window, dict):
            windows = list(current_metadata.get("source_windows") or [])
            windows.append(source_window)
            patch["source_windows"] = windows[-20:]

        patch["last_access_time"] = now
        patch["dedup_merge_count"] = (
            int(current_metadata.get("dedup_merge_count") or 0) + 1
        )
        patch["dedup_last_merged_at"] = now

        current_importance = clamp_float(
            current_metadata.get("importance"), default=0.5
        )
        merged_importance = max(
            current_importance, clamp_float(importance, default=0.5)
        )

        content_changed = merged_content.strip() != current_content.strip()
        if content_changed:
            patch["canonical_summary"] = merged_content
        try:
            if content_changed:
                # 内容变化必须重建向量/FTS/图/原子；replace_memory 先写新记忆再删旧，
                # 任何一步失败都会回滚，不会留下半同步状态。
                target_id = await self.replace_memory(
                    memory_id,
                    content=merged_content,
                    metadata=patch,
                    importance=merged_importance,
                )
            else:
                if not await self.update_memory(
                    memory_id,
                    {"importance": merged_importance, "metadata": patch},
                ):
                    return None
                target_id = memory_id
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                f"[MemoryEngine] 合并近重复记忆后重建索引失败 (memory_id={memory_id})",
                exc_info=True,
            )
            await self._mark_index_rebuild_needed(
                "dedup_merge_rebuild_failed", memory_id
            )
            return None

        if atoms and self.atom_store is not None and self.atom_enabled:
            await self._append_atoms_to_memory(target_id, atoms, metadata)

        dedup_cfg = self.config.get("memory_dedup") or {}
        if source_messages and bool(dedup_cfg.get("merge_sources", True)):
            max_sources = int(dedup_cfg.get("max_source_messages", 50) or 50)
            existing_sources = await self.get_memory_source(target_id)
            combined: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in [*existing_sources, *source_messages]:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                combined.append(item)
            try:
                await self.save_memory_source(target_id, combined[-max_sources:])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    f"[MemoryEngine] 合并来源消息失败 (memory_id={target_id})",
                    exc_info=True,
                )

        logger.info(
            f"[MemoryEngine] 已合并近重复记忆到 #{target_id}"
            f"（相似度合并次数={patch['dedup_merge_count']}，"
            f"正文{'已重建' if content_changed else '未变'}）"
        )
        return target_id

    @staticmethod
    def _merge_memory_text(existing: str, incoming: str) -> str:
        """Return the merged summary body for a dedup merge.

        Keeps the more complete variant when one body already contains the
        other, and otherwise concatenates the non-overlapping sentences so facts
        introduced by the newer summary stay retrievable. When the union exceeds
        ``_MERGED_CONTENT_MAX_CHARS`` the longer single body wins, which bounds
        the cost of re-embedding repeated merges.

        Args:
            existing: Body stored on the existing memory.
            incoming: Body of the incoming summary.

        Returns:
            Body to persist on the surviving memory.
        """
        left = str(existing or "").strip()
        right = str(incoming or "").strip()
        if not left:
            return right
        if not right:
            return left
        if left == right or right in left:
            return left
        if left in right:
            return right

        seen: set[str] = set()
        merged = ""
        for text in (left, right):
            for segment in _SENTENCE_BOUNDARY.split(text):
                piece = segment.strip()
                if not piece or piece in seen:
                    continue
                seen.add(piece)
                if not merged:
                    merged = piece
                elif merged[-1].isascii() and merged[-1] not in "。！？；\n":
                    merged = f"{merged} {piece}"
                else:
                    merged += piece
        if len(merged) > _MERGED_CONTENT_MAX_CHARS:
            return left if len(left) >= len(right) else right
        return merged or (left if len(left) >= len(right) else right)

    async def _mark_index_rebuild_needed(self, reason: str, memory_id: int) -> None:
        """Flag startup index maintenance after an unsafe merge rebuild.

        Args:
            reason: Short machine-readable reason recorded in the log.
            memory_id: Memory whose derived indexes may be stale.
        """
        logger.error(
            f"[MemoryEngine] {reason}: 检索索引可能未同步，"
            f"已标记 needs_index_rebuild 以便启动时重建 (memory_id={memory_id})"
        )
        writer = getattr(self, "_write_transaction", None)
        if self.db_connection is None or not callable(writer):
            return
        try:
            async with writer() as connection:
                await connection.execute(
                    "INSERT OR REPLACE INTO migration_status (key, value, updated_at) "
                    "VALUES (?, ?, datetime('now'))",
                    ("needs_index_rebuild", "true"),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                "[MemoryEngine] 写入 needs_index_rebuild 标记失败", exc_info=True
            )

    @staticmethod
    def _merge_unique_values(existing: Any, incoming: Any) -> list[Any]:
        """Union two metadata lists without reordering existing entries."""
        merged: list[Any] = []
        seen: set[str] = set()
        for item in [*(existing or []), *(incoming or [])]:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    async def _append_atoms_to_memory(
        self,
        memory_id: int,
        atoms: list,
        metadata: dict[str, Any],
    ) -> None:
        """Attach newly classified atoms to an existing memory, skipping duplicates."""
        if self.db_connection is None or self.atom_store is None:
            return
        cursor = await self.db_connection.execute(
            "SELECT content FROM memory_atoms WHERE parent_memory_id = ?",
            (int(memory_id),),
        )
        existing_contents = {
            str(row[0]).strip() for row in await cursor.fetchall() if row[0]
        }
        session_id = metadata.get("session_id")
        persona_id = metadata.get("persona_id")
        for atom in atoms:
            content = str(getattr(atom, "content", "") or "").strip()
            if not content or content in existing_contents:
                continue
            atom.metadata["adapter_id"] = metadata.get("adapter_id")
            atom.parent_memory_id = memory_id
            atom.session_id = atom.session_id or session_id
            atom.persona_id = atom.persona_id or persona_id
            try:
                await self.atom_store.insert(atom)
                existing_contents.add(content)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    f"[MemoryEngine] 合并记忆原子失败 (memory_id={memory_id}): {content[:60]}",
                    exc_info=True,
                )

    async def _delete_graph_and_atoms_for_batch(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        if self.graph_memory_manager is not None:
            deleted = await self.graph_memory_manager.batch_delete_memories(memory_ids)
            if deleted is False:
                raise RuntimeError("Graph deletion deferred until rebuild completes")
        if self.atom_store is not None:
            await self.atom_store.batch_delete_by_parent(memory_ids)

    async def _drop_legacy_documents_fts_triggers(
        self, connection: aiosqlite.Connection
    ):
        cursor = await connection.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND tbl_name='documents'
              AND sql LIKE '%documents_fts%'
        """)
        rows = await cursor.fetchall()
        for row in rows:
            trigger_name = row[0]
            await connection.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')
            logger.warning(f"已清理旧 LivingMemory FTS 触发器: {trigger_name}")

    async def search_narrative_history(
        self, query: str, k: int, origins: tuple[str, ...], personas: tuple[str, ...]
    ) -> list[HybridResult]:
        """Recall same-source historical material with a network-free baseline."""
        from ..retrieval.narrative_history import search_local

        return await search_local(self, query, k, origins, personas)

    async def search_narrative_semantic(
        self,
        query: str,
        k: int,
        origin: str,
        personas: tuple[str, ...],
        *,
        execution_context=None,
        diagnostics=None,
    ) -> list[HybridResult]:
        """Optional scoped semantic enrichment; do not migrate old IDs or use broad caches."""
        from ..retrieval.narrative_history import allowed_memory

        retriever = self.dual_route_retriever or self.hybrid_retriever
        if not str(query).strip() or retriever is None:
            return []
        # Keep final lifecycle validation on the original deadline. Optional
        # retrieval/reranking must return before consuming that reserve.
        from ..retrieval.narrative_plan import NarrativeSearchPlan

        retrieval_context = execution_context
        if execution_context is not None and hasattr(
            execution_context, "with_deadline"
        ):
            reserve = min(0.35, execution_context.budget.remaining() * 0.2)
            retrieval_context = execution_context.with_deadline(
                execution_context.budget.deadline - reserve
            )

        async def validate_candidates(rows):
            scoped = [
                row for row in rows if allowed_memory(row.metadata, (origin,), personas)
            ]
            return await self._apply_atom_policy(
                self._filter_by_retrieval_policy(scoped)
            )

        # Exact ceilings retain the previous expansion capacity, but each
        # layer receives its own budget instead of multiplying the last one.
        document = getattr(retriever, "document_retriever", retriever)
        outer_k = max(k * 4, 20)
        initial = max(20, int(getattr(document, "rerank_candidates", 20)))
        plan = NarrativeSearchPlan(
            k,
            validate_candidates,
            initial=initial,
            document_ceiling=outer_k
            * (2 if self.dual_route_retriever else 1)
            * int(getattr(document, "candidate_multiplier", 3)),
            graph_ceiling=outer_k * 2,
        )
        options = {
            "execution_context": retrieval_context,
            "diagnostics": diagnostics,
            "narrative_plan": plan,
        }
        results = await retriever.search(query, outer_k, origin, None, **options)
        if not results:
            # An optional expansion/fusion failure must not erase a previously
            # validated batch. These are hints; the checks below run again.
            results = list(
                {
                    r.doc_id: r for r in (*plan.document_fallback, *plan.graph_fallback)
                }.values()
            )
        results = [
            row for row in results if allowed_memory(row.metadata, (origin,), personas)
        ]
        results = self._filter_by_retrieval_policy(results)
        if execution_context is not None:
            outcome = await execution_context.run(
                "semantic_atom_policy",
                self._apply_atom_policy,
                results,
                discriminator={"document_ids": [row.doc_id for row in results]},
            )
            return (
                (outcome.get("value") or [])[:k]
                if outcome.get("state") == "completed"
                else []
            )
        return (await self._apply_atom_policy(results))[:k]

    def schedule_recall_access_touch(self, memory_ids, atom_ids=()) -> str:
        """Bounded owner-only access touch after candidates survive final filtering."""
        docs = tuple(
            sorted(
                {
                    int(value)
                    for value in memory_ids
                    if isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                }
            )
        )
        atoms = tuple(
            sorted(
                {
                    int(value)
                    for value in atom_ids
                    if isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                }
            )
        )
        if not docs and not atoms:
            return "empty"
        task_key = (docs, atoms)
        active = self._recall_access_tasks.get(task_key)
        if active is not None and not active.done():
            return "coalesced"
        if active is not None:
            self._recall_access_tasks.pop(task_key, None)
        for stale_key, stale_task in tuple(self._recall_access_tasks.items()):
            if stale_task.done():
                self._recall_access_tasks.pop(stale_key, None)
        if len(self._recall_access_tasks) >= self._recall_access_task_limit:
            return "busy"
        task = self._create_tracked_task(
            self._update_access_times_internal(list(docs), list(atoms)),
            clean_context=True,
        )
        self._recall_access_tasks[task_key] = task

        def release(completed, key=task_key):
            if self._recall_access_tasks.get(key) is completed:
                self._recall_access_tasks.pop(key, None)

        import contextvars

        task.add_done_callback(release, context=contextvars.Context())
        return "scheduled"

    async def _update_memory_status_guarded(
        self,
        memory_id: int,
        metadata_updates: dict[str, Any],
    ) -> bool:
        async with self.source_state_guard.hold():
            return await self._update_memory_status_with_guard_locked(
                memory_id,
                metadata_updates,
            )

    async def _update_memory_status_with_guard_locked(
        self,
        memory_id: int,
        metadata_updates: dict[str, Any],
    ) -> bool:
        """Serialize status changes with graph reads/pruning and reactivation.

        Archived-to-active status is committed only after graph indexing has
        completed and its FAISS references still exist. The shared write guard
        covers preparation and commit; graph queries do not acquire it and
        continue to filter against committed source status.
        """
        status = str(metadata_updates.get("status") or "").strip()
        if status not in {"active", "archived", "deleted"}:
            return False

        memory = await self.get_memory(memory_id)
        if not memory:
            return False
        current_metadata = safe_json_dict(memory.get("metadata", {}))
        current_status = str(current_metadata.get("status") or "active")
        reactivating = status == "active" and current_status != "active"
        target_metadata = current_metadata.copy()
        target_metadata.update(metadata_updates)
        target_metadata["updated_at"] = time.time()

        if reactivating:
            # Provider work happens while the source is still archived. The
            # shared guard excludes concurrent prune and status commits; graph
            # queries do not acquire it and continue to use committed status.
            try:
                if self.graph_memory_manager is not None:
                    indexed = await self.graph_memory_manager.index_memory(
                        memory_id,
                        str(memory.get("text") or ""),
                        target_metadata,
                    )
                    if indexed is False:
                        raise RuntimeError("graph indexing deferred during rebuild")
            except asyncio.CancelledError:
                async with self.source_state_guard.hold():
                    await self._persist_graph_repair_marker_cancel_safe_locked(
                        memory_id, "reactivation_prepare_cancelled"
                    )
                raise
            except Exception:
                async with self.source_state_guard.hold():
                    await self._persist_graph_repair_marker_cancel_safe_locked(
                        memory_id, "reactivation_prepare_failed"
                    )
                logger.error(
                    "[更新] 来源仍保持归档状态；图索引恢复失败并已登记待修复",
                    exc_info=True,
                )
                return False

        graph_affecting = metadata_updates.keys() & {
            "participants",
            "participant_identities",
            "topics",
            "key_facts",
            "canonical_summary",
            "session_id",
            "persona_id",
            "importance",
            "create_time",
            "summary_schema_version",
            "source_window",
        }
        expected_revision = (
            self._graph_source_revision(memory) if reactivating else None
        )

        async with self.source_state_guard.hold():
            latest = await self.get_memory(memory_id)
            if not latest:
                if reactivating:
                    await self._persist_graph_repair_marker_cancel_safe_locked(
                        memory_id, "source_missing_at_reactivation_commit"
                    )
                return False
            latest_metadata = safe_json_dict(latest.get("metadata", {}))
            latest_status = str(latest_metadata.get("status") or "active")
            if reactivating and (
                latest_status != current_status
                or self._graph_source_revision(latest) != expected_revision
            ):
                await self._persist_graph_repair_marker_cancel_safe_locked(
                    memory_id, "source_changed_during_reactivation"
                )
                return False

            if reactivating and self.graph_memory_manager is not None:
                complete = await self.graph_memory_manager.source_graph_index_complete(
                    memory_id
                )
                if not complete:
                    await self._persist_graph_repair_marker_cancel_safe_locked(
                        memory_id, "graph_vector_missing_at_reactivation_commit"
                    )
                    return False
            if reactivating and not self._is_memory_vector_indexed(memory_id):
                await self._persist_graph_repair_marker_cancel_safe_locked(
                    memory_id, "main_vector_missing_at_reactivation_commit"
                )
                return False

            if self.hybrid_retriever is None:
                if reactivating:
                    await self._persist_graph_repair_marker_cancel_safe_locked(
                        memory_id, "metadata_writer_unavailable"
                    )
                return False

            commit_updates = dict(metadata_updates)
            if reactivating:
                commit_updates.update(
                    needs_graph_repair=False,
                    graph_repair_reason=None,
                )
            success = await self.hybrid_retriever.update_metadata(
                memory_id,
                commit_updates,
            )
            if not success:
                if reactivating:
                    await self._persist_graph_repair_marker_cancel_safe_locked(
                        memory_id, "metadata_activation_failed"
                    )
                return False

            self._invalidate_search_cache()

        if (
            graph_affecting
            and not reactivating
            and status == "active"
            and self.graph_memory_manager is not None
        ):
            try:
                await self.graph_memory_manager.index_memory(
                    memory_id,
                    str(latest.get("text") or ""),
                    {**latest_metadata, **metadata_updates},
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                async with self.source_state_guard.hold():
                    await self._persist_graph_repair_marker_cancel_safe_locked(
                        memory_id, "active_graph_metadata_reindex_failed"
                    )
                raise
        return True

    async def _mark_graph_repair_needed_locked(
        self,
        memory_id: int,
        reason: str,
    ) -> None:
        """Persist a non-sensitive repair marker while the shared guard is held."""
        async with self._write_db() as db:
            cursor = await db.execute(
                "SELECT metadata FROM documents WHERE id = ?",
                (int(memory_id),),
            )
            row = await cursor.fetchone()
            if row is None:
                return
            metadata = safe_json_dict(row["metadata"])
            metadata["needs_graph_repair"] = True
            metadata["graph_repair_reason"] = str(reason)
            metadata["graph_repair_updated_at"] = time.time()
            await db.execute(
                "UPDATE documents SET metadata = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), int(memory_id)),
            )

    async def _persist_graph_repair_marker_cancel_safe_locked(
        self,
        memory_id: int,
        reason: str,
    ) -> None:
        """Finish the SQL marker write before a repeated cancellation can release the guard."""
        from .memory_engine_batch import _await_task_to_completion

        marker_task = asyncio.create_task(
            self._mark_graph_repair_needed_locked(memory_id, reason)
        )
        cancelled, error = await _await_task_to_completion(marker_task)
        if error is not None:
            raise RuntimeError("graph repair marker persistence failed") from error
        if cancelled:
            raise asyncio.CancelledError

    @staticmethod
    def _graph_source_revision(memory: dict[str, Any]) -> str:
        metadata = safe_json_dict(memory.get("metadata", {}))
        graph_fields = (
            "participants",
            "participant_identities",
            "topics",
            "key_facts",
            "canonical_summary",
            "session_id",
            "persona_id",
            "importance",
            "create_time",
            "summary_schema_version",
            "source_window",
        )
        payload = {
            "text": str(memory.get("text") or ""),
            "metadata": {key: metadata.get(key) for key in graph_fields},
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def update_importance(self, memory_id: int, new_importance: float) -> bool:
        """
        更新记忆重要性

        Args:
            memory_id: 记忆ID
            new_importance: 新重要性值(0-1)

        Returns:
            bool: 是否更新成功
        """
        return await self.update_memory(memory_id, {"importance": new_importance})

    _DECAY_BATCH_SIZE = 1000

    def _decay_updates_for_rows(
        self,
        rows,
        *,
        decay_rate: float,
        days: int,
        access_window_start: float,
        max_access_count: float,
        access_decay_multiplier: float,
        protected_threshold: float,
    ) -> "tuple[list[tuple[str, int]], int]":
        """计算一批行的衰减结果（纯计算，供线程池与等价性对照复用）。

        逐行数学与 B10 之前的实现**完全一致**，只额外做一次"新序列化文本是否
        与库内文本逐字节相同"的判断：相同就不必写库（写不写最终状态一样）。

        Returns:
            (需要写入的 ``(metadata_json, id)`` 列表, 计算结果无变化而被跳过的行数)
        """
        updates: list[tuple[str, int]] = []
        skipped_noop = 0
        for row in rows:
            metadata = safe_json_dict(row["metadata"])
            importance = clamp_float(metadata.get("importance"), default=0.5)
            if importance >= protected_threshold:
                continue
            access_count = safe_float(metadata.get("access_count"), 0.0)
            last_access_time = safe_float(metadata.get("last_access_time"), 0.0)

            recent_access_factor = (
                1.0 if last_access_time >= access_window_start else 0.5
            )
            access_factor = min(1.0, access_count / max(1.0, max_access_count))
            effective_decay_rate = decay_rate * (
                1 - 0.5 * access_factor * recent_access_factor
            )
            decay_factor = (1 - effective_decay_rate) ** days
            metadata["importance"] = max(
                0.01,
                round(importance * decay_factor, 4),
            )
            metadata["access_count"] = int(access_count * access_decay_multiplier)
            serialized = json.dumps(metadata, ensure_ascii=False)
            if serialized == row["metadata"]:
                # 写到库里也是同一个字节串：跳过 UPDATE，最终状态不变。
                skipped_noop += 1
                continue
            updates.append((serialized, int(row["id"])))
        return updates, skipped_noop
