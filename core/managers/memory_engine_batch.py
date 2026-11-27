"""批量维护、衰减与访问统计。"""

import asyncio
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import aiosqlite
from astrbot.api import logger

from ...storage.atom_store import AtomStore
from ...storage.sqlite_connection import open_connection
from ..memory_scope import (
    adapter_sql,
    current_adapter_access,
)
from ..processors.atom_classifier import classify_atoms
from ..retrieval.hybrid_retriever import HybridResult
from ..utils.json_utils import safe_json_dict
from ..utils.number_utils import clamp_float, safe_float
from .agentic_recall_types import AgenticRecallOutcome as AgenticRecallOutcome


class MemoryEngineBatchMixin:
    """MemoryEngine 的 MemoryEngineBatchMixin 分组。

    宿主共享状态由 ``MemoryEngine`` 独占持有，本模块只通过 self 访问。
    """

    # 访问统计合并窗口：窗口内的多次召回合并为一次 UPDATE
    _ACCESS_UPDATE_DEBOUNCE_SECONDS = 0.05

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

    async def batch_delete_memories(self, memory_ids: list[int]) -> int:
        """Batch delete multiple memories using bulk SQL operations."""
        if not memory_ids:
            return 0

        if self.db_connection is None:
            logger.error("[批量删除] 数据库连接未初始化")
            return 0

        if current_adapter_access.get() is not None:
            ownership, ownership_params = adapter_sql("metadata")
            slots = ",".join("?" for _ in memory_ids)
            cursor = await self.db_connection.execute(
                f"SELECT id FROM documents WHERE id IN ({slots}) AND {ownership}",
                (*memory_ids, *ownership_params),
            )
            memory_ids = [int(row[0]) for row in await cursor.fetchall()]
        self._invalidate_search_cache()
        total_deleted = 0
        sql_batch_size = 200

        for i in range(0, len(memory_ids), sql_batch_size):
            batch = memory_ids[i : i + sql_batch_size]
            placeholders = ",".join("?" * len(batch))
            op_id = await self._start_write_op(
                "batch_delete",
                {
                    "memory_ids": batch,
                    "batch_offset": i,
                    "batch_size": len(batch),
                },
            )
            batch_deleted = 0

            try:
                # 1. Batch delete from BM25 FTS
                async with self._write_db() as db:
                    await db.execute(
                        f"DELETE FROM livingmemory_memories_fts WHERE doc_id IN ({placeholders})",
                        batch,
                    )
                await self._advance_write_op(
                    op_id,
                    "bm25_deleted",
                    payload_patch={"memory_ids": batch},
                )

                # 2. Look up UUIDs and delete from FAISS vector DB
                cursor = await self.db_connection.execute(
                    f"SELECT id, doc_id FROM documents WHERE id IN ({placeholders})",
                    batch,
                )
                uuid_rows = await cursor.fetchall()
                found_ids = [int(row["id"]) for row in uuid_rows]
                if found_ids:
                    deleted_vector_ids = await self.vector_retriever.delete_documents(
                        found_ids
                    )
                    if set(deleted_vector_ids) != set(found_ids):
                        raise RuntimeError(
                            "批量向量删除不完整: "
                            f"expected={found_ids}, deleted={deleted_vector_ids}"
                        )
                await self._advance_write_op(
                    op_id,
                    "faiss_deleted",
                    payload_patch={"memory_ids": batch, "found_ids": found_ids},
                )

                # 3. Batch delete from documents table
                async with self._write_db() as db:
                    cursor = await db.execute(
                        f"DELETE FROM documents WHERE id IN ({placeholders})",
                        batch,
                    )
                    await db.execute(
                        f"DELETE FROM memory_sources WHERE memory_id IN ({placeholders})",
                        batch,
                    )
                # FaissVecDB 与本引擎共享 documents 表；向量删除可能已移除
                # 对应行，因此以删除前查到的记录数作为准确结果。
                batch_deleted = len(found_ids)
                await self._advance_write_op(
                    op_id,
                    "documents_deleted",
                    payload_patch={
                        "memory_ids": batch,
                        "found_ids": found_ids,
                        "deleted_count": batch_deleted,
                    },
                )

                # 4. Batch delete graph artifacts and atoms
                await self._delete_graph_and_atoms_for_batch(batch)
                await self._advance_write_op(
                    op_id,
                    "graph_atoms_deleted",
                    payload_patch={"memory_ids": batch, "deleted_count": batch_deleted},
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._advance_write_op(
                    op_id,
                    "batch_delete_failed",
                    status="needs_repair",
                    error=str(e),
                    payload_patch={
                        "memory_ids": batch,
                        "deleted_count": batch_deleted,
                    },
                )
                logger.error(
                    f"[批量删除] 批次删除失败 (offset={i}, size={len(batch)})",
                    exc_info=True,
                )
                raise

            await self._advance_write_op(
                op_id,
                "completed",
                status="completed",
                payload_patch={"memory_ids": batch, "deleted_count": batch_deleted},
            )
            total_deleted += batch_deleted

        if total_deleted:
            logger.info(f"[批量删除] 共删除 {total_deleted} 条记忆")
        return total_deleted

    async def cleanup_old_memories(
        self,
        days_threshold: int | None = None,
        importance_threshold: float | None = None,
    ) -> int:
        """
        清理旧记忆（使用分批处理避免内存问题）

        删除超过阈值且重要性低的记忆

        Args:
            days_threshold: 天数阈值,默认从配置读取
            importance_threshold: 重要性阈值,默认从配置读取

        Returns:
            int: 删除的记忆数量
        """
        # 使用配置或参数值
        days = (
            self.config.get("cleanup_days_threshold", 30)
            if days_threshold is None
            else days_threshold
        )
        importance = (
            self.config.get("cleanup_importance_threshold", 0.3)
            if importance_threshold is None
            else importance_threshold
        )
        ownership, ownership_params = adapter_sql("metadata")
        try:
            days = int(days)
            importance = float(importance)
        except (TypeError, ValueError):
            logger.error(
                f"清理参数格式错误: days_threshold={days}, importance_threshold={importance}"
            )
            return 0

        if days < 0:
            logger.error(f"清理参数无效: days_threshold={days}（必须 >= 0）")
            return 0

        cutoff_time = time.time() - (days * 86400)

        try:
            if self.db_connection is None:
                return 0

            batch_size = 500
            last_id = 0
            candidates: list[int] = []

            # 使用主键 keyset 分页流式读取，避免 OFFSET 分页的 O(N²) 开销。
            while True:
                cursor = await self.db_connection.execute(
                    f"SELECT id, metadata FROM documents WHERE id > ? AND {ownership} ORDER BY id LIMIT ?",
                    (last_id, *ownership_params, batch_size),
                )
                rows = await cursor.fetchall()
                if not rows:
                    break
                last_id = int(rows[-1]["id"])

                raw_metadata = [r["metadata"] for r in rows]
                parsed = await asyncio.to_thread(
                    lambda: [safe_json_dict(m) for m in raw_metadata]
                )

                for row, metadata in zip(rows, parsed):
                    if str(metadata.get("status") or "active") != "active":
                        continue
                    create_time = safe_float(metadata.get("create_time"), time.time())
                    doc_importance = clamp_float(
                        metadata.get("importance"), default=0.5
                    )
                    if create_time < cutoff_time and doc_importance < importance:
                        candidates.append(int(row["id"]))

            if not candidates:
                return 0
            if self.config.get("auto_archived_enabled", False):
                logger.info(f"[清理] 发现 {len(candidates)} 条候选记忆，开始归档")
                return await self.archive_memories(candidates)

            logger.info(f"[清理] 发现 {len(candidates)} 条候选记忆，开始批量删除")
            deleted_count = await self.batch_delete_memories(candidates)
            logger.info(f"[清理] 完成，已删除 {deleted_count} 条旧记忆")
            return deleted_count
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("[清理] 清理旧记忆失败", exc_info=True)
            return 0

    async def purge_expired_archived_memories(
        self, retention_days: int | None = None
    ) -> int:
        """Permanently purge ``status='archived'`` documents (and graph/atom rows) past retention TTL."""
        days_raw = (
            self.config.get("archived_retention_days", 90)
            if retention_days is None
            else retention_days
        )
        try:
            days = int(days_raw)
        except (TypeError, ValueError):
            days = 90
        if days <= 0 or self.db_connection is None:
            return 0

        cutoff_time = time.time() - (days * 86400)
        ownership, ownership_params = adapter_sql("metadata")
        try:
            batch_size = 500
            last_id = 0
            expired_ids: list[int] = []
            while True:
                cursor = await self.db_connection.execute(
                    f"""
                    SELECT id, metadata FROM documents
                    WHERE id > ? AND {ownership}
                      AND COALESCE(json_extract(metadata, '$.status'), 'active') = 'archived'
                    ORDER BY id LIMIT ?
                    """,
                    (last_id, *ownership_params, batch_size),
                )
                rows = await cursor.fetchall()
                if not rows:
                    break
                last_id = int(rows[-1]["id"])
                raw_metadata = [r["metadata"] for r in rows]
                parsed = await asyncio.to_thread(
                    lambda: [safe_json_dict(m) for m in raw_metadata]
                )
                for row, metadata in zip(rows, parsed):
                    archived_ts = safe_float(
                        metadata.get("archived_at")
                        or metadata.get("last_access_time")
                        or metadata.get("create_time"),
                        0.0,
                    )
                    if archived_ts > 0 and archived_ts < cutoff_time:
                        expired_ids.append(int(row["id"]))

            if not expired_ids:
                return 0
            logger.info(
                f"[归档清理] 发现 {len(expired_ids)} 条超期归档记忆（TTL={days}天），开始彻底清除"
            )
            return await self.batch_delete_memories(expired_ids)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("[归档清理] 清除过期归档记忆失败", exc_info=True)
            return 0

    async def archive_memories(
        self,
        memory_ids: list[int],
        *,
        audit=None,
        expected_revision: str | None = None,
    ) -> int:
        """Fence archive writes against graph-source validation and pruning."""
        async with self.source_state_guard.hold():
            if expected_revision is not None:
                if len(memory_ids) != 1:
                    raise ValueError("expected_revision requires an individual memory")
                await self._check_memory_revision(memory_ids[0], expected_revision)
            return await self._archive_memories_locked(memory_ids, audit=audit)

    async def restore_memory(
        self, memory_id: int, *, expected_revision: str | None = None
    ) -> bool:
        """Fence the complete restore transaction against stale-vector pruning."""
        async with self.source_state_guard.hold():
            if expected_revision is not None:
                await self._check_memory_revision(memory_id, expected_revision)
            return await self._restore_memory_locked(memory_id)

    # 运行时旧会话迁移的全局完成标记（migration_status.key）
    _RUNTIME_MIGRATION_FLAG_KEY = "runtime_session_migration_v1"
    _RUNTIME_MIGRATION_MARKER_PREFIX = "migrated_umo_"

    async def _runtime_session_migration_needed(self, unified_msg_origin: str) -> bool:
        """判断当前会话是否还需要跑一次旧格式 session_id 迁移。

        热路径（每次召回）调用：已完成的迁移由进程内状态短路，只有进程内首次
        调用或未完成迁移的会话才会查 ``migration_status``。库里没有旧格式
        session_id 数据时写一次全局完成标记，之后本进程不再触发任何迁移任务。

        Args:
            unified_msg_origin: 当前会话的 unified_msg_origin。

        Returns:
            True 表示需要调度后台迁移任务。
        """
        if self._runtime_migration_finished:
            return False
        if unified_msg_origin in self._runtime_migration_checked_sessions:
            return False
        if self.db_connection is None:
            return False

        if not self._runtime_migration_flag_loaded:
            self._runtime_migration_flag_loaded = True
            flag = await self._read_migration_status_value(
                self._RUNTIME_MIGRATION_FLAG_KEY
            )
            if str(flag or "").casefold() == "true":
                self._runtime_migration_finished = True
                return False
            if not await self._has_legacy_session_rows():
                await self._write_runtime_migration_flag()
                self._runtime_migration_finished = True
                return False

        marker = await self._read_migration_status_value(
            f"{self._RUNTIME_MIGRATION_MARKER_PREFIX}{unified_msg_origin}"
        )
        if str(marker or "").casefold() == "true":
            self._runtime_migration_checked_sessions.add(unified_msg_origin)
            return False
        return True

    async def _read_migration_status_value(self, key: str) -> str | None:
        """读取 migration_status 单键值；读取失败按缺失处理。"""
        if self.db_connection is None:
            return None
        try:
            cursor = await self.db_connection.execute(
                "SELECT value FROM migration_status WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[自动迁移] 读取 migration_status 失败", exc_info=True)
            return None
        return str(row[0]) if row and row[0] is not None else None

    async def _has_legacy_session_rows(self) -> bool:
        """判断 documents 中是否还有未 unified_msg_origin 化的旧会话数据。"""
        if self.db_connection is None:
            return False
        try:
            cursor = await self.db_connection.execute(
                "SELECT 1 FROM documents WHERE json_valid(metadata) "
                "AND json_extract(metadata, '$.session_id') IS NOT NULL "
                "AND json_extract(metadata, '$.session_id') NOT LIKE '%:%' "
                "LIMIT 1"
            )
            return await cursor.fetchone() is not None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[自动迁移] 检查旧会话数据失败", exc_info=True)
            return False

    async def _write_runtime_migration_flag(self) -> None:
        """写入“旧会话数据已清理完毕”的全局标记（失败只记日志）。"""
        if self.db_connection is None:
            return
        try:
            async with self._write_transaction() as connection:
                await connection.execute(
                    "INSERT OR REPLACE INTO migration_status "
                    "(key, value, updated_at) VALUES (?, 'true', datetime('now'))",
                    (self._RUNTIME_MIGRATION_FLAG_KEY,),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[自动迁移] 写入运行期迁移标记失败", exc_info=True)

    async def _finish_runtime_session_migration(self, unified_msg_origin: str) -> None:
        """记录单会话迁移完成；若库里已无旧数据则写全局完成标记。"""
        self._runtime_migration_checked_sessions.add(unified_msg_origin)
        if self._runtime_migration_finished:
            return
        if await self._has_legacy_session_rows():
            return
        await self._write_runtime_migration_flag()
        self._runtime_migration_finished = True

    def _schedule_runtime_session_migration(self, unified_msg_origin: str) -> None:
        """调度单会话后台迁移任务（同一会话 in-flight 去重）。"""
        if unified_msg_origin in self._runtime_migration_inflight:
            return
        self._runtime_migration_inflight.add(unified_msg_origin)
        task = self._create_tracked_task(
            self._migrate_session_data_if_needed(unified_msg_origin)
        )
        task.add_done_callback(
            lambda _task, umo=unified_msg_origin: (
                self._runtime_migration_inflight.discard(umo)
            )
        )

    async def _migrate_session_data_if_needed(self, unified_msg_origin: str) -> None:
        """
        运行时自动迁移：将旧格式的session_id更新为unified_msg_origin格式

        支持各种平台的旧格式（通用匹配策略）：
        - WebChat UUID: "ac8c2cef-959e-4146-ad22-c82d0230ad06"
        - WebChat带前缀: "webchat!astrbot!ac8c2cef-959e-4146-ad22-c82d0230ad06"
        - QQ号: "123456789"
        - 其他平台: 任意字符串

        目标格式: "platform:message_type:session_id"

        策略：
        1. 从unified_msg_origin解析出：platform、message_type、session_id
        2. 生成所有可能的旧格式匹配候选（递归拆分）
        3. 查找匹配任一候选且不含冒号的旧记录
        4. 批量更新为unified_msg_origin
        5. 使用unified_msg_origin本身作为迁移标记（避免重复）

        Args:
            unified_msg_origin: 完整的统一消息来源（格式：platform:type:session_id）
        """

        try:
            # 1. 解析 unified_msg_origin
            parts = unified_msg_origin.split(":", 2)
            if len(parts) != 3:
                logger.warning(
                    f"[自动迁移] unified_msg_origin 格式不正确: {unified_msg_origin}"
                )
                return

            platform_id, message_type, full_session_id = parts

            # 2. 生成所有可能的旧格式匹配候选
            # 对于 "webchat!astrbot!ac8c2cef-..." 会生成:
            #   ["webchat!astrbot!ac8c2cef-...", "astrbot!ac8c2cef-...", "ac8c2cef-..."]
            # 对于 "123456789" 会生成: ["123456789"]
            candidates = [full_session_id]

            # 按感叹号递归拆分
            if "!" in full_session_id:
                parts_by_bang = full_session_id.split("!")
                for i in range(1, len(parts_by_bang)):
                    candidates.append("!".join(parts_by_bang[i:]))

            logger.info(f"[自动迁移] 开始检查会话，候选匹配: {candidates}")

            # 3. 检查是否已迁移（使用unified_msg_origin本身作为标记）
            migration_key = f"migrated_umo_{unified_msg_origin}"
            if self.db_connection is None:
                return
            cursor = await self.db_connection.execute(
                "SELECT value FROM migration_status WHERE key = ?", (migration_key,)
            )
            row = await cursor.fetchone()
            if row and row[0] == "true":
                # 已迁移过，跳过；同时写入进程内缓存，后续召回不再触发本任务
                self._runtime_migration_checked_sessions.add(unified_msg_origin)
                return

            # 4. 查找所有需要迁移的记录
            # 条件：session_id 匹配任一候选 且 不包含冒号（旧格式标识）
            placeholders = " OR ".join(
                ["json_extract(metadata, '$.session_id') = ?" for _ in candidates]
            )
            query = f"""
                SELECT id, metadata FROM documents
                WHERE ({placeholders})
                AND json_extract(metadata, '$.session_id') NOT LIKE '%:%'
            """

            cursor = await self.db_connection.execute(query, tuple(candidates))
            rows = list(await cursor.fetchall())

            if not rows:
                logger.info("[自动迁移] 未找到需要迁移的旧数据")
                # 即使没有旧数据也标记为已检查，避免重复查询
                async with self._write_transaction() as connection:
                    await connection.execute(
                        "INSERT OR REPLACE INTO migration_status (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                        (migration_key, "true"),
                    )
                await self._finish_runtime_session_migration(unified_msg_origin)
                return

            logger.info(f"[自动迁移] 找到 {len(rows)} 条旧数据需要迁移")

            # 5. 批量更新
            updates: list[tuple[str, int]] = []
            for row in rows:
                doc_id = row[0]
                metadata_str = row[1]

                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                old_session_id = metadata.get("session_id", "unknown")

                # 更新为unified_msg_origin格式
                metadata["session_id"] = unified_msg_origin
                metadata["migrated_at"] = time.time()
                metadata["old_session_id"] = old_session_id  # 保留旧值便于追溯

                updates.append((json.dumps(metadata, ensure_ascii=False), doc_id))

            updated_count = len(updates)
            async with self._write_transaction() as connection:
                if updates:
                    await connection.executemany(
                        "UPDATE documents SET metadata = ? WHERE id = ?",
                        updates,
                    )
                await connection.execute(
                    "INSERT OR REPLACE INTO migration_status (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    (migration_key, "true"),
                )

            logger.info(
                f"[自动迁移] 完成！已更新 {updated_count} 条记录 -> {unified_msg_origin}"
            )
            await self._finish_runtime_session_migration(unified_msg_origin)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[自动迁移] 迁移失败: {e}", exc_info=True)

    async def rebuild_graph_index(self) -> dict[str, int]:
        """Stream active documents into a safe graph-memory rebuild."""
        if self.graph_memory_manager is None:
            return {"rebuilt": 0, "skipped": 0}

        if self.db_connection is None:
            raise RuntimeError("memory database is not initialized")

        async def active_memory_batches():
            last_id = 0
            batch_size = 200
            while True:
                cursor = await self.db_connection.execute(
                    """
                    SELECT id, text, metadata
                    FROM documents
                    WHERE id > ?
                      AND COALESCE(
                          json_extract(metadata, '$.status'), 'active'
                      ) = 'active'
                    ORDER BY id
                    LIMIT ?
                    """,
                    (last_id, batch_size),
                )
                rows = await cursor.fetchall()
                if not rows:
                    break

                batch: list[tuple[int, str, dict[str, Any]]] = []
                for row in rows:
                    metadata = row["metadata"] or {}
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    elif not isinstance(metadata, dict):
                        metadata = {}
                    batch.append((int(row["id"]), str(row["text"] or ""), metadata))

                last_id = int(rows[-1]["id"])
                yield batch

        self._invalidate_search_cache()
        return await self.graph_memory_manager.rebuild_memory_batches(
            active_memory_batches()
        )

    async def apply_daily_decay(self, decay_rate: float, days: int = 1) -> int:
        """
        批量应用重要性衰减

        B10：改为按 id 分页流式处理（内存有界）并按批提交；**逐行数学与返回的
        "影响记录数"语义与改动前逐字一致**（见 notes/b10.md 第 3 节的等价性论证）。

        原子性与旧实现的唯一差异：旧实现"全部算完再一次性 executemany"，中间失败
        时零写入；本实现每批一个短事务，极端情况下（DB 长时间被锁 / 进程被杀 /
        metadata 无法解析）会留下"前若干批已落库"。失败时会把已处理进度写进日志，
        便于事后核对。

        Args:
            decay_rate: 每日衰减率 (0-1)
            days: 衰减天数（用于补偿错过的天数）

        Returns:
            int: 受影响的记忆数量（= 通过保护的阈值过滤、参与计算的行数）
        """
        if decay_rate <= 0 or days <= 0:
            return 0

        if self.db_connection is None:
            logger.error("[衰减] 数据库连接未初始化")
            return 0

        # 进度计数放在 try 之外，异常时才能报告"已经写进去多少"。
        last_id = 0
        affected = 0
        written = 0
        skipped_noop = 0
        batches = 0

        try:
            access_window_days = float(
                self.config.get("access_decay_window_days", 30.0)
            )
            max_access_count = float(self.config.get("access_decay_max_count", 10.0))
            access_decay_multiplier = float(
                self.config.get("access_count_decay_multiplier", 0.5)
            )
            protected_threshold = clamp_float(
                self.config.get("protected_importance_threshold"), default=0.6
            )
            access_window_start = time.time() - max(1.0, access_window_days) * 86400.0
            access_decay_multiplier = max(0.0, min(1.0, access_decay_multiplier))

            # 与旧实现相同的候选行谓词，额外加 json_valid(metadata) 隔离损坏行，
            # 避免单条坏 JSON 触发 SQLite OperationalError 中止整趟衰减。
            candidate_predicate = (
                "json_valid(metadata) AND ("
                "json_extract(metadata, '$.importance') IS NOT NULL "
                "OR metadata LIKE '%\"importance\"%'"
                ")"
            )
            # 快照上界：只处理本次调用开始时已存在的行，等价于旧实现"一次性 SELECT"
            # 的候选集（之后插入的行 id 更大，本趟不处理）。
            cursor = await self.db_connection.execute("SELECT MAX(id) FROM documents")
            max_row = await cursor.fetchone()
            max_id = int(max_row[0]) if max_row and max_row[0] is not None else 0
            if max_id <= 0:
                return 0

            while last_id < max_id:
                cursor = await self.db_connection.execute(
                    "SELECT id, metadata FROM documents "
                    f"WHERE ({candidate_predicate}) AND id > ? AND id <= ? "
                    "ORDER BY id LIMIT ?",
                    (last_id, max_id, self._DECAY_BATCH_SIZE),
                )
                rows = await cursor.fetchall()
                if not rows:
                    break
                last_id = int(rows[-1]["id"])
                batches += 1

                # 卸载逐行 JSON 解析与数值计算，避免阻塞事件循环（每批一次）。
                def _compute_batch(batch_rows):
                    batch_updates, batch_skipped = self._decay_updates_for_rows(
                        batch_rows,
                        decay_rate=decay_rate,
                        days=days,
                        access_window_start=access_window_start,
                        max_access_count=max_access_count,
                        access_decay_multiplier=access_decay_multiplier,
                        protected_threshold=protected_threshold,
                    )
                    sql_rows = []
                    for raw_json, doc_id in batch_updates:
                        parsed = json.loads(raw_json)
                        sql_rows.append(
                            (
                                parsed["importance"],
                                access_decay_multiplier,
                                doc_id,
                            )
                        )
                    return sql_rows, batch_skipped

                sql_params, skipped = await asyncio.to_thread(_compute_batch, rows)
                skipped_noop += skipped
                affected += len(sql_params) + skipped
                if sql_params:
                    # 每批一个短事务：写锁分段释放，不再把整趟衰减压在一个长事务里。
                    # 使用 SQL 侧 json_extract(metadata, '$.access_count') * multiplier
                    # 保留快照之后并发召回写入的 access_count 增量。
                    async with self._write_db() as db:
                        await db.executemany(
                            "UPDATE documents "
                            "SET metadata = json_set("
                            "metadata, '$.importance', ?, '$.access_count', "
                            "CAST(COALESCE(json_extract(metadata, '$.access_count'), 0) * ? AS INTEGER)"
                            ") WHERE id = ? AND json_valid(metadata)",
                            sql_params,
                        )
                    written += len(sql_params)

            if affected == 0:
                return 0

            logger.info(
                f"[衰减] 批量衰减完成: 衰减率={decay_rate}, 天数={days}, "
                f"访问窗口={access_window_days:.1f}天, 影响记录={affected} "
                f"(写入={written}, 无变化跳过={skipped_noop}, 批次={batches})"
            )

            self._invalidate_search_cache()
            return affected

        except asyncio.CancelledError:
            if written:
                logger.warning(
                    f"[衰减] 衰减被取消，已提交 {written}/{affected} 条"
                    f"（最后处理 id={last_id}, 批次={batches}）；重试可能重复衰减该前缀"
                )
            raise
        except Exception as e:
            logger.error(f"[衰减] 批量衰减失败: {e}", exc_info=True)
            if written:
                # 分批提交后不再有全或无原子性：返回**已提交条数**而不是 0，
                # 否则调用方会把「已部分衰减」误读成「什么都没发生」。
                # 明细（最后处理的 id、批次数、重试语义）见上面两条日志。
                logger.error(
                    f"[衰减] 失败前已提交 {written}/{affected} 条"
                    f"（最后处理 id={last_id}, 批次={batches}）；重试可能重复衰减该前缀"
                )
                return written
            return 0

    async def _update_access_times_internal(
        self, doc_ids: list[int], atom_ids=None
    ) -> bool:
        """Coalesce access time and count increments through the debounce window."""
        unique_ids = [
            int(doc_id)
            for doc_id in doc_ids
            if isinstance(doc_id, int) and not isinstance(doc_id, bool) and doc_id > 0
        ]
        if not unique_ids or self.db_connection is None:
            return False

        self._access_update_pending.update(unique_ids)
        if atom_ids:
            self._access_atom_update_pending.update(
                int(atom_id)
                for atom_id in atom_ids
                if isinstance(atom_id, int)
                and not isinstance(atom_id, bool)
                and atom_id > 0
            )

        flush_task = self._access_flush_task
        if flush_task is not None and not flush_task.done():
            await asyncio.shield(flush_task)
            return True

        current_task = asyncio.current_task()
        self._access_flush_task = current_task
        try:
            await self._deferred_flush_access_updates()
            return True
        finally:
            if self._access_flush_task is current_task:
                self._access_flush_task = None

    async def _deferred_flush_access_updates(self) -> None:
        await asyncio.sleep(self._ACCESS_UPDATE_DEBOUNCE_SECONDS)
        while self._access_update_pending:
            if not await self._flush_access_updates():
                break

    async def _flush_access_updates(self) -> bool:
        """把挂起的访问统计写入数据库（失败时保留挂起项，下次重试）。"""
        doc_ids = sorted(self._access_update_pending)
        atom_ids = sorted(self._access_atom_update_pending)
        if not doc_ids:
            self._access_atom_update_pending.clear()
            return False
        written = await self._write_access_updates(doc_ids, atom_ids)
        if written:
            self._access_update_pending.difference_update(doc_ids)
            self._access_atom_update_pending.difference_update(atom_ids)
        return written

    async def _write_access_updates(
        self, doc_ids: list[int], atom_ids: list[int]
    ) -> bool:
        """单条原子 SQL 批量递增访问时间与次数。"""
        unique_ids = list(dict.fromkeys(int(doc_id) for doc_id in doc_ids))
        if not unique_ids or self.db_connection is None:
            return False

        current_time = time.time()
        try:
            # 单条原子 SQL，避免并发召回任务对同一记忆产生丢失更新
            placeholders = ",".join("?" * len(unique_ids))
            async with self._write_transaction() as connection:
                cursor = await connection.execute(
                    f"""
                    UPDATE documents
                    SET metadata = CASE
                        WHEN json_valid(metadata) THEN json_set(
                            json_set(metadata, '$.last_access_time', ?),
                            '$.access_count',
                            MIN(
                                COALESCE(
                                    CAST(json_extract(metadata, '$.access_count') AS INTEGER),
                                    0
                                ) + 1,
                                1000000
                            )
                        )
                        ELSE json_set('{{}}', '$.last_access_time', ?, '$.access_count', 1)
                    END
                    WHERE id IN ({placeholders})
                    """,
                    (current_time, current_time, *unique_ids),
                )
                if atom_ids and self.atom_store is not None:
                    unique_atoms = list(
                        dict.fromkeys(int(atom_id) for atom_id in atom_ids)
                    )
                    await connection.execute(
                        f"UPDATE memory_atoms SET last_accessed_at = ? "
                        f"WHERE id IN ({','.join('?' for _ in unique_atoms)}) "
                        "AND status = 'active' AND expires_at > ?",
                        (current_time, *unique_atoms, current_time),
                    )
            return cursor.rowcount > 0
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Access statistics are best effort; retain SQLite details for diagnosis.
            logger.warning(
                f"Failed to update memory access statistics "
                f"(db_path={self.db_path}, "
                f"sqlite_errorcode={getattr(e, 'sqlite_errorcode', None)}, "
                f"sqlite_errorname={getattr(e, 'sqlite_errorname', None)}, "
                f"doc_ids={unique_ids}): {e}",
                exc_info=True,
            )
            return False

    async def _archive_memories_locked(
        self, memory_ids: list[int], *, audit=None
    ) -> int:
        """Remove memories from retrieval indexes while retaining their documents."""
        if not memory_ids or self.db_connection is None:
            return 0

        unique_ids = list(dict.fromkeys(int(memory_id) for memory_id in memory_ids))
        documents = await self.faiss_db.document_storage.get_documents(
            metadata_filters={},
            ids=unique_ids,
            offset=0,
            limit=len(unique_ids),
        )
        active_documents = []
        metadata_updates: list[tuple[str, int]] = []
        archived_at = time.time()
        for document in documents:
            metadata = safe_json_dict(document.get("metadata"))
            if str(metadata.get("status") or "active") == "archived":
                continue
            metadata.update(
                {
                    k: v
                    for k, v in (audit or {}).items()
                    if k not in {"status", "archived_at"}
                }
            )
            metadata["status"] = "archived"
            metadata["archived_at"] = archived_at
            active_documents.append(document)
            metadata_updates.append(
                (json.dumps(metadata, ensure_ascii=False), int(document["id"]))
            )
        if not active_documents:
            return 0

        archived_ids = [int(document["id"]) for document in active_documents]
        placeholders = ",".join("?" * len(archived_ids))
        async with self._write_db() as db:
            await db.executemany(
                "UPDATE documents SET metadata = ? WHERE id = ?",
                metadata_updates,
            )
            await db.execute(
                f"DELETE FROM livingmemory_memories_fts WHERE doc_id IN ({placeholders})",
                archived_ids,
            )

        embedding_storage = getattr(self.faiss_db, "embedding_storage", None)
        embedding_delete = getattr(embedding_storage, "delete", None)
        if callable(embedding_delete):
            await embedding_delete(archived_ids)
        else:
            logger.warning("[归档] 当前 AstrBot 不支持独立删除向量，已保留软归档状态")
            async with self._write_transaction() as connection:
                await connection.execute(
                    "INSERT OR REPLACE INTO migration_status (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    ("needs_index_rebuild", "true"),
                )

        await self._delete_graph_and_atoms_for_batch(archived_ids)
        self._invalidate_search_cache()
        logger.info(f"[归档] 已归档 {len(archived_ids)} 条记忆")
        return len(archived_ids)

    async def _restore_memory_locked(self, memory_id: int) -> bool:
        """Restore one archived document and rebuild every retrieval index."""
        if self.db_connection is None:
            return False
        memory = await self.get_memory(memory_id)
        if not memory:
            return False
        metadata = safe_json_dict(memory.get("metadata"))
        source_status = str(metadata.get("status") or "active")
        if source_status != "archived":
            if source_status != "active":
                return False
            if not await self._restore_index_coverage_complete(memory_id):
                await self._persist_graph_repair_marker_cancel_safe_locked(
                    memory_id, "active_restore_coverage_missing"
                )
                return False
            if metadata.get("needs_graph_repair"):
                metadata["needs_graph_repair"] = False
                metadata["graph_repair_reason"] = None
                marker_task = asyncio.create_task(
                    self._commit_restore_metadata_sql(memory_id, metadata)
                )
                cancelled, error = await _await_task_to_completion(marker_task)
                if error is not None:
                    raise RuntimeError("active repair marker clear failed") from error
                if cancelled:
                    raise asyncio.CancelledError
            return True

        embedding_storage = getattr(self.faiss_db, "embedding_storage", None)
        embedding_insert = getattr(embedding_storage, "insert", None)
        embedding_delete = getattr(embedding_storage, "delete", None)
        provider = getattr(self.faiss_db, "embedding_provider", None)
        get_embedding = getattr(provider, "get_embedding_with_retry", None)
        if not callable(get_embedding):
            get_embedding = getattr(provider, "get_embedding", None)
        if not callable(embedding_insert) or not callable(get_embedding):
            raise RuntimeError("当前 AstrBot 不支持恢复已归档向量")

        import numpy as np

        content = str(memory.get("text") or "")
        vector_inserted = False
        fts_inserted = False
        activation_committed = False
        preserve_restore_indexes = False
        metadata["status"] = "active"
        metadata["restored_at"] = time.time()
        metadata.pop("archived_at", None)
        metadata["needs_graph_repair"] = False
        metadata["graph_repair_reason"] = None
        try:
            vector = np.asarray(await get_embedding(content), dtype=np.float32)
            await embedding_insert(vector, memory_id)
            vector_inserted = True
            if self.bm25_retriever is None:
                raise RuntimeError("BM25 检索器未初始化")
            await self.bm25_retriever.add_document(memory_id, content, metadata)
            fts_inserted = True

            atoms = classify_atoms(
                key_facts=list(metadata.get("key_facts") or []),
                topics=list(metadata.get("topics") or []),
                participants=list(metadata.get("participants") or []),
                parent_importance=clamp_float(metadata.get("importance"), default=0.5),
                session_id=metadata.get("session_id"),
                persona_id=metadata.get("persona_id"),
            )
            for atom in atoms:
                atom.parent_memory_id = memory_id
            if atoms and self.atom_store is not None:
                await self.atom_store.insert_many(atoms)
                if self.index_writer is not None and hasattr(
                    self.index_writer, "sync_atoms_added"
                ):
                    await self.index_writer.sync_atoms_added(atoms)
            if self.graph_memory_manager is not None:
                indexed = await self.graph_memory_manager.index_memory(
                    memory_id, content, metadata, atoms
                )
                if indexed is False:
                    raise RuntimeError("graph indexing deferred during rebuild")
            if not await self._restore_index_coverage_complete(memory_id):
                raise RuntimeError("restored retrieval indexes do not cover the source")

            # Until a separate committed-row read proves otherwise, any
            # failure after this point must retain all rebuilt indexes.
            preserve_restore_indexes = True
            commit_task = asyncio.create_task(
                self._commit_restore_metadata_sql(memory_id, metadata)
            )
            commit_cancelled, commit_error = await _await_task_to_completion(
                commit_task
            )
            if commit_error is not None:
                verify_task = asyncio.create_task(
                    self._verify_restore_commit(memory_id, metadata)
                )
                verify_cancelled, verify_error = await _await_task_to_completion(
                    verify_task
                )
                outcome = verify_task.result() if verify_error is None else "unknown"
                coverage_ok = False
                coverage_cancelled = False
                coverage_error = None
                if outcome == "committed":
                    coverage_task = asyncio.create_task(
                        self._restore_index_coverage_complete(memory_id)
                    )
                    (
                        coverage_cancelled,
                        coverage_error,
                    ) = await _await_task_to_completion(coverage_task)
                    if coverage_error is None:
                        coverage_ok = bool(coverage_task.result())

                caller_cancelled = (
                    commit_cancelled
                    or verify_cancelled
                    or coverage_cancelled
                    or isinstance(commit_error, asyncio.CancelledError)
                )
                if (
                    outcome == "not_committed"
                    and verify_error is None
                    and not caller_cancelled
                ):
                    preserve_restore_indexes = False
                    if isinstance(commit_error, asyncio.CancelledError):
                        raise asyncio.CancelledError
                    raise commit_error

                if outcome == "committed" and coverage_ok:
                    activation_committed = True
                    if caller_cancelled:
                        raise asyncio.CancelledError
                    raise commit_error

                await self._persist_graph_repair_marker_cancel_safe_locked(
                    memory_id, "restore_commit_outcome_uncertain"
                )
                if caller_cancelled:
                    raise asyncio.CancelledError
                raise RuntimeError(
                    "restore commit or coverage could not be confirmed; indexes were retained and repair was recorded"
                ) from commit_error

            activation_committed = True
            self._invalidate_search_cache()
            if commit_cancelled:
                raise asyncio.CancelledError
            return True
        except BaseException:
            if activation_committed or preserve_restore_indexes:
                raise

            async def rollback_restore() -> None:
                try:
                    if fts_inserted:
                        async with self._write_db() as db:
                            await db.execute(
                                "DELETE FROM livingmemory_memories_fts WHERE doc_id = ?",
                                (memory_id,),
                            )
                    if vector_inserted and callable(embedding_delete):
                        await embedding_delete([memory_id])
                    if self.graph_memory_manager is not None:
                        await self.graph_memory_manager.delete_memory(memory_id)
                    if self.atom_store is not None:
                        await self.atom_store.delete_by_parent(memory_id)
                finally:
                    await self._mark_graph_repair_needed_locked(
                        memory_id, "restore_reindex_failed"
                    )

            cleanup = asyncio.create_task(rollback_restore())
            cancelled_during_cleanup, cleanup_error = await _await_task_to_completion(
                cleanup
            )
            if cleanup_error is not None:
                logger.error(
                    "[恢复] 回滚未完全完成；源状态仍归档，repair标记已尽力持久化",
                    exc_info=(
                        type(cleanup_error),
                        cleanup_error,
                        cleanup_error.__traceback__,
                    ),
                )
            if cancelled_during_cleanup:
                raise asyncio.CancelledError
            raise

    async def _commit_restore_metadata_sql(
        self,
        memory_id: int,
        metadata: dict[str, Any],
    ) -> None:
        """SQL-only final commit task; caller retains SourceStateGuard."""
        serialized = json.dumps(metadata, ensure_ascii=False)
        async with self._write_db() as db:
            await db.execute(
                "UPDATE documents SET metadata = ? WHERE id = ?",
                (serialized, memory_id),
            )

    async def _verify_restore_commit(
        self,
        memory_id: int,
        expected_metadata: dict[str, Any],
    ) -> str:
        """Read the committed row independently after an ambiguous SQL task."""
        reader = None
        close_reader = False
        db_path = getattr(self, "db_path", None)
        try:
            if isinstance(db_path, (str, Path)) and str(db_path) not in (
                "",
                ":memory:",
            ):
                reader = await open_connection(str(db_path), isolation_level=None)
                reader.row_factory = aiosqlite.Row
                close_reader = True
            else:
                reader = self.db_connection
            cursor = await reader.execute(
                "SELECT metadata FROM documents WHERE id = ?",
                (int(memory_id),),
            )
            row = await cursor.fetchone()
            if row is None:
                return "unknown"
            actual = safe_json_dict(row["metadata"])
            if actual == expected_metadata and actual.get("status") == "active":
                return "committed"
            if actual.get("status") != "active":
                return "not_committed"
            return "unknown"
        except Exception:
            logger.error(
                "[恢复] 无法确认状态提交结果；保留索引并登记修复", exc_info=True
            )
            return "unknown"
        finally:
            if close_reader and reader is not None:
                await reader.close()

    async def _restore_index_coverage_complete(self, memory_id: int) -> bool:
        if not self._is_memory_vector_indexed(memory_id):
            return False
        if self.graph_memory_manager is not None:
            return await self.graph_memory_manager.source_graph_index_complete(
                memory_id
            )
        return True

    def _is_memory_vector_indexed(self, memory_id: int) -> bool:
        """Fail closed unless the main FAISS index confirms the restored ID."""
        import faiss

        storage = getattr(self.faiss_db, "embedding_storage", None)
        index = getattr(storage, "index", None)
        if index is None:
            return False
        try:
            mapped = faiss.downcast_index(index)
            id_map = getattr(mapped, "id_map", None)
            if id_map is None:
                return False
            return int(memory_id) in {
                int(value) for value in faiss.vector_to_array(id_map)
            }
        except Exception:
            return False

    def schedule_legacy_session_migration(self, unified_msg_origin: str) -> str:
        """Schedule only owner-authorized legacy migration with keyed bounded ownership."""
        if (
            not isinstance(unified_msg_origin, str)
            or not unified_msg_origin.strip()
            or self.db_connection is None
        ):
            return "unavailable"
        key = unified_msg_origin.strip()
        active = self._session_migration_tasks.get(key)
        if active is not None and not active.done():
            return "coalesced"
        if active is not None:
            self._session_migration_tasks.pop(key, None)
        for stale_key, stale_task in tuple(self._session_migration_tasks.items()):
            if stale_task.done():
                self._session_migration_tasks.pop(stale_key, None)
        if len(self._session_migration_tasks) >= self._session_migration_task_limit:
            return "busy"
        task = self._create_tracked_task(
            self._migrate_session_data_if_needed(key), clean_context=True
        )
        self._session_migration_tasks[key] = task

        def release(completed, session_key=key):
            if self._session_migration_tasks.get(session_key) is completed:
                self._session_migration_tasks.pop(session_key, None)

        import contextvars

        task.add_done_callback(release, context=contextvars.Context())
        return "scheduled"


async def _await_task_to_completion(
    task: asyncio.Task,
) -> tuple[bool, BaseException | None]:
    """Settle shielded cleanup despite repeated cancellation of the caller."""
    cancellation_seen = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation_seen = True
        except BaseException:
            break
    try:
        task.result()
    except BaseException as exc:
        return cancellation_seen, exc
    return cancellation_seen, None
