"""
统一记忆引擎 - MemoryEngine
提供统一的记忆管理接口,整合所有底层组件
"""

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

import aiosqlite

from astrbot.api import logger

from ...storage.atom_store import AtomStore
from ...storage.graph_store import GraphStore
from ...storage.schema import (
    DB_VERSION_SCHEMA_STATEMENTS,
    DOCUMENTS_DOC_ID_INDEX_STATEMENT,
    DOCUMENTS_INDEX_STATEMENTS,
    MIGRATION_STATUS_SCHEMA_STATEMENTS,
)
from ...storage.sqlite_connection import open_connection
from ...storage.sqlite_write import settle, write_transaction
from ..maintenance.graph_index_repair import RepairPlan
from ..maintenance.source_state_guard import SourceStateGuard
from ..managers.atom_lifecycle_manager import AtomLifecycleManager
from ..managers.graph_memory_manager import GraphMemoryManager
from ..processors.graph_extractor import GraphExtractor
from ..processors.text_processor import TextProcessor
from ..retrieval.atom_retriever import AtomRetriever
from ..retrieval.atom_vector_retriever import AtomVectorRetriever
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.dual_route_retriever import DualRouteRetriever
from ..retrieval.graph_keyword_retriever import GraphKeywordRetriever
from ..retrieval.graph_retriever import GraphRetriever
from ..retrieval.graph_vector_retriever import GraphVectorRetriever
from ..retrieval.hybrid_retriever import HybridResult, HybridRetriever
from ..retrieval.memory_index_writer import MemoryIndexWriter
from ..retrieval.rrf_fusion import RRFFusion
from ..retrieval.vector_retriever import VectorRetriever
from ..retrieval.vector_search import retrieve_documents
from .adapter_migrator import AdapterMigrator
from .memory_engine_agentic import MemoryEngineAgenticMixin
from .memory_engine_batch import MemoryEngineBatchMixin
from .memory_engine_crud import MemoryEngineCrudMixin
from .memory_engine_recall import MemoryEngineRecallMixin
from .memory_engine_stats import MemoryEngineStatsMixin
from .memory_engine_write_ops import MemoryEngineWriteOpsMixin


class MemoryEngine(
    MemoryEngineRecallMixin,
    MemoryEngineCrudMixin,
    MemoryEngineBatchMixin,
    MemoryEngineStatsMixin,
    MemoryEngineAgenticMixin,
    MemoryEngineWriteOpsMixin,
):
    """
    统一记忆引擎

    整合BM25检索、向量检索和混合检索,提供完整的记忆管理接口。

    主要功能:
    1. 记忆CRUD操作(添加、检索、更新、删除)
    2. 自动化记忆整理和清理
    3. 重要性评估和时间衰减
    4. 会话隔离和统计

    ID管理体系说明：
    ==================
    本系统使用三层存储架构，统一使用整数ID作为主键：

    1. **DocumentStorage (FAISS内部)**
       - 表: documents (SQLite，由SQLAlchemy管理)
       - 主键: id (INTEGER, AUTOINCREMENT) - 这是统一的整数标识符
       - UUID字段: doc_id (TEXT) - FAISS内部使用的UUID字符串
       - 关系: id ←→ doc_id (一对一映射)

    2. **BM25 FTS5索引**
       - 表: livingmemory_memories_fts (SQLite FTS5虚拟表)
       - 字段: doc_id (UNINDEXED) - 引用documents.id的整数
       - 注意: 只存储分词后的内容，metadata从documents表读取

    3. **FAISS向量索引**
       - 存储: EmbeddingStorage (FAISS索引文件)
       - 索引ID: 使用documents.id作为向量的整数索引

    插件对外接口：
    - add_memory() 返回: int (documents.id)
    - search_memories() 返回: HybridResult包含doc_id (int)
    - update_memory(memory_id: int) 参数: documents.id
    - delete_memory(memory_id: int) 参数: documents.id

    同步保证：
    - 添加: 先插入DocumentStorage获取id，再用此id插入BM25和FAISS
    - 更新: 通过vector_retriever更新DocumentStorage (自动同步)
    - 删除: 先删除BM25，再通过FaissVecDB.delete()删除DocumentStorage和向量
    """

    def __init__(
        self,
        db_path: str,
        faiss_db,
        graph_vector_db=None,
        llm_provider=None,
        config: dict[str, Any] | None = None,
        rerank_provider_resolver: Callable[[], Any] | None = None,
    ):
        """
        初始化记忆引擎

        Args:
            db_path: SQLite数据库路径
            faiss_db: FAISS向量数据库实例
            llm_provider: LLM提供者(可选,用于高级功能)
            config: 配置字典,支持以下参数:
                - rrf_k: RRF参数,默认60
                - decay_rate: 时间衰减率,默认0.01
                - importance_weight: 重要性权重,默认1.0
                - min_importance_for_retrieval: 召回最低重要性,默认0.0
                - fallback_enabled: 启用退化机制,默认True
                - cleanup_days_threshold: 清理天数阈值,默认30
                - cleanup_importance_threshold: 清理重要性阈值,默认0.3
                - stopwords_path: 停用词文件路径(可选)
            rerank_provider_resolver: 返回RerankProvider实例的可调用对象(可选)。
                动态解析以适配AstrBot的Provider实例重建,传None时跳过重排序。
        """
        self.db_path = db_path
        self.faiss_db = faiss_db
        self.graph_vector_db = graph_vector_db
        self.llm_provider = llm_provider
        self.config = config or {}
        self.rerank_provider_resolver = rerank_provider_resolver
        self.source_state_guard = SourceStateGuard()
        self.graph_enabled = bool(self.config.get("graph_memory_enabled", False))
        self.atom_enabled = bool(
            self.config.get(
                "atom_enabled",
                self.config.get("graph_memory_atom_enabled", True),
            )
        )

        # 确保数据库目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 后台任务跟踪
        self._pending_tasks: set[asyncio.Task] = set()
        self._session_migration_tasks: dict[str, asyncio.Task] = {}
        self._session_migration_task_limit = 16
        self._recall_access_tasks: dict[tuple, asyncio.Task] = {}
        self._recall_access_task_limit = 16

        # 旧会话运行时迁移（unified_msg_origin 化）：进程内缓存，避免已迁移用户
        # 每次召回都创建后台任务、查 migration_status 或打 INFO 日志。
        self._runtime_migration_finished = False
        self._runtime_migration_flag_loaded = False
        self._runtime_migration_checked_sessions: set[str] = set()
        self._runtime_migration_inflight: set[str] = set()

        # 初始化组件(在initialize中完成)
        self.text_processor = None
        self.bm25_retriever = None
        self.vector_retriever = None
        self.rrf_fusion = None
        self.hybrid_retriever = None
        self.graph_store = None
        self.graph_extractor = None
        self.graph_keyword_retriever = None
        self.graph_vector_retriever = None
        self.graph_retriever = None
        self.graph_memory_manager = None
        self.dual_route_retriever = None
        self.atom_store = None
        self.atom_lifecycle_manager = None
        self.atom_vector_retriever = None
        self.atom_retriever = None
        self.db_connection = None
        self.source_loader = None
        self._db_write_lock = asyncio.Lock()
        self._write_op_repair_enabled = bool(
            self.config.get("write_op_repair_enabled", True)
        )
        self._write_op_max_retries = int(self.config.get("write_op_max_retries", 3))
        self._search_cache_enabled = bool(self.config.get("search_cache_enabled", True))
        self._search_cache_ttl = float(
            self.config.get("search_cache_ttl_seconds", 45.0)
        )
        self._search_cache_max_size = int(self.config.get("search_cache_max_size", 256))
        self._search_cache_generation = 0
        self._search_cache: OrderedDict[
            tuple[Any, ...], tuple[float, list[HybridResult]]
        ] = OrderedDict()
        # 访问统计合并写：同一时间窗口内多次召回只落一次库
        self._access_update_pending: set[int] = set()
        self._access_atom_update_pending: set[int] = set()
        self._access_flush_task: asyncio.Task | None = None
        self.index_maintenance_status: dict[str, Any] = {
            "state": "idle",
            "current": 0,
            "total": 0,
            "message": "",
        }

    def refresh_config(self, new_config: dict[str, Any]) -> None:
        """Hot-refresh runtime config copy, retriever parameters, and search cache."""
        if not isinstance(new_config, dict):
            return
        self.config.update(new_config)
        self._search_cache_enabled = bool(
            self.config.get("search_cache_enabled", self._search_cache_enabled)
        )
        self._search_cache_ttl = float(
            self.config.get("search_cache_ttl_seconds", self._search_cache_ttl)
        )
        self._search_cache_max_size = int(
            self.config.get("search_cache_max_size", self._search_cache_max_size)
        )
        if self.rrf_fusion is not None and "rrf_k" in new_config:
            self.rrf_fusion.k = int(self.config.get("rrf_k", 60))
        if self.hybrid_retriever is not None:
            hr = self.hybrid_retriever
            hr.config = self.config
            hr.decay_rate = self.config.get("decay_rate", hr.decay_rate)
            hr.fallback_enabled = self.config.get(
                "fallback_enabled", hr.fallback_enabled
            )
            hr.rerank_enabled = bool(
                self.config.get("rerank_enabled", hr.rerank_enabled)
            )
            hr.rerank_candidates = max(
                2,
                min(
                    100,
                    int(
                        self.config.get("rerank_candidates", hr.rerank_candidates)
                        or hr.rerank_candidates
                    ),
                ),
            )
            if "rerank_excerpt_max_chars" in new_config:
                hr.rerank_excerpt_max_chars = max(
                    120,
                    min(
                        2400,
                        int(
                            self.config.get(
                                "rerank_excerpt_max_chars",
                                hr.rerank_excerpt_max_chars,
                            )
                            or hr.rerank_excerpt_max_chars
                        ),
                    ),
                )
            if "rerank_min_score" in new_config:
                hr.rerank_min_score = max(
                    0.0,
                    min(
                        1.0,
                        float(self.config.get("rerank_min_score", hr.rerank_min_score)),
                    ),
                )
            if "rerank_high_confidence_score" in new_config:
                hr.rerank_high_confidence_score = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self.config.get(
                                "rerank_high_confidence_score",
                                hr.rerank_high_confidence_score,
                            )
                        ),
                    ),
                )
            if "rerank_medium_confidence_cap" in new_config:
                hr.rerank_medium_confidence_cap = max(
                    1,
                    int(
                        self.config.get(
                            "rerank_medium_confidence_cap",
                            hr.rerank_medium_confidence_cap,
                        )
                        or hr.rerank_medium_confidence_cap
                    ),
                )
            if "rerank_timeout_seconds" in new_config:
                hr.rerank_timeout_seconds = max(
                    0.1,
                    float(
                        self.config.get(
                            "rerank_timeout_seconds", hr.rerank_timeout_seconds
                        )
                        or hr.rerank_timeout_seconds
                    ),
                )
            hr.fusion_mode = (
                (self.config.get("fusion_mode", hr.fusion_mode) or "score")
                .strip()
                .lower()
            )
            hr.fusion_bm25_weight = min(
                1.0,
                max(
                    0.0,
                    float(self.config.get("fusion_bm25_weight", hr.fusion_bm25_weight)),
                ),
            )
            hr.candidate_multiplier = max(
                1,
                min(
                    10,
                    int(
                        self.config.get(
                            "recall_candidate_multiplier",
                            hr.candidate_multiplier,
                        )
                        or 1
                    ),
                ),
            )
            hr.fusion_agreement_bonus = min(
                1.0,
                max(
                    0.0,
                    float(
                        self.config.get(
                            "fusion_agreement_bonus", hr.fusion_agreement_bonus
                        )
                    ),
                ),
            )
        if self.dual_route_retriever is not None:
            dr = self.dual_route_retriever
            dr.config = self.config
            dr.document_route_weight = float(
                self.config.get("document_route_weight", dr.document_route_weight)
            )
            dr.graph_route_weight = float(
                self.config.get("graph_route_weight", dr.graph_route_weight)
            )
            dr.cross_route_bonus = float(
                self.config.get("cross_route_bonus", dr.cross_route_bonus)
            )
            dr.dynamic_route_weighting = bool(
                self.config.get("dynamic_route_weighting", dr.dynamic_route_weighting)
            )
            dr.graph_route_timeout = float(
                self.config.get(
                    "graph_route_timeout_seconds",
                    self.config.get(
                        "graph_route_timeout",
                        getattr(dr, "graph_route_timeout", 1.5),
                    ),
                )
            )
            dr.rerank_post_fusion = bool(
                self.config.get("rerank_post_fusion", dr.rerank_post_fusion)
            )
            dr.rerank_candidates = max(
                2,
                min(
                    100,
                    int(
                        self.config.get("rerank_candidates", dr.rerank_candidates)
                        or dr.rerank_candidates
                    ),
                ),
            )
            dr.atom_route_weight = float(
                self.config.get("atom_route_weight", dr.atom_route_weight)
            )
            dr.affinity_session_bonus = float(
                self.config.get("affinity_session_bonus", dr.affinity_session_bonus)
            )
            dr.affinity_speaker_bonus = float(
                self.config.get("affinity_speaker_bonus", dr.affinity_speaker_bonus)
            )
            dr.affinity_multi_route_bonus = float(
                self.config.get(
                    "affinity_multi_route_bonus", dr.affinity_multi_route_bonus
                )
            )
            dr.temporal_recent_bonus = float(
                self.config.get("temporal_recent_bonus", dr.temporal_recent_bonus)
            )
        if self.atom_retriever is not None and hasattr(self.atom_retriever, "config"):
            self.atom_retriever.config = self.config
        if self.atom_vector_retriever is not None and hasattr(
            self.atom_vector_retriever, "config"
        ):
            self.atom_vector_retriever.config = self.config
        if self.graph_retriever is not None and hasattr(self.graph_retriever, "config"):
            self.graph_retriever.config = self.config
        self._invalidate_search_cache()

    async def initialize(self):
        """初始化引擎：打开数据库、建表、迁移归属、装配文档路与图谱路组件。"""
        await self._open_database()
        await self._create_tables()
        from ..retrieval.narrative_history import ensure_scope_index

        await ensure_scope_index(self.db_connection)
        await self._migrate_adapter_ownership()
        await self._init_document_route()
        if self.graph_enabled and self.graph_vector_db is not None:
            await self._init_graph_route()
        if self._write_op_repair_enabled:
            await self._repair_incomplete_write_ops()

    async def _open_database(self) -> None:
        """打开主连接并应用进程级 journal 策略与忙等超时。"""
        self.db_connection = await open_connection(
            self.db_path, apply_journal_mode=True
        )
        self.db_connection.row_factory = aiosqlite.Row

    async def _migrate_adapter_ownership(self) -> None:
        """为历史记忆补写适配器归属。"""
        self.adapter_migrator = AdapterMigrator(
            self.db_path,
            self.config.get("graph_document_path"),
            self.config.get("adapter_instances"),
        )
        await self.adapter_migrator.migrate()

    async def _init_document_route(self) -> None:
        """装配文本处理、BM25、向量、混合检索与写侧索引维护器。"""
        stopwords_path = self.config.get("stopwords_path")
        self.text_processor = TextProcessor(
            stopwords_path,
            tokenizer=str(self.config.get("tokenizer", "jieba")),
        )

        self.rrf_fusion = RRFFusion(k=self.config.get("rrf_k", 60))

        self.bm25_retriever = BM25Retriever(
            self.db_path, self.text_processor, self.config, cache_reads=True
        )
        await self.bm25_retriever.initialize()
        self.vector_retriever = VectorRetriever(self.faiss_db, self.config)
        self.hybrid_retriever = HybridRetriever(
            self.bm25_retriever,
            self.vector_retriever,
            self.rrf_fusion,
            self.config,
            rerank_provider_resolver=self.rerank_provider_resolver,
        )
        self.index_writer = MemoryIndexWriter(
            self.bm25_retriever, self.vector_retriever, self._write_transaction
        )

    async def _init_graph_route(self) -> None:
        """装配图谱存储、原子生命周期与双路检索。"""
        self.graph_store = GraphStore(self.db_path, cache_reads=True)
        await self.graph_store.initialize()

        self.atom_store = AtomStore(
            self.db_path,
            cache_reads=True,
            text_processor=self.text_processor,
        )
        await self.atom_store.initialize()

        if self.atom_enabled:
            self.atom_lifecycle_manager = AtomLifecycleManager(
                self.atom_store, self.config
            )
            emb_provider = getattr(self.faiss_db, "embedding_provider", None)
            if emb_provider is not None and bool(
                self.config.get("atom_vector_enabled", True)
            ):
                atom_index_path = str(
                    self.config.get("atom_index_path")
                    or (Path(self.db_path).parent / "livingmemory_atoms.index")
                )
                self.atom_vector_retriever = AtomVectorRetriever(
                    self.atom_store,
                    emb_provider,
                    index_path=atom_index_path,
                    config=self.config,
                )
                await self.atom_vector_retriever.initialize()
                self.atom_store.set_vector_sync(self.atom_vector_retriever)
                if self.index_writer is not None:
                    self.index_writer.atom_vector_retriever = self.atom_vector_retriever
            self.atom_retriever = AtomRetriever(
                self.atom_store,
                self.config,
                atom_vector_retriever=self.atom_vector_retriever,
            )
            await self.atom_lifecycle_manager.start()

        self.graph_extractor = GraphExtractor(self.config)
        self.graph_keyword_retriever = GraphKeywordRetriever(
            self.graph_store,
            self.text_processor,
            self.config,
        )
        self.graph_vector_retriever = GraphVectorRetriever(
            self.graph_vector_db,
            self.config,
        )
        self.graph_retriever = GraphRetriever(
            self.graph_keyword_retriever,
            self.graph_vector_retriever,
            self.rrf_fusion,
            self.config,
            source_validity_checker=self.get_active_graph_source_ids,
        )
        self.graph_memory_manager = GraphMemoryManager(
            self.graph_store,
            self.graph_vector_retriever,
            self.graph_extractor,
            source_validity_checker=self.get_active_graph_source_ids,
            source_state_guard=self.source_state_guard,
        )
        self.dual_route_retriever = DualRouteRetriever(
            self.hybrid_retriever,
            self.graph_retriever,
            self.get_memory,
            self.config,
            memory_batch_loader=partial(retrieve_documents, self.faiss_db),
            atom_retriever=self.atom_retriever,
        )

    async def rebuild_atom_vector_index(
        self,
        *,
        batch_size: int = 32,
        request_delay: float = 0.0,
        progress_callback=None,
    ) -> dict[str, int]:
        """Batch-build or rebuild livingmemory_atoms.index from all active atoms."""
        if self.atom_vector_retriever is None:
            return {"total": 0, "indexed": 0, "failed": 0}
        result = await self.atom_vector_retriever.rebuild_from_store(
            batch_size=batch_size,
            request_delay=request_delay,
            progress_callback=progress_callback,
        )
        self._invalidate_search_cache()
        return result

    async def close(self):
        """关闭数据库连接和清理资源"""
        if self.atom_lifecycle_manager is not None:
            await self.atom_lifecycle_manager.stop()
        if self._pending_tasks:
            for task in self._pending_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
        if self.db_connection:
            try:
                await self._flush_access_updates()
            except (asyncio.CancelledError, Exception):
                logger.warning("关闭前写入访问统计失败", exc_info=True)
            if self.db_connection.in_transaction:
                await self.db_connection.rollback()
            await self.db_connection.close()
        # B10：释放连接池里缓存的空闲连接（顺序：先关池，再关常驻连接已在上方完成）。
        for store in (self.atom_store, self.graph_store, self.bm25_retriever):
            close = getattr(store, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    logger.warning("[MemoryEngine] 关闭存储连接池失败", exc_info=True)
        if self.atom_vector_retriever is not None:
            try:
                await self.atom_vector_retriever.close()
            except Exception:
                logger.warning("[MemoryEngine] 关闭原子向量索引失败", exc_info=True)
        if self.graph_vector_db is not None:
            await self.graph_vector_db.close()

    async def _close_read_connections(self) -> None:
        """Close reusable retrieval read connections (test doubles are skipped)."""
        for component in (self.atom_store, self.bm25_retriever):
            close = getattr(component, "close", None)
            if close is not None and asyncio.iscoroutinefunction(close):
                await close()

    @asynccontextmanager
    async def _write_transaction(self):
        """Serialize isolated writes without upgrading a concurrent read snapshot."""
        if self.db_connection is None:
            raise RuntimeError("memory database is not initialized")

        async with self._db_write_lock:
            # Preserve recovery for a legacy write left open on the main
            # connection. Ordinary SELECT cursors do not set in_transaction;
            # their snapshots remain separate from the new writer below.
            if self.db_connection.in_transaction:
                await settle(self.db_connection.rollback())
            # Recall and diagnostics can have a live SELECT on db_connection
            # while FAISS/BM25 commits on another connection. Reusing that
            # reader for BEGIN IMMEDIATE fails with SQLITE_BUSY_SNAPSHOT,
            # regardless of busy_timeout. It also exposes uncommitted writes
            # to readers sharing the connection. Keep the writer private.
            async with write_transaction(self) as connection:
                yield connection

    def _create_tracked_task(
        self, coro, *, clean_context: bool = False
    ) -> asyncio.Task:
        """Create and track a background task, auto-discarding on completion."""
        if clean_context:
            import contextvars

            task = contextvars.Context().run(asyncio.create_task, coro)
        else:
            task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        import contextvars

        task.add_done_callback(
            self._on_background_task_done, context=contextvars.Context()
        )
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        """取回后台任务异常并记录，避免静默失败。"""
        self._pending_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"[MemoryEngine] 后台任务失败: {exc}", exc_info=exc)

    async def _create_tables(self):
        """创建数据库表

        注意：documents 表主要由 FAISS 的 DocumentStorage 类创建和管理。
        这里使用 CREATE TABLE IF NOT EXISTS 确保兼容性：
        - 如果 FAISS 已创建，不会重复创建（IF NOT EXISTS）
        - 如果 FAISS 未创建（极端情况），插件仍能正常工作
        - 插件需要直接操作此表进行高频更新（如访问时间）
        """
        async with self._write_transaction() as connection:
            await self._drop_legacy_documents_fts_triggers(connection)

            # documents表 - 与FAISS共享，IF NOT EXISTS确保不重复创建
            await connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                text TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                updated_at TEXT
            )
        """)

            # 兼容旧版插件创建的简化 documents 表，确保 FAISS DocumentStorage 所需字段存在
            cursor = await connection.execute("PRAGMA table_info(documents)")
            column_rows = await cursor.fetchall()
            existing_columns = {row[1] for row in column_rows}

            missing_columns = []
            if "doc_id" not in existing_columns:
                await connection.execute("ALTER TABLE documents ADD COLUMN doc_id TEXT")
                missing_columns.append("doc_id")
            if "created_at" not in existing_columns:
                await connection.execute(
                    "ALTER TABLE documents ADD COLUMN created_at TEXT"
                )
                missing_columns.append("created_at")
            if "updated_at" not in existing_columns:
                await connection.execute(
                    "ALTER TABLE documents ADD COLUMN updated_at TEXT"
                )
                missing_columns.append("updated_at")

            if missing_columns:
                logger.warning(
                    "[MemoryEngine] 检测到旧版 documents 表结构，已补齐字段: "
                    f"{', '.join(missing_columns)}"
                )

            # 回填旧数据，避免 doc_id/timestamp 缺失导致删除与展示异常
            await connection.execute("""
            UPDATE documents
            SET doc_id = 'legacy-' || id
            WHERE doc_id IS NULL OR TRIM(doc_id) = ''
        """)
            await connection.execute("""
            UPDATE documents
            SET created_at = datetime('now')
            WHERE created_at IS NULL OR TRIM(CAST(created_at AS TEXT)) = ''
        """)
            await connection.execute("""
            UPDATE documents
            SET updated_at = COALESCE(created_at, datetime('now'))
            WHERE updated_at IS NULL OR TRIM(CAST(updated_at AS TEXT)) = ''
        """)

            await self._create_write_ops_table(connection)

            # 索引 DDL 与迁移/校验器共用 storage.schema 中的单一定义
            for statement in DOCUMENTS_INDEX_STATEMENTS:
                await connection.execute(statement)
            await connection.execute(DOCUMENTS_DOC_ID_INDEX_STATEMENT)

            await connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_sources (
                memory_id INTEGER PRIMARY KEY,
                source_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

            # 版本管理 / 迁移状态表：与 DBMigration、IndexValidator 共用同一份 DDL
            for statement in DB_VERSION_SCHEMA_STATEMENTS:
                await connection.execute(statement)
            for statement in MIGRATION_STATUS_SCHEMA_STATEMENTS:
                await connection.execute(statement)

            # 检查是否需要初始化版本信息
            cursor = await connection.execute("SELECT COUNT(*) FROM db_version")
            version_result = await cursor.fetchone()
            version_count = version_result[0] if version_result else 0

            if version_count == 0:
                # 全新数据库，设置初始版本为最新迁移版本
                from datetime import datetime, timezone

                from ...storage.db_migration import DBMigration

                await connection.execute(
                    """
                    INSERT INTO db_version (version, description, migrated_at, migration_duration_seconds)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        DBMigration.CURRENT_VERSION,
                        "初始版本 - 当前架构",
                        datetime.now(timezone.utc).isoformat(),
                        0.0,
                    ),
                )

                logger.info(f"已初始化数据库版本信息: v{DBMigration.CURRENT_VERSION}")

    async def get_active_graph_source_ids(
        self, source_ids: Sequence[int] | set[int]
    ) -> set[int]:
        """Read current source validity using the graph rebuild's SQL rule.

        This deliberately queries the main documents table on every call so an
        archive, deletion, or source revision takes effect on the next query.
        Missing database state fails closed.
        """
        if self.db_connection is None or not source_ids:
            return set()

        active_ids: set[int] = set()
        unique_ids = sorted({int(source_id) for source_id in source_ids})
        batch_size = 800
        for start in range(0, len(unique_ids), batch_size):
            batch = unique_ids[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            cursor = await self.db_connection.execute(
                "SELECT id FROM documents WHERE id IN ("
                + placeholders
                + ") AND COALESCE(json_extract(metadata, '$.status'), 'active') = 'active'",
                batch,
            )
            active_ids.update(int(row[0]) for row in await cursor.fetchall())
        return active_ids

    async def plan_graph_vector_prune(self) -> RepairPlan:
        """Return a dry-run plan for stale vectors; performs no index writes."""
        if self.graph_memory_manager is None:
            raise RuntimeError("graph memory is not initialized")
        return await self.graph_memory_manager.plan_stale_vector_prune()

    async def apply_graph_vector_prune(self, plan: RepairPlan) -> tuple[int, ...]:
        """Apply the exact stale-vector IDs from a reviewed plan."""
        if self.graph_memory_manager is None:
            raise RuntimeError("graph memory is not initialized")
        return await self.graph_memory_manager.apply_stale_vector_prune(plan)

    # ==================== 核心记忆操作 ====================

    # ==================== 高级功能 ====================
