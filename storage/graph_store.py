"""SQLite-backed graph-memory storage."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from astrbot.api import logger

from .graph_store_read import GraphStoreReadMixin
from .graph_store_write import GraphStoreWriteMixin
from .schema import GRAPH_SCHEMA_STATEMENTS
from .sqlite_connection import open_connection
from .sqlite_write import write_transaction


class GraphStore(GraphStoreWriteMixin, GraphStoreReadMixin):
    """Persist graph nodes, edges, and searchable entries."""

    _SQLITE_BATCH_SIZE = 500
    _NODE_TOKEN_QUERY_BATCH_SIZE = 200
    _READ_POOL_SIZE = 4

    # initialize() 中赋值：trigram tokenizer 不可用时保持 False，供节点检索选择回退路径
    _node_fts_available: bool

    def __init__(self, db_path: str, *, cache_reads: bool = False):
        self.db_path = db_path
        # 召回链路只读查询使用小型连接池并发执行 WAL 读（最多 4 条），既避免每次
        # open/PRAGMA 开销，也不再把 5-7 个图查询串行锁在同一把 _read_lock 上。
        # 写路径仍按操作单独连接以隔离事务。
        self._cache_reads = cache_reads
        self._read_pool: list[aiosqlite.Connection] = []
        self._read_semaphore: asyncio.Semaphore | None = None
        self._read_pool_closed: bool = False
        self._closing_tasks: set[asyncio.Task[Any]] = set()
        # Backward-compat alias for callers/tests inspecting _read_conn.
        self._read_conn: aiosqlite.Connection | None = None
        self._read_lock: asyncio.Lock | None = None

    def _get_read_lock(self) -> asyncio.Lock:
        if self._read_lock is None:
            self._read_lock = asyncio.Lock()
        return self._read_lock

    def _get_read_semaphore(self) -> asyncio.Semaphore:
        if self._read_semaphore is None:
            self._read_semaphore = asyncio.Semaphore(self._READ_POOL_SIZE)
        return self._read_semaphore

    async def initialize(self) -> None:
        """Create tables used by the graph-memory layer."""
        async with write_transaction(self, foreign_keys=True) as db:
            for statement in GRAPH_SCHEMA_STATEMENTS:
                await db.execute(statement)
            # External-content triggers also cover graph snapshot replacement.
            existing = await db.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'livingmemory_graph_nodes_fts'"
            )
            needs_backfill = await existing.fetchone() is None
            try:
                await db.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS livingmemory_graph_nodes_fts
                    USING fts5(canonical_value, content='graph_nodes', content_rowid='id', tokenize='trigram')
                """)
                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS lmem_nodes_insert AFTER INSERT ON graph_nodes BEGIN
                        INSERT INTO livingmemory_graph_nodes_fts(rowid, canonical_value)
                        VALUES (new.id, new.canonical_value);
                    END
                """)
                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS lmem_nodes_delete AFTER DELETE ON graph_nodes BEGIN
                        INSERT INTO livingmemory_graph_nodes_fts(livingmemory_graph_nodes_fts, rowid, canonical_value)
                        VALUES ('delete', old.id, old.canonical_value);
                    END
                """)
                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS lmem_nodes_update AFTER UPDATE OF canonical_value ON graph_nodes BEGIN
                        INSERT INTO livingmemory_graph_nodes_fts(livingmemory_graph_nodes_fts, rowid, canonical_value)
                        VALUES ('delete', old.id, old.canonical_value);
                        INSERT INTO livingmemory_graph_nodes_fts(rowid, canonical_value)
                        VALUES (new.id, new.canonical_value);
                    END
                """)
                if needs_backfill:
                    await db.execute(
                        "INSERT INTO livingmemory_graph_nodes_fts(livingmemory_graph_nodes_fts) VALUES ('rebuild')"
                    )
                self._node_fts_available = True
            except aiosqlite.OperationalError:
                self._node_fts_available = False
                logger.warning(
                    "SQLite trigram tokenizer unavailable; using node substring search"
                )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_edges_semantic
                ON graph_edges(source_node_id, target_node_id, relation_type)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_entries_scope_latest
                ON graph_entries(session_id, persona_id, source_memory_id, id DESC)
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_entries_session_id ON graph_entries(session_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_entries_persona_id ON graph_entries(persona_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_entries_edge_id ON graph_entries(edge_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_entry_nodes_node ON graph_entry_nodes(node_id)"
            )

    @asynccontextmanager
    async def _connect(self, *, discard: bool = False):
        """创建新的SQLite连接（busy_timeout 与 journal 策略由连接层统一设置）。"""
        db = await open_connection(self.db_path)
        try:
            await db.execute("PRAGMA foreign_keys = ON")
            yield db
        finally:
            await db.close()

    @staticmethod
    def _interrupt_conn_quietly(conn: aiosqlite.Connection) -> None:
        try:
            raw_conn = getattr(conn, "_conn", None)
            interrupt = getattr(raw_conn, "interrupt", None)
            if callable(interrupt):
                interrupt()
        except Exception:
            pass

    @staticmethod
    async def _close_conn_quietly(conn: aiosqlite.Connection) -> None:
        try:
            await conn.close()
        except Exception:
            pass

    def _schedule_close_conn(self, conn: aiosqlite.Connection) -> None:
        self._interrupt_conn_quietly(conn)
        try:
            task = asyncio.create_task(self._close_conn_quietly(conn))
        except RuntimeError:
            return
        self._closing_tasks.add(task)
        task.add_done_callback(self._closing_tasks.discard)

    @asynccontextmanager
    async def _read_connect(self):
        """Borrow a pooled read-only WAL connection for concurrent reads."""
        if not self._cache_reads:
            async with self._connect() as db:
                yield db
            return
        self._read_pool_closed = False
        sem = self._get_read_semaphore()
        async with sem:
            from_pool = bool(self._read_pool)
            conn = (
                self._read_pool.pop()
                if from_pool
                else await open_connection(self.db_path)
            )
            self._read_conn = conn
            broken = False
            cancelled = False
            try:
                if not from_pool:
                    try:
                        cur = await conn.execute("PRAGMA query_only = ON")
                        await cur.close()
                    except Exception:
                        pass
                yield conn
            except BaseException as exc:
                broken = True
                cancelled = isinstance(exc, asyncio.CancelledError)
                raise
            finally:
                if (
                    broken
                    or self._read_pool_closed
                    or len(self._read_pool) >= self._READ_POOL_SIZE
                ):
                    if cancelled:
                        self._schedule_close_conn(conn)
                    else:
                        await self._close_conn_quietly(conn)
                else:
                    self._read_pool.append(conn)
                self._read_conn = self._read_pool[-1] if self._read_pool else None

    async def close(self) -> None:
        """关闭所有复用的只读连接（插件卸载 / 临时 shadow store 清理时调用）。"""
        self._read_pool_closed = True
        conns = list(self._read_pool)
        self._read_pool.clear()
        self._read_conn = None
        for conn in conns:
            await self._close_conn_quietly(conn)
        if self._closing_tasks:
            pending = list(self._closing_tasks)
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_json(payload: dict[str, Any] | None) -> str:
        return json.dumps(payload or {}, ensure_ascii=False)

    @staticmethod
    def _chunked(items: list[int], size: int) -> list[list[int]]:
        return [items[index : index + size] for index in range(0, len(items), size)]


__all__ = ["GraphStore"]
