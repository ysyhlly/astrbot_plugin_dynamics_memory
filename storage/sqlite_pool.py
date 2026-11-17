"""可复用的 aiosqlite 连接池（B10 / T2-L01）。

背景
----
`AtomStore` / `GraphStore` / `BM25Retriever` 原来在每个存储操作里
`aiosqlite.connect()` 新建连接，并重跑 `PRAGMA journal_mode = WAL` 等设置
（三处 `_connect()` 共 40+ 个调用点）。`aiosqlite.connect()` 会为每个连接创建
一个工作线程，`journal_mode` 也需要一次短暂的库级加锁，因此高频召回/写入下
这部分固定开销可观；而 `journal_mode` 是**持久化**的库属性，重复设置没有收益。

设计
----
"借出—归还"式小连接池：**每次 `async with pool.connection()` 独占一个连接**，
因此事务边界与回滚语义与"每操作一条新连接"**完全一致**——同一时刻不会有两个
协程共享同一连接，也就不存在"别人的语句混进我的事务 / 被我的 commit 提交"的问题。

- 连接创建与 PRAGMA 只在池需要增长时发生一次，之后复用；
- 归还前执行一次 `rollback()`，等价于原实现 `finally: db.close()` 对未提交事务的
  丢弃（`rollback()` 在已提交/无事务时是 no-op）；
- 异常路径与显式 `discard=True` 的连接直接关闭，不再复用（避免把未知状态的连接
  或带 `ATTACH`/`PRAGMA foreign_keys=OFF` 等会话级状态的连接放回池里）；
- 池关闭后仍可用（退化为一次性连接），因此收尾顺序不会让调用方失败。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable

import aiosqlite

from .sqlite_write import settle

#: 池中最多缓存的空闲连接数（并发借出超过它时会临时新建并在归还时关闭）。
DEFAULT_POOL_SIZE = 4


async def _safe_close(db: Any) -> None:
    """尽力关闭连接；任何失败都不应影响调用方。"""
    try:
        await settle(db.close())
    except Exception:
        pass


class SQLiteConnectionPool:
    """按需创建、按需复用 aiosqlite 连接的小连接池。

    Attributes:
        created_count: 迄今为止真正 `connect()` 出来的连接数（自证用）。
        pragma_runs: 迄今执行的 PRAGMA 语句数（自证用）。
        reused_count: 从空闲池成功复用的次数（自证用）。
        borrowed_count: 借出次数（含新建），用于算复用率。
    """

    def __init__(
        self,
        db_path: str,
        *,
        pragmas: Iterable[str] = (),
        size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        self._db_path = db_path
        self._pragmas = tuple(pragmas)
        self._size = max(1, int(size))
        self._idle: list[Any] = []
        self._lock = asyncio.Lock()
        self._closed = False
        # 统计（只读，供自证与诊断）
        self.created_count = 0
        self.pragma_runs = 0
        self.reused_count = 0
        self.borrowed_count = 0
        self.discarded_count = 0

    async def _new_connection(self) -> Any:
        db = await aiosqlite.connect(self._db_path)
        self.created_count += 1
        try:
            for statement in self._pragmas:
                await db.execute(statement)
                self.pragma_runs += 1
        except BaseException:
            await _safe_close(db)
            raise
        return db

    async def _release(self, db: Any, *, discard: bool) -> None:
        # 与"每次操作新建连接"等价的收尾：丢弃未提交事务。
        try:
            await settle(db.rollback())
        except Exception:
            discard = True
        if discard or self._closed:
            self.discarded_count += 1
            await _safe_close(db)
            return
        async with self._lock:
            if len(self._idle) < self._size:
                self._idle.append(db)
                return
        # 池已满：多出来的连接直接关闭，回到"随用随开"的行为。
        self.discarded_count += 1
        await _safe_close(db)

    @asynccontextmanager
    async def connection(self, *, discard: bool = False) -> AsyncIterator[Any]:
        """借出一个连接；退出时归还（或按 discard 关闭）。

        Args:
            discard: True 表示本块结束后不再复用该连接（用于会话级状态被改动、
                或有 `ATTACH` 等副作用的情境）。

        Yields:
            `aiosqlite.Connection`。调用方独占它直到退出本 `async with`。
        """
        if self._closed:
            # 收尾之后仍允许使用：退化为一次性连接，语义与旧实现相同。
            db = await self._new_connection()
            self.borrowed_count += 1
            try:
                yield db
            finally:
                if discard:
                    self.discarded_count += 1
                await _safe_close(db)
            return

        db = None
        async with self._lock:
            if self._idle:
                db = self._idle.pop()
                self.reused_count += 1
        if db is None:
            db = await self._new_connection()
        self.borrowed_count += 1

        failed = False
        try:
            yield db
        except BaseException:
            failed = True
            raise
        finally:
            await self._release(db, discard=discard or failed)

    async def close_all(self) -> None:
        """关闭所有空闲连接并标记池已关闭（幂等；之后仍可安全借出）。"""
        self._closed = True
        async with self._lock:
            idle, self._idle = self._idle, []
        for db in idle:
            await _safe_close(db)

    def stats(self) -> dict[str, int]:
        """返回池计数快照（自证/诊断用）。"""
        return {
            "created": self.created_count,
            "pragmas": self.pragma_runs,
            "reused": self.reused_count,
            "borrowed": self.borrowed_count,
            "discarded": self.discarded_count,
            "idle": len(self._idle),
        }


__all__ = ["DEFAULT_POOL_SIZE", "SQLiteConnectionPool"]
