"""
会话存储层 - ConversationStore
负责管理会话和消息的持久化存储,使用 SQLite 数据库
"""

import contextvars
from .conversation_store_messages import ConversationStoreMessagesMixin
from .conversation_summary_progress import ConversationSummaryProgress


import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

from astrbot.api import logger

from ..core.models.conversation_models import Message, Session, serialize_to_json
from .sqlite_connection import open_connection


from .series_conflicts import SeriesConflictStore, valid_event


class ConversationStore(SeriesConflictStore, ConversationSummaryProgress):
    """
    会话存储管理器

    职责:
    - 管理会话和消息的持久化存储
    - 提供 CRUD 操作接口
    - 支持群聊场景的数据查询
    """

    def __init__(self, db_path: str):
        """
        初始化存储层

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

        # 确保数据库目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """初始化数据库连接并创建表结构"""
        self.connection = await open_connection(self.db_path, apply_journal_mode=True)
        self.connection.row_factory = aiosqlite.Row
        await self._create_tables()
        await self._create_indexes()
        await self.initialize_series_conflicts()
        await self.initialize_summary_progress()
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS narrative_events (
                event_key TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                event_id TEXT NOT NULL, envelope TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_narrative_scope_time ON narrative_events(session_id, timestamp DESC)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_narrative_event_id ON narrative_events(event_id)"
        )
        await self.connection.commit()

        logger.info(f"[ConversationStore] 数据库初始化完成: {self.db_path}")

    async def close(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("[ConversationStore] 数据库连接已关闭")

    async def _create_tables(self) -> None:
        """创建数据库表结构"""
        # sessions 表 - 会话元数据
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                platform TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_active_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                participants TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)

        # messages 表 - 消息记录
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT,
                group_id TEXT,
                platform TEXT,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        await self.connection.commit()

    async def _create_indexes(self) -> None:
        """创建索引以优化查询性能"""
        # sessions 表索引
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_id ON sessions(session_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_last_active ON sessions(last_active_at DESC)"
        )

        # messages 表索引
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, timestamp DESC)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_sender ON messages(session_id, sender_id, timestamp DESC)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON messages(timestamp DESC)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_session_id ON messages(session_id, id)"
        )

        await self.connection.commit()

    # ==================== 会话管理 ====================

    async def create_session(self, session_id: str, platform: str) -> Session:
        """
        创建新会话

        Args:
            session_id: 会话唯一标识
            platform: 平台类型

        Returns:
            Session: 创建的会话对象
        """
        now = time.time()

        # 确保 platform 是字符串类型
        if not isinstance(platform, str):
            # 如果是 PlatformMetadata 对象，提取 name 属性
            platform = getattr(platform, "name", str(platform))
            logger.warning(
                f"[create_session] platform 参数不是字符串类型，已自动转换为: {platform}"
            )

        async with self._write_lock:
            await self.connection.execute(
                """
                INSERT INTO sessions (session_id, platform, created_at, last_active_at, message_count, participants, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
            """,
                (session_id, platform, now, now, 0, "[]", "{}"),
            )
            await self.connection.commit()

        session = await self.get_session(session_id)
        if session is not None:
            logger.debug(f"[ConversationStore] 会话已存在: {session_id}")
            return session

        session = Session(
            id=0,
            session_id=session_id,
            platform=platform,
            created_at=now,
            last_active_at=now,
            message_count=0,
            participants=[],
            metadata={},
        )

        logger.debug(f"[ConversationStore] 创建会话: {session_id}")
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            Optional[Session]: 会话对象,不存在则返回 None
        """
        async with self.connection.execute(
            """
            SELECT id, session_id, platform, created_at, last_active_at,
                   message_count, participants, metadata
            FROM sessions
            WHERE session_id = ?
        """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return Session.from_dict(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "platform": row["platform"],
                "created_at": row["created_at"],
                "last_active_at": row["last_active_at"],
                "message_count": row["message_count"],
                "participants": row["participants"],
                "metadata": row["metadata"],
            }
        )

    async def update_session_activity(self, session_id: str) -> None:
        """
        更新会话最后活跃时间

        Args:
            session_id: 会话ID
        """
        now = time.time()

        async with self._write_lock:
            await self.connection.execute(
                """
                UPDATE sessions
                SET last_active_at = ?
                WHERE session_id = ?
            """,
                (now, session_id),
            )
            await self.connection.commit()

    async def get_recent_sessions(self, limit: int = 10) -> list[Session]:
        """
        获取最近活跃的会话

        Args:
            limit: 返回数量限制

        Returns:
            List[Session]: 会话列表
        """
        async with self.connection.execute(
            """
            SELECT id, session_id, platform, created_at, last_active_at,
                   message_count, participants, metadata
            FROM sessions
            ORDER BY last_active_at DESC
            LIMIT ?
        """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

        sessions = []
        for row in rows:
            sessions.append(
                Session.from_dict(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "platform": row["platform"],
                        "created_at": row["created_at"],
                        "last_active_at": row["last_active_at"],
                        "message_count": row["message_count"],
                        "participants": row["participants"],
                        "metadata": row["metadata"],
                    }
                )
            )

        return sessions

    # ==================== 消息管理 ====================

    @staticmethod
    def _row_to_message(row: aiosqlite.Row) -> Message:
        """把 messages 表行转换为 Message 对象。"""
        return Message.from_dict(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "sender_id": row["sender_id"],
                "sender_name": row["sender_name"],
                "group_id": row["group_id"],
                "platform": row["platform"],
                "timestamp": row["timestamp"],
                "metadata": row["metadata"],
            }
        )

    async def list_session_ids(self) -> list[str]:
        """Return every known session id for import session-id remapping."""
        async with self.connection.execute("SELECT session_id FROM sessions") as cursor:
            rows = await cursor.fetchall()
        return [str(row["session_id"]) for row in rows if row["session_id"]]

    async def add_message(self, message: Message) -> int:
        """
        添加消息到数据库

        Args:
            message: 消息对象

        Returns:
            int: 消息ID
        """
        platform = message.platform or "unknown"
        if not isinstance(platform, str):
            platform = getattr(platform, "name", str(platform))
            logger.warning(
                f"[add_message] platform 参数不是字符串类型，已自动转换为: {platform}"
            )

        sender_id = message.sender_id or message.session_id
        content = Message.content_to_text(message.content)
        now = time.time()
        async with self._write_lock:
            try:
                await self.connection.execute(
                    """
                    INSERT INTO sessions (
                        session_id, platform, created_at, last_active_at,
                        message_count, participants, metadata
                    )
                    VALUES (?, ?, ?, ?, 0, '[]', '{}')
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (message.session_id, platform, now, message.timestamp),
                )

                cursor = await self.connection.execute(
                    """
                    INSERT INTO messages (
                        session_id, role, content, sender_id, sender_name,
                        group_id, platform, timestamp, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.session_id,
                        message.role,
                        content,
                        sender_id,
                        message.sender_name,
                        message.group_id,
                        platform,
                        message.timestamp,
                        serialize_to_json(message.metadata),
                    ),
                )

                message_id = cursor.lastrowid if cursor.lastrowid else 0

                await self.connection.execute(
                    """
                    UPDATE sessions
                    SET message_count = message_count + 1,
                        last_active_at = ?,
                        participants = CASE
                            WHEN ? = '' THEN participants
                            WHEN EXISTS (
                                SELECT 1
                                FROM json_each(COALESCE(NULLIF(participants, ''), '[]'))
                                WHERE value = ?
                            ) THEN participants
                            ELSE json_insert(
                                COALESCE(NULLIF(participants, ''), '[]'),
                                '$[#]',
                                ?
                            )
                        END
                    WHERE session_id = ?
                    """,
                    (
                        message.timestamp,
                        sender_id,
                        sender_id,
                        sender_id,
                        message.session_id,
                    ),
                )
                await self._commit_write_transaction()
            except BaseException:
                # aiosqlite work already queued on its worker thread is ordered
                # before this rollback. Keep the writer lock until that rollback
                # completes so timeout cancellation cannot leave a late row.
                try:
                    await asyncio.shield(self.connection.rollback())
                except BaseException:
                    pass
                raise

        logger.debug(
            f"[ConversationStore] 添加消息: session={message.session_id}, role={message.role}"
        )
        return message_id

    async def get_messages(
        self, session_id: str, limit: int = 50, sender_id: str | None = None
    ) -> list[Message]:
        """
        获取会话消息 (支持按发送者过滤)

        Args:
            session_id: 会话ID
            limit: 限制数量
            sender_id: 可选,按发送者ID过滤

        Returns:
            List[Message]: 消息列表 (按时间升序)
        """
        if sender_id:
            # 按发送者过滤
            query = """
                SELECT id, session_id, role, content, sender_id, sender_name,
                       group_id, platform, timestamp, metadata
                FROM messages
                WHERE session_id = ? AND sender_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            """
            params = (session_id, sender_id, limit)
        else:
            # 获取所有消息
            query = """
                SELECT id, session_id, role, content, sender_id, sender_name,
                       group_id, platform, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            """
            params = (session_id, limit)

        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        messages = []
        for row in rows:
            messages.append(self._row_to_message(row))

        # 反转列表,返回时间升序
        messages.reverse()
        return messages

    async def get_message_count(self, session_id: str) -> int:
        """
        获取会话的消息总数

        Args:
            session_id: 会话ID

        Returns:
            int: 消息数量
        """
        async with self.connection.execute(
            """
            SELECT COUNT(*) as count
            FROM messages
            WHERE session_id = ?
        """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["count"])

    async def native_summary_window_start(
        self, session_id: str, messages: list[Message], metadata: dict[str, Any],
    ) -> int | None:
        """Locate an unchanged window covering the first unsummarized row."""
        if not messages:
            return None
        first = messages[0]
        async with self.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=? AND (timestamp,id) < (?,?)",
            (session_id, first.timestamp, first.id),
        ) as cursor:
            start = (await cursor.fetchone())[0]
        progress = int(metadata.get("last_summarized_index", 0) or 0)
        if not start <= progress < start + len(messages):
            return None
        current = await self.get_messages_range(session_id, start, len(messages))
        return start if current == messages else None

    async def write_native_pending_summary(
        self, session_id: str, messages: list[Message], pending: dict[str, Any],
    ) -> bool:
        """Persist the failed rows at their current offsets after a summarized-head trim."""
        if len(messages) < 2:
            return False
        async with self._write_lock:
            committed = False
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                session = await self.get_session(session_id)
                if session is None:
                    return False
                start = await self.native_summary_window_start(session_id, messages, session.metadata)
                if start is None:
                    return False
                session.metadata["pending_summary"] = {
                    **pending, "start_index": start, "end_index": start + len(messages),
                }
                await self.connection.execute(
                    "UPDATE sessions SET metadata=? WHERE session_id=?",
                    (json.dumps(session.metadata, ensure_ascii=False), session_id),
                )
                await self._commit_write_transaction()
                committed = True
                return True
            finally:
                if not committed:
                    await self._settle_write(self.connection.rollback())

    async def commit_summary_cursor(
        self, session_id: str, timestamp: float, message_id: int, *,
        window_messages: list[Message] | None = None,
    ) -> bool:
        """Commit the final row after verifying that the summary covers the current prefix."""
        if self.connection is None or type(message_id) is not int or message_id <= 0:
            return False
        async with self._write_lock:
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                async with self.connection.execute(
                    "SELECT id FROM messages WHERE session_id=? AND id=? AND timestamp=?",
                    (session_id, message_id, timestamp),
                ) as cursor:
                    anchor = await cursor.fetchone()
                if anchor is None:
                    await self.connection.rollback()
                    return False
                async with self.connection.execute(
                    "SELECT s.metadata, COUNT(m.id) AS position FROM sessions s LEFT JOIN messages m "
                    "ON m.session_id=s.session_id AND (m.timestamp < ? OR (m.timestamp=? AND m.id<=?)) "
                    "WHERE s.session_id=? GROUP BY s.session_id",
                    (timestamp, timestamp, message_id, session_id),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    await self.connection.rollback()
                    return False
                metadata = json.loads(row["metadata"] or "{}")
                if window_messages is not None:
                    start = await self.native_summary_window_start(session_id, window_messages, metadata)
                    if start is None or start + len(window_messages) != row["position"]:
                        await self.connection.rollback()
                        return False
                metadata["last_summarized_index"] = max(int(metadata.get("last_summarized_index", 0) or 0), int(row["position"]))
                metadata["pending_summary"] = None
                metadata["last_summary_message_v1"] = {"id": message_id, "timestamp": timestamp}
                await self.connection.execute("UPDATE sessions SET metadata=? WHERE session_id=?",
                    (json.dumps(metadata, ensure_ascii=False), session_id))
                try:
                    await self._commit_write_transaction()
                except asyncio.CancelledError as exc:
                    # The shared helper drains the SQLite commit before raising.
                    # Preserve a durable cursor receipt for reflection cleanup.
                    async with self.connection.execute("SELECT metadata FROM sessions WHERE session_id=?", (session_id,)) as cursor:
                        saved = await cursor.fetchone()
                    if saved and json.loads(saved[0]).get("last_summary_message_v1") == metadata["last_summary_message_v1"]:
                        exc.summary_cursor_committed = (session_id, message_id, timestamp)
                    raise
                return True
            except BaseException:
                await self.connection.rollback()
                raise

    async def trim_session_messages(
        self,
        session_id: str,
        delete_count: int,
        *,
        delete_unsummarized: bool = False,
    ) -> int:
        """Trim oldest messages, optionally including unprocessed history at the hard cap."""
        if delete_count <= 0:
            return 0

        # 整个读-改-写都在写锁内完成，避免与 add_message 交错导致
        # message_count / last_summarized_index 漂移（TOCTOU）。
        async with self._write_lock:
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                async with self.connection.execute(
                    """
                    SELECT
                        s.metadata,
                        COUNT(m.id) AS actual_count
                    FROM sessions s
                    LEFT JOIN messages m ON m.session_id = s.session_id
                    WHERE s.session_id = ?
                    GROUP BY s.session_id
                    """,
                    (session_id,),
                ) as cursor:
                    row = await cursor.fetchone()

                if not row:
                    return 0

                try:
                    metadata = json.loads(row["metadata"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}

                try:
                    last_summarized_index = int(
                        metadata.get("last_summarized_index", 0) or 0
                    )
                except (TypeError, ValueError):
                    last_summarized_index = 0
                last_summarized_index = max(0, last_summarized_index)

                actual_count = int(row["actual_count"] or 0)

                if last_summarized_index > actual_count and not delete_unsummarized:
                    metadata["last_summarized_index"] = 0
                    await self.connection.execute(
                        """
                        UPDATE sessions
                        SET metadata = ?,
                            message_count = ?
                        WHERE session_id = ?
                        """,
                        (
                            json.dumps(metadata, ensure_ascii=False),
                            actual_count,
                            session_id,
                        ),
                    )
                    await self._commit_write_transaction()
                    logger.warning(
                        f"[ConversationStore] 阻止清理未总结消息并重置 last_summarized_index: "
                        f"{session_id} ({last_summarized_index} > {actual_count})"
                    )
                    return 0

                if last_summarized_index > actual_count:
                    last_summarized_index = 0
                safe_delete_count = min(
                    delete_count,
                    actual_count if delete_unsummarized else last_summarized_index,
                )
                if safe_delete_count <= 0:
                    return 0

                cursor = await self.connection.execute(
                    """
                    DELETE FROM messages
                    WHERE id IN (
                        SELECT id FROM messages
                        WHERE session_id = ?
                        ORDER BY timestamp ASC, id ASC
                        LIMIT ?
                    )
                    """,
                    (session_id, safe_delete_count),
                )
                deleted_count = max(0, cursor.rowcount)
                if deleted_count <= 0:
                    return 0

                metadata["last_summarized_index"] = max(
                    0, last_summarized_index - deleted_count
                )
                # pending_summary 存的是**绝对下标**：滑动窗口从头部删除后必须整体前移，
                # 否则下一轮会从旧下标起步，跳过 [新游标, 旧 pending_start) 区间
                pending = metadata.get("pending_summary")
                if isinstance(pending, dict):
                    try:
                        pending_start = int(pending.get("start_index", 0) or 0)
                        pending_end = int(pending.get("end_index", 0) or 0)
                    except (TypeError, ValueError):
                        metadata.pop("pending_summary", None)
                    else:
                        shifted = dict(pending)
                        shifted["start_index"] = max(0, pending_start - deleted_count)
                        shifted["end_index"] = max(0, pending_end - deleted_count)
                        metadata["pending_summary"] = shifted
                kept_count = max(0, actual_count - deleted_count)
                self._clamp_summary_metadata(metadata, kept_count)
                if metadata.get("managed_summary_progress_v1"):
                    await self._cleanup_managed_summary_progress(session_id)
                await self.connection.execute(
                    """
                    UPDATE sessions
                    SET message_count = ?,
                        metadata = ?
                    WHERE session_id = ?
                    """,
                    (
                        kept_count,
                        json.dumps(metadata, ensure_ascii=False),
                        session_id,
                    ),
                )
                await self._commit_write_transaction()
            finally:
                # A cancelled aiosqlite statement can still be queued, so let
                # rollback run behind it before another writer gets this lock.
                await self._settle_write(self.connection.rollback())
        return deleted_count

    async def delete_session_messages(self, session_id: str) -> int:
        """
        删除会话的所有消息

        Args:
            session_id: 会话ID

        Returns:
            int: 删除的消息数量
        """
        async with self._write_lock:
            cursor = await self.connection.execute(
                """
                DELETE FROM messages
                WHERE session_id = ?
            """,
                (session_id,),
            )

            deleted_count = cursor.rowcount
            if session_id.startswith("narrative:v1:"):
                await self.connection.execute("DELETE FROM managed_summary_progress WHERE session_id=?", (session_id,))
                await self.connection.execute("DELETE FROM managed_summary_refusals WHERE session_id=?", (session_id,))

            await self.connection.execute(
                """
                UPDATE sessions
                SET message_count = 0
                WHERE session_id = ?
            """,
                (session_id,),
            )
            await self.connection.commit()

        logger.info(
            f"[ConversationStore] 删除会话消息: session={session_id}, count={deleted_count}"
        )
        return deleted_count

    async def get_session_messages_asc(
        self, session_id: str, limit: int | None = None
    ) -> list[Message]:
        """按会话顺序返回消息，含完整 metadata。

        排序键为 ``timestamp ASC, id ASC``，与滑动窗口/游标语义
        （get_messages_range 的 timestamp 序，trim 的 timestamp,id 序）保持一致；
        常规情况下与插入顺序（id 升序）等价，时钟跳变时以时间序为准。

        Args:
            session_id: 会话ID
            limit: 限制数量，None 表示不限制

        Returns:
            List[Message]: 消息列表（升序）
        """
        query = """
            SELECT id, session_id, role, content, sender_id, sender_name,
                   group_id, platform, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
        """
        params: tuple = (session_id,)
        if limit is not None:
            query += " LIMIT ?"
            params = (session_id, limit)

        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        messages = []
        for row in rows:
            messages.append(self._row_to_message(row))
        return messages

    async def get_session_message_refs_asc(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """按会话顺序返回对账所需的最小字段（id/role/metadata）。

        供对话历史对账使用，避免构造完整 Message 对象（跳过 content 等大字段）。
        排序键与 get_session_messages_asc 一致（``timestamp ASC, id ASC``）。

        Args:
            session_id: 会话ID

        Returns:
            List[dict]: [{"id", "role", "metadata"}]（metadata 已解析为 dict）
        """
        query = """
            SELECT id, role, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
        """
        refs: list[dict[str, Any]] = []
        async with self.connection.execute(query, (session_id,)) as cursor:
            rows = await cursor.fetchall()
        for row in rows:
            metadata = row["metadata"] or "{}"
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            refs.append(
                {
                    "id": row["id"],
                    "role": row["role"],
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }
            )
        return refs

    async def get_last_message(self, session_id: str) -> Message | None:
        """获取会话最后一条消息（按会话顺序）。

        Args:
            session_id: 会话ID

        Returns:
            Message 或 None
        """
        async with self.connection.execute(
            """
            SELECT id, session_id, role, content, sender_id, sender_name,
                   group_id, platform, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
        """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None
        return self._row_to_message(row)

    @staticmethod
    def _clamp_summary_metadata(metadata: dict, kept_count: int) -> dict:
        """把总结游标与待处理范围钳制到 ``kept_count`` 条消息以内。

        规则（滑动窗口、按位置删除、按 id 删除、元数据写入共用一套，避免语义分叉）：

        - ``last_summarized_index`` 收敛到 ``min(cursor, kept_count)``；
        - ``pending_summary`` 起点越过末尾则整条清除；终点越界则收敛；
          收敛后不足一轮（< 2 条）则清除。

        Args:
            metadata: 会话元数据（原地修改）
            kept_count: 保留的消息条数

        Returns:
            dict: 实际被修正的键（被清除的键值为 ``None``）
        """
        if (
            "last_summarized_index" not in metadata
            and "pending_summary" not in metadata
        ):
            return {}
        repaired: dict = {}
        try:
            cursor = int(metadata.get("last_summarized_index", 0) or 0)
        except (TypeError, ValueError):
            cursor = 0
        cursor = max(0, cursor)
        clamped_cursor = min(cursor, kept_count)
        metadata["last_summarized_index"] = clamped_cursor
        if clamped_cursor != cursor:
            repaired["last_summarized_index"] = clamped_cursor

        pending = metadata.get("pending_summary")
        if isinstance(pending, dict):
            before = json.dumps(pending, sort_keys=True, ensure_ascii=False)
            try:
                pending_start = int(pending.get("start_index", 0) or 0)
                pending_end = int(pending.get("end_index", 0) or 0)
            except (TypeError, ValueError):
                pending = None
            else:
                if pending_start >= kept_count:
                    pending = None
                else:
                    clamped_end = min(pending_end, kept_count)
                    if clamped_end - pending_start < 2:
                        pending = None
                    else:
                        pending["end_index"] = clamped_end
            if pending is None:
                metadata.pop("pending_summary", None)
            else:
                metadata["pending_summary"] = pending
            after_obj = metadata.get("pending_summary")
            after = (
                json.dumps(after_obj, sort_keys=True, ensure_ascii=False)
                if after_obj is not None
                else None
            )
            if before != after:
                repaired["pending_summary"] = after_obj
        return repaired

    async def update_session_metadata_atomic(
        self, session_id: str, metadata: dict
    ) -> dict:
        """原子写入会话元数据，并在同一临界区内钳制总结游标。

        与滑动窗口清理、对账删除共用 ``_write_lock``，因此"读消息数 -> 钳制 ->
        写入"不会被 trim/delete 穿插，避免把越界游标写回（进而使新一轮被跳过总结）。

        Args:
            session_id: 会话ID
            metadata: 完整的元数据字典（会被钳制后写入）

        Returns:
            dict: 实际被修正的键（无修正时为空字典）
        """
        async with self._write_lock:
            kept_count = await self.get_message_count(session_id)
            repaired = self._clamp_summary_metadata(metadata, kept_count)
            await self.connection.execute(
                """
                UPDATE sessions
                SET metadata = ?
                WHERE session_id = ?
                """,
                (json.dumps(metadata, ensure_ascii=False), session_id),
            )
            await self.connection.commit()
        return repaired

    async def delete_messages_from_position(
        self, session_id: str, position: int, message_ids: list[int] | None = None
    ) -> int:
        """从指定位置（0-based，按会话顺序）开始删除会话消息，并同步计数与总结游标。

        用于对话历史对账：WebUI 编辑/重试会截断 LLM 历史，本方法原子地删除
        插件会话库中对应的被撤销消息，同时修正 message_count、
        last_summarized_index 与 pending_summary，避免游标漂移。

        位置排序键为 ``timestamp ASC, id ASC``，与滑动窗口/游标语义保持一致
        （常规下等价于插入顺序；时钟跳变时以时间序为准）。

        Args:
            session_id: 会话ID
            position: 0-based 起始位置（该位置及之后的消息被删除，等于保留条数）；
                      position < 0 时不执行任何删除
            message_ids: 可选。给出时**只删除这些 id**（对账快照里确认失效的消息）。
                按位置/时间边界删除会连带删掉"快照之后并发新增的合法消息"，并在库中
                留下永久缺口，因此对账应传入快照 id 集合；此时 position 不参与删除条件

        Returns:
            int: 删除的消息数量
        """
        ids = [int(i) for i in (message_ids or []) if i is not None]
        if not ids and position < 0:
            return 0

        async with self._write_lock:
            if ids:
                # 精确按 id 删除：并发新增的消息不受影响
                placeholders = ",".join("?" for _ in ids)
                cursor = await self.connection.execute(
                    f"""
                    DELETE FROM messages
                    WHERE session_id = ? AND id IN ({placeholders})
                    """,
                    (session_id, *ids),
                )
                deleted_count = max(0, cursor.rowcount)
                if deleted_count <= 0:
                    return 0
                kept_count = await self.get_message_count(session_id)
            else:
                # 定位起始消息（按会话顺序），position == 0 表示从第一条删除
                async with self.connection.execute(
                    """
                    SELECT id, timestamp FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC, id ASC
                    LIMIT 1 OFFSET ?
                    """,
                    (session_id, position),
                ) as cursor:
                    row = await cursor.fetchone()

                if not row:
                    return 0
                start_id = row["id"]
                start_timestamp = row["timestamp"]

                cursor = await self.connection.execute(
                    """
                    DELETE FROM messages
                    WHERE session_id = ?
                      AND (timestamp > ? OR (timestamp = ? AND id >= ?))
                    """,
                    (session_id, start_timestamp, start_timestamp, start_id),
                )
                deleted_count = max(0, cursor.rowcount)
                if deleted_count <= 0:
                    return 0

                kept_count = position

            # 读取会话行以同步 message_count 与总结游标
            async with self.connection.execute(
                """
                SELECT metadata, message_count
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ) as cursor:
                session_row = await cursor.fetchone()

            if not session_row:
                await self.connection.commit()
                return deleted_count

            try:
                metadata = json.loads(session_row["metadata"] or "{}")
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}

            # 游标/待处理范围钳制：与按 id 删除、元数据原子写入共用同一套规则
            self._clamp_summary_metadata(metadata, kept_count)
            if metadata.get("managed_summary_progress_v1"):
                await self._cleanup_managed_summary_progress(session_id)
                await self._update_managed_summary_cursor(session_id, metadata)

            await self.connection.execute(
                """
                UPDATE sessions
                SET message_count = ?,
                    metadata = ?
                WHERE session_id = ?
                """,
                (
                    kept_count,
                    json.dumps(metadata, ensure_ascii=False),
                    session_id,
                ),
            )
            await self.connection.commit()

            logger.info(
                f"[ConversationStore] 删除会话消息（对账回滚）: session={session_id}, "
                f"position={position}, count={deleted_count}"
            )
            return deleted_count

    # ==================== 高级查询 ====================

    async def get_user_message_stats(self, session_id: str) -> dict[str, int]:
        """
        获取会话中各用户的消息统计 (群聊场景)

        Args:
            session_id: 会话ID

        Returns:
            Dict[str, int]: {sender_id: message_count}
        """
        async with self.connection.execute(
            """
            SELECT sender_id, COUNT(*) as count
            FROM messages
            WHERE session_id = ? AND role = 'user'
            GROUP BY sender_id
        """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        stats = {}
        for row in rows:
            stats[row["sender_id"]] = row["count"]

        return stats

    async def update_message_metadata(self, message_id: int, metadata: dict) -> bool:
        """
        更新消息的metadata

        Args:
            message_id: 消息ID
            metadata: 新的metadata字典

        Returns:
            bool: 是否更新成功
        """
        try:
            async with self._write_lock:
                await self.connection.execute(
                    """
                    UPDATE messages
                    SET metadata = ?
                    WHERE id = ?
                    """,
                    (json.dumps(metadata, ensure_ascii=False), message_id),
                )
                await self.connection.commit()
            logger.debug(f"[ConversationStore] 更新消息metadata: id={message_id}")
            return True
        except Exception as e:
            logger.error(f"更新消息metadata失败: {e}", exc_info=True)
            return False

    async def search_messages(
        self, session_id: str, keyword: str, limit: int = 20
    ) -> list[Message]:
        """
        搜索会话中包含关键词的消息

        Args:
            session_id: 会话ID
            keyword: 搜索关键词
            limit: 限制数量

        Returns:
            List[Message]: 匹配的消息列表
        """
        async with self.connection.execute(
            """
            SELECT id, session_id, role, content, sender_id, sender_name,
                   group_id, platform, timestamp, metadata
            FROM messages
            WHERE session_id = ? AND content LIKE ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        """,
            (session_id, f"%{keyword}%", limit),
        ) as cursor:
            rows = await cursor.fetchall()

        messages = []
        for row in rows:
            messages.append(self._row_to_message(row))

        return messages

    async def get_messages_range(
        self, session_id: str, offset: int = 0, limit: int = 50
    ) -> list[Message]:
        """
        按范围获取会话消息（使用 SQL OFFSET/LIMIT）

        Args:
            session_id: 会话ID
            offset: 跳过的消息数量（从最旧的开始计算）
            limit: 获取的消息数量

        Returns:
            List[Message]: 消息列表（按时间升序）
        """
        # 使用子查询确保按时间升序后再应用 OFFSET/LIMIT
        query = """
            SELECT id, session_id, role, content, sender_id, sender_name,
                   group_id, platform, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
            LIMIT ? OFFSET ?
        """

        async with self.connection.execute(
            query, (session_id, limit, offset)
        ) as cursor:
            rows = await cursor.fetchall()

        messages = []
        for row in rows:
            messages.append(self._row_to_message(row))

        logger.debug(
            f"[get_messages_range] session={session_id}, offset={offset}, "
            f"limit={limit}, 实际获取={len(messages)}条"
        )

        return messages

    async def sync_message_counts(self) -> dict[str, int]:
        """
        同步所有会话的 message_count 与实际消息数量

        用于修复 message_count 不一致的问题（如删除消息后未更新计数）

        Returns:
            Dict[str, int]: {session_id: 修正后的count}
        """
        fixed_sessions = {}

        try:
            async with self._write_lock:
                async with self.connection.execute(
                    """
                    SELECT s.session_id,
                           s.message_count AS recorded_count,
                           COUNT(m.id) AS actual_count
                    FROM sessions s
                    LEFT JOIN messages m ON m.session_id = s.session_id
                    GROUP BY s.session_id
                    HAVING s.message_count != COUNT(m.id)
                    """
                ) as cursor:
                    rows = await cursor.fetchall()

                for row in rows:
                    session_id = row["session_id"]
                    recorded_count = row["recorded_count"]
                    actual_count = int(row["actual_count"] or 0)
                    await self.connection.execute(
                        """
                        UPDATE sessions
                        SET message_count = ?
                        WHERE session_id = ?
                        """,
                        (actual_count, session_id),
                    )
                    fixed_sessions[session_id] = actual_count
                    logger.info(
                        f"[ConversationStore] 修复会话 message_count: "
                        f"{session_id} ({recorded_count} -> {actual_count})"
                    )

                if fixed_sessions:
                    await self.connection.commit()
                    logger.info(
                        f"[ConversationStore] 共修复 {len(fixed_sessions)} 个会话的 message_count"
                    )
                else:
                    logger.info(
                        "[ConversationStore] 所有会话的 message_count 均正确，无需修复"
                    )

            return fixed_sessions

        except Exception as e:
            logger.error(f"同步 message_count 失败: {e}", exc_info=True)
            return {}

    @staticmethod
    async def _settle_write(operation) -> None:
        """Drain an enqueued SQLite commit or rollback before releasing the writer."""
        task = contextvars.Context().run(asyncio.create_task, operation)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                break
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _commit_write_transaction(self) -> None:
        if self.connection is None:
            raise RuntimeError("数据库连接未初始化")
        try:
            await self._settle_write(self.connection.commit())
        except BaseException:
            # If the commit already completed, rollback has no work to do.
            await self._settle_write(self.connection.rollback())
            raise

    async def ingest_narrative_event(self, event: dict) -> bool:
        """Atomically archive evidence and append the actual conversation message."""
        from ..core.narrative_api import scope_key, timestamp

        if not valid_event(event):
            return False
        scope = event["scope"]
        session_id = scope_key(scope)
        event_id = str(event.get("event_id", "")).strip()
        text = str(event.get("text", "")).strip()
        if not event_id or not text:
            return False
        occurred_at = timestamp(event["occurred_at"])
        event_key = session_id + ":" + event_id
        actor = str(event.get("actor_id") or "bot")
        outgoing = event["type"] == "delivery.success"
        metadata = {
            "schema_version": 1, "source_event_id": event_id, "scope": scope,
            "fact_kind": "receipt" if outgoing else "observation",
            "is_bot_message": outgoing,
        }
        async with self._write_lock:
            try:
                cursor = await self.connection.execute(
                    "INSERT OR IGNORE INTO narrative_events VALUES (?, ?, ?, ?, ?)",
                    (event_key, session_id, event_id, json.dumps(event, ensure_ascii=False), occurred_at),
                )
                if cursor.rowcount == 0:
                    await self.record_series_collision(event_key, event)
                    await self.connection.commit()
                    return False
                await self.connection.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, platform, created_at, last_active_at, message_count, participants, metadata) VALUES (?, ?, ?, ?, 0, '[]', '{}')",
                    (session_id, scope["platform"], occurred_at, occurred_at),
                )
                await self.connection.execute(
                    "INSERT INTO messages (session_id, role, content, sender_id, sender_name, group_id, platform, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_id, "assistant" if outgoing else "user", text, actor, actor,
                     scope["umo"] if ":GroupMessage:" in scope["umo"] else None,
                     scope["platform"], occurred_at, json.dumps(metadata, ensure_ascii=False)),
                )
                await self.connection.execute(
                    "UPDATE sessions SET message_count=message_count+1, last_active_at=MAX(last_active_at, ?) WHERE session_id=?",
                    (occurred_at, session_id),
                )
                session = await self.get_session(session_id)
                if session.metadata.get("managed_summary_progress_v1"):
                    # A delayed receipt can sort before the previous cursor.
                    await self._update_managed_summary_cursor(session_id, session.metadata)
                    await self._save_managed_summary_metadata(session_id, session.metadata)
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise
        return True

    async def ingest_owned_private_original_event(
        self, event: dict, *, source_platform: str
    ) -> bool:
        """Atomically archive one authentic private event and project one UMO row.

        The canonical narrative envelope remains keyed by its four-field scope,
        while the single physical conversation row stays under the host UMO so
        existing private history windows, reflection, and source snapshots see it.
        A prior archive key blocks replay even if the raw row was later trimmed.
        """
        import math

        from ..core.memory_scope import classify_umo
        from ..core.narrative_api import scope_key, timestamp

        if (not isinstance(event, dict) or event.get("schema_version") != 1
                or event.get("type") != "message.received"):
            return False
        scope = event.get("scope")
        if not isinstance(scope, dict):
            return False
        try:
            canonical_scope = {key: scope[key] for key in
                               ("platform", "umo", "persona_id", "instance_id")}
            narrative_session = scope_key(canonical_scope)
        except (KeyError, TypeError, ValueError):
            return False
        umo = canonical_scope["umo"]
        if classify_umo(umo) != "private":
            return False
        event_id = event.get("event_id")
        actor_id = event.get("actor_id")
        text = event.get("text")
        if (not isinstance(event_id, str) or not event_id.startswith("cd-original-input-v1:")
                or not isinstance(actor_id, str) or not actor_id.strip()
                or not isinstance(text, str) or not text.strip()
                or not isinstance(source_platform, str) or not source_platform.strip()):
            return False
        try:
            occurred_at = timestamp(event.get("occurred_at"))
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(occurred_at) or occurred_at <= 0:
            return False
        archive_key = narrative_session + ":" + event_id
        envelope = {**event, "scope": canonical_scope, "occurred_at": occurred_at}
        metadata = {
            "schema_version": 1,
            "source_event_id": event_id,
            "scope": canonical_scope,
            "fact_kind": "observation",
            "is_bot_message": False,
        }
        if self.connection is None:
            raise RuntimeError("数据库连接未初始化")
        async with self._write_lock:
            try:
                # Gate by stable physical event identity, not persona/card scope:
                # the same source event projected under another character must
                # not append a second UMO row or resurrect a trimmed one.
                async with self.connection.execute(
                    "SELECT event_key FROM narrative_events WHERE event_id=? "
                    "AND json_extract(envelope, '$.scope.umo')=? LIMIT 1",
                    (event_id, umo),
                ) as cursor:
                    existing = await cursor.fetchone()
                    if existing is not None:
                        await self.record_series_collision(existing["event_key"], envelope)
                        await self.connection.commit()
                        return False
                await self.connection.execute(
                    "INSERT INTO narrative_events VALUES (?, ?, ?, ?, ?)",
                    (archive_key, narrative_session, event_id,
                     json.dumps(envelope, ensure_ascii=False), occurred_at),
                )
                await self.connection.execute(
                    "INSERT INTO sessions (session_id, platform, created_at, last_active_at, message_count, participants, metadata) "
                    "VALUES (?, ?, ?, ?, 0, '[]', '{}') ON CONFLICT(session_id) DO NOTHING",
                    (umo, source_platform.strip(), occurred_at, occurred_at),
                )
                await self.connection.execute(
                    "INSERT INTO messages (session_id, role, content, sender_id, sender_name, group_id, platform, timestamp, metadata) "
                    "VALUES (?, 'user', ?, ?, ?, NULL, ?, ?, ?)",
                    (umo, text, actor_id, actor_id, source_platform.strip(), occurred_at,
                     json.dumps(metadata, ensure_ascii=False)),
                )
                await self.connection.execute(
                    "UPDATE sessions SET message_count=message_count+1, last_active_at=MAX(last_active_at, ?), "
                    "participants=CASE WHEN EXISTS (SELECT 1 FROM json_each(COALESCE(NULLIF(participants,''),'[]')) WHERE value=?) "
                    "THEN participants ELSE json_insert(COALESCE(NULLIF(participants,''),'[]'),'$[#]',?) END "
                    "WHERE session_id=?",
                    (occurred_at, actor_id, actor_id, umo),
                )
                await self._commit_write_transaction()
            except BaseException:
                await self.connection.rollback()
                raise
        return True

    async def recall_narrative_events(self, scope: dict, query: str, limit: int = 5) -> list[dict]:
        """Recall exact-scope evidence without promoting user statements to fact."""
        from ..core.narrative_api import scope_key

        async with self.connection.execute(
            "SELECT envelope FROM narrative_events WHERE session_id=? ORDER BY timestamp DESC LIMIT 200",
            (scope_key(scope),),
        ) as cursor:
            rows = await cursor.fetchall()
        events = [json.loads(row[0]) for row in rows]
        terms = set(str(query).casefold().split())
        events.sort(key=lambda ev: sum(term in ev["text"].casefold() for term in terms), reverse=True)
        return [{
            "text": ("用户发言：" if ev["type"] == "message.received" else "已发送：") + ev["text"],
            "source_id": ev["event_id"], "occurred_at": ev["occurred_at"],
            "scope": dict(ev["scope"]),
            "fact_kind": "observation" if ev["type"] == "message.received" else "receipt",
            "confidence": 1.0,
        } for ev in events[:limit]]

    async def _read_narrative_event_rows(self, event_keys: list[str]) -> dict:
        """Read each exact primary key once; callers still verify scope/type/author."""
        if self.connection is None:
            return {}
        unique = list(dict.fromkeys(event_keys))
        rows = {}
        for offset in range(0, len(unique), 200):
            chunk = unique[offset:offset + 200]
            placeholders = ",".join("?" for _ in chunk)
            async with self.connection.execute(
                f"SELECT event_key,session_id,envelope FROM narrative_events WHERE event_key IN ({placeholders})", chunk,
            ) as cursor:
                for key, session_id, raw in await cursor.fetchall():
                    try:
                        event = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(event, dict):
                        rows[key] = (session_id, event)
        return rows

    async def load_narrative_events(self, scope: dict, source_event_ids: list[str]) -> dict:
        """Request-local batch of authoritative events; no persistent text cache."""
        from ..core.narrative_api import scope_key
        if self.connection is None or not isinstance(source_event_ids, list) or len(source_event_ids) > 1000:
            return {}
        session = scope_key(scope)
        ids = list(dict.fromkeys(eid.strip() for eid in source_event_ids if isinstance(eid, str) and eid.strip()))
        stored = await self._read_narrative_event_rows([session + ":" + eid for eid in ids])
        result = {}
        for eid in ids:
            row = stored.get(session + ":" + eid)
            if row is None or row[0] != session:
                continue
            event = row[1]
            if (event.get("event_id") != eid or event.get("type") not in {"message.received", "delivery.success"}
                    or not isinstance(event.get("scope"), dict) or any(
                        event["scope"].get(field) != scope.get(field)
                        for field in ("platform", "umo", "persona_id", "instance_id"))):
                continue
            result[eid] = event
        return result

    async def load_narrative_event(self, scope: dict, source_event_id: str) -> dict | None:
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            return None
        return (await self.load_narrative_events(scope, [source_event_id])).get(source_event_id.strip())

    @staticmethod
    def _narrative_source_plan(metadata: dict):
        from ..core.narrative_api import scope_key
        if not isinstance(metadata, dict):
            return None
        evidence = metadata.get("narrative_provenance") or []
        window = metadata.get("source_window")
        expected = window.get("session_id") if isinstance(window, dict) else None
        if not isinstance(evidence, list) or not 0 < len(evidence) <= 1000:
            return None
        keys = []
        for item in evidence:
            if not isinstance(item, dict) or not item.get("source_id") or not isinstance(item.get("scope"), dict):
                return None
            try:
                key = scope_key(item["scope"])
            except (TypeError, ValueError):
                return None
            if key != expected:
                return None
            keys.append(key + ":" + str(item["source_id"]))
        return evidence, keys, expected

    async def load_narrative_sources_many(self, metadatas: list[dict]) -> list:
        """Deduplicate source reads across selected summaries, preserving every plan."""
        from ..core.narrative_api import timestamp
        if not isinstance(metadatas, list) or len(metadatas) > 64:
            raise ValueError("narrative_source_batch_limit")
        plans = [self._narrative_source_plan(metadata) for metadata in metadatas]
        stored = await self._read_narrative_event_rows([key for plan in plans if plan for key in plan[1]])
        outputs = []
        for plan in plans:
            if plan is None:
                outputs.append(None)
                continue
            evidence, keys, expected = plan
            output = []
            for item, key in zip(evidence, keys):
                row = stored.get(key)
                if row is None or row[0] != expected:
                    output = None
                    break
                event, scope = row[1], item["scope"]
                if (event.get("event_id") != str(item["source_id"])
                        or event.get("type") not in {"message.received", "delivery.success"}
                        or not isinstance(event.get("scope"), dict)
                        or any(event["scope"].get(k) != scope.get(k) for k in ("platform", "umo", "persona_id", "instance_id"))):
                    output = None
                    break
                outgoing = event["type"] == "delivery.success"
                kind = "receipt" if outgoing else "observation"
                if item.get("fact_kind") != kind:
                    output = None
                    break
                output.append({
                    "id": 0, "session_id": expected, "role": "assistant" if outgoing else "user",
                    "content": event["text"], "sender_id": event["actor_id"], "sender_name": event["actor_id"],
                    "group_id": scope["umo"], "platform": scope["platform"], "timestamp": timestamp(event["occurred_at"]),
                    "metadata": {"schema_version": 1, "source_event_id": event["event_id"], "scope": scope,
                                 "fact_kind": kind, "is_bot_message": outgoing},
                })
            outputs.append(output)
        return outputs

    async def load_narrative_sources(self, metadata: dict) -> list[dict] | None:
        return (await self.load_narrative_sources_many([metadata]))[0]

    async def get_message_by_id(self, session_id: str, message_id: int) -> dict | None:
        """Read one exact message row without changing session state or access times."""
        if self.connection is None:
            return None
        if not isinstance(session_id, str) or not session_id.strip():
            return None
        if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
            return None
        async with self.connection.execute(
            "SELECT id,session_id,role,content,sender_id,sender_name,group_id,platform,timestamp,metadata "
            "FROM messages WHERE session_id=? AND id=? LIMIT 1",
            (session_id, message_id),
        ) as cursor:
            row = await cursor.fetchone()
        if not row or str(row["session_id"]) != session_id:
            return None
        return {key: row[key] for key in (
            "id", "session_id", "role", "content", "sender_id", "sender_name",
            "group_id", "platform", "timestamp", "metadata",
        )}

    async def get_activity_sources(self, umo: str, message_ids: list[str]) -> dict:
        """Resolve physical group IDs and managed envelopes from the owning store."""
        import hashlib

        from ..core.narrative_api import scope_key, timestamp

        if self.connection is None or not 1 <= len(message_ids) <= 32:
            return {}
        found = {message_id: [] for message_id in message_ids}
        params = ",".join("?" for _ in message_ids)
        message_id_expr = "CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.platform_message_id') END"
        async with self.connection.execute(
            f"SELECT *, {message_id_expr} AS platform_id FROM messages "
            f"WHERE session_id=? AND role='user' AND {message_id_expr} IN ({params})",
            (umo, *message_ids),
        ) as cursor:
            for row in await cursor.fetchall():
                record = dict(row)
                record["metadata"] = json.loads(record["metadata"])
                found[record.pop("platform_id")].append(record)

        event_ids = {
            "in:" + hashlib.sha256(f"{umo}|{message_id}".encode()).hexdigest(): message_id
            for message_id in message_ids
        }
        async with self.connection.execute(
            f"SELECT event_id,session_id,envelope FROM narrative_events "
            f"WHERE event_id IN ({params})", tuple(event_ids)
        ) as cursor:
            for row in await cursor.fetchall():
                try:
                    event = json.loads(row["envelope"])
                except (TypeError, ValueError):
                    continue
                message_id = event_ids[row["event_id"]]
                if (not valid_event(event) or event["type"] != "message.received"
                        or event["event_id"] != row["event_id"]
                        or event["scope"]["umo"] != umo or event.get("source_ids") != [message_id]):
                    continue
                scope = event["scope"]
                if scope_key(scope) != row["session_id"]:
                    continue
                found[message_id].append({
                    "id": 0, "session_id": row["session_id"], "role": "user",
                    "content": event["text"], "sender_id": event["actor_id"],
                    "sender_name": event["actor_id"], "group_id": umo,
                    "platform": scope["platform"], "timestamp": timestamp(event["occurred_at"]),
                    "metadata": {
                        "source_event_id": event["event_id"], "scope": scope,
                        "fact_kind": "observation", "schema_version": 1,
                    },
                })
        return found
