"""SQLite 连接策略：按数据目录能力选择 journal mode。

WAL 依赖数据库同目录的 ``-shm`` 共享内存文件与 POSIX 文件锁。Docker Desktop
把宿主盘 bind mount 成 9p / gRPC-FUSE / virtiofs 等共享文件系统时，这些语义
不被完整支持，新连接加入 WAL 会失败并报 ``SQLITE_IOERR``（disk I/O error）。
插件启动时先识别挂载类型并探测 WAL，不可用就整体回退 rollback journal，
再把历史 WAL 库安全转回，避免运行期反复失败。
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

from astrbot.api import logger

JOURNAL_MODE_WAL = "WAL"
JOURNAL_MODE_DELETE = "DELETE"

BUSY_TIMEOUT_MS = 10000
_UNSET = object()

# 插件管理的 SQLite 库；判定不支持 WAL 时这些库都要转回 rollback journal。
_PLUGIN_DB_FILES = (
    "livingmemory.db",
    "livingmemory_graph_documents.db",
    "conversations.db",
)
_PROBE_DB_NAME = ".wal_probe.db"

# 无法承载 WAL 共享内存索引的文件系统：Docker Desktop / 虚拟机的共享挂载、
# 网络文件系统、FUSE 类挂载。overlay/ext4/xfs/btrfs 等本地文件系统不在此列。
_UNSAFE_FS_TYPES = frozenset(
    {
        "9p",
        "9p2000",
        "drvfs",
        "virtiofs",
        "grpcfuse",
        "osxfs",
        "vboxsf",
        "vmhgfs",
        "prl_fs",
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "smbfs",
        "afs",
        "ceph",
        "glusterfs",
        "lustre",
        "sshfs",
        "ecryptfs",
    }
)

# 进程级 journal 策略：resolve_journal_mode() 启动时确定，之后所有连接复用。
_journal_mode: str = JOURNAL_MODE_WAL
_resolved_dir: str | None = None


async def _close_unreturned_connection(connect_task) -> None:
    """Drain a pending SQLite open and close any connection it created."""
    try:
        db = await connect_task
    except (Exception, asyncio.CancelledError):
        return
    try:
        await db.close()
    except Exception:
        logger.exception("Failed to close a partially opened SQLite connection")


async def open_connection(
    db_path: str,
    *,
    apply_journal_mode: bool = False,
    isolation_level: Any = _UNSET,
) -> aiosqlite.Connection:
    """打开插件统一配置的 SQLite 连接。

    journal mode 会持久化在数据库文件里，只需在启动期对每个库设置一次
    （``apply_journal_mode=True``），之后新建连接继承即可。热路径连接额外执行
    ``PRAGMA journal_mode`` 会先读数据库头再写，遇到并发提交会立刻
    ``database is locked`` 且不进入 busy 等待，因此只设置 ``busy_timeout``。

    Args:
        db_path: 数据库文件路径。
        apply_journal_mode: 是否为该库写入进程级 journal mode（仅启动期）。
        isolation_level: 可选覆写 ``aiosqlite.connect`` 的事务隔离级别（如 ``None``）。

    Returns:
        已设置 ``busy_timeout`` 的连接。
    """
    # The SQLite worker may finish opening even after its caller is cancelled.
    connect_coro = (
        aiosqlite.connect(db_path)
        if isolation_level is _UNSET
        else aiosqlite.connect(db_path, isolation_level=isolation_level)
    )
    connect_task = asyncio.ensure_future(connect_coro)
    try:
        db = await asyncio.shield(connect_task)
        cursor = await db.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        await cursor.close()
        # WAL + NORMAL is durable enough for an embedded plugin store and avoids
        # an fsync per commit; rollback journal keeps FULL to stay crash-safe.
        cursor = await db.execute(
            "PRAGMA synchronous = NORMAL"
            if _journal_mode == JOURNAL_MODE_WAL
            else "PRAGMA synchronous = FULL"
        )
        await cursor.close()
        cursor = await db.execute("PRAGMA temp_store = MEMORY")
        await cursor.close()
        if apply_journal_mode:
            # Enable incremental auto_vacuum on fresh DBs (and prepare existing DBs
            # for their next VACUUM) so PRAGMA incremental_vacuum actually reclaims pages.
            av_cursor = await db.execute("PRAGMA auto_vacuum")
            av_row = await av_cursor.fetchone()
            await av_cursor.close()
            if av_row is None or int(av_row[0]) != 2:
                av_set = await db.execute("PRAGMA auto_vacuum = INCREMENTAL")
                await av_set.fetchall()
                await av_set.close()
            # This PRAGMA returns a row. Until its cursor is finalized it can
            # retain the mode-change lock and block the first separate writer.
            cursor = await db.execute(f"PRAGMA journal_mode = {_journal_mode}")
            await cursor.close()
        return db
    except BaseException:
        # SQLite closes after any queued PRAGMA finishes. Even a second
        # cancellation must leave the worker enough time to finish closing.
        cleanup = asyncio.create_task(_close_unreturned_connection(connect_task))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        raise


async def resolve_journal_mode(data_dir: Path) -> str:
    """确定数据目录可用的 journal mode，必要时把历史 WAL 库转回。

    必须在建立任何 SQLite 连接之前调用（其余连接都通过
    :func:`open_connection` 复用这里确定的策略）。

    Args:
        data_dir: 存放插件数据库的数据目录。

    Returns:
        ``WAL`` 或 ``DELETE``。

    Raises:
        RuntimeError: 判定不支持 WAL，但已有 WAL 库无法转回 rollback journal。
    """
    global _journal_mode, _resolved_dir

    data_dir = Path(data_dir)
    cache_key = str(data_dir.resolve())
    if cache_key == _resolved_dir:
        return _journal_mode

    data_dir.mkdir(parents=True, exist_ok=True)
    mount_type = _mount_fstype(data_dir)
    if mount_type is not None and _is_unsafe_fstype(mount_type):
        detail = f"挂载类型 {mount_type}"
        wal_supported = False
    else:
        detail = "WAL 探测失败"
        wal_supported = await asyncio.to_thread(_probe_wal_support, data_dir)

    if wal_supported:
        _journal_mode = JOURNAL_MODE_WAL
        logger.info(f"SQLite WAL 可用，journal mode = WAL: {data_dir}")
    else:
        for db_name in _PLUGIN_DB_FILES:
            db_path = data_dir / db_name
            if not db_path.exists():
                continue
            try:
                await asyncio.to_thread(_downgrade_to_rollback, db_path)
            except sqlite3.Error as exc:
                raise RuntimeError(
                    "数据目录不支持 SQLite WAL，且无法把已有数据库转回 "
                    f"rollback journal: {db_path} ({exc})。"
                    "请把插件数据目录迁移到 Docker 命名卷或本地磁盘后重试。"
                ) from exc
        _journal_mode = JOURNAL_MODE_DELETE
        logger.warning(
            f"数据目录不支持 SQLite WAL（{detail}），已回退 rollback journal: "
            f"{data_dir}。WAL 需要 -shm 共享内存与文件锁，Docker Desktop "
            "bind mount、网络盘等共享文件系统不保证该语义；把数据目录迁移到 "
            "Docker 命名卷或本地磁盘后会自动恢复 WAL。"
        )

    _resolved_dir = cache_key
    return _journal_mode


def _probe_wal_support(data_dir: Path) -> bool:
    """用同目录临时库验证 WAL 与多连接写入是否可用。"""
    probe_path = data_dir / _PROBE_DB_NAME
    _remove_db_files(probe_path)
    connections: list[sqlite3.Connection] = []
    try:
        first = sqlite3.connect(str(probe_path), timeout=10, isolation_level=None)
        connections.append(first)
        mode_row = first.execute("PRAGMA journal_mode = WAL").fetchone()
        if mode_row is None or str(mode_row[0]).lower() != "wal":
            return False
        first.execute("CREATE TABLE probe (value INTEGER)")
        first.execute("INSERT INTO probe VALUES (1)")
        # 第二条连接加入 WAL 并写入，触发 -shm 共享内存与文件锁路径。
        second = sqlite3.connect(str(probe_path), timeout=10, isolation_level=None)
        connections.append(second)
        second.execute("BEGIN IMMEDIATE")
        second.execute("INSERT INTO probe VALUES (2)")
        second.execute("COMMIT")
        second.execute("SELECT COUNT(*) FROM probe").fetchone()
        return True
    except sqlite3.Error as exc:
        logger.debug(f"SQLite WAL 探测失败: {exc}")
        return False
    finally:
        for connection in connections:
            connection.close()
        _remove_db_files(probe_path)


def _downgrade_to_rollback(db_path: Path) -> None:
    """把已有 WAL 库转回 rollback journal，保留全部数据。

    WAL 库在不支持 ``-shm`` 的文件系统上无法用普通方式打开；
    ``locking_mode = EXCLUSIVE`` 让 WAL 索引留在堆内存，从而能 checkpoint
    并切回 rollback journal。
    """
    connection = sqlite3.connect(str(db_path), timeout=10, isolation_level=None)
    try:
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA locking_mode = EXCLUSIVE")
        mode_row = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        if mode_row is None or str(mode_row[0]).lower() != "delete":
            raise sqlite3.OperationalError(f"journal mode 仍为 {mode_row}")
    finally:
        connection.close()


def _mount_fstype(path: Path) -> str | None:
    """返回 ``path`` 所在挂载点的文件系统类型；读不到挂载表时返回 None。"""
    try:
        with open("/proc/self/mounts", encoding="utf-8", errors="replace") as mounts:
            lines = mounts.readlines()
    except OSError:
        return None

    target = str(path.resolve())
    best_mount = ""
    best_type: str | None = None
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point = _unescape_mount_field(fields[1])
        if not _mount_contains(mount_point, target):
            continue
        if len(mount_point) >= len(best_mount):
            best_mount = mount_point
            best_type = fields[2]
    return best_type


def _mount_contains(mount_point: str, target: str) -> bool:
    if mount_point == "/":
        return target.startswith("/")
    return target == mount_point or target.startswith(mount_point + "/")


def _unescape_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _is_unsafe_fstype(fstype: str) -> bool:
    if fstype in _UNSAFE_FS_TYPES:
        return True
    return fstype.startswith("fuse.")


def _remove_db_files(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            Path(f"{db_path}{suffix}").unlink()
        except FileNotFoundError:
            continue
