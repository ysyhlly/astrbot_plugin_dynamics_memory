"""SQLite journal 策略：WAL 能力探测、回退与连接配置。"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from astrbot_plugin_dynamics_memory.storage import sqlite_connection


@pytest.fixture(autouse=True)
def _reset_journal_policy():
    mode = sqlite_connection._journal_mode
    resolved = sqlite_connection._resolved_dir
    yield
    sqlite_connection._journal_mode = mode
    sqlite_connection._resolved_dir = resolved


def test_mount_path_matching_is_component_aware():
    assert sqlite_connection._mount_contains("/", "/a/b")
    assert sqlite_connection._mount_contains("/a", "/a/b")
    assert not sqlite_connection._mount_contains("/a", "/ab")
    assert (
        sqlite_connection._unescape_mount_field("/mnt/with\040space")
        == "/mnt/with space"
    )


def test_unsafe_filesystem_types_are_rejected():
    assert sqlite_connection._is_unsafe_fstype("9p")
    assert sqlite_connection._is_unsafe_fstype("fuse.grpcfuse")
    assert sqlite_connection._is_unsafe_fstype("nfs4")

    assert not sqlite_connection._is_unsafe_fstype("ext4")
    assert not sqlite_connection._is_unsafe_fstype("overlay")
    assert not sqlite_connection._is_unsafe_fstype("fuseblk")


@pytest.mark.asyncio
async def test_resolve_keeps_wal_when_directory_supports_it(tmp_path):
    mode = await sqlite_connection.resolve_journal_mode(tmp_path)

    assert mode == sqlite_connection.JOURNAL_MODE_WAL
    for suffix in ("", "-wal", "-shm", "-journal"):
        assert not (tmp_path / f".wal_probe.db{suffix}").exists()


@pytest.mark.asyncio
async def test_resolve_falls_back_on_shared_mount(tmp_path, monkeypatch):
    """Docker Desktop 的 9p bind mount 不能承载 WAL，探测通过也不采用。"""
    monkeypatch.setattr(sqlite_connection, "_mount_fstype", lambda _path: "9p")
    monkeypatch.setattr(sqlite_connection, "_probe_wal_support", lambda _dir: True)

    mode = await sqlite_connection.resolve_journal_mode(tmp_path)

    assert mode == sqlite_connection.JOURNAL_MODE_DELETE


@pytest.mark.asyncio
async def test_resolve_converts_existing_wal_database(tmp_path, monkeypatch):
    db_path = tmp_path / "livingmemory.db"
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
    connection.execute("INSERT INTO documents VALUES (1)")
    connection.commit()
    connection.close()
    monkeypatch.setattr(sqlite_connection, "_mount_fstype", lambda _path: None)
    monkeypatch.setattr(sqlite_connection, "_probe_wal_support", lambda _dir: False)

    mode = await sqlite_connection.resolve_journal_mode(tmp_path)

    assert mode == sqlite_connection.JOURNAL_MODE_DELETE
    connection = sqlite3.connect(db_path)
    try:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        connection.close()
    assert journal_mode == "delete"
    assert count == 1


@pytest.mark.asyncio
async def test_resolve_raises_when_wal_database_cannot_be_converted(
    tmp_path, monkeypatch
):
    (tmp_path / "livingmemory.db").write_bytes(b"not a sqlite database")
    monkeypatch.setattr(sqlite_connection, "_mount_fstype", lambda _path: None)
    monkeypatch.setattr(sqlite_connection, "_probe_wal_support", lambda _dir: False)

    with pytest.raises(RuntimeError, match="rollback journal"):
        await sqlite_connection.resolve_journal_mode(tmp_path)


@pytest.mark.asyncio
async def test_open_connection_keeps_hot_path_free_of_journal_mode(tmp_path):
    connection = await sqlite_connection.open_connection(str(tmp_path / "memory.db"))
    try:
        cursor = await connection.execute("PRAGMA journal_mode")
        journal_mode = (await cursor.fetchone())[0]
        cursor = await connection.execute("PRAGMA busy_timeout")
        busy_timeout = (await cursor.fetchone())[0]
    finally:
        await connection.close()

    # 新建库保持 SQLite 默认 rollback journal；journal mode 只在启动期设置。
    assert journal_mode == "delete"
    assert busy_timeout == sqlite_connection.BUSY_TIMEOUT_MS


@pytest.mark.asyncio
async def test_open_connection_applies_journal_mode_when_requested(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sqlite_connection, "_journal_mode", sqlite_connection.JOURNAL_MODE_DELETE
    )

    connection = await sqlite_connection.open_connection(
        str(tmp_path / "memory.db"), apply_journal_mode=True
    )
    try:
        cursor = await connection.execute("PRAGMA journal_mode")
        journal_mode = (await cursor.fetchone())[0]
    finally:
        await connection.close()

    assert journal_mode == "delete"


@pytest.mark.asyncio
async def test_open_connection_sets_wal_safe_pragmas(tmp_path):
    """WAL 下使用 NORMAL 同步级别、FILE 临时表与受控页缓存/零 mmap。"""
    sqlite_connection._journal_mode = sqlite_connection.JOURNAL_MODE_WAL
    connection = await sqlite_connection.open_connection(str(tmp_path / "pragmas.db"))
    try:
        cursor = await connection.execute("PRAGMA synchronous")
        assert (await cursor.fetchone())[0] == 1  # NORMAL
        cursor = await connection.execute("PRAGMA temp_store")
        assert (await cursor.fetchone())[0] == 1  # FILE
        cursor = await connection.execute("PRAGMA mmap_size")
        assert (await cursor.fetchone())[0] == 0
        cursor = await connection.execute("PRAGMA cache_size")
        assert (await cursor.fetchone())[0] < 0
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["connect", "pragma"])
async def test_cancelled_open_waits_for_sqlite_close(tmp_path, monkeypatch, stage):
    started = asyncio.Event()
    release_connect = asyncio.Event()
    closing = asyncio.Event()
    release_close = asyncio.Event()
    closed = asyncio.Event()
    original_connect = sqlite_connection.aiosqlite.connect
    real_close = None

    async def connect(db_path):
        nonlocal real_close
        db = await original_connect(db_path)
        real_close = db.close

        async def close():
            closing.set()
            await release_close.wait()
            await real_close()
            closed.set()

        db.close = close
        if stage == "connect":
            started.set()
            await release_connect.wait()
        else:

            async def block_pragma(_statement):
                started.set()
                await asyncio.Future()

            db.execute = block_pragma
        return db

    monkeypatch.setattr(sqlite_connection.aiosqlite, "connect", connect)
    opening = asyncio.create_task(
        sqlite_connection.open_connection(str(tmp_path / "cancelled.db"))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        opening.cancel()
        await asyncio.sleep(0)
        assert not opening.done()
        release_connect.set()
        await asyncio.wait_for(closing.wait(), timeout=2)
        for _ in range(2):
            opening.cancel()
            await asyncio.sleep(0)
            assert not opening.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(opening, timeout=2)
        assert closed.is_set()
    finally:
        release_connect.set()
        release_close.set()
        await asyncio.gather(opening, return_exceptions=True)
        if real_close is not None and not closed.is_set():
            await real_close()


@pytest.mark.asyncio
async def test_failed_pragma_closes_sqlite_connection(tmp_path, monkeypatch):
    original_connect = sqlite_connection.aiosqlite.connect
    db = None

    async def connect(db_path):
        nonlocal db
        db = await original_connect(db_path)
        execute = db.execute

        async def bad_pragma(_statement):
            await execute("PRAGMA =")

        db.execute = bad_pragma
        return db

    monkeypatch.setattr(sqlite_connection.aiosqlite, "connect", connect)
    try:
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            await sqlite_connection.open_connection(str(tmp_path / "invalid.db"))
        with pytest.raises(ValueError, match="closed|no active connection"):
            await db.execute_fetchall("SELECT 1")
    finally:
        if db is not None:
            await db.close()
