import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from astrbot_plugin_dynamics_memory.storage.sqlite_write import write_transaction
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.tests.test_memory_engine import _FakeFaissDB
from astrbot_plugin_dynamics_memory.tests.test_narrative_api import SCOPE, envelope


@pytest.mark.asyncio
async def test_cross_connection_vector_delete_never_holds_write_transaction(tmp_path):
    engine = MemoryEngine(
        str(tmp_path / "memory.db"), _FakeFaissDB(), config={"fallback_enabled": True}
    )
    await engine.initialize()
    try:
        await engine.db_connection.execute(
            "INSERT INTO documents(id,doc_id,text,metadata) VALUES(1,'uuid-1','summary','{}')"
        )
        await engine.db_connection.commit()
        transactions = []

        async def delete(uuid):
            transactions.append(engine.db_connection.in_transaction)
            async with aiosqlite.connect(engine.db_path, timeout=0.1) as second:
                await second.execute("DELETE FROM documents WHERE doc_id=?", (uuid,))
                await second.commit()
            return True

        engine.faiss_db.delete = delete
        assert (
            await asyncio.wait_for(engine._delete_document_indexes_for_batch([1]), 2)
            == 1
        )
        assert transactions == [False]
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_vector_failure_retains_document_and_is_not_success(tmp_path):
    engine = MemoryEngine(
        str(tmp_path / "memory.db"), _FakeFaissDB(), config={"fallback_enabled": True}
    )
    await engine.initialize()
    try:
        await engine.db_connection.execute(
            "INSERT INTO documents(id,doc_id,text,metadata) VALUES(1,'uuid-1','summary','{}')"
        )
        await engine.db_connection.commit()
        engine.faiss_db.delete = AsyncMock(return_value=False)
        with pytest.raises(RuntimeError, match="not confirmed"):
            await engine._delete_document_indexes_for_batch([1])
        engine.faiss_db.delete.assert_awaited_once_with("uuid-1")
        assert await (
            await engine.db_connection.execute("SELECT 1 FROM documents WHERE id=1")
        ).fetchone()
        assert not engine.db_connection.in_transaction
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_transaction_cancellation_rolls_back_and_releases_writer(tmp_path):
    owner = SimpleNamespace(db_path=str(tmp_path / "test.db"))
    async with aiosqlite.connect(owner.db_path) as db:
        await db.execute("CREATE TABLE sample(id INTEGER)")
        await db.commit()
    started = asyncio.Event()

    async def pending():
        async with write_transaction(owner) as db:
            await db.execute("INSERT INTO sample VALUES(1)")
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(pending())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with write_transaction(owner) as db:
        await db.execute("INSERT INTO sample VALUES(2)")
    async with aiosqlite.connect(owner.db_path) as db:
        assert await (await db.execute("SELECT id FROM sample")).fetchall() == [(2,)]


@pytest.mark.asyncio
async def test_concurrent_writers_commit_independently(tmp_path):
    owner = SimpleNamespace(db_path=str(tmp_path / "test.db"))
    async with aiosqlite.connect(owner.db_path) as db:
        await db.execute("CREATE TABLE sample(id INTEGER UNIQUE)")
        await db.commit()

    async def add(n):
        async with write_transaction(owner) as db:
            await db.execute("INSERT INTO sample VALUES(?)", (n,))

    await asyncio.gather(*(add(i) for i in range(30)))
    async with aiosqlite.connect(owner.db_path) as db:
        assert (await (await db.execute("SELECT COUNT(*) FROM sample")).fetchone())[
            0
        ] == 30


@pytest.mark.asyncio
async def test_journal_failure_stops_before_index_mutation(tmp_path):
    engine = MemoryEngine(
        str(tmp_path / "memory.db"), _FakeFaissDB(), config={"fallback_enabled": True}
    )
    await engine.initialize()
    try:
        await engine.db_connection.execute("DROP TABLE memory_write_ops")
        engine.faiss_db.insert = AsyncMock()
        with pytest.raises(aiosqlite.OperationalError):
            await engine.add_memory("summary")
        engine.faiss_db.insert.assert_not_awaited()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_source_recovery_is_exact_scope_typed_and_idempotent(tmp_path):
    store = ConversationStore(str(tmp_path / "conversations.db"))
    await store.initialize()
    engine = MemoryEngine(
        str(tmp_path / "memory.db"), _FakeFaissDB(), config={"fallback_enabled": True}
    )
    await engine.initialize()
    engine.source_loader = store.load_narrative_sources
    evidence = [{"source_id": "a", "scope": dict(SCOPE), "fact_kind": "observation"}]
    metadata = {
        "narrative_provenance": evidence,
        "source_window": {"session_id": scope_key(SCOPE)},
        "has_source": False,
        "fact_kind": "subjective",
    }
    try:
        await store.ingest_narrative_event(envelope())
        await engine.db_connection.execute(
            "INSERT INTO documents(id,doc_id,text,metadata) VALUES(1,?,?,?)",
            ("uuid-1", "summary", json.dumps(metadata)),
        )
        await engine.db_connection.commit()
        result = await engine.repair_retained_sources()
        assert result["repaired"] == 1
        source = await engine.get_memory_source(1)
        assert source[0]["content"] == envelope()["text"]
        assert source[0]["metadata"]["fact_kind"] == "observation"
        assert isinstance(source[0]["timestamp"], float)
        assert (await engine.repair_retained_sources())["repaired"] == 0
        altered = {
            **metadata,
            "narrative_provenance": [
                {**evidence[0], "scope": {**SCOPE, "umo": "qq:GroupMessage:2"}}
            ],
        }
        assert await store.load_narrative_sources(altered) is None
        altered = {
            **metadata,
            "narrative_provenance": [{**evidence[0], "fact_kind": "receipt"}],
        }
        assert await store.load_narrative_sources(altered) is None
    finally:
        await engine.close()
        await store.close()


@pytest.mark.asyncio
async def test_only_known_sqlite_failures_are_requeued(tmp_path):
    engine = MemoryEngine(
        str(tmp_path / "memory.db"), _FakeFaissDB(), config={"fallback_enabled": True}
    )
    await engine.initialize()
    try:
        for reason in [
            "database is locked",
            "source document missing",
            "PROHIBITED_CONTENT",
        ]:
            op = await engine._start_write_op("add", {}, memory_id=1)
            await engine.db_connection.execute(
                "UPDATE memory_write_ops SET status='failed',error=? WHERE id=?",
                (reason, op),
            )
            await engine.db_connection.commit()
        assert await engine.requeue_sqlite_failures() == 1
        rows = await (
            await engine.db_connection.execute(
                "SELECT status FROM memory_write_ops ORDER BY id"
            )
        ).fetchall()
        assert [r[0] for r in rows] == ["needs_repair", "failed", "failed"]
    finally:
        await engine.close()
