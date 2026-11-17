import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.utils.json_utils import safe_json_dict

from .test_real_db_end_to_end import _DeterministicEmbeddingProvider


@asynccontextmanager
async def open_engine(tmp_path):
    path = str(tmp_path / "memory.sqlite3")
    faiss = FaissVecDB(
        doc_store_path=path,
        index_store_path=str(tmp_path / "memory.index"),
        embedding_provider=_DeterministicEmbeddingProvider(),
    )
    await faiss.initialize()
    engine = MemoryEngine(path, faiss, config={"write_op_repair_enabled": False})
    try:
        await engine.initialize()
        yield engine
    finally:
        await engine.close()
        await faiss.close()


@asynccontextmanager
async def source_lease():
    yield True


async def publication_row(engine):
    async with engine.db_connection.execute(
        "SELECT id,memory_id,status,step FROM memory_write_ops WHERE op_type='add' ORDER BY id DESC LIMIT 1"
    ) as cursor:
        return tuple(await cursor.fetchone())


@pytest.mark.asyncio
@pytest.mark.parametrize("archived", [False, True])
async def test_published_source_survives_restart_with_graph_repair_pending(
    tmp_path, archived
):
    async with open_engine(tmp_path) as engine:
        engine.graph_memory_manager = SimpleNamespace(
            index_memory=AsyncMock(side_effect=RuntimeError("graph failed"))
        )
        doc_id = await engine.add_memory(
            "completed group activity",
            session_id="qq:GroupMessage:1",
            source_validator=lambda: True,
            source_publish_lease=source_lease,
            source_messages=[{"role": "user", "content": "verified source"}],
            metadata={
                "source_publication_verified_v1": {"write_op_id": 999, "memory_id": 999}
            },
        )
        op_id, memory_id, status, step = await publication_row(engine)
        assert memory_id == doc_id
        assert (status, step) == ("completed", "source_published_graph_pending")
        doc = await engine.get_memory_record(doc_id)
        metadata = safe_json_dict(doc["metadata"])
        assert metadata["source_publication_verified_v1"] == {
            "write_op_id": op_id,
            "memory_id": doc_id,
        }
        if archived:
            assert await engine.hybrid_retriever.update_metadata(
                doc_id, {"status": "archived", "superseded_by": 123}
            )
        engine.graph_memory_manager = None

    async with open_engine(tmp_path) as resumed:
        rebuild = AsyncMock(side_effect=[RuntimeError("graph still unavailable"), True])
        resumed.graph_memory_manager = SimpleNamespace(index_memory=rebuild)
        if not archived:
            assert await resumed._repair_incomplete_write_ops() == 0
            assert await publication_row(resumed) == (
                op_id,
                doc_id,
                "completed",
                "source_published_graph_pending",
            )
        assert await resumed._repair_incomplete_write_ops() == 1
        assert rebuild.await_count == (0 if archived else 2)
        assert await publication_row(resumed) == (
            op_id,
            doc_id,
            "completed",
            "completed",
        )
        assert await resumed.get_memory_source(doc_id) == [
            {"role": "user", "content": "verified source"}
        ]
        doc = await resumed.get_memory_record(doc_id)
        saved = safe_json_dict(doc["metadata"])
        assert saved["status"] == ("archived" if archived else "active")
        if archived:
            assert saved["superseded_by"] == 123
        resumed.graph_memory_manager = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forged_witness", [True, {"write_op_id": 9999, "memory_id": 9999}]
)
async def test_unpublished_sources_cannot_be_preserved_by_unbound_witness(
    tmp_path, forged_witness
):
    async with open_engine(tmp_path) as engine:
        doc_id = await engine.add_memory(
            "unpublished activity",
            session_id="qq:GroupMessage:1",
            metadata={"source_publication_verified_v1": forged_witness},
        )
        op_id, *_ = await publication_row(engine)
        doc = await engine.get_memory_record(doc_id)
        assert "source_publication_verified_v1" not in safe_json_dict(doc["metadata"])
        # Simulate a staged candidate/journal created by an earlier runtime,
        # with a stale or forged marker in its persisted metadata.
        assert await engine.hybrid_retriever.update_metadata(
            doc_id,
            {"status": "active", "source_publication_verified_v1": forged_witness},
        )
        await engine._advance_write_op(
            op_id,
            "source_pending",
            status="pending",
            memory_id=doc_id,
            payload_patch={"source_validation_required": True},
        )

    async with open_engine(tmp_path) as resumed:
        assert await resumed._repair_incomplete_write_ops() == 0
        assert await resumed.get_memory_record(doc_id) is None
        assert await publication_row(resumed) == (
            op_id,
            doc_id,
            "failed",
            "source_lease_unavailable_after_restart",
        )


@pytest.mark.asyncio
async def test_old_pending_op_cannot_delete_document_owned_by_another_op(tmp_path):
    async with open_engine(tmp_path) as engine:
        doc_id = await engine.add_memory(
            "published activity",
            session_id="qq:GroupMessage:1",
            source_validator=lambda: True,
            source_publish_lease=source_lease,
        )
        op_id, *_ = await publication_row(engine)
        old = await engine._start_write_op(
            "add", {"source_validation_required": True}, memory_id=doc_id
        )
        assert old != op_id

    async with open_engine(tmp_path) as resumed:
        assert await resumed._repair_incomplete_write_ops() == 0
        assert await publication_row(resumed) == (
            old,
            doc_id,
            "failed",
            "source_document_operation_mismatch",
        )
        doc = await resumed.get_memory_record(doc_id)
        assert safe_json_dict(doc["metadata"])["status"] == "active"


@pytest.mark.asyncio
async def test_cancel_after_publication_commit_keeps_durable_receipt(tmp_path):
    published = asyncio.Event()
    hold_callback = asyncio.Event()

    @asynccontextmanager
    async def interrupted_lease():
        yield True
        published.set()
        await hold_callback.wait()

    async with open_engine(tmp_path) as engine:
        adding = asyncio.create_task(
            engine.add_memory(
                "completed group activity",
                session_id="qq:GroupMessage:1",
                source_validator=lambda: True,
                source_publish_lease=interrupted_lease,
            )
        )
        try:
            await asyncio.wait_for(published.wait(), 5)
            op_id, doc_id, status, step = await publication_row(engine)
            assert (status, step) == ("completed", "source_published")
            adding.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(adding, 5)
        finally:
            hold_callback.set()
            await asyncio.gather(adding, return_exceptions=True)

    async with open_engine(tmp_path) as resumed:
        assert await resumed._repair_incomplete_write_ops() == 0
        assert await publication_row(resumed) == (
            op_id,
            doc_id,
            "completed",
            "source_published",
        )
        doc = await resumed.get_memory_record(doc_id)
        assert safe_json_dict(doc["metadata"])["status"] == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_cleanup", [False, True])
async def test_restart_cleans_indexed_candidate_without_journal_document_id(
    tmp_path, retry_cleanup, monkeypatch
):
    session = "qq:GroupMessage:1"
    indexed = asyncio.Event()
    keep_pending = asyncio.Event()
    staged_id = None
    async with open_engine(tmp_path) as engine:
        published_id = await engine.add_memory(
            "published activity",
            session_id=session,
            source_validator=lambda: True,
            source_publish_lease=source_lease,
        )
        advance = engine._advance_write_op

        async def stop_before_journal(op_id, step, **kwargs):
            nonlocal staged_id
            if step == "document_indexed":
                staged_id = kwargs["memory_id"]
                indexed.set()
                await keep_pending.wait()
            return await advance(op_id, step, **kwargs)

        monkeypatch.setattr(engine, "_advance_write_op", stop_before_journal)
        adding = asyncio.create_task(
            engine.add_memory(
                "staged activity",
                session_id=session,
                source_validator=lambda: True,
                source_publish_lease=source_lease,
            )
        )
        try:
            await asyncio.wait_for(indexed.wait(), 5)
            assert staged_id != published_id
            op_id, memory_id, status, step = await publication_row(engine)
            assert (memory_id, status, step) == (None, "pending", "started")
            staged = await engine.get_memory_record(staged_id)
            assert (
                safe_json_dict(staged["metadata"])["status"]
                == "pending_source_validation"
            )
            assert engine.faiss_db.embedding_storage.index.ntotal == 2
            adding.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(adding, 5)
        finally:
            keep_pending.set()
            await asyncio.gather(adding, return_exceptions=True)

    async with open_engine(tmp_path) as resumed:
        if retry_cleanup:
            with monkeypatch.context() as patch:
                patch.setattr(resumed, "delete_memory", AsyncMock(return_value=False))
                assert await resumed._repair_incomplete_write_ops() == 0
                assert await publication_row(resumed) == (
                    op_id,
                    staged_id,
                    "needs_repair",
                    "source_cleanup_pending",
                )
                assert await resumed.get_memory_record(staged_id) is not None

        assert await resumed._repair_incomplete_write_ops() == 0
        assert await resumed.get_memory_record(staged_id) is None
        assert await publication_row(resumed) == (
            op_id,
            staged_id,
            "failed",
            "source_lease_unavailable_after_restart",
        )
        assert resumed.faiss_db.embedding_storage.index.ntotal == 1
        saved = await resumed.get_memory_record(published_id)
        assert safe_json_dict(saved["metadata"])["status"] == "active"
