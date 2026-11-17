"""Recovery of persisted operations across failures without replaying source leases."""

import json
from unittest.mock import AsyncMock

import pytest

from tests.integration.test_real_db_end_to_end import real_db_stack as real_db_stack


@pytest.fixture
def engine(real_db_stack):
    return real_db_stack["memory_engine"]


async def operation(engine, identifier):
    row = await (
        await engine.db_connection.execute(
            "SELECT status,step,payload,error FROM memory_write_ops WHERE id=?",
            (identifier,),
        )
    ).fetchone()
    return {
        "status": row[0],
        "step": row[1],
        "payload": json.loads(row[2]),
        "error": row[3],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["add", "delete"])
async def test_missing_document_identity_is_not_replayed(engine, kind):
    op = await engine._start_write_op(kind, {})
    assert await engine._repair_incomplete_write_ops() == 0
    assert (await operation(engine, op))["status"] == "failed"
    assert await engine._repair_incomplete_write_ops() == 0


@pytest.mark.asyncio
async def test_unknown_operation_does_not_starve_following_repair(engine):
    unknown = await engine._start_write_op("obsolete-write-format", {})
    valid = await engine._start_write_op("delete", {}, memory_id=999)
    assert await engine._repair_incomplete_write_ops() == 1
    assert (await operation(engine, unknown))["status"] == "pending"
    assert (await operation(engine, valid))["status"] == "completed"


@pytest.mark.asyncio
async def test_add_repair_restores_source_and_is_idempotent(engine):
    memory = await engine.add_memory(
        "coffee memory", session_id="fixture:GroupMessage:room", persona_id="p"
    )
    source = [
        {
            "id": 1,
            "role": "user",
            "content": "coffee source",
            "session_id": "fixture:GroupMessage:room",
            "sender_id": "u1",
            "timestamp": 1234,
            "metadata": {},
        }
    ]
    await engine.hybrid_retriever.update_metadata(memory, {"has_source": True})
    op = await engine._start_write_op(
        "add", {"source_messages": source, "atoms": []}, memory_id=memory
    )
    assert await engine._repair_incomplete_write_ops() == 1
    assert (await engine.get_memory_source(memory))[0]["content"] == "coffee source"
    assert (await operation(engine, op))["status"] == "completed"
    assert await engine._repair_incomplete_write_ops() == 0


@pytest.mark.asyncio
async def test_missing_source_stays_pending_and_cannot_become_completed(engine):
    memory = await engine.add_memory(
        "source required", session_id="fixture:GroupMessage:room", persona_id="p"
    )
    await engine.hybrid_retriever.update_metadata(memory, {"has_source": True})
    op = await engine._start_write_op("add", {}, memory_id=memory)
    assert await engine._repair_incomplete_write_ops() == 0
    row = await operation(engine, op)
    assert row["status"] == "needs_repair" and row["step"] == "source_pending"
    assert await engine.get_memory_source(memory) == []


@pytest.mark.asyncio
async def test_request_bound_source_lease_is_never_replayed_after_restart(engine):
    memory = await engine.add_memory(
        "guarded draft", session_id="fixture:FriendMessage:u1", persona_id="p"
    )
    op = await engine._start_write_op(
        "add", {"source_validation_required": True}, memory_id=memory
    )
    await engine.hybrid_retriever.update_metadata(
        memory, {"status": "pending_source_validation", "write_op_id": op}
    )
    assert await engine._repair_incomplete_write_ops() == 0
    row = await operation(engine, op)
    assert row["status"] == "failed"
    assert row["step"] == "source_lease_unavailable_after_restart"
    assert await engine.get_memory(memory) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [None, {}, "1", [], ["invalid"]])
async def test_invalid_batch_recovery_never_deletes_documents(engine, ids):
    memory = await engine.add_memory(
        "keep this source", session_id="fixture:GroupMessage:room", persona_id="p"
    )
    op = await engine._start_write_op("batch_delete", {"memory_ids": ids})
    assert await engine._repair_incomplete_write_ops() == 0
    assert (await operation(engine, op))["status"] == "failed"
    assert await engine.get_memory(memory) is not None


@pytest.mark.asyncio
async def test_batch_recovery_deletes_ids_and_cleans_sources(engine):
    first = await engine.add_memory(
        "one source", session_id="fixture:GroupMessage:room", persona_id="p"
    )
    second = await engine.add_memory(
        "two source", session_id="fixture:GroupMessage:room", persona_id="p"
    )
    op = await engine._start_write_op(
        "batch_delete", {"memory_ids": [first, str(first), second]}
    )
    assert await engine._repair_incomplete_write_ops() == 1
    assert (
        await engine.get_memory(first) is None
        and await engine.get_memory(second) is None
    )
    row = await operation(engine, op)
    assert row["status"] == "completed" and row["payload"]["deleted_count"] == 3


@pytest.mark.asyncio
async def test_transient_repair_failure_remains_durable(engine):
    op = await engine._start_write_op("delete", {}, memory_id=99)
    original = engine._repair_delete_write_op
    engine._repair_delete_write_op = AsyncMock(
        side_effect=RuntimeError("fixture transient failure")
    )
    assert await engine._repair_incomplete_write_ops() == 0
    assert (await operation(engine, op))["status"] == "needs_repair"
    engine._repair_delete_write_op = original
    assert await engine._repair_incomplete_write_ops() == 1
    assert (await operation(engine, op))["status"] == "completed"
