"""A failed shadow index rebuild must preserve the live searchable index."""

from unittest.mock import AsyncMock

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.core.validators.index_maintenance import (
    IndexValidator,
)
from tests.integration.test_real_db_end_to_end import real_db_stack as real_db_stack


@pytest.mark.asyncio
@pytest.mark.parametrize("row_failure", [False, True])
async def test_shadow_insert_retries_rows_and_keeps_old_index_if_failure_ratio_exceeded(
    real_db_stack, monkeypatch, row_failure
):
    engine = real_db_stack["memory_engine"]
    ids = [
        await engine.add_memory(f"coffee item {i}", session_id="s", persona_id="p")
        for i in range(3)
    ]
    validator = IndexValidator(real_db_stack["memory_db_path"], engine.faiss_db)
    execute = aiosqlite.Connection.execute
    executemany = aiosqlite.Connection.executemany
    tables = []
    create = validator._create_bm25_shadow

    async def create_shadow(name):
        table = await create(name)
        tables.append(table)
        return table

    monkeypatch.setattr(validator, "_create_bm25_shadow", create_shadow)

    def batch(db, sql, parameters):
        if tables and sql.startswith("INSERT INTO " + tables[0]):

            async def fail():
                raise aiosqlite.OperationalError("simulated batch failure")

            return fail()
        return executemany(db, sql, parameters)

    def single(db, sql, parameters=None):
        if (
            row_failure
            and tables
            and sql.startswith("INSERT INTO " + tables[0])
            and parameters[0] == ids[1]
        ):

            async def fail():
                raise aiosqlite.OperationalError("simulated row failure")

            return fail()
        return execute(db, sql, parameters)

    monkeypatch.setattr(aiosqlite.Connection, "executemany", batch)
    monkeypatch.setattr(aiosqlite.Connection, "execute", single)
    progress = AsyncMock()
    result = await validator._rebuild_bm25_index(
        engine, 3, {"batch_size": 3, "max_failure_ratio": 0}, progress
    )
    assert result["switched"] is (not row_failure)
    assert result["errors"] == int(row_failure)
    assert result["processed"] == 3 - int(row_failure)
    progress.assert_awaited_once()
    # Even failure of one rebuilt row cannot replace the working live index.
    rows = await engine.bm25_retriever.search("coffee", 10)
    assert {r.doc_id for r in rows} == set(ids)
    async with aiosqlite.connect(real_db_stack["memory_db_path"]) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE name=?", (tables[0],)
        )
        assert await cursor.fetchone() is None
