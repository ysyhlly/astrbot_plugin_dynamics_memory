"""Exercise full staged publication with real FAISS, FTS and SQLite stores."""

import pytest

from .test_guarded_publication_recovery import open_engine, source_lease
from astrbot_plugin_dynamics_memory.core.utils.json_utils import safe_json_dict


@pytest.mark.asyncio
async def test_source_publication_survives_a_concurrent_old_reader(tmp_path):
    async with open_engine(tmp_path) as engine:
        await engine._start_write_op("add", {"sample": 1})
        await engine._start_write_op("add", {"sample": 2})
        cursor = await engine.db_connection.execute(
            "SELECT id FROM memory_write_ops ORDER BY id"
        )
        try:
            assert await cursor.fetchone() is not None
            memory_id = await engine.add_memory(
                "verified offline summary",
                session_id="qq:FriendMessage:1",
                source_validator=lambda: True,
                source_publish_lease=source_lease,
                source_messages=[
                    {"role": "user", "content": "retained offline source"}
                ],
            )
        finally:
            await cursor.close()
        record = await engine.get_memory_record(memory_id)
        metadata = safe_json_dict(record["metadata"])
        assert metadata["status"] == "active"
        op = metadata["write_op_id"]
        assert metadata["source_publication_verified_v1"] == {
            "write_op_id": op,
            "memory_id": memory_id,
        }
        async with engine.db_connection.execute(
            "SELECT status,step,memory_id FROM memory_write_ops WHERE id=?", (op,)
        ) as result:
            assert tuple(await result.fetchone()) == (
                "completed",
                "source_published",
                memory_id,
            )
        assert await engine.get_memory_source(memory_id) == [
            {"role": "user", "content": "retained offline source"}
        ]
