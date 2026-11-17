"""Exercise admin edits through real Quart requests and SQLite/FAISS storage."""

from unittest.mock import AsyncMock
import pytest
from quart import Quart
from astrbot_plugin_dynamics_memory.core.page_api_modules.memory_handler import (
    MemoryHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import PageApiUtils


async def edit(engine, payload, method="update_memory"):
    app = Quart(__name__)
    handler = MemoryHandler(PageApiUtils())
    async with app.test_request_context("/edit", method="POST", json=payload):
        return await getattr(handler, method)(engine)


async def seed(engine):
    return await engine.add_memory(
        "coffee original",
        session_id="fixture:GroupMessage:room",
        persona_id="reader",
        importance=0.5,
        metadata={"topics": ["coffee"], "key_facts": ["likes coffee"]},
    )


@pytest.mark.asyncio
async def test_structured_edit_rebuilds_all_fields_and_retains_history(real_db_stack):
    engine = real_db_stack["memory_engine"]
    old = await seed(engine)
    result = await edit(
        engine,
        {
            "memory_id": old,
            "field": "structured",
            "reason": "user correction",
            "value": {
                "content": "tea corrected",
                "topics": ["tea", "tea", " "],
                "key_facts": ["likes tea"],
                "importance": 9,
                "status": "active",
                "type": "PREFERENCE",
            },
        },
    )
    assert result["status"] == "ok", result
    new = result["data"]["new_memory_id"]
    assert new != old and await engine.get_memory(old) is None
    row = await engine.get_memory(new)
    meta = PageApiUtils.normalize_metadata(row["metadata"])
    assert row["text"] == "tea corrected" and meta["importance"] == 0.9
    assert meta["topics"] == ["tea"] and meta["key_facts"] == ["likes tea"]
    assert (
        meta["update_reason"] == "user correction" and len(meta["update_history"]) == 5
    )
    assert (
        meta["previous_content"] == "coffee original"
        and meta["canonical_summary"] == "tea corrected"
    )
    same = await edit(
        engine, {"memory_id": new, "field": "content", "value": "tea corrected"}
    )
    assert same["data"]["new_memory_id"] == new


@pytest.mark.asyncio
async def test_batch_archive_restore_updates_actual_search_index(real_db_stack):
    engine = real_db_stack["memory_engine"]
    mid = await seed(engine)
    for status in ("archived", "active"):
        result = await edit(
            engine,
            {"memory_ids": [mid, "bad", 999999], "field": "status", "value": status},
            "batch_update_memories",
        )
        assert result["data"] == {
            "updated_count": 1,
            "failed_count": 2,
            "total": 3,
            "failed_ids": ["bad", 999999],
        }
        record = await engine.get_memory_record(mid)
        assert PageApiUtils.normalize_metadata(record["metadata"])["status"] == status
        hits = await engine.search_memories(
            "coffee",
            session_id="fixture:GroupMessage:room",
            persona_id="reader",
            touch_access=False,
        )
        assert (mid in {r.doc_id for r in hits}) is (status == "active")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value,expected", [("importance", 8, 0.8), ("type", "FACT", "FACT")]
)
async def test_batch_scalar_edits_persist(real_db_stack, field, value, expected):
    engine = real_db_stack["memory_engine"]
    mid = await seed(engine)
    result = await edit(
        engine,
        {"memory_ids": [str(mid)], "field": field, "value": value},
        "batch_update_memories",
    )
    assert result["data"]["updated_count"] == 1
    meta = PageApiUtils.normalize_metadata((await engine.get_memory(mid))["metadata"])
    assert meta["importance" if field == "importance" else "memory_type"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("structured", []),
        ("structured", {"content": ""}),
        ("structured", {"topics": "bad"}),
        ("structured", {"importance": 11}),
        ("structured", {"status": "invalid"}),
        ("structured", {"type": ""}),
        ("importance", "bad"),
        ("status", "invalid"),
        ("type", ""),
        ("unknown", "anything"),
    ],
)
async def test_invalid_edit_does_not_change_original(real_db_stack, field, value):
    engine = real_db_stack["memory_engine"]
    mid = await seed(engine)
    before = await engine.get_memory(mid)
    result = await edit(engine, {"memory_id": mid, "field": field, "value": value})
    assert result["status"] == "error"
    assert await engine.get_memory(mid) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", [True, False, 1.5, -1, 0, "1.0", None])
async def test_bad_id_never_targets_another_row(real_db_stack, bad_id):
    engine = real_db_stack["memory_engine"]
    mid = await seed(engine)
    before = await engine.get_memory(mid)
    assert (
        await edit(engine, {"memory_id": bad_id, "field": "type", "value": "WRONG"})
    )["status"] == "error"
    for method in ("batch_delete_memories", "batch_update_memories"):
        result = await edit(
            engine,
            {"memory_ids": [bad_id], "field": "status", "value": "deleted"},
            method,
        )
        assert result["data"]["failed_ids"] == [bad_id]
    assert await engine.get_memory(mid) == before


@pytest.mark.asyncio
async def test_backend_failure_reports_failure_and_preserves_row(
    real_db_stack, monkeypatch
):
    engine = real_db_stack["memory_engine"]
    mid = await seed(engine)
    before = await engine.get_memory(mid)
    monkeypatch.setattr(
        engine, "replace_memory", AsyncMock(side_effect=OSError("disk full"))
    )
    assert (await edit(engine, {"memory_id": mid, "field": "content", "value": "new"}))[
        "status"
    ] == "error"
    for outcome in (False, OSError("disk full")):
        mock = (
            AsyncMock(side_effect=outcome)
            if isinstance(outcome, Exception)
            else AsyncMock(return_value=outcome)
        )
        monkeypatch.setattr(engine, "update_memory", mock)
        assert (
            await edit(engine, {"memory_id": mid, "field": "importance", "value": 0.8})
        )["status"] == "error"
        batch = await edit(
            engine,
            {"memory_ids": [mid], "field": "type", "value": "FACT"},
            "batch_update_memories",
        )
        assert batch["data"]["failed_ids"] == [mid]
    assert await engine.get_memory(mid) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", [("status", "bad"), ("importance", "bad"), ("type", "")]
)
async def test_invalid_batch_value_has_no_side_effects(real_db_stack, field, value):
    engine = real_db_stack["memory_engine"]
    mid = await seed(engine)
    before = await engine.get_memory(mid)
    result = await edit(
        engine,
        {"memory_ids": [mid], "field": field, "value": value},
        "batch_update_memories",
    )
    assert result["data"]["failed_ids"] == [mid]
    assert await engine.get_memory(mid) == before


@pytest.mark.asyncio
async def test_single_status_transitions_and_deletion(real_db_stack):
    engine = real_db_stack["memory_engine"]
    mid = await seed(engine)
    for status in ("active", "archived", "active"):
        assert (
            await edit(engine, {"memory_id": mid, "field": "status", "value": status})
        )["status"] == "ok"
    result = await edit(
        engine, {"memory_ids": [mid, str(mid), "bad"]}, "batch_delete_memories"
    )
    assert result["data"]["deleted_count"] == 1 and result["data"]["failed_ids"] == [
        "bad"
    ]
    assert await engine.get_memory_record(mid) is None
