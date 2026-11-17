"""Extra unit tests covering remaining branches of MemoryHandlerUpdateMixin."""

from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.memory_revision import (
    MemoryRevisionConflict,
    memory_revision,
)
from astrbot_plugin_dynamics_memory.core.page_api import PluginPageApi
from tests.test_page_api import FakePlugin, _mock_page_request, _patch_page_request


@pytest.fixture
def api():
    return PluginPageApi(FakePlugin(ready=True))


@pytest.mark.asyncio
async def test_update_memory_revision_conflict_and_replace_error(api):
    engine = api.plugin.initializer.memory_engine
    engine.get_memory_record = AsyncMock(return_value=None)

    req = _mock_page_request(
        get_json={
            "memory_id": 1,
            "field": "importance",
            "value": 5,
            "expected_revision": "stale-rev",
        }
    )
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "error" and res["code"] == "memory_revision_conflict"

    record = {"id": 1, "text": "hello", "metadata": {"status": "active"}}
    engine.get_memory_record = AsyncMock(return_value=record)
    req = _mock_page_request(
        get_json={
            "memory_id": 1,
            "field": "importance",
            "value": 5,
            "expected_revision": "wrong-rev",
        }
    )
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "error" and res["code"] == "memory_revision_conflict"

    engine.replace_memory = AsyncMock(side_effect=MemoryRevisionConflict())
    req = _mock_page_request(
        get_json={"memory_id": 1, "field": "content", "value": "updated"}
    )
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "error" and res["code"] == "memory_revision_conflict"

    engine.replace_memory = AsyncMock(side_effect=RuntimeError("replace failed"))
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "error" and "replace failed" in res["message"]


@pytest.mark.asyncio
async def test_update_memory_archive_restore_deleted_type_and_reason(api):
    engine = api.plugin.initializer.memory_engine
    active_record = {
        "id": 1,
        "text": "hello",
        "metadata": {"status": "active", "memory_type": "GENERAL"},
    }
    archived_record = {
        "id": 1,
        "text": "hello",
        "metadata": {"status": "archived", "memory_type": "GENERAL"},
    }

    # archive conflict and archive failure (0 archived)
    engine.get_memory_record = AsyncMock(return_value=active_record)
    engine.archive_memories = AsyncMock(side_effect=MemoryRevisionConflict())
    req = _mock_page_request(
        get_json={"memory_id": 1, "field": "status", "value": "archived"}
    )
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["code"] == "memory_revision_conflict"

    engine.archive_memories = AsyncMock(return_value=0)
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "error" and "归档失败" in res["message"]

    # restore conflict and restore failure
    engine.get_memory_record = AsyncMock(return_value=archived_record)
    engine.restore_memory = AsyncMock(side_effect=MemoryRevisionConflict())
    req = _mock_page_request(
        get_json={"memory_id": 1, "field": "status", "value": "active"}
    )
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["code"] == "memory_revision_conflict"

    engine.restore_memory = AsyncMock(return_value=False)
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "error" and "恢复失败" in res["message"]

    # status -> deleted and type update with reason
    engine.get_memory_record = AsyncMock(return_value=active_record)
    engine.update_memory = AsyncMock(return_value=True)
    req = _mock_page_request(
        get_json={
            "memory_id": 1,
            "field": "status",
            "value": "deleted",
            "reason": "cleanup",
        }
    )
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "ok"
    assert (
        engine.update_memory.await_args.args[1]["metadata"]["update_reason"]
        == "cleanup"
    )

    req = _mock_page_request(get_json={"memory_id": 1, "field": "type", "value": "   "})
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "error"

    req = _mock_page_request(
        get_json={
            "memory_id": 1,
            "field": "type",
            "value": "FACTUAL",
            "expected_revision": memory_revision(active_record),
        }
    )
    with _patch_page_request(req):
        res = await api.update_memory()
    assert res["status"] == "ok"


@pytest.mark.asyncio
async def test_batch_update_memories_validation_and_mixed_results(api):
    engine = api.plugin.initializer.memory_engine

    with _patch_page_request(_mock_page_request(get_json={"memory_ids": []})):
        res = await api.batch_update_memories()
    assert res["status"] == "error"

    with _patch_page_request(_mock_page_request(get_json={"memory_ids": [1]})):
        res = await api.batch_update_memories()
    assert res["status"] == "error"

    with _patch_page_request(
        _mock_page_request(get_json={"memory_ids": [1], "field": "bad", "value": "x"})
    ):
        res = await api.batch_update_memories()
    assert res["status"] == "error"

    async def fake_update(memory_id, updates):
        if memory_id == 2:
            return False
        if memory_id == 3:
            raise RuntimeError("boom")
        return True

    engine.update_memory = AsyncMock(side_effect=fake_update)

    with _patch_page_request(
        _mock_page_request(
            get_json={
                "memory_ids": [1, "bad", 2, 3],
                "field": "status",
                "value": "archived",
            }
        )
    ):
        res = await api.batch_update_memories()
    assert res["status"] == "ok"
    assert res["data"]["updated_count"] == 1
    assert res["data"]["failed_count"] == 3

    with _patch_page_request(
        _mock_page_request(
            get_json={"memory_ids": [1], "field": "status", "value": "invalid"}
        )
    ):
        res = await api.batch_update_memories()
    assert res["data"]["failed_count"] == 1

    with _patch_page_request(
        _mock_page_request(
            get_json={"memory_ids": [1], "field": "importance", "value": 999}
        )
    ):
        res = await api.batch_update_memories()
    assert res["data"]["failed_count"] == 1

    with _patch_page_request(
        _mock_page_request(
            get_json={"memory_ids": [1], "field": "importance", "value": 8}
        )
    ):
        res = await api.batch_update_memories()
    assert res["data"]["updated_count"] == 1

    with _patch_page_request(
        _mock_page_request(get_json={"memory_ids": [1], "field": "type", "value": "  "})
    ):
        res = await api.batch_update_memories()
    assert res["data"]["failed_count"] == 1

    with _patch_page_request(
        _mock_page_request(
            get_json={"memory_ids": [1], "field": "type", "value": "EVENT"}
        )
    ):
        res = await api.batch_update_memories()
    assert res["data"]["updated_count"] == 1
