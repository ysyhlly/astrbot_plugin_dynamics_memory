"""Admin HTTP adapters validate payloads and serialize bounded storage repair."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from quart import Quart
from astrbot_plugin_dynamics_memory.core.page_api import PluginPageApi
from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import PageApiUtils


def api():
    instance = PluginPageApi.__new__(PluginPageApi)
    instance.utils = PageApiUtils()
    engine = NS(
        requeue_sqlite_failures=AsyncMock(return_value=2),
        repair_retained_sources=AsyncMock(return_value={"repaired": 1}),
    )
    instance._ensure_plugin_ready = AsyncMock(
        return_value=({"memory_engine": engine}, None)
    )
    instance.plugin = NS(
        correct_insight=AsyncMock(return_value={"revision": 2}),
        list_evidence_insights=AsyncMock(return_value={"ok": True, "items": []}),
    )
    return instance, engine


async def call(instance, method, payload):
    app = Quart(__name__)
    async with app.test_request_context("/admin", method="POST", json=payload):
        return await getattr(instance, method)()


@pytest.mark.asyncio
async def test_bounded_repair_serializes_and_never_calls_models():
    instance, engine = api()
    running = 0

    async def repair(limit, after):
        nonlocal running
        running += 1
        assert running == 1
        await asyncio.sleep(0.01)
        running -= 1
        return {"repaired": 1}

    engine.repair_retained_sources.side_effect = repair
    results = await asyncio.gather(
        *(
            call(
                instance,
                "repair_storage",
                {"limit": 999, "after": -1, "requeue_sqlite": True},
            )
            for _ in range(2)
        )
    )
    assert all(
        r["data"]
        == {
            "repaired": 1,
            "requeued_sqlite_operations": 2,
            "model_calls": 0,
            "embedding_calls": 0,
        }
        for r in results
    )
    assert all(
        c.args == (200, 0) for c in engine.repair_retained_sources.await_args_list
    )
    for bad in ([1], {"limit": "bad"}):
        assert (await call(instance, "repair_storage", bad))["status"] == "error"
    assert engine.repair_retained_sources.await_count == 2


@pytest.mark.asyncio
async def test_correction_requires_integer_revision_and_preserves_errors():
    instance, _ = api()
    valid = {
        "target_id": "atom1",
        "action": "dispute",
        "expected_revision": 1,
        "reason": "user correction",
        "action_id": "action1",
    }
    assert (await call(instance, "correct_evidence_insight", valid)) == {
        "ok": True,
        "revision": 2,
    }
    instance.plugin.correct_insight.assert_awaited_with(
        "atom1", 1, "dispute", "", "user correction", "action1"
    )
    for bad in (
        [1],
        {**valid, "expected_revision": True},
        {**valid, "reason": 1},
        {**valid, "replacement": {}},
        {},
    ):
        assert (await call(instance, "correct_evidence_insight", bad)) == {
            "ok": False,
            "error": "invalid_payload",
        }
    for exc, error in [
        (RuntimeError("stale"), "stale_revision_or_unavailable"),
        (ValueError("missing"), "invalid_or_missing_target"),
    ]:
        instance.plugin.correct_insight.side_effect = exc
        assert (await call(instance, "correct_evidence_insight", valid)) == {
            "ok": False,
            "error": error,
        }
    assert await instance.list_evidence_insights() == {"ok": True, "items": []}
    instance.plugin.list_evidence_insights.assert_awaited_with(100)
    unavailable = {"status": "error", "code": "not_ready"}
    instance._ensure_plugin_ready.return_value = (None, unavailable)
    for method in (
        "repair_storage",
        "correct_evidence_insight",
        "list_evidence_insights",
    ):
        assert await call(instance, method, {}) == unavailable
