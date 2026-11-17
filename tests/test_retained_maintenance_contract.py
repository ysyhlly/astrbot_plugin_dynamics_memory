"""Retained maintenance reports failures and runs scheduled rebuild paths."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot_plugin_dynamics_memory.core.plugin_initializer import PluginInitializer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["ready", "partial", "failed", "cancelled", "error", "migration"]
)
async def test_maintenance_reports_real_completion_state(outcome):
    owner = PluginInitializer.__new__(PluginInitializer)
    owner.memory_engine = NS()
    owner._graph_index_requires_rebuild = False
    owner._index_maintenance_status = {}
    owner._set_index_maintenance_status = Mock()
    owner._run_graph_index_rebuild = AsyncMock(return_value={"rebuilt": 3})

    async def rebuild(engine, **kwargs):
        await kwargs["progress_callback"](2, 3, "progress")
        return {
            "success": outcome != "failed",
            "partial": outcome == "partial",
            "processed": 3,
            "total": 3,
        }

    validator = NS(
        provider_fingerprint_changed=AsyncMock(return_value=(outcome != "migration")),
        get_migration_status=AsyncMock(
            return_value=(outcome == "migration", 3 if outcome == "migration" else 0)
        ),
        rebuild_indexes=AsyncMock(side_effect=rebuild),
    )
    owner.index_validator = validator
    if outcome in {"error", "cancelled"}:
        validator.provider_fingerprint_changed.side_effect = (
            OSError("provider unavailable")
            if outcome == "error"
            else asyncio.CancelledError()
        )
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await owner._run_index_maintenance()
    else:
        await owner._run_index_maintenance()
    state = owner._set_index_maintenance_status.call_args.kwargs["state"]
    assert state == {"error": "failed", "migration": "ready"}.get(outcome, outcome)
    if outcome in {"ready", "migration"}:
        if outcome == "ready":
            owner._run_graph_index_rebuild.assert_awaited_once()
        else:
            owner._run_graph_index_rebuild.assert_not_awaited()
    else:
        owner._run_graph_index_rebuild.assert_not_awaited()
