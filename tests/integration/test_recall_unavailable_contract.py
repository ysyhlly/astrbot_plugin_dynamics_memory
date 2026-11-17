"""Public recall reports explicit unavailability instead of exposing summaries."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from tests.integration.test_public_evidence_pipeline import pipeline as pipeline, SCOPE
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.recall_runtime import BoundedRecallTaskPool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "terminating",
        "initializing",
        "denied",
        "auth_error",
        "auth_timeout",
        "closed",
        "disabled",
        "invalid_purpose",
        "no_evidence",
    ],
)
async def test_public_recall_unavailable_paths_do_not_expose_unverified_text(
    pipeline, mode
):
    p, provider, _ = pipeline
    p._recall_runtime = BoundedRecallTaskPool()
    p.initializer.memory_engine = NS(
        search_narrative_history=AsyncMock(return_value=[]),
        search_narrative_semantic=AsyncMock(return_value=[]),
    )
    service = p.evidence_service
    purpose = "reply"
    if mode == "terminating":
        p._terminating = True
    elif mode == "initializing":
        p.initializer.is_initialized = False
    elif mode == "denied":
        p._narrative_scope_allowed = AsyncMock(return_value=False)
    elif mode == "auth_error":
        p._narrative_scope_allowed = AsyncMock(side_effect=OSError("unavailable"))
    elif mode == "auth_timeout":

        async def slow(*a):
            await asyncio.sleep(0.2)
            return True

        p._narrative_scope_allowed = slow
        p.config_manager = ConfigManager(
            {"recall_engine": {"narrative_total_timeout_seconds": 0.1}}
        )
    elif mode == "closed":
        await p._recall_runtime.close()
    elif mode == "disabled":
        p.config_manager = ConfigManager({"recall_engine": {"top_k": 0}})
    elif mode == "invalid_purpose":
        purpose = "unknown"
    elif mode == "no_evidence":
        p.evidence_service = None
    try:
        result = await p.recall_context(SCOPE, "coffee", purpose=purpose)
        assert result["items"] == [] and result["evidence_status"] in {
            "unavailable",
            "empty",
            "timeout",
            "invalid",
        }
        provider.text_chat.assert_not_awaited()
    finally:
        p.evidence_service = service
        await p._recall_runtime.close()
