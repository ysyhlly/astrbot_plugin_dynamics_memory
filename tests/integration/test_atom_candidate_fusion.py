"""Atom-only candidates are hydrated from the correct parent scope before fusion."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from tests.integration.test_real_db_end_to_end import real_db_stack as real_db_stack
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import HybridResult


@pytest.mark.asyncio
async def test_atom_only_parent_hydration_keeps_scope_and_persona(real_db_stack):
    engine = real_db_stack["memory_engine"]
    first = await engine.add_memory("first fact", session_id="local", persona_id="p")
    second = await engine.add_memory("second fact", session_id="local", persona_id="p")
    foreign = await engine.add_memory(
        "foreign secret", session_id="other", persona_id="p"
    )
    wrong_persona = await engine.add_memory(
        "other persona secret", session_id="local", persona_id="q"
    )
    engine.atom_retriever = NS(
        search=AsyncMock(
            return_value=[
                NS(parent_memory_id=mid, base_score=0.8)
                for mid in (first, second, foreign, wrong_persona)
            ]
        )
    )
    row = await engine.get_memory(first)
    original = HybridResult(first, 0.8, 0.1, 0.7, 0.9, row["text"], row["metadata"])
    result = await engine._attach_agentic_atoms([original], "fact", 5, "local", "p", {})
    assert {r.doc_id for r in result} == {first, second}
    assert {r.content for r in result} == {"first fact", "second fact"}
    assert all(
        0 < r.final_score <= 1 and r.score_breakdown["atom_score"] == 0.8
        for r in result
    )
    assert original.score_breakdown is None
