"""Graph candidates must be hydrated from canonical documents before fusion."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from tests.integration.test_real_db_end_to_end import real_db_stack as real_db_stack
from astrbot_plugin_dynamics_memory.core.retrieval.dual_route_retriever import (
    DualRouteRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.graph_retriever import GraphResult
from astrbot_plugin_dynamics_memory.core.retrieval.narrative_plan import (
    NarrativeSearchPlan,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_graph_expands_past_stale_foreign_top_candidate(real_db_stack, batched):
    engine = real_db_stack["memory_engine"]
    foreign = await engine.add_memory(
        "foreign private secret", session_id="foreign", persona_id="other"
    )
    local = await engine.add_memory(
        "canonical local fact", session_id="local", persona_id="reader"
    )
    graph_rows = [
        GraphResult(
            mid,
            0.9,
            0.1,
            0.8,
            0.8,
            "stale index text",
            {"session_id": "local", "persona_id": "reader"},
        )
        for mid in (foreign, local)
    ]
    graph = NS(search=AsyncMock(side_effect=lambda query, k, *a, **kw: graph_rows[:k]))
    docs = NS(search=AsyncMock(return_value=[]))

    async def validate(rows):
        return [
            r
            for r in rows
            if r.metadata.get("session_id") == "local"
            and r.metadata.get("persona_id") == "reader"
        ]

    plan = NarrativeSearchPlan(1, validate, initial=1, graph_ceiling=4)
    retriever = DualRouteRetriever(
        docs,
        graph,
        engine.get_memory,
        memory_batch_loader=engine.faiss_db.document_storage.get_documents
        if batched
        else None,
    )
    result = await retriever.search(
        "关系沿革", k=1, session_id="local", persona_id="reader", narrative_plan=plan
    )
    assert [r.doc_id for r in result] == [local]
    assert result[0].content == "canonical local fact"
    assert [call.args[1] for call in graph.search.await_args_list] == [1, 2]
    assert plan.graph_fallback[0].doc_id == local
