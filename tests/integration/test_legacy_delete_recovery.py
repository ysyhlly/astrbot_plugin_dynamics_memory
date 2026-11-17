"""Retained hybrid deletion API must restore lexical recall if vector deletion fails."""

from unittest.mock import AsyncMock
import pytest
from tests.integration.test_real_db_end_to_end import real_db_stack as real_db_stack


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["none", "vector_false", "vector_error", "bm25_false", "bm25_error"]
)
async def test_legacy_delete_consistency(real_db_stack, monkeypatch, failure):
    engine = real_db_stack["memory_engine"]
    mid = await engine.add_memory(
        "coffee original", session_id="s", persona_id="p", importance=0.9
    )
    retriever = engine.hybrid_retriever
    if failure != "none":
        target = (
            retriever.vector_retriever
            if failure.startswith("vector")
            else retriever.bm25_retriever
        )
        mock = (
            AsyncMock(side_effect=OSError("simulated storage failure"))
            if failure.endswith("error")
            else AsyncMock(return_value=False)
        )
        monkeypatch.setattr(target, "delete_document", mock)
    assert await retriever.delete_memory(mid) is (failure == "none")
    rows = await retriever.bm25_retriever.search("coffee", 5)
    assert (mid in {r.doc_id for r in rows}) is (failure != "none")
    assert (await engine.get_memory(mid) is None) is (failure == "none")
