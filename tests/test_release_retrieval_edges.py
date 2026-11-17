"""Scoring excerpts preserve corrections; partial startup releases all owned resources."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.plugin_initializer import PluginInitializer
from astrbot_plugin_dynamics_memory.core.retrieval.rerank_text import scoring_excerpt


def document():
    lines = ["header"] + [f"filler {i} " + ("x" * 25) for i in range(30)]
    lines[8] = "coffee tea preference"
    lines[20] = "correction: no longer coffee"
    lines[25] = "tea also liked"
    return lines


def test_excerpt_preserves_corrections_and_neighbours_without_rewriting():
    lines = document()
    text = "\n".join(lines)
    result = scoring_excerpt("coffee tea", text, 400)
    assert len(result) < len(text) and "[…]" in result
    for idx in (0, 1, 7, 8, 9, 19, 20, 21):
        assert lines[idx] in result
    assert all(line in lines or line == "[…]" for line in result.splitlines())
    assert lines[20] in result


@pytest.mark.parametrize(
    "query,limit",
    [
        ("?", 400),
        ("z", 400),
        ("什么怎么", 400),
        ("absent", 400),
        ("coffee", 10),
    ],
)
def test_excerpt_keeps_full_source_without_safe_discriminating_excerpt(query, limit):
    text = "\n".join(document())
    assert scoring_excerpt(query, text, limit) == text


def test_excerpt_handles_chinese_matches_and_equal_strength_ties():
    lines = document()
    lines[8] = "喜欢咖啡"
    lines[25] = "后来更正喜欢咖啡的说法"
    text = "\n".join(lines)
    result = scoring_excerpt("咖啡", text, 550)
    assert lines[8] in result and lines[25] in result
    assert scoring_excerpt("coffee", "\n".join(["coffee" * 100] * 3), 100) == "\n".join(
        ["coffee" * 100] * 3
    )
    assert scoring_excerpt("coffee", "coffee" * 1000, 20) == "coffee" * 1000


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", [False, True])
@pytest.mark.parametrize("with_engine", [False, True])
async def test_partial_initialization_closes_every_owned_resource_and_is_retryable(
    failing, with_engine
):
    owner = PluginInitializer.__new__(PluginInitializer)

    def closer():
        return AsyncMock(side_effect=OSError("close failed") if failing else None)

    persister = NS(aclose=closer())
    scheduler = NS(stop=closer())
    store = NS(close=closer())
    engine = NS(close=closer())
    vector = NS(close=closer())
    graph = NS(close=closer())
    owner._index_persisters = [persister]
    owner.decay_scheduler = scheduler
    owner.conversation_manager = NS(store=store)
    owner.memory_engine = engine if with_engine else None
    owner.db, owner.graph_db = vector, graph
    await owner._teardown_partial_init()
    for mock in (persister.aclose, scheduler.stop, store.close, vector.close):
        mock.assert_awaited_once()
    if with_engine:
        engine.close.assert_awaited_once()
        graph.close.assert_not_awaited()
    else:
        graph.close.assert_awaited_once()
    assert all(
        getattr(owner, name) is None
        for name in (
            "db",
            "graph_db",
            "memory_engine",
            "conversation_manager",
            "decay_scheduler",
        )
    )
    await owner._teardown_partial_init()
    vector.close.assert_awaited_once()
