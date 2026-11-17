import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import HybridRetriever
from astrbot_plugin_dynamics_memory.core.retrieval.rrf_fusion import FusedResult


def rows(text='alpha', scope='s'):
    return [FusedResult(i, 1.0, None, None, body, {'session_id': scope})
            for i, body in enumerate((text, 'beta'))]


def setup():
    provider = SimpleNamespace(rerank=AsyncMock(return_value=[
        SimpleNamespace(index=0, relevance_score=.2),
        SimpleNamespace(index=1, relevance_score=.9)]))
    retriever = object.__new__(HybridRetriever)
    retriever.config = {}
    retriever.rerank_provider_resolver = lambda: provider
    return retriever, provider


@pytest.mark.asyncio
async def test_score_cache_keeps_fresh_metadata_and_content_changes_miss():
    retriever, provider = setup()
    await retriever._apply_rerank('q', rows(scope='first'), 1)
    current = await retriever._apply_rerank('q', rows(scope='second'), 2)
    assert provider.rerank.await_count == 1
    assert all(row.metadata['session_id'] == 'second' for row in current)
    await retriever._apply_rerank('q', rows(text='edited'), 2)
    assert provider.rerank.await_count == 2


@pytest.mark.asyncio
async def test_single_candidate_skips_remote():
    retriever, provider = setup()
    await retriever._apply_rerank('q', rows()[:1], 1)
    provider.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_identical_queries_coalesce():
    retriever, provider = setup()
    await asyncio.gather(*(retriever._apply_rerank('q', rows(), 2) for _ in range(6)))
    assert provider.rerank.await_count == 1
    assert not retriever._rerank_locks


@pytest.mark.asyncio
async def test_failure_cooldown_preserves_local_results_and_recovers():
    retriever, provider = setup()
    provider.rerank.side_effect = RuntimeError('400 arrearage')
    for query in ('q', 'different'):
        result = await retriever._apply_rerank(query, rows(), 2)
        assert [r.doc_id for r in result] == [0, 1]
    assert provider.rerank.await_count == 1
    retriever._rerank_retry_at = 0
    provider.rerank.side_effect = None
    assert (await retriever._apply_rerank('q', rows(), 2))[0].doc_id == 1
    assert provider.rerank.await_count == 2


@pytest.mark.asyncio
async def test_provider_replacement_invalidates_scores():
    retriever, old = setup()
    await retriever._apply_rerank('q', rows(), 2)
    _, new = setup()
    retriever.rerank_provider_resolver = lambda: new
    await retriever._apply_rerank('q', rows(), 2)
    assert old.rerank.await_count == new.rerank.await_count == 1


@pytest.mark.asyncio
async def test_cache_expiry_and_invalid_response():
    retriever, provider = setup()
    await retriever._apply_rerank('q', rows(), 2)
    for key, (_, values) in retriever._rerank_cache.items():
        retriever._rerank_cache[key] = (0, values)
    provider.rerank.return_value = [SimpleNamespace(index=-1, relevance_score=float('nan'))]
    result = await retriever._apply_rerank('q', rows(), 2)
    assert [r.doc_id for r in result] == [0, 1]
    assert provider.rerank.await_count == 2


@pytest.mark.asyncio
async def test_unrelated_queries_do_not_wait_for_slow_request():
    retriever, provider = setup()
    started, release = asyncio.Event(), asyncio.Event()
    original = provider.rerank.return_value
    async def remote(query, documents):
        if query == 'slow':
            started.set()
            await release.wait()
        return original
    provider.rerank.side_effect = remote
    slow = asyncio.create_task(retriever._apply_rerank('slow', rows(), 2))
    await started.wait()
    try:
        await asyncio.wait_for(retriever._apply_rerank('fast', rows(), 2), .5)
    finally:
        release.set()
        await slow


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_lock():
    retriever, provider = setup()
    started = asyncio.Event()
    async def remote(**kwargs):
        started.set()
        await asyncio.Event().wait()
    provider.rerank.side_effect = remote
    task = asyncio.create_task(retriever._apply_rerank('q', rows(), 2))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not retriever._rerank_locks
    assert retriever._rerank_retry_at == 0
