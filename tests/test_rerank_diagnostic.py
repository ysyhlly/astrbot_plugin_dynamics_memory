from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin


def _plugin(engine_config, resolver=None):
    engine = SimpleNamespace(config=dict(engine_config))
    if resolver is not None:
        engine.rerank_provider_resolver = resolver
    plugin = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    plugin.initializer = SimpleNamespace(memory_engine=engine)
    return plugin


def test_enabled_rerank_without_provider_is_reported_unconfigured_without_fallback():
    calls = []

    def resolver(provider_id):
        calls.append(provider_id)
        return "qwr-provider"

    plugin = _plugin({"rerank_enabled": True, "rerank_provider_id": ""}, resolver)

    result = plugin._effective_rerank_status()

    assert result == {
        "enabled": True,
        "provider_configured": False,
        "resolver_available": True,
        "effective": "skipped_unconfigured",
    }
    assert calls == []


@pytest.mark.parametrize(
    "resolver",
    [
        lambda _provider_id: "fixture-resolved-provider",
        lambda _provider_id: (_ for _ in ()).throw(
            RuntimeError("fixture resolver failure")
        ),
    ],
)
def test_configured_rerank_is_explicitly_unprobed_without_calling_resolver(resolver):
    calls = []

    def counted(provider_id):
        calls.append(provider_id)
        return resolver(provider_id)

    plugin = _plugin(
        {"rerank_enabled": True, "rerank_provider_id": "fixture-reranker"}, counted
    )

    result = plugin._effective_rerank_status()

    assert result == {
        "enabled": True,
        "provider_configured": True,
        "resolver_available": True,
        "effective": "configured_not_probed",
    }
    assert calls == []


def test_configured_rerank_without_resolver_is_reported_unavailable():
    plugin = _plugin({"rerank_enabled": True, "rerank_provider_id": "fixture-reranker"})

    result = plugin._effective_rerank_status()

    assert result == {
        "enabled": True,
        "provider_configured": True,
        "resolver_available": False,
        "effective": "resolver_unavailable",
    }
