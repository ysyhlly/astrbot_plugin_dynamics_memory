"""Prevent native System One providers from receiving memory summary prompts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import astrbot_plugin_dynamics_memory.core.plugin_initializer as initializer_module
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.plugin_initializer import PluginInitializer
from astrbot_plugin_dynamics_memory.core.processors.memory_processor import MemoryProcessor


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, "deleted-model", "decision-model"])
async def test_summary_uses_text_provider_when_default_is_systemone(configured):
    decision = SimpleNamespace(is_systemone_provider=True, text_chat=AsyncMock())
    text = SimpleNamespace(
        text_chat=AsyncMock(return_value=SimpleNamespace(completion_text="summary"))
    )
    context = Mock()
    context.get_provider_by_id.return_value = (
        decision if configured == "decision-model" else None
    )
    context.get_using_provider.return_value = decision
    context.get_all_providers.return_value = [decision, text]
    processor = MemoryProcessor(context=context, llm_provider=configured)

    assert (
        await processor._call_llm_with_retry("Summarize this conversation", "Memory")
        == "summary"
    )
    decision.text_chat.assert_not_awaited()
    text.text_chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_configured_text_provider_is_resolved_again_after_reload():
    first, replacement = SimpleNamespace(), SimpleNamespace()
    context = Mock()
    context.get_provider_by_id.side_effect = [first, replacement]
    processor = MemoryProcessor(context=context, llm_provider="text-model")

    assert await processor._get_current_llm_provider() is first
    assert await processor._get_current_llm_provider() is replacement
    context.get_using_provider.assert_not_called()


@pytest.mark.asyncio
async def test_no_summary_provider_when_only_systemone_is_available():
    decision = SimpleNamespace(is_systemone_provider=True)
    context = Mock()
    context.get_using_provider.return_value = decision
    context.get_all_providers.return_value = [decision]

    assert await MemoryProcessor(context=context)._get_current_llm_provider() is None
    assert await MemoryProcessor(llm_provider=decision)._get_current_llm_provider() is None


@pytest.mark.parametrize("configured", [None, "decision-model"])
@pytest.mark.asyncio
async def test_initializer_excludes_systemone(monkeypatch, tmp_path, configured):
    class TextProvider:
        pass

    class DecisionProvider(TextProvider):
        is_systemone_provider = True

    monkeypatch.setattr(initializer_module, "Provider", TextProvider)
    decision, text = DecisionProvider(), TextProvider()
    context = Mock()
    context.get_provider_by_id.return_value = decision
    context.get_all_embedding_providers.return_value = []
    context.get_using_provider.return_value = decision
    context.get_all_providers.return_value = [decision, text]
    config = ConfigManager({"provider_settings": {"llm_provider_id": configured}})
    initializer = PluginInitializer(context, config, str(tmp_path))

    await initializer._initialize_providers(silent=False)

    assert initializer.llm_provider is text


@pytest.mark.asyncio
async def test_initializer_waits_when_only_decision_models_are_available(
    monkeypatch, tmp_path
):
    class DecisionProvider:
        is_systemone_provider = True

    monkeypatch.setattr(initializer_module, "Provider", DecisionProvider)
    decision = DecisionProvider()
    context = Mock()
    context.get_all_embedding_providers.return_value = []
    context.get_using_provider.return_value = decision
    context.get_all_providers.return_value = [decision]
    initializer = PluginInitializer(context, ConfigManager(), str(tmp_path))

    await initializer._initialize_providers(silent=False)

    assert initializer.llm_provider is None
