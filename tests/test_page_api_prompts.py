"""Tests for PromptHandler WebUI prompt management API."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from quart import Quart

from astrbot_plugin_dynamics_memory.core.page_api_modules.prompt_handler import (
    PromptHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import PageApiUtils


class _FakePromptManager:
    def __init__(self) -> None:
        self.prompts = {
            "reflection_summary": "custom summary prompt",
            "agent_recall_policy": "recall policy prompt",
        }
        self.defaults = {
            "reflection_summary": "default summary prompt",
            "agent_recall_policy": "default recall policy prompt",
        }

    def list_prompts(self):
        return [
            {"id": pid, "content": content} for pid, content in self.prompts.items()
        ]

    def get_categories(self):
        return {
            "reflection": {"name": "Reflection"},
            "agent_recall": {"name": "Agent Recall"},
        }

    def get_prompt_detail(self, prompt_id: str):
        if prompt_id not in self.prompts:
            raise KeyError(f"未知提示词: {prompt_id}")
        return {"id": prompt_id, "content": self.prompts[prompt_id]}

    def get_default_content(self, prompt_id: str):
        if prompt_id not in self.defaults:
            raise KeyError(f"未知提示词: {prompt_id}")
        return self.defaults[prompt_id]

    def get_prompt(self, prompt_id: str):
        if prompt_id not in self.prompts:
            raise KeyError(f"未知提示词: {prompt_id}")
        return self.prompts[prompt_id]

    def update_prompt(self, prompt_id: str, content: str):
        if prompt_id not in self.prompts:
            raise KeyError(f"未知提示词: {prompt_id}")
        self.prompts[prompt_id] = content

    def reset_prompt(self, prompt_id: str):
        if prompt_id not in self.defaults:
            raise KeyError(f"未知提示词: {prompt_id}")
        self.prompts[prompt_id] = self.defaults[prompt_id]


def test_beta_recall_enabled_variants():
    assert PromptHandler(PageApiUtils(), plugin=None)._beta_recall_enabled() is True
    assert (
        PromptHandler(PageApiUtils(), plugin=SimpleNamespace())._beta_recall_enabled()
        is True
    )

    class _DisabledCfg:
        def get(self, key, default=None):
            if key == "agent_tools.enable_recall_tool":
                return True
            if key == "agent_tools.enable_agentic_recall_beta":
                return False
            return default

    class _EnabledCfg:
        def get(self, key, default=None):
            return True

    class _BrokenCfg:
        def get(self, key, default=None):
            raise RuntimeError("cfg broken")

    assert (
        PromptHandler(
            PageApiUtils(), plugin=SimpleNamespace(config_manager=_DisabledCfg())
        )._beta_recall_enabled()
        is False
    )
    assert (
        PromptHandler(
            PageApiUtils(),
            plugin=SimpleNamespace(
                initializer=SimpleNamespace(config_manager=_EnabledCfg())
            ),
        )._beta_recall_enabled()
        is True
    )
    assert (
        PromptHandler(
            PageApiUtils(), plugin=SimpleNamespace(config_manager=_BrokenCfg())
        )._beta_recall_enabled()
        is True
    )


def test_get_manager_raises_when_uninitialized(monkeypatch):
    import astrbot_plugin_dynamics_memory.core.prompts.prompt_manager as pm_mod

    monkeypatch.setattr(pm_mod, "get_prompt_manager", lambda: None)
    with pytest.raises(RuntimeError, match="PromptManager"):
        PromptHandler._get_manager()

    fake = _FakePromptManager()
    monkeypatch.setattr(pm_mod, "get_prompt_manager", lambda: fake)
    assert PromptHandler._get_manager() is fake


@pytest.mark.asyncio
async def test_prompt_handler_crud_and_error_paths(monkeypatch):
    import astrbot_plugin_dynamics_memory.core.prompts.prompt_manager as pm_mod

    fake = _FakePromptManager()
    monkeypatch.setattr(pm_mod, "get_prompt_manager", lambda: fake)
    app = Quart(__name__)

    class _DisabledCfg:
        def get(self, key, default=None):
            if key == "agent_tools.enable_recall_tool":
                return True
            if key == "agent_tools.enable_agentic_recall_beta":
                return False
            return default

    handler = PromptHandler(
        PageApiUtils(), plugin=SimpleNamespace(config_manager=_DisabledCfg())
    )
    async with app.test_request_context("/prompts"):
        listed = await handler.list_prompts()
    assert listed["status"] == "ok"
    assert [p["id"] for p in listed["data"]["prompts"]] == ["reflection_summary"]
    assert [c["id"] for c in listed["data"]["categories"]] == ["reflection"]

    async with app.test_request_context("/prompts/detail"):
        missing = await handler.get_prompt_detail()
    assert missing["status"] == "error"

    async with app.test_request_context("/prompts/detail?id=reflection_summary"):
        detail = await handler.get_prompt_detail()
    assert (
        detail["status"] == "ok"
        and detail["data"]["content"] == "custom summary prompt"
    )

    async with app.test_request_context("/prompts/detail?id=unknown"):
        missing_key = await handler.get_prompt_detail()
    assert missing_key["status"] == "error"

    async with app.test_request_context("/prompts/default"):
        no_id = await handler.get_prompt_default()
    assert no_id["status"] == "error"

    async with app.test_request_context("/prompts/default?id=reflection_summary"):
        default_res = await handler.get_prompt_default()
    assert (
        default_res["status"] == "ok"
        and default_res["data"]["content"] == "default summary prompt"
    )

    async with app.test_request_context("/prompts/default?id=unknown"):
        bad_default = await handler.get_prompt_default()
    assert bad_default["status"] == "error"

    async with app.test_request_context("/prompts/update", method="POST", json={}):
        empty_id = await handler.update_prompt()
    assert empty_id["status"] == "error"

    async with app.test_request_context(
        "/prompts/update",
        method="POST",
        json={"id": "reflection_summary", "content": ""},
    ):
        empty_content = await handler.update_prompt()
    assert empty_content["status"] == "error"

    async with app.test_request_context(
        "/prompts/update",
        method="POST",
        json={"id": "reflection_summary", "content": "updated summary"},
    ):
        updated = await handler.update_prompt()
    assert updated["status"] == "ok"
    assert fake.prompts["reflection_summary"] == "updated summary"

    async with app.test_request_context(
        "/prompts/update", method="POST", json={"id": "unknown", "content": "x"}
    ):
        bad_update = await handler.update_prompt()
    assert bad_update["status"] == "error"

    async with app.test_request_context("/prompts/reset", method="POST", json={}):
        bad_reset_id = await handler.reset_prompt()
    assert bad_reset_id["status"] == "error"

    async with app.test_request_context(
        "/prompts/reset", method="POST", json={"id": "reflection_summary"}
    ):
        reset_res = await handler.reset_prompt()
    assert reset_res["status"] == "ok"
    assert reset_res["data"]["content"] == "default summary prompt"

    async with app.test_request_context(
        "/prompts/reset", method="POST", json={"id": "unknown"}
    ):
        bad_reset = await handler.reset_prompt()
    assert bad_reset["status"] == "error"

    monkeypatch.setattr(pm_mod, "get_prompt_manager", lambda: None)
    for path, method, body, fn in (
        ("/prompts", "GET", None, handler.list_prompts),
        ("/prompts/detail?id=x", "GET", None, handler.get_prompt_detail),
        ("/prompts/default?id=x", "GET", None, handler.get_prompt_default),
        ("/prompts/update", "POST", {"id": "x", "content": "y"}, handler.update_prompt),
        ("/prompts/reset", "POST", {"id": "x"}, handler.reset_prompt),
    ):
        ctx = (
            app.test_request_context(path, method=method, json=body)
            if body is not None
            else app.test_request_context(path, method=method)
        )
        async with ctx:
            err = await fn()
        assert err["status"] == "error"
