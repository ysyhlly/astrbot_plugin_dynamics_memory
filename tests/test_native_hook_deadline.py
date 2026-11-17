from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from astrbot_plugin_dynamics_memory.core.recall_runtime import RecallDeadline


PLUGIN_MAIN = Path(__file__).resolve().parents[1] / "main.py"


def _extract_hook():
    tree = ast.parse(PLUGIN_MAIN.read_text(encoding="utf-8"))
    plugin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LivingMemoryPlugin"
    )
    method = next(
        node
        for node in plugin.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "handle_memory_recall"
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)

    class ShortDeadline:
        @staticmethod
        def start(**_kwargs):
            return RecallDeadline.start(total_seconds=0.09, final_reserve_seconds=0.02)

    namespace = {
        "asyncio": asyncio,
        "timed_recall_phase": __import__(
            "astrbot_plugin_dynamics_memory.core.recall_runtime",
            fromlist=["timed_recall_phase"],
        ).timed_recall_phase,
        "RecallDeadline": ShortDeadline,
        "is_managed_event": lambda _event: False,
    }
    exec(compile(module, str(PLUGIN_MAIN), "exec"), namespace)
    return namespace["handle_memory_recall"]


def _harness(*, projection, owner, collect):
    calls = []
    event = SimpleNamespace(
        unified_msg_origin="synthetic:FriendMessage:user",
        get_extra=lambda key, default=None: (
            {"schema_version": 1, "projection_restricted": True}
            if key == "_autonomous_blocking_context_v1"
            else default
        ),
    )
    request = SimpleNamespace()
    instance = SimpleNamespace(
        series_runtime_allowed=lambda: True,
        initializer=SimpleNamespace(is_initialized=True),
        event_handler=SimpleNamespace(
            handle_memory_recall=lambda *_args, **_kwargs: None
        ),
        _install_private_injection_getter=lambda *_args: None,
        _prepare_native_memory_projection=projection,
        _remember_blocking_owner=owner,
        _native_memory_recall_v1=collect,
        _native_projection_restricted=lambda _event: True,
        _native_clear_request=lambda _req: calls.append("clear"),
    )
    instance.handle_memory_recall = MethodType(_extract_hook(), instance)
    return instance, event, request, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_stage", ["projection", "owner", "ready"])
async def test_native_hook_applies_same_entry_deadline_to_projection_owner_and_ready(
    slow_stage,
):
    async def immediate(*_args, **_kwargs):
        return {"schema_version": 1, "scope": {}}

    async def slow(*_args, **_kwargs):
        await asyncio.sleep(0.5)
        return {"schema_version": 1, "scope": {}}

    projection = slow if slow_stage == "projection" else immediate
    owner = slow if slow_stage == "owner" else immediate

    async def collect(*_args, **kwargs):
        if slow_stage == "ready" and kwargs.get("collect_only"):
            await asyncio.sleep(0.5)
        return True

    instance, event, request, calls = _harness(
        projection=projection,
        owner=owner,
        collect=collect,
    )
    start = asyncio.get_running_loop().time()
    await instance.handle_memory_recall(event, request)
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 0.18
    assert request._lm_private_hook_outcome_v1 == "deadline_exceeded"
    if slow_stage == "ready":
        assert calls == ["clear"]
