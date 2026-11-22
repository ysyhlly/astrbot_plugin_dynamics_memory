"""Tests for shared ProviderBudget, telemetry, timezone, provider fallback, exclude_groups, and quiet_hours in Dynamics Memory."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.memory_scope import is_event_memory_allowed
from astrbot_plugin_dynamics_memory.core.processors.memory_processor import (
    MemoryProcessor,
)
from astrbot_plugin_dynamics_memory.core.series import SeriesMixin, _day_bounds
from astrbot_plugin_dynamics_memory.series_panels import (
    ProviderBudgetBusyError,
    is_provider_budget_busy_error,
    is_series_group_excluded,
    is_series_in_quiet_hours,
    resolve_series_provider,
    resolve_series_timezone,
    resolve_series_timezone_name,
    series_llm_call,
)


def _chat_ctx(
    *,
    activated: bool = True,
    rhythm_timezone: str = "Asia/Tokyo",
    reply_provider: str = "shared-reply-p",
    exclude_groups: list[str] | None = None,
    in_quiet_hours: bool = False,
    budget=None,
    telemetry=None,
    providers: dict | None = None,
    using_provider=None,
):
    class ChatOwner:
        def get_series_shared_config(self):
            return {
                "reply_provider": reply_provider,
                "decision_provider": "shared-decision-p",
                "rhythm_timezone": rhythm_timezone,
                "exclude_groups": list(exclude_groups or []),
                "takeover_groups": ["qq:GroupMessage:1001"],
                "takeover_all_groups": False,
                "persona_id": "p-main",
                "ui_language": "zh-CN",
                "quiet_hours": {
                    "daily_rhythm_enabled": True,
                    "force_sleep": False,
                    "in_quiet_hours": in_quiet_hours,
                },
            }

    provider_map = dict(providers or {})
    ctx = SimpleNamespace(
        get_all_stars=lambda: {
            "chat": SimpleNamespace(
                name="astrbot_plugin_chat_dynamics",
                activated=activated,
                star_cls=ChatOwner(),
            )
        },
        get_provider_by_id=lambda pid: provider_map.get(pid),
        get_using_provider=lambda *_a, **_kw: using_provider,
        get_all_providers=lambda: list(provider_map.values()),
    )
    if budget is not None:
        ctx._chat_dynamics_provider_budget = budget
    if telemetry is not None:
        ctx._chat_dynamics_series_telemetry = telemetry
    return ctx


class _FakeBudget:
    supports_series_telemetry = True

    def __init__(self, *, busy: bool = False):
        self.busy = busy
        self.calls: list[dict] = []

    async def run(
        self,
        provider_id: str,
        purpose: str,
        invoke,
        *,
        plugin_id: str = "unknown",
        acquire_timeout: float | None = None,
    ):
        self.calls.append(
            {
                "provider_id": provider_id,
                "purpose": purpose,
                "plugin_id": plugin_id,
                "acquire_timeout": acquire_timeout,
            }
        )
        if self.busy:
            raise ProviderBudgetBusyError(
                "busy", provider=provider_id, purpose=purpose, reason="acquire_timeout"
            )
        res = invoke()
        if hasattr(res, "__await__"):
            res = await res
        return res


@pytest.mark.asyncio
async def test_memory_processor_uses_shared_budget_and_surfaces_busy_error():
    provider = SimpleNamespace(
        provider_config={"id": "shared-reply-p"},
        text_chat=AsyncMock(
            return_value=SimpleNamespace(completion_text='{"ok": true}')
        ),
    )
    budget = _FakeBudget(busy=False)
    ctx = _chat_ctx(budget=budget, providers={"shared-reply-p": provider})
    proc = MemoryProcessor(context=ctx, config={})
    result = await proc._call_llm_with_retry("hello", "sys", purpose="memory_summary")
    assert result == '{"ok": true}'
    assert budget.calls
    assert budget.calls[0]["plugin_id"] == "astrbot_plugin_dynamics_memory"
    assert budget.calls[0]["provider_id"] == "shared-reply-p"
    assert budget.calls[0]["purpose"] == "memory_summary"

    busy_budget = _FakeBudget(busy=True)
    ctx_busy = _chat_ctx(budget=busy_budget, providers={"shared-reply-p": provider})
    proc_busy = MemoryProcessor(context=ctx_busy, config={})
    with pytest.raises(Exception) as exc_info:
        await proc_busy._call_llm_with_retry("hello", "sys", purpose="memory_summary")
    assert is_provider_budget_busy_error(exc_info.value)


@pytest.mark.asyncio
async def test_memory_provider_fallback_vs_explicit_override():
    explicit_p = SimpleNamespace(
        provider_config={"id": "explicit-bg"}, text_chat=AsyncMock()
    )
    shared_p = SimpleNamespace(
        provider_config={"id": "shared-reply-p"}, text_chat=AsyncMock()
    )
    host_p = SimpleNamespace(
        provider_config={"id": "host-default"}, text_chat=AsyncMock()
    )

    # 1. Explicit background_llm_provider_id wins
    ctx = _chat_ctx(
        providers={"explicit-bg": explicit_p, "shared-reply-p": shared_p},
        using_provider=host_p,
    )
    proc_explicit = MemoryProcessor(
        context=ctx, config={"background_llm_provider_id": "explicit-bg"}
    )
    assert await proc_explicit._get_current_llm_provider() is explicit_p

    # 2. Blank explicit config falls back to Chat Dynamics shared reply_provider
    proc_shared = MemoryProcessor(context=ctx, config={})
    assert await proc_shared._get_current_llm_provider() is shared_p

    # 3. Standalone without Chat Dynamics falls back to host default provider
    standalone_ctx = SimpleNamespace(
        get_all_stars=lambda: {},
        get_provider_by_id=lambda _pid: None,
        get_using_provider=lambda *_a, **_kw: host_p,
        get_all_providers=lambda: [host_p],
    )
    proc_standalone = MemoryProcessor(context=standalone_ctx, config={})
    assert await proc_standalone._get_current_llm_provider() is host_p


@pytest.mark.asyncio
async def test_memory_standalone_fallback_without_chat_dynamics():
    standalone_ctx = SimpleNamespace(get_all_stars=lambda: {})
    assert resolve_series_timezone_name(standalone_ctx) == "Asia/Shanghai"
    assert resolve_series_provider(standalone_ctx, "") == ""
    assert is_series_group_excluded(standalone_ctx, umo="qq:GroupMessage:1001") is False
    assert is_series_in_quiet_hours(standalone_ctx) is False

    out = await series_llm_call(
        standalone_ctx,
        "p1",
        "astrbot_plugin_dynamics_memory",
        "memory_summary",
        lambda: SimpleNamespace(completion_text="standalone"),
    )
    assert out.completion_text == "standalone"


def test_memory_timezone_inheritance_and_day_bounds():
    ctx_ny = _chat_ctx(rhythm_timezone="America/New_York")
    tz_ny = resolve_series_timezone(ctx_ny)
    assert getattr(tz_ny, "key", "") == "America/New_York"
    # 2026-11-22 02:00 UTC is still 2026-11-21 21:00 in America/New_York,
    # while it is already 2026-11-22 10:00 in Asia/Shanghai.
    ts = datetime.datetime(2026, 11, 22, 2, 0, tzinfo=datetime.timezone.utc).timestamp()
    start_ny, end_ny = _day_bounds("today", now_ts=ts, tz=tz_ny)
    start_sh, _ = _day_bounds("today", now_ts=ts, tz=resolve_series_timezone(None))
    assert start_ny != start_sh
    assert end_ny - start_ny == 86400.0


def test_memory_exclude_groups_and_scope_status():
    excluded_umo = "qq:GroupMessage:9001"
    allowed_umo = "qq:GroupMessage:1001"
    ctx = _chat_ctx(exclude_groups=[excluded_umo], in_quiet_hours=True)
    assert is_series_in_quiet_hours(ctx) is True

    ev_excluded = SimpleNamespace(
        unified_msg_origin=excluded_umo,
        get_group_id=lambda: "9001",
        get_sender_id=lambda: "u1",
        get_platform_name=lambda: "qq",
    )
    ev_allowed = SimpleNamespace(
        unified_msg_origin=allowed_umo,
        get_group_id=lambda: "1001",
        get_sender_id=lambda: "u1",
        get_platform_name=lambda: "qq",
    )
    assert is_event_memory_allowed({}, ev_excluded, context=ctx) is False
    assert is_event_memory_allowed({}, ev_allowed, context=ctx) is True

    class _DummyMemoryPlugin(SeriesMixin):
        def __init__(self, context):
            self.context = context
            self._series_enabled = True
            self._terminating = False
            self.config_manager = {}
            self.initializer = SimpleNamespace(is_initialized=True)

        def _series_memory_db_path(self):
            return None

    plugin = _DummyMemoryPlugin(ctx)
    st_excl = plugin.get_series_scope_status(excluded_umo)
    assert st_excl["enabled"] is False
    assert st_excl["excluded_by_core"] is True
    assert "主控排除" in st_excl["summary"]

    st_ok = plugin.get_series_scope_status(allowed_umo)
    assert st_ok["enabled"] is True
    assert st_ok["excluded_by_core"] is False
