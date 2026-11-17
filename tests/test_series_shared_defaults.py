"""Tests for read-only series shared defaults inheritance in Dynamics Memory."""

from __future__ import annotations

from types import SimpleNamespace

from astrbot_plugin_dynamics_memory.series_panels import (
    get_series_defaults,
    inherit_provider,
)


def _chat_context(*, activated: bool = True, reply_provider: str = "series-reply"):
    class ChatOwner:
        def get_series_shared_config(self):
            return {
                "takeover_all_groups": False,
                "takeover_groups": ["qq:GroupMessage:42"],
                "exclude_groups": [],
                "reply_provider": reply_provider,
                "decision_provider": "series-decision",
                "persona_id": "persona-a",
                "bot_names": ["AnAn"],
                "rhythm_timezone": "Asia/Shanghai",
                "quiet_hours": {"daily_rhythm_enabled": True},
                "ui_language": "en-US",
                "api_key": "secret-should-not-pass",
            }

    return SimpleNamespace(
        get_all_stars=lambda: {
            "chat": SimpleNamespace(
                name="astrbot_plugin_chat_dynamics",
                activated=activated,
                star_cls=ChatOwner(),
            )
        }
    )


def test_memory_inherits_series_provider_and_language_when_empty():
    ctx = _chat_context()
    assert inherit_provider("", ctx, role="reply") == "series-reply"
    assert inherit_provider("explicit-llm", ctx, role="reply") == "explicit-llm"
    defaults = get_series_defaults(ctx)
    assert defaults["ui_language"] == "en-US"
    assert "api_key" not in defaults


def test_memory_safe_fallback_when_chat_dynamics_missing_or_disabled():
    missing_ctx = SimpleNamespace(get_all_stars=lambda: {})
    assert inherit_provider("", missing_ctx, role="reply") == ""
    assert get_series_defaults(missing_ctx) == {}

    disabled_ctx = _chat_context(activated=False)
    assert inherit_provider("", disabled_ctx, role="reply") == ""
    assert get_series_defaults(disabled_ctx) == {}
