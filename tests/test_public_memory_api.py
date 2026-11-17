"""Public consumer contracts: active registry, readiness, scoped read-only pages."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from astrbot_plugin_dynamics_memory.api import public


@pytest.mark.parametrize("mapping", [True, False])
def test_registry_reload_disable_and_public_accessors(monkeypatch, mapping):
    monkeypatch.setattr(public, "_get_active_plugin_impl", lambda: None)
    engine, connection = object(), object()
    owner = NS(
        initializer=NS(
            is_initialized=True,
            memory_engine=engine,
            conversation_manager=NS(store=NS(connection=connection)),
        )
    )
    row = NS(name="astrbot_plugin_dynamics_memory", star_cls=owner, activated=True)
    entries = [row]
    context = NS(get_all_stars=lambda: {"memory": entries[0]} if mapping else entries)
    assert public.find_plugin(context) is owner
    assert public.get_memory_engine(owner) is engine
    assert (
        public.get_conversation_manager(owner) is owner.initializer.conversation_manager
    )
    assert public.unavailable_stage(context) == ""
    row.activated = False
    assert public.find_plugin(context) is None
    assert public.unavailable_stage(context) == "not_detected"
    row.activated = True
    replacement = NS(initializer=NS(is_initialized=False))
    row.star_cls = replacement
    assert public.find_plugin(context) is replacement
    assert public.unavailable_stage(context) == "not_ready"
    assert public.unavailable_reason(context)


@pytest.mark.asyncio
async def test_readiness_failure_timeout_and_component_requirements():
    owner = NS(
        initializer=NS(
            is_initialized=False, ensure_initialized=AsyncMock(return_value=False)
        )
    )
    assert not await public.ensure_ready(owner)
    owner.initializer.ensure_initialized.side_effect = TimeoutError()
    assert not await public.ensure_ready(owner)
    owner.initializer.is_initialized = True
    owner.initializer.memory_engine = object()
    assert await public.ensure_ready(owner)
    assert not await public.ensure_ready(owner, require=("conversation_manager",))
    owner.series_runtime_allowed = lambda: False
    assert not await public.ensure_ready(owner)
    assert public.get_memory_engine(owner) is None
    assert not await public.ensure_ready(None)


def test_broken_registry_and_properties_never_escape(monkeypatch):
    def broken(*args):
        raise RuntimeError("unloaded")

    monkeypatch.setattr(public, "_get_active_plugin_impl", broken)
    assert public.get_active_plugin() is None
    assert (
        public.find_plugin(NS(get_all_stars=broken, get_registered_star=broken)) is None
    )
    assert public.get_memory_engine(None) is None
    assert public.get_conversation_manager(None) is None
    assert public.get_history_reader(None) is None
    with pytest.raises(public.Unavailable):
        public.require_plugin(None)
    with pytest.raises(public.Unavailable):
        public.require_engine(None)


@pytest.mark.asyncio
async def test_history_pages_keep_watermark_and_scope(tmp_path, monkeypatch):
    async with aiosqlite.connect(tmp_path / "history.db") as db:
        await db.execute(
            "CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, sender_id TEXT, sender_name TEXT, timestamp, metadata)"
        )
        await db.executemany(
            "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)",
            [
                (
                    1,
                    "qq:GroupMessage:a",
                    "user",
                    "first",
                    "u",
                    None,
                    123,
                    '{"kind":"source"}',
                ),
                (
                    2,
                    "qq:GroupMessage:a",
                    "assistant",
                    "second",
                    "bot",
                    "Bot",
                    "bad",
                    "broken json",
                ),
                (3, "qq:FriendMessage:u", "user", "private", "u", "U", 125, "{}"),
                (4, "qq:GroupMessage:b", "user", "other", "v", "V", 126, "[]"),
            ],
        )
        await db.commit()
        assert set(await public.query_history_sessions(db)) == {
            "qq:GroupMessage:a",
            "qq:GroupMessage:b",
        }
        bounds = await public.query_history_bounds(db, "qq:GroupMessage:a")
        assert bounds["count"] == 2 and bounds["high"] == 2
        rows = await public.query_history_page(
            db, "qq:GroupMessage:a", 0, bounds["high"], 1
        )
        assert [r["content"] for r in rows] == ["first"] and rows[0]["name"] == "u"
        rows = await public.query_history_page(
            db, "qq:GroupMessage:a", 1, bounds["high"], 10
        )
        assert rows[0]["metadata"] == {} and rows[0]["ts"] == 0
        assert (await public.query_history_bounds(db, "missing"))["count"] == 0
        owner = NS(
            initializer=NS(
                is_initialized=True, conversation_manager=NS(store=NS(connection=db))
            )
        )
        monkeypatch.setattr(public, "_get_active_plugin_impl", lambda: owner)
        assert public.get_history_reader(None) is db
    with pytest.raises(public.Unavailable):
        await public.query_history_page(None, "qq:GroupMessage:a", 0, 10, 1)
