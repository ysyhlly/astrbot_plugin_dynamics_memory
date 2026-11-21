"""Unit tests covering InitializerFaissMixin and SeriesMixin edge/error paths."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import subprocess
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core import plugin_initializer_faiss as faiss_mod
from astrbot_plugin_dynamics_memory.core.base.exceptions import InitializationError
from astrbot_plugin_dynamics_memory.core.plugin_initializer_faiss import (
    InitializerFaissMixin,
    _bridge_env_is_risky,
    _faiss_error_details,
    _installed_faiss_version,
    _is_faiss_binding_mismatch,
    _sanitize_path,
    _should_try_faiss_generic,
    _warn_if_ascii_bridge_missing,
)
from astrbot_plugin_dynamics_memory.core.series import (
    SeriesMixin,
    _activity_timestamp,
    active_entries,
    find_role,
    manifest,
    value,
)


def _proc(
    returncode: int, stderr: str = "", stdout: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["python", "-c", "import faiss"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_faiss_diagnostic_helpers(monkeypatch, tmp_path):
    assert _sanitize_path("C:/plain/ascii.index") == "C:/plain/ascii.index"
    assert _sanitize_path("C:/用户/测试/vec.index") == "C:/[***]/[***]/vec.index"

    neg = _proc(-4, "illegal instruction")
    assert "进程被信号 4 终止" in _faiss_error_details(neg)
    assert _should_try_faiss_generic(neg) is True
    assert _should_try_faiss_generic(_proc(1, "AVX2 unsupported")) is True
    assert _should_try_faiss_generic(_proc(1, "some unrelated error")) is False

    assert _is_faiss_binding_mismatch("missing _superkmeans symbol") is True
    assert _is_faiss_binding_mismatch("Python binding version mismatch") is True
    assert _is_faiss_binding_mismatch("plain import error") is False

    def missing_pkg(_name):
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(faiss_mod.metadata, "version", missing_pkg)
    assert _installed_faiss_version() == "未知"

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("USERPROFILE", "C:\\测试用户")
    assert _bridge_env_is_risky() is True
    monkeypatch.setattr(faiss_mod, "_bridge_warning_logged", False)
    monkeypatch.setattr(faiss_mod, "bridge_installed", lambda: False)
    _warn_if_ascii_bridge_missing()
    assert faiss_mod._bridge_warning_logged is True
    _warn_if_ascii_bridge_missing()


def test_check_faiss_runtime_branches(monkeypatch):
    mixin = InitializerFaissMixin()

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=10)

    monkeypatch.setattr(faiss_mod.subprocess, "run", raise_timeout)
    with pytest.raises(InitializationError, match="FAISS 运行时检查失败"):
        mixin._check_faiss_runtime()

    monkeypatch.setattr(
        faiss_mod.subprocess,
        "run",
        lambda *_a, **_kw: _proc(1, "superkmeans symbol error"),
    )
    monkeypatch.setattr(faiss_mod.metadata, "version", lambda _name: "1.14.2")
    with pytest.raises(InitializationError, match="1.14.2"):
        mixin._check_faiss_runtime()

    monkeypatch.setattr(
        faiss_mod.subprocess, "run", lambda *_a, **_kw: _proc(1, "module missing")
    )
    with pytest.raises(InitializationError, match="无法在当前 Python 环境中加载"):
        mixin._check_faiss_runtime()

    calls = []

    def fallback_succeeds(*_a, **kwargs):
        calls.append(kwargs.get("env", {}).get("FAISS_OPT_LEVEL"))
        return _proc(0) if len(calls) == 2 else _proc(1, "avx unsupported")

    monkeypatch.setattr(faiss_mod.subprocess, "run", fallback_succeeds)
    monkeypatch.delenv("FAISS_OPT_LEVEL", raising=False)
    mixin._check_faiss_runtime()
    assert os.environ.get("FAISS_OPT_LEVEL") == "generic"

    calls.clear()

    def fallback_fails(*_a, **_kw):
        calls.append(1)
        if len(calls) == 1:
            return _proc(1, "avx unsupported")
        return _proc(1, "generic also failed")

    monkeypatch.setattr(faiss_mod.subprocess, "run", fallback_fails)
    with pytest.raises(InitializationError, match="generic"):
        mixin._check_faiss_runtime()


def test_faiss_index_cleanup_helpers(tmp_path, monkeypatch):
    mixin = InitializerFaissMixin()

    empty_file = tmp_path / "empty.index"
    empty_file.write_bytes(b"")
    assert mixin._discard_empty_index(str(empty_file)) is True
    assert not empty_file.exists()

    non_empty = tmp_path / "good.index"
    non_empty.write_bytes(b"abc")
    assert mixin._discard_empty_index(str(non_empty)) is False
    assert mixin._discard_empty_index(str(tmp_path / "missing.index")) is False

    assert (
        mixin._discard_corrupt_index(str(non_empty), RuntimeError("No such file"))
        is False
    )
    assert (
        mixin._discard_corrupt_index(str(non_empty), RuntimeError("corrupt header"))
        is True
    )
    assert not non_empty.exists()

    monkeypatch.setattr(
        os, "remove", lambda _p: (_ for _ in ()).throw(OSError("locked"))
    )
    assert (
        mixin._discard_corrupt_index(str(tmp_path / "x"), RuntimeError("bad")) is False
    )

    vec_db = NS(embedding_storage=NS(index=NS(d=768)))
    assert mixin._index_dimension(vec_db) == 768
    idx = tmp_path / "wrong_dim.index"
    idx.write_bytes(b"123")
    monkeypatch.setattr(os, "remove", lambda p: os.unlink(p))
    assert mixin._reset_incompatible_index(vec_db, str(idx), 768) is False
    assert mixin._reset_incompatible_index(vec_db, str(idx), 384) is True
    assert not idx.exists()


@pytest.mark.asyncio
async def test_series_discovery_and_shared_evidence_guards():
    assert _activity_timestamp(True) is None
    assert _activity_timestamp(-1) is None
    assert _activity_timestamp(float("inf")) is None
    assert _activity_timestamp(100.0) == 100.0

    assert value({"k": 1}, "k") == 1
    assert value(NS(k=2), "k") == 2

    assert active_entries(NS()) == []
    assert active_entries(NS(get_all_stars=lambda: "bad")) == []

    async def coro_caps():
        return {"schema_version": 1}

    coro_holder = NS(get_dynamics_capabilities=lambda: coro_caps())
    assert manifest(coro_holder) == {}
    assert manifest(NS(get_dynamics_capabilities=lambda: 1 / 0)) == {}

    class DummyType:
        pass

    ctx = NS(
        get_all_stars=lambda: {
            "a": NS(name="a", activated=False, star_cls=NS()),
            "b": NS(name="b", activated=True, star_cls=DummyType),
            "w1": NS(
                name="w1",
                activated=True,
                star_cls=NS(
                    get_dynamics_capabilities=lambda: {
                        "schema_version": 1,
                        "enabled": True,
                        "ready": True,
                        "capabilities": {"character": 1},
                    }
                ),
            ),
        }
    )
    assert find_role(ctx, "character") is not None
    assert find_role(ctx, "missing") is None

    class DummySeries(SeriesMixin):
        def __init__(self):
            self.context = ctx
            self.initializer = NS(is_initialized=True)
            self._series_enabled = True
            self._terminating = False

        async def _narrative_scope_allowed(self, _scope, _uid=None):
            return True

        async def _memory_scope_allowed(self, _scope, _uid=None):
            return True

        async def get_evidence_view(self, *_args, **_kwargs):
            return {"ok": True}

    s = DummySeries()
    assert s.series_runtime_allowed() is True
    assert s.get_dynamics_capabilities()["ready"] is True
    with pytest.raises(ValueError, match="bounded list"):
        await s.get_shared_evidence({"umo": "p:GroupMessage:1"}, "not-a-list")
    with pytest.raises(ValueError, match="bounded list"):
        await s.get_shared_evidence({"umo": "p:GroupMessage:1"}, ["x"] * 65)
    assert await s.get_shared_evidence({"umo": "p:GroupMessage:1"}, []) == {"ok": True}


@pytest.mark.asyncio
async def test_series_context_routes_and_activity_validation(monkeypatch):
    class _Store:
        list_series_conflicts = AsyncMock(return_value=[{"id": "c1"}])
        resolve_series_conflict = AsyncMock(return_value={"resolved": True})
        get_series_conflict = AsyncMock(return_value={"id": "c1"})

    async def coro_learning(**_kw):
        return {}

    async def failing_world(_scope):
        raise RuntimeError("world offline")

    world_star = NS(
        get_dynamics_capabilities=lambda: {
            "schema_version": 1,
            "enabled": True,
            "ready": True,
            "capabilities": {"character": 1},
        },
        get_character_context=failing_world,
    )
    learning_star = NS(get_learning_context=lambda **_kw: coro_learning())
    stars = {
        "world": NS(name="world", activated=True, star_cls=world_star),
        "astrbot_plugin_dynamics_learning": NS(
            name="astrbot_plugin_dynamics_learning",
            activated=True,
            star_cls=learning_star,
        ),
    }

    class DummySeries(SeriesMixin):
        def __init__(self):
            self.context = NS(
                get_all_stars=lambda: stars,
                get_config=lambda: {"dashboard": {"username": "admin"}},
                register_web_api=lambda *_a, **_kw: None,
            )
            self.initializer = NS(
                is_initialized=True,
                conversation_manager=NS(store=_Store()),
                memory_engine=NS(add_memory=AsyncMock(return_value=1)),
            )
            self._series_enabled = True
            self._terminating = False
            self.allow_scope = True
            self.allow_later = True

        async def _ensure_plugin_ready(self, wait=False):
            return True, ""

        async def _narrative_scope_allowed(self, _scope, _uid=None):
            res = self.allow_scope
            self.allow_scope = self.allow_later
            return res

        async def _memory_scope_allowed(self, _scope, _uid=None):
            return True

        async def get_evidence_view(self, *_args, **_kwargs):
            return {"evidence": []}

    s = DummySeries()
    ctx_res = await s.get_series_context(
        {"umo": "p:GroupMessage:1", "platform": "p"},
        [],
        learning_request={
            "scope": "p:GroupMessage:1",
            "persona": "per",
            "member": "u1",
            "baseline": "b",
            "host_version": "1",
        },
    )
    assert ctx_res["available"] is True and ctx_res["learning"] is None

    s.allow_later = False
    invalidated = await s.get_series_context(
        {"umo": "p:GroupMessage:1", "platform": "p"}, []
    )
    assert invalidated == {
        "schema_version": 1,
        "available": False,
        "reason": "source_invalidated",
    }

    s._series_enabled = False
    assert (await s.get_series_context({"umo": "p:GroupMessage:1"}, []))[
        "reason"
    ] == "scope_or_owner_unavailable"
    s._series_enabled = True

    import astrbot.api.web as web_mod
    import astrbot.dashboard.api.auth as auth_mod

    monkeypatch.setattr(web_mod, "request", NS(_request=None), raising=False)
    monkeypatch.setattr(
        web_mod,
        "json_response",
        lambda data, status_code=200: (status_code, data),
        raising=False,
    )
    monkeypatch.setattr(
        auth_mod,
        "require_dashboard_user",
        AsyncMock(return_value="admin"),
        raising=False,
    )

    assert (await s._series_route(s.series_status)())[0] == 403

    monkeypatch.setattr(
        web_mod,
        "request",
        NS(
            _request=object(),
            query={"id": "c1"},
            json=AsyncMock(return_value={"bad": 1}),
        ),
        raising=False,
    )
    assert (await s._series_route(s.series_resolve_conflict)())[0] == 409

    monkeypatch.setattr(
        web_mod,
        "request",
        NS(
            _request=object(),
            query={"id": "c1"},
            json=AsyncMock(return_value={"id": "c1", "revision": 1, "reason": "ok"}),
        ),
        raising=False,
    )
    assert (await s._series_route(s.series_status)())[0] == 200
    assert (await s._series_route(s.series_conflict_list)())[0] == 200
    assert (await s._series_route(s.series_conflict_detail)())[0] == 200
    s.initializer.conversation_manager.store.get_series_conflict.assert_awaited_with(
        "c1"
    )
    assert (await s._series_route(s.series_resolve_conflict)())[0] == 200

    assert (await s.record_activity_experience("bad"))["reason"] == "invalid_payload"
    assert (
        await s.record_activity_experience(
            {"schema_version": 1, "scope": {"umo": "bad"}}
        )
    )["reason"] == "invalid_group_umo"
    assert (
        await s.record_activity_experience(
            {
                "schema_version": 1,
                "scope": {"umo": "p:GroupMessage:g1", "platform": "other"},
            }
        )
    )["reason"] == "invalid_platform"
    activity_payload = {
        "schema_version": 1,
        "scope": {"umo": "p:GroupMessage:g1", "platform": "p"},
        "activity_id": "a1",
        "title": "t",
        "summary": "sum",
        "occurred_at": time.time() - 1,
        "participants": [{"uid": "u1", "name": "U1"}],
        "source_refs": [],
    }
    assert (await s.record_activity_experience(activity_payload))[
        "reason"
    ] == "source_store_unavailable"
    s.initializer.conversation_manager.source_publication_lease = lambda *_a: None
    assert (await s.record_activity_experience(activity_payload))[
        "reason"
    ] == "memory_index_unavailable"
