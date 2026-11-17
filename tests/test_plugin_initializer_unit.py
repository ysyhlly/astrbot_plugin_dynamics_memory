"""Unit tests covering PluginInitializer lifecycle, retry, and recovery branches."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.base.exceptions import ProviderNotReadyError
from astrbot_plugin_dynamics_memory.core.plugin_initializer import PluginInitializer


def _context():
    return SimpleNamespace(
        platform_manager=SimpleNamespace(get_insts=lambda: []),
        get_config=lambda: {},
        get_provider_by_id=Mock(return_value=None),
        get_all_embedding_providers=Mock(return_value=[]),
        get_all_providers=Mock(return_value=[]),
        get_using_provider_async=AsyncMock(return_value=None),
    )


@pytest.mark.asyncio
async def test_initialize_success_and_already_initialized(monkeypatch, tmp_path):
    init = PluginInitializer(_context(), ConfigManager(), str(tmp_path))
    monkeypatch.setattr(
        init, "_wait_for_providers_non_blocking", AsyncMock(return_value=True)
    )

    async def complete():
        init._initialization_complete = True

    monkeypatch.setattr(
        init, "_complete_initialization", AsyncMock(side_effect=complete)
    )
    assert await init.initialize() is True
    assert init.is_initialized is True
    # Already initialized early return
    assert await init.initialize() is True
    assert await init.ensure_initialized(timeout=0.05) is True


@pytest.mark.asyncio
async def test_initialize_missing_providers_starts_retry_and_stops(
    monkeypatch, tmp_path
):
    init = PluginInitializer(_context(), ConfigManager(), str(tmp_path))
    monkeypatch.setattr(
        init, "_wait_for_providers_non_blocking", AsyncMock(return_value=False)
    )

    async def noop_retry():
        await asyncio.sleep(10)

    monkeypatch.setattr(init, "_retry_initialization", noop_retry)
    assert await init.initialize() is False
    assert init._retry_task is not None
    # Duplicate call does not overwrite running retry task
    first_task = init._retry_task
    init._start_retry_task_if_needed()
    assert init._retry_task is first_task
    await init.stop_background_tasks()
    assert init._retry_task is None


@pytest.mark.asyncio
async def test_initialize_exception_triggers_teardown_and_retry(monkeypatch, tmp_path):
    init = PluginInitializer(_context(), ConfigManager(), str(tmp_path))
    monkeypatch.setattr(
        init, "_wait_for_providers_non_blocking", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        init,
        "_complete_initialization",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    teardown = AsyncMock()
    start_retry = Mock()
    monkeypatch.setattr(init, "_teardown_partial_init", teardown)
    monkeypatch.setattr(init, "_start_retry_task_if_needed", start_retry)
    assert await init.initialize() is False
    assert "boom" in (init.error_message or "")
    teardown.assert_awaited_once()
    start_retry.assert_called_once()


@pytest.mark.asyncio
async def test_wait_for_providers_and_retry_initialization(monkeypatch, tmp_path):
    init = PluginInitializer(_context(), ConfigManager(), str(tmp_path))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    calls = 0

    async def init_providers(silent=True):
        nonlocal calls
        calls += 1
        if calls >= 2:
            init.embedding_provider = object()
            init.llm_provider = object()

    monkeypatch.setattr(init, "_initialize_providers", init_providers)
    assert await init._wait_for_providers_non_blocking(max_wait=5.0) is True
    assert init._providers_ready is True

    # Retry loop failure -> _initialization_failed = True
    retry_init = PluginInitializer(_context(), ConfigManager(), str(tmp_path))
    retry_init._max_provider_attempts = 5
    monkeypatch.setattr(retry_init, "_initialize_providers", AsyncMock())
    await retry_init._retry_initialization()
    assert retry_init.is_failed is True
    assert await retry_init.ensure_initialized(timeout=0.05) is False

    # Retry loop succeeds on 2nd attempt after transient teardown
    ok_init = PluginInitializer(_context(), ConfigManager(), str(tmp_path))
    ok_init._max_provider_attempts = 5
    attempt = 0

    async def ready_later(silent=True):
        ok_init.embedding_provider = object()
        ok_init.llm_provider = object()

    async def complete_later():
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise RuntimeError("transient")
        ok_init._initialization_complete = True

    monkeypatch.setattr(ok_init, "_initialize_providers", ready_later)
    monkeypatch.setattr(ok_init, "_complete_initialization", complete_later)
    monkeypatch.setattr(ok_init, "_teardown_partial_init", AsyncMock())
    await ok_init._retry_initialization()
    assert ok_init.is_initialized is True


@pytest.mark.asyncio
async def test_open_vector_storage_validation_and_backup_restore_errors(tmp_path):
    init = PluginInitializer(_context(), ConfigManager(), str(tmp_path))
    with pytest.raises(ProviderNotReadyError):
        await init._open_vector_storage()
    init.embedding_provider = object()
    with pytest.raises(ProviderNotReadyError):
        await init._open_vector_storage()

    init.db_migration = SimpleNamespace(db_path=str(tmp_path / "missing.db"))
    assert await init._restore_db_backup(str(tmp_path)) is False

    bad_store = SimpleNamespace(
        sync_message_counts=AsyncMock(side_effect=RuntimeError("db err"))
    )
    await init._repair_message_counts(bad_store)


@pytest.mark.asyncio
async def test_run_scheduled_index_rebuild_and_on_done_callbacks(tmp_path):
    init = PluginInitializer(_context(), ConfigManager(), str(tmp_path))
    init.memory_engine = SimpleNamespace(
        rebuild_graph_index=AsyncMock(return_value={"rebuilt": 2, "skipped": 0}),
        index_maintenance_status={},
    )
    init.index_validator = SimpleNamespace(
        rebuild_indexes=AsyncMock(
            return_value={
                "success": True,
                "partial": False,
                "processed": 2,
                "total": 2,
                "message": "ok",
            }
        )
    )
    await init._run_scheduled_index_rebuild(
        "test", 2, force_full_vector=True, rebuild_graph=True
    )
    assert init.index_maintenance_status["state"] == "ready"
    init.index_validator.rebuild_indexes.assert_awaited_once()

    # _on_retry_task_done & _on_index_maintenance_done with exception
    failed_task = asyncio.Future()
    failed_task.set_exception(RuntimeError("task boom"))
    init._on_retry_task_done(failed_task)
    init._index_maintenance_task = failed_task
    init._on_index_maintenance_done(failed_task)
    assert init._index_maintenance_task is None
