"""Initializer APIs must refuse startup after failed migration checks."""

import sqlite3
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.plugin_initializer import PluginInitializer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "no_manager",
        "up_to_date",
        "success",
        "backup_missing",
        "failed_restore",
        "check_error",
    ],
)
async def test_migration_success_and_failure_preserve_original_database(
    tmp_path, scenario
):
    original = tmp_path / "database.db"
    backup = tmp_path / "backup.db"
    for path in (original, backup):
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE memories (value TEXT)")
            db.execute("INSERT INTO memories VALUES ('original')")
    owner = PluginInitializer.__new__(PluginInitializer)
    owner.config_manager = ConfigManager()

    async def migrate(**kwargs):
        if scenario == "failed_restore":
            with sqlite3.connect(original) as db:
                db.execute("UPDATE memories SET value='partially changed'")
            return {"success": False, "message": "failed"}
        return {"success": True, "message": "complete"}

    manager = NS(
        db_path=str(original),
        needs_migration=AsyncMock(return_value=scenario != "up_to_date"),
        create_backup=AsyncMock(
            return_value=None if scenario == "backup_missing" else str(backup)
        ),
        migrate=AsyncMock(side_effect=migrate),
    )
    owner.db_migration = None if scenario == "no_manager" else manager
    if scenario == "check_error":
        manager.needs_migration.side_effect = OSError("database inaccessible")
    if scenario in {"backup_missing", "failed_restore"}:
        with pytest.raises(Exception, match="迁移"):
            await owner._check_and_migrate_database()
    else:
        await owner._check_and_migrate_database()
    if scenario in {"no_manager", "up_to_date", "backup_missing", "check_error"}:
        manager.migrate.assert_not_awaited()
    with sqlite3.connect(original) as db:
        assert db.execute("SELECT value FROM memories").fetchone() == ("original",)
