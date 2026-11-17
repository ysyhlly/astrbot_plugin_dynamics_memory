"""Upgrade old sidecars while retaining corrections and refusing future schemas."""

import sqlite3

import pytest

from astrbot_plugin_dynamics_memory.storage.evidence_sidecar import (
    EvidenceSidecar,
    SCHEMA_VERSION,
)
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key
from tests.test_evidence_typed_view import _service_with_source_backed_atom, SCOPE_A


@pytest.mark.asyncio
async def test_old_nonnullable_action_ids_upgrade_with_backup_and_revision_preserved(
    tmp_path,
):
    sidecar, service, typed, atom = await _service_with_source_backed_atom(tmp_path)
    path = sidecar.path
    await service.correct_insight(atom, 0, "withdraw", reason="original correction")
    await service.close()
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE evidence_schema SET schema_version=1,retry_extension_version=0"
        )
        db.execute("UPDATE evidence_materials SET status='failed',next_attempt_at=0")
        db.execute("ALTER TABLE evidence_corrections RENAME TO old_corrections")
        db.execute("""CREATE TABLE evidence_corrections(
            correction_id TEXT PRIMARY KEY,scope_key TEXT NOT NULL,correction_revision INTEGER NOT NULL,
            target_id TEXT NOT NULL,action TEXT NOT NULL,replacement_atom_id TEXT,
            affected_ids_json TEXT NOT NULL,reason TEXT NOT NULL,action_id TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,UNIQUE(scope_key,correction_revision),UNIQUE(scope_key,action_id))""")
        db.execute("""INSERT INTO evidence_corrections SELECT correction_id,scope_key,correction_revision,
            target_id,action,replacement_atom_id,affected_ids_json,reason,COALESCE(action_id,''),created_at FROM old_corrections""")
        db.execute("DROP TABLE old_corrections")
    upgraded = EvidenceSidecar(path)
    await upgraded.initialize()
    rows = await upgraded.list_admin_atoms()
    assert any(row["atom_id"] == atom and row["status"] == "withdrawn" for row in rows)
    assert (
        upgraded.peek_evidence_revision(scope_key(SCOPE_A))["correction_revision"] == 1
    )
    backups = list(tmp_path.glob(path.name + ".schema-v1-*.bak"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as db:
        assert (
            db.execute("SELECT schema_version FROM evidence_schema").fetchone()[0] == 1
        )
        assert (
            db.execute("SELECT reason FROM evidence_corrections").fetchone()[0]
            == "original correction"
        )
    with sqlite3.connect(path) as db:
        assert (
            db.execute("SELECT schema_version FROM evidence_schema").fetchone()[0]
            == SCHEMA_VERSION
        )
        assert (
            db.execute("SELECT action_id FROM evidence_corrections").fetchone()[0]
            is None
        )
    assert (await upgraded.list_admin_materials())[0]["next_attempt_at"] > 0


@pytest.mark.asyncio
async def test_future_schema_is_refused_without_rewriting_records(tmp_path):
    path = tmp_path / "future.db"
    sidecar = EvidenceSidecar(path)
    await sidecar.initialize()
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE evidence_schema SET schema_version=?", (SCHEMA_VERSION + 10,)
        )
        db.execute("CREATE TABLE future_source(id INTEGER PRIMARY KEY, body TEXT)")
        db.execute("INSERT INTO future_source VALUES(1,'future data')")
    reopened = EvidenceSidecar(path)
    with pytest.raises(RuntimeError, match="newer"):
        await reopened.initialize()
    with sqlite3.connect(path) as db:
        assert (
            db.execute("SELECT schema_version FROM evidence_schema").fetchone()[0]
            == SCHEMA_VERSION + 10
        )
        assert (
            db.execute("SELECT body FROM future_source").fetchone()[0] == "future data"
        )


@pytest.mark.asyncio
async def test_admin_status_tracks_supersession_and_invalidated_dependents(tmp_path):
    sidecar, service, typed, atom = await _service_with_source_backed_atom(tmp_path)
    try:
        result = await service.correct_insight(
            atom,
            0,
            "supersede",
            replacement="corrected interpretation",
            reason="reviewed",
            action_id="one-review",
        )
        rows = await sidecar.list_admin_atoms(200)
        assert next(r for r in rows if r["atom_id"] == atom)["status"] == "superseded"
        assert all("claim" not in row and "text" not in row for row in rows)
        again = await service.correct_insight(
            atom,
            0,
            "supersede",
            replacement="corrected interpretation",
            reason="reviewed",
            action_id="one-review",
        )
        assert again["correction_revision"] == result["correction_revision"]
        with pytest.raises((ValueError, RuntimeError)):
            await service.correct_insight(atom, 0, "withdraw", reason="stale")
    finally:
        await service.close()
