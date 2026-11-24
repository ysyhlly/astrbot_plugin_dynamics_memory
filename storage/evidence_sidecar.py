"""Sidecar SQLite owner for evidence projections and append-only corrections."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sqlite_connection import (
    JOURNAL_MODE_DELETE,
    JOURNAL_MODE_WAL,
    resolve_journal_mode,
)

# Retry columns are an additive v3 extension. Keeping the marker at v3 lets
# the deployed rc7 reader reopen this sidecar during a source-only rollback.
SCHEMA_VERSION = 3
RETRY_EXTENSION_VERSION = 1
RETRY_DELAYS_SECONDS = (300, 900, 1800, 3600)
MAX_INVALID_SCHEMA_ATTEMPTS = 3


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class EvidenceSidecar:
    """Small SQLite sidecar; original conversation and memory databases are read-only here."""

    _bounded_queue_claim_v1 = True

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # Revision snapshots are RAM-only. Ordinary SQLite work must not block
        # their synchronous callers on the bot's event loop.
        self._revision_lock = threading.Lock()
        self._revision_cache: dict[str, dict] = {}
        self._ready = False
        self._read_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="evidence-read"
        )
        self._read_local = threading.local()
        self._read_stats_lock = threading.Lock()
        self._read_stats: dict[str, dict] = {}
        self._journal_mode = JOURNAL_MODE_WAL

    async def initialize(self) -> None:
        journal_mode = await resolve_journal_mode(self.path.parent)
        await asyncio.to_thread(self._initialize_sync, journal_mode)

    async def close(self) -> None:
        """Checkpoint and release sidecar file handles after background work stops."""
        await asyncio.to_thread(
            self._read_executor.shutdown, wait=True, cancel_futures=True
        )
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        with self._revision_lock:
            self._ready = False
        if self._journal_mode != JOURNAL_MODE_WAL:
            return
        with self._lock, self._session() as db:
            try:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
                db.execute("PRAGMA incremental_vacuum(128)").fetchall()
            except sqlite3.Error:
                pass

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=5000")
        return db

    @contextmanager
    def _session(self):
        db = self._connect()
        try:
            yield db
        finally:
            db.close()

    async def _read(self, operation, *args):
        queued = time.perf_counter()

        def run():
            started = time.perf_counter()
            timing = {
                "queue_ms": (started - queued) * 1000,
                "sql_ms": 0.0,
                "json_ms": 0.0,
            }
            self._read_local.timing = timing
            try:
                return operation(*args)
            finally:
                timing["total_ms"] = (time.perf_counter() - queued) * 1000
                with self._read_stats_lock:
                    previous = self._read_stats.get(operation.__name__, {})
                    self._read_stats[operation.__name__] = {
                        **{key: round(value, 3) for key, value in timing.items()},
                        "calls": previous.get("calls", 0) + 1,
                    }
                del self._read_local.timing

        return await asyncio.get_running_loop().run_in_executor(
            self._read_executor, run
        )

    def read_diagnostics(self) -> dict:
        """Bounded counters/timings only; never retain scopes, query text or rows."""
        with self._read_stats_lock:
            return copy.deepcopy(self._read_stats)

    def _read_rows(self, db, sql, params):
        started = time.perf_counter()
        try:
            return db.execute(sql, params).fetchall()
        finally:
            timing = getattr(self._read_local, "timing", None)
            if timing is not None:
                timing["sql_ms"] += (time.perf_counter() - started) * 1000

    def _read_json(self, value):
        started = time.perf_counter()
        try:
            return json.loads(value)
        finally:
            timing = getattr(self._read_local, "timing", None)
            if timing is not None:
                timing["json_ms"] += (time.perf_counter() - started) * 1000

    @contextmanager
    def _read_session(self):
        # WAL readers do not queue behind Python's writer lock. All SELECTs in
        # this operation see one committed snapshot, including multi-table reads.
        # Correction publication and final revision checks remain authoritative.
        with self._session() as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            try:
                yield db
            finally:
                db.rollback()

    def _initialize_sync(self, journal_mode: str = JOURNAL_MODE_WAL) -> None:
        with self._revision_lock:
            self._ready = False
        self._backup_before_schema_upgrade()
        with self._lock, self._session() as db:
            try:
                av_row = db.execute("PRAGMA auto_vacuum").fetchone()
                if av_row is None or int(av_row[0]) != 2:
                    db.execute("PRAGMA auto_vacuum=INCREMENTAL").fetchall()
            except sqlite3.Error:
                pass
            requested = (
                JOURNAL_MODE_WAL
                if journal_mode == JOURNAL_MODE_WAL
                else JOURNAL_MODE_DELETE
            )
            if requested == JOURNAL_MODE_WAL:
                try:
                    row = db.execute("PRAGMA journal_mode=WAL").fetchone()
                    self._journal_mode = (
                        JOURNAL_MODE_WAL
                        if row and str(row[0]).upper() == JOURNAL_MODE_WAL
                        else JOURNAL_MODE_DELETE
                    )
                except sqlite3.Error:
                    db.execute("PRAGMA journal_mode=DELETE").fetchall()
                    self._journal_mode = JOURNAL_MODE_DELETE
            else:
                db.execute("PRAGMA journal_mode=DELETE").fetchall()
                self._journal_mode = JOURNAL_MODE_DELETE
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_schema (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), schema_version INTEGER NOT NULL,
                retry_extension_version INTEGER NOT NULL DEFAULT 0
            )""")
            schema_columns = {
                row[1] for row in db.execute("PRAGMA table_info(evidence_schema)")
            }
            if "retry_extension_version" not in schema_columns:
                db.execute(
                    "ALTER TABLE evidence_schema ADD COLUMN retry_extension_version INTEGER NOT NULL DEFAULT 0"
                )
            row = db.execute(
                "SELECT schema_version FROM evidence_schema WHERE singleton=1"
            ).fetchone()
            if row and int(row[0]) > SCHEMA_VERSION:
                raise RuntimeError("Evidence sidecar schema is newer than this plugin")
            db.execute(
                "INSERT OR IGNORE INTO evidence_schema(singleton,schema_version) VALUES (1,?)",
                (SCHEMA_VERSION,),
            )
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_materials (
                material_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, material_kind TEXT NOT NULL,
                role TEXT NOT NULL, conversation_id TEXT NOT NULL, snapshot_fingerprint TEXT NOT NULL,
                source_position INTEGER NOT NULL, source_family_id TEXT NOT NULL DEFAULT '',
                text_hash TEXT NOT NULL, projection_key TEXT NOT NULL,
                card_version TEXT NOT NULL, status TEXT NOT NULL, diagnostics_code TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            )""")
            material_columns = {
                row[1] for row in db.execute("PRAGMA table_info(evidence_materials)")
            }
            extension_row = db.execute(
                "SELECT retry_extension_version FROM evidence_schema WHERE singleton=1"
            ).fetchone()
            extension_applied = bool(
                extension_row and int(extension_row[0]) >= RETRY_EXTENSION_VERSION
            )
            if "source_family_id" not in material_columns:
                db.execute(
                    "ALTER TABLE evidence_materials ADD COLUMN source_family_id TEXT NOT NULL DEFAULT ''"
                )
            if "attempt_count" not in material_columns:
                db.execute(
                    "ALTER TABLE evidence_materials ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
                )
            if "next_attempt_at" not in material_columns:
                db.execute(
                    "ALTER TABLE evidence_materials ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0"
                )
            if "outcome_code" not in material_columns:
                db.execute(
                    "ALTER TABLE evidence_materials ADD COLUMN outcome_code TEXT NOT NULL DEFAULT ''"
                )
            if not extension_applied:
                # Existing failed rows had no retry metadata. Delay only this
                # one additive migration pass so deploy cannot flood the bounded queue.
                db.execute(
                    "UPDATE evidence_materials SET next_attempt_at=? "
                    "WHERE status='failed' AND next_attempt_at=0",
                    (time.time() + 300,),
                )
            for row in db.execute(
                "SELECT material_id,scope_key,conversation_id,source_position,text_hash "
                "FROM evidence_materials WHERE source_family_id='' AND material_kind='host_record_only'"
            ).fetchall():
                # Old schema rows lack a trusted full-history prefix anchor. Keep
                # them exact-material-only; never infer cross-snapshot identity.
                db.execute(
                    "UPDATE evidence_materials SET source_family_id='' WHERE material_id=?",
                    (str(row[0]),),
                )
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_atoms (
                atom_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, material_id TEXT NOT NULL,
                kind TEXT NOT NULL, status TEXT NOT NULL, root_ids_json TEXT NOT NULL,
                root_material_ids_json TEXT NOT NULL DEFAULT '[]', atom_json TEXT NOT NULL,
                revision INTEGER NOT NULL, created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(material_id) REFERENCES evidence_materials(material_id)
            )""")
            atom_columns = {
                row[1] for row in db.execute("PRAGMA table_info(evidence_atoms)")
            }
            if "root_material_ids_json" not in atom_columns:
                db.execute(
                    "ALTER TABLE evidence_atoms ADD COLUMN root_material_ids_json TEXT NOT NULL DEFAULT '[]'"
                )
                for atom_row in db.execute(
                    "SELECT atom_id,root_ids_json,atom_json FROM evidence_atoms"
                ).fetchall():
                    old_roots = [str(x) for x in json.loads(atom_row[1])]
                    atom_json = json.loads(atom_row[2])
                    evidence_roots = [x for x in old_roots if x.startswith("ev1:")]
                    material_roots = [
                        x for x in old_roots if x.startswith(("mat1:", "leg1:"))
                    ]
                    # Also recover material provenance embedded in old atom JSON.
                    material_roots.extend(
                        str(x)
                        for x in atom_json.get("root_material_ids", [])
                        if str(x).startswith(("mat1:", "leg1:"))
                    )
                    atom_json["root_evidence_ids"] = sorted(set(evidence_roots))
                    atom_json["root_material_ids"] = sorted(set(material_roots))
                    db.execute(
                        "UPDATE evidence_atoms SET root_ids_json=?,root_material_ids_json=?,atom_json=? WHERE atom_id=?",
                        (
                            _json(sorted(set(evidence_roots))),
                            _json(sorted(set(material_roots))),
                            _json(atom_json),
                            str(atom_row[0]),
                        ),
                    )
                # Old families were computed without a trusted prefix anchor.
                db.execute(
                    "UPDATE evidence_materials SET source_family_id='' WHERE material_kind='host_record_only'"
                )
                for correction in db.execute(
                    "SELECT correction_id,affected_ids_json FROM evidence_corrections"
                ).fetchall():
                    affected = [
                        str(x)
                        for x in json.loads(correction[1])
                        if not str(x).startswith("fam1:")
                    ]
                    db.execute(
                        "UPDATE evidence_corrections SET affected_ids_json=? WHERE correction_id=?",
                        (_json(affected), str(correction[0])),
                    )
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_corrections (
                correction_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, correction_revision INTEGER NOT NULL,
                target_id TEXT NOT NULL, action TEXT NOT NULL, replacement_atom_id TEXT,
                affected_ids_json TEXT NOT NULL, reason TEXT NOT NULL, action_id TEXT,
                created_at REAL NOT NULL, UNIQUE(scope_key, correction_revision), UNIQUE(scope_key, action_id)
            )""")
            correction_columns = {
                row[1]: row
                for row in db.execute("PRAGMA table_info(evidence_corrections)")
            }
            if (
                "action_id" in correction_columns
                and int(correction_columns["action_id"][3]) != 0
            ):
                # v1 stored empty action IDs as a unique value; v2 uses NULL for
                # callers without an idempotency key so multiple corrections work.
                db.execute(
                    "DROP INDEX IF EXISTS idx_evidence_corrections_scope_revision"
                )
                db.execute(
                    "ALTER TABLE evidence_corrections RENAME TO evidence_corrections_v1"
                )
                db.execute("""CREATE TABLE evidence_corrections (
                    correction_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, correction_revision INTEGER NOT NULL,
                    target_id TEXT NOT NULL, action TEXT NOT NULL, replacement_atom_id TEXT,
                    affected_ids_json TEXT NOT NULL, reason TEXT NOT NULL, action_id TEXT,
                    created_at REAL NOT NULL, UNIQUE(scope_key, correction_revision), UNIQUE(scope_key, action_id)
                )""")
                db.execute("""INSERT INTO evidence_corrections(
                    correction_id,scope_key,correction_revision,target_id,action,replacement_atom_id,
                    affected_ids_json,reason,action_id,created_at
                ) SELECT correction_id,scope_key,correction_revision,target_id,action,replacement_atom_id,
                    affected_ids_json,reason,NULLIF(action_id,''),created_at FROM evidence_corrections_v1""")
                db.execute("DROP TABLE evidence_corrections_v1")
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_scope_revisions (
                scope_key TEXT PRIMARY KEY, correction_revision INTEGER NOT NULL
            )""")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_atoms_scope_status ON evidence_atoms(scope_key,status,kind)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_atoms_material ON evidence_atoms(material_id,status)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_corrections_scope_revision ON evidence_corrections(scope_key,correction_revision)"
            )
            # Queue metadata contains only source locators, never source text.
            # Additive tables are ignored by older readers during source-only rollback.
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_derivation_queue (
                material_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, umo TEXT NOT NULL DEFAULT '',
                source_locator_json TEXT NOT NULL, correction_revision INTEGER NOT NULL,
                projection_key TEXT NOT NULL, text_hash TEXT NOT NULL,
                status TEXT NOT NULL, available_at REAL NOT NULL,
                lease_id TEXT NOT NULL DEFAULT '', lease_until REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            )""")
            queue_columns = {
                row[1]
                for row in db.execute("PRAGMA table_info(evidence_derivation_queue)")
            }
            if "umo" not in queue_columns:
                db.execute(
                    "ALTER TABLE evidence_derivation_queue ADD COLUMN umo TEXT NOT NULL DEFAULT ''"
                )
                db.execute("""UPDATE evidence_derivation_queue SET umo=COALESCE(
                    json_extract(source_locator_json,'$.scope.umo'),'') WHERE umo=''""")
            if "projection_key" not in queue_columns:
                db.execute(
                    "ALTER TABLE evidence_derivation_queue ADD COLUMN projection_key TEXT NOT NULL DEFAULT ''"
                )
            if "text_hash" not in queue_columns:
                db.execute(
                    "ALTER TABLE evidence_derivation_queue ADD COLUMN text_hash TEXT NOT NULL DEFAULT ''"
                )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_queue_due ON evidence_derivation_queue(status,available_at,created_at)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_queue_scope ON evidence_derivation_queue(scope_key,status,available_at)"
            )
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_queue_admissions (
                admission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_key TEXT NOT NULL, material_id TEXT NOT NULL, admitted_at REAL NOT NULL
            )""")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidence_admissions_scope_time ON evidence_queue_admissions(scope_key,admitted_at)"
            )
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_queue_cursor (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), last_scope_key TEXT NOT NULL DEFAULT ''
            )""")
            db.execute(
                "INSERT OR IGNORE INTO evidence_queue_cursor(singleton,last_scope_key) VALUES (1,'')"
            )
            if not self._retry_extension_columns_valid(db):
                raise RuntimeError(
                    "Evidence retry metadata columns do not match the expected contract"
                )
            db.execute(
                "UPDATE evidence_schema SET retry_extension_version=? WHERE singleton=1",
                (RETRY_EXTENSION_VERSION,),
            )
            db.execute(
                "UPDATE evidence_schema SET schema_version=? WHERE singleton=1",
                (SCHEMA_VERSION,),
            )
            db.commit()
            self._load_revision_cache(db)
            with self._revision_lock:
                self._ready = True

    def _backup_before_schema_upgrade(self) -> None:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return
        uri = self.path.resolve().as_uri() + "?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, timeout=2) as source:
                tables = {
                    row[0]
                    for row in source.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                current = None
                if "evidence_schema" in tables:
                    row = source.execute(
                        "SELECT schema_version FROM evidence_schema WHERE singleton=1"
                    ).fetchone()
                    current = int(row[0]) if row else 0
                material_columns = set()
                if "evidence_materials" in tables:
                    material_columns = {
                        str(row[1])
                        for row in source.execute(
                            "PRAGMA table_info(evidence_materials)"
                        ).fetchall()
                    }
                marker_columns = set()
                marker_version = 0
                if "evidence_schema" in tables:
                    marker_columns = {
                        str(row[1])
                        for row in source.execute(
                            "PRAGMA table_info(evidence_schema)"
                        ).fetchall()
                    }
                    if "retry_extension_version" in marker_columns:
                        marker = source.execute(
                            "SELECT retry_extension_version FROM evidence_schema WHERE singleton=1"
                        ).fetchone()
                        marker_version = int(marker[0]) if marker else 0
                columns_valid = self._retry_extension_contract_from_columns(
                    source, material_columns
                )
                if (
                    current is not None
                    and current >= SCHEMA_VERSION
                    and columns_valid
                    and "retry_extension_version" in marker_columns
                    and marker_version >= RETRY_EXTENSION_VERSION
                ):
                    return
                suffix = "none" if current is None else str(current)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup_path = self.path.with_name(
                    f"{self.path.name}.schema-v{suffix}-{stamp}.bak"
                )
                if backup_path.exists():
                    raise RuntimeError(
                        "Evidence sidecar migration backup already exists"
                    )
                with sqlite3.connect(str(backup_path), timeout=2) as backup:
                    source.backup(backup)
                    if backup.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        backup.close()
                        backup_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            "Evidence sidecar migration backup failed validation"
                        )
        except sqlite3.DatabaseError:
            raise RuntimeError(
                "Existing evidence sidecar is not a readable SQLite database"
            ) from None

    @staticmethod
    def _retry_extension_contract_from_columns(
        connection, material_columns: set[str]
    ) -> bool:
        expected = {
            "attempt_count": ("INTEGER", 1, "0"),
            "next_attempt_at": ("REAL", 1, "0"),
            "outcome_code": ("TEXT", 1, "''"),
        }
        info = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(evidence_materials)")
        }
        if not set(expected) <= material_columns:
            return False
        for name, (column_type, not_null, default) in expected.items():
            row = info.get(name)
            if (
                row is None
                or str(row[2]).upper() != column_type
                or int(row[3]) != not_null
                or str(row[4]) != default
            ):
                return False
        schema_info = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(evidence_schema)")
        }
        marker = schema_info.get("retry_extension_version")
        return bool(
            marker
            and str(marker[2]).upper() == "INTEGER"
            and int(marker[3]) == 1
            and str(marker[4]) == "0"
        )

    @classmethod
    def _retry_extension_columns_valid(cls, db) -> bool:
        columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(evidence_materials)")
        }
        return cls._retry_extension_contract_from_columns(db, columns)

    def _load_revision_cache(self, db) -> None:
        values: dict[str, dict] = {}
        for row in db.execute(
            "SELECT scope_key,correction_revision FROM evidence_scope_revisions"
        ):
            scope_key = str(row[0])
            values[scope_key] = self._revision_projection(db, scope_key, int(row[1]))
        with self._revision_lock:
            self._revision_cache = values

    def _revision_projection(self, db, scope_key: str, revision: int) -> dict:
        rows = db.execute(
            "SELECT correction_revision,target_id,action,replacement_atom_id,affected_ids_json "
            "FROM evidence_corrections WHERE scope_key=? ORDER BY correction_revision DESC",
            (scope_key,),
        ).fetchall()
        revocations = []
        for row in rows:
            revocations.append(
                {
                    "revision": int(row[0]),
                    "target_id": str(row[1]),
                    "action": str(row[2]),
                    "replacement_atom_id": row[3],
                    "affected_ids": json.loads(row[4]),
                }
            )
        return {
            "schema_version": 1,
            "available": True,
            "scope_key": scope_key,
            "correction_revision": revision,
            "revocations": revocations,
        }

    def peek_evidence_revision(self, scope_key: str) -> dict:
        unavailable = {
            "schema_version": 1,
            "available": False,
            "scope_key": scope_key,
            "correction_revision": None,
            "revocations": [],
        }
        # A correction commit and cache publication form one critical section.
        # Fail closed during it instead of blocking the event loop or exposing
        # the pre-correction revision after a durable correction has committed.
        if not self._revision_lock.acquire(blocking=False):
            return unavailable
        try:
            if not self._ready:
                return unavailable
            value = self._revision_cache.get(scope_key)
            if value is None:
                return {
                    "schema_version": 1,
                    "available": True,
                    "scope_key": scope_key,
                    "correction_revision": 0,
                    "revocations": [],
                }
            if not value["revocations"]:
                return {**value, "revocations": []}
            return copy.deepcopy(value)
        finally:
            self._revision_lock.release()

    async def upsert_material(self, descriptor: dict) -> str:
        return await asyncio.to_thread(self._upsert_material_sync, descriptor)

    async def has_ready_family(self, descriptor: dict) -> bool:
        return await asyncio.to_thread(self._has_ready_family_sync, descriptor)

    def _has_ready_family_sync(self, descriptor: dict) -> bool:
        family_id = str(descriptor.get("source_family_id", ""))
        if not family_id:
            return False
        with self._lock, self._session() as db:
            row = db.execute(
                "SELECT 1 FROM evidence_materials WHERE scope_key=? AND source_family_id=? "
                "AND text_hash=? AND projection_key=? AND status='ready' LIMIT 1",
                (
                    descriptor["scope_key"],
                    family_id,
                    descriptor["text_hash"],
                    descriptor["projection_key"],
                ),
            ).fetchone()
            return row is not None

    def _upsert_material_sync(self, descriptor: dict) -> str:
        now = time.time()
        with self._lock, self._session() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO evidence_materials(
                material_id,scope_key,material_kind,role,conversation_id,snapshot_fingerprint,
                source_position,source_family_id,text_hash,projection_key,card_version,status,diagnostics_code,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'',?,?)
            ON CONFLICT(material_id) DO UPDATE SET
                material_kind=excluded.material_kind,role=excluded.role,
                conversation_id=excluded.conversation_id,snapshot_fingerprint=excluded.snapshot_fingerprint,
                source_position=excluded.source_position,text_hash=excluded.text_hash,
                source_family_id=excluded.source_family_id,
                projection_key=excluded.projection_key,
                card_version=excluded.card_version,
                status=CASE WHEN evidence_materials.status IN ('invalidated','terminal')
                            THEN evidence_materials.status
                            WHEN evidence_materials.text_hash=excluded.text_hash
                                  AND evidence_materials.projection_key=excluded.projection_key
                                  AND evidence_materials.status IN ('ready','failed','deferred')
                            THEN evidence_materials.status ELSE 'queued' END,
                attempt_count=CASE WHEN evidence_materials.text_hash=excluded.text_hash
                                        AND evidence_materials.projection_key=excluded.projection_key
                                   THEN evidence_materials.attempt_count ELSE 0 END,
                next_attempt_at=CASE WHEN evidence_materials.text_hash=excluded.text_hash
                                          AND evidence_materials.projection_key=excluded.projection_key
                                     THEN evidence_materials.next_attempt_at ELSE 0 END,
                outcome_code=CASE WHEN evidence_materials.text_hash=excluded.text_hash
                                       AND evidence_materials.projection_key=excluded.projection_key
                                  THEN evidence_materials.outcome_code ELSE '' END,
                diagnostics_code=CASE WHEN evidence_materials.text_hash=excluded.text_hash
                                           AND evidence_materials.projection_key=excluded.projection_key
                                      THEN evidence_materials.diagnostics_code ELSE '' END,
                updated_at=excluded.updated_at""",
                (
                    descriptor["material_id"],
                    descriptor["scope_key"],
                    descriptor["material_kind"],
                    descriptor["role"],
                    descriptor["conversation_id"],
                    descriptor["snapshot_fingerprint"],
                    int(descriptor["position"]),
                    descriptor.get("source_family_id", ""),
                    descriptor["text_hash"],
                    descriptor["projection_key"],
                    descriptor["card_version"],
                    "queued",
                    now,
                    now,
                ),
            )
            row = db.execute(
                "SELECT status FROM evidence_materials WHERE material_id=?",
                (descriptor["material_id"],),
            ).fetchone()
            db.commit()
            return str(row[0]) if row else "queued"

    async def material_retry_state(self, material_id: str, scope_key: str) -> dict:
        return await asyncio.to_thread(
            self._material_retry_state_sync, material_id, scope_key
        )

    def _material_retry_state_sync(self, material_id: str, scope_key: str) -> dict:
        with self._lock, self._session() as db:
            row = db.execute(
                "SELECT status,attempt_count,next_attempt_at,outcome_code FROM evidence_materials "
                "WHERE material_id=? AND scope_key=?",
                (material_id, scope_key),
            ).fetchone()
        if not row:
            return {
                "status": "missing",
                "attempt_count": 0,
                "next_attempt_at": 0.0,
                "outcome_code": "",
            }
        return {
            "status": str(row[0]),
            "attempt_count": int(row[1]),
            "next_attempt_at": float(row[2]),
            "outcome_code": str(row[3]),
        }

    async def material_retry_due(
        self, material_id: str, scope_key: str, now: float | None = None
    ) -> bool:
        state = await self.material_retry_state(material_id, scope_key)
        status, outcome = state["status"], state["outcome_code"]
        if status in {"ready", "invalidated", "terminal"} or outcome in {
            "policy_refusal",
            "source_unavailable",
        }:
            return False
        if (
            outcome == "invalid_schema"
            and state["attempt_count"] >= MAX_INVALID_SCHEMA_ATTEMPTS
        ):
            return False
        if status == "deferred" and outcome == "bounded_queue":
            return True
        now_value = time.time() if now is None else now
        return (
            status in {"queued", "failed", "deferred"}
            and state["next_attempt_at"] <= now_value
        )

    async def admit_material_job(
        self,
        descriptor: dict,
        now: float,
        queue_size: int = 128,
        owner_ready: bool = True,
    ) -> str:
        return await asyncio.to_thread(
            self._admit_material_job_sync, descriptor, now, queue_size, owner_ready
        )

    def _admit_material_job_sync(
        self, descriptor: dict, now: float, queue_size: int, owner_ready: bool = True
    ) -> str:
        material_id = str(descriptor.get("material_id", ""))
        scope_key = str(descriptor.get("scope_key", ""))
        locator = descriptor.get("source_locator")
        if (
            not material_id
            or not scope_key
            or not isinstance(locator, dict)
            or not locator.get("kind")
        ):
            return "invalid_locator"
        source_scope = (
            locator.get("scope") if isinstance(locator.get("scope"), dict) else {}
        )
        umo = str(source_scope.get("umo") or "")
        locator_json = _json(locator)
        capacity = max(1, int(queue_size))
        with self._lock, self._session() as db:
            db.execute("BEGIN IMMEDIATE")
            material = db.execute(
                "SELECT status,attempt_count,next_attempt_at,outcome_code,text_hash,projection_key "
                "FROM evidence_materials WHERE material_id=? AND scope_key=?",
                (material_id, scope_key),
            ).fetchone()
            if not material:
                db.rollback()
                return "missing_material"
            status, attempts, retry_at, outcome = (
                str(material[0]),
                int(material[1]),
                float(material[2]),
                str(material[3]),
            )
            if (
                status in {"ready", "invalidated", "terminal"}
                or outcome in {"policy_refusal", "source_unavailable"}
                or (
                    outcome == "invalid_schema"
                    and attempts >= MAX_INVALID_SCHEMA_ATTEMPTS
                )
            ):
                db.execute(
                    "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                    (material_id, scope_key),
                )
                db.commit()
                return "terminal"
            existing = db.execute(
                "SELECT status,available_at,lease_id,lease_until,source_locator_json,correction_revision,projection_key,text_hash "
                "FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                (material_id, scope_key),
            ).fetchone()
            if existing:
                old_status, old_available, old_lease, old_lease_until = (
                    str(existing[0]),
                    float(existing[1]),
                    str(existing[2]),
                    float(existing[3]),
                )
                same_source = (
                    str(existing[6]) == str(descriptor.get("projection_key", ""))
                    and str(existing[7]) == str(descriptor.get("text_hash", ""))
                    and int(existing[5])
                    == int(descriptor.get("correction_revision", 0))
                )
                if (
                    same_source
                    and old_status == "leased"
                    and old_lease
                    and old_lease_until > now
                ):
                    db.commit()
                    return "duplicate"
                if (
                    same_source
                    and not owner_ready
                    and old_status in {"queued", "deferred", "owner_wait"}
                ):
                    db.execute(
                        "UPDATE evidence_derivation_queue SET status='owner_wait',lease_id='',lease_until=0,updated_at=? "
                        "WHERE material_id=? AND scope_key=?",
                        (now, material_id, scope_key),
                    )
                    db.execute(
                        "UPDATE evidence_materials SET status='deferred',diagnostics_code='policy_unavailable',updated_at=? "
                        "WHERE material_id=? AND scope_key=? AND status NOT IN ('ready','invalidated','terminal')",
                        (now, material_id, scope_key),
                    )
                    db.commit()
                    return "owner_wait"
                if same_source and old_status == "queued":
                    db.commit()
                    return "duplicate"
                if same_source and old_status == "owner_wait":
                    db.commit()
                    return "deferred"
                if same_source and old_status == "deferred" and old_available > now:
                    db.commit()
                    return "deferred"
            if status == "failed" and retry_at > now:
                db.execute(
                    """INSERT INTO evidence_derivation_queue(
                    material_id,scope_key,umo,source_locator_json,correction_revision,projection_key,text_hash,status,available_at,lease_id,lease_until,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,'deferred',?,'',0,?,?)
                ON CONFLICT(material_id) DO UPDATE SET status='deferred',available_at=excluded.available_at,
                    umo=excluded.umo,source_locator_json=excluded.source_locator_json,correction_revision=excluded.correction_revision,
                    projection_key=excluded.projection_key,text_hash=excluded.text_hash,
                    lease_id='',lease_until=0,updated_at=excluded.updated_at""",
                    (
                        material_id,
                        scope_key,
                        umo,
                        locator_json,
                        int(descriptor.get("correction_revision", 0)),
                        str(descriptor.get("projection_key", "")),
                        str(descriptor.get("text_hash", "")),
                        retry_at,
                        now,
                        now,
                    ),
                )
                db.commit()
                return "deferred"
            active = db.execute(
                "SELECT COUNT(*) FROM evidence_derivation_queue WHERE status IN ('queued','leased')"
            ).fetchone()[0]
            next_status = (
                "owner_wait"
                if not owner_ready
                else "queued"
                if int(active) < capacity
                else "deferred"
            )
            available_at = now if next_status == "queued" else max(now, retry_at)
            db.execute(
                """INSERT INTO evidence_derivation_queue(
                material_id,scope_key,umo,source_locator_json,correction_revision,projection_key,text_hash,status,available_at,lease_id,lease_until,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,'',0,?,?)
            ON CONFLICT(material_id) DO UPDATE SET scope_key=excluded.scope_key,
                umo=excluded.umo,source_locator_json=excluded.source_locator_json,correction_revision=excluded.correction_revision,
                projection_key=excluded.projection_key,text_hash=excluded.text_hash,
                status=excluded.status,available_at=excluded.available_at,lease_id='',lease_until=0,updated_at=excluded.updated_at""",
                (
                    material_id,
                    scope_key,
                    umo,
                    locator_json,
                    int(descriptor.get("correction_revision", 0)),
                    str(descriptor.get("projection_key", "")),
                    str(descriptor.get("text_hash", "")),
                    next_status,
                    available_at,
                    now,
                    now,
                ),
            )
            db.execute(
                "UPDATE evidence_materials SET status=?,diagnostics_code=?,updated_at=? WHERE material_id=? AND scope_key=?",
                (
                    "queued" if next_status == "queued" else "deferred",
                    ""
                    if next_status == "queued"
                    else "policy_unavailable"
                    if next_status == "owner_wait"
                    else "bounded_queue",
                    now,
                    material_id,
                    scope_key,
                ),
            )
            db.commit()
            return next_status

    async def claim_due_materials(
        self,
        now: float,
        queue_size: int = 128,
        per_scope_hourly_limit: int = 24,
        lease_seconds: float = 120,
        batch_size: int = 4,
    ) -> list[dict]:
        return await asyncio.to_thread(
            self._claim_due_materials_sync,
            now,
            queue_size,
            per_scope_hourly_limit,
            lease_seconds,
            batch_size,
        )

    def _claim_due_materials_sync(
        self,
        now: float,
        queue_size: int,
        per_scope_hourly_limit: int,
        lease_seconds: float,
        batch_size: int,
    ) -> list[dict]:
        claimed = []
        with self._lock, self._session() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "DELETE FROM evidence_queue_admissions WHERE admitted_at<=?",
                (now - 86400,),
            )
            db.execute(
                "UPDATE evidence_derivation_queue SET status='queued',lease_id='',lease_until=0,available_at=?,updated_at=? "
                "WHERE status='leased' AND lease_until<=?",
                (now, now, now),
            )
            # Queue payloads can contain large source locators. Keep only the
            # small scope inventory, then fetch one eligible head per scope.
            # This preserves the round-robin order without materializing the
            # entire deferred backlog on every four-item claim.
            scopes = [
                str(row[0])
                for row in db.execute(
                    "SELECT DISTINCT q.scope_key FROM evidence_derivation_queue q "
                    "JOIN evidence_materials m ON m.material_id=q.material_id AND m.scope_key=q.scope_key "
                    "WHERE q.status IN ('queued','deferred') AND q.available_at<=? ORDER BY q.scope_key",
                    (now,),
                )
            ]
            head_sql = (
                "SELECT q.material_id,q.scope_key,q.source_locator_json,q.correction_revision,q.status,q.available_at, "
                "m.material_kind,m.role,m.conversation_id,m.snapshot_fingerprint,m.source_position,m.source_family_id, "
                "m.text_hash,m.projection_key,m.card_version,m.status,m.attempt_count,m.next_attempt_at,m.outcome_code, "
                "COALESCE(r.correction_revision,0),q.projection_key,q.text_hash "
                "FROM evidence_derivation_queue q "
                "JOIN evidence_materials m ON m.material_id=q.material_id AND m.scope_key=q.scope_key "
                "LEFT JOIN evidence_scope_revisions r ON r.scope_key=q.scope_key "
                "WHERE q.scope_key=? AND q.available_at<=? "
            )
            cursor = db.execute(
                "SELECT last_scope_key FROM evidence_queue_cursor WHERE singleton=1"
            ).fetchone()
            last_scope = str(cursor[0]) if cursor else ""
            if scopes and last_scope in scopes:
                pivot = scopes.index(last_scope) + 1
                scopes = scopes[pivot:] + scopes[:pivot]
            active_count = int(
                db.execute(
                    "SELECT COUNT(*) FROM evidence_derivation_queue WHERE status IN ('queued','leased')"
                ).fetchone()[0]
            )
            seen_scopes = set()
            for scope_id in scopes:
                if len(claimed) >= max(1, int(batch_size)):
                    break
                if scope_id in seen_scopes:
                    continue
                # A deferred head item must not block work that already owns a
                # queue slot in this scope. Prefer queued work while capacity
                # is full; once a slot opens, the oldest eligible row wins.
                capacity = max(1, int(queue_size))
                status_sql = (
                    "AND q.status='queued' "
                    if active_count >= capacity
                    else "AND q.status IN ('queued','deferred') "
                )
                due = db.execute(
                    head_sql
                    + status_sql
                    + "ORDER BY q.created_at,q.material_id LIMIT 1",
                    (scope_id, now),
                ).fetchone()
                if due is None:
                    continue
                if str(due[4]) == "deferred" and active_count >= capacity:
                    continue
                material_id = str(due[0])
                material_status, attempts, retry_at, outcome = (
                    str(due[15]),
                    int(due[16]),
                    float(due[17]),
                    str(due[18]),
                )
                if (
                    material_status in {"ready", "invalidated", "terminal"}
                    or outcome in {"policy_refusal", "source_unavailable"}
                    or (
                        outcome == "invalid_schema"
                        and attempts >= MAX_INVALID_SCHEMA_ATTEMPTS
                    )
                ):
                    db.execute(
                        "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                        (material_id, scope_id),
                    )
                    continue
                if int(due[3]) != int(due[19]):
                    db.execute(
                        "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                        (material_id, scope_id),
                    )
                    db.execute(
                        "UPDATE evidence_materials SET status='deferred',diagnostics_code='revision_changed',updated_at=? "
                        "WHERE material_id=? AND scope_key=?",
                        (now, material_id, scope_id),
                    )
                    continue
                if str(due[20]) != str(due[13]) or str(due[21]) != str(due[12]):
                    db.execute(
                        "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                        (material_id, scope_id),
                    )
                    db.execute(
                        "UPDATE evidence_materials SET status='deferred',diagnostics_code='projection_changed',updated_at=? "
                        "WHERE material_id=? AND scope_key=?",
                        (now, material_id, scope_id),
                    )
                    continue
                if material_status == "failed" and retry_at > now:
                    db.execute(
                        "UPDATE evidence_derivation_queue SET status='deferred',available_at=?,updated_at=? "
                        "WHERE material_id=? AND scope_key=?",
                        (retry_at, now, material_id, scope_id),
                    )
                    continue
                admitted = db.execute(
                    "SELECT COUNT(*),MIN(admitted_at) FROM evidence_queue_admissions WHERE scope_key=? AND admitted_at>?",
                    (scope_id, now - 3600),
                ).fetchone()
                if int(admitted[0]) >= max(1, int(per_scope_hourly_limit)):
                    available_at = (
                        float(admitted[1]) + 3600
                        if admitted[1] is not None
                        else now + 3600
                    )
                    db.execute(
                        "UPDATE evidence_derivation_queue SET status='deferred',available_at=?,updated_at=? "
                        "WHERE scope_key=? AND status IN ('queued','deferred') AND available_at<=? "
                        "AND EXISTS (SELECT 1 FROM evidence_materials m WHERE "
                        "m.material_id=evidence_derivation_queue.material_id AND m.scope_key=evidence_derivation_queue.scope_key)",
                        (available_at, now, scope_id, now),
                    )
                    db.execute(
                        "UPDATE evidence_materials SET status='deferred',diagnostics_code='scope_hourly_limit',updated_at=? "
                        "WHERE scope_key=? AND material_id IN (SELECT material_id FROM evidence_derivation_queue WHERE scope_key=?)",
                        (now, scope_id, scope_id),
                    )
                    continue
                locator_text = str(due[2] or "")
                try:
                    locator = json.loads(locator_text)
                except (TypeError, ValueError):
                    locator = None
                if not isinstance(locator, dict) or not locator.get("kind"):
                    db.execute(
                        "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                        (material_id, scope_id),
                    )
                    db.execute(
                        "UPDATE evidence_materials SET status='failed',diagnostics_code='source_unavailable',outcome_code='source_unavailable',updated_at=? "
                        "WHERE material_id=? AND scope_key=?",
                        (now, material_id, scope_id),
                    )
                    continue
                lease_id = uuid.uuid4().hex
                lease_until = now + max(5.0, float(lease_seconds))
                db.execute(
                    "UPDATE evidence_derivation_queue SET status='leased',lease_id=?,lease_until=?,updated_at=? "
                    "WHERE material_id=? AND scope_key=? AND status IN ('queued','deferred')",
                    (lease_id, lease_until, now, material_id, scope_id),
                )
                db.execute(
                    "INSERT INTO evidence_queue_admissions(scope_key,material_id,admitted_at) VALUES (?,?,?)",
                    (scope_id, material_id, now),
                )
                db.execute(
                    "UPDATE evidence_queue_cursor SET last_scope_key=? WHERE singleton=1",
                    (scope_id,),
                )
                if str(due[4]) == "deferred":
                    active_count += 1
                seen_scopes.add(scope_id)
                claimed.append(
                    {
                        "material_id": material_id,
                        "scope_key": scope_id,
                        "source_locator": locator,
                        "correction_revision": int(due[3]),
                        "material_kind": str(due[6]),
                        "role": str(due[7]),
                        "conversation_id": str(due[8]),
                        "snapshot_fingerprint": str(due[9]),
                        "position": int(due[10]),
                        "source_family_id": str(due[11]),
                        "text_hash": str(due[12]),
                        "projection_key": str(due[13]),
                        "card_version": str(due[14]),
                        "lease_id": lease_id,
                        "lease_until": lease_until,
                    }
                )
            db.commit()
        return claimed

    async def renew_material_lease(
        self,
        material_id: str,
        scope_key: str,
        lease_id: str,
        now: float,
        lease_seconds: float,
    ) -> bool:
        return await asyncio.to_thread(
            self._renew_material_lease_sync,
            material_id,
            scope_key,
            lease_id,
            now,
            lease_seconds,
        )

    def _renew_material_lease_sync(
        self,
        material_id: str,
        scope_key: str,
        lease_id: str,
        now: float,
        lease_seconds: float,
    ) -> bool:
        with self._lock, self._session() as db:
            cursor = db.execute(
                "UPDATE evidence_derivation_queue SET lease_until=?,updated_at=? "
                "WHERE material_id=? AND scope_key=? AND status='leased' AND lease_id=? AND lease_until>?",
                (
                    now + max(5.0, float(lease_seconds)),
                    now,
                    material_id,
                    scope_key,
                    lease_id,
                    now,
                ),
            )
            db.commit()
            return cursor.rowcount == 1

    async def release_material_lease(
        self, material_id: str, scope_key: str, lease_id: str, now: float
    ) -> bool:
        return await asyncio.to_thread(
            self._release_material_lease_sync, material_id, scope_key, lease_id, now
        )

    def _release_material_lease_sync(
        self, material_id: str, scope_key: str, lease_id: str, now: float
    ) -> bool:
        with self._lock, self._session() as db:
            cursor = db.execute(
                "UPDATE evidence_derivation_queue SET status='queued',available_at=?,lease_id='',lease_until=0,updated_at=? "
                "WHERE material_id=? AND scope_key=? AND status='leased' AND lease_id=? AND lease_until>?",
                (now, now, material_id, scope_key, lease_id, now),
            )
            db.commit()
            return cursor.rowcount == 1

    async def due_policy_umos(self, now: float) -> list[str]:
        return await asyncio.to_thread(self._due_policy_umos_sync, now)

    def _due_policy_umos_sync(self, now: float) -> list[str]:
        with self._lock, self._session() as db:
            rows = db.execute(
                "SELECT DISTINCT umo FROM evidence_derivation_queue "
                "WHERE status IN ('queued','deferred') AND available_at<=? AND umo<>'' "
                "ORDER BY umo",
                (now,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    async def pause_policy_jobs(self, umo: str, now: float) -> int:
        return await asyncio.to_thread(self._pause_policy_jobs_sync, umo, now)

    def _pause_policy_jobs_sync(self, umo: str, now: float) -> int:
        with self._lock, self._session() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE evidence_materials SET status='deferred',diagnostics_code='policy_unavailable',"
                "updated_at=? WHERE status NOT IN ('ready','invalidated','terminal') "
                "AND material_id IN (SELECT material_id FROM evidence_derivation_queue "
                "WHERE umo=? AND status IN ('queued','deferred'))",
                (now, umo),
            )
            cursor = db.execute(
                "UPDATE evidence_derivation_queue SET status='owner_wait',lease_id='',lease_until=0,updated_at=? "
                "WHERE umo=? AND status IN ('queued','deferred')",
                (now, umo),
            )
            db.commit()
            return max(0, int(cursor.rowcount))

    async def wake_policy_deferred(self, umo: str, now: float) -> int:
        # Most trusted-owner observations have no waiting jobs. A WAL read
        # avoids acquiring the contended writer lock for two no-op UPDATEs.
        if not await self._read(self._has_policy_waiters_sync, umo):
            return 0
        return await asyncio.to_thread(self._wake_policy_deferred_sync, umo, now)

    def _has_policy_waiters_sync(self, umo: str) -> bool:
        with self._read_session() as db:
            return bool(
                self._read_rows(
                    db,
                    "SELECT 1 FROM evidence_derivation_queue WHERE umo=? AND status='owner_wait' LIMIT 1",
                    (umo,),
                )
            )

    def _wake_policy_deferred_sync(self, umo: str, now: float) -> int:
        with self._lock, self._session() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE evidence_materials SET next_attempt_at=CASE WHEN outcome_code='policy_unavailable' "
                "THEN ? ELSE next_attempt_at END,diagnostics_code='',updated_at=? "
                "WHERE status='deferred' AND diagnostics_code='policy_unavailable' AND material_id IN "
                "(SELECT material_id FROM evidence_derivation_queue WHERE umo=? AND status='owner_wait')",
                (now, now, umo),
            )
            cursor = db.execute(
                "UPDATE evidence_derivation_queue SET status='deferred',available_at=MAX(?,COALESCE(("
                "SELECT next_attempt_at FROM evidence_materials WHERE evidence_materials.material_id="
                "evidence_derivation_queue.material_id),0)),updated_at=? "
                "WHERE umo=? AND status='owner_wait'",
                (now, now, umo),
            )
            db.commit()
            return max(0, int(cursor.rowcount))

    async def record_material_failure(
        self,
        material_id: str,
        scope_key: str,
        outcome_code: str,
        lease_id: str | None = None,
        now: float | None = None,
        admission_spent: bool = False,
    ) -> None:
        await asyncio.to_thread(
            self._record_material_failure_sync,
            material_id,
            scope_key,
            outcome_code,
            lease_id,
            now,
            admission_spent,
        )

    def _record_material_failure_sync(
        self,
        material_id: str,
        scope_key: str,
        outcome_code: str,
        lease_id: str | None = None,
        now: float | None = None,
        admission_spent: bool = False,
    ) -> None:
        allowed = {
            "timeout",
            "policy_refusal",
            "policy_unavailable",
            "transport_failure",
            "invalid_schema",
            "cancelled",
            "provider_unavailable",
            "source_unavailable",
        }
        code = outcome_code if outcome_code in allowed else "transport_failure"
        now = time.time() if now is None else float(now)
        with self._lock, self._session() as db:
            db.execute("BEGIN IMMEDIATE")
            if lease_id is not None:
                lease = db.execute(
                    "SELECT 1 FROM evidence_derivation_queue WHERE material_id=? AND scope_key=? "
                    "AND status='leased' AND lease_id=? AND lease_until>?",
                    (material_id, scope_key, lease_id, now),
                ).fetchone()
                if not lease:
                    db.rollback()
                    return
            row = db.execute(
                "SELECT attempt_count FROM evidence_materials WHERE material_id=? AND scope_key=?",
                (material_id, scope_key),
            ).fetchone()
            if not row:
                db.commit()
                return
            attempts = int(row[0]) + (
                0
                if code
                in {
                    "cancelled",
                    "provider_unavailable",
                    "policy_unavailable",
                    "source_unavailable",
                }
                else 1
            )
            if code in {"policy_refusal", "source_unavailable"}:
                retry_at = 0.0
            elif code == "cancelled":
                retry_at = now
            elif code == "provider_unavailable":
                retry_at = now + RETRY_DELAYS_SECONDS[0]
            elif code == "policy_unavailable":
                retry_at = now + RETRY_DELAYS_SECONDS[0]
            elif code == "invalid_schema" and attempts >= MAX_INVALID_SCHEMA_ATTEMPTS:
                retry_at = 0.0
            else:
                retry_at = (
                    now
                    + RETRY_DELAYS_SECONDS[
                        min(max(attempts - 1, 0), len(RETRY_DELAYS_SECONDS) - 1)
                    ]
                )
            next_status = (
                "deferred"
                if code in {"cancelled", "provider_unavailable", "policy_unavailable"}
                else "failed"
            )
            db.execute(
                "UPDATE evidence_materials SET status=?,attempt_count=?,next_attempt_at=?,"
                "outcome_code=?,diagnostics_code=?,updated_at=? WHERE material_id=? AND scope_key=?",
                (
                    next_status,
                    attempts,
                    retry_at,
                    code,
                    code,
                    now,
                    material_id,
                    scope_key,
                ),
            )
            terminal = code in {"policy_refusal", "source_unavailable"} or (
                code == "invalid_schema" and attempts >= MAX_INVALID_SCHEMA_ATTEMPTS
            )
            if code == "policy_unavailable" and not admission_spent:
                db.execute(
                    "DELETE FROM evidence_queue_admissions WHERE admission_id=("
                    "SELECT MAX(admission_id) FROM evidence_queue_admissions "
                    "WHERE scope_key=? AND material_id=?)",
                    (scope_key, material_id),
                )
            if terminal:
                db.execute(
                    "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                    (material_id, scope_key),
                )
            else:
                db.execute(
                    "UPDATE evidence_derivation_queue SET status=?,available_at=?,lease_id='',lease_until=0,updated_at=? "
                    "WHERE material_id=? AND scope_key=?",
                    (
                        "queued"
                        if code == "cancelled"
                        else "owner_wait"
                        if code == "policy_unavailable"
                        else "deferred",
                        retry_at,
                        now,
                        material_id,
                        scope_key,
                    ),
                )
            db.commit()

    async def get_materials(
        self,
        scope_key: str,
        material_ids: list[str],
        family_ids: list[str] | None = None,
    ) -> dict[str, dict]:
        return await self._read(
            self._get_materials_sync, scope_key, material_ids, family_ids or []
        )

    async def get_material_descriptors(
        self, scope_key: str, material_ids: list[str]
    ) -> dict[str, dict]:
        """Read exact scoped material descriptors without returning source text.

        Ready rows no longer have a derivation-queue locator. For event source
        rows only, the canonical ``event:<id>`` snapshot fingerprint is enough
        to reconstruct the event locator when its ev1 identity recomputes to
        the selected primary key. Other material kinds remain locator-less.
        """
        return await self._read(
            self._get_material_descriptors_sync, scope_key, material_ids
        )

    def _get_material_descriptors_sync(
        self, scope_key: str, material_ids: list[str]
    ) -> dict[str, dict]:
        if (
            not isinstance(scope_key, str)
            or not scope_key
            or not isinstance(material_ids, list)
        ):
            return {}
        ids = list(
            dict.fromkeys(
                value for value in material_ids if isinstance(value, str) and value
            )
        )[:128]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._read_session() as db:
            rows = self._read_rows(
                db,
                "SELECT m.material_id,m.scope_key,m.material_kind,m.role,m.conversation_id,"
                "m.snapshot_fingerprint,m.source_position,m.source_family_id,m.text_hash,"
                "m.projection_key,m.card_version,m.status,COALESCE(v.correction_revision,0) "
                f"FROM evidence_materials m LEFT JOIN evidence_scope_revisions v "
                f"ON v.scope_key=m.scope_key WHERE m.scope_key=? AND m.material_id IN ({placeholders})",
                (scope_key, *ids),
            )
        # Import the pure identity helper lazily to keep SQLite storage free of
        # plugin/runtime initialization and to avoid a module import cycle.
        from ..core.evidence_projection import event_evidence_id

        result = {}
        for row in rows:
            material_id = str(row[0])
            descriptor = {
                "material_id": material_id,
                "scope_key": str(row[1]),
                "material_kind": str(row[2]),
                "role": str(row[3]),
                "conversation_id": str(row[4]),
                "snapshot_fingerprint": str(row[5]),
                "position": int(row[6]),
                "source_family_id": str(row[7]),
                "text_hash": str(row[8]),
                "projection_key": str(row[9]),
                "card_version": str(row[10]),
                "status": str(row[11]),
                "correction_revision": int(row[12]),
                "source_locator": None,
            }
            fingerprint = descriptor["snapshot_fingerprint"]
            if (
                descriptor["material_kind"] in {"source_event", "program_receipt"}
                and fingerprint.startswith("event:")
                and len(fingerprint) > len("event:")
            ):
                event_id = fingerprint[len("event:") :]
                if event_evidence_id(scope_key, event_id) == material_id:
                    descriptor["source_locator"] = {
                        "kind": "event",
                        "event_id": event_id,
                        "scope_key": scope_key,
                    }
            result[material_id] = descriptor
        return result

    def _get_materials_sync(
        self, scope_key: str, material_ids: list[str], family_ids: list[str]
    ) -> dict[str, dict]:
        if not material_ids and not family_ids:
            return {}
        with self._read_session() as db:
            clauses = []
            params: list = [scope_key]
            if material_ids:
                clauses.append(
                    "material_id IN (" + ",".join("?" for _ in material_ids) + ")"
                )
                params.extend(material_ids)
            if family_ids:
                clauses.append(
                    "source_family_id IN (" + ",".join("?" for _ in family_ids) + ")"
                )
                params.extend(family_ids)
            match_sql = "(" + " OR ".join(clauses) + ")"
            rows = self._read_rows(
                db,
                f"SELECT material_id,status,diagnostics_code,source_family_id,text_hash,projection_key,material_kind,role,"
                f"conversation_id,snapshot_fingerprint,source_position,card_version FROM evidence_materials "
                f"WHERE scope_key=? AND {match_sql}",
                tuple(params),
            )
            result = {
                str(r[0]): {
                    "material_id": str(r[0]),
                    "status": str(r[1]),
                    "scope_key": scope_key,
                    "diagnostics_code": str(r[2]),
                    "source_family_id": str(r[3]),
                    "text_hash": str(r[4]),
                    "projection_key": str(r[5]),
                    "material_kind": str(r[6]),
                    "role": str(r[7]),
                    "conversation_id": str(r[8]),
                    "snapshot_fingerprint": str(r[9]),
                    "position": int(r[10]),
                    "card_version": str(r[11]),
                }
                for r in rows
            }
            actual_ids = list(result)
            if not actual_ids:
                return result
            placeholders = ",".join("?" for _ in actual_ids)
            atoms = self._read_rows(
                db,
                f"SELECT atom_id,material_id,kind,status,root_ids_json,root_material_ids_json,atom_json,revision FROM evidence_atoms "
                f"WHERE scope_key=? AND material_id IN ({placeholders}) ORDER BY created_at,atom_id",
                (scope_key, *actual_ids),
            )
            for r in atoms:
                result.setdefault(
                    str(r[1]), {"status": "ready", "diagnostics_code": ""}
                ).setdefault("atoms", []).append(
                    {
                        "atom_id": str(r[0]),
                        "material_id": str(r[1]),
                        "kind": str(r[2]),
                        "status": str(r[3]),
                        "root_evidence_ids": self._read_json(r[4]),
                        "root_material_ids": self._read_json(r[5]),
                        "atom": self._read_json(r[6]),
                        "revision": int(r[7]),
                    }
                )
            return result

    async def mark_material(
        self, material_id: str, scope_key: str, status: str, diagnostics_code: str = ""
    ) -> None:
        await asyncio.to_thread(
            self._mark_material_sync, material_id, scope_key, status, diagnostics_code
        )

    def _mark_material_sync(
        self, material_id: str, scope_key: str, status: str, diagnostics_code: str
    ) -> None:
        with self._lock, self._session() as db:
            if status == "deferred" and diagnostics_code == "bounded_queue":
                # Capacity deferral is not a model attempt; preserve an older
                # failure and its retry timestamp.
                db.execute(
                    "UPDATE evidence_materials SET status=CASE WHEN status='failed' THEN 'failed' ELSE 'deferred' END,"
                    "diagnostics_code=?,updated_at=? WHERE material_id=? AND scope_key=?",
                    (diagnostics_code[:48], time.time(), material_id, scope_key),
                )
                db.execute(
                    "UPDATE evidence_derivation_queue SET status='deferred',available_at=?,lease_id='',lease_until=0,updated_at=? "
                    "WHERE material_id=? AND scope_key=?",
                    (time.time(), time.time(), material_id, scope_key),
                )
            else:
                db.execute(
                    "UPDATE evidence_materials SET status=?,diagnostics_code=?,updated_at=? WHERE material_id=? AND scope_key=?",
                    (
                        status,
                        diagnostics_code[:48],
                        time.time(),
                        material_id,
                        scope_key,
                    ),
                )
            db.commit()

    async def store_atoms(
        self,
        descriptor: dict,
        atoms: list[dict],
        lease_id: str | None = None,
        now: float | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._store_atoms_sync, descriptor, atoms, lease_id, now
        )

    def _store_atoms_sync(
        self,
        descriptor: dict,
        atoms: list[dict],
        lease_id: str | None = None,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else float(now)
        with self._lock, self._session() as db:
            db.execute("BEGIN IMMEDIATE")
            if lease_id is not None:
                lease = db.execute(
                    "SELECT 1 FROM evidence_derivation_queue WHERE material_id=? AND scope_key=? "
                    "AND status='leased' AND lease_id=? AND lease_until>?",
                    (descriptor["material_id"], descriptor["scope_key"], lease_id, now),
                ).fetchone()
                if not lease:
                    db.rollback()
                    return False
            current = db.execute(
                "SELECT status,text_hash,projection_key FROM evidence_materials "
                "WHERE material_id=? AND scope_key=?",
                (descriptor["material_id"], descriptor["scope_key"]),
            ).fetchone()
            if current and (
                str(current[0]) == "invalidated"
                or str(current[1]) != str(descriptor["text_hash"])
                or str(current[2]) != str(descriptor["projection_key"])
            ):
                db.rollback()
                return False
            revision_row = db.execute(
                "SELECT correction_revision FROM evidence_scope_revisions WHERE scope_key=?",
                (descriptor["scope_key"],),
            ).fetchone()
            current_revision = int(revision_row[0]) if revision_row else 0
            if current_revision != int(descriptor.get("correction_revision", 0)):
                db.execute(
                    "UPDATE evidence_materials SET status='deferred',diagnostics_code='revision_changed',updated_at=? "
                    "WHERE material_id=? AND scope_key=?",
                    (now, descriptor["material_id"], descriptor["scope_key"]),
                )
                db.execute(
                    "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                    (descriptor["material_id"], descriptor["scope_key"]),
                )
                db.commit()
                return False
            db.execute(
                """INSERT INTO evidence_materials(
                material_id,scope_key,material_kind,role,conversation_id,snapshot_fingerprint,
                source_position,source_family_id,text_hash,projection_key,card_version,status,diagnostics_code,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,'ready','',?,?)
            ON CONFLICT(material_id) DO UPDATE SET status='ready',diagnostics_code='',
                attempt_count=0,next_attempt_at=0,outcome_code='',updated_at=excluded.updated_at""",
                (
                    descriptor["material_id"],
                    descriptor["scope_key"],
                    descriptor["material_kind"],
                    descriptor["role"],
                    descriptor["conversation_id"],
                    descriptor["snapshot_fingerprint"],
                    int(descriptor["position"]),
                    descriptor.get("source_family_id", ""),
                    descriptor["text_hash"],
                    descriptor["projection_key"],
                    descriptor["card_version"],
                    now,
                    now,
                ),
            )
            corrected = db.execute(
                "SELECT a.root_ids_json,a.atom_json,a.root_material_ids_json FROM evidence_atoms a "
                "JOIN evidence_corrections c ON c.target_id=a.atom_id "
                "WHERE a.scope_key=? AND c.action IN ('withdraw','dispute','supersede')",
                (descriptor["scope_key"],),
            ).fetchall()
            correction_blocklist = []
            for old in corrected:
                old_atom = json.loads(old[1])
                correction_blocklist.append(
                    (
                        str(old_atom.get("kind", "")),
                        " ".join(str(old_atom.get("claim", "")).casefold().split()),
                        set(json.loads(old[0])) | set(json.loads(old[2])),
                    )
                )
            for atom in atoms:
                atom_id = atom["atom_id"]
                root_ids = sorted(
                    {
                        str(x)
                        for x in atom.get("root_evidence_ids", [])
                        if str(x).startswith("ev1:")
                    }
                )
                root_material_ids = sorted(
                    {
                        str(x)
                        for x in atom.get("root_material_ids", [])
                        if str(x).startswith(("mat1:", "leg1:"))
                    }
                )
                identity_refs = set(root_ids) | set(root_material_ids)
                claim_key = " ".join(str(atom.get("claim", "")).casefold().split())
                if any(
                    kind == str(atom.get("kind", ""))
                    and old_claim == claim_key
                    and old_roots == identity_refs
                    for kind, old_claim, old_roots in correction_blocklist
                ):
                    atom = dict(atom)
                    atom["status"] = "invalidated"
                db.execute(
                    """INSERT INTO evidence_atoms(
                    atom_id,scope_key,material_id,kind,status,root_ids_json,root_material_ids_json,atom_json,revision,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(atom_id) DO UPDATE SET
                    root_ids_json=excluded.root_ids_json,atom_json=excluded.atom_json,
                    root_material_ids_json=excluded.root_material_ids_json,
                    status=CASE WHEN evidence_atoms.status IN ('withdrawn','disputed','superseded','invalidated')
                                THEN evidence_atoms.status ELSE excluded.status END,
                    revision=excluded.revision,updated_at=excluded.updated_at""",
                    (
                        atom_id,
                        descriptor["scope_key"],
                        descriptor["material_id"],
                        atom["kind"],
                        atom.get("status", "reported_unverified"),
                        _json(root_ids),
                        _json(root_material_ids),
                        _json(atom),
                        current_revision,
                        now,
                        now,
                    ),
                )
            db.execute(
                "UPDATE evidence_materials SET status='ready',diagnostics_code='',updated_at=? WHERE material_id=?",
                (now, descriptor["material_id"]),
            )
            db.execute(
                "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                (descriptor["material_id"], descriptor["scope_key"]),
            )
            db.commit()
            return True

    async def get_atoms_by_ids(
        self, scope_key: str, atom_ids: list[str]
    ) -> dict[str, dict]:
        """Load exact, scope-bound atom rows for the typed evidence-view adapter."""
        return await self._read(self._get_atoms_by_ids_sync, scope_key, atom_ids)

    def _get_atoms_by_ids_sync(
        self, scope_key: str, atom_ids: list[str]
    ) -> dict[str, dict]:
        ids = list(
            dict.fromkeys(
                str(value) for value in atom_ids if isinstance(value, str) and value
            )
        )[:128]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._read_session() as db:
            rows = self._read_rows(
                db,
                "SELECT atom_id,material_id,kind,status,root_ids_json,root_material_ids_json,atom_json,revision "
                f"FROM evidence_atoms WHERE scope_key=? AND atom_id IN ({placeholders})",
                (scope_key, *ids),
            )
        return {
            str(row[0]): {
                "atom_id": str(row[0]),
                "material_id": str(row[1]),
                "kind": str(row[2]),
                "status": str(row[3]),
                "root_evidence_ids": self._read_json(row[4]),
                "root_material_ids": self._read_json(row[5]),
                "atom": self._read_json(row[6]),
                "revision": int(row[7]),
            }
            for row in rows
        }

    async def list_atoms(self, scope_key: str, limit: int = 100) -> list[dict]:
        return await self._read(self._list_atoms_sync, scope_key, limit)

    def _list_atoms_sync(self, scope_key: str, limit: int) -> list[dict]:
        with self._read_session() as db:
            rows = self._read_rows(
                db,
                "SELECT atom_id,kind,status,atom_json,revision FROM evidence_atoms WHERE scope_key=? ORDER BY updated_at DESC LIMIT ?",
                (scope_key, max(1, min(int(limit), 200))),
            )
            return [
                {
                    "atom_id": r[0],
                    "kind": r[1],
                    "status": r[2],
                    "atom": self._read_json(r[3]),
                    "revision": int(r[4]),
                }
                for r in rows
            ]

    async def list_admin_atoms(self, limit: int = 100) -> list[dict]:
        return await self._read(self._list_admin_atoms_sync, limit)

    def _list_admin_atoms_sync(self, limit: int) -> list[dict]:
        with self._read_session() as db:
            rows = self._read_rows(
                db,
                "SELECT a.atom_id,a.scope_key,a.kind,a.status,a.root_ids_json,a.root_material_ids_json,a.revision,a.updated_at, "
                "COALESCE(r.correction_revision,0) FROM evidence_atoms a "
                "LEFT JOIN evidence_scope_revisions r ON r.scope_key=a.scope_key "
                "ORDER BY a.updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            )
            scopes = {str(r[1]) for r in rows}
            effective_status = {}
            if scopes:
                placeholders = ",".join("?" for _ in scopes)
                correction_rows = self._read_rows(
                    db,
                    "SELECT scope_key,target_id,action,replacement_atom_id,affected_ids_json "
                    f"FROM evidence_corrections WHERE scope_key IN ({placeholders}) ORDER BY correction_revision",
                    tuple(scopes),
                )
                for correction in correction_rows:
                    scope = str(correction[0])
                    target_id = str(correction[1])
                    action = str(correction[2])
                    replacement_id = str(correction[3] or "")
                    target_status = {
                        "withdraw": "withdrawn",
                        "dispute": "disputed",
                        "supersede": "superseded",
                    }[action]
                    effective_status[(scope, target_id)] = target_status
                    for affected_id in json.loads(correction[4]):
                        affected_id = str(affected_id)
                        if affected_id != replacement_id and affected_id != target_id:
                            effective_status[(scope, affected_id)] = "invalidated"
            return [
                {
                    "atom_id": str(r[0]),
                    "scope_key": str(r[1]),
                    "kind": str(r[2]),
                    "status": effective_status.get((str(r[1]), str(r[0])), str(r[3])),
                    "source_count": len(set(json.loads(r[4]))),
                    "material_count": len(set(json.loads(r[5]))),
                    "revision": int(r[6]),
                    "updated_at": float(r[7]),
                    "correction_revision": int(r[8]),
                }
                for r in rows
            ]

    async def list_admin_materials(self, limit: int = 100) -> list[dict]:
        return await self._read(self._list_admin_materials_sync, limit)

    def _list_admin_materials_sync(self, limit: int) -> list[dict]:
        with self._read_session() as db:
            rows = self._read_rows(
                db,
                "SELECT m.material_id,m.scope_key,m.material_kind,m.role,m.status,m.updated_at, "
                "COALESCE(r.correction_revision,0),m.attempt_count,m.next_attempt_at,m.outcome_code "
                "FROM evidence_materials m "
                "LEFT JOIN evidence_scope_revisions r ON r.scope_key=m.scope_key "
                "WHERE m.status IN ('queued','deferred','failed') "
                "ORDER BY m.updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            )
            return [
                {
                    "material_id": str(r[0]),
                    "scope_key": str(r[1]),
                    "material_kind": str(r[2]),
                    "role": str(r[3]),
                    "status": str(r[4]),
                    "updated_at": float(r[5]),
                    "correction_revision": int(r[6]),
                    "attempt_count": int(r[7]),
                    "next_attempt_at": float(r[8]),
                    "outcome_code": str(r[9]),
                }
                for r in rows
            ]

    async def correct(
        self,
        target_id: str,
        expected_revision: int,
        action: str,
        replacement: str = "",
        reason: str = "",
        action_id: str = "",
    ) -> dict:
        return await asyncio.to_thread(
            self._correct_sync,
            target_id,
            expected_revision,
            action,
            replacement,
            reason,
            action_id,
        )

    def _correct_sync(
        self,
        target_id: str,
        expected_revision: int,
        action: str,
        replacement: str,
        reason: str,
        action_id: str,
    ) -> dict:
        if action not in {"withdraw", "dispute", "supersede"}:
            raise ValueError("unsupported correction action")
        if (
            not target_id
            or len(target_id) > 160
            or not reason.strip()
            or len(reason) > 500
        ):
            raise ValueError("target and a bounded correction reason are required")
        if action == "supersede" and (
            not replacement.strip() or len(replacement) > 1200
        ):
            raise ValueError("supersede requires a bounded replacement")
        now = time.time()
        cache_update = None
        with self._lock, self._session() as db:
            db.execute("BEGIN IMMEDIATE")
            target = db.execute(
                "SELECT atom_id,scope_key,material_id,kind,status,root_ids_json,atom_json,root_material_ids_json "
                "FROM evidence_atoms WHERE atom_id=?",
                (target_id,),
            ).fetchone()
            if target:
                scope_key = str(target[1])
                roots = set(json.loads(target[5]))
                material_roots = set(json.loads(target[7]))
                material_roots.add(str(target[2]))
            else:
                material = db.execute(
                    "SELECT material_id,scope_key FROM evidence_materials WHERE material_id=?",
                    (target_id,),
                ).fetchone()
                if not material:
                    db.rollback()
                    raise ValueError("insight or material not found")
                scope_key = str(material[1])
                roots = set()
                material_roots = {str(material[0])}
            families = set()
            for material_id in tuple(material_roots):
                family = db.execute(
                    "SELECT source_family_id FROM evidence_materials WHERE material_id=? AND scope_key=?",
                    (material_id, scope_key),
                ).fetchone()
                if family and family[0]:
                    families.add(str(family[0]))
            if action_id:
                previous = db.execute(
                    "SELECT correction_revision,target_id,action,replacement_atom_id,affected_ids_json "
                    "FROM evidence_corrections WHERE scope_key=? AND action_id=?",
                    (scope_key, action_id),
                ).fetchone()
                if previous:
                    db.commit()
                    return {
                        "schema_version": 1,
                        "scope_key": scope_key,
                        "correction_revision": int(previous[0]),
                        "target_id": previous[1],
                        "action": previous[2],
                        "replacement_atom_id": previous[3],
                        "affected_ids": json.loads(previous[4]),
                        "idempotent_replay": True,
                    }
            revision_row = db.execute(
                "SELECT correction_revision FROM evidence_scope_revisions WHERE scope_key=?",
                (scope_key,),
            ).fetchone()
            revision = int(revision_row[0]) if revision_row else 0
            if revision != int(expected_revision):
                db.rollback()
                raise RuntimeError("stale correction revision")
            affected = set(roots) | set(material_roots) | families
            if target:
                affected.add(str(target[0]))
            rows = db.execute(
                "SELECT atom_id,material_id,root_ids_json,root_material_ids_json FROM evidence_atoms WHERE scope_key=? AND status IN ('active','reported_unverified','pending_continuity')",
                (scope_key,),
            ).fetchall()
            for row in rows:
                atom_roots = set(json.loads(row[2]))
                atom_material_roots = set(json.loads(row[3]))
                material_row = db.execute(
                    "SELECT source_family_id FROM evidence_materials WHERE material_id=? AND scope_key=?",
                    (row[1], scope_key),
                ).fetchone()
                same_family = material_row and str(material_row[0]) in families
                if (
                    atom_roots & roots
                    or atom_material_roots & material_roots
                    or str(row[1]) in material_roots
                    or same_family
                ):
                    affected.add(str(row[0]))
            new_revision = revision + 1
            replacement_atom_id = None
            if not target:
                db.execute(
                    "UPDATE evidence_materials SET status='invalidated',diagnostics_code='admin_correction',updated_at=? "
                    "WHERE material_id=? AND scope_key=?",
                    (now, target_id, scope_key),
                )
                db.execute(
                    "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                    (target_id, scope_key),
                )
            if action == "supersede":
                if target:
                    target_atom = json.loads(target[6])
                    replacement_atom_id = (
                        "atm1:"
                        + hashlib.sha256(
                            (
                                scope_key
                                + "\0"
                                + str(target[3])
                                + "\0"
                                + replacement.strip().casefold()
                                + "\0"
                                + str(new_revision)
                            ).encode()
                        ).hexdigest()
                    )
                    next_atom = dict(target_atom)
                    next_atom.update(
                        {
                            "atom_id": replacement_atom_id,
                            "claim": replacement.strip(),
                            "status": "active",
                            "revision": new_revision,
                            "supersedes": target_id,
                            "derived_from": [target_id],
                        }
                    )
                    next_roots = sorted(roots)
                    next_material_roots = sorted(material_roots)
                    next_atom["root_evidence_ids"] = next_roots
                    next_atom["root_material_ids"] = next_material_roots
                    db.execute(
                        """INSERT INTO evidence_atoms(atom_id,scope_key,material_id,kind,status,root_ids_json,root_material_ids_json,atom_json,revision,created_at,updated_at)
                                  VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            replacement_atom_id,
                            scope_key,
                            str(target[2]),
                            str(target[3]),
                            "active",
                            _json(next_roots),
                            _json(next_material_roots),
                            _json(next_atom),
                            new_revision,
                            now,
                            now,
                        ),
                    )
                    affected.add(replacement_atom_id)
                else:
                    db.rollback()
                    raise ValueError(
                        "material-only supersede requires a derived insight first"
                    )
            for material_id in material_roots:
                db.execute(
                    "UPDATE evidence_materials SET status='invalidated',diagnostics_code='source_correction',updated_at=? "
                    "WHERE material_id=? AND scope_key=?",
                    (now, material_id, scope_key),
                )
                db.execute(
                    "DELETE FROM evidence_derivation_queue WHERE material_id=? AND scope_key=?",
                    (material_id, scope_key),
                )
            correction_id = "cor1:" + uuid.uuid4().hex
            affected_ids = sorted(affected)
            db.execute(
                "INSERT INTO evidence_corrections(correction_id,scope_key,correction_revision,target_id,action,replacement_atom_id,affected_ids_json,reason,action_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    correction_id,
                    scope_key,
                    new_revision,
                    target_id,
                    action,
                    replacement_atom_id,
                    _json(affected_ids),
                    reason.strip(),
                    action_id[:160] if action_id else None,
                    now,
                ),
            )
            db.execute(
                "INSERT INTO evidence_scope_revisions(scope_key,correction_revision) VALUES (?,?) "
                "ON CONFLICT(scope_key) DO UPDATE SET correction_revision=excluded.correction_revision",
                (scope_key, new_revision),
            )
            cache_update = self._revision_projection(db, scope_key, new_revision)
            # Publish only committed state, without a gap in which a reader
            # could accept the old revision after the correction commits.
            with self._revision_lock:
                db.commit()
                self._revision_cache[scope_key] = cache_update
        return {
            "schema_version": 1,
            "scope_key": scope_key,
            "correction_revision": new_revision,
            "target_id": target_id,
            "action": action,
            "replacement_atom_id": replacement_atom_id,
            "affected_ids": affected_ids,
            "idempotent_replay": False,
        }

    async def prune_old(
        self,
        retention_days: int = 30,
        *,
        now: float | None = None,
        queue_retention_days: int | None = None,
        max_rows_per_scope: int = 2000,
    ) -> dict[str, int]:
        """清理过期失效/滞留 material、队列项、孤儿 atom/correction，并执行 WAL 截断与增量收缩。"""
        return await asyncio.to_thread(
            self._prune_old_sync,
            retention_days,
            now,
            queue_retention_days,
            max_rows_per_scope,
        )

    @classmethod
    def prune_file_sync(
        cls,
        path: str | Path,
        retention_days: int = 30,
        now: float | None = None,
        *,
        queue_retention_days: int | None = None,
        max_rows_per_scope: int = 2000,
    ) -> dict[str, int]:
        """对指定 sidecar SQLite 文件执行一次同步安全清理与 WAL 截断（不启动线程池）。"""
        db_path = Path(path)
        if not db_path.is_file():
            return {
                "materials_pruned": 0,
                "atoms_pruned": 0,
                "corrections_pruned": 0,
                "queue_pruned": 0,
                "admissions_pruned": 0,
                "wal_checkpointed": 0,
            }
        db = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=5000")
            return cls._execute_prune_on_connection(
                db,
                retention_days=retention_days,
                now=now,
                queue_retention_days=queue_retention_days,
                max_rows_per_scope=max_rows_per_scope,
            )
        finally:
            db.close()

    def _prune_old_sync(
        self,
        retention_days: int = 30,
        now: float | None = None,
        queue_retention_days: int | None = None,
        max_rows_per_scope: int = 2000,
    ) -> dict[str, int]:
        with self._lock, self._session() as db:
            stats = self._execute_prune_on_connection(
                db,
                retention_days=retention_days,
                now=now,
                queue_retention_days=queue_retention_days,
                max_rows_per_scope=max_rows_per_scope,
                on_committed=self._load_revision_cache,
            )
        return stats

    @staticmethod
    def _execute_prune_on_connection(
        db: sqlite3.Connection,
        *,
        retention_days: int = 30,
        now: float | None = None,
        queue_retention_days: int | None = None,
        max_rows_per_scope: int = 2000,
        on_committed=None,
    ) -> dict[str, int]:
        now_ts = time.time() if now is None else float(now)
        days = max(1, int(retention_days))
        q_days = (
            max(1, int(queue_retention_days))
            if queue_retention_days is not None
            else min(days, 14)
        )
        per_scope_cap = max(0, int(max_rows_per_scope))
        cutoff = now_ts - days * 86400.0
        queue_cutoff = now_ts - q_days * 86400.0
        admissions_cutoff = min(cutoff, now_ts - 86400.0)

        existing_tables = {
            str(row[0])
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        atoms_pruned = 0
        materials_pruned = 0
        corrections_pruned = 0
        queue_pruned = 0
        admissions_pruned = 0

        if {"evidence_materials", "evidence_atoms"} <= existing_tables:
            db.execute("BEGIN IMMEDIATE")
            try:
                # 1. 清理已过期的非活跃 atom（保留所有活跃/待验证/替代 atom）
                cur = db.execute(
                    "DELETE FROM evidence_atoms "
                    "WHERE status NOT IN ("
                    "'active','verified','verified_utterance',"
                    "'reported_unverified','pending_continuity','observed_user_boundary'"
                    ") AND updated_at < ?",
                    (cutoff,),
                )
                atoms_pruned += max(0, cur.rowcount)

                # 2. 清理引用不存在 material 的孤儿 atom
                cur = db.execute(
                    "DELETE FROM evidence_atoms "
                    "WHERE material_id NOT IN (SELECT material_id FROM evidence_materials)"
                )
                atoms_pruned += max(0, cur.rowcount)

                # 3. 清理滞留/过期的推导队列项（queued/deferred/owner_wait）与按 scope 上限裁剪
                if "evidence_derivation_queue" in existing_tables:
                    cur = db.execute(
                        "DELETE FROM evidence_derivation_queue "
                        "WHERE status IN ('queued','deferred','owner_wait') AND updated_at < ?",
                        (queue_cutoff,),
                    )
                    queue_pruned += max(0, cur.rowcount)
                    if per_scope_cap > 0:
                        q_scopes = [
                            str(r[0])
                            for r in db.execute(
                                "SELECT scope_key FROM evidence_derivation_queue "
                                "GROUP BY scope_key HAVING COUNT(*) > ?",
                                (per_scope_cap,),
                            ).fetchall()
                        ]
                        for sc in q_scopes:
                            cur = db.execute(
                                "DELETE FROM evidence_derivation_queue "
                                "WHERE scope_key = ? AND status IN ('queued','deferred','owner_wait') "
                                "AND material_id NOT IN ("
                                "  SELECT material_id FROM evidence_derivation_queue "
                                "  WHERE scope_key = ? "
                                "  ORDER BY updated_at DESC LIMIT ?"
                                ")",
                                (sc, sc, per_scope_cap),
                            )
                            queue_pruned += max(0, cur.rowcount)

                # 4. 清理已失效/终态/失败及滞留过期的 ready/queued/deferred/owner_wait material
                #    （只要无剩余 atom 引用即可安全清理；活跃/替代 atom 仍被严格保护）
                cur = db.execute(
                    "DELETE FROM evidence_materials "
                    "WHERE status IN ('invalidated','terminal','failed','ready','queued','deferred','owner_wait') "
                    "AND updated_at < ? "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM evidence_atoms a WHERE a.material_id = evidence_materials.material_id"
                    ")",
                    (cutoff,),
                )
                materials_pruned += max(0, cur.rowcount)

                # 按 scope 限制未被引用的历史 material 行数，防止高活跃群无限膨胀
                if per_scope_cap > 0:
                    m_scopes = [
                        str(r[0])
                        for r in db.execute(
                            "SELECT scope_key FROM evidence_materials "
                            "GROUP BY scope_key HAVING COUNT(*) > ?",
                            (per_scope_cap,),
                        ).fetchall()
                    ]
                    for sc in m_scopes:
                        cur = db.execute(
                            "DELETE FROM evidence_materials "
                            "WHERE scope_key = ? "
                            "AND NOT EXISTS ("
                            "  SELECT 1 FROM evidence_atoms a WHERE a.material_id = evidence_materials.material_id"
                            ") "
                            "AND material_id NOT IN ("
                            "  SELECT material_id FROM evidence_materials "
                            "  WHERE scope_key = ? "
                            "  ORDER BY updated_at DESC LIMIT ?"
                            ")",
                            (sc, sc, per_scope_cap),
                        )
                        materials_pruned += max(0, cur.rowcount)

                # 5. 清理孤儿队列项与过期准入流水
                if "evidence_derivation_queue" in existing_tables:
                    cur = db.execute(
                        "DELETE FROM evidence_derivation_queue "
                        "WHERE material_id NOT IN (SELECT material_id FROM evidence_materials)"
                    )
                    queue_pruned += max(0, cur.rowcount)

                if "evidence_queue_admissions" in existing_tables:
                    cur = db.execute(
                        "DELETE FROM evidence_queue_admissions WHERE admitted_at < ?",
                        (admissions_cutoff,),
                    )
                    admissions_pruned += max(0, cur.rowcount)

                # 6. 清理已过期且目标/替代 atom 均已不存在的孤儿 correction（绝不回退 scope_revisions）
                if "evidence_corrections" in existing_tables:
                    cur = db.execute(
                        "DELETE FROM evidence_corrections "
                        "WHERE created_at < ? "
                        "AND target_id NOT IN (SELECT atom_id FROM evidence_atoms) "
                        "AND target_id NOT IN (SELECT material_id FROM evidence_materials) "
                        "AND (replacement_atom_id IS NULL "
                        "     OR replacement_atom_id NOT IN (SELECT atom_id FROM evidence_atoms))",
                        (cutoff,),
                    )
                    corrections_pruned += max(0, cur.rowcount)

                db.commit()
            except Exception:
                db.rollback()
                raise

            if callable(on_committed):
                on_committed(db)

        wal_checkpointed = 0
        try:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            wal_checkpointed = 1
        except sqlite3.Error:
            pass
        try:
            db.execute("PRAGMA incremental_vacuum").fetchall()
        except sqlite3.Error:
            pass

        return {
            "materials_pruned": materials_pruned,
            "atoms_pruned": atoms_pruned,
            "corrections_pruned": corrections_pruned,
            "queue_pruned": queue_pruned,
            "admissions_pruned": admissions_pruned,
            "wal_checkpointed": wal_checkpointed,
        }
