"""Version-triggered data backup manager.

Automatically backs up all plugin data files when the plugin version changes,
storing each backup under a version-tagged directory for easy recovery.
Also prunes expired or oversized backups so the data directory stays bounded.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from astrbot.api import logger


# Match metadata.yaml; backup manifests use this plugin version.
def _read_plugin_version() -> str:
    """Read the plugin version from metadata.yaml, the single source of truth.

    The previous hard-coded constant (2.7.0-rc.9) had drifted away from
    metadata.yaml and the @register decorator (both 2.7.0-rc.14), so the
    "version changed" automatic backup never fired again after rc.9.

    Returns:
        The version string, or "0.0.0-unknown" when it cannot be read.
    """
    try:
        meta = Path(__file__).resolve().parents[2] / "metadata.yaml"
        for line in meta.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "version":
                cleaned = value.strip().strip('"').strip("'")
                return cleaned or "0.0.0-unknown"
    except Exception:
        pass
    return "0.0.0-unknown"


PLUGIN_VERSION = _read_plugin_version()

# Retention policies enforced by BackupManager.prune().
KEEP_VERSION_BACKUPS = 2
KEEP_ADAPTER_BACKUPS = 1
KEEP_SCOPE_BACKUPS = 3
MAX_BACKUP_TOTAL_MB = 512

_VERSION_FILE = ".plugin_version"
_BACKUP_INFO_FILE = "backup_info.json"

_VERSION_DIR_RE = re.compile(r"^v(?:\d|unknown)")
_ADAPTER_BACKUP_PREFIX = "adapter_ownership_"
_SCOPE_BACKUP_PREFIX = "scope_migration_"
_DAILY_BACKUP_GLOBS: tuple[str, ...] = ("*_backup_*.db", "*_backup_*.db.gz")
_SIDECAR_SCHEMA_BAK_GLOB = "evidence_sidecar.sqlite3.schema-*.bak"
KEEP_SIDECAR_SCHEMA_BACKUPS = 1


def _is_daily_backup(entry: _BackupEntry) -> bool:
    if entry.is_dir:
        return False
    name = entry.path.name
    return any(fnmatch.fnmatch(name, pat) for pat in _DAILY_BACKUP_GLOBS)


def _is_staging_or_bak(entry: _BackupEntry) -> bool:
    name = entry.path.name
    return (
        name.startswith(".tmp_")
        or name.endswith(".tmp")
        or name.endswith(".bak")
    )


class _BackupEntry(NamedTuple):
    """A direct child of the backups directory with its retention metadata."""

    path: Path
    is_dir: bool
    mtime: float
    size: int


_SQLITE_HEADER = b"SQLite format 3\x00"

# SQLite databases backed up via SQLite online backup API (`src.backup(dst)`).
_SQLITE_DB_FILES: tuple[str, ...] = (
    "livingmemory.db",
    "livingmemory_graph_documents.db",
    "conversations.db",
    "evidence_sidecar.sqlite3",
)

# Plain metadata files included in every backup.
_PLAIN_BACKUP_FILES: tuple[str, ...] = ("decay_state.json",)

# Rebuildable FAISS vector indexes (skipped by default to keep backups compact).
_FAISS_INDEX_FILES: tuple[str, ...] = (
    "livingmemory.index",
    "livingmemory_graph.index",
)

# Kept for backward-compatible introspection.
_BACKUP_PATTERNS: list[str] = [
    *_SQLITE_DB_FILES,
    *_PLAIN_BACKUP_FILES,
    *_FAISS_INDEX_FILES,
]


def _is_sqlite_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER
    except OSError:
        return False


def _backup_sqlite_file(src: Path, dst: Path) -> bool:
    """Back up a SQLite file using chunked online backup API for WAL consistency.

    Returns True when SQLite online backup was used, False when ``src`` is a
    non-SQLite placeholder file copied directly. Never falls back to raw
    ``shutil.copy2`` on a live SQLite database when online backup fails.
    """
    if _is_sqlite_file(src):
        src_conn: sqlite3.Connection | None = None
        dst_conn: sqlite3.Connection | None = None
        try:
            src_conn = sqlite3.connect(str(src), timeout=10.0)
            src_conn.execute("PRAGMA busy_timeout=10000")
            dst_conn = sqlite3.connect(str(dst), timeout=10.0)
            src_conn.backup(dst_conn, pages=256, sleep=0.05)
            return True
        except sqlite3.Error as exc:
            logger.error(
                f"[BackupManager] SQLite online backup 失败 ({src.name})，中止该库备份避免产出损坏副本: {exc}"
            )
            if dst_conn is not None:
                try:
                    dst_conn.close()
                except Exception:
                    pass
                dst_conn = None
            dst.unlink(missing_ok=True)
            raise
        finally:
            if dst_conn is not None:
                dst_conn.close()
            if src_conn is not None:
                src_conn.close()
    shutil.copy2(src, dst)
    return False


def _is_version_backup(entry: _BackupEntry) -> bool:
    """Return True if an entry is a version backup directory or .tar.gz archive."""
    name = entry.path.name
    if entry.is_dir:
        return bool(_VERSION_DIR_RE.match(name))
    return name.endswith(".tar.gz") and bool(_VERSION_DIR_RE.match(name))


class BackupManager:
    """Detect version changes and create full data backups."""

    def __init__(
        self,
        data_dir: str,
        *,
        compress: bool = True,
        include_faiss: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.version_file = self.data_dir / _VERSION_FILE
        self.compress = bool(compress)
        self.include_faiss = bool(include_faiss)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stored_version(self) -> str | None:
        """Return the last-known plugin version, or None on first run."""
        if not self.version_file.exists():
            return None
        try:
            return self.version_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def write_current_version(self) -> None:
        """Persist the current plugin version."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.version_file.write_text(PLUGIN_VERSION, encoding="utf-8")

    def needs_backup(self) -> bool:
        """Return True when the plugin version has changed (or is fresh)."""
        stored = self.get_stored_version()
        if stored is None:
            return True  # first install — backup for safety
        return stored != PLUGIN_VERSION

    def _stage_backup_files(
        self, target_dir: Path, *, include_faiss: bool
    ) -> tuple[int, list[str], list[str]]:
        """Populate ``target_dir`` with consistent DB snapshots and plain files."""
        copied_count = 0
        backed_up_files: list[str] = []
        online_backed_up: list[str] = []

        for db_name in _SQLITE_DB_FILES:
            src = self.data_dir / db_name
            if not src.is_file():
                continue
            dest = target_dir / db_name
            try:
                used_online = _backup_sqlite_file(src, dest)
                copied_count += 1
                backed_up_files.append(db_name)
                if used_online:
                    online_backed_up.append(db_name)
                else:
                    # Non-SQLite fallback: preserve any companion -wal/-shm files
                    for suffix in ("-wal", "-shm"):
                        side = self.data_dir / f"{db_name}{suffix}"
                        if side.is_file():
                            shutil.copy2(side, target_dir / side.name)
                            copied_count += 1
                            backed_up_files.append(side.name)
            except OSError as exc:
                logger.error(f"[BackupManager] 备份数据库失败 {db_name}: {exc}")

        plain_files: tuple[str, ...] = (
            (*_PLAIN_BACKUP_FILES, *_FAISS_INDEX_FILES)
            if include_faiss
            else _PLAIN_BACKUP_FILES
        )
        for name in plain_files:
            src = self.data_dir / name
            if not src.is_file():
                continue
            try:
                shutil.copy2(src, target_dir / name)
                copied_count += 1
                backed_up_files.append(name)
            except OSError as exc:
                logger.error(f"[BackupManager] 备份文件失败 {name}: {exc}")

        return copied_count, backed_up_files, online_backed_up

    def backup_if_needed(
        self,
        *,
        compress: bool | None = None,
        include_faiss: bool | None = None,
    ) -> str | None:
        """Create a version backup when the version changed.

        By default creates a compressed ``v<old>.tar.gz`` archive using SQLite
        online backup (``src.backup(dst)``) for WAL consistency, includes
        ``evidence_sidecar.sqlite3``, and skips rebuildable FAISS ``.index``
        files unless ``include_faiss=True``.
        """
        if not self.needs_backup():
            return None

        use_compress = self.compress if compress is None else bool(compress)
        use_faiss = self.include_faiss if include_faiss is None else bool(include_faiss)

        stored = self.get_stored_version()
        old_label = stored or "unknown"
        backups_root = self.data_dir / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        if use_compress:
            archive_path = backups_root / f"v{old_label}.tar.gz"
            logger.info(
                f"[BackupManager] 检测到版本变更 ({old_label} → {PLUGIN_VERSION})，"
                f"正在创建压缩备份 {archive_path} (include_faiss={use_faiss}) ..."
            )
            with tempfile.TemporaryDirectory(
                dir=str(backups_root), prefix=f".tmp_v{old_label}_"
            ) as staging_str:
                staging_dir = Path(staging_str)
                copied_count, backed_up_files, online_backed_up = (
                    self._stage_backup_files(staging_dir, include_faiss=use_faiss)
                )
                info = {
                    "plugin_version": PLUGIN_VERSION,
                    "previous_version": old_label,
                    "backup_timestamp": datetime.now(timezone.utc).isoformat(),
                    "backup_unix_time": time.time(),
                    "files_copied": copied_count,
                    "data_dir": str(self.data_dir),
                    "format": "tar.gz",
                    "include_faiss": use_faiss,
                    "sqlite_online_backup": online_backed_up,
                    "files": sorted(backed_up_files),
                }
                info_path = staging_dir / _BACKUP_INFO_FILE
                info_path.write_text(
                    json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                tmp_archive = backups_root / f".v{old_label}.tar.gz.tmp"
                try:
                    with tarfile.open(tmp_archive, "w:gz") as tar:
                        for item in sorted(staging_dir.iterdir()):
                            if item.is_file():
                                tar.add(item, arcname=item.name)
                    os.replace(tmp_archive, archive_path)
                finally:
                    if tmp_archive.exists():
                        tmp_archive.unlink(missing_ok=True)

                # Sidecar metadata file lets list_backups() read metadata in O(1)
                # without streaming/decompressing 200MB+ .tar.gz archives.
                meta_path = backups_root / f"v{old_label}.meta.json"
                try:
                    meta_path.write_text(
                        json.dumps(info, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except OSError:
                    pass

            self.write_current_version()
            logger.info(
                f"[BackupManager] 压缩备份完成: {copied_count} 个文件 → {archive_path}"
            )
            return str(archive_path)

        backup_dir = backups_root / f"v{old_label}"
        existed_before = backup_dir.exists()
        backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[BackupManager] 检测到版本变更 ({old_label} → {PLUGIN_VERSION})，"
            f"正在备份数据到 {backup_dir} (include_faiss={use_faiss}) ..."
        )
        try:
            copied_count, backed_up_files, online_backed_up = self._stage_backup_files(
                backup_dir, include_faiss=use_faiss
            )
            info = {
                "plugin_version": PLUGIN_VERSION,
                "previous_version": old_label,
                "backup_timestamp": datetime.now(timezone.utc).isoformat(),
                "backup_unix_time": time.time(),
                "files_copied": copied_count,
                "data_dir": str(self.data_dir),
                "format": "directory",
                "include_faiss": use_faiss,
                "sqlite_online_backup": online_backed_up,
                "files": sorted(backed_up_files),
            }
            info_path = backup_dir / _BACKUP_INFO_FILE
            info_path.write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            if not existed_before and backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            raise
        self.write_current_version()
        logger.info(f"[BackupManager] 备份完成: {copied_count} 个文件 → {backup_dir}")
        return str(backup_dir)

    async def backup_if_needed_async(
        self,
        *,
        compress: bool | None = None,
        include_faiss: bool | None = None,
    ) -> str | None:
        """异步版本：通过 asyncio.to_thread 将同步文件 I/O 卸载到线程池。"""
        return await asyncio.to_thread(
            self.backup_if_needed,
            compress=compress,
            include_faiss=include_faiss,
        )

    @staticmethod
    def prune(
        data_dir: str,
        *,
        keep_days: int = 7,
        max_total_mb: int = MAX_BACKUP_TOTAL_MB,
    ) -> dict[str, int]:
        """Remove expired and oversized backups under ``<data_dir>/backups``.

        Only direct children of the backups directory are touched. Every child
        is resolved first and skipped if it escapes the backups directory.
        Retention per category:

        - ``v<digit>...`` directories keep the newest KEEP_VERSION_BACKUPS;
        - ``adapter_ownership_*`` directories keep the newest
          KEEP_ADAPTER_BACKUPS;
        - ``scope_migration_*`` files keep the newest KEEP_SCOPE_BACKUPS;
        - ``*_backup_*.db`` files older than ``keep_days`` are removed.

        If the remaining entries exceed ``max_total_mb``, the oldest entries are
        removed until the cap is met or nothing else can be removed. The newest
        version directory and the newest daily DB backup are never removed by
        the cap.

        Args:
            data_dir: Plugin data directory that owns the backups folder.
            keep_days: Retention period for daily DB backups, in days.
            max_total_mb: Total backups size cap, in MiB.

        Returns:
            Counters of removed files/directories and bytes, plus the final
            backups size in bytes.
        """
        stats = {
            "files_removed": 0,
            "dirs_removed": 0,
            "bytes_removed": 0,
            "total_bytes": 0,
        }
        backups_root = Path(data_dir) / "backups"
        if not backups_root.is_dir():
            return stats

        root = backups_root.resolve()
        entries: list[_BackupEntry] = []
        for child in backups_root.iterdir():
            try:
                path = child.resolve()
            except OSError as exc:
                logger.warning(
                    f"[BackupManager] Skipping unresolvable backup path {child}: {exc}"
                )
                continue
            if path.parent != root:
                logger.warning(
                    f"[BackupManager] Skipping path outside backups dir: {child}"
                )
                continue
            try:
                mtime = path.stat().st_mtime
                is_dir = path.is_dir()
                size = (
                    sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                    if is_dir
                    else path.stat().st_size
                )
            except OSError as exc:
                logger.warning(
                    f"[BackupManager] Skipping unreadable backup entry {child}: {exc}"
                )
                continue
            entries.append(_BackupEntry(path, is_dir, mtime, size))

        removed: set[Path] = set()

        def _remove(entry: _BackupEntry) -> None:
            try:
                if entry.is_dir:
                    shutil.rmtree(entry.path)
                else:
                    entry.path.unlink()
            except OSError as exc:
                logger.warning(
                    f"[BackupManager] Failed to remove backup entry {entry.path}: {exc}"
                )
                return
            removed.add(entry.path)
            stats["bytes_removed"] += entry.size
            if entry.is_dir:
                stats["dirs_removed"] += 1
            else:
                stats["files_removed"] += 1
            if not entry.is_dir and entry.path.name.endswith(".tar.gz"):
                sidecar_meta = entry.path.with_name(
                    entry.path.name[: -len(".tar.gz")] + ".meta.json"
                )
                try:
                    if sidecar_meta.is_file():
                        meta_size = sidecar_meta.stat().st_size
                        sidecar_meta.unlink()
                        removed.add(sidecar_meta)
                        stats["bytes_removed"] += meta_size
                        stats["files_removed"] += 1
                except OSError:
                    pass

        # Remove leftover .bak / .tmp_* / *.tmp staging files unconditionally.
        for entry in entries:
            if _is_staging_or_bak(entry):
                _remove(entry)

        # Category retention: keep the newest entries of each backup source.
        version_entries = sorted(
            (e for e in entries if e.path not in removed and _is_version_backup(e)),
            key=lambda e: e.mtime,
            reverse=True,
        )
        for entry in version_entries[KEEP_VERSION_BACKUPS:]:
            _remove(entry)

        adapter_dirs = sorted(
            (
                e
                for e in entries
                if e.path not in removed
                and e.is_dir
                and e.path.name.startswith(_ADAPTER_BACKUP_PREFIX)
            ),
            key=lambda e: e.mtime,
            reverse=True,
        )
        for entry in adapter_dirs[KEEP_ADAPTER_BACKUPS:]:
            _remove(entry)

        scope_files = sorted(
            (
                e
                for e in entries
                if e.path not in removed
                and not e.is_dir
                and e.path.name.startswith(_SCOPE_BACKUP_PREFIX)
            ),
            key=lambda e: e.mtime,
            reverse=True,
        )
        for entry in scope_files[KEEP_SCOPE_BACKUPS:]:
            _remove(entry)

        cutoff = time.time() - keep_days * 86400
        for entry in entries:
            if (
                entry.path not in removed
                and _is_daily_backup(entry)
                and entry.mtime < cutoff
            ):
                _remove(entry)

        # Total size cap: drop oldest entries first, never the protected ones.
        # Prefer protecting compressed archives (.tar.gz / .db.gz) over legacy
        # uncompressed files/dirs so giant uncompressed copies are not exempt
        # from the total size cap once a compressed backup exists.
        remaining = [e for e in entries if e.path not in removed]
        total_bytes = sum(e.size for e in remaining)
        limit = max_total_mb * 1024 * 1024
        if total_bytes > limit:
            protected: set[Path] = set()
            kept_versions = [e for e in remaining if _is_version_backup(e)]
            if kept_versions:
                compressed_versions = [
                    e for e in kept_versions if not e.is_dir and e.path.name.endswith(".tar.gz")
                ]
                pool = compressed_versions or kept_versions
                protected.add(max(pool, key=lambda e: e.mtime).path)
            kept_daily = [e for e in remaining if _is_daily_backup(e)]
            if kept_daily:
                compressed_daily = [
                    e for e in kept_daily if e.path.name.endswith(".db.gz")
                ]
                pool = compressed_daily or kept_daily
                protected.add(max(pool, key=lambda e: e.mtime).path)
            for entry in sorted(remaining, key=lambda e: e.mtime):
                if total_bytes <= limit:
                    break
                if entry.path in protected:
                    continue
                _remove(entry)
                if entry.path in removed:
                    total_bytes -= entry.size

        # Prune stale evidence_sidecar schema migration .bak files stored in data_dir.
        data_root = Path(data_dir)
        if data_root.is_dir():
            sidecar_baks: list[tuple[float, int, Path]] = []
            for bak in data_root.glob(_SIDECAR_SCHEMA_BAK_GLOB):
                if not bak.is_file():
                    continue
                try:
                    st = bak.stat()
                    sidecar_baks.append((st.st_mtime, st.st_size, bak))
                except OSError:
                    continue
            sidecar_baks.sort(key=lambda item: item[0], reverse=True)
            for idx, (mtime, size, bak) in enumerate(sidecar_baks):
                if idx >= KEEP_SIDECAR_SCHEMA_BACKUPS or mtime < cutoff:
                    try:
                        bak.unlink()
                        stats["files_removed"] += 1
                        stats["bytes_removed"] += size
                    except OSError as exc:
                        logger.warning(
                            f"[BackupManager] Failed to remove sidecar backup {bak}: {exc}"
                        )

        stats["total_bytes"] = sum(e.size for e in entries if e.path not in removed)
        logger.info(
            f"[BackupManager] Backup prune finished: removed "
            f"{stats['files_removed']} files / {stats['dirs_removed']} dirs "
            f"({stats['bytes_removed']} bytes), "
            f"{stats['total_bytes']} bytes kept"
        )
        return stats

    async def prune_async(
        self,
        *,
        keep_days: int = 7,
        max_total_mb: int = MAX_BACKUP_TOTAL_MB,
    ) -> dict[str, int]:
        """Async wrapper that runs prune() in the thread pool.

        Args:
            keep_days: Retention period for daily DB backups, in days.
            max_total_mb: Total backups size cap, in MiB.

        Returns:
            Same counters as prune().
        """
        return await asyncio.to_thread(
            self.prune,
            str(self.data_dir),
            keep_days=keep_days,
            max_total_mb=max_total_mb,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def list_backups(data_dir: str) -> list[dict]:
        """Enumerate existing backups (both .tar.gz archives and legacy directories)."""
        backups_path = Path(data_dir) / "backups"
        if not backups_path.exists():
            return []

        result: list[dict] = []
        for entry in sorted(backups_path.iterdir(), reverse=True):
            if entry.is_dir():
                info_path = entry / _BACKUP_INFO_FILE
                info: dict = {}
                if info_path.exists():
                    try:
                        info = json.loads(info_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        pass
                info.setdefault("directory", str(entry))
                info.setdefault("name", entry.name)
                info.setdefault("format", "directory")
                files = [p.name for p in entry.iterdir() if p.is_file()]
                info.setdefault("files", files)
                info.setdefault("file_count", len(files))
                result.append(info)
            elif entry.is_file() and entry.name.endswith(".tar.gz"):
                info = {}
                files: list[str] = []
                meta_path = entry.with_name(
                    entry.name[: -len(".tar.gz")] + ".meta.json"
                )
                loaded_from_sidecar = False
                if meta_path.is_file():
                    try:
                        parsed = json.loads(meta_path.read_text(encoding="utf-8"))
                        if isinstance(parsed, dict):
                            info = parsed
                            raw_files = info.get("files")
                            if isinstance(raw_files, list):
                                files = [str(f) for f in raw_files]
                            loaded_from_sidecar = True
                    except (json.JSONDecodeError, OSError):
                        loaded_from_sidecar = False
                if not loaded_from_sidecar:
                    try:
                        with tarfile.open(entry, "r:gz") as tar:
                            for member in tar.getmembers():
                                if not member.isfile():
                                    continue
                                basename = Path(member.name).name
                                files.append(basename)
                                if basename == _BACKUP_INFO_FILE:
                                    extracted = tar.extractfile(member)
                                    if extracted is not None:
                                        try:
                                            info = json.loads(
                                                extracted.read().decode("utf-8")
                                            )
                                        except (
                                            json.JSONDecodeError,
                                            UnicodeDecodeError,
                                            OSError,
                                        ):
                                            pass
                    except (tarfile.TarError, OSError):
                        pass
                info.setdefault("archive", str(entry))
                info.setdefault("directory", str(entry))
                info.setdefault("name", entry.name[: -len(".tar.gz")])
                info.setdefault("format", "tar.gz")
                info.setdefault("files", files)
                info.setdefault("file_count", len(files))
                result.append(info)

        result.sort(key=lambda item: str(item.get("name", "")), reverse=True)
        return result


__all__ = [
    "KEEP_ADAPTER_BACKUPS",
    "KEEP_SCOPE_BACKUPS",
    "KEEP_VERSION_BACKUPS",
    "MAX_BACKUP_TOTAL_MB",
    "BackupManager",
    "PLUGIN_VERSION",
]
