"""Offline, non-destructive LivingMemory migration. Standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_file(path):
    with path.open("rb") as f:
        return f.read(16) == b"SQLite format 3\x00"


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)


def database_summary(path):
    with closing(readonly(path)) as conn:
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("database integrity failed: " + path.name)
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            name: conn.execute(
                'SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"'
            ).fetchone()[0]
            for name in tables
        }


def snapshot(source, config):
    source, config = Path(source).resolve(), Path(config).resolve()
    if not source.is_dir() or not config.is_file():
        raise ValueError("source data directory/config not found")
    files, databases = {}, {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink sources are not accepted")
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        # Old backups remain intact at the source. Copy active stores, indexes,
        # stopwords and custom assets; transient rebuilds are not live indexes.
        if (
            "backups" in path.relative_to(source).parts
            or ".bak" in path.name
            or ".corrupt_" in path.name
            or ".tmp" in path.name
        ):
            continue
        if path.name.endswith("-shm") or (
            path.name.endswith("-wal") and path.stat().st_size == 0
        ):
            continue
        files[rel] = {"sha256": sha(path), "size": path.stat().st_size}
        if not path.name.endswith("-wal") and sqlite_file(path):
            databases[rel] = database_summary(path)
    settings = json.loads(config.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("configuration must be an object")
    payload = {"files": files, "databases": databases, "config_sha256": sha(config)}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return {
        **payload,
        "fingerprint": fingerprint,
        "file_count": len(files),
        "bytes": sum(x["size"] for x in files.values()),
        "configuration_keys": sorted(settings),
        "source": str(source),
    }


def migrate(source, config, destination, fingerprint, *, offline=False):
    """Require quiescent source AND target; caller stops both via host first."""
    if not offline:
        raise ValueError("stop source and target plugins, then pass --offline")
    source, config, destination = (
        Path(source).resolve(),
        Path(config).resolve(),
        Path(destination).resolve(),
    )
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("source and destination must be independent")
    target_was_empty = destination.is_dir() and not any(destination.iterdir())
    if destination.exists() and not target_was_empty:
        raise ValueError(
            "destination must be absent or empty; do not overwrite initialized data"
        )
    before = snapshot(source, config)
    if fingerprint != before["fingerprint"]:
        raise ValueError("source changed; preview again")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".dynamics-memory-import-", dir=destination.parent)
    ).resolve()
    try:
        for rel in before["files"]:
            if rel.endswith("-wal"):
                continue
            src, dst = source / rel, staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if rel in before["databases"]:
                with (
                    closing(readonly(src)) as src_db,
                    closing(sqlite3.connect(dst)) as dst_db,
                ):
                    src_db.backup(dst_db)
                    dst_db.commit()
                if database_summary(dst) != before["databases"][rel]:
                    raise ValueError("database row counts changed: " + rel)
            else:
                shutil.copy2(src, dst)
                if sha(dst) != before["files"][rel]["sha256"]:
                    raise ValueError("file changed: " + rel)
        if snapshot(source, config)["fingerprint"] != fingerprint:
            raise ValueError("source changed during copy; keep plugins stopped")
        settings = json.loads(config.read_text(encoding="utf-8-sig"))
        settings["series_runtime_enabled"] = False
        (staging / "migration-settings.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = {
            **before,
            "source_unchanged": True,
            "runtime_enabled": False,
            "excluded": [
                "historical backups",
                "corrupt indexes",
                "rebuild temporary files",
                "SQLite shared-memory files",
            ],
            "config_application": "Copy migration-settings.json to the NEW plugin config while stopped; then review runtime switch.",
        }
        (staging / "migration-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Rename to a previously absent path; neither live data directory is replaced.
        if target_was_empty:
            destination.rmdir()  # Fails if another process wrote anything meanwhile.
        os.rename(staging, destination)
        return report
    except BaseException:
        # Only the verified temporary directory created above is removed.
        if staging.parent == destination.parent and staging.name.startswith(
            ".dynamics-memory-import-"
        ):
            shutil.rmtree(staging)
        if target_was_empty and not destination.exists():
            destination.mkdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--destination")
    parser.add_argument("--apply-fingerprint")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.apply_fingerprint:
        if not args.destination:
            parser.error("--destination required for import")
        result = migrate(
            args.source,
            args.config,
            args.destination,
            args.apply_fingerprint,
            offline=args.offline,
        )
    else:
        result = snapshot(args.source, args.config)
    # Avoid printing imported model settings or source text.
    print(
        json.dumps(
            {k: result[k] for k in ("fingerprint", "file_count", "bytes", "databases")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
