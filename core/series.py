"""Versioned companions, ownership and shared evidence without replica stores."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..series_panels import (
    invalidate_presence_cache,
    is_series_group_excluded,
    resolve_series_timezone,
)
from .memory_scope import parse_value_list
from .memory_source import serialize_source_messages

# Contribution time is recorded by Games on receipt. Some adapters provide
# second precision or a skewed source clock; never substitute the claimed time.
ACTIVITY_SOURCE_CLOCK_SKEW = 300


def _activity_timestamp(raw):
    if type(raw) not in (int, float):
        return None
    try:
        ts = float(raw)
    except OverflowError:
        return None
    return ts if math.isfinite(ts) and ts > 0 else None


PLUGIN = "astrbot_plugin_dynamics_memory"
_FALLBACK_PLUGIN_VERSION = "0.2.0"
_VERSION_LINE_RE = re.compile(r"^version:\s*['\"]?([^\s'\"#]+)", re.MULTILINE)


def _load_plugin_version(
    metadata_path: Path | None = None, *, fallback: str = _FALLBACK_PLUGIN_VERSION
) -> str:
    """Dynamically read plugin version from metadata.yaml with safe fallback."""
    target = (
        metadata_path
        if metadata_path is not None
        else Path(__file__).resolve().parents[1] / "metadata.yaml"
    )
    try:
        raw = target.read_text(encoding="utf-8-sig")
    except OSError:
        return fallback
    match = _VERSION_LINE_RE.search(raw)
    if match:
        version = match.group(1).strip()
        if version:
            return version
    return fallback


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_from_ts(raw) -> str:
    ts = _activity_timestamp(raw)
    if ts is None:
        return _iso_now()
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return _iso_now()


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _compact_text(raw: str, limit: int = 72) -> str:
    line = " ".join(str(raw or "").split()).strip()
    if len(line) <= limit:
        return line
    return line[: max(1, limit - 1)].rstrip() + "…"


def _short_scope(umo: str) -> str:
    text = str(umo or "").strip()
    if not text:
        return ""
    parts = text.split(":")
    return parts[-1] if len(parts) >= 3 and parts[-1] else text[:32]


_TZ_SHANGHAI = timezone(timedelta(hours=8))
_UNBOUND_PERSONA_SENTINELS = frozenset({"__none__", "[%None]"})


def _day_bounds(
    date_str: str, *, now_ts: float | None = None, tz=None
) -> tuple[float, float] | None:
    zone = tz or _TZ_SHANGHAI
    raw = str(date_str or "").strip().lower()
    if not raw or raw == "today":
        stamp = time.time() if now_ts is None else float(now_ts)
        now_local = datetime.fromtimestamp(stamp, tz=zone)
        start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            return None
        start = parsed.replace(tzinfo=zone)
    start_ts = start.timestamp()
    return start_ts, (start + timedelta(days=1)).timestamp()


def _resolve_time_window(
    date_str: str,
    start_ts: float | None = None,
    end_ts: float | None = None,
    *,
    now_ts: float | None = None,
    tz=None,
) -> tuple[bool, tuple[float, float] | None]:
    if start_ts is not None or end_ts is not None:
        if (
            isinstance(start_ts, bool)
            or isinstance(end_ts, bool)
            or not isinstance(start_ts, (int, float))
            or not isinstance(end_ts, (int, float))
        ):
            return False, None
        s_val = float(start_ts)
        e_val = float(end_ts)
        if not (math.isfinite(s_val) and math.isfinite(e_val) and e_val > s_val):
            return False, None
        return True, (s_val, e_val)
    raw = str(date_str or "").strip().lower()
    if raw in {"all", "*"}:
        return True, None
    bounds = _day_bounds(raw, now_ts=now_ts, tz=tz)
    if bounds is None:
        return False, None
    return True, bounds


def _open_ro_sqlite(raw_path: Path | str | None) -> sqlite3.Connection | None:
    if not raw_path:
        return None
    text = str(raw_path).strip()
    if not text or text == ":memory:":
        return None
    try:
        path = Path(text)
        if not path.is_file():
            return None
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=0.15)
        except sqlite3.Error:
            conn = sqlite3.connect(str(path), timeout=0.15)
            conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA busy_timeout = 150")
        conn.row_factory = sqlite3.Row
        return conn
    except (OSError, sqlite3.Error, ValueError):
        return None


def value(row, key, default=None):
    return (
        row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)
    )


def active_entries(context):
    getter = getattr(context, "get_all_stars", None)
    if not callable(getter):
        return []
    rows = getter()
    if not isinstance(rows, (list, tuple, dict)):
        return []
    entries = (
        rows.items() if isinstance(rows, dict) else (("", row) for row in (rows or []))
    )
    result = []
    for key, row in entries:
        if not value(row, "activated", True):
            continue
        owner = value(row, "star_cls") or value(row, "star") or value(row, "plugin")
        if owner is not None and not isinstance(owner, type):
            result.append((str(value(row, "name") or key), owner))
    return result


def manifest(owner):
    getter = getattr(owner, "get_dynamics_capabilities", None)
    if not callable(getter) or inspect.iscoroutinefunction(getter):
        return {}
    try:
        data = getter()
        if inspect.isawaitable(data):
            if inspect.iscoroutine(data):
                data.close()
            return {}
        return (
            data
            if isinstance(data, dict)
            and type(data.get("schema_version")) is int
            and data["schema_version"] == 1
            else {}
        )
    except Exception:
        return {}


def find_role(context, role):
    found = []
    for _, owner in active_entries(context):
        data = manifest(owner)
        caps = data.get("capabilities", {})
        if (
            data.get("enabled") is True
            and isinstance(caps, dict)
            and type(caps.get(role)) is int
            and caps[role] == 1
        ):
            found.append((owner, data))
    if len(found) == 1 and found[0][1].get("ready") is True:
        return found[0][0]
    return None


class SeriesMixin:
    def series_conflicts(self):
        # Do not recursively call other memory manifests (they may also check us).
        return sorted(
            {
                name
                for name, owner in active_entries(getattr(self, "context", None))
                if owner is not self
                and (
                    name in {"astrbot_plugin_livingmemory", "LivingMemory", PLUGIN}
                    or (
                        callable(getattr(owner, "handle_memory_recall", None))
                        and callable(getattr(owner, "get_evidence_view", None))
                    )
                )
            }
        )

    def series_runtime_allowed(self):
        return (
            getattr(self, "_series_enabled", True) is True
            and not getattr(self, "_terminating", False)
            and not self.series_conflicts()
        )

    def get_dynamics_capabilities(self):
        return {
            "schema_version": 1,
            "provider_id": PLUGIN,
            "display_name": "群间记忆",
            "enabled": getattr(self, "_series_enabled", False) is True,
            "ready": self.series_runtime_allowed()
            and bool(getattr(self.initializer, "is_initialized", False)),
            "capabilities": {"memory": 1, "evidence": 1},
            "api_version": 1,
            "conflicts": self.series_conflicts(),
        }

    def _series_data_dir(self) -> Path | None:
        raw = getattr(self, "_data_dir", None)
        if not raw:
            initializer = getattr(self, "initializer", None)
            raw = getattr(initializer, "data_dir", None)
        if not raw:
            return None
        try:
            return Path(str(raw))
        except (TypeError, ValueError):
            return None

    def _series_memory_db_path(self) -> Path | None:
        initializer = getattr(self, "initializer", None)
        engine = getattr(initializer, "memory_engine", None)
        bm25 = getattr(
            getattr(engine, "hybrid_retriever", None), "bm25_retriever", None
        )
        for candidate in (
            getattr(bm25, "db_path", None),
            getattr(getattr(initializer, "db_migration", None), "db_path", None),
        ):
            if candidate and str(candidate).strip() not in {"", ":memory:"}:
                return Path(str(candidate))
        data_dir = self._series_data_dir()
        return (data_dir / "livingmemory.db") if data_dir is not None else None

    def _series_conversation_db_path(self) -> Path | None:
        initializer = getattr(self, "initializer", None)
        store = getattr(
            getattr(initializer, "conversation_manager", None), "store", None
        )
        candidate = getattr(store, "db_path", None)
        if candidate and str(candidate).strip() not in {"", ":memory:"}:
            return Path(str(candidate))
        data_dir = self._series_data_dir()
        return (data_dir / "conversations.db") if data_dir is not None else None

    def _count_pending_source_conflicts(self) -> int:
        conn = _open_ro_sqlite(self._series_conversation_db_path())
        if conn is None:
            return 0
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM series_source_conflicts WHERE status = 'pending'"
            ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0
        finally:
            conn.close()

    def _memory_health_snapshot(self) -> tuple[int, str]:
        conn = _open_ro_sqlite(self._series_memory_db_path())
        if conn is None:
            return 0, _iso_now()
        try:
            row = conn.execute(
                "SELECT COUNT(*), "
                "MAX(COALESCE(CAST(json_extract(metadata, '$.update_time') AS REAL), "
                "CAST(json_extract(metadata, '$.create_time') AS REAL), 0)) "
                "FROM documents "
                "WHERE json_valid(metadata) "
                "AND COALESCE(json_extract(metadata, '$.status'), 'active') = 'active'"
            ).fetchone()
            if not row:
                return 0, _iso_now()
            count = int(row[0] or 0)
            latest_ts = row[1]
            return count, _iso_from_ts(latest_ts) if latest_ts else _iso_now()
        except sqlite3.Error:
            return 0, _iso_now()
        finally:
            conn.close()

    def get_series_summary(self) -> dict:
        """Return lightweight read-only badge and pulse summary for series_presence v2."""
        try:
            provider_conflicts = self.series_conflicts()
            enabled = getattr(self, "_series_enabled", False) is True
            initialized = bool(
                getattr(getattr(self, "initializer", None), "is_initialized", False)
            )
            ready = self.series_runtime_allowed() and initialized
            pending_conflicts = self._count_pending_source_conflicts()
            active_memories, updated_at = self._memory_health_snapshot()

            if provider_conflicts:
                count = len(provider_conflicts)
                return {
                    "badge": {
                        "count": count,
                        "tone": "danger",
                        "label": f"{count} 项插件冲突",
                    },
                    "pulse": {
                        "headline": f"检测到旧记忆插件冲突（{', '.join(provider_conflicts[:2])}）",
                        "updated_at": _iso_now(),
                    },
                }
            if pending_conflicts > 0:
                return {
                    "badge": {
                        "count": pending_conflicts,
                        "tone": "warm",
                        "label": f"{pending_conflicts} 条来源冲突",
                    },
                    "pulse": {
                        "headline": f"{pending_conflicts} 条来源冲突待审查 · 共 {active_memories} 条活跃记忆",
                        "updated_at": updated_at,
                    },
                }
            if not enabled:
                return {
                    "badge": {"count": 0, "tone": "muted", "label": "待接入"},
                    "pulse": {
                        "headline": f"未开启系列记忆接入 · 现有 {active_memories} 条记忆",
                        "updated_at": updated_at,
                    },
                }
            if not ready:
                return {
                    "badge": {"count": 0, "tone": "info", "label": "初始化中"},
                    "pulse": {
                        "headline": "记忆引擎初始化中",
                        "updated_at": updated_at,
                    },
                }
            return {
                "badge": {"count": 0, "tone": "accent", "label": "运行正常"},
                "pulse": {
                    "headline": f"已沉淀 {active_memories} 条活跃记忆 · 来源无冲突",
                    "updated_at": updated_at,
                },
            }
        except Exception:
            return {
                "badge": {"count": 0, "tone": "muted", "label": "状态暂不可用"},
                "pulse": {"headline": "群间记忆状态暂不可用", "updated_at": _iso_now()},
            }

    def series_search_items(
        self, query: str, scope: str = "", limit: int = 20, persona: str = ""
    ) -> list[dict]:
        """Search memories and graph entities safely read-only for Ctrl+K federated search."""
        try:
            if not isinstance(query, str):
                return []
            keyword = query.strip()[:120]
            if not keyword:
                return []
            try:
                bounded_limit = max(1, min(int(limit), 50))
            except (TypeError, ValueError):
                bounded_limit = 20
            scope_filter = str(scope or "").strip()
            persona_filter = str(persona or "").strip()
            if persona_filter in _UNBOUND_PERSONA_SENTINELS:
                return []
            pattern = f"%{_escape_like(keyword)}%"

            conn = _open_ro_sqlite(self._series_memory_db_path())
            if conn is None:
                return []

            results: list[dict] = []
            try:
                mem_sql = (
                    "SELECT id, content, metadata FROM documents "
                    "WHERE json_valid(metadata) "
                    "AND COALESCE(json_extract(metadata, '$.status'), 'active') = 'active' "
                    "AND LOWER(COALESCE(content, '')) LIKE LOWER(?) ESCAPE '\\' "
                )
                params: list[object] = [pattern]
                if scope_filter:
                    mem_sql += "AND json_extract(metadata, '$.session_id') = ? "
                    params.append(scope_filter)
                if persona_filter:
                    mem_sql += "AND json_extract(metadata, '$.persona_id') = ? "
                    params.append(persona_filter)
                mem_sql += (
                    "ORDER BY COALESCE(CAST(json_extract(metadata, '$.importance') AS REAL), 0) DESC, "
                    "id DESC LIMIT ?"
                )
                params.append(bounded_limit)
                try:
                    for row in conn.execute(mem_sql, params).fetchall():
                        doc_id = int(row["id"])
                        try:
                            meta = json.loads(row["metadata"] or "{}")
                        except (TypeError, ValueError):
                            meta = {}
                        mem_type = str(meta.get("memory_type") or "FACT").upper()
                        sess = str(meta.get("session_id") or "")
                        sub_parts = [f"记忆 #{doc_id}", mem_type]
                        short_sess = _short_scope(sess)
                        if short_sess:
                            sub_parts.append(short_sess)
                        results.append(
                            {
                                "kind": "memory",
                                "title": _compact_text(
                                    row["content"] or f"记忆 #{doc_id}", 72
                                ),
                                "subtitle": " · ".join(sub_parts),
                                "plugin_id": PLUGIN,
                                "page": "dashboard",
                                "deep_link": {
                                    "plugin": PLUGIN,
                                    "page": "memory",
                                    "memory_id": doc_id,
                                    "umo": sess or scope_filter,
                                },
                            }
                        )
                except sqlite3.Error:
                    pass

                remaining = bounded_limit - len(results)
                if remaining > 0:
                    try:
                        if scope_filter or persona_filter:
                            node_sql = (
                                "SELECT DISTINCT n.id, n.node_type, n.node_value "
                                "FROM graph_nodes n "
                                "JOIN graph_entry_nodes en ON en.node_id = n.id "
                                "JOIN graph_entries e ON e.id = en.entry_id "
                                "WHERE (LOWER(n.node_value) LIKE LOWER(?) ESCAPE '\\' "
                                "OR LOWER(n.canonical_value) LIKE LOWER(?) ESCAPE '\\') "
                            )
                            node_params: list[object] = [pattern, pattern]
                            if scope_filter:
                                node_sql += "AND e.session_id = ? "
                                node_params.append(scope_filter)
                            if persona_filter:
                                node_sql += "AND e.persona_id = ? "
                                node_params.append(persona_filter)
                            node_sql += "ORDER BY n.id DESC LIMIT ?"
                            node_params.append(remaining)
                            node_rows = conn.execute(node_sql, node_params).fetchall()
                        else:
                            node_sql = (
                                "SELECT id, node_type, node_value FROM graph_nodes "
                                "WHERE LOWER(node_value) LIKE LOWER(?) ESCAPE '\\' "
                                "OR LOWER(canonical_value) LIKE LOWER(?) ESCAPE '\\' "
                                "ORDER BY id DESC LIMIT ?"
                            )
                            node_rows = conn.execute(
                                node_sql, (pattern, pattern, remaining)
                            ).fetchall()
                        for row in node_rows:
                            node_id = int(row["id"])
                            node_val = _compact_text(row["node_value"] or "", 72)
                            node_type = str(row["node_type"] or "entity")
                            results.append(
                                {
                                    "kind": "entity",
                                    "title": node_val or f"实体 #{node_id}",
                                    "subtitle": f"图谱实体 · {node_type}",
                                    "plugin_id": PLUGIN,
                                    "page": "dashboard",
                                    "deep_link": {
                                        "plugin": PLUGIN,
                                        "page": "graph",
                                        "node_id": node_id,
                                        "query": node_val,
                                        "umo": scope_filter,
                                    },
                                }
                            )
                    except sqlite3.Error:
                        pass
            finally:
                conn.close()
            return results[:bounded_limit]
        except Exception:
            return []

    def _scope_config_version(self, scope: str, enabled: bool | None) -> str:
        """Compute deterministic config-backed version string for optimistic locking."""
        cfg = getattr(self, "config_manager", None)
        whitelist_on = (
            bool(cfg.get("access_control.whitelist_enabled", False))
            if cfg is not None and callable(getattr(cfg, "get", None))
            else False
        )
        allowed_raw = (
            cfg.get("access_control.allowed_ids", "")
            if cfg is not None and callable(getattr(cfg, "get", None))
            else ""
        )
        disabled_raw = (
            cfg.get("access_control.disabled_ids", "")
            if cfg is not None and callable(getattr(cfg, "get", None))
            else ""
        )
        digest_src = "|".join(
            [
                str(scope or ""),
                str(int(bool(getattr(self, "_series_enabled", False)))),
                str(int(whitelist_on)),
                ",".join(sorted(parse_value_list(allowed_raw))),
                ",".join(sorted(parse_value_list(disabled_raw))),
            ]
        )
        digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:10]
        return f"v1:{int(bool(enabled))}:{digest}"

    def get_series_scope_status(self, umo: str) -> dict:
        """Return scope status for a group/session UMO in the cross-plugin matrix."""
        try:
            scope = str(umo or "").strip()
            conflicts = self.series_conflicts()
            runtime_allowed = self.series_runtime_allowed()
            ctx = getattr(self, "context", None)
            excluded_by_core = bool(scope and is_series_group_excluded(ctx, umo=scope))
            enabled: bool | None = runtime_allowed and not excluded_by_core
            cfg = getattr(self, "config_manager", None)
            writable = bool(
                cfg is not None and callable(getattr(cfg, "apply_updates", None))
            )
            whitelist_on = False
            disabled_locally = False
            parts = scope.split(":") if scope else []
            group_id = parts[-1] if len(parts) >= 3 else ""
            if cfg is not None and callable(getattr(cfg, "get", None)) and scope:
                whitelist_on = bool(cfg.get("access_control.whitelist_enabled", False))
                disabled_set = {
                    item.casefold()
                    for item in parse_value_list(
                        cfg.get("access_control.disabled_ids", "")
                    )
                }
                disabled_locally = bool(
                    scope.casefold() in disabled_set
                    or (group_id and group_id.casefold() in disabled_set)
                )
                if enabled and disabled_locally:
                    enabled = False
                elif enabled and whitelist_on:
                    allowed = {
                        item.casefold()
                        for item in parse_value_list(
                            cfg.get("access_control.allowed_ids", "")
                        )
                    }
                    enabled = bool(
                        scope.casefold() in allowed
                        or (group_id and group_id.casefold() in allowed)
                    )

            total_count = 0
            activity_today = 0
            persona_id = ""
            day_bounds = _day_bounds("today", tz=resolve_series_timezone(ctx))
            start_ts, end_ts = day_bounds if day_bounds else (0.0, time.time() + 1.0)
            conn = _open_ro_sqlite(self._series_memory_db_path())
            if conn is not None and scope:
                try:
                    row = conn.execute(
                        "SELECT COUNT(*), "
                        "SUM(CASE WHEN COALESCE(CAST(json_extract(metadata, '$.create_time') AS REAL), 0) >= ? "
                        "AND COALESCE(CAST(json_extract(metadata, '$.create_time') AS REAL), 0) < ? THEN 1 ELSE 0 END) "
                        "FROM documents "
                        "WHERE json_valid(metadata) "
                        "AND COALESCE(json_extract(metadata, '$.status'), 'active') = 'active' "
                        "AND json_extract(metadata, '$.session_id') = ?",
                        (start_ts, end_ts, scope),
                    ).fetchone()
                    if row:
                        total_count = int(row[0] or 0)
                        activity_today = int(row[1] or 0)
                    p_row = conn.execute(
                        "SELECT json_extract(metadata, '$.persona_id') FROM documents "
                        "WHERE json_valid(metadata) "
                        "AND COALESCE(json_extract(metadata, '$.status'), 'active') = 'active' "
                        "AND json_extract(metadata, '$.session_id') = ? "
                        "AND COALESCE(json_extract(metadata, '$.persona_id'), '') != '' "
                        "ORDER BY id DESC LIMIT 1",
                        (scope,),
                    ).fetchone()
                    if p_row and p_row[0]:
                        persona_id = str(p_row[0])
                except sqlite3.Error:
                    pass
                finally:
                    conn.close()

            if conflicts:
                summary = f"存在旧插件冲突（{conflicts[0]}）"
                mode = "conflict"
            elif excluded_by_core:
                summary = f"主控排除 · 现有 {total_count} 条记忆"
                mode = "excluded_by_core"
            elif not getattr(self, "_series_enabled", False):
                summary = f"未开启接入 · 现有 {total_count} 条记忆"
                mode = "runtime_disabled"
            elif disabled_locally:
                summary = f"已按群关闭 · 现有 {total_count} 条记忆"
                mode = "disabled"
            elif enabled is False:
                summary = f"未在白名单 · 现有 {total_count} 条记忆"
                mode = "whitelist"
            else:
                summary = f"共 {total_count} 条记忆 · 今日 +{activity_today}"
                mode = "whitelist" if whitelist_on else "open"

            return {
                "umo": scope,
                "enabled": enabled,
                "writable": writable,
                "supported": writable,
                "mode": mode,
                "excluded_by_core": excluded_by_core,
                "persona_id": persona_id,
                "activity_today": activity_today,
                "summary": summary,
                "detail": summary,
                "version": self._scope_config_version(scope, enabled),
            }
        except Exception:
            return {
                "umo": str(umo or "").strip(),
                "enabled": None,
                "writable": False,
                "supported": False,
                "mode": "error",
                "persona_id": "",
                "activity_today": None,
                "summary": "群间记忆状态暂不可用",
                "detail": "群间记忆状态暂不可用",
                "version": "0",
            }

    async def set_series_scope_enabled(
        self,
        umo: str,
        enabled: bool,
        expected_version: str | int | None = None,
    ) -> dict[str, Any]:
        """Optimistically toggle whether a UMO is enabled in Dynamics Memory."""
        scope = str(umo or "").strip()
        if not scope:
            return {
                "ok": False,
                "umo": "",
                "enabled": False,
                "version": "0",
                "reason": "invalid_umo",
            }
        current = self.get_series_scope_status(scope)
        current_version = str(current.get("version") or "0")
        if (
            expected_version is not None
            and str(expected_version).strip() != ""
            and str(expected_version) != current_version
        ):
            return {
                "ok": False,
                "umo": scope,
                "enabled": bool(current.get("enabled")),
                "version": current_version,
                "reason": "version_conflict",
            }
        if self.series_conflicts():
            return {
                "ok": False,
                "umo": scope,
                "enabled": False,
                "version": current_version,
                "reason": "plugin_conflict",
            }

        cfg = getattr(self, "config_manager", None)
        apply_updates = getattr(cfg, "apply_updates", None) if cfg is not None else None
        if not callable(apply_updates):
            return {
                "ok": False,
                "umo": scope,
                "enabled": bool(current.get("enabled")),
                "version": current_version,
                "reason": "config_not_writable",
            }

        whitelist_on = bool(cfg.get("access_control.whitelist_enabled", False))
        allowed_list = parse_value_list(cfg.get("access_control.allowed_ids", ""))
        disabled_list = parse_value_list(cfg.get("access_control.disabled_ids", ""))
        parts = scope.split(":")
        group_id = parts[-1] if len(parts) >= 3 else ""
        match_keys = {scope.casefold()}
        if group_id:
            match_keys.add(group_id.casefold())

        updates: dict[str, Any] = {}
        if enabled:
            new_disabled = [
                item for item in disabled_list if item.casefold() not in match_keys
            ]
            if new_disabled != disabled_list:
                updates["access_control.disabled_ids"] = "\n".join(new_disabled)
            if whitelist_on:
                if not any(item.casefold() in match_keys for item in allowed_list):
                    allowed_list.append(scope)
                updates["access_control.allowed_ids"] = "\n".join(allowed_list)
            elif not updates:
                updates["access_control.disabled_ids"] = "\n".join(new_disabled)
        else:
            if whitelist_on:
                new_allowed = [
                    item for item in allowed_list if item.casefold() not in match_keys
                ]
                updates["access_control.allowed_ids"] = "\n".join(new_allowed)
            else:
                if not any(item.casefold() in match_keys for item in disabled_list):
                    disabled_list.append(scope)
                updates["access_control.disabled_ids"] = "\n".join(disabled_list)

        try:
            apply_updates(updates)
        except Exception as exc:
            return {
                "ok": False,
                "umo": scope,
                "enabled": bool(current.get("enabled")),
                "version": current_version,
                "reason": f"save_failed:{type(exc).__name__}",
            }

        try:
            invalidate_presence_cache(PLUGIN, context=getattr(self, "context", None))
        except Exception:
            pass

        updated = self.get_series_scope_status(scope)
        reason = (
            "excluded_by_core"
            if enabled and updated.get("excluded_by_core")
            else "runtime_disabled"
            if enabled and not getattr(self, "_series_enabled", False)
            else "updated"
        )
        return {
            "ok": True,
            "umo": scope,
            "enabled": bool(updated.get("enabled")),
            "version": str(updated.get("version") or "0"),
            "reason": reason,
        }

    async def activate_series_runtime(self) -> dict[str, Any]:
        """Activate Dynamics Memory series runtime (used by Setup Wizard step 4)."""
        conflicts = self.series_conflicts()
        if conflicts:
            return {
                "ok": False,
                "plugin_id": PLUGIN,
                "enabled": bool(getattr(self, "_series_enabled", False)),
                "ready": False,
                "reason": "plugin_conflict",
                "conflicts": conflicts,
            }
        if getattr(self, "_terminating", False):
            return {
                "ok": False,
                "plugin_id": PLUGIN,
                "enabled": False,
                "ready": False,
                "reason": "terminating",
            }

        self._series_enabled = True
        cfg = getattr(self, "config_manager", None)
        persisted = False
        if cfg is not None:
            apply_updates = getattr(cfg, "apply_updates", None)
            if callable(apply_updates):
                try:
                    apply_updates({"series_runtime_enabled": True})
                    persisted = True
                except Exception:
                    persisted = False
            if not persisted:
                raw_cfg = getattr(cfg, "_raw_config", None)
                if isinstance(raw_cfg, dict):
                    raw_cfg["series_runtime_enabled"] = True
                    save_fn = getattr(raw_cfg, "save_config", None)
                    if callable(save_fn):
                        try:
                            save_fn()
                            persisted = True
                        except Exception:
                            pass

        initializer = getattr(self, "initializer", None)
        is_initialized = bool(getattr(initializer, "is_initialized", False))
        if not is_initialized:
            init_coro_fn = getattr(self, "_initialize_plugin", None)
            create_task = getattr(self, "_create_tracked_task", None)
            if callable(init_coro_fn) and callable(create_task):
                try:
                    create_task(init_coro_fn())
                except RuntimeError:
                    pass
            else:
                wake = getattr(initializer, "wake_retry", None)
                if callable(wake):
                    wake(reset_attempts=True)
        else:
            ensure_rt = getattr(self, "_ensure_runtime_components", None)
            if callable(ensure_rt):
                res = ensure_rt()
                if inspect.isawaitable(res):
                    await res

        try:
            invalidate_presence_cache(PLUGIN, context=getattr(self, "context", None))
        except Exception:
            pass

        return {
            "ok": True,
            "plugin_id": PLUGIN,
            "enabled": True,
            "ready": bool(getattr(initializer, "is_initialized", False)),
            "persisted": persisted,
            "reason": "activated",
        }

    def get_journal_events(
        self,
        persona: str = "",
        scope: str = "",
        date: str = "today",
        limit: int = 20,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> list[dict]:
        """Return today's important memory events for World's Living Journal feed."""
        try:
            tz = resolve_series_timezone(getattr(self, "context", None))
            ok, bounds = _resolve_time_window(date, start_ts, end_ts, tz=tz)
            if not ok:
                return []
            try:
                bounded_limit = max(1, min(int(limit), 50))
            except (TypeError, ValueError):
                bounded_limit = 20
            persona_filter = str(persona or "").strip()
            scope_filter = str(scope or "").strip()
            if persona_filter in _UNBOUND_PERSONA_SENTINELS:
                return []
            if scope_filter and not persona_filter:
                return []

            conn = _open_ro_sqlite(self._series_memory_db_path())
            if conn is None:
                return []
            events: list[dict] = []
            try:
                sql = (
                    "SELECT id, content, metadata FROM documents "
                    "WHERE json_valid(metadata) "
                    "AND COALESCE(json_extract(metadata, '$.status'), 'active') = 'active' "
                )
                params: list[object] = []
                if bounds is not None:
                    sql += (
                        "AND COALESCE(CAST(json_extract(metadata, '$.create_time') AS REAL), 0) >= ? "
                        "AND COALESCE(CAST(json_extract(metadata, '$.create_time') AS REAL), 0) < ? "
                    )
                    params.extend(bounds)
                if scope_filter:
                    sql += "AND json_extract(metadata, '$.session_id') = ? "
                    params.append(scope_filter)
                if persona_filter:
                    sql += "AND json_extract(metadata, '$.persona_id') = ? "
                    params.append(persona_filter)
                sql += (
                    "ORDER BY COALESCE(CAST(json_extract(metadata, '$.create_time') AS REAL), 0) DESC, "
                    "COALESCE(CAST(json_extract(metadata, '$.importance') AS REAL), 0) DESC, "
                    "id DESC LIMIT ?"
                )
                params.append(bounded_limit)
                for row in conn.execute(sql, params).fetchall():
                    doc_id = str(row["id"])
                    try:
                        meta = json.loads(row["metadata"] or "{}")
                    except (TypeError, ValueError):
                        meta = {}
                    mem_type = str(meta.get("memory_type") or "FACT").upper()
                    try:
                        imp = float(meta.get("importance") or 0.5)
                    except (TypeError, ValueError):
                        imp = 0.5
                    is_high = (imp <= 1.0 and imp >= 0.75) or imp >= 7.5
                    tone = (
                        "warm"
                        if is_high
                        else "info"
                        if mem_type in {"EVENT", "EPISODIC"}
                        else "accent"
                    )
                    sess = str(meta.get("session_id") or "")
                    sub_parts = [mem_type, f"记忆 #{doc_id}"]
                    short_sess = _short_scope(sess)
                    if short_sess:
                        sub_parts.append(short_sess)
                    events.append(
                        {
                            "id": f"memory:{doc_id}",
                            "category": "memory",
                            "source_label": "群间记忆",
                            "tone": tone,
                            "occurred_at": _iso_from_ts(meta.get("create_time")),
                            "title": _compact_text(
                                row["content"] or f"沉淀新记忆 #{doc_id}", 64
                            ),
                            "subtitle": " · ".join(sub_parts),
                            "target_plugin": PLUGIN,
                            "target_route": "dashboard",
                            "target_id": doc_id,
                        }
                    )
            except sqlite3.Error:
                return []
            finally:
                conn.close()
            return events
        except Exception:
            return []

    async def get_shared_evidence(
        self, scope, source_refs, *, purpose="reply", card_version=""
    ):
        """World/learning read existing validated references; never ingest their drafts.

        Keep source checks in get_evidence_view. Callers cannot turn learned
        policies, world plans or their own prose into new user observations.
        """
        if not isinstance(source_refs, list) or len(source_refs) > 64:
            raise ValueError("source_refs must be a bounded list")
        return await self.get_evidence_view(
            scope,
            source_refs=source_refs,
            purpose=purpose,
            card_version=card_version,
            allow_derivation=False,
        )

    async def get_series_context(
        self,
        scope,
        source_refs,
        *,
        purpose="reply",
        card_version="",
        learning_request=None,
    ):
        """Read-only composite, with separate owners and no second prompt injection."""
        if not self.series_runtime_allowed() or not await self._narrative_scope_allowed(
            scope
        ):
            return {
                "schema_version": 1,
                "available": False,
                "reason": "scope_or_owner_unavailable",
            }
        world = find_role(self.context, "character")
        world_data = None
        if world is not None:
            try:
                world_data = await asyncio.wait_for(
                    world.get_character_context(scope), 1.0
                )
            except (TimeoutError, ValueError, RuntimeError):
                world_data = None
            if find_role(self.context, "character") is not world:
                world_data = None
        learning = None
        # Learning uses a different, intentionally narrower request contract.
        # The caller must supply the exact current persona fingerprint/baseline.
        if isinstance(learning_request, dict) and learning_request.get(
            "scope"
        ) == scope.get("umo"):
            required = {"scope", "persona", "member", "baseline", "host_version"}
            if set(learning_request) == required and all(
                isinstance(v, str) for v in learning_request.values()
            ):
                candidates = [
                    p
                    for n, p in active_entries(self.context)
                    if n == "astrbot_plugin_dynamics_learning"
                ]
                if len(candidates) == 1 and await self._narrative_scope_allowed(
                    scope, learning_request["member"]
                ):
                    owner = candidates[0]
                    getter = getattr(owner, "get_learning_context", None)
                    if callable(getter) and not inspect.iscoroutinefunction(getter):
                        learning = getter(**learning_request)
                        if inspect.isawaitable(learning):
                            if inspect.iscoroutine(learning):
                                learning.close()
                            learning = None
        evidence = await self.get_shared_evidence(
            scope, source_refs, purpose=purpose, card_version=card_version
        )
        if world is not None and find_role(self.context, "character") is not world:
            world_data = None
        if learning is not None and not any(
            n == "astrbot_plugin_dynamics_learning" and p is owner
            for n, p in active_entries(self.context)
        ):
            learning = None
        if not self.series_runtime_allowed() or not await self._narrative_scope_allowed(
            scope
        ):
            return {
                "schema_version": 1,
                "available": False,
                "reason": "source_invalidated",
            }
        return {
            "schema_version": 1,
            "available": True,
            "memory": evidence,
            "world": world_data,
            "learning": learning,
            "ownership": {
                "memory": "observations_and_evidence",
                "world": "character_and_life",
                "learning": "reviewed_behavior",
            },
        }

    def register_series_routes(self):
        from ..series_workspace import register_panel_api

        for path, handler, methods, title in (
            ("series", self.series_status, ["GET"], "群间记忆接入与冲突状态"),
            (
                "series/conflicts",
                self.series_conflict_list,
                ["GET"],
                "群间记忆来源冲突",
            ),
            ("series/detail", self.series_conflict_detail, ["GET"], "查看来源冲突差异"),
            (
                "series/resolve",
                self.series_resolve_conflict,
                ["POST"],
                "标记来源冲突已审查",
            ),
        ):
            register_panel_api(
                self,
                self.context.register_web_api,
                f"/{PLUGIN}/page/{path}",
                self._series_route(handler),
                methods,
                title,
            )

    @staticmethod
    def _series_route(handler):
        async def serve():
            from ..series_panels import _web_json_response

            try:
                return _web_json_response(await handler())
            except PermissionError as exc:
                return _web_json_response(
                    {
                        "status": "error",
                        "message": str(exc) or "需要 Dashboard 管理员登录",
                    },
                    status_code=403,
                )
            except ValueError:
                return _web_json_response(
                    {
                        "status": "error",
                        "message": "参数无效或来源冲突版本已改变，请刷新后重试。",
                    },
                    status_code=409,
                )

        return serve

    async def _series_admin(self, *, check_csrf: bool = False):
        from ..series_panels import require_series_admin

        return await require_series_admin(self.context, check_csrf=check_csrf)

    async def series_status(self):
        await self._series_admin()
        entries = active_entries(self.context)
        return {
            "status": "ok",
            "data": {
                **self.get_dynamics_capabilities(),
                "base": "LivingMemory 2.10.0+local.3",
                "version": _load_plugin_version(),
                "world_ready": find_role(self.context, "character") is not None,
                "learning_installed": any(
                    n == "astrbot_plugin_dynamics_learning" for n, _ in entries
                ),
                "games_installed": any(
                    n == "astrbot_plugin_dynamics_games" for n, _ in entries
                ),
                "migration": "migration.py：预览后在源与目标均停用时导入；旧数据保持原样。",
                "data_ownership": "世界维护设定与生活；学习维护行为；游戏负责活动进度与成果；记忆维护经历与事实来源。",
            },
        }

    async def series_conflict_list(self):
        await self._series_admin()
        ready, error = await self._ensure_plugin_ready(wait=False)
        if not ready:
            return {"status": "error", "message": error}
        store = self.initializer.conversation_manager.store
        return {"status": "ok", "data": await store.list_series_conflicts()}

    async def series_resolve_conflict(self):
        actor = await self._series_admin(check_csrf=True)
        from astrbot.api.web import request

        body = await request.json()
        if not isinstance(body, dict) or set(body) != {"id", "revision", "reason"}:
            raise ValueError("invalid conflict review")
        ready, error = await self._ensure_plugin_ready(wait=False)
        if not ready:
            return {"status": "error", "message": error}
        result = (
            await self.initializer.conversation_manager.store.resolve_series_conflict(
                body["id"], body["revision"], body["reason"], actor
            )
        )
        return {"status": "ok", "data": result}

    async def series_conflict_detail(self):
        await self._series_admin()
        from astrbot.api.web import request

        identifier = request.query.get("id", "")
        ready, error = await self._ensure_plugin_ready(wait=False)
        if not ready:
            return {"status": "error", "message": error}
        data = await self.initializer.conversation_manager.store.get_series_conflict(
            identifier
        )
        return {"status": "ok", "data": data}

    async def _resolve_activity_sources(self, store, scope, refs, occurred_at):
        sources = await store.get_activity_sources(scope["umo"], list(refs))
        ordered = []
        for message_id, (uid, claimed_time) in refs.items():
            matches = sources.get(message_id, ())
            if not matches:
                return [], "source_unresolved"
            rows = []
            for row in matches:
                stamp = _activity_timestamp(row.get("timestamp"))
                source_scope = (row.get("metadata") or {}).get("scope") or {}
                source_umo = source_scope.get("umo") or row.get("session_id")
                if (
                    source_umo != scope["umo"]
                    or row.get("sender_id") != uid
                    or row.get("role") != "user"
                    or stamp is None
                    or abs(stamp - claimed_time) > ACTIVITY_SOURCE_CLOCK_SKEW
                    or stamp > occurred_at + ACTIVITY_SOURCE_CLOCK_SKEW
                ):
                    return [], "source_identity_mismatch"
                platform = str(row.get("platform") or scope["platform"])
                if not await self._memory_scope_allowed(
                    {**scope, "platform": platform}, uid
                ):
                    return [], "source_scope_denied"
                rows.append(row)
            # One physical delivery can exist under multiple managed personas
            # and in passive history. Conflicting content is ambiguous.
            if any(row["content"] != rows[0]["content"] for row in rows[1:]):
                return [], "source_ambiguous"
            row = rows[0]
            retained = serialize_source_messages([row])
            if len(retained) != 1:
                return [], "source_unresolved"
            retained[0]["metadata"]["platform_message_id"] = message_id
            ordered.extend(retained)
        return ordered, ""

    async def record_activity_experience(self, payload: dict) -> dict:
        """Archive a group activity and attach only verified original messages."""

        def result(reason="", doc_id=0):
            response = {"schema_version": 1, "recorded": bool(doc_id and not reason)}
            if reason:
                response["reason"] = reason
            if doc_id:
                response["doc_id"] = doc_id
            return response

        if (
            not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
        ):
            return result("invalid_payload")
        if not self.series_runtime_allowed():
            return result("runtime_not_allowed")
        initializer = getattr(self, "initializer", None)
        if not getattr(initializer, "is_initialized", False):
            return result("not_initialized")
        engine = getattr(initializer, "memory_engine", None)
        if not callable(getattr(engine, "add_memory", None)):
            return result("engine_unavailable")

        supplied_scope = payload.get("scope")
        umo = supplied_scope.get("umo") if isinstance(supplied_scope, dict) else ""
        parts = umo.split(":") if isinstance(umo, str) else []
        if (
            len(parts) != 3
            or parts[1] != "GroupMessage"
            or any(not p or p.strip() != p for p in parts)
        ):
            return result("invalid_group_umo")
        platform = parts[0]
        if supplied_scope.get("platform") != platform:
            return result("invalid_platform")
        scope = {"umo": umo, "platform": platform}
        activity_id = str(payload.get("activity_id") or "").strip()[:64]
        event_kind = str(payload.get("event_kind") or "completed").strip()[:32]
        category = str(payload.get("category") or "creation").strip()[:32]
        category_label = str(payload.get("category_label") or "协作活动").strip()[:32]
        title = str(payload.get("title") or "").strip()[:200]
        summary = str(payload.get("summary") or "").strip()[:2200]
        if not activity_id or not title or not summary:
            return result("missing_fields")
        occurred_at = _activity_timestamp(payload.get("occurred_at"))
        if (
            occurred_at is None
            or occurred_at > time.time() + ACTIVITY_SOURCE_CLOCK_SKEW
        ):
            return result("invalid_occurred_at")

        participants = payload.get("participants")
        if not isinstance(participants, list) or not 1 <= len(participants) <= 50:
            return result("invalid_participants")
        members = {}
        for participant in participants:
            uid = participant.get("uid") if isinstance(participant, dict) else None
            if not isinstance(uid, str) or not 0 < len(uid) <= 80 or uid.strip() != uid:
                return result("invalid_participants")
            members[uid] = str(participant.get("name") or uid)[:80]

        raw_refs = payload.get("source_refs", [])
        if not isinstance(raw_refs, list) or len(raw_refs) > 32:
            return result("invalid_source_refs")
        refs = {}
        for raw in raw_refs:
            if not isinstance(raw, dict):
                return result("invalid_source_refs")
            message_id, uid = raw.get("message_id"), raw.get("uid")
            ts = _activity_timestamp(raw.get("ts"))
            if (
                not isinstance(message_id, str)
                or not 0 < len(message_id) <= 128
                or message_id.strip() != message_id
                or not isinstance(uid, str)
                or uid not in members
                or ts is None
                or ts > occurred_at + ACTIVITY_SOURCE_CLOCK_SKEW
            ):
                return result("invalid_source_refs")
            if message_id in refs and refs[message_id] != (uid, ts):
                return result("invalid_source_refs")
            refs[message_id] = (uid, ts)

        async def activity_scope_allowed():
            for uid in members:
                if not await self._memory_scope_allowed(scope, uid):
                    return False
            return True

        if not await activity_scope_allowed():
            return result("scope_not_allowed")
        manager = getattr(initializer, "conversation_manager", None)
        store = getattr(manager, "store", None)
        lease = getattr(manager, "source_publication_lease", None)
        if not callable(lease) or (
            refs and not callable(getattr(store, "get_activity_sources", None))
        ):
            return result("source_store_unavailable")

        exp_key = f"dg:{umo}:{activity_id}:{event_kind}"
        bm25 = getattr(
            getattr(engine, "hybrid_retriever", None), "bm25_retriever", None
        )
        if not callable(getattr(bm25, "_read_connect", None)):
            return result("memory_index_unavailable")

        lock = getattr(self, "_dg_experience_lock", None)
        if lock is None:
            lock = self._dg_experience_lock = asyncio.Lock()
        async with lock:
            saved_refs = {
                message_id: list(identity) for message_id, identity in refs.items()
            }
            async with bm25._read_connect() as db:
                cursor = await db.execute(
                    f"SELECT d.id, json_extract(d.metadata, '$.activity_source_verified_v1'), "
                    "json_extract(d.metadata, '$.occurred_at'), "
                    "json_extract(d.metadata, '$.activity_source_refs') "
                    f"FROM {bm25.doc_table} d "
                    "WHERE json_valid(d.metadata) "
                    "AND json_extract(d.metadata, '$.experience_key') = ? "
                    "AND COALESCE(json_extract(d.metadata, '$.status'), 'active') IN ('active', 'archived') "
                    "LIMIT 1",
                    (exp_key,),
                )
                prior = await cursor.fetchone()
                if prior:
                    if prior[1] != 1:
                        return result("legacy_experience_unverified")
                    if (
                        prior[2] != occurred_at
                        or json.loads(prior[3] or "{}") != saved_refs
                    ):
                        return result("activity_source_conflict")
                    return result("duplicate", int(prior[0]))

            source_messages = []
            if refs:
                source_messages, error = await self._resolve_activity_sources(
                    store, scope, refs, occurred_at
                )
                if error:
                    return result(error)

            async def sources_current():
                if not await activity_scope_allowed():
                    return False
                if not refs:
                    return True
                current, error = await self._resolve_activity_sources(
                    store, scope, refs, occurred_at
                )
                return not error and current == source_messages

            names = "、".join(list(members.values())[:20])
            content = f"[群间共同经历·{category_label}]《{title}》：参与成员 {names}。{summary}"
            try:
                doc_id = await engine.add_memory(
                    content=content[:2400],
                    session_id=umo,
                    importance=0.78 if event_kind == "completed" else 0.62,
                    metadata={
                        "memory_type": "event",
                        "source_plugin": "astrbot_plugin_dynamics_games",
                        "activity_id": activity_id,
                        "activity_category": category,
                        "event_kind": event_kind,
                        "experience_key": exp_key,
                        "participant_uids": list(members),
                        "occurred_at": occurred_at,
                        "create_time": occurred_at,
                        "activity_source_verified_v1": 1,
                        "activity_source_refs": saved_refs,
                    },
                    preserve_create_time=True,
                    source_messages=source_messages or None,
                    source_validator=sources_current,
                    source_publish_lease=lambda: lease(sources_current),
                )
            except ValueError as exc:
                if str(exc) != "source_snapshot_changed":
                    raise
                return result("source_snapshot_changed")
            return result(doc_id=doc_id) if doc_id else result("memory_write_failed")
