"""Resolve access control, user aliases, and memory isolation scopes."""

from __future__ import annotations

import inspect
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .. import series_panels
from ..series_panels import is_series_group_excluded

GLOBAL_MEMORY_SCOPE = "livingmemory:global"


@dataclass(frozen=True)
class AdapterAccess:
    """Adapter ownership and allowed instances for one request."""

    adapter_id: str
    adapter_ids: tuple[str, ...] | None


current_adapter_access: ContextVar[AdapterAccess | None] = ContextVar(
    "livingmemory_adapter_access", default=None
)


@contextmanager
def adapter_access(config: Any, event: Any, *, write: bool = False):
    """Apply adapter policy to this task and reset it when the request finishes."""
    adapter_id = (
        event if isinstance(event, str) else _event_value(event, "get_platform_id")
    )
    if not adapter_id:
        adapter_id = _event_value(event, "unified_msg_origin").partition(":")[0]
    shared = _config_get(config, "filtering_settings.shared_adapter_ids")
    allowed = None
    if write:
        allowed = (adapter_id,)
    elif shared is not None:
        members = parse_value_list(shared)
        allowed = tuple(sorted(members)) if adapter_id in members else (adapter_id,)
    token = current_adapter_access.set(AdapterAccess(adapter_id, allowed))
    try:
        yield
    finally:
        current_adapter_access.reset(token)


def adapter_sql(metadata: str) -> tuple[str, list[str]]:
    """Build a parameterized ownership predicate for an internal SQL expression."""
    access = current_adapter_access.get()
    if access is None or access.adapter_ids is None:
        return "1", []
    values = list(access.adapter_ids)
    return (
        f"json_extract({metadata}, '$.adapter_id') IN ({','.join('?' for _ in values)})",
        values,
    )


def adapter_visible(metadata: dict[str, Any]) -> bool:
    """Apply the current adapter boundary to a loaded memory."""
    access = current_adapter_access.get()
    return (
        access is None
        or access.adapter_ids is None
        or metadata.get("adapter_id") in access.adapter_ids
    )


def adapter_cache_key() -> tuple[str, ...] | None:
    """Return the adapter dimension for request-independent retrieval caches."""
    access = current_adapter_access.get()
    return access.adapter_ids if access is not None else None


def infer_adapter_id(
    metadata: dict[str, Any], sources=(), instances=None
) -> str | None:
    """Recover ownership from structured provenance without matching user IDs."""
    if metadata.get("adapter_id"):
        return str(metadata["adapter_id"])
    raw_sources = [item.get("session_id") for item in sources if isinstance(item, dict)]
    if not raw_sources:
        window = metadata.get("source_window") or {}
        raw_sources = [metadata.get("source_session_id") or window.get("session_id")]
    if not any(raw_sources):
        raw_sources = [metadata.get("old_session_id") or metadata.get("session_id")]
    owners = set()
    for value in raw_sources:
        parts = str(value or "").split(":", 2)
        if len(parts) != 3 or parts[0] == "livingmemory":
            continue
        prefix = parts[0]
        if instances and prefix not in instances:
            matching = [key for key, kind in instances.items() if kind == prefix]
            if len(matching) > 1:
                return None
            if matching:
                prefix = matching[0]
        owners.add(prefix)
    return owners.pop() if len(owners) == 1 else None


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        current: Any = config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return default


def parse_value_list(value: Any) -> list[str]:
    """Parse comma, semicolon, or newline separated configuration values."""
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[,;\n]", str(value or ""))
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def parse_identity_aliases(value: Any) -> dict[str, str]:
    """Parse one ``source=canonical name`` mapping per line."""
    aliases: dict[str, str] = {}
    for line in str(value or "").splitlines():
        source, separator, target = line.partition("=")
        if not separator:
            continue
        source = source.strip()
        target = target.strip()
        if source and target:
            aliases[source.casefold()] = target
    return aliases


def _event_value(event: Any, method_name: str, attribute_name: str = "") -> str:
    method = getattr(event, method_name, None)
    if callable(method):
        try:
            value = method()
            return str(value).strip() if value is not None else ""
        except Exception:
            return ""
    value = getattr(event, attribute_name or method_name, "")
    return str(value).strip() if value is not None else ""


def resolve_event_identity(config: Any, event: Any) -> str:
    """Return the configured canonical identity for the event sender."""
    sender_id = _event_value(event, "get_sender_id", "sender_id")
    sender_name = _event_value(event, "get_sender_name", "sender_name")
    platform = _event_value(event, "get_platform_name", "platform").casefold()
    aliases = parse_identity_aliases(
        _config_get(config, "access_control.identity_aliases", "")
    )
    candidates = (
        f"{platform}:{sender_id}" if platform and sender_id else "",
        sender_id,
        sender_name,
    )
    for candidate in candidates:
        if candidate and candidate.casefold() in aliases:
            return aliases[candidate.casefold()]
    return (
        sender_id
        or sender_name
        or _event_value(event, "unified_msg_origin", "unified_msg_origin")
    )


def resolve_sender_alias(
    aliases_value: Any,
    platform: str,
    sender_id: str,
    sender_name: str | None,
) -> str | None:
    aliases = parse_identity_aliases(aliases_value)
    for candidate in (f"{platform}:{sender_id}", sender_id, sender_name or ""):
        if candidate and candidate.casefold() in aliases:
            return aliases[candidate.casefold()]
    return sender_name


@dataclass(frozen=True, slots=True)
class EventMemoryAccess:
    """一次事件请求内解析出的白名单、人格与记忆作用域。"""

    allowed: bool
    persona_id: str | None
    scope: str | None


async def resolve_event_memory_access(
    config: Any, context: Any, event: Any
) -> EventMemoryAccess:
    """一次解析事件的白名单、人格过滤与记忆作用域。

    Args:
        config: ConfigManager 或宿主配置。
        context: 插件 Context；人格解析依赖其 conversation/persona manager。
        event: 当前消息事件。

    Returns:
        白名单不通过时 ``allowed=False``，其余字段为 None。
    """
    if is_event_excluded_by_series(context, event) or not is_event_memory_allowed(
        config, event
    ):
        return EventMemoryAccess(allowed=False, persona_id=None, scope=None)
    persona_id = None
    if _config_get(config, "filtering_settings.use_persona_filtering", True):
        from .utils import get_persona_id

        persona_id = await get_persona_id(context, event)
    return EventMemoryAccess(
        allowed=True,
        persona_id=persona_id,
        scope=resolve_memory_scope(config, event),
    )


def get_effective_series_shared_config(context: Any) -> dict[str, Any]:
    """Read shared defaults from chat_dynamics via series_panels with extended key fallback."""
    if context is None:
        return {}
    result: dict[str, Any] = {}
    try:
        base = series_panels.get_series_shared_config(context)
        if isinstance(base, dict):
            result.update(base)
    except Exception:
        pass

    # Compat: before series_panels.py is synced, read newly added Wave 1 keys
    # directly from chat_dynamics owner if present (strictly synchronous/read-only).
    extended_str_keys = (
        "light_analysis_provider",
        "vision_provider",
        "embedding_provider",
        "rerank_provider",
    )
    need_extended = any(
        k not in result for k in (*extended_str_keys, "private_shared_group_sources")
    )
    if need_extended:
        try:
            owner = series_panels.find_plugin_owner(
                context, "astrbot_plugin_chat_dynamics"
            )
            getter = (
                getattr(owner, "get_series_shared_config", None)
                if owner is not None
                else None
            )
            if callable(getter) and not inspect.iscoroutinefunction(getter):
                raw = getter()
                if inspect.isawaitable(raw):
                    if inspect.iscoroutine(raw):
                        raw.close()
                    raw = None
                if isinstance(raw, dict) and raw.get("enabled") is not False:
                    for key in extended_str_keys:
                        if key not in result:
                            val = raw.get(key)
                            result[key] = val.strip() if isinstance(val, str) else ""
                    if "private_shared_group_sources" not in result:
                        result["private_shared_group_sources"] = parse_value_list(
                            raw.get("private_shared_group_sources")
                        )
                    if "persona_id" not in result or not result.get("persona_id"):
                        val = raw.get("persona_id")
                        if isinstance(val, str) and val.strip():
                            result["persona_id"] = val.strip()
        except Exception:
            pass
    return result


def is_member_privacy_muted(
    context: Any, umo: str, member_id: str, scope: str = "memory"
) -> bool:
    """Zero-coupling dynamic lookup for unified cross-plugin privacy mute."""
    clean_umo = str(umo or "").strip()
    clean_member = str(member_id or "").strip()
    clean_scope = str(scope or "").strip()
    if context is None or not clean_umo or not clean_member:
        return False

    try:
        owner = series_panels.find_plugin_owner(context, "astrbot_plugin_chat_dynamics")
        if owner is None:
            return False
        getter = getattr(owner, "get_series_privacy_muted", None)
        if not callable(getter) or inspect.iscoroutinefunction(getter):
            return False
        try:
            raw = getter(clean_umo, clean_member, clean_scope)
        except TypeError:
            try:
                raw = getter(umo=clean_umo, member_id=clean_member, scope=clean_scope)
            except TypeError:
                raw = getter(clean_umo, clean_member)
        if inspect.isawaitable(raw):
            if inspect.iscoroutine(raw):
                raw.close()
            return False
        if isinstance(raw, dict):
            return raw.get("muted") is True
        return raw is True
    except Exception:
        return False


def is_event_privacy_muted(context: Any, event: Any, scope: str = "memory") -> bool:
    """Return True if the event sender is privacy-muted in this UMO for the given scope."""
    if context is None or event is None:
        return False
    umo = _event_value(event, "unified_msg_origin", "unified_msg_origin")
    member_id = _event_value(event, "get_sender_id", "sender_id")
    if not umo or not member_id:
        return False
    return is_member_privacy_muted(context, umo, member_id, scope)


def is_event_excluded_by_series(context: Any, event: Any) -> bool:
    """Return True if chat_dynamics excludes this event's UMO or group_id."""
    if context is None or event is None:
        return False
    session_id = _event_value(event, "unified_msg_origin", "unified_msg_origin")
    group_id = _event_value(event, "get_group_id", "group_id")
    return is_series_group_excluded(context, umo=session_id, group_id=group_id)


def is_event_memory_allowed(config: Any, event: Any, context: Any = None) -> bool:
    """Apply the plugin-level allowlist/disabled_ids and core exclude_groups guard consistently."""
    if is_event_excluded_by_series(context, event):
        return False
    session_id = _event_value(event, "unified_msg_origin", "unified_msg_origin")
    group_id = _event_value(event, "get_group_id", "group_id")
    disabled = {
        value.casefold()
        for value in parse_value_list(
            _config_get(config, "access_control.disabled_ids", "")
        )
    }
    if disabled:
        if (session_id and session_id.casefold() in disabled) or (
            group_id and group_id.casefold() in disabled
        ):
            return False
    if not _config_get(config, "access_control.whitelist_enabled", False):
        return True
    allowed = {
        value.casefold()
        for value in parse_value_list(
            _config_get(config, "access_control.allowed_ids", "")
        )
    }
    if not allowed:
        return False

    sender_id = _event_value(event, "get_sender_id", "sender_id")
    platform = _event_value(event, "get_platform_name", "platform")
    identity = resolve_event_identity(config, event)
    candidates = {
        sender_id,
        identity,
        session_id,
        group_id,
        f"{platform}:{sender_id}" if platform and sender_id else "",
    }
    return any(
        candidate and candidate.casefold() in allowed for candidate in candidates
    )


def resolve_user_memory_scope(config: Any, event: Any) -> str:
    """构造用户级作用域；身份别名与运行时读写共用同一份配置。"""
    platform = _event_value(event, "get_platform_name", "platform").casefold()
    identity = resolve_event_identity(config, event).casefold()
    return f"livingmemory:user:{platform or 'unknown'}:{identity}"


def resolve_memory_scope(config: Any, event: Any) -> str | None:
    """Resolve the retrieval/storage scope while preserving legacy defaults."""
    session_id = _event_value(event, "unified_msg_origin", "unified_msg_origin")
    isolated_sessions = set(
        parse_value_list(
            _config_get(config, "filtering_settings.isolated_sessions", "")
        )
    )
    if session_id in isolated_sessions:
        return session_id

    mode = str(
        _config_get(config, "filtering_settings.memory_scope_mode", "legacy")
    ).casefold()
    if mode == "session":
        return session_id
    if mode == "global":
        return GLOBAL_MEMORY_SCOPE
    if mode == "user":
        return resolve_user_memory_scope(config, event)

    use_session = bool(
        _config_get(config, "filtering_settings.use_session_filtering", True)
    )
    if use_session:
        return session_id
    return GLOBAL_MEMORY_SCOPE if isolated_sessions else None


def is_suspicious_session_id(session_id: str | None) -> bool:
    """Detect error strings leaked into unified_msg_origin.

    Some platform adapters surface initialization errors through the message
    origin; memory writes/recalls tagged with such an id would be unreachable
    in later sessions, so entry points log a warning when they see one.

    Args:
        session_id: unified_msg_origin of the current event.

    Returns:
        bool: True when the id looks like a leaked error string.
    """
    return bool(session_id) and "error:" in str(session_id).lower()


__all__ = [
    "EventMemoryAccess",
    "GLOBAL_MEMORY_SCOPE",
    "MemoryAccessPolicy",
    "MemoryReadScope",
    "classify_umo",
    "get_effective_series_shared_config",
    "is_event_memory_allowed",
    "is_event_privacy_muted",
    "is_member_privacy_muted",
    "is_suspicious_session_id",
    "memory_metadata_matches_policy",
    "normalize_interaction_types",
    "parse_identity_aliases",
    "parse_value_list",
    "resolve_event_identity",
    "resolve_memory_access_policy",
    "resolve_verified_group_sessions",
    "resolve_memory_scope",
    "resolve_memory_write_scope",
    "resolve_sender_alias",
]


def normalize_interaction_types(value: Any) -> tuple[str, ...] | None:
    """Normalize a single interaction_type or tuple/sequence of types."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if item is not None)


@dataclass(frozen=True)
class MemoryReadScope:
    """A single, provenance-bounded selector used before candidate ranking."""

    session_ids: tuple[str, ...] | None
    interaction_types: tuple[str, ...] | None
    require_persona: bool = False
    source_platform: str | None = None
    source_session_id: str | None = None
    read_only: bool = False
    excluded_session_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryAccessPolicy:
    """Trusted event-derived read/write boundaries for one memory turn."""

    message_kind: str
    write_scope: str
    write_interaction_type: str
    write_visibility: str
    read_scopes: tuple[MemoryReadScope, ...]


def classify_umo(umo: str | None) -> str:
    """Classify only explicit AstrBot message types; unknown origins fail closed."""
    parts = str(umo or "").split(":")
    if len(parts) < 2:
        return "unknown"
    message_type = parts[1].casefold()
    if "group" in message_type:
        return "group"
    if "friend" in message_type or "private" in message_type:
        return "private"
    return "unknown"


def _umo_platform(umo: str) -> str:
    return str(umo or "").split(":", 1)[0].strip().casefold()


def _event_platform_identity(event: Any) -> str:
    platform_id = _event_value(event, "get_platform_id", "platform_id")
    if platform_id:
        return platform_id.casefold()
    return _event_value(event, "get_platform_name", "platform").casefold()


def memory_metadata_matches_policy(
    metadata: Any,
    *,
    interaction_type: str | tuple[str, ...] | None = None,
    source_platform: str | None = None,
    source_session_id: str | None = None,
    excluded_session_ids: tuple[str, ...] | set[str] | list[str] | None = None,
) -> bool:
    """Fail closed on private, unknown, excluded, or cross-platform derived records."""
    if not isinstance(metadata, dict):
        return False
    allowed_types = normalize_interaction_types(interaction_type)
    if (
        allowed_types is not None
        and metadata.get("interaction_type") not in allowed_types
    ):
        return False
    if (
        source_session_id is not None
        and metadata.get("source_session_id") != source_session_id
    ):
        return False
    if excluded_session_ids:
        excluded = set(excluded_session_ids)
        if (
            metadata.get("session_id") in excluded
            or metadata.get("source_session_id") in excluded
        ):
            return False
    if (
        allowed_types is not None
        and set(allowed_types).issubset({"group_chat", "unknown"})
        and (
            metadata.get("memory_visibility") == "private"
            or classify_umo(metadata.get("source_session_id")) == "private"
        )
    ):
        return False
    if source_platform:
        expected = source_platform.casefold()
        declared = str(metadata.get("source_platform") or "").casefold()
        if declared == expected:
            return True
        origin = metadata.get("source_session_id") or metadata.get("session_id")
        if not (
            isinstance(origin, str)
            and _umo_platform(origin) == expected
            and classify_umo(origin) == "group"
        ):
            return False
    return True


def _resolve_private_shared_group_sources(
    config: Any, context: Any = None
) -> list[str]:
    """Resolve private_shared_group_sources from local config or shared chat_dynamics config (str or list)."""
    local_sources = parse_value_list(
        _config_get(config, "filtering_settings.private_shared_group_sources", "")
    )
    if local_sources:
        return local_sources
    if context is not None:
        shared = get_effective_series_shared_config(context)
        shared_sources = parse_value_list(shared.get("private_shared_group_sources"))
        if shared_sources:
            return shared_sources
    return []


def resolve_memory_access_policy(
    config: Any,
    event: Any,
    *,
    persona_id: str | None = None,
    verified_group_sessions: Any = (),
    context: Any = None,
) -> MemoryAccessPolicy:
    """Resolve one-way group-to-private reads and event-owned write scope.

    Group sources are admitted only when explicitly configured, present in the
    conversation store, tagged as GroupMessage, and on the current platform.
    The current user need not have participated in a shared group session.
    """
    session_id = _event_value(event, "unified_msg_origin", "unified_msg_origin")
    kind = classify_umo(session_id)
    share_enabled = bool(
        _config_get(config, "filtering_settings.enable_group_to_private_recall", False)
    )
    write_scope = resolve_memory_write_scope(config, event)
    interaction_type = {
        "group": "group_chat",
        "private": "private_chat",
    }.get(kind, "unknown")
    # Unknown and private writes are private by default. Only a verified group
    # event can produce group-visible memory.
    visibility = "group" if kind == "group" else "private"

    if kind == "group":
        isolated_sessions = set(
            parse_value_list(
                _config_get(config, "filtering_settings.isolated_sessions", "")
            )
        )
        mode = str(
            _config_get(config, "filtering_settings.memory_scope_mode", "legacy")
        ).casefold()
        use_session = bool(
            _config_get(config, "filtering_settings.use_session_filtering", True)
        )
        # Respect session isolation (isolated_sessions, mode="session", and
        # legacy default use_session_filtering=True) in a single read scope.
        if (
            session_id in isolated_sessions
            or mode == "session"
            or (mode not in {"global", "user"} and use_session)
        ):
            own_scope = (
                MemoryReadScope((session_id,), ("group_chat", "unknown"))
                if session_id
                else MemoryReadScope((), ("group_chat", "unknown"))
            )
            return MemoryAccessPolicy(
                kind, write_scope, interaction_type, visibility, (own_scope,)
            )
        if mode == "user":
            user_scope = resolve_user_memory_scope(config, event)
            return MemoryAccessPolicy(
                kind,
                write_scope,
                interaction_type,
                visibility,
                (MemoryReadScope((user_scope,), ("group_chat", "unknown")),),
            )

        # Cross-group group recall is explicitly enabled (mode="global" or
        # legacy with use_session_filtering=False). Require explicit group_chat
        # provenance on the current platform (which already covers the current
        # group's group_chat records, avoiding a redundant subset search),
        # exclude configured isolated_sessions, and keep older untyped records
        # available in their own group session only.
        current_platform = _event_platform_identity(event)
        excluded = tuple(sorted(isolated_sessions))
        scopes = [
            MemoryReadScope(
                None,
                ("group_chat",),
                source_platform=current_platform or None,
                excluded_session_ids=excluded,
            )
        ]
        if session_id:
            scopes.append(MemoryReadScope((session_id,), ("unknown",)))
        return MemoryAccessPolicy(
            kind, write_scope, interaction_type, visibility, tuple(scopes)
        )

    own_scope_id = (
        session_id
        if kind == "private" and share_enabled
        else resolve_memory_scope(config, event) or session_id
    )
    own_scope = (
        MemoryReadScope((own_scope_id,), None)
        if own_scope_id
        else MemoryReadScope((), None)
    )
    if kind != "private" or not _config_get(
        config, "filtering_settings.enable_group_to_private_recall", False
    ):
        return MemoryAccessPolicy(
            kind, write_scope, interaction_type, visibility, (own_scope,)
        )

    current_platform = _event_platform_identity(event)
    configured_sources = set(
        _resolve_private_shared_group_sources(config, context=context)
    )
    verified_sources = {
        str(source).strip()
        for source in (verified_group_sessions or ())
        if str(source).strip()
    }
    configured_instance = str(
        _config_get(config, "filtering_settings.private_shared_group_instance_id", "")
        or ""
    ).strip()
    if not configured_instance and context is not None:
        shared_persona = str(
            get_effective_series_shared_config(context).get("persona_id") or ""
        ).strip()
        if shared_persona:
            configured_instance = shared_persona
    if not persona_id or not configured_instance or persona_id != configured_instance:
        return MemoryAccessPolicy(
            kind, write_scope, interaction_type, visibility, (own_scope,)
        )
    shared_sources = sorted(
        source
        for source in configured_sources & verified_sources
        if classify_umo(source) == "group"
        and current_platform
        and _umo_platform(source) == current_platform
    )
    scopes = [own_scope]
    if shared_sources:
        scopes.extend(
            MemoryReadScope(
                None,
                ("group_chat",),
                require_persona=True,
                source_platform=current_platform,
                source_session_id=source,
                read_only=True,
            )
            for source in shared_sources
        )
    return MemoryAccessPolicy(
        kind, write_scope, interaction_type, visibility, tuple(scopes)
    )


def resolve_memory_write_scope(config: Any, event: Any) -> str:
    """Keep private writes session-local whenever one-way sharing is enabled."""
    session_id = _event_value(event, "unified_msg_origin", "unified_msg_origin")
    if classify_umo(session_id) == "private" and _config_get(
        config, "filtering_settings.enable_group_to_private_recall", False
    ):
        return session_id
    return resolve_memory_scope(config, event) or session_id


async def resolve_verified_group_sessions(
    config: Any,
    event: Any,
    conversation_manager: Any,
    *,
    context: Any = None,
) -> tuple[str, ...]:
    """Return configured group UMOs backed by this platform's captured sessions."""
    if not _config_get(
        config, "filtering_settings.enable_group_to_private_recall", False
    ):
        return ()
    if not callable(getattr(conversation_manager, "get_session_info", None)):
        return ()
    platform_id = _event_platform_identity(event)
    platform_name = _event_value(event, "get_platform_name", "platform").casefold()
    if not platform_id or not platform_name:
        return ()
    configured = _resolve_private_shared_group_sources(config, context=context)
    verified = []
    for source in configured:
        parts = source.split(":")
        if (
            classify_umo(source) != "group"
            or len(parts) < 3
            or parts[0].casefold() != platform_id
        ):
            continue
        session = await conversation_manager.get_session_info(source)
        if (
            session is not None
            and str(session.platform or "").casefold() == platform_name
        ):
            verified.append(source)
    return tuple(sorted(set(verified)))


async def fetch_group_culture_context(
    context: Any,
    umo: str,
    *,
    query: str = "",
    limit: int = 4,
    timeout: float = 0.4,
) -> list[dict[str, str]]:
    """Query approved group slang/memes from group_chronicle with zero hard coupling.

    Always returns a bounded list of ``{"term": str, "meaning": str}`` dicts and
    never raises on missing plugin, mock objects, timeout, or malformed returns.
    """
    if context is None or not isinstance(umo, str) or not umo.strip():
        return []
    limit = max(1, min(int(limit), 8))
    try:
        owner = series_panels.find_plugin_owner(
            context, "astrbot_plugin_group_chronicle"
        )
    except Exception:
        return []
    if owner is None:
        return []
    getter = getattr(owner, "get_culture_context", None)
    if not callable(getter):
        return []

    raw: Any = None
    try:
        try:
            res = getter(umo.strip(), query=str(query or ""), limit=limit)
        except TypeError:
            try:
                res = getter(umo.strip(), limit=limit)
            except TypeError:
                res = getter(umo.strip())
        if inspect.isawaitable(res):
            import asyncio

            raw = await asyncio.wait_for(res, timeout=max(0.05, float(timeout)))
        else:
            raw = res
    except Exception:
        return []

    candidates: Any = None
    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, dict):
        for key in ("items", "entries", "slang", "memes", "culture"):
            val = raw.get(key)
            if isinstance(val, list):
                candidates = val
                break
    if not isinstance(candidates, list):
        return []

    normalized: list[dict[str, str]] = []
    seen_terms: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if isinstance(status, str) and status.strip():
            if status.strip().lower() not in {
                "approved",
                "active",
                "verified",
                "published",
            }:
                continue
        term = ""
        for k in ("term", "name", "phrase", "title", "keyword"):
            val = item.get(k)
            if isinstance(val, str) and val.strip():
                term = re.sub(r"\s+", " ", val).strip()[:40]
                break
        meaning = ""
        for k in ("meaning", "explanation", "description", "summary", "content"):
            val = item.get(k)
            if isinstance(val, str) and val.strip():
                meaning = re.sub(r"\s+", " ", val).strip()[:120]
                break
        if not term or not meaning:
            continue
        dedup_key = term.casefold()
        if dedup_key in seen_terms:
            continue
        seen_terms.add(dedup_key)
        normalized.append({"term": term, "meaning": meaning})
        if len(normalized) >= limit:
            break
    return normalized
