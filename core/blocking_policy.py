"""Optional, reload-safe adapter to Autonomous Blocking's public service."""

from __future__ import annotations

import inspect
import time
from typing import Any


PLUGIN_NAME = "astrbot_plugin_autonomous_blocking"


def _value(item: Any, name: str, default=None):
    return (
        item.get(name, default)
        if isinstance(item, dict)
        else getattr(item, name, default)
    )


def _matches_name(value: Any) -> bool:
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        return False
    leaf = text.rsplit("/", 1)[-1]
    parts = leaf.split(".")
    return PLUGIN_NAME in parts or leaf == PLUGIN_NAME


class BlockingPolicyGate:
    """Resolve and apply the current optional blocking policy for one source."""

    def __init__(self, host_context: Any):
        self.host_context = host_context
        self.owner_ttl_seconds = 1800.0
        self.max_owner_contexts = 256
        self._owner_contexts: dict[tuple[str, str], dict[str, Any]] = {}
        self.wakeup = None

    def _discover(self) -> tuple[str, Any]:
        catalog = getattr(self.host_context, "get_all_stars", None)
        if not callable(catalog):
            return "unavailable", None
        try:
            entries = catalog()
            entries = (
                entries.items()
                if isinstance(entries, dict)
                else (("", item) for item in entries or ())
            )
            for key, item in entries:
                if not _value(item, "activated", True):
                    continue
                names = (
                    key,
                    _value(item, "name"),
                    _value(item, "root_dir_name"),
                    _value(item, "module_path"),
                )
                if not any(_matches_name(value) for value in names):
                    continue
                service = (
                    _value(item, "star_cls")
                    or _value(item, "star")
                    or _value(item, "plugin")
                    or item
                )
                if service is not None and not isinstance(service, type):
                    return "found", service
            return "absent", None
        except Exception:
            return "unavailable", None

    @staticmethod
    def _plain(value: Any):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(key): BlockingPolicyGate._plain(item) for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [BlockingPolicyGate._plain(item) for item in value]
        raise TypeError("blocking owner context is not plain data")

    def _platform(self, platform_id: str, scope: dict):
        lookup = getattr(self.host_context, "get_platform_inst", None)
        if not callable(lookup):
            return None
        platform = lookup(platform_id)
        if inspect.isawaitable(platform):
            return platform
        return self._validate_platform(platform, platform_id, scope)

    @staticmethod
    def _validate_platform(platform: Any, platform_id: str, scope: dict):
        if platform is None:
            return None
        try:
            meta = platform.meta()
            if str(_value(meta, "id", "")) != platform_id:
                return None
            if not str(_value(meta, "name", "")):
                return None
            status = _value(platform, "status")
            if (
                status is not None
                and str(_value(status, "value", status)).lower() != "running"
            ):
                return None
            return platform
        except Exception:
            return None

    @staticmethod
    def _scope_key(scope: dict) -> tuple[str, str] | None:
        umo = scope.get("umo")
        platform_id = umo.split(":", 1)[0] if isinstance(umo, str) else ""
        if not platform_id:
            return None
        return platform_id, umo

    def _drop_platform_contexts(
        self, platform_id: str, owner_id: str | None = None
    ) -> None:
        for cached_key, cached_value in list(self._owner_contexts.items()):
            if cached_key[0] == platform_id and (
                owner_id is None or cached_value.get("owner_id") != owner_id
            ):
                self._owner_contexts.pop(cached_key, None)

    async def observe_trusted_event(self, event: Any, scope: dict) -> bool | None:
        """Cache only a verified owner context derived from a genuine event."""
        state, service = self._discover()
        if state == "absent":
            return True
        if state != "found":
            return None
        key = self._scope_key(scope) if isinstance(scope, dict) else None
        if not key:
            return None
        platform_getter = getattr(event, "get_platform_id", None)
        self_getter = getattr(event, "get_self_id", None)
        if not callable(platform_getter) or not callable(self_getter):
            return None
        try:
            platform_id = str(platform_getter() or "").strip()
            self_id = str(self_getter() or "").strip()
        except Exception:
            return None
        event_umo = str(getattr(event, "unified_msg_origin", "") or "")
        if (
            not platform_id
            or not self_id
            or platform_id != key[0]
            or event_umo != key[1]
        ):
            return None
        self._drop_platform_contexts(platform_id, self_id)
        if getattr(service, "enabled", True) is False:
            self._drop_platform_contexts(platform_id)
            return True
        platform = self._platform(platform_id, scope)
        if inspect.isawaitable(platform):
            platform = await platform
            platform = self._validate_platform(platform, platform_id, scope)
        if platform is None:
            self._drop_platform_contexts(platform_id)
            return None
        resolve = getattr(service, "resolve_context", None)
        if not callable(resolve):
            return None
        try:
            context = resolve(event=event, scope=dict(scope))
            if inspect.isawaitable(context):
                context = await context
            if not isinstance(context, dict) or not context.get("ready"):
                self._owner_contexts.pop(key, None)
                return None
            owner = context.get("owner")
            owner_id = _value(owner, "self_id", "")
            owner_platform_id = _value(owner, "platform_instance_id", "")
            if (
                not isinstance(owner_id, str)
                or not owner_id.strip()
                or owner_id.strip() != self_id
                or str(owner_platform_id or "") != platform_id
                or str(context.get("umo", "")) != event_umo
                or event_umo != key[1]
            ):
                self._owner_contexts.pop(key, None)
                return None
            context = self._plain(context)
        except Exception:
            self._owner_contexts.pop(key, None)
            return None
        self._owner_contexts[key] = {
            "context": context,
            "owner_id": self_id,
            "service_id": id(service),
            "platform_object_id": id(platform),
            "captured_at": time.monotonic(),
        }
        if len(self._owner_contexts) > self.max_owner_contexts:
            oldest = min(
                self._owner_contexts,
                key=lambda item: self._owner_contexts[item]["captured_at"],
            )
            self._owner_contexts.pop(oldest, None)
        if callable(self.wakeup):
            try:
                result = self.wakeup(scope)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
        return True

    async def owner_ready(self, umo: str) -> bool | None:
        """Check whether this UMO has a live owner captured from a real event."""
        state, service = self._discover()
        if state == "absent":
            return True
        if state != "found":
            return None
        if getattr(service, "enabled", True) is False:
            return True
        platform_id = umo.split(":", 1)[0] if isinstance(umo, str) else ""
        key = (platform_id, umo) if platform_id else None
        cached = self._owner_contexts.get(key) if key else None
        if (
            not cached
            or time.monotonic() - cached["captured_at"] > self.owner_ttl_seconds
        ):
            if key:
                self._owner_contexts.pop(key, None)
            return None
        if cached["service_id"] != id(service):
            self._owner_contexts.pop(key, None)
            return None
        context = cached["context"]
        owner = context.get("owner") if isinstance(context, dict) else None
        if (
            str(context.get("umo", "")) != umo
            or str(_value(owner, "self_id", "")) != cached["owner_id"]
            or str(_value(owner, "platform_instance_id", "")) != platform_id
        ):
            self._owner_contexts.pop(key, None)
            return None
        platform = self._platform(
            platform_id, {"platform": context.get("scope", {}).get("platform", "")}
        )
        if inspect.isawaitable(platform):
            platform = await platform
            platform = self._validate_platform(platform, platform_id, {})
        if platform is None or id(platform) != cached["platform_object_id"]:
            self._owner_contexts.pop(key, None)
            return None
        return True

    async def check(self, descriptor: dict, material: dict) -> bool | None:
        """Return True when allowed, False when blocked, None when unverifiable."""
        state, service = self._discover()
        if state == "absent":
            return True
        if state != "found":
            return None
        if getattr(service, "enabled", True) is False:
            return True

        locator = descriptor.get("source_locator")
        scope = locator.get("scope") if isinstance(locator, dict) else None
        if not isinstance(scope, dict) or not all(
            isinstance(scope.get(key), str) and scope[key].strip()
            for key in ("platform", "umo", "persona_id", "instance_id")
        ):
            return None
        snapshot_fn = getattr(service, "policy_snapshot", None)
        filter_fn = getattr(service, "filter_material", None)
        if not all(callable(fn) for fn in (snapshot_fn, filter_fn)):
            return None
        try:
            key = self._scope_key(scope)
            cached = self._owner_contexts.get(key) if key else None
            if (
                not cached
                or time.monotonic() - cached["captured_at"] > self.owner_ttl_seconds
            ):
                return None
            if cached["service_id"] != id(service):
                return None
            context = cached["context"]
            owner_id = _value(context.get("owner"), "self_id", "")
            owner_platform_id = _value(context.get("owner"), "platform_instance_id", "")
            if (
                not owner_id
                or str(owner_id) != cached["owner_id"]
                or str(owner_platform_id or "") != key[0]
                or str(context.get("umo", "")) != key[1]
            ):
                return None
            platform_id, _ = key
            platform = self._platform(platform_id, scope)
            if inspect.isawaitable(platform):
                platform = await platform
                platform = self._validate_platform(platform, platform_id, scope)
            if platform is None:
                return None
            if id(platform) != cached["platform_object_id"]:
                return None
            snapshot = snapshot_fn(context)
            if inspect.isawaitable(snapshot):
                snapshot = await snapshot
            if not isinstance(snapshot, dict) or not isinstance(
                snapshot.get("rules"), list
            ):
                return None

            actor_id = material.get("actor_id")
            exact_event = (
                descriptor.get("material_kind") == "source_event"
                and material.get("source_status") == "exact_event"
                and isinstance(material.get("source_event_id"), str)
                and isinstance(actor_id, str)
                and bool(actor_id.strip())
            )
            source_ids = {actor_id.strip()} if exact_event else set()
            frame = {
                "frame_id": str(descriptor.get("material_id", "")),
                "text": str(material.get("text", "")),
                "author_uid": actor_id if exact_event else "",
                "source_uids": [actor_id] if exact_event else [],
                "subject_id": "",
                "provenance_complete": bool(exact_event),
            }
            active_shields = set()
            for rule in snapshot["rules"]:
                if not isinstance(rule, dict):
                    return None
                if str(rule.get("mode", "")).lower() == "shield" and rule.get(
                    "active", True
                ):
                    target = str(rule.get("target_uid") or "").strip()
                    if not target:
                        return None
                    active_shields.add(target)
            bundle = {"context_items": [frame]}
            filtered = filter_fn(context, bundle, "role_input")
            if inspect.isawaitable(filtered):
                filtered = await filtered
            if not isinstance(filtered, dict) or not isinstance(
                filtered.get("context_items"), list
            ):
                return None
            if active_shields and (
                not exact_event or source_ids.intersection(active_shields)
            ):
                return False
            return any(
                isinstance(row, dict) and row.get("frame_id") == frame["frame_id"]
                for row in filtered["context_items"]
            )
        except Exception:
            return None
