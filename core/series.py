"""Versioned companions, ownership and shared evidence without replica stores."""
from __future__ import annotations

import asyncio
import inspect
import json
import math
import time

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


def value(row, key, default=None):
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def active_entries(context):
    getter = getattr(context, "get_all_stars", None)
    if not callable(getter):
        return []
    rows = getter()
    if not isinstance(rows, (list, tuple, dict)):
        return []
    entries = rows.items() if isinstance(rows, dict) else (("", row) for row in (rows or []))
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
        return data if isinstance(data, dict) and type(data.get("schema_version")) is int and data["schema_version"] == 1 else {}
    except Exception:
        return {}


def find_role(context, role):
    found = []
    for _, owner in active_entries(context):
        data = manifest(owner)
        caps = data.get("capabilities", {})
        if (data.get("enabled") is True and isinstance(caps, dict)
                and type(caps.get(role)) is int and caps[role] == 1):
            found.append((owner, data))
    if len(found) == 1 and found[0][1].get("ready") is True:
        return found[0][0]
    return None


class SeriesMixin:
    def series_conflicts(self):
        # Do not recursively call other memory manifests (they may also check us).
        return sorted({name for name, owner in active_entries(getattr(self, "context", None))
                       if owner is not self and (
                           name in {"astrbot_plugin_livingmemory", "LivingMemory", PLUGIN}
                           or (callable(getattr(owner, "handle_memory_recall", None))
                               and callable(getattr(owner, "get_evidence_view", None))))})

    def series_runtime_allowed(self):
        return (getattr(self, "_series_enabled", True) is True
                and not getattr(self, "_terminating", False) and not self.series_conflicts())

    def get_dynamics_capabilities(self):
        return {"schema_version": 1, "provider_id": PLUGIN, "display_name": "群间记忆",
                "enabled": getattr(self, "_series_enabled", False) is True,
                "ready": self.series_runtime_allowed() and bool(getattr(self.initializer, "is_initialized", False)),
                "capabilities": {"memory": 1, "evidence": 1}, "api_version": 1,
                "conflicts": self.series_conflicts()}

    async def get_shared_evidence(self, scope, source_refs, *, purpose="reply", card_version=""):
        """World/learning read existing validated references; never ingest their drafts.

        Keep source checks in get_evidence_view. Callers cannot turn learned
        policies, world plans or their own prose into new user observations.
        """
        if not isinstance(source_refs, list) or len(source_refs) > 64:
            raise ValueError("source_refs must be a bounded list")
        return await self.get_evidence_view(scope, source_refs=source_refs,
                                            purpose=purpose, card_version=card_version,
                                            allow_derivation=False)

    async def get_series_context(self, scope, source_refs, *, purpose="reply", card_version="", learning_request=None):
        """Read-only composite, with separate owners and no second prompt injection."""
        if not self.series_runtime_allowed() or not await self._narrative_scope_allowed(scope):
            return {"schema_version": 1, "available": False, "reason": "scope_or_owner_unavailable"}
        world = find_role(self.context, "character")
        world_data = None
        if world is not None:
            try:
                world_data = await asyncio.wait_for(world.get_character_context(scope), 1.0)
            except (TimeoutError, ValueError, RuntimeError):
                world_data = None
            if find_role(self.context, "character") is not world:
                world_data = None
        learning = None
        # Learning uses a different, intentionally narrower request contract.
        # The caller must supply the exact current persona fingerprint/baseline.
        if isinstance(learning_request, dict) and learning_request.get("scope") == scope.get("umo"):
            required = {"scope", "persona", "member", "baseline", "host_version"}
            if set(learning_request) == required and all(isinstance(v, str) for v in learning_request.values()):
                candidates = [p for n, p in active_entries(self.context) if n == "astrbot_plugin_dynamics_learning"]
                if len(candidates) == 1 and await self._narrative_scope_allowed(scope, learning_request['member']):
                    owner = candidates[0]
                    getter = getattr(owner, "get_learning_context", None)
                    if callable(getter) and not inspect.iscoroutinefunction(getter):
                        learning = getter(**learning_request)
                        if inspect.isawaitable(learning):
                            if inspect.iscoroutine(learning):
                                learning.close()
                            learning = None
        evidence = await self.get_shared_evidence(scope, source_refs, purpose=purpose, card_version=card_version)
        if world is not None and find_role(self.context, "character") is not world:
            world_data = None
        if learning is not None and not any(n == 'astrbot_plugin_dynamics_learning' and p is owner for n, p in active_entries(self.context)):
            learning = None
        if not self.series_runtime_allowed() or not await self._narrative_scope_allowed(scope):
            return {"schema_version": 1, "available": False, "reason": "source_invalidated"}
        return {"schema_version": 1, "available": True, "memory": evidence,
                "world": world_data, "learning": learning,
                "ownership": {"memory": "observations_and_evidence", "world": "character_and_life",
                              "learning": "reviewed_behavior"}}

    def register_series_routes(self):
        for path, handler, methods, title in (
            ('series', self.series_status, ['GET'], '群间记忆接入与冲突状态'),
            ('series/conflicts', self.series_conflict_list, ['GET'], '群间记忆来源冲突'),
            ('series/detail', self.series_conflict_detail, ['GET'], '查看来源冲突差异'),
            ('series/resolve', self.series_resolve_conflict, ['POST'], '标记来源冲突已审查'),
        ):
            self.context.register_web_api(f"/{PLUGIN}/page/{path}", self._series_route(handler), methods, title)

    @staticmethod
    def _series_route(handler):
        async def serve():
            from astrbot.api.web import json_response
            try:
                return json_response(await handler())
            except PermissionError:
                return json_response({'status': 'error', 'message': '需要 Dashboard 管理员登录'}, status_code=403)
            except ValueError:
                return json_response({'status': 'error', 'message': '参数无效或来源冲突版本已改变，请刷新后重试。'}, status_code=409)
        return serve

    async def _series_admin(self):
        from astrbot.api.web import request
        from astrbot.dashboard.api.auth import require_dashboard_user
        raw = getattr(request, "_request", None)
        if raw is None:
            raise PermissionError("需要 Dashboard 管理员登录")
        actor = await require_dashboard_user(raw)
        expected = self.context.get_config().get("dashboard", {}).get("username")
        if not actor or expected and actor != expected:
            raise PermissionError("需要 Dashboard 管理员登录")
        return str(actor)

    async def series_status(self):
        await self._series_admin()
        entries = active_entries(self.context)
        return {"status": "ok", "data": {**self.get_dynamics_capabilities(),
                "base": "LivingMemory 2.10.0+local.3", "version": "0.1.0",
                "world_ready": find_role(self.context, "character") is not None,
                "learning_installed": any(n == "astrbot_plugin_dynamics_learning" for n, _ in entries),
                "games_installed": any(n == "astrbot_plugin_dynamics_games" for n, _ in entries),
                "migration": "migration.py：预览后在源与目标均停用时导入；旧数据保持原样。",
                "data_ownership": "世界维护设定与生活；学习维护行为；游戏负责活动进度与成果；记忆维护经历与事实来源。"}}

    async def series_conflict_list(self):
        await self._series_admin()
        ready, error = await self._ensure_plugin_ready(wait=False)
        if not ready:
            return {"status": "error", "message": error}
        store = self.initializer.conversation_manager.store
        return {"status": "ok", "data": await store.list_series_conflicts()}

    async def series_resolve_conflict(self):
        actor = await self._series_admin()
        from astrbot.api.web import request
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {"id", "revision", "reason"}:
            raise ValueError("invalid conflict review")
        ready, error = await self._ensure_plugin_ready(wait=False)
        if not ready:
            return {"status": "error", "message": error}
        result = await self.initializer.conversation_manager.store.resolve_series_conflict(
            body["id"], body["revision"], body["reason"], actor)
        return {"status": "ok", "data": result}

    async def series_conflict_detail(self):
        await self._series_admin()
        from astrbot.api.web import request
        identifier = request.args.get('id', '')
        ready, error = await self._ensure_plugin_ready(wait=False)
        if not ready:
            return {"status": "error", "message": error}
        data = await self.initializer.conversation_manager.store.get_series_conflict(identifier)
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
                if (source_umo != scope["umo"] or row.get("sender_id") != uid
                        or row.get("role") != "user" or stamp is None
                        or abs(stamp - claimed_time) > ACTIVITY_SOURCE_CLOCK_SKEW
                        or stamp > occurred_at + ACTIVITY_SOURCE_CLOCK_SKEW):
                    return [], "source_identity_mismatch"
                platform = str(row.get("platform") or scope["platform"])
                if not await self._memory_scope_allowed({**scope, "platform": platform}, uid):
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

        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
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
        if len(parts) != 3 or parts[1] != "GroupMessage" or any(not p or p.strip() != p for p in parts):
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
        if occurred_at is None or occurred_at > time.time() + ACTIVITY_SOURCE_CLOCK_SKEW:
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
            if (not isinstance(message_id, str) or not 0 < len(message_id) <= 128
                    or message_id.strip() != message_id or not isinstance(uid, str)
                    or uid not in members or ts is None or ts > occurred_at + ACTIVITY_SOURCE_CLOCK_SKEW):
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
        if not callable(lease) or (refs and not callable(getattr(store, "get_activity_sources", None))):
            return result("source_store_unavailable")

        exp_key = f"dg:{umo}:{activity_id}:{event_kind}"
        bm25 = getattr(getattr(engine, "hybrid_retriever", None), "bm25_retriever", None)
        if not callable(getattr(bm25, "_read_connect", None)):
            return result("memory_index_unavailable")

        lock = getattr(self, "_dg_experience_lock", None)
        if lock is None:
            lock = self._dg_experience_lock = asyncio.Lock()
        async with lock:
            saved_refs = {message_id: list(identity) for message_id, identity in refs.items()}
            async with bm25._read_connect() as db:
                cursor = await db.execute(
                    f"SELECT d.id, json_extract(d.metadata, '$.activity_source_verified_v1'), "
                    "json_extract(d.metadata, '$.occurred_at'), "
                    "json_extract(d.metadata, '$.activity_source_refs') "
                    f"FROM {bm25.doc_table} d "
                    "WHERE json_valid(d.metadata) "
                    "AND json_extract(d.metadata, '$.experience_key') = ? "
                    "AND COALESCE(json_extract(d.metadata, '$.status'), 'active') IN ('active', 'archived') "
                    "LIMIT 1", (exp_key,),
                )
                prior = await cursor.fetchone()
                if prior:
                    if prior[1] != 1:
                        return result("legacy_experience_unverified")
                    if prior[2] != occurred_at or json.loads(prior[3] or "{}") != saved_refs:
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
                        "memory_type": "event", "source_plugin": "astrbot_plugin_dynamics_games",
                        "activity_id": activity_id, "activity_category": category,
                        "event_kind": event_kind, "experience_key": exp_key,
                        "participant_uids": list(members), "occurred_at": occurred_at,
                        "create_time": occurred_at, "activity_source_verified_v1": 1,
                        "activity_source_refs": saved_refs,
                    },
                    preserve_create_time=True, source_messages=source_messages or None,
                    source_validator=sources_current,
                    source_publish_lease=lambda: lease(sources_current),
                )
            except ValueError as exc:
                if str(exc) != "source_snapshot_changed":
                    raise
                return result("source_snapshot_changed")
            return result(doc_id=doc_id) if doc_id else result("memory_write_failed")
