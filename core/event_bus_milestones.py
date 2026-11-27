"""Wave 3: Dynamics Event Bus subscription (`topic_archived`) & cross-5-plugin
Memory Scrapbook × 3D Room Keepsake milestone trigger.

Zero hard imports of companion plugins: all cross-plugin interactions go through
``series_panels.find_plugin_owner`` with strict return-type validation and graceful
degradation when any companion plugin is absent or busy.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
from astrbot.api import logger

from .. import series_panels
from .memory_scope import is_member_privacy_muted

PLUGIN_ID = "astrbot_plugin_dynamics_memory"

_MILESTONE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS memory_milestones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER NOT NULL,
        umo TEXT NOT NULL,
        day_key TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        participants_json TEXT NOT NULL DEFAULT '[]',
        image_path TEXT NOT NULL DEFAULT '',
        keepsake_id TEXT NOT NULL DEFAULT '',
        world_deep_link TEXT NOT NULL DEFAULT '',
        memory_deep_link TEXT NOT NULL DEFAULT '',
        event_emitted INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'recorded',
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_milestones_umo_day
    ON memory_milestones(umo, day_key)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_milestones_memory
    ON memory_milestones(memory_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS archived_topic_dedup (
        dedup_key TEXT PRIMARY KEY,
        umo TEXT NOT NULL,
        topic_id TEXT NOT NULL DEFAULT '',
        memory_id INTEGER,
        created_at REAL NOT NULL
    )
    """,
)

_MILESTONE_KEYWORD_RE = re.compile(
    r"周年|纪念日|生日|百天|第\s*\d+\s*天|里程碑|通关|夺冠|毕业|一起完成|跨年|庆功|新纪录|合集|获奖|首胜",
    re.IGNORECASE,
)


def _clean_str(val: Any, max_len: int = 240) -> str:
    if not isinstance(val, str):
        return ""
    return re.sub(r"\s+", " ", val).strip()[:max_len]


def _normalize_importance(val: Any) -> float:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return 0.5
    if num > 1.0:
        num /= 10.0
    return max(0.0, min(1.0, num))


class MilestoneStore:
    """SQLite persistence for milestone keepsakes and archived-topic dedup keys."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode = WAL")
            await db.execute("PRAGMA busy_timeout = 5000")
            for stmt in _MILESTONE_SCHEMA:
                await db.execute(stmt)
            await db.commit()
        self._initialized = True

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        try:
            participants = json.loads(row["participants_json"] or "[]")
            if not isinstance(participants, list):
                participants = []
        except Exception:
            participants = []
        return {
            "id": int(row["id"]),
            "memory_id": int(row["memory_id"]),
            "umo": str(row["umo"]),
            "day_key": str(row["day_key"]),
            "title": str(row["title"]),
            "summary": str(row["summary"]),
            "participants": participants,
            "image_path": str(row["image_path"] or ""),
            "keepsake_id": str(row["keepsake_id"] or ""),
            "world_deep_link": str(row["world_deep_link"] or ""),
            "memory_deep_link": str(row["memory_deep_link"] or ""),
            "event_emitted": bool(row["event_emitted"]),
            "status": str(row["status"] or "recorded"),
            "created_at": float(row["created_at"]),
        }

    async def has_archived_topic(self, dedup_key: str) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT 1 FROM archived_topic_dedup WHERE dedup_key = ?",
                (dedup_key,),
            )
            return (await cur.fetchone()) is not None

    async def record_archived_topic(
        self, dedup_key: str, umo: str, topic_id: str, memory_id: int | None
    ) -> None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO archived_topic_dedup
                (dedup_key, umo, topic_id, memory_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dedup_key, umo, topic_id, memory_id, time.time()),
            )
            await db.commit()

    async def count_archived_topics(self) -> int:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT COUNT(*) FROM archived_topic_dedup")
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def count_milestones_for_day(self, umo: str, day_key: str) -> int:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT COUNT(*) FROM memory_milestones WHERE umo = ? AND day_key = ?",
                (umo, day_key),
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def get_milestone_for_memory(self, memory_id: int) -> dict[str, Any] | None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM memory_milestones WHERE memory_id = ? ORDER BY id DESC LIMIT 1",
                (int(memory_id),),
            )
            row = await cur.fetchone()
            return self._row_to_dict(row) if row is not None else None

    async def save_milestone(
        self,
        *,
        memory_id: int,
        umo: str,
        day_key: str,
        title: str,
        summary: str,
        participants: list[str],
        image_path: str,
        keepsake_id: str,
        world_deep_link: str,
        memory_deep_link: str,
        event_emitted: bool,
        status: str,
    ) -> dict[str, Any]:
        await self.initialize()
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                INSERT INTO memory_milestones (
                    memory_id, umo, day_key, title, summary, participants_json,
                    image_path, keepsake_id, world_deep_link, memory_deep_link,
                    event_emitted, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(memory_id),
                    umo,
                    day_key,
                    title[:120],
                    summary[:400],
                    json.dumps(participants[:12], ensure_ascii=False),
                    image_path[:500],
                    keepsake_id[:120],
                    world_deep_link[:300],
                    memory_deep_link[:300],
                    1 if event_emitted else 0,
                    status[:40],
                    now,
                ),
            )
            mid = int(cur.lastrowid)
            await db.commit()
            cur2 = await db.execute(
                "SELECT * FROM memory_milestones WHERE id = ?", (mid,)
            )
            return self._row_to_dict(await cur2.fetchone())

    async def list_milestones(
        self, *, umo: str | None = None, limit: int = 30
    ) -> dict[str, Any]:
        await self.initialize()
        limit = max(1, min(int(limit), 100))
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if umo:
                cur_t = await db.execute(
                    "SELECT COUNT(*) FROM memory_milestones WHERE umo = ?", (umo,)
                )
                total = int((await cur_t.fetchone())[0])
                cur = await db.execute(
                    "SELECT * FROM memory_milestones WHERE umo = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                    (umo, limit),
                )
            else:
                cur_t = await db.execute("SELECT COUNT(*) FROM memory_milestones")
                total = int((await cur_t.fetchone())[0])
                cur = await db.execute(
                    "SELECT * FROM memory_milestones ORDER BY created_at DESC, id DESC LIMIT ?",
                    (limit,),
                )
            rows = await cur.fetchall()
            return {"items": [self._row_to_dict(r) for r in rows], "total": total}


class EventBusAndMilestoneService:
    """Manages Dynamics Event Bus subscription and cross-plugin milestone keepsakes."""

    DAILY_MILESTONE_LIMIT_PER_GROUP = 1

    def __init__(self, plugin: Any, db_path: str | Path) -> None:
        self.plugin = plugin
        self.store = MilestoneStore(db_path)
        self._topic_sub_token: str | None = None
        self._last_archived_at: float | None = None
        self._session_archived_count: int = 0

    def ensure_subscribed(self) -> bool:
        """Dynamically subscribe to chat_dynamics ``topic_archived`` event."""
        if self._topic_sub_token:
            return True
        ctx = getattr(self.plugin, "context", None)
        if ctx is None:
            return False
        try:
            owner = series_panels.find_plugin_owner(ctx, "astrbot_plugin_chat_dynamics")
        except Exception:
            return False
        if owner is None:
            return False
        sub_fn = getattr(owner, "subscribe_dynamics_event", None)
        if not callable(sub_fn) or inspect.iscoroutinefunction(sub_fn):
            return False
        try:
            try:
                token = sub_fn(
                    "topic_archived", self.handle_topic_archived, owner=PLUGIN_ID
                )
            except TypeError:
                token = sub_fn("topic_archived", self.handle_topic_archived)
            if isinstance(token, str) and token.strip():
                self._topic_sub_token = token.strip()
                logger.info(
                    "[LivingMemory] Subscribed to Dynamics Event Bus topic_archived (token=%s)",
                    self._topic_sub_token[:24],
                )
                return True
        except Exception as exc:
            logger.debug("[LivingMemory] Event Bus subscription skipped: %s", exc)
        return False

    def unsubscribe(self) -> bool:
        """Safely unsubscribe from Dynamics Event Bus on plugin terminate."""
        token = self._topic_sub_token
        self._topic_sub_token = None
        if not token:
            return False
        ctx = getattr(self.plugin, "context", None)
        if ctx is None:
            return False
        try:
            owner = series_panels.find_plugin_owner(ctx, "astrbot_plugin_chat_dynamics")
            unsub_fn = (
                getattr(owner, "unsubscribe_dynamics_event", None) if owner else None
            )
            if callable(unsub_fn):
                res = unsub_fn(token)
                return res is True
        except Exception as exc:
            logger.debug("[LivingMemory] Event Bus unsubscribe ignored: %s", exc)
        return False

    async def handle_topic_archived(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Archive high-SNR topic summaries dispatched by chat_dynamics Event Bus."""
        if not isinstance(payload, dict):
            return {"accepted": False, "reason": "invalid_payload"}

        runtime_allowed = getattr(self.plugin, "series_runtime_allowed", None)
        if callable(runtime_allowed) and runtime_allowed() is False:
            return {"accepted": False, "reason": "runtime_disabled"}

        umo = _clean_str(payload.get("umo") or payload.get("session_id"), 160)
        topic_id = _clean_str(payload.get("topic_id"), 120)
        summary = _clean_str(
            payload.get("summary")
            or payload.get("topic_summary")
            or payload.get("topic"),
            800,
        )
        if not umo or len(summary) < 4:
            return {"accepted": False, "reason": "missing_umo_or_summary"}

        ctx = getattr(self.plugin, "context", None)
        try:
            if series_panels.is_series_group_excluded(ctx, umo=umo):
                return {"accepted": False, "reason": "group_excluded"}
        except Exception:
            pass

        # Filter privacy-muted participants
        raw_participants = payload.get("participants")
        kept_names: list[str] = []
        kept_identities: list[dict[str, Any]] = []
        muted_count = 0
        total_participants = 0

        if isinstance(raw_participants, list):
            for entry in raw_participants:
                member_id = ""
                display_name = ""
                if isinstance(entry, dict):
                    member_id = _clean_str(
                        entry.get("sender_id")
                        or entry.get("member_id")
                        or entry.get("id"),
                        80,
                    )
                    display_name = _clean_str(
                        entry.get("display_name") or entry.get("name") or member_id,
                        60,
                    )
                elif isinstance(entry, str) and entry.strip():
                    member_id = entry.strip()[:80]
                    display_name = member_id
                if not member_id and not display_name:
                    continue
                total_participants += 1
                if member_id and is_member_privacy_muted(ctx, umo, member_id, "memory"):
                    muted_count += 1
                    continue
                if display_name and display_name not in kept_names:
                    kept_names.append(display_name)
                if member_id:
                    kept_identities.append(
                        {
                            "sender_id": member_id,
                            "display_name": display_name or member_id,
                        }
                    )

        if (
            total_participants > 0
            and not kept_names
            and muted_count == total_participants
        ):
            return {"accepted": False, "reason": "all_participants_muted"}

        # Dedup check
        raw_key = f"{umo}|{topic_id}" if topic_id else f"{umo}|{summary[:160]}"
        dedup_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]
        if await self.store.has_archived_topic(dedup_key):
            return {"accepted": False, "reason": "duplicate"}

        try:
            msg_count = max(0, int(payload.get("message_count") or 0))
        except (TypeError, ValueError):
            msg_count = 0

        raw_topics = payload.get("topics")
        topics = (
            [_clean_str(t, 32) for t in raw_topics if _clean_str(t, 32)][:5]
            if isinstance(raw_topics, list)
            else []
        )
        title = _clean_str(payload.get("title") or payload.get("topic_title"), 80)
        if title and title not in topics:
            topics.insert(0, title[:32])

        importance = round(min(0.88, max(0.66, 0.66 + min(msg_count, 22) * 0.01)), 2)
        if isinstance(payload.get("importance"), (int, float)):
            importance = round(
                max(importance, _normalize_importance(payload["importance"])), 2
            )

        initializer = getattr(self.plugin, "initializer", None)
        engine = getattr(initializer, "memory_engine", None)
        conv_mgr = getattr(initializer, "conversation_manager", None)

        metadata: dict[str, Any] = {
            "session_id": umo,
            "source_session_id": umo,
            "memory_origin": "topic_archived",
            "memory_type": "EVENT",
            "interaction_type": "group_chat"
            if ":GroupMessage:" in umo
            else "private_chat",
            "memory_visibility": "group" if ":GroupMessage:" in umo else "private",
            "importance": importance,
            "status": "active",
            "topic_id": topic_id,
            "participants": kept_names,
            "participant_identities": kept_identities,
            "topics": topics[:5],
            "persona_summary": summary,
            "source_message_count": msg_count,
            "ended_at": payload.get("ended_at"),
            "create_time": time.time(),
        }

        doc_id: int | None = None
        if engine is not None and callable(getattr(engine, "add_memory", None)):
            try:
                doc_id = await engine.add_memory(
                    content=summary,
                    session_id=umo,
                    importance=importance,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.warning(
                    "[LivingMemory] topic_archived 写入记忆引擎失败: %s", exc
                )
                return {
                    "accepted": False,
                    "reason": f"engine_error:{type(exc).__name__}",
                }
        elif conv_mgr is not None and callable(getattr(conv_mgr, "add_message", None)):
            try:
                await conv_mgr.add_message(
                    session_id=umo,
                    role="system",
                    content=f"[归档话题摘要] {summary}",
                )
            except Exception as exc:
                logger.warning(
                    "[LivingMemory] topic_archived 写入待总结队列失败: %s", exc
                )
                return {"accepted": False, "reason": "queue_error"}
        else:
            return {"accepted": False, "reason": "engine_unavailable"}

        await self.store.record_archived_topic(dedup_key, umo, topic_id, doc_id)
        self._session_archived_count += 1
        self._last_archived_at = time.time()

        milestone_result = None
        if doc_id is not None and self.is_milestone_candidate(
            summary, metadata, importance
        ):
            milestone_result = await self.trigger_milestone_keepsake(
                memory_id=int(doc_id),
                umo=umo,
                content=summary,
                metadata=metadata,
                force=False,
            )

        return {
            "accepted": True,
            "doc_id": doc_id,
            "dedup_key": dedup_key,
            "filtered_muted_count": muted_count,
            "milestone": milestone_result,
        }

    @classmethod
    def is_milestone_candidate(
        cls, content: str, metadata: dict[str, Any], importance: float
    ) -> bool:
        """Determine whether a stored memory qualifies as a scrapbook milestone."""
        if not isinstance(metadata, dict):
            return False
        if metadata.get("milestone") is True:
            return True
        norm_imp = _normalize_importance(importance)
        origin = str(metadata.get("memory_origin") or "").strip().lower()
        if origin == "activity_experience" and norm_imp >= 0.78:
            return True

        identities = metadata.get("participant_identities")
        human_count = (
            sum(1 for i in identities if isinstance(i, dict) and not i.get("is_bot"))
            if isinstance(identities, list)
            else 0
        )
        participants = metadata.get("participants")
        if human_count == 0 and isinstance(participants, list):
            human_count = len([p for p in participants if str(p or "").strip()])

        has_kw = bool(_MILESTONE_KEYWORD_RE.search(str(content or "")))
        if norm_imp >= 0.85 and (human_count >= 2 or has_kw):
            return True
        if norm_imp >= 0.80 and human_count >= 2 and has_kw:
            return True
        return False

    async def _call_visual_polaroid(
        self, *, prompt: str, persona_id: str, umo: str
    ) -> str:
        ctx = getattr(self.plugin, "context", None)
        if ctx is None:
            return ""
        try:
            owner = series_panels.find_plugin_owner(ctx, "astrbot_plugin_visual_media")
        except Exception:
            return ""
        if owner is None:
            return ""
        gen_fn = getattr(owner, "generate_series_artwork", None)
        if not callable(gen_fn):
            return ""
        try:
            res = gen_fn(
                prompt,
                persona_id=persona_id,
                umo=umo,
                style="polaroid",
                source_plugin="dynamics_memory",
            )
            if inspect.isawaitable(res):
                res = await asyncio.wait_for(res, timeout=15.0)
            if isinstance(res, dict) and res.get("ok") is True:
                path = res.get("image_path")
                if isinstance(path, str) and path.strip():
                    return path.strip()
        except Exception as exc:
            logger.debug("[LivingMemory] visual_media polaroid skipped: %s", exc)
        return ""

    async def _call_world_keepsake(
        self,
        *,
        persona_id: str,
        umo: str,
        title: str,
        summary: str,
        sender_name: str,
        deep_link: str,
        image_path: str,
    ) -> tuple[str, str]:
        ctx = getattr(self.plugin, "context", None)
        if ctx is None:
            return "", ""
        try:
            owner = series_panels.find_plugin_owner(
                ctx, "astrbot_plugin_dynamics_world"
            )
        except Exception:
            return "", ""
        if owner is None:
            return "", ""
        reg_fn = getattr(owner, "register_external_keepsake", None)
        if not callable(reg_fn):
            return "", ""
        try:
            res = reg_fn(
                persona_id=persona_id,
                umo=umo,
                title=title,
                summary=summary,
                sender_name=sender_name,
                source_plugin="dynamics_memory",
                deep_link=deep_link,
                image_path=image_path,
                kind="memory_postcard",
            )
            if inspect.isawaitable(res):
                res = await asyncio.wait_for(res, timeout=5.0)
            if isinstance(res, dict):
                kid = res.get("keepsake_id") or res.get("id") or ""
                kid_str = str(kid).strip() if isinstance(kid, (str, int)) else ""
                wlink = res.get("deep_link") or res.get("world_deep_link") or ""
                wlink_str = (
                    wlink.strip()
                    if isinstance(wlink, str) and wlink.strip()
                    else (
                        f"astrbot_plugin_dynamics_world?page=cabin&keepsake={kid_str}"
                        if kid_str
                        else ""
                    )
                )
                return kid_str, wlink_str
        except Exception as exc:
            logger.debug("[LivingMemory] dynamics_world keepsake skipped: %s", exc)
        return "", ""

    async def _emit_milestone_event(self, payload: dict[str, Any]) -> bool:
        ctx = getattr(self.plugin, "context", None)
        if ctx is None:
            return False
        try:
            owner = series_panels.find_plugin_owner(ctx, "astrbot_plugin_chat_dynamics")
        except Exception:
            return False
        if owner is None:
            return False
        emit_fn = getattr(owner, "emit_dynamics_event", None)
        if not callable(emit_fn):
            return False
        try:
            res = emit_fn("milestone_achieved", payload)
            if inspect.isawaitable(res):
                res = await asyncio.wait_for(res, timeout=5.0)
            return isinstance(res, int) and res >= 0
        except Exception as exc:
            logger.debug("[LivingMemory] emit milestone_achieved skipped: %s", exc)
        return False

    async def trigger_milestone_keepsake(
        self,
        *,
        memory_id: int,
        umo: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Generate polaroid artwork, register 3D Room keepsake, emit event, and persist deep links."""
        meta = metadata if isinstance(metadata, dict) else {}
        clean_umo = _clean_str(umo or meta.get("session_id"), 160)
        clean_summary = _clean_str(
            meta.get("persona_summary") or meta.get("canonical_summary") or content, 300
        )
        if not clean_umo or not clean_summary:
            return {"ok": False, "reason": "missing_umo_or_summary"}

        day_key = datetime.now().strftime("%Y-%m-%d")
        if not force:
            existing_mem = await self.store.get_milestone_for_memory(memory_id)
            if existing_mem is not None:
                return {
                    "ok": False,
                    "reason": "already_recorded",
                    "milestone": existing_mem,
                }
            today_count = await self.store.count_milestones_for_day(clean_umo, day_key)
            if today_count >= self.DAILY_MILESTONE_LIMIT_PER_GROUP:
                return {
                    "ok": False,
                    "reason": "daily_limit_reached",
                    "day_key": day_key,
                }

        topics = meta.get("topics")
        topic_title = (
            _clean_str(topics[0], 32)
            if isinstance(topics, list) and topics and isinstance(topics[0], str)
            else ""
        )
        title = topic_title or clean_summary[:24]
        persona_id = _clean_str(meta.get("persona_id"), 80)

        participants: list[str] = []
        raw_p = meta.get("participants")
        if isinstance(raw_p, list):
            participants = [_clean_str(p, 40) for p in raw_p if _clean_str(p, 40)][:8]
        sender_name = participants[0] if participants else ""

        memory_deep_link = (
            f"{PLUGIN_ID}?page=memory&memory_id={int(memory_id)}&umo={clean_umo}"
        )
        polaroid_prompt = f"温馨纪念拍立得手账：{title}。{clean_summary[:140]}"

        image_path = await self._call_visual_polaroid(
            prompt=polaroid_prompt, persona_id=persona_id, umo=clean_umo
        )
        keepsake_id, world_deep_link = await self._call_world_keepsake(
            persona_id=persona_id,
            umo=clean_umo,
            title=title,
            summary=clean_summary,
            sender_name=sender_name,
            deep_link=memory_deep_link,
            image_path=image_path,
        )
        now = time.time()
        event_emitted = await self._emit_milestone_event(
            {
                "umo": clean_umo,
                "source_plugin": "dynamics_memory",
                "memory_id": int(memory_id),
                "title": title,
                "summary": clean_summary,
                "participants": participants,
                "image_path": image_path,
                "keepsake_id": keepsake_id,
                "deep_link": memory_deep_link,
                "world_deep_link": world_deep_link,
                "created_at": now,
            }
        )

        status = (
            "synced_world"
            if keepsake_id
            else ("polaroid_ready" if image_path else "recorded")
        )
        record = await self.store.save_milestone(
            memory_id=int(memory_id),
            umo=clean_umo,
            day_key=day_key,
            title=title,
            summary=clean_summary,
            participants=participants,
            image_path=image_path,
            keepsake_id=keepsake_id,
            world_deep_link=world_deep_link,
            memory_deep_link=memory_deep_link,
            event_emitted=event_emitted,
            status=status,
        )

        # Best-effort write-back to memory metadata so Memory Detail shows keepsake link
        initializer = getattr(self.plugin, "initializer", None)
        engine = getattr(initializer, "memory_engine", None)
        if engine is not None and callable(getattr(engine, "update_memory", None)):
            try:
                await engine.update_memory(
                    int(memory_id),
                    metadata_updates={
                        "milestone_keepsake": {
                            "milestone_id": record["id"],
                            "title": title,
                            "keepsake_id": keepsake_id,
                            "world_deep_link": world_deep_link,
                            "memory_deep_link": memory_deep_link,
                            "image_path": image_path,
                            "status": status,
                            "created_at": record["created_at"],
                        }
                    },
                )
            except Exception as exc:
                logger.debug("[LivingMemory] 写回纪念品深链元数据忽略: %s", exc)

        return {"ok": True, "milestone": record}

    def schedule_if_milestone(
        self, memory_id: int, content: str, metadata: dict[str, Any], importance: float
    ) -> None:
        """Non-blocking hook invoked when a new memory is persisted."""
        if not self.is_milestone_candidate(content, metadata, importance):
            return
        umo = _clean_str(
            metadata.get("session_id") or metadata.get("source_session_id"), 160
        )
        if not umo:
            return
        create_task = getattr(self.plugin, "_create_tracked_task", None)
        coro = self.trigger_milestone_keepsake(
            memory_id=int(memory_id),
            umo=umo,
            content=content,
            metadata=metadata,
            force=False,
        )
        try:
            if callable(create_task):
                create_task(coro)
            else:
                asyncio.create_task(coro)
        except RuntimeError:
            coro.close()

    async def get_status_and_milestones(
        self, *, umo: str | None = None, limit: int = 30
    ) -> dict[str, Any]:
        """Return Event Bus subscription state and recorded milestone keepsakes."""
        self.ensure_subscribed()
        data = await self.store.list_milestones(umo=umo, limit=limit)
        archived_total = await self.store.count_archived_topics()
        return {
            "event_bus": {
                "subscribed": bool(self._topic_sub_token),
                "topic_archived_total": archived_total,
                "session_archived_count": self._session_archived_count,
                "last_archived_at": self._last_archived_at,
            },
            "daily_limit_per_group": self.DAILY_MILESTONE_LIMIT_PER_GROUP,
            **data,
        }
