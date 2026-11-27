"""Self-service member memory card rendering and member correction proposal queue.

Provides:
- ``CorrectionQueueStore``: persistent SQLite queue for member-submitted memory
  correction proposals, pending admin approval in the Dashboard.
- ``MemberMemoryCardService``: strictly scoped personal memory query, Pillow
  card image rendering (with graceful plain-text fallback), and safe member
  correction submission.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
from astrbot.api import logger

from ..storage.sqlite_connection import open_connection as _open_sqlite_connection
from .memory_scope import (
    is_event_excluded_by_series,
    is_event_memory_allowed,
    is_event_privacy_muted,
    resolve_event_identity,
    resolve_memory_scope,
)

_CORRECTION_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS memory_correction_proposals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        requester_id TEXT NOT NULL,
        requester_name TEXT NOT NULL DEFAULT '',
        original_preview TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL,
        proposed_text TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending',
        reviewer TEXT NOT NULL DEFAULT '',
        review_note TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL,
        reviewed_at REAL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_correction_proposals_status
    ON memory_correction_proposals(status, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_correction_proposals_memory
    ON memory_correction_proposals(memory_id, status)
    """,
)

_CJK_FONT_CANDIDATES = (
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    # Linux / Docker
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

_RAW_ID_RE = re.compile(r"(?<!#)\b\d{6,12}\b")


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _sanitize_card_text(text: str, *, max_len: int = 110) -> str:
    """Compact whitespace and mask raw long numeric account IDs in card text."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = _RAW_ID_RE.sub(lambda m: f"***{m.group(0)[-3:]}", cleaned)
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def _format_ts(ts: Any) -> str:
    try:
        val = float(ts)
        if val > 100_000_000_000:
            val /= 1000.0
        if val > 0:
            return datetime.fromtimestamp(val).strftime("%m-%d")
    except Exception:
        pass
    return ""


def _event_str(event: Any, method_name: str, attr_name: str = "") -> str:
    fn = getattr(event, method_name, None)
    if callable(fn):
        try:
            val = fn()
            if val is not None:
                return str(val).strip()
        except Exception:
            pass
    val = getattr(event, attr_name or method_name, "")
    return str(val).strip() if val is not None else ""


def _memory_belongs_to_member(
    metadata: dict[str, Any],
    sender_id: str,
    sender_name: str,
    canonical_identity: str,
) -> bool:
    """Return True iff the memory record belongs to or explicitly involves the member."""
    sid = str(sender_id or "").strip()
    sname = str(sender_name or "").strip().casefold()
    cid = str(canonical_identity or "").strip().casefold()

    if sid:
        meta_sender = str(metadata.get("sender_id") or "").strip()
        if meta_sender and meta_sender == sid:
            return True

    identities = metadata.get("participant_identities")
    if isinstance(identities, list):
        for item in identities:
            if not isinstance(item, dict) or item.get("is_bot"):
                continue
            item_sid = str(item.get("sender_id") or "").strip()
            item_name = str(item.get("display_name") or "").strip().casefold()
            if sid and item_sid and item_sid == sid:
                return True
            if sname and item_name and item_name == sname:
                return True
            if cid and item_name and item_name == cid:
                return True

    participants = metadata.get("participants")
    if isinstance(participants, list):
        for p in participants:
            p_norm = str(p or "").strip().casefold()
            if not p_norm:
                continue
            if (
                (sid and p_norm == sid.casefold())
                or (sname and p_norm == sname)
                or (cid and p_norm == cid)
            ):
                return True

    return False


def _session_matches(metadata: dict[str, Any], allowed_sessions: set[str]) -> bool:
    for key in ("session_id", "source_session_id"):
        val = str(metadata.get(key) or "").strip()
        if val and val in allowed_sessions:
            return True
    return False


class CorrectionQueueStore:
    """SQLite-backed pending queue for member memory correction proposals."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._initialized = False

    @asynccontextmanager
    async def _open_db(
        self, *, apply_journal_mode: bool = False, readonly: bool = False
    ):
        db = await _open_sqlite_connection(
            self.db_path,
            apply_journal_mode=apply_journal_mode,
            readonly=readonly,
        )
        try:
            yield db
        finally:
            await db.close()

    async def initialize(self) -> None:
        if self._initialized:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with self._open_db(apply_journal_mode=True) as db:
            for stmt in _CORRECTION_SCHEMA:
                await db.execute(stmt)
            await db.commit()
        self._initialized = True

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "memory_id": int(row["memory_id"]),
            "session_id": str(row["session_id"]),
            "requester_id": str(row["requester_id"]),
            "requester_name": str(row["requester_name"] or ""),
            "original_preview": str(row["original_preview"] or ""),
            "reason": str(row["reason"] or ""),
            "proposed_text": str(row["proposed_text"] or ""),
            "status": str(row["status"] or "pending"),
            "reviewer": str(row["reviewer"] or ""),
            "review_note": str(row["review_note"] or ""),
            "created_at": float(row["created_at"]),
            "reviewed_at": float(row["reviewed_at"])
            if row["reviewed_at"] is not None
            else None,
        }

    async def submit(
        self,
        *,
        memory_id: int,
        session_id: str,
        requester_id: str,
        requester_name: str,
        original_preview: str,
        reason: str,
        proposed_text: str = "",
    ) -> dict[str, Any]:
        await self.initialize()
        now = time.time()
        clean_reason = str(reason or "").strip()[:500]
        clean_proposed = str(proposed_text or "").strip()[:800]
        clean_preview = str(original_preview or "").strip()[:300]
        if not clean_reason:
            raise ValueError("纠错说明不能为空")

        async with self._open_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                INSERT INTO memory_correction_proposals (
                    memory_id, session_id, requester_id, requester_name,
                    original_preview, reason, proposed_text, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    int(memory_id),
                    str(session_id),
                    str(requester_id),
                    str(requester_name or "")[:80],
                    clean_preview,
                    clean_reason,
                    clean_proposed,
                    now,
                ),
            )
            proposal_id = int(cursor.lastrowid)
            await db.commit()
            cur = await db.execute(
                "SELECT * FROM memory_correction_proposals WHERE id = ?",
                (proposal_id,),
            )
            row = await cur.fetchone()
            return self._row_to_dict(row)

    async def list_proposals(
        self, *, status: str = "all", limit: int = 50
    ) -> dict[str, Any]:
        await self.initialize()
        limit = max(1, min(int(limit), 200))
        norm_status = str(status or "all").strip().lower()
        async with self._open_db(readonly=True) as db:
            db.row_factory = aiosqlite.Row
            cur_pending = await db.execute(
                "SELECT COUNT(*) AS cnt FROM memory_correction_proposals WHERE status = 'pending'"
            )
            pending_count = int((await cur_pending.fetchone())["cnt"])

            if norm_status in {"pending", "approved", "rejected"}:
                cur_total = await db.execute(
                    "SELECT COUNT(*) AS cnt FROM memory_correction_proposals WHERE status = ?",
                    (norm_status,),
                )
                total = int((await cur_total.fetchone())["cnt"])
                cur = await db.execute(
                    """
                    SELECT * FROM memory_correction_proposals
                    WHERE status = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (norm_status, limit),
                )
            else:
                cur_total = await db.execute(
                    "SELECT COUNT(*) AS cnt FROM memory_correction_proposals"
                )
                total = int((await cur_total.fetchone())["cnt"])
                cur = await db.execute(
                    """
                    SELECT * FROM memory_correction_proposals
                    ORDER BY CASE WHEN status = 'pending' THEN 0 ELSE 1 END,
                             created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = await cur.fetchall()
            return {
                "items": [self._row_to_dict(row) for row in rows],
                "pending_count": pending_count,
                "total": total,
            }

    async def get_proposal(self, proposal_id: int) -> dict[str, Any] | None:
        await self.initialize()
        async with self._open_db(readonly=True) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM memory_correction_proposals WHERE id = ?",
                (int(proposal_id),),
            )
            row = await cur.fetchone()
            return self._row_to_dict(row) if row is not None else None

    async def review(
        self,
        proposal_id: int,
        *,
        decision: str,
        reviewer: str = "admin",
        review_note: str = "",
        applied_text: str = "",
    ) -> dict[str, Any]:
        await self.initialize()
        norm_decision = str(decision or "").strip().lower()
        if norm_decision not in {"approve", "reject"}:
            raise ValueError("decision 必须为 approve 或 reject")
        next_status = "approved" if norm_decision == "approve" else "rejected"
        now = time.time()

        async with self._open_db() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM memory_correction_proposals WHERE id = ?",
                (int(proposal_id),),
            )
            row = await cur.fetchone()
            if row is None:
                raise LookupError("纠错申请不存在")
            if str(row["status"]) != "pending":
                raise RuntimeError("该纠错申请已被处理")

            next_proposed = (
                str(applied_text).strip()[:800]
                if applied_text
                else str(row["proposed_text"] or "")
            )
            await db.execute(
                """
                UPDATE memory_correction_proposals
                SET status = ?, reviewer = ?, review_note = ?, proposed_text = ?, reviewed_at = ?
                WHERE id = ?
                """,
                (
                    next_status,
                    str(reviewer or "admin")[:80],
                    str(review_note or "").strip()[:500],
                    next_proposed,
                    now,
                    int(proposal_id),
                ),
            )
            await db.commit()
            cur2 = await db.execute(
                "SELECT * FROM memory_correction_proposals WHERE id = ?",
                (int(proposal_id),),
            )
            return self._row_to_dict(await cur2.fetchone())


class MemberMemoryCardService:
    """Query strictly scoped personal memories and render member cards."""

    def __init__(self, temp_dir: str | Path | None = None) -> None:
        self._temp_dir = (
            Path(temp_dir)
            if temp_dir
            else Path(tempfile.gettempdir()) / "livingmemory_cards"
        )

    async def collect_member_card_data(
        self,
        memory_engine: Any,
        event: Any,
        config_manager: Any,
        context: Any,
    ) -> dict[str, Any]:
        """Collect strictly isolated personal & shared memories for the sender."""
        if is_event_excluded_by_series(context, event) or not is_event_memory_allowed(
            config_manager, event
        ):
            return {"allowed": False, "reason": "scope_denied"}
        if is_event_privacy_muted(context, event, "memory"):
            return {"allowed": False, "reason": "privacy_muted"}

        umo = _event_str(event, "unified_msg_origin")
        sender_id = _event_str(event, "get_sender_id", "sender_id")
        sender_name = (
            _event_str(event, "get_sender_name", "sender_name") or sender_id or "群友"
        )
        canonical_identity = resolve_event_identity(config_manager, event)
        resolved_scope = resolve_memory_scope(config_manager, event)
        allowed_sessions = {s for s in (umo, resolved_scope) if s}

        if not sender_id and not sender_name:
            return {"allowed": False, "reason": "missing_identity"}

        db = getattr(memory_engine, "db_connection", None)
        if db is None:
            return {"allowed": False, "reason": "engine_unavailable"}

        cursor = await db.execute(
            """
            SELECT id, text, metadata, created_at
            FROM documents
            WHERE COALESCE(json_extract(metadata, '$.status'), 'active') = 'active'
            ORDER BY COALESCE(CAST(json_extract(metadata, '$.create_time') AS REAL), 0) DESC,
                     id DESC
            LIMIT 300
            """
        )
        rows = await cursor.fetchall()

        personal: list[dict[str, Any]] = []
        shared: list[dict[str, Any]] = []
        topic_counter: Counter[str] = Counter()

        for row in rows:
            doc_id = int(
                row["id"] if isinstance(row, dict) or hasattr(row, "keys") else row[0]
            )
            raw_text = str(
                row["text"]
                if isinstance(row, dict) or hasattr(row, "keys")
                else row[1] or ""
            )
            raw_meta = (
                row["metadata"]
                if isinstance(row, dict) or hasattr(row, "keys")
                else row[2]
            )
            metadata = _parse_metadata(raw_meta)

            if not _session_matches(metadata, allowed_sessions):
                continue
            if not _memory_belongs_to_member(
                metadata, sender_id, sender_name, canonical_identity
            ):
                continue

            summary_src = (
                metadata.get("persona_summary")
                or metadata.get("canonical_summary")
                or raw_text
            )
            summary = _sanitize_card_text(str(summary_src or ""), max_len=96)
            if not summary:
                continue

            imp_raw = metadata.get("importance", 0.5)
            try:
                importance = float(imp_raw)
                if importance > 1.0:
                    importance = importance / 10.0
            except (TypeError, ValueError):
                importance = 0.5

            topics = metadata.get("topics")
            if isinstance(topics, list):
                for topic in topics:
                    t_str = str(topic or "").strip()
                    if t_str and len(t_str) <= 24:
                        topic_counter[t_str] += 1

            entry = {
                "id": doc_id,
                "summary": summary,
                "memory_type": str(metadata.get("memory_type") or "GENERAL").upper(),
                "importance": round(importance, 2),
                "date": _format_ts(metadata.get("create_time")),
            }

            identities = metadata.get("participant_identities")
            human_participants = (
                [
                    item
                    for item in identities
                    if isinstance(item, dict) and not item.get("is_bot")
                ]
                if isinstance(identities, list)
                else []
            )
            if len(human_participants) >= 2:
                shared.append(entry)
            else:
                personal.append(entry)

        personal.sort(key=lambda item: (item["importance"], item["id"]), reverse=True)
        shared.sort(key=lambda item: (item["importance"], item["id"]), reverse=True)

        group_tail = umo.split(":")[-1] if ":" in umo else umo
        group_label = (
            f"群聊 ···{group_tail[-4:]}"
            if len(group_tail) > 4
            else (group_tail or "当前会话")
        )

        return {
            "allowed": True,
            "member_name": sender_name[:32],
            "sender_id_tail": sender_id[-4:] if len(sender_id) >= 4 else sender_id,
            "group_label": group_label,
            "session_id": umo,
            "total_count": len(personal) + len(shared),
            "top_topics": [topic for topic, _ in topic_counter.most_common(6)],
            "personal_memories": personal[:6],
            "shared_highlights": shared[:3],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    @staticmethod
    def _load_font(size: int) -> Any:
        from PIL import ImageFont

        for font_path in _CJK_FONT_CANDIDATES:
            if Path(font_path).is_file():
                try:
                    return ImageFont.truetype(font_path, size=size)
                except Exception:
                    continue
        return ImageFont.load_default()

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list[str]:
        lines: list[str] = []
        buf = ""
        width = 0.0
        for ch in text:
            w = 1.0 if ord(ch) > 127 else 0.55
            if width + w > max_chars and buf:
                lines.append(buf)
                buf = ch
                width = w
            else:
                buf += ch
                width += w
        if buf:
            lines.append(buf)
        return lines or [""]

    def render_card_image(
        self, card_data: dict[str, Any], output_path: Path | None = None
    ) -> Path | None:
        """Render a clean PNG card image via Pillow; returns None on fallback."""
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            logger.debug(
                "[MemberCard] Pillow unavailable, using text fallback: %s", exc
            )
            return None

        try:
            width = 860
            pad = 40
            personal = card_data.get("personal_memories") or []
            shared = card_data.get("shared_highlights") or []
            topics = card_data.get("top_topics") or []

            # Pre-wrap text to calculate exact canvas height
            wrapped_personal = [
                (item, self._wrap_text(item["summary"], 38)[:2]) for item in personal
            ]
            wrapped_shared = [
                (item, self._wrap_text(item["summary"], 38)[:2]) for item in shared
            ]

            height = 230
            if topics:
                height += 72
            height += 48 + (
                sum(32 + len(lines) * 26 for _, lines in wrapped_personal)
                if wrapped_personal
                else 52
            )
            if wrapped_shared:
                height += 54 + sum(32 + len(lines) * 26 for _, lines in wrapped_shared)
            height += 86
            height = max(420, min(height, 1800))

            img = Image.new("RGB", (width, height), (247, 248, 252))
            draw = ImageDraw.Draw(img)

            f_title = self._load_font(28)
            f_sub = self._load_font(16)
            f_sec = self._load_font(18)
            f_body = self._load_font(17)
            f_small = self._load_font(14)

            # Header card
            draw.rounded_rectangle(
                [(pad, 28), (width - pad, 148)],
                radius=18,
                fill=(30, 41, 59),
            )
            member_name = str(card_data.get("member_name") or "群友")
            group_label = str(card_data.get("group_label") or "当前群聊")
            total_count = int(card_data.get("total_count") or 0)
            draw.text(
                (pad + 26, 48),
                f"{member_name} 的记忆卡片",
                fill=(248, 250, 252),
                font=f_title,
            )
            draw.text(
                (pad + 26, 98),
                f"{group_label}  ·  与你相关的长期记忆 {total_count} 条  ·  {card_data.get('generated_at', '')}",
                fill=(148, 163, 184),
                font=f_sub,
            )

            y = 172

            # Top topics pills
            if topics:
                draw.text((pad, y), "常聊话题", fill=(71, 85, 105), font=f_sec)
                y += 30
                x = pad
                for topic in topics[:6]:
                    pill_text = f"#{topic}"
                    tw = max(64, int(len(pill_text) * 15) + 22)
                    if x + tw > width - pad:
                        break
                    draw.rounded_rectangle(
                        [(x, y), (x + tw, y + 30)],
                        radius=15,
                        fill=(224, 231, 255),
                    )
                    draw.text(
                        (x + 12, y + 6), pill_text, fill=(55, 48, 163), font=f_small
                    )
                    x += tw + 10
                y += 48

            # Personal memories
            draw.text((pad, y), "个人记忆摘要", fill=(30, 41, 59), font=f_sec)
            y += 32
            if not wrapped_personal:
                draw.rounded_rectangle(
                    [(pad, y), (width - pad, y + 48)],
                    radius=12,
                    fill=(255, 255, 255),
                    outline=(226, 232, 240),
                )
                draw.text(
                    (pad + 18, y + 14),
                    "暂时还没有沉淀下你的专属记忆，多在群里聊聊天吧。",
                    fill=(100, 116, 139),
                    font=f_body,
                )
                y += 62
            else:
                for item, lines in wrapped_personal:
                    box_h = 22 + len(lines) * 25
                    draw.rounded_rectangle(
                        [(pad, y), (width - pad, y + box_h)],
                        radius=12,
                        fill=(255, 255, 255),
                        outline=(226, 232, 240),
                    )
                    badge = f"#{item['id']}"
                    draw.rounded_rectangle(
                        [(pad + 14, y + 11), (pad + 86, y + 35)],
                        radius=8,
                        fill=(238, 242, 255),
                    )
                    draw.text(
                        (pad + 22, y + 14), badge, fill=(67, 56, 202), font=f_small
                    )
                    ty = y + 11
                    for line in lines:
                        draw.text((pad + 98, ty), line, fill=(15, 23, 42), font=f_body)
                        ty += 25
                    if item.get("date"):
                        draw.text(
                            (width - pad - 62, y + 14),
                            str(item["date"]),
                            fill=(148, 163, 184),
                            font=f_small,
                        )
                    y += box_h + 10

            # Shared highlights
            if wrapped_shared:
                y += 8
                draw.text((pad, y), "群内共同经历", fill=(30, 41, 59), font=f_sec)
                y += 32
                for item, lines in wrapped_shared:
                    box_h = 22 + len(lines) * 25
                    draw.rounded_rectangle(
                        [(pad, y), (width - pad, y + box_h)],
                        radius=12,
                        fill=(255, 251, 235),
                        outline=(253, 230, 138),
                    )
                    draw.text(
                        (pad + 16, y + 13),
                        f"#{item['id']}",
                        fill=(180, 83, 9),
                        font=f_small,
                    )
                    ty = y + 11
                    for line in lines:
                        draw.text((pad + 92, ty), line, fill=(69, 26, 3), font=f_body)
                        ty += 25
                    y += box_h + 10

            # Footer tip
            draw.line(
                [(pad, height - 56), (width - pad, height - 56)],
                fill=(226, 232, 240),
                width=1,
            )
            draw.text(
                (pad, height - 40),
                "发现记错了？发送  /纠错 <编号> <正确说明>  即可提交修正申请（仅展示你本人在本群的记忆）",
                fill=(100, 116, 139),
                font=f_small,
            )

            if output_path is None:
                self._temp_dir.mkdir(parents=True, exist_ok=True)
                output_path = (
                    self._temp_dir / f"memory_card_{int(time.time() * 1000)}.png"
                )
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format="PNG")
            return output_path
        except Exception as exc:
            logger.warning(
                "[MemberCard] Failed to render PNG card, falling back to text: %s", exc
            )
            return None

    @staticmethod
    def render_card_text(card_data: dict[str, Any]) -> str:
        """Render clean plain-text memory card fallback."""
        member = card_data.get("member_name") or "群友"
        group_label = card_data.get("group_label") or "当前会话"
        total = int(card_data.get("total_count") or 0)
        lines = [f"【{member} 的记忆卡片】({group_label} · 共 {total} 条相关记忆)"]
        topics = card_data.get("top_topics") or []
        if topics:
            lines.append("常聊话题：" + " ".join(f"#{t}" for t in topics[:6]))

        personal = card_data.get("personal_memories") or []
        if personal:
            lines.append("\n个人记忆：")
            for item in personal:
                date_part = f" ({item['date']})" if item.get("date") else ""
                lines.append(f"- #{item['id']}{date_part} {item['summary']}")
        else:
            lines.append("\n暂未收录与你直接相关的个人记忆。")

        shared = card_data.get("shared_highlights") or []
        if shared:
            lines.append("\n共同经历：")
            for item in shared:
                lines.append(f"- #{item['id']} {item['summary']}")

        lines.append("\n提示：发现记忆有误可发送 /纠错 <编号> <说明> 提交修正。")
        return "\n".join(lines)

    async def submit_member_correction(
        self,
        *,
        memory_engine: Any,
        correction_store: CorrectionQueueStore,
        event: Any,
        config_manager: Any,
        context: Any,
        memory_id: int,
        description: str,
    ) -> dict[str, Any]:
        """Validate ownership/scope and enqueue a member correction proposal."""
        if is_event_excluded_by_series(context, event) or not is_event_memory_allowed(
            config_manager, event
        ):
            return {"ok": False, "error": "当前会话未启用记忆功能。"}
        if is_event_privacy_muted(context, event, "memory"):
            return {"ok": False, "error": "你已开启记忆隐私静音，无法提交纠错。"}

        desc = str(description or "").strip()
        if not desc:
            return {
                "ok": False,
                "error": "请填写纠错说明，用法：/纠错 <记忆编号> <说明>",
            }

        memory = await memory_engine.get_memory_by_id(int(memory_id))
        if not memory:
            return {"ok": False, "error": f"未找到编号为 #{memory_id} 的记忆。"}

        metadata = _parse_metadata(memory.get("metadata"))
        if str(metadata.get("status") or "active").lower() != "active":
            return {
                "ok": False,
                "error": f"记忆 #{memory_id} 当前不是活跃状态，无法纠错。",
            }

        umo = _event_str(event, "unified_msg_origin")
        resolved_scope = resolve_memory_scope(config_manager, event)
        allowed_sessions = {s for s in (umo, resolved_scope) if s}
        if not _session_matches(metadata, allowed_sessions):
            return {
                "ok": False,
                "error": "只能对当前群聊/会话中与你相关的记忆提交纠错。",
            }

        sender_id = _event_str(event, "get_sender_id", "sender_id")
        sender_name = (
            _event_str(event, "get_sender_name", "sender_name") or sender_id or "群友"
        )
        canonical_identity = resolve_event_identity(config_manager, event)
        if not _memory_belongs_to_member(
            metadata, sender_id, sender_name, canonical_identity
        ):
            return {"ok": False, "error": "只能对你本人参与或相关的记忆提交纠错。"}

        preview = _sanitize_card_text(
            str(
                metadata.get("persona_summary")
                or metadata.get("canonical_summary")
                or memory.get("text")
                or ""
            ),
            max_len=160,
        )
        proposal = await correction_store.submit(
            memory_id=int(memory_id),
            session_id=umo,
            requester_id=sender_id or sender_name,
            requester_name=sender_name,
            original_preview=preview,
            reason=desc,
            proposed_text=desc,
        )
        return {"ok": True, "proposal": proposal}
