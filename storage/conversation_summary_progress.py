"""Durable reflection progress for receipt-managed conversations."""

import asyncio
import hashlib
import json
import time

from ..core.memory_source import summary_window_fingerprint
from ..core.models.conversation_models import Message


TERMINAL_SUMMARY_STAGES = {"policy", "provider_terminal"}
SUMMARY_RETRY_LIMIT = 3
SUMMARY_COOLDOWN_SECONDS = 900
ROLLBACK_PAUSE = "managed_summary_source_rollback_v1"


class ConversationSummaryProgress:
    async def initialize_summary_progress(self) -> None:
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS managed_summary_progress (
                session_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('saved', 'refused')),
                PRIMARY KEY (session_id, message_id)
            )
        """)
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS managed_summary_refusals (
                session_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                window TEXT NOT NULL,
                PRIMARY KEY (session_id, fingerprint)
            )
        """)
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_managed_pending_summary_schedule
            ON sessions (COALESCE(json_extract(metadata,'$.pending_summary.recovery_after'),
                                  json_extract(metadata,'$.pending_summary.retry_after'), 0), id)
            WHERE session_id LIKE 'narrative:v1:%' AND json_valid(metadata)
                AND json_type(metadata,'$.pending_summary')='object'
        """)

    async def managed_summary_recovery_candidates(self, after_id: int, now: float, limit: int):
        """Read a bounded page of due scopes and an already archived actor."""
        async with self.connection.execute("""
            SELECT s.id, s.session_id, e.envelope FROM (
                SELECT id, session_id FROM sessions
                WHERE session_id LIKE 'narrative:v1:%' AND json_valid(metadata)
                    AND json_type(metadata,'$.pending_summary')='object'
                    AND COALESCE(json_extract(metadata,'$.pending_summary.recovery_after'),
                                 json_extract(metadata,'$.pending_summary.retry_after'), 0) <= ?
                    AND id > ?
                ORDER BY id LIMIT ?
            ) s LEFT JOIN narrative_events e ON e.event_key=(
                SELECT n.event_key FROM narrative_events n
                WHERE n.session_id=s.session_id AND json_extract(n.envelope,'$.type')='message.received'
                ORDER BY n.timestamp DESC LIMIT 1
            ) ORDER BY s.id
        """, (now, after_id, limit)) as cursor:
            return await cursor.fetchall()

    async def _managed_summary_rows(self, session_id, message_ids):
        async with self.connection.execute("""
            SELECT m.* FROM messages m
            JOIN json_each(?) j ON m.id=j.value
            WHERE m.session_id=? ORDER BY CAST(j.key AS INTEGER)
        """, (json.dumps(message_ids), session_id)) as cursor:
            return [self._row_to_message(row) for row in await cursor.fetchall()]

    async def _mark_managed_summary_rows(self, session_id, messages, state):
        await self.connection.executemany("""
            INSERT INTO managed_summary_progress (session_id, message_id, state)
            VALUES (?, ?, ?)
            ON CONFLICT (session_id, message_id) DO UPDATE SET state=excluded.state
        """, [(session_id, message.id, state) for message in messages])

    async def _save_managed_summary_metadata(self, session_id, metadata):
        await self.connection.execute(
            "UPDATE sessions SET metadata=? WHERE session_id=?",
            (json.dumps(metadata, ensure_ascii=False), session_id),
        )

    async def _cleanup_managed_summary_progress(self, session_id):
        await self.connection.execute("""
            DELETE FROM managed_summary_progress
            WHERE session_id=? AND message_id NOT IN (SELECT id FROM messages WHERE session_id=?)
        """, (session_id, session_id))
        # An explicitly deleted row cannot be part of a future exact retry.
        # Retain the surviving refused rows outside automatic generation.

    async def _adopt_legacy_summary_progress(self, session_id, metadata):
        """Freeze the original refused/pending range before allowing new plans."""
        progress = max(0, int(metadata.get("last_summarized_index", 0) or 0))
        pending = metadata.get("pending_summary")
        if isinstance(pending, dict):
            start = max(0, int(pending.get("start_index", 0) or 0))
            end = max(start, int(pending.get("end_index", start) or start))
            progress = min(progress, start)
            if end > start:
                messages = await self.get_messages_range(session_id, start, end - start)
                if len(messages) == end - start:
                    pending = self._managed_summary_window(pending, messages)
                    if pending.get("failure_stage") in TERMINAL_SUMMARY_STAGES:
                        await self._save_managed_refusal(session_id, pending, messages)
                        pending = None
                else:
                    # Preserve any surviving part of a legacy refusal without
                    # ever submitting it as a shortened retry.
                    if pending.get("failure_stage") in TERMINAL_SUMMARY_STAGES:
                        fingerprint = hashlib.sha256(f"legacy:{start}:{end}".encode()).hexdigest()
                        await self._save_managed_refusal(session_id, {
                            **pending, "fingerprint": fingerprint,
                            "message_ids": [message.id for message in messages],
                            "retry_unavailable_reason": "legacy_range_incomplete",
                        }, messages)
                    pending = None
        await self.connection.execute("""
            INSERT OR IGNORE INTO managed_summary_progress (session_id, message_id, state)
            SELECT session_id, id, 'saved' FROM messages
            WHERE session_id=? ORDER BY timestamp, id LIMIT ?
        """, (session_id, progress))
        metadata["managed_summary_progress_v1"] = True
        metadata["pending_summary"] = pending
        await self._update_managed_summary_cursor(session_id, metadata)
        await self._save_managed_summary_metadata(session_id, metadata)

    @staticmethod
    def _managed_summary_window(pending, messages):
        return {
            **pending,
            "message_ids": [message.id for message in messages],
            "fingerprint": summary_window_fingerprint(messages),
        }

    async def _save_managed_refusal(self, session_id, window, messages):
        await self.connection.execute("""
            INSERT INTO managed_summary_refusals (session_id, fingerprint, window)
            VALUES (?, ?, ?)
            ON CONFLICT (session_id, fingerprint) DO UPDATE SET window=excluded.window
        """, (session_id, window["fingerprint"], json.dumps(window)))
        await self._mark_managed_summary_rows(session_id, messages, "refused")

    async def _update_managed_summary_cursor(self, session_id, metadata):
        async with self.connection.execute("""
            SELECT m.timestamp, m.id FROM messages m
            LEFT JOIN managed_summary_progress p ON p.session_id=m.session_id AND p.message_id=m.id
            WHERE m.session_id=? AND COALESCE(p.state, '')!='saved'
            ORDER BY m.timestamp, m.id LIMIT 1
        """, (session_id,)) as cursor:
            first_unsaved = await cursor.fetchone()
        if first_unsaved is None:
            position = await self.get_message_count(session_id)
        else:
            timestamp, message_id = first_unsaved
            async with self.connection.execute("""
                SELECT COUNT(*) FROM messages WHERE session_id=?
                AND (timestamp<? OR (timestamp=? AND id<?))
            """, (session_id, timestamp, timestamp, message_id)) as cursor:
                position = (await cursor.fetchone())[0]
        metadata["last_summarized_index"] = position

    async def _managed_summary_position(self, session_id, message):
        async with self.connection.execute("""
            SELECT COUNT(*) FROM messages WHERE session_id=?
            AND (timestamp<? OR (timestamp=? AND id<?))
        """, (session_id, message.timestamp, message.timestamp, message.id)) as cursor:
            return (await cursor.fetchone())[0]

    async def get_managed_summary_window(
        self, session_id: str, limit: int, threshold: int, *, retry_refused: bool = False,
    ) -> tuple[int, list[Message], dict | None]:
        """Return one untreated run or the exact original pending/refused snapshot."""
        async with self._write_lock:
            session = await self.get_session(session_id)
            if session is None:
                return 0, [], None
            metadata = session.metadata
            try:
                if not metadata.get("managed_summary_progress_v1"):
                    await self._adopt_legacy_summary_progress(session_id, metadata)
                    await self._commit_write_transaction()
                else:
                    pending = metadata.get("pending_summary")
                    if (isinstance(pending, dict)
                            and pending.get("failure_stage") == "provider_terminal"
                            and pending.get("error_code") == ROLLBACK_PAUSE
                            and "preserved_pending_summary" in pending):
                        metadata["pending_summary"] = pending["preserved_pending_summary"]
                        await self._update_managed_summary_cursor(session_id, metadata)
                        await self._save_managed_summary_metadata(session_id, metadata)
                        await self._commit_write_transaction()
            except BaseException:
                await self.connection.rollback()
                raise
            if retry_refused:
                return await self._get_managed_refused_window(session_id)
            pending = metadata.get("pending_summary")
            if isinstance(pending, dict):
                messages = await self._managed_summary_rows(session_id, pending.get("message_ids", []))
                if messages and summary_window_fingerprint(messages) == pending.get("fingerprint"):
                    if time.time() < float(pending.get("retry_after", 0) or 0):
                        return 0, [], None
                    return await self._managed_summary_position(session_id, messages[0]), messages, pending
                # Rows changed or were explicitly deleted. The remaining
                # current rows can form new windows through the usual planner.
                metadata["last_stale_pending_summary_v1"] = {
                    **pending, "retry_unavailable_reason": "source_snapshot_changed",
                }
                metadata["pending_summary"] = None
                await self._save_managed_summary_metadata(session_id, metadata)
                await self._commit_write_transaction()
            async with self.connection.execute("""
                SELECT COUNT(*) FROM messages m LEFT JOIN managed_summary_progress p
                    ON p.session_id=m.session_id AND p.message_id=m.id
                WHERE m.session_id=? AND p.message_id IS NULL
            """, (session_id,)) as cursor:
                available = (await cursor.fetchone())[0]
            if available < threshold:
                return 0, [], None
            async with self.connection.execute("""
                SELECT m.*, p.state FROM messages m LEFT JOIN managed_summary_progress p
                    ON p.session_id=m.session_id AND p.message_id=m.id
                WHERE m.session_id=? AND (m.timestamp,m.id) >= (
                    SELECT n.timestamp,n.id FROM messages n LEFT JOIN managed_summary_progress q
                        ON q.session_id=n.session_id AND q.message_id=n.id
                    WHERE n.session_id=? AND q.message_id IS NULL ORDER BY n.timestamp,n.id LIMIT 1
                ) ORDER BY m.timestamp,m.id LIMIT ?
            """, (session_id, session_id, limit)) as cursor:
                rows = await cursor.fetchall()
            messages = []
            for row in rows:
                if row["state"] is not None:
                    break
                messages.append(self._row_to_message(row))
            if not messages:
                return 0, [], None
            return await self._managed_summary_position(session_id, messages[0]), messages, None

    async def _get_managed_refused_window(self, session_id):
        changed = False
        result = (0, [], None)
        async with self.connection.execute("""
            SELECT window FROM managed_summary_refusals
            WHERE session_id=? AND json_extract(window,'$.retry_unavailable_reason') IS NULL
            ORDER BY rowid LIMIT 32
        """, (session_id,)) as cursor:
            rows = await cursor.fetchall()
        try:
            for row in rows:
                window = json.loads(row["window"])
                messages = await self._managed_summary_rows(session_id, window["message_ids"])
                if messages and summary_window_fingerprint(messages) == window["fingerprint"]:
                    result = (await self._managed_summary_position(session_id, messages[0]), messages, window)
                    break
                window["retry_unavailable_reason"] = "source_snapshot_changed"
                await self.connection.execute("""
                    UPDATE managed_summary_refusals SET window=? WHERE session_id=? AND fingerprint=?
                """, (json.dumps(window), session_id, window["fingerprint"]))
                changed = True
            if changed:
                await self._commit_write_transaction()
            return result
        except BaseException:
            if changed:
                await self.connection.rollback()
            raise

    async def write_managed_summary_progress(
        self, session_id: str, messages: list[Message], pending: dict | None,
        *, retry_refused: bool = False,
    ) -> bool:
        """Persist a generation outcome by exact row snapshot, never by index alone."""
        window = self._managed_summary_window(pending or {}, messages)
        async with self._write_lock:
            try:
                session = await self.get_session(session_id)
                current_rows = await self._managed_summary_rows(session_id, window["message_ids"])
                if session is None or summary_window_fingerprint(current_rows) != window["fingerprint"]:
                    return False
                metadata = session.metadata
                fingerprint = window["fingerprint"]
                if pending is None:
                    await self._mark_managed_summary_rows(session_id, current_rows, "saved")
                    if retry_refused:
                        await self.connection.execute("""
                            DELETE FROM managed_summary_refusals WHERE session_id=? AND fingerprint=?
                        """, (session_id, fingerprint))
                elif retry_refused or pending["failure_stage"] in TERMINAL_SUMMARY_STAGES:
                    await self._save_managed_refusal(session_id, window, current_rows)
                if not retry_refused:
                    metadata["pending_summary"] = (
                        window if pending and pending["failure_stage"] not in TERMINAL_SUMMARY_STAGES else None
                    )
                await self._update_managed_summary_cursor(session_id, metadata)
                await self._save_managed_summary_metadata(session_id, metadata)
                try:
                    await self._commit_write_transaction()
                except asyncio.CancelledError as exc:
                    if pending is None:
                        anchor = messages[-1]
                        exc.summary_cursor_committed = (session_id, anchor.id, anchor.timestamp)
                    raise
                return True
            except BaseException:
                await self.connection.rollback()
                raise
