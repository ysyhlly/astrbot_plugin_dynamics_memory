"""Same-source collision journal; no semantic guessing or raw-event overwrite."""
from __future__ import annotations

import hashlib
import json
import math
import time


def event_fingerprint(event):
    # Transport retry timestamps/card revisions can change. Physical statement,
    # actor and receipt identity cannot change silently for a stable source ID.
    data = {k: event.get(k) for k in ("type", "actor_id", "text", "receipt")}
    data["text"] = str(data.get("text") or "").strip()
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def valid_event(event):
    from ..core.narrative_api import scope_key, timestamp
    if not isinstance(event, dict) or type(event.get("schema_version")) is not int or event["schema_version"] != 1:
        return False
    if event.get("type") not in {"message.received", "delivery.success"}:
        return False
    if any(not isinstance(event.get(k), str) or not event[k].strip() for k in ("event_id", "actor_id", "text")):
        return False
    if len(event["event_id"]) > 512 or len(event["text"]) > 131072:
        return False
    try:
        scope_key(event.get("scope"))
        occurred = timestamp(event.get("occurred_at"))
        event_fingerprint(event)
        return math.isfinite(occurred) and occurred >= 0
    except (TypeError, ValueError, OverflowError):
        return False


class SeriesConflictStore:
    async def initialize_series_conflicts(self):
        await self.connection.execute('''CREATE TABLE IF NOT EXISTS series_source_conflicts (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, event_id TEXT NOT NULL,
            original_hash TEXT NOT NULL, incoming_hash TEXT NOT NULL, incoming_envelope TEXT NOT NULL,
            created_at REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending', revision INTEGER NOT NULL DEFAULT 0,
            reviewer TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '')''')
        await self.connection.execute('CREATE INDEX IF NOT EXISTS idx_series_conflicts_status ON series_source_conflicts(status,created_at)')

    async def record_series_collision(self, event_key, incoming):
        """Called within the original ingest transaction/write lock."""
        async with self.connection.execute('SELECT session_id,envelope FROM narrative_events WHERE event_key=?', (event_key,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return
        original = json.loads(row['envelope'])
        old_hash, new_hash = event_fingerprint(original), event_fingerprint(incoming)
        if old_hash == new_hash:
            return
        identifier = hashlib.sha256((event_key + ':' + new_hash).encode()).hexdigest()
        await self.connection.execute('''INSERT OR IGNORE INTO series_source_conflicts
            (id,session_id,event_id,original_hash,incoming_hash,incoming_envelope,created_at)
            VALUES (?,?,?,?,?,?,?)''', (identifier, row['session_id'], incoming['event_id'], old_hash,
                                      new_hash, json.dumps(incoming, ensure_ascii=False), time.time()))

    async def list_series_conflicts(self, limit=100):
        limit = max(1, min(int(limit), 100))
        async with self.connection.execute('''SELECT id,session_id,event_id,original_hash,incoming_hash,
                created_at,status,revision,reviewer,reason FROM series_source_conflicts
                ORDER BY created_at DESC LIMIT ?''', (limit,)) as cursor:
            rows = [dict(row) for row in await cursor.fetchall()]
        return {'items': rows, 'raw_source_returned': False}

    async def resolve_series_conflict(self, identifier, revision, reason, reviewer):
        if (not isinstance(identifier, str) or type(revision) is not int or revision < 0
                or not isinstance(reason, str) or not reason.strip() or len(reason) > 1000
                or not isinstance(reviewer, str) or not reviewer):
            raise ValueError('invalid review')
        async with self._write_lock:
            try:
                cur = await self.connection.execute('''UPDATE series_source_conflicts SET status='reviewed',revision=revision+1,reason=?,reviewer=?
                    WHERE id=? AND revision=? AND status='pending' ''', (reason, reviewer, identifier, revision))
                if cur.rowcount != 1:
                    raise ValueError('conflict missing or revision changed')
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise
        return {'id': identifier, 'revision': revision + 1, 'status': 'reviewed', 'source_unchanged': True}

    async def get_series_conflict(self, identifier):
        if not isinstance(identifier, str) or len(identifier) != 64:
            raise ValueError('invalid conflict id')
        async with self.connection.execute('''SELECT c.id,c.revision,c.status,c.incoming_envelope,n.envelope
            FROM series_source_conflicts c LEFT JOIN narrative_events n
            ON n.session_id=c.session_id AND n.event_id=c.event_id WHERE c.id=?''', (identifier,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise ValueError('conflict not found')
        def readable(raw):
            data = json.loads(raw) if raw else {}
            return {k: data.get(k) for k in ('scope', 'actor_id', 'text', 'occurred_at', 'type')}
        return {'id': row['id'], 'revision': row['revision'], 'status': row['status'],
                'original': readable(row['envelope']), 'incoming': readable(row['incoming_envelope'])}
