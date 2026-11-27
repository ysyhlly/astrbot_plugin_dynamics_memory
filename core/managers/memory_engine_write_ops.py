"""
MemoryEngine 的 MemoryEngineWriteOpsMixin 拆分模块
自动从 core/managers/memory_engine.py 拆分，保持行为不变
"""

import asyncio
import json
import time
from typing import Any

import aiosqlite
from astrbot.api import logger

from ...storage.atom_store import AtomStore
from ...storage.schema import WRITE_OPS_SCHEMA_STATEMENTS
from ..models.memory_atom import AtomStatus, AtomType, DecayType, MemoryAtom
from ..utils.json_utils import safe_json_dict


class MemoryEngineWriteOpsMixin:
    """MemoryEngine 拆分模块：MemoryEngineWriteOpsMixin

    下述类级注解声明本模块依赖的宿主共享状态（由 MemoryEngine.__init__ 赋值），
    仅作类型约束，不在此赋默认值。
    """

    # 宿主共享状态契约
    db_connection: aiosqlite.Connection | None
    config: dict[str, Any]
    faiss_db: Any
    atom_enabled: bool
    atom_store: AtomStore | None
    hybrid_retriever: Any | None
    dual_route_retriever: Any | None
    graph_memory_manager: Any | None
    _write_op_max_retries: int

    def _write_db(self):
        return self._write_transaction()

    async def repair_retained_sources(self, limit=50, after=0):
        """Recover original source bytes from this plugin's own durable event archive."""
        loader = getattr(self, "source_loader", None)
        if not callable(loader) or self.db_connection is None:
            return {"repaired": 0, "available": False}
        cursor = await self.db_connection.execute(
            "SELECT d.id,d.metadata FROM documents d LEFT JOIN memory_sources s ON s.memory_id=d.id "
            "WHERE d.id>? AND s.memory_id IS NULL AND json_valid(d.metadata) AND json_array_length(json_extract(d.metadata,'$.narrative_provenance'))>0 "
            "ORDER BY d.id LIMIT ?",
            (max(0, int(after)), max(1, min(int(limit), 200))),
        )
        rows = await cursor.fetchall()
        repaired = 0
        unavailable = 0
        for row in rows:
            metadata = json.loads(row["metadata"])
            try:
                source = await loader(metadata)
            except (ValueError, TypeError, KeyError):
                source = None
            if not source:
                unavailable += 1
                continue
            async with self._write_db() as db:
                current = await (
                    await db.execute(
                        "SELECT metadata FROM documents WHERE id=?", (row["id"],)
                    )
                ).fetchone()
                if current is None:
                    continue
                current_metadata = safe_json_dict(current[0])
                if current_metadata.get("narrative_provenance") != metadata.get(
                    "narrative_provenance"
                ):
                    unavailable += 1
                    continue
                current_metadata.update(
                    has_source=True, source_message_count=len(source)
                )
                await db.execute(
                    "INSERT OR IGNORE INTO memory_sources(memory_id,source_json,created_at,updated_at) VALUES(?,?,?,?)",
                    (
                        row["id"],
                        json.dumps(source, ensure_ascii=False),
                        time.time(),
                        time.time(),
                    ),
                )
                await db.execute(
                    "UPDATE documents SET metadata=? WHERE id=?",
                    (json.dumps(current_metadata, ensure_ascii=False), row["id"]),
                )
            repaired += 1
        if repaired:
            self._invalidate_search_cache()
        return {
            "repaired": repaired,
            "source_unavailable": unavailable,
            "scanned": len(rows),
            "last_id": rows[-1]["id"] if rows else after,
            "available": True,
        }

    async def requeue_sqlite_failures(self):
        async with self._write_db() as db:
            cursor = await db.execute(
                "UPDATE memory_write_ops SET status='needs_repair',retry_count=0,updated_at=? "
                "WHERE status='failed' AND lower(error) LIKE '%database is locked%'",
                (time.time(),),
            )
        return int(cursor.rowcount or 0)

    async def stored_summary_window(
        self, session_id, start, end, *, snapshot_fingerprint=None
    ):
        if self.db_connection is None:
            return None
        window_predicate = (
            "(json_extract(d.metadata,'$.source_window.snapshot_fingerprint')=? OR EXISTS ("
            "SELECT 1 FROM json_each(d.metadata,'$.source_windows') w "
            "WHERE json_extract(w.value,'$.snapshot_fingerprint')=?)) "
            if snapshot_fingerprint
            else "json_extract(d.metadata,'$.source_window.start_index')=? AND json_extract(d.metadata,'$.source_window.end_index')=? "
        )
        window_params = (
            (snapshot_fingerprint, snapshot_fingerprint)
            if snapshot_fingerprint
            else (start, end)
        )
        cursor = await self.db_connection.execute(
            "SELECT d.id FROM documents d LEFT JOIN memory_sources s ON s.memory_id=d.id "
            "WHERE json_valid(d.metadata) AND json_extract(d.metadata,'$.source_session_id')=? "
            "AND "
            + window_predicate
            + "AND COALESCE(json_extract(d.metadata,'$.status'),'active')='active' "
            "AND json_extract(d.metadata,'$.superseded_by') IS NULL "
            "AND (COALESCE(json_extract(d.metadata,'$.has_source'),0)=0 OR s.memory_id IS NOT NULL) ORDER BY d.id LIMIT 2",
            (session_id, *window_params),
        )
        rows = await cursor.fetchall()
        return rows[0]["id"] if len(rows) == 1 else None

    async def _create_write_ops_table(self, connection=None) -> None:
        """Create the resumable write-operation log."""
        connection = connection if connection is not None else self.db_connection
        if connection is None:
            return
        for statement in WRITE_OPS_SCHEMA_STATEMENTS:
            await connection.execute(statement)
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_summary_snapshot_v1 ON documents ("
            "json_extract(metadata,'$.source_session_id'), "
            "json_extract(metadata,'$.source_window.snapshot_fingerprint')) WHERE json_valid(metadata)"
        )

    async def _start_write_op(
        self,
        op_type: str,
        payload: dict[str, Any] | None = None,
        memory_id: int | None = None,
    ) -> int | None:
        """Record the beginning of a multi-store write operation."""
        if self.db_connection is None:
            return None
        now = time.time()
        try:
            async with self._write_db() as db:
                cursor = await db.execute(
                    """
                    INSERT INTO memory_write_ops(
                        op_type, memory_id, status, step, payload,
                        created_at, updated_at
                    ) VALUES (?, ?, 'pending', 'started', ?, ?, ?)
                    """,
                    (
                        op_type,
                        memory_id,
                        json.dumps(payload or {}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                return int(cursor.lastrowid)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "[MemoryEngine] 写操作日志创建失败，未开始修改记忆 sqlite_code=%s sqlite_name=%s",
                getattr(exc, "sqlite_errorcode", None),
                getattr(exc, "sqlite_errorname", None),
                exc_info=True,
            )
            raise

    async def _advance_write_op(
        self,
        op_id: int | None,
        step: str,
        *,
        status: str = "pending",
        memory_id: int | None = None,
        error: str | None = None,
        payload_patch: dict[str, Any] | None = None,
    ) -> None:
        """Advance a write-operation log entry."""
        if op_id is None or self.db_connection is None:
            return

        try:
            async with self._write_db() as db:
                if status == "completed":
                    error = None
                current_payload: dict[str, Any] = {}
                if payload_patch:
                    cursor = await db.execute(
                        "SELECT payload FROM memory_write_ops WHERE id = ?",
                        (op_id,),
                    )
                    row = await cursor.fetchone()
                    if row and row[0]:
                        try:
                            loaded = json.loads(row[0])
                            current_payload = loaded if isinstance(loaded, dict) else {}
                        except (json.JSONDecodeError, TypeError):
                            current_payload = {}
                    current_payload.update(payload_patch)

                fields = ["step = ?", "updated_at = ?"]
                params: list[Any] = [step, time.time()]
                if memory_id is not None:
                    fields.append("memory_id = ?")
                    params.append(memory_id)
                if error is not None:
                    fields.append("error = ?")
                    params.append(error[:1000])
                    fields.append("retry_count = retry_count + 1")
                    # 达到重试上限后转为终态 failed，避免待修复记录永久滞留。
                    fields.append(
                        "status = CASE WHEN retry_count + 1 >= ? THEN 'failed' ELSE ? END"
                    )
                    params.append(self._write_op_max_retries)
                    params.append(status)
                else:
                    fields.append("status = ?")
                    params.append(status)
                    if status == "completed":
                        fields.append("error = NULL")
                if payload_patch:
                    fields.append("payload = ?")
                    params.append(json.dumps(current_payload, ensure_ascii=False))
                params.append(op_id)
                await db.execute(
                    f"UPDATE memory_write_ops SET {', '.join(fields)} WHERE id = ?",
                    params,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "[MemoryEngine] 写操作日志更新失败 sqlite_code=%s sqlite_name=%s",
                getattr(exc, "sqlite_errorcode", None),
                getattr(exc, "sqlite_errorname", None),
                exc_info=True,
            )
            raise

    def _serialize_atom_for_repair(self, atom: Any) -> dict[str, Any]:
        """Convert a MemoryAtom-like object into JSON-safe repair payload."""
        atom_type = getattr(atom, "atom_type", AtomType.UNKNOWN)
        decay_type = getattr(atom, "decay_type", DecayType.EXPONENTIAL)
        status = getattr(atom, "status", AtomStatus.ACTIVE)
        return {
            "parent_memory_id": int(getattr(atom, "parent_memory_id", 0) or 0),
            "atom_type": getattr(atom_type, "value", str(atom_type)),
            "content": str(getattr(atom, "content", "")),
            "entities": list(getattr(atom, "entities", []) or []),
            "importance": float(getattr(atom, "importance", 0.5) or 0.5),
            "confidence": float(getattr(atom, "confidence", 0.7) or 0.7),
            "created_at": float(
                getattr(atom, "created_at", time.time()) or time.time()
            ),
            "last_accessed_at": float(
                getattr(atom, "last_accessed_at", time.time()) or time.time()
            ),
            "last_reinforced_at": getattr(atom, "last_reinforced_at", None),
            "event_time": getattr(atom, "event_time", None),
            "ttl_days": float(getattr(atom, "ttl_days", 30.0) or 30.0),
            "expires_at": float(getattr(atom, "expires_at", 0.0) or 0.0),
            "status": getattr(status, "value", str(status)),
            "reinforcement_count": int(getattr(atom, "reinforcement_count", 0) or 0),
            "decay_type": getattr(decay_type, "value", str(decay_type)),
            "session_id": getattr(atom, "session_id", None),
            "persona_id": getattr(atom, "persona_id", None),
            "metadata": dict(getattr(atom, "metadata", {}) or {}),
        }

    def _deserialize_atom_from_repair(
        self,
        payload: dict[str, Any],
        parent_memory_id: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> MemoryAtom | None:
        """Rebuild a MemoryAtom from repair payload."""
        content = str(payload.get("content") or "")
        if not content.strip():
            return None

        try:
            atom_type = AtomType(payload.get("atom_type") or AtomType.UNKNOWN.value)
        except ValueError:
            atom_type = AtomType.UNKNOWN
        try:
            decay_type = DecayType(
                payload.get("decay_type") or DecayType.EXPONENTIAL.value
            )
        except ValueError:
            decay_type = DecayType.EXPONENTIAL
        try:
            status = AtomStatus(payload.get("status") or AtomStatus.ACTIVE.value)
        except ValueError:
            status = AtomStatus.ACTIVE

        return MemoryAtom(
            parent_memory_id=parent_memory_id,
            atom_type=atom_type,
            content=content,
            entities=[str(item) for item in payload.get("entities", []) if item],
            importance=float(payload.get("importance", 0.5) or 0.5),
            confidence=float(payload.get("confidence", 0.7) or 0.7),
            created_at=float(payload.get("created_at", time.time()) or time.time()),
            last_accessed_at=float(
                payload.get("last_accessed_at", time.time()) or time.time()
            ),
            last_reinforced_at=payload.get("last_reinforced_at"),
            event_time=payload.get("event_time"),
            ttl_days=float(payload.get("ttl_days", 30.0) or 30.0),
            expires_at=float(payload.get("expires_at", 0.0) or 0.0),
            status=status,
            reinforcement_count=int(payload.get("reinforcement_count", 0) or 0),
            decay_type=decay_type,
            session_id=payload.get("session_id") or session_id,
            persona_id=payload.get("persona_id") or persona_id,
            metadata=dict(payload.get("metadata") or {}),
        )

    async def _repair_incomplete_write_ops(self) -> int:
        """Best-effort replay for incomplete add/delete operations."""
        if self.db_connection is None:
            return 0

        try:
            cursor = await self.db_connection.execute(
                """
                SELECT id, op_type, memory_id, status, step, payload, retry_count
                FROM memory_write_ops
                WHERE (status IN ('pending', 'needs_repair')
                    OR (status = 'completed' AND step = 'source_published_graph_pending'))
                  AND retry_count < ?
                ORDER BY updated_at ASC, id ASC
                LIMIT 25
                """,
                (self._write_op_max_retries,),
            )
            rows = await cursor.fetchall()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("[MemoryEngine] 读取待修复写操作失败", exc_info=True)
            return 0

        repaired = 0
        for row in rows:
            payload = safe_json_dict(row["payload"])
            try:
                op_type = row["op_type"]
                memory_id = row["memory_id"]
                if op_type == "add":
                    ok = await self._repair_add_write_op(
                        int(row["id"]),
                        int(memory_id) if memory_id is not None else None,
                        payload,
                    )
                elif op_type == "delete":
                    ok = await self._repair_delete_write_op(
                        int(row["id"]),
                        int(memory_id) if memory_id is not None else None,
                    )
                elif op_type == "batch_delete":
                    ok = await self._repair_batch_delete_write_op(
                        int(row["id"]),
                        payload,
                    )
                else:
                    ok = False
                repaired += 1 if ok else 0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    f"[MemoryEngine] 修复写操作失败 (op_id={row['id']})",
                    exc_info=True,
                )
                if row["step"] == "source_published_graph_pending":
                    await self._record_published_graph_error(int(row["id"]), str(e))
                else:
                    await self._advance_write_op(
                        int(row["id"]),
                        str(row["step"] or "repair_failed"),
                        status="needs_repair",
                        error=str(e),
                    )

        if repaired:
            logger.info(f"[MemoryEngine] 已修复 {repaired} 个未完成写操作")
            self._invalidate_search_cache()
        return repaired

    async def _repair_add_write_op(
        self,
        op_id: int,
        memory_id: int | None,
        payload: dict[str, Any],
    ) -> bool:
        if payload.get("source_validation_required") is True:
            if memory_id is None:
                # The hidden parent can commit before the journal records its ID.
                # Match only an unpublished row with this operation's own integer ID.
                async with self.db_connection.execute(
                    """
                    SELECT id FROM documents
                    WHERE CASE WHEN json_valid(metadata) THEN
                        json_type(metadata, '$.write_op_id') = 'integer'
                        AND json_extract(metadata, '$.write_op_id') = ?
                        AND json_extract(metadata, '$.status') = 'pending_source_validation'
                        AND json_type(metadata, '$.source_publication_verified_v1') IS NULL
                    END LIMIT 2
                    """,
                    (op_id,),
                ) as cursor:
                    candidates = await cursor.fetchall()
                if len(candidates) > 1:
                    await self._advance_write_op(
                        op_id,
                        "source_document_operation_ambiguous",
                        status="failed",
                        error="multiple unpublished documents reference this operation",
                    )
                    return False
                if candidates:
                    memory_id = int(candidates[0]["id"])
            if memory_id is not None:
                # Read the canonical document directly. A retrieval miss or
                # transient database error must not be treated as proof that
                # source publication never committed.
                async with self.db_connection.execute(
                    "SELECT text, metadata FROM documents WHERE id = ?", (memory_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                if row is not None:
                    metadata = safe_json_dict(row["metadata"])
                    write_op_id = metadata.get("write_op_id")
                    if write_op_id is not None and (
                        type(write_op_id) is not int or write_op_id != op_id
                    ):
                        await self._advance_write_op(
                            op_id,
                            "source_document_operation_mismatch",
                            status="failed",
                            memory_id=memory_id,
                            error="source document belongs to another write operation",
                        )
                        return False
                    witness = metadata.get("source_publication_verified_v1")
                    if (
                        type(write_op_id) is int
                        and write_op_id == op_id
                        and isinstance(witness, dict)
                        and type(witness.get("write_op_id")) is int
                        and witness["write_op_id"] == op_id
                        and type(witness.get("memory_id")) is int
                        and witness["memory_id"] == memory_id
                        and metadata.get("status") in {"active", "archived"}
                    ):
                        return await self._repair_published_source_write(
                            op_id, memory_id, row["text"], metadata
                        )

            # Only unpublished candidates are discarded. A fresh runtime
            # lease cannot be recreated from a serialized source payload.
            if memory_id is not None and self.hybrid_retriever is not None:
                try:
                    await self.hybrid_retriever.update_metadata(
                        int(memory_id), {"status": "pending_source_validation"}
                    )
                except asyncio.CancelledError:
                    try:
                        await asyncio.shield(self.delete_memory(int(memory_id)))
                    except BaseException:
                        pass
                    raise
                except Exception:
                    logger.info(
                        "[MemoryEngine] route=guarded_recovery step=mark_pending outcome=failed"
                    )
                if not await self.delete_memory(memory_id):
                    await self._advance_write_op(
                        op_id,
                        "source_cleanup_pending",
                        status="needs_repair",
                        memory_id=memory_id,
                        error="unpublished source document deletion failed",
                    )
                    return False
            await self._advance_write_op(
                op_id,
                "source_lease_unavailable_after_restart",
                status="failed",
                memory_id=memory_id,
                error="fresh source publication lease is unavailable after restart",
            )
            return False
        if memory_id is None:
            await self._advance_write_op(
                op_id,
                "unrepairable",
                status="failed",
                error="missing memory_id for add repair",
            )
            return False

        memory = await self.get_memory(int(memory_id))
        if memory is None:
            await self._advance_write_op(
                op_id,
                "source_missing",
                status="failed",
                memory_id=int(memory_id),
                error="source document missing",
            )
            return False

        metadata = memory.get("metadata") or payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = safe_json_dict(metadata)
        if metadata.get("has_source") and not await self.get_memory_source(
            int(memory_id)
        ):
            retained_sources = payload.get("source_messages")
            loader = getattr(self, "source_loader", None)
            if not retained_sources and callable(loader):
                retained_sources = await loader(metadata)
            if not isinstance(retained_sources, list) or not retained_sources:
                await self._advance_write_op(
                    op_id,
                    "source_pending",
                    status="needs_repair",
                    memory_id=int(memory_id),
                )
                return False
            await self.save_memory_source(int(memory_id), retained_sources)
            await self._advance_write_op(
                op_id, "source_repaired", memory_id=int(memory_id)
            )
        content = str(memory.get("text") or "")
        session_id = metadata.get("session_id") or payload.get("session_id")
        persona_id = metadata.get("persona_id") or payload.get("persona_id")

        atom_payloads = payload.get("failed_atoms") or payload.get("atoms", []) or []
        atoms: list[MemoryAtom] = []
        for atom_payload in atom_payloads:
            if isinstance(atom_payload, dict):
                atom = self._deserialize_atom_from_repair(
                    atom_payload,
                    int(memory_id),
                    session_id,
                    persona_id,
                )
                if atom is not None:
                    atoms.append(atom)

        if self.atom_store is not None and atoms and self.atom_enabled:
            existing_atoms = await self.atom_store.get_by_parent(int(memory_id))
            if payload.get("failed_atoms"):
                existing_keys = {
                    (
                        atom.content,
                        atom.atom_type.value,
                        atom.session_id,
                        atom.persona_id,
                    )
                    for atom in existing_atoms
                }
                atoms_to_insert = [
                    atom
                    for atom in atoms
                    if (
                        atom.content,
                        atom.atom_type.value,
                        atom.session_id,
                        atom.persona_id,
                    )
                    not in existing_keys
                ]
                if atoms_to_insert:
                    await self.atom_store.insert_many(atoms_to_insert)
            elif not existing_atoms:
                await self.atom_store.insert_many(atoms)
            await self._advance_write_op(op_id, "atoms_repaired", memory_id=memory_id)

        if self.graph_memory_manager is not None and content.strip():
            await self.graph_memory_manager.index_memory(
                int(memory_id),
                content,
                metadata,
                atoms or None,
            )
            await self._advance_write_op(op_id, "graph_repaired", memory_id=memory_id)

        await self._advance_write_op(
            op_id,
            "completed",
            status="completed",
            memory_id=int(memory_id),
        )
        return True

    async def _repair_published_source_write(
        self, op_id: int, memory_id: int, content: str, metadata: dict[str, Any]
    ) -> bool:
        # Source text and atoms were saved before publication. An archived
        # record must stay archived and never reenter the graph.
        if (
            self.graph_memory_manager is not None
            and metadata.get("status") == "active"
            and not metadata.get("superseded_by")
            and str(content or "").strip()
        ):
            atoms = (
                await self.atom_store.get_by_parent(memory_id)
                if self.atom_store is not None and self.atom_enabled
                else None
            )
            try:
                indexed = await self.graph_memory_manager.index_memory(
                    memory_id,
                    content,
                    metadata,
                    atoms or None,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_published_graph_error(op_id, str(exc))
                return False
            if indexed is False:
                await self._record_published_graph_error(op_id, "graph repair deferred")
                return False
        await self._advance_write_op(
            op_id,
            "completed",
            status="completed",
            memory_id=memory_id,
        )
        return True

    async def _record_published_graph_error(self, op_id: int, error: str) -> None:
        # Keep this source publication terminal for older code's startup repair.
        async with self._write_db() as db:
            await db.execute(
                "UPDATE memory_write_ops SET status='completed', step='source_published_graph_pending', "
                "error=?, retry_count=retry_count+1, updated_at=? WHERE id=?",
                (error[:1000], time.time(), op_id),
            )

    async def _repair_delete_write_op(
        self,
        op_id: int,
        memory_id: int | None,
    ) -> bool:
        if memory_id is None:
            await self._advance_write_op(
                op_id,
                "unrepairable",
                status="failed",
                error="missing memory_id for delete repair",
            )
            return False

        if self.db_connection is not None:
            # Replay the document/vector/BM25 delete first, mirroring the normal
            # delete flow. If the process crashed before or during that step, the
            # document must be removed here too; otherwise the memory stays
            # searchable while its derived graph/atom/source data is already gone.
            cursor = await self.db_connection.execute(
                "SELECT 1 FROM documents WHERE id = ?", (int(memory_id),)
            )
            if (
                await cursor.fetchone() is not None
                and self.hybrid_retriever is not None
            ):
                await self.hybrid_retriever.delete_memory(int(memory_id))

            async with self._write_db() as db:
                await db.execute(
                    "DELETE FROM memory_sources WHERE memory_id = ?", (int(memory_id),)
                )
        if self.graph_memory_manager is not None:
            await self.graph_memory_manager.delete_memory(int(memory_id))
        if self.atom_store is not None:
            await self.atom_store.delete_by_parent(int(memory_id))

        await self._advance_write_op(
            op_id,
            "completed",
            status="completed",
            memory_id=int(memory_id),
        )
        return True

    async def _repair_batch_delete_write_op(
        self,
        op_id: int,
        payload: dict[str, Any],
    ) -> bool:
        memory_ids_raw = payload.get("memory_ids") or []
        if not isinstance(memory_ids_raw, list):
            await self._advance_write_op(
                op_id,
                "unrepairable",
                status="failed",
                error="missing memory_ids for batch delete repair",
            )
            return False

        memory_ids: list[int] = []
        for raw_id in memory_ids_raw:
            try:
                memory_ids.append(int(raw_id))
            except (TypeError, ValueError):
                continue

        if not memory_ids:
            await self._advance_write_op(
                op_id,
                "unrepairable",
                status="failed",
                error="empty memory_ids for batch delete repair",
            )
            return False

        await self._delete_document_indexes_for_batch(memory_ids)
        await self._delete_graph_and_atoms_for_batch(memory_ids)
        await self._advance_write_op(
            op_id,
            "completed",
            status="completed",
            payload_patch={"deleted_count": len(memory_ids)},
        )
        return True

    async def _delete_document_indexes_for_batch(self, memory_ids: list[int]) -> int:
        if not memory_ids or self.db_connection is None:
            return 0
        placeholders = ",".join("?" * len(memory_ids))
        cursor = await self.db_connection.execute(
            f"SELECT id, doc_id FROM documents WHERE id IN ({placeholders})", memory_ids
        )
        rows = await cursor.fetchall()
        async with self._write_db() as db:
            await db.execute(
                f"DELETE FROM livingmemory_memories_fts WHERE doc_id IN ({placeholders})",
                memory_ids,
            )
        # The SQLite transaction is finished before the SDK opens its writer.
        # A failed vector delete must retain the repair operation, not claim success.
        for row in rows:
            if row["doc_id"]:
                deleted = await self.faiss_db.delete(row["doc_id"])
                if deleted is False:
                    raise RuntimeError("Vector deletion was not confirmed")
        async with self._write_db() as db:
            await db.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})", memory_ids
            )
            await db.execute(
                f"DELETE FROM memory_sources WHERE memory_id IN ({placeholders})",
                memory_ids,
            )
        return len(rows)

    async def cleanup_write_ops(
        self, retention_days: int = 14, *, payload_retention_days: int = 3
    ) -> int:
        """Prune aged rows and strip bloated audit payloads from ``memory_write_ops``."""
        if self.db_connection is None:
            return 0
        now_ts = time.time()
        cutoff = now_ts - max(1, int(retention_days)) * 86400
        payload_cutoff = now_ts - max(1, int(payload_retention_days)) * 86400
        stale_cutoff = now_ts - max(30, int(retention_days) * 2) * 86400
        async with self._write_db() as db:
            table_check = await db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_write_ops' LIMIT 1"
            )
            exists = await table_check.fetchone() is not None
            await table_check.close()
            if not exists:
                return 0
            # 1. Strip bulky JSON payloads from older completed ops while keeping audit row metadata.
            await db.execute(
                "UPDATE memory_write_ops SET payload = '{}' "
                "WHERE status = 'completed' AND updated_at < ? AND length(payload) > 256",
                (payload_cutoff,),
            )
            # 2. Delete expired terminal rows.
            cursor = await db.execute(
                "DELETE FROM memory_write_ops WHERE status IN ('completed', 'failed') AND updated_at < ?",
                (cutoff,),
            )
            deleted = max(0, int(cursor.rowcount or 0))
            # 3. Delete long-abandoned pending/needs_repair rows.
            stale_cur = await db.execute(
                "DELETE FROM memory_write_ops WHERE status IN ('pending', 'needs_repair') AND updated_at < ?",
                (stale_cutoff,),
            )
            deleted += max(0, int(stale_cur.rowcount or 0))
            return deleted
