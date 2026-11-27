"""B-side memory health checkup, smart dedup/contradiction governance, and member correction review API."""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any

from quart import request

from astrbot.api import logger

from ..memory_scope import adapter_sql

if TYPE_CHECKING:
    from .utils import PageApiUtils


_TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[\u4e00-\u9fff]")
_POS_MARKERS = (
    "喜欢",
    "爱吃",
    "最爱",
    "偏好",
    "支持",
    "习惯",
    "想要",
    "觉得不错",
    "很满意",
)
_NEG_MARKERS = (
    "讨厌",
    "不喜欢",
    "不吃",
    "不爱",
    "反感",
    "反对",
    "忌口",
    "过敏",
    "不想",
    "讨厌吃",
    "不喝",
)
_STOP_TERMS = frozenset(
    {
        "喜欢",
        "讨厌",
        "不喜欢",
        "觉得",
        "今天",
        "昨天",
        "最近",
        "大家",
        "群友",
        "聊天",
        "讨论",
        "表示",
        "认为",
        "这个",
        "那个",
        "我们",
        "他们",
        "自己",
        "可以",
        "比较",
    }
)


def _parse_meta(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_importance(val: Any) -> float:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return 0.5
    if num > 1.0:
        num = num / 10.0
    return max(0.0, min(1.0, num))


def _ngrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", "", str(text or "").lower())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _extract_terms(text: str, metadata: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    topics = metadata.get("topics")
    if isinstance(topics, list):
        for t in topics:
            s = str(t or "").strip().lower()
            if len(s) >= 2 and s not in _STOP_TERMS:
                terms.add(s)
    # Extract CJK 2-4 char chunks & alphanumeric words
    for word in re.findall(
        r"[a-z0-9]{2,}|[\u4e00-\u9fff]{2,4}", str(text or "").lower()
    ):
        if word not in _STOP_TERMS:
            terms.add(word)
    return terms


def _polarity(text: str) -> int:
    """Return +1 for positive preference, -1 for negative preference, 0 for neutral."""
    s = str(text or "")
    has_neg = any(m in s for m in _NEG_MARKERS)
    if has_neg:
        return -1
    has_pos = any(m in s for m in _POS_MARKERS)
    if has_pos:
        return 1
    return 0


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / float(len(a | b))


class GovernanceHandler:
    """Memory health checkup, dedup merge, bulk governance, and correction review."""

    def __init__(self, utils: PageApiUtils, plugin: Any) -> None:
        self.utils = utils
        self.plugin = plugin

    async def get_health_report(self, memory_engine: Any) -> dict[str, Any]:
        """Scan active memories & graph for duplicates, contradictions, dead items, and orphan nodes."""
        args = request.args
        session_id = self.utils.optional_text(args.get("session_id"))
        try:
            stale_days = max(1, min(int(args.get("stale_days", 30)), 3650))
            dup_threshold = max(0.5, min(float(args.get("dup_threshold", 0.72)), 0.99))
            limit = max(5, min(int(args.get("limit", 20)), 60))
        except (TypeError, ValueError):
            return self.utils.error("体检参数无效")

        db = getattr(memory_engine, "db_connection", None)
        if db is None:
            return self.utils.error("数据库连接未初始化")

        ownership, ownership_params = adapter_sql("metadata")
        where = [
            ownership,
            "COALESCE(json_extract(metadata, '$.status'), 'active') = 'active'",
        ]
        params: list[Any] = list(ownership_params)
        if session_id:
            where.append("json_extract(metadata, '$.session_id') = ?")
            params.append(session_id)

        try:
            cursor = await db.execute(
                f"""
                SELECT id, text, metadata
                FROM documents
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(CAST(json_extract(metadata, '$.create_time') AS REAL), 0) DESC,
                         id DESC
                LIMIT 400
                """,
                params,
            )
            rows = await cursor.fetchall()
        except Exception as exc:
            logger.error("[Governance] 查询记忆失败: %s", exc, exc_info=True)
            return self.utils.error(str(exc))

        now = time.time()
        records: list[dict[str, Any]] = []
        dead_memories: list[dict[str, Any]] = []

        for row in rows:
            doc_id = int(row["id"] if hasattr(row, "keys") else row[0])
            text = str(row["text"] if hasattr(row, "keys") else row[1] or "").strip()
            meta = _parse_meta(row["metadata"] if hasattr(row, "keys") else row[2])
            summary = str(
                meta.get("persona_summary") or meta.get("canonical_summary") or text
            ).strip()
            imp = _normalize_importance(meta.get("importance", 0.5))
            create_time = float(meta.get("create_time") or 0)
            if create_time > 100_000_000_000:
                create_time /= 1000.0
            age_days = (
                max(0.0, (now - create_time) / 86400.0) if create_time > 0 else 0.0
            )
            access_count = int(meta.get("access_count") or 0)
            scope = str(meta.get("session_id") or "")
            persona = str(meta.get("persona_id") or "")

            rec = {
                "id": doc_id,
                "summary": summary,
                "preview": summary[:140] + ("…" if len(summary) > 140 else ""),
                "importance": round(imp, 3),
                "session_id": scope,
                "persona_id": persona,
                "age_days": round(age_days, 1),
                "access_count": access_count,
                "ngrams": _ngrams(summary),
                "terms": _extract_terms(summary, meta),
                "polarity": _polarity(summary),
            }
            records.append(rec)

            if age_days >= stale_days and access_count == 0 and imp <= 0.45:
                dead_memories.append(
                    {
                        "id": doc_id,
                        "session_id": scope,
                        "importance": round(imp, 3),
                        "age_days": round(age_days, 1),
                        "access_count": access_count,
                        "preview": rec["preview"],
                    }
                )

        # Group by (session_id, persona_id) for bounded pairwise duplicate & contradiction scan
        by_scope: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for rec in records:
            by_scope.setdefault((rec["session_id"], rec["persona_id"]), []).append(rec)

        duplicates: list[dict[str, Any]] = []
        contradictions: list[dict[str, Any]] = []
        seen_dup_ids: set[int] = set()

        for group in by_scope.values():
            bounded = group[:120]
            n = len(bounded)
            for i in range(n):
                a = bounded[i]
                for j in range(i + 1, n):
                    b = bounded[j]
                    sim = _jaccard(a["ngrams"], b["ngrams"])
                    if sim >= dup_threshold and len(duplicates) < limit:
                        if a["id"] not in seen_dup_ids or b["id"] not in seen_dup_ids:
                            keep, dup = (
                                (a, b)
                                if (a["importance"], a["id"])
                                >= (b["importance"], b["id"])
                                else (b, a)
                            )
                            duplicates.append(
                                {
                                    "keep_id": keep["id"],
                                    "duplicate_id": dup["id"],
                                    "similarity": round(sim, 3),
                                    "session_id": keep["session_id"],
                                    "keep_preview": keep["preview"],
                                    "duplicate_preview": dup["preview"],
                                    "keep_importance": keep["importance"],
                                    "duplicate_importance": dup["importance"],
                                }
                            )
                            seen_dup_ids.add(dup["id"])
                            continue

                    if (
                        len(contradictions) < limit
                        and a["polarity"] != 0
                        and b["polarity"] != 0
                        and a["polarity"] == -b["polarity"]
                    ):
                        shared_terms = sorted(
                            t
                            for t in (a["terms"] & b["terms"])
                            if not any(m in t for m in _POS_MARKERS + _NEG_MARKERS)
                        )
                        if shared_terms:
                            contradictions.append(
                                {
                                    "memory_a_id": a["id"],
                                    "memory_b_id": b["id"],
                                    "session_id": a["session_id"],
                                    "shared_terms": shared_terms[:4],
                                    "preview_a": a["preview"],
                                    "preview_b": b["preview"],
                                    "reason": f"围绕「{'、'.join(shared_terms[:2])}」存在正反偏好冲突",
                                }
                            )

        duplicates.sort(key=lambda x: x["similarity"], reverse=True)
        dead_memories.sort(key=lambda x: (-x["age_days"], x["importance"]))

        orphan_nodes: list[dict[str, Any]] = []
        graph_store = self.utils.get_graph_store(memory_engine)
        if graph_store is not None and callable(
            getattr(graph_store, "list_orphan_nodes", None)
        ):
            try:
                orphan_nodes = await graph_store.list_orphan_nodes(limit=limit)
            except Exception as exc:
                logger.warning("[Governance] 读取孤立节点失败: %s", exc)

        pending_corrections = 0
        correction_store = getattr(self.plugin, "correction_store", None)
        if correction_store is not None:
            try:
                corr = await correction_store.list_proposals(status="pending", limit=1)
                pending_corrections = int(corr.get("pending_count", 0))
            except Exception:
                pending_corrections = 0

        penalty = min(
            70,
            len(duplicates) * 4
            + len(contradictions) * 6
            + min(len(dead_memories), 15) * 1
            + min(len(orphan_nodes), 15) * 1,
        )
        health_score = max(30, 100 - penalty)

        return self.utils.ok(
            {
                "summary": {
                    "scanned_memories": len(records),
                    "duplicate_count": len(duplicates),
                    "contradiction_count": len(contradictions),
                    "dead_count": len(dead_memories),
                    "orphan_node_count": len(orphan_nodes),
                    "pending_corrections": pending_corrections,
                    "health_score": health_score,
                },
                "duplicates": duplicates[:limit],
                "contradictions": contradictions[:limit],
                "dead_memories": dead_memories[:limit],
                "orphan_nodes": orphan_nodes[:limit],
            }
        )

    async def merge_duplicates(self, memory_engine: Any) -> dict[str, Any]:
        """Preview or execute merging duplicate memories into one primary record."""
        payload = await request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return self.utils.error("请求体必须是 JSON 对象")

        try:
            keep_id = int(payload.get("keep_id"))
            raw_merge_ids = payload.get("merge_ids") or []
            if not isinstance(raw_merge_ids, list) or not raw_merge_ids:
                return self.utils.error("merge_ids 不能为空")
            merge_ids = sorted({int(x) for x in raw_merge_ids if int(x) != keep_id})
        except (TypeError, ValueError):
            return self.utils.error("keep_id 或 merge_ids 必须是整数")

        if not merge_ids:
            return self.utils.error("至少需要一个不同的待合并记忆 ID")

        dry_run = bool(payload.get("dry_run", False))
        custom_content = str(payload.get("merged_content") or "").strip()

        keep_mem = await memory_engine.get_memory_by_id(keep_id)
        if not keep_mem:
            return self.utils.error(f"主记忆 #{keep_id} 不存在")

        merge_mems: list[dict[str, Any]] = []
        for mid in merge_ids:
            mem = await memory_engine.get_memory_by_id(mid)
            if mem:
                merge_mems.append(mem)
        if not merge_mems:
            return self.utils.error("未找到有效的待合并记忆")

        keep_meta = _parse_meta(keep_mem.get("metadata"))
        base_text = str(keep_mem.get("text") or "").strip()
        merged_topics = list(
            dict.fromkeys(str(t) for t in (keep_meta.get("topics") or []) if t)
        )
        merged_facts = list(
            dict.fromkeys(str(f) for f in (keep_meta.get("key_facts") or []) if f)
        )
        max_imp = _normalize_importance(keep_meta.get("importance", 0.5))

        extra_parts: list[str] = []
        for mem in merge_mems:
            m_meta = _parse_meta(mem.get("metadata"))
            max_imp = max(max_imp, _normalize_importance(m_meta.get("importance", 0.5)))
            for t in m_meta.get("topics") or []:
                s = str(t or "").strip()
                if s and s not in merged_topics:
                    merged_topics.append(s)
            for f in m_meta.get("key_facts") or []:
                s = str(f or "").strip()
                if s and s not in merged_facts:
                    merged_facts.append(s)
            m_text = str(mem.get("text") or "").strip()
            if (
                m_text
                and m_text not in base_text
                and _jaccard(_ngrams(base_text), _ngrams(m_text)) < 0.9
            ):
                extra_parts.append(m_text)

        merged_content = custom_content or (
            base_text if not extra_parts else f"{base_text}；{'；'.join(extra_parts)}"
        )
        merged_topics = merged_topics[:5]
        merged_facts = merged_facts[:5]

        preview_data = {
            "dry_run": dry_run,
            "keep_id": keep_id,
            "merge_ids": [int(m["id"]) for m in merge_mems],
            "merged_content": merged_content,
            "merged_topics": merged_topics,
            "merged_key_facts": merged_facts,
            "importance": round(max_imp, 3),
        }
        if dry_run:
            return self.utils.ok(preview_data)

        history = list(keep_meta.get("update_history") or [])
        history.append(
            {
                "timestamp": time.time(),
                "source": "governance_merge",
                "merged_ids": [int(m["id"]) for m in merge_mems],
            }
        )

        try:
            await memory_engine.update_memory(
                keep_id,
                content=merged_content,
                metadata_updates={
                    "topics": merged_topics,
                    "key_facts": merged_facts,
                    "importance": round(max_imp, 3),
                    "persona_summary": merged_content,
                    "update_history": history[-20:],
                },
            )
            archive_ids = [int(m["id"]) for m in merge_mems]
            archive_fn = getattr(memory_engine, "archive_memories", None)
            if callable(archive_fn):
                await archive_fn(archive_ids)
            else:
                for aid in archive_ids:
                    await memory_engine.update_memory(
                        aid, metadata_updates={"status": "archived"}
                    )
        except Exception as exc:
            logger.error("[Governance] 合并重复记忆失败: %s", exc, exc_info=True)
            return self.utils.error(f"合并失败: {exc}")

        return self.utils.ok({**preview_data, "archived_ids": archive_ids})

    async def bulk_action(self, memory_engine: Any) -> dict[str, Any]:
        """Batch archive or downweight low-value/dead memories."""
        payload = await request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return self.utils.error("请求体必须是 JSON 对象")

        action = str(payload.get("action") or "").strip().lower()
        if action not in {"archive", "downweight"}:
            return self.utils.error("action 必须为 archive 或 downweight")

        raw_ids = payload.get("ids") or []
        if not isinstance(raw_ids, list) or not raw_ids:
            return self.utils.error("ids 不能为空")
        try:
            ids = sorted({int(x) for x in raw_ids})[:100]
            factor = max(0.1, min(float(payload.get("factor", 0.5)), 0.95))
            floor = max(0.05, min(float(payload.get("floor", 0.1)), 0.5))
        except (TypeError, ValueError):
            return self.utils.error("参数格式无效")

        succeeded: list[int] = []
        failed: list[dict[str, Any]] = []

        if action == "archive":
            archive_fn = getattr(memory_engine, "archive_memories", None)
            if callable(archive_fn):
                try:
                    await archive_fn(ids)
                    succeeded = list(ids)
                except Exception as exc:
                    return self.utils.error(f"批量归档失败: {exc}")
            else:
                for mid in ids:
                    try:
                        ok = await memory_engine.update_memory(
                            mid, metadata_updates={"status": "archived"}
                        )
                        if ok:
                            succeeded.append(mid)
                        else:
                            failed.append({"id": mid, "error": "not_found"})
                    except Exception as exc:
                        failed.append({"id": mid, "error": str(exc)})
        else:
            for mid in ids:
                try:
                    mem = await memory_engine.get_memory_by_id(mid)
                    if not mem:
                        failed.append({"id": mid, "error": "not_found"})
                        continue
                    meta = _parse_meta(mem.get("metadata"))
                    curr = _normalize_importance(meta.get("importance", 0.5))
                    next_imp = round(max(floor, curr * factor), 3)
                    await memory_engine.update_memory(
                        mid, metadata_updates={"importance": next_imp}
                    )
                    succeeded.append(mid)
                except Exception as exc:
                    failed.append({"id": mid, "error": str(exc)})

        return self.utils.ok(
            {
                "action": action,
                "succeeded": succeeded,
                "failed": failed,
                "count": len(succeeded),
            }
        )

    async def list_corrections(self) -> dict[str, Any]:
        """List member-submitted memory correction proposals."""
        store = getattr(self.plugin, "correction_store", None)
        if store is None:
            return self.utils.ok({"items": [], "pending_count": 0, "total": 0})
        status = str(request.args.get("status", "all") or "all").strip().lower()
        try:
            limit = max(1, min(int(request.args.get("limit", 50)), 200))
        except (TypeError, ValueError):
            limit = 50
        try:
            data = await store.list_proposals(status=status, limit=limit)
            return self.utils.ok(data)
        except Exception as exc:
            logger.error("[Governance] 获取纠错队列失败: %s", exc, exc_info=True)
            return self.utils.error(str(exc))

    async def review_correction(self, memory_engine: Any) -> dict[str, Any]:
        """Approve (revise target memory) or reject a pending correction proposal."""
        store = getattr(self.plugin, "correction_store", None)
        if store is None:
            return self.utils.error("纠错队列未初始化")

        payload = await request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return self.utils.error("请求体必须是 JSON 对象")

        try:
            proposal_id = int(payload.get("id"))
        except (TypeError, ValueError):
            return self.utils.error("id 必须是整数")

        decision = str(payload.get("decision") or "").strip().lower()
        if decision not in {"approve", "reject"}:
            return self.utils.error("decision 必须为 approve 或 reject")
        review_note = str(payload.get("review_note") or "").strip()
        replacement = str(payload.get("replacement") or "").strip()

        proposal = await store.get_proposal(proposal_id)
        if proposal is None:
            return self.utils.error("纠错申请不存在")
        if proposal["status"] != "pending":
            return self.utils.error("该纠错申请已被处理")

        applied_text = ""
        if decision == "approve":
            memory_id = int(proposal["memory_id"])
            mem = await memory_engine.get_memory_by_id(memory_id)
            if not mem:
                return self.utils.error(f"目标记忆 #{memory_id} 不存在，无法批准修订")
            meta = _parse_meta(mem.get("metadata"))
            applied_text = (
                replacement
                or proposal.get("proposed_text")
                or proposal.get("reason")
                or ""
            )
            if not applied_text:
                return self.utils.error("修订内容不能为空")
            history = list(meta.get("update_history") or [])
            history.append(
                {
                    "timestamp": time.time(),
                    "source": "member_correction",
                    "proposal_id": proposal_id,
                    "requester_id": proposal.get("requester_id", ""),
                    "previous_preview": str(mem.get("text") or "")[:160],
                    "reason": proposal.get("reason", ""),
                }
            )
            try:
                await memory_engine.update_memory(
                    memory_id,
                    content=applied_text,
                    metadata_updates={
                        "persona_summary": applied_text,
                        "update_history": history[-20:],
                    },
                )
            except Exception as exc:
                logger.error("[Governance] 应用纠错修订失败: %s", exc, exc_info=True)
                return self.utils.error(f"修订记忆失败: {exc}")

        try:
            updated = await store.review(
                proposal_id,
                decision=decision,
                reviewer="dashboard_admin",
                review_note=review_note,
                applied_text=applied_text,
            )
        except (LookupError, RuntimeError, ValueError) as exc:
            return self.utils.error(str(exc))

        return self.utils.ok({"proposal": updated})

    async def list_milestones(self) -> dict[str, Any]:
        """List Event Bus subscription state and recorded milestone keepsakes."""
        svc = getattr(self.plugin, "event_bus_service", None)
        if svc is None:
            return self.utils.ok(
                {
                    "event_bus": {
                        "subscribed": False,
                        "topic_archived_total": 0,
                        "session_archived_count": 0,
                        "last_archived_at": None,
                    },
                    "daily_limit_per_group": 1,
                    "items": [],
                    "total": 0,
                }
            )
        umo = self.utils.optional_text(request.args.get("umo"))
        try:
            limit = max(1, min(int(request.args.get("limit", 30)), 100))
        except (TypeError, ValueError):
            limit = 30
        try:
            data = await svc.get_status_and_milestones(umo=umo, limit=limit)
            return self.utils.ok(data)
        except Exception as exc:
            logger.error("[Governance] 读取里程碑状态失败: %s", exc, exc_info=True)
            return self.utils.error(str(exc))

    async def trigger_milestone(self, memory_engine: Any) -> dict[str, Any]:
        """Manually generate a polaroid keepsake milestone from a memory ID."""
        svc = getattr(self.plugin, "event_bus_service", None)
        if svc is None:
            return self.utils.error("里程碑服务未初始化")
        payload = await request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return self.utils.error("请求体必须是 JSON 对象")
        try:
            memory_id = int(payload.get("memory_id"))
        except (TypeError, ValueError):
            return self.utils.error("memory_id 必须是整数")
        force = bool(payload.get("force", False))

        mem = await memory_engine.get_memory_by_id(memory_id)
        if not mem:
            return self.utils.error(f"记忆 #{memory_id} 不存在")
        meta = _parse_meta(mem.get("metadata"))
        umo = str(meta.get("session_id") or meta.get("source_session_id") or "").strip()
        if not umo:
            return self.utils.error("该记忆缺少所属会话 UMO，无法生成里程碑")
        res = await svc.trigger_milestone_keepsake(
            memory_id=memory_id,
            umo=umo,
            content=str(mem.get("text") or ""),
            metadata=meta,
            force=force,
        )
        if not res.get("ok"):
            reason = res.get("reason")
            msg = (
                "该群今日已达里程碑纪念品生成上限（每日 1 次），勾选强制可覆盖"
                if reason == "daily_limit_reached"
                else (
                    "该记忆已生成过里程碑纪念品"
                    if reason == "already_recorded"
                    else f"生成失败: {reason}"
                )
            )
            return self.utils.error(msg)
        return self.utils.ok(res)
