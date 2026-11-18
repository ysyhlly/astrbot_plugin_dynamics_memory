"""Small, body-aware helpers for the per-request native recall bridge."""

from __future__ import annotations

import copy
import re
from typing import Any

NATIVE_MEMORY_REQUEST_KEY = "_lm_native_memory_v1"
NATIVE_MEMORY_VALIDATE_KEY = "_lm_native_memory_validate_v1"


def request_identity(value: Any) -> Any:
    """Return a deep copy of JSON-like request data for callback identity checks."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return None


def quote_frames(bundle: Any) -> list[dict[str, Any]]:
    """Extract only authored user quotations backed by canonical sources."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        return []
    frames: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source_view in bundle.get("source_views", []):
        if not isinstance(source_view, dict):
            continue
        view = source_view.get("evidence_view")
        if not isinstance(view, dict) or view.get("available") is not True:
            continue
        materials = source_view.get("native_materials", [])
        event_proofs = source_view.get("native_events", [])
        material_rows = {
            str(row.get("material_id")): row
            for row in materials
            if isinstance(row, dict) and isinstance(row.get("material_id"), str)
        }
        material_ids = set(material_rows)
        scope = source_view.get("scope")
        scope_id = str(scope.get("umo", "")) if isinstance(scope, dict) else ""
        for atom in view.get("atoms", []):
            if not isinstance(atom, dict):
                continue
            if (
                atom.get("kind") != "utterance_fact"
                or atom.get("status") != "verified_utterance"
            ):
                continue
            claim = atom.get("claim")
            if not isinstance(claim, str) or not claim.strip() or len(claim) > 500:
                continue
            refs = atom.get("source_refs", [])
            valid_refs = []
            for ref in refs if isinstance(refs, list) else []:
                if not isinstance(ref, dict) or ref.get("author_kind") != "user":
                    continue
                source_kind = ref.get("source_kind", ref.get("source_status"))
                valid_ref = (
                    source_kind == "conversation_row"
                    and isinstance(ref.get("material_id"), str)
                    and ref["material_id"] in material_ids
                    and isinstance(ref.get("row_hash"), str)
                    and isinstance(ref.get("author_id"), str)
                    and bool(ref.get("author_id"))
                    and isinstance(ref.get("exact_span"), list)
                    and len(ref["exact_span"]) == 2
                    and all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in ref["exact_span"]
                    )
                    and ref["exact_span"][0] == 0
                    and ref["exact_span"][1] == len(claim)
                ) or (
                    source_kind in {"exact_event", "event"}
                    and isinstance(ref.get("source_event_id"), str)
                    and bool(ref.get("source_event_id"))
                    and isinstance(ref.get("author_id"), str)
                    and bool(ref.get("author_id"))
                    and isinstance(ref.get("exact_span"), list)
                    and any(
                        isinstance(proof, dict)
                        and proof.get("kind") == "event"
                        and proof.get("atom_id") == atom.get("atom_id")
                        and proof.get("event_id") == ref.get("source_event_id")
                        and proof.get("actor_id") == ref.get("author_id")
                        and proof.get("text_hash") == ref.get("text_hash")
                        and proof.get("exact_span") == ref.get("exact_span")
                        for proof in event_proofs
                    )
                )
                if valid_ref:
                    valid_refs.append(ref)
            if not valid_refs:
                continue
            atom_id = str(atom.get("atom_id", ""))
            if not atom_id or (scope_id, atom_id) in seen:
                continue
            seen.add((scope_id, atom_id))
            first_ref = valid_refs[0]
            occurred_at = first_ref.get("occurred_at")
            source_kind = first_ref.get("source_kind", first_ref.get("source_status"))
            source_uids = [str(ref["author_id"]) for ref in valid_refs]
            subjects = atom.get("subject_ids", [])
            if isinstance(subjects, list):
                source_uids.extend(
                    str(subject)
                    for subject in subjects
                    if isinstance(subject, str) and subject.strip()
                )
            retrieval_rows = [
                material_rows.get(str(ref.get("material_id")))
                for ref in valid_refs
                if isinstance(ref.get("material_id"), str)
            ]
            retrieval_rows = [
                row
                for row in retrieval_rows
                if isinstance(row, dict) and row.get("retrieval_hit") is True
            ]
            retrieval_rank = min(
                (
                    int(row["retrieval_rank"])
                    for row in retrieval_rows
                    if isinstance(row.get("retrieval_rank"), int)
                    and not isinstance(row.get("retrieval_rank"), bool)
                ),
                default=None,
            )
            frames.append(
                {
                    "atom_id": atom_id,
                    "scope_umo": scope_id,
                    "text": claim,
                    "source_uids": list(dict.fromkeys(source_uids)),
                    "author_uid": str(first_ref["author_id"]),
                    "occurred_at": occurred_at,
                    "source_kind": source_kind,
                    "source_material_ids": list(
                        dict.fromkeys(
                            str(ref["material_id"])
                            for ref in valid_refs
                            if isinstance(ref.get("material_id"), str)
                        )
                    ),
                    "source_event_ids": list(
                        dict.fromkeys(
                            str(ref["source_event_id"])
                            for ref in valid_refs
                            if isinstance(ref.get("source_event_id"), str)
                        )
                    ),
                    "retrieval_hit": bool(retrieval_rows),
                    "retrieval_rank": retrieval_rank,
                    "provenance_complete": True,
                }
            )
    return frames[:12]


def relevant_frames(
    frames: list[dict[str, Any]], query: str, tokens: list[str] | None = None
) -> list[dict[str, Any]]:
    """Keep bounded retrieval matches using LivingMemory's tokenizer output."""
    text = str(query or "").strip().casefold()
    if not text:
        return []
    terms = [
        str(token).strip().casefold() for token in (tokens or []) if str(token).strip()
    ]
    if not terms:
        parts = [part for part in re.split(r"[^\w\u3400-\u9fff]+", text) if part]
        terms = []
        for part in parts:
            if re.search(r"[\u3400-\u9fff]", part):
                terms.extend(
                    part[index : index + 2] for index in range(max(0, len(part) - 1))
                )
                if len(part) == 1:
                    terms.append(part)
            else:
                terms.append(part)
        terms = [term for term in terms if term not in {"我们", "什么", "一下", "之前"}]
    ranked = []
    for index, frame in enumerate(frames):
        haystack = str(frame.get("text", "")).casefold()
        score = sum(1 for term in terms if term in haystack)
        if score:
            ranked.append((score, -index, frame))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    matched = [row[2] for row in ranked[:5]]
    if len(matched) >= 3:
        return matched
    fallback = sorted(
        (
            frame
            for frame in frames
            if frame.get("retrieval_hit") is True and frame not in matched
        ),
        key=lambda frame: (
            frame.get("retrieval_rank")
            if isinstance(frame.get("retrieval_rank"), int)
            else 10_000
        ),
    )[: 3 - len(matched)]
    return [*matched, *fallback]


def append_quote_surface(req: Any, frames: list[dict[str, Any]]) -> bool:
    """Append a provider-only temporary fragment; never edit stored history."""
    if not frames:
        return False
    parts = getattr(req, "extra_user_content_parts", None)
    if not isinstance(parts, list):
        return False
    try:
        from astrbot.core.agent.message import TextPart
    except Exception:
        return False
    lines = ["已核验的历史用户原话（仅供本轮参考）："]
    for frame in frames:
        lines.append(
            f"- 来源会话 {frame['scope_umo']}；发言者 {frame['author_uid']}；"
            f"时间 {frame.get('occurred_at') or '未知'}；来源类型 {frame.get('source_kind')}："
            f"“{frame['text']}”"
        )
    part = TextPart(text="\n".join(lines))
    marker = getattr(part, "mark_as_temp", None)
    if callable(marker):
        part = marker()
    try:
        from .recall_runtime import remember_private_segment

        remember_private_segment(
            req, part, source_level="verified_quote", item_count=len(frames)
        )
    except Exception:
        pass
    parts.append(part)
    return True
