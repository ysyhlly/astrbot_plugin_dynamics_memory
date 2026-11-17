"""Pure, provenance-safe helpers for LivingMemory evidence projections."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

PROJECTION_VERSION = "evidence-view-v1"
PURPOSES = frozenset({"reply", "decision", "history_quote", "reflection", "replay"})
MAX_MATERIALS = 64
MAX_TEXT_CHARS = 10000


def stable_id(prefix: str, parts: list[Any] | tuple[Any, ...]) -> str:
    value = json.dumps(
        list(parts), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def history_material_id(
    scope_key: str,
    conversation_id: str,
    snapshot_fingerprint: str,
    position: int,
    text: str,
) -> str:
    return stable_id(
        "mat1:",
        [
            scope_key,
            conversation_id,
            snapshot_fingerprint,
            int(position),
            content_hash(text),
        ],
    )


def history_material_family_id(
    scope_key: str,
    conversation_id: str,
    position: int,
    text: str,
    source_prefix_anchor: str,
) -> str:
    """Cross-snapshot identity only when the host supplies a trusted full-prefix anchor."""
    if not isinstance(source_prefix_anchor, str) or not source_prefix_anchor.strip():
        return ""
    return stable_id(
        "fam1:",
        [
            scope_key,
            conversation_id,
            source_prefix_anchor.strip(),
            int(position),
            content_hash(text),
        ],
    )


def legacy_material_id(scope_key: str, document_id: str, text: str) -> str:
    return stable_id("leg1:", [scope_key, str(document_id), content_hash(text)])


def event_evidence_id(scope_key: str, source_event_id: str) -> str:
    return stable_id("ev1:", [scope_key, str(source_event_id)])


def conversation_row_hash(row: dict) -> str:
    """Bind a private stored row to its exact database identity and contents."""
    return stable_id(
        "row1:",
        [
            str(row.get("session_id", "")),
            int(row.get("id", 0)),
            str(row.get("role", "")),
            str(row.get("sender_id", "")),
            str(row.get("platform", "")),
            float(row.get("timestamp", 0)),
            str(row.get("content", "")),
        ],
    )


def projection_key(
    scope_key: str,
    material_id: str,
    text: str,
    correction_revision: int,
    card_version: str,
    purpose: str,
) -> str:
    version_dependency = (
        str(card_version)
        if purpose in {"style_compatibility", "candidate_compatibility"}
        else ""
    )
    return stable_id(
        "pv1:",
        [
            scope_key,
            material_id,
            content_hash(text),
            PROJECTION_VERSION,
            int(correction_revision),
            version_dependency,
            purpose,
        ],
    )


def normalize_host_material(
    scope_key: str,
    raw: dict,
    *,
    conversation_id: str = "",
    snapshot_fingerprint: str = "",
    card_version: str = "",
    correction_revision: int = 0,
    purpose: str = "reply",
) -> tuple[dict, dict] | None:
    """Accept a host history row only with a trusted host conversation/snapshot identity.

    Arbitrary sender IDs, timestamps, event IDs and delivery claims are discarded.
    """
    if not isinstance(raw, dict):
        return None
    kind = raw.get("source_kind", raw.get("material_kind", "host_history"))
    if kind != "host_history":
        return None
    text = raw.get("text", raw.get("content"))
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        return None
    role = raw.get("role")
    if role not in {"user", "assistant"}:
        return None
    conv = raw.get("conversation_id", conversation_id)
    snapshot = raw.get("snapshot_fingerprint", snapshot_fingerprint)
    if not isinstance(conv, str) or not conv.strip() or len(conv) > 512:
        return None
    if not isinstance(snapshot, str) or not snapshot.strip() or len(snapshot) > 256:
        return None
    position = raw.get("position", raw.get("source_position"))
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        return None
    material_id = history_material_id(
        scope_key, conv.strip(), snapshot.strip(), position, text
    )
    text_hash = content_hash(text)
    anchor = raw.get("source_prefix_anchor", raw.get("message_anchor", ""))
    epoch = raw.get("history_epoch", "")
    if not isinstance(anchor, str) or len(anchor) > 256:
        anchor = ""
    if not isinstance(epoch, str) or len(epoch) > 256:
        epoch = ""
    family_anchor = (
        stable_id("ha1:", [epoch.strip(), anchor.strip()])
        if anchor.strip() and epoch.strip()
        else ""
    )
    family_id = history_material_family_id(
        scope_key, conv.strip(), position, text, family_anchor
    )
    projection_identity = family_id or material_id
    descriptor = {
        "material_id": material_id,
        "scope_key": scope_key,
        "material_kind": "host_record_only",
        "source_kind": "host_history",
        "correction_revision": int(correction_revision),
        "source_family_id": family_id,
        "source_prefix_anchor": anchor.strip() if family_id else "",
        "history_epoch": epoch.strip() if family_id else "",
        "source_locator": {
            "kind": "host_history",
            "conversation_id": conv.strip(),
            "snapshot_fingerprint": snapshot.strip(),
            "position": position,
            "role": role,
            "source_prefix_anchor": anchor.strip() if family_id else "",
            "history_epoch": epoch.strip() if family_id else "",
            "correction_revision": int(correction_revision),
            "card_version": str(card_version)[:256],
            "purpose": purpose,
        },
        "role": role,
        "conversation_id": conv.strip(),
        "snapshot_fingerprint": snapshot.strip(),
        "position": position,
        "text_hash": text_hash,
        "projection_key": projection_key(
            scope_key,
            projection_identity,
            text,
            correction_revision,
            card_version,
            purpose,
        ),
        "card_version": str(card_version)[:256],
    }
    material = {
        **descriptor,
        "source_kind": "host_history",
        "text": text,
        "source_status": "host_record_only",
        "author_id": None,
        "occurred_at": None,
        "delivery_status": "unknown",
        "source_event_id": None,
        "root_evidence_ids": [],
        "root_material_ids": [material_id],
    }
    return descriptor, material


def normalize_event_material(
    scope_key: str,
    scope: dict,
    raw: dict,
    *,
    card_version: str = "",
    correction_revision: int = 0,
    purpose: str = "reply",
) -> tuple[dict, dict] | None:
    if not isinstance(raw, dict) or raw.get("source_kind") != "event":
        return None
    source_scope = raw.get("scope")
    if not isinstance(source_scope, dict) or any(
        source_scope.get(k) != scope.get(k)
        for k in ("platform", "umo", "persona_id", "instance_id")
    ):
        return None
    event_id = raw.get("event_id", raw.get("source_event_id"))
    event_type = raw.get("event_type", raw.get("type"))
    if (
        not isinstance(event_id, str)
        or not event_id.strip()
        or event_type
        not in {"message.received", "delivery.success", "tool.execution_receipt"}
    ):
        return None
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        return None
    if event_type == "tool.execution_receipt":
        receipt = raw.get("receipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("source") != "astrbot_runtime"
            or receipt.get("status") != "success"
            or not isinstance(receipt.get("operation_id"), str)
            or not receipt["operation_id"].strip()
        ):
            return None
    evidence_id = event_evidence_id(scope_key, event_id.strip())
    descriptor = {
        "material_id": evidence_id,
        "scope_key": scope_key,
        "source_kind": "event",
        "correction_revision": int(correction_revision),
        "source_family_id": evidence_id,
        "material_kind": "program_receipt"
        if event_type == "tool.execution_receipt"
        else "source_event",
        "role": "assistant" if event_type != "message.received" else "user",
        "conversation_id": str(scope["umo"]),
        "snapshot_fingerprint": "event:" + event_id.strip(),
        "position": int(raw.get("source_position", 0)),
        "text_hash": content_hash(text),
        "projection_key": projection_key(
            scope_key, evidence_id, text, correction_revision, card_version, purpose
        ),
        "card_version": str(card_version)[:256],
        "source_locator": {
            "kind": "event",
            "event_id": event_id.strip(),
            "event_type": event_type,
            "scope": {
                key: str(source_scope.get(key, ""))
                for key in ("platform", "umo", "persona_id", "instance_id")
            },
            "correction_revision": int(correction_revision),
            "card_version": str(card_version)[:256],
            "purpose": purpose,
        },
    }
    actor_id = str(raw.get("actor_id") or "")[:160] or None
    material = {
        **descriptor,
        "source_kind": "event",
        "scope": dict(source_scope),
        "text": text,
        "source_status": "exact_event",
        # Keep the trusted archive actor in its input form as well as the
        # canonical projection field so a second normalization is idempotent.
        "actor_id": actor_id,
        "author_id": actor_id,
        "occurred_at": raw.get("occurred_at"),
        "delivery_status": "success"
        if event_type == "delivery.success"
        else "verified"
        if event_type == "tool.execution_receipt"
        else "received",
        "source_event_id": event_id.strip(),
        "event_type": event_type,
        "receipt": raw.get("receipt")
        if event_type == "tool.execution_receipt"
        else None,
        "root_evidence_ids": [evidence_id],
        "root_material_ids": [],
    }
    return descriptor, material


def normalize_conversation_row(
    scope_key: str,
    scope: dict,
    raw: dict,
    *,
    source_memory_id: int,
    card_version: str = "",
    correction_revision: int = 0,
    purpose: str = "reply",
) -> tuple[dict, dict] | None:
    """Normalize one exact private user row anchored by a retained source snapshot.

    The row is material evidence, never a platform event: it receives a mat1
    identity and cannot create ev1 roots, delivery receipts, or narrative events.
    """
    if not isinstance(raw, dict) or raw.get("role") != "user":
        return None
    if not isinstance(scope, dict) or any(
        not isinstance(scope.get(key), str) or not scope[key].strip()
        for key in ("platform", "umo", "persona_id", "instance_id")
    ):
        return None
    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or session_id != scope.get("umo"):
        return None
    row_id = raw.get("id")
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
        return None
    try:
        memory_id = int(source_memory_id)
    except (TypeError, ValueError):
        return None
    if isinstance(source_memory_id, bool) or memory_id <= 0:
        return None
    content = raw.get("content")
    if (
        not isinstance(content, str)
        or not content.strip()
        or len(content) > MAX_TEXT_CHARS
    ):
        return None
    sender_id = raw.get("sender_id")
    if not isinstance(sender_id, str) or not sender_id.strip():
        return None
    sender_id = sender_id.strip()
    # The session fallback and legacy placeholders do not identify a speaker.
    if sender_id == session_id or sender_id.casefold() in {
        "unknown",
        "none",
        "null",
        "user",
    }:
        return None
    platform = raw.get("platform")
    if not isinstance(platform, str) or not platform.strip():
        return None
    try:
        timestamp_value = float(raw.get("timestamp"))
    except (TypeError, ValueError):
        return None
    if not (
        timestamp_value == timestamp_value and abs(timestamp_value) != float("inf")
    ):
        return None
    row = {
        "id": row_id,
        "session_id": session_id,
        "role": "user",
        "sender_id": sender_id,
        "platform": platform.strip(),
        "timestamp": timestamp_value,
        "content": content,
    }
    row_hash = conversation_row_hash(row)
    material_id = stable_id(
        "mat1:",
        [scope_key, "conversation_row", memory_id, session_id, row_id, row_hash],
    )
    text_hash = content_hash(content)
    descriptor = {
        "material_id": material_id,
        "scope_key": scope_key,
        "source_kind": "conversation_row",
        "material_kind": "conversation_row",
        "correction_revision": int(correction_revision),
        "source_family_id": "",
        "role": "user",
        "conversation_id": session_id,
        "snapshot_fingerprint": row_hash,
        "position": row_id,
        "text_hash": text_hash,
        "projection_key": projection_key(
            scope_key, material_id, content, correction_revision, card_version, purpose
        ),
        "card_version": str(card_version)[:256],
        "source_locator": {
            "kind": "conversation_row",
            "source_store": "livingmemory.conversations.messages",
            "session_id": session_id,
            "row_id": row_id,
            "row_hash": row_hash,
            "source_memory_id": memory_id,
            "scope": {
                key: str(scope[key])
                for key in ("platform", "umo", "persona_id", "instance_id")
            },
            "correction_revision": int(correction_revision),
            "card_version": str(card_version)[:256],
            "purpose": purpose,
        },
    }
    material = {
        **descriptor,
        **row,
        "text": content,
        "actor_id": sender_id,
        "author_id": sender_id,
        "source_status": "conversation_row",
        "occurred_at": timestamp_value,
        "delivery_status": "unknown",
        "source_event_id": None,
        "root_evidence_ids": [],
        "root_material_ids": [material_id],
        "source_memory_id": memory_id,
        "row_hash": row_hash,
    }
    return descriptor, material


def normalize_legacy_summary(
    scope_key: str,
    raw: dict,
    *,
    card_version: str = "",
    correction_revision: int = 0,
    purpose: str = "reply",
) -> tuple[dict, dict] | None:
    """Wrap an old summary as an unverified legacy source, without inventing chat identity."""
    if not isinstance(raw, dict) or raw.get("source_kind") != "legacy_summary":
        return None
    text = raw.get("text")
    document_id = raw.get("document_id")
    if not isinstance(document_id, (str, int)) or not str(document_id).strip():
        return None
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        return None
    material_id = legacy_material_id(scope_key, str(document_id), text)
    descriptor = {
        "material_id": material_id,
        "scope_key": scope_key,
        "material_kind": "legacy_summary",
        "source_kind": "legacy_summary",
        "correction_revision": int(correction_revision),
        "source_family_id": "",
        "document_id": str(document_id),
        "role": "summary",
        "conversation_id": "",
        "snapshot_fingerprint": "",
        "position": -1,
        "text_hash": content_hash(text),
        "projection_key": projection_key(
            scope_key, material_id, text, correction_revision, card_version, purpose
        ),
        "card_version": str(card_version)[:256],
        "source_locator": {
            "kind": "legacy_summary",
            "document_id": str(document_id),
            "source_umo": str(raw.get("source_umo") or ""),
            "correction_revision": int(correction_revision),
            "card_version": str(card_version)[:256],
            "purpose": purpose,
        },
    }
    return descriptor, {
        **descriptor,
        "text": text,
        "source_status": "legacy_summary",
        "author_id": None,
        "occurred_at": None,
        "delivery_status": "unknown",
        "source_event_id": None,
        "root_evidence_ids": [],
        "root_material_ids": [material_id],
    }


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;\n])\s*|(?<=[.!?])\s+(?=[^.!?])")
_CONTINUITY = re.compile(
    r"(答应|答应了|承诺|约好|约定|我会|我来|我帮|会帮|下次|明天|今晚|以后|到时候|记得|别再|不要再|不再|不叫|停止|先别|拒绝|做完|已经完成|正在处理|还没做|尚未|等我|我去|我负责|不想|不愿意|不能|不会|I'll|I will|I can|I promised|promise|promised|agreed|next time|tomorrow|tonight|don't call|do not call|stop|won't|will not|not yet|haven't done|finished|complete)",
    re.IGNORECASE,
)


def pending_continuity_spans(
    text: str,
    *,
    max_spans: int = 3,
    max_total_chars: int = 1200,
    max_span_chars: int = 500,
) -> list[dict]:
    """Return only complete candidate sentences; never truncate a condition or negation."""
    output = []
    offset = 0
    total = 0
    for chunk in _SENTENCE_BOUNDARY.split(text):
        if not chunk:
            continue
        start = text.find(chunk, offset)
        if start < 0:
            continue
        end = start + len(chunk)
        offset = end
        sentence = chunk.strip()
        trim_left = len(chunk) - len(chunk.lstrip())
        trim_right = len(chunk.rstrip())
        sentence_start = start + trim_left
        sentence_end = start + trim_right
        if (
            not sentence
            or len(sentence) > max_span_chars
            or not _CONTINUITY.search(sentence)
        ):
            continue
        if total + len(sentence) > max_total_chars:
            continue
        output.append(
            {
                "start": sentence_start,
                "end": sentence_end,
                "text": sentence,
                "kind": "pending_continuity",
                "claim_status": "unverified",
                "source_status": "host_record_only",
            }
        )
        total += len(sentence)
        if len(output) >= max_spans:
            break
    return output


def atom_id(
    scope_key: str,
    kind: str,
    claim: str,
    roots: list[str] | tuple[str, ...],
    revision: int,
) -> str:
    normalized = " ".join(str(claim).casefold().split())
    return stable_id(
        "atm1:", [scope_key, kind, normalized, sorted(set(roots)), int(revision)]
    )
