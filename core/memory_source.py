"""Structured source-conversation helpers for durable important memories."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .evidence_projection import conversation_row_hash
from .models.conversation_models import Message


def serialize_source_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Store only fields required for review and deterministic re-summarization."""
    serialized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            data = message
        elif hasattr(message, "to_dict"):
            data = message.to_dict()
        else:
            data = {
                key: getattr(message, key, None)
                for key in (
                    "id",
                    "session_id",
                    "role",
                    "content",
                    "sender_id",
                    "sender_name",
                    "group_id",
                    "platform",
                    "timestamp",
                )
            }
        content = Message.content_to_text(data.get("content"))
        if not content:
            continue
        raw_metadata = data.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        try:
            message_id = int(data.get("id") or 0)
        except (TypeError, ValueError):
            message_id = 0
        try:
            timestamp = float(data.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        serialized.append(
            {
                "id": message_id,
                "session_id": str(data.get("session_id") or ""),
                "role": str(data.get("role") or "user"),
                "content": content,
                "sender_id": str(data.get("sender_id") or "unknown"),
                "sender_name": data.get("sender_name"),
                "group_id": data.get("group_id"),
                "platform": data.get("platform"),
                "timestamp": timestamp,
                "metadata": {
                    "is_bot_message": bool(metadata.get("is_bot_message", False)),
                    **{
                        key: metadata[key]
                        for key in (
                            "schema_version",
                            "source_event_id",
                            "scope",
                            "fact_kind",
                        )
                        if key in metadata
                    },
                },
            }
        )
    return serialized


def restore_source_messages(source: list[dict[str, Any]]) -> list[Message]:
    """Restore source rows to the model expected by MemoryProcessor."""
    return [Message.from_dict(item) for item in source]


def summary_window_fingerprint(messages: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            serialize_source_messages(messages),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


__all__ = [
    "build_source_row_refs_v1",
    "restore_source_messages",
    "serialize_source_messages",
]


def build_source_row_refs_v1(
    messages: list[Any], session_id: str
) -> list[dict[str, Any]]:
    """Build body-free locators for the exact private user rows summarized.

    The inputs are the rows already selected for this summary operation. This
    helper never infers source identity from model output and never retains
    message text in the returned metadata.
    """
    if (
        not isinstance(messages, list)
        or not isinstance(session_id, str)
        or not session_id
    ):
        return []
    refs: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for message in messages:
        if isinstance(message, dict):
            data = message
        elif hasattr(message, "to_dict"):
            data = message.to_dict()
        else:
            data = {
                key: getattr(message, key, None)
                for key in (
                    "id",
                    "session_id",
                    "role",
                    "content",
                    "sender_id",
                    "platform",
                    "timestamp",
                )
            }
        if data.get("session_id") != session_id or data.get("role") != "user":
            continue
        row_id = data.get("id")
        if (
            isinstance(row_id, bool)
            or not isinstance(row_id, int)
            or row_id <= 0
            or row_id in seen_ids
        ):
            continue
        sender_id = data.get("sender_id")
        if (
            not isinstance(sender_id, str)
            or not sender_id.strip()
            or sender_id == session_id
            or sender_id.casefold() in {"unknown", "none", "null", "user"}
        ):
            continue
        platform = data.get("platform")
        if not isinstance(platform, str) or not platform.strip():
            continue
        content = Message.content_to_text(data.get("content"))
        if not content:
            continue
        try:
            timestamp = float(data.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(timestamp):
            continue
        canonical_row = {
            "id": row_id,
            "session_id": session_id,
            "role": "user",
            "sender_id": sender_id.strip(),
            "platform": platform.strip(),
            "timestamp": timestamp,
            "content": content,
        }
        refs.append(
            {
                "id": row_id,
                "session_id": session_id,
                "role": "user",
                "sender_id": sender_id.strip(),
                "platform": platform.strip(),
                "timestamp": timestamp,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "row_hash": conversation_row_hash(canonical_row),
            }
        )
        seen_ids.add(row_id)
    return refs
