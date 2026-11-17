"""Optimistic versions for the editable contents of a memory record."""

import hashlib
import json

from .utils.json_utils import safe_json_dict


class MemoryRevisionConflict(ValueError):
    code = "memory_revision_conflict"

    def __init__(self):
        super().__init__("记忆已更新，请刷新后重试")


def memory_revision(record):
    metadata = safe_json_dict(record.get("metadata")).copy()
    for field in (
        "access_count",
        "last_access_time",
        "needs_graph_repair",
        "graph_repair_reason",
        "graph_repair_updated_at",
    ):
        metadata.pop(field, None)
    payload = json.dumps(
        [record.get("id"), record.get("text"), metadata],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
