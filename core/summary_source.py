"""Identity of document text and its live-atom projection for private recall."""

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class AtomProjectionProof:
    document_sha256: str
    atoms: tuple[tuple[int, str], ...]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def project_live_atoms(rows) -> tuple[str, tuple[tuple[int, str], ...]]:
    """Preserve SQL order and bind each ID to its own text, including duplicates."""
    atoms = [(int(row[0]), str(row[2])) for row in rows]
    return (
        "\n".join(dict.fromkeys(text for _, text in atoms)),
        tuple((atom_id, text_sha256(text)) for atom_id, text in atoms),
    )


def summary_metadata_sha256(metadata: dict) -> str:
    """Bind source ownership and fields rendered by format_memories_for_injection."""
    keys = (
        "adapter_id",
        "create_time",
        "importance",
        "interaction_type",
        "key_facts",
        "memory_visibility",
        "participant_identities",
        "participants",
        "persona_summary",
        "session_id",
        "source_session_id",
        "source_time_end",
        "source_time_start",
        "summary_schema_version",
        "time_tags",
        "topics",
    )
    rendered = {key: metadata.get(key) for key in keys}
    return text_sha256(json.dumps(rendered, sort_keys=True, ensure_ascii=False))
