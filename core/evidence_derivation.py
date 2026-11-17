"""Pure validation for background evidence model output."""

from __future__ import annotations

import json
import re


class InvalidDerivationSchema(ValueError):
    """The provider returned output that cannot be safely consumed."""


_ALLOWED_KINDS = {
    "utterance_fact", "reported_claim", "commitment", "boundary", "refusal",
    "execution_claim", "execution_receipt",
}


def parse_derivation_response(text: str, expected_material_ids: set[str]) -> dict[str, list[dict]]:
    """Parse a complete model response without touching persistent memory."""
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        document = json.loads(cleaned)
    except (TypeError, ValueError):
        raise InvalidDerivationSchema("json") from None
    rows = document.get("materials") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise InvalidDerivationSchema("materials")
    expected = {str(value) for value in expected_material_ids}
    result: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise InvalidDerivationSchema("row")
        material_id = row.get("material_id")
        if not isinstance(material_id, str) or material_id not in expected or material_id in result:
            raise InvalidDerivationSchema("material_id")
        atoms = row.get("atoms")
        if not isinstance(atoms, list):
            raise InvalidDerivationSchema("atoms")
        for atom in atoms[:12]:
            if not isinstance(atom, dict):
                raise InvalidDerivationSchema("atom")
            if (atom.get("kind") not in _ALLOWED_KINDS or
                    not isinstance(atom.get("claim"), str) or not atom["claim"].strip() or
                    not isinstance(atom.get("span_start"), int) or isinstance(atom["span_start"], bool) or
                    not isinstance(atom.get("span_end"), int) or isinstance(atom["span_end"], bool) or
                    not isinstance(atom.get("conditions"), str) or
                    atom.get("polarity") not in {"positive", "negative", "unknown"}):
                raise InvalidDerivationSchema("atom_fields")
        result[material_id] = atoms[:12]
    if set(result) != expected:
        raise InvalidDerivationSchema("incomplete")
    return result
