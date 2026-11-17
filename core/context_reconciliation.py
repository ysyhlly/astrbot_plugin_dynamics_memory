"""Conservative presentation deduplication, leaving every evidence atom intact."""
import json


def reconcile_rows(rows):
    """Collapse repeated attributed claims, never merge people or actual events.

    Polarity, time bounds and conditions are part of identity. Contrary claims
    remain visible as attributed claims; only explicit correction can supersede
    evidence. This is deliberately not a keyword-based truth classifier.
    """
    result, seen = [], set()
    for row in rows:
        subjects = row.get('subject_ids')
        scope = row.get('scope')
        if (row.get('kind') not in {'reported_claim', 'boundary', 'refusal'}
                or not isinstance(subjects, list) or not subjects
                or not isinstance(scope, dict) or not scope
                or not isinstance(row.get('claim'), str)):
            result.append(row)
            continue
        identity = {key: row.get(key) for key in (
            'kind', 'status', 'scope', 'conditions', 'polarity', 'valid_from', 'valid_until', 'source_status')}
        identity['subject_ids'] = sorted(subjects)
        identity['claim'] = ' '.join(row['claim'].split())
        key = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result
