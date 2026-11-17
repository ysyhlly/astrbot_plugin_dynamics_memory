"""Fast same-conversation history retrieval without embedding or cache fallback."""

from __future__ import annotations

import asyncio
import json
import re

from .hybrid_retriever import HybridResult


def scope_expressions(prefix=""):
    meta = (
        f"CASE WHEN json_valid({prefix}metadata) THEN {prefix}metadata ELSE '{{}}' END"
    )
    source = f"json_extract({meta}, '$.source_session_id')"
    window = f"json_extract({meta}, '$.source_window.session_id')"
    origin = (
        f"CASE WHEN NULLIF({source}, '') IS NOT NULL AND NULLIF({window}, '') IS NOT NULL "
        f"AND {source} != {window} THEN NULL ELSE "
        f"COALESCE(NULLIF({source}, ''), NULLIF({window}, ''), json_extract({meta}, '$.session_id')) END"
    )
    return (
        meta,
        origin,
        f"COALESCE(json_extract({meta}, '$.status'), 'active')",
        f"COALESCE(json_extract({meta}, '$.persona_id'), '')",
    )


async def ensure_scope_index(connection):
    """Index derived provenance, retaining legacy aliases and contradiction rules."""
    _, origin, status, persona = scope_expressions()
    await connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_narrative_scope_v1 ON documents ({origin}, {status}, {persona})"
    )


def persona_names(scope: dict) -> tuple[str, ...]:
    """Accept only bounded names supplied by the authoritative character bridge."""
    names = [scope["persona_id"]]
    aliases = scope.get("persona_aliases", [])
    if isinstance(aliases, list):
        names.extend(
            alias.strip()
            for alias in aliases[:4]
            if isinstance(alias, str) and 0 < len(alias.strip()) <= 100
        )
    return tuple(dict.fromkeys(names))


def source_origin(metadata: dict) -> str | None:
    """Prefer original conversation provenance and reject contradictory sources."""
    window = metadata.get("source_window")
    window_origin = window.get("session_id") if isinstance(window, dict) else None
    source = metadata.get("source_session_id")
    if source and window_origin and source != window_origin:
        return None
    return source or window_origin or metadata.get("session_id")


def allowed_memory(
    metadata: dict, origins: tuple[str, ...], personas: tuple[str, ...]
) -> bool:
    if source_origin(metadata) not in origins:
        return False
    persona = metadata.get("persona_id")
    return persona in personas or persona in (None, "", "default", "[%None]")


async def search_local(
    engine, query: str, k: int, origins: tuple[str, ...], personas: tuple[str, ...]
) -> list[HybridResult]:
    """Read exact source matches first, including documents missing an FTS entry."""
    if k <= 0 or not origins or not engine.db_connection:
        return []
    meta, origin, status, persona = scope_expressions("d.")
    allowed_personas = tuple(dict.fromkeys((*personas, "", "default", "[%None]")))
    filters = [
        f"{status} = 'active'",
        f"{origin} IN ({','.join('?' for _ in origins)})",
        f"{persona} IN ({','.join('?' for _ in allowed_personas)})",
    ]
    params = [*origins, *allowed_personas]
    tokens = []
    if str(query).strip() and engine.text_processor:
        try:
            tokens = await asyncio.wait_for(
                engine.text_processor.tokenize_async(
                    str(query)[:1000], remove_stopwords=True
                ),
                0.35,
            )
        except (TimeoutError, Exception):
            tokens = []
    tokens = list(dict.fromkeys(str(token) for token in tokens if str(token).strip()))[
        :16
    ]
    if not tokens:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", str(query)[:1000])[:8]
    rows = []
    if tokens:
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        try:
            cursor = await engine.db_connection.execute(
                f"SELECT d.id,d.text,{meta},bm25(livingmemory_memories_fts) FROM livingmemory_memories_fts "
                f"JOIN documents d ON d.id=livingmemory_memories_fts.doc_id "
                f"WHERE livingmemory_memories_fts MATCH ? AND {' AND '.join(filters)} ORDER BY bm25(livingmemory_memories_fts) LIMIT ?",
                [match, *params, max(k * 4, 20)],
            )
            rows = await cursor.fetchall()
        except Exception:
            # A partial legacy FTS index must not hide valid scoped documents.
            rows = []
    selected = {
        int(row[0]): (row, 1.0 - index / max(len(rows), 1))
        for index, row in enumerate(rows)
    }
    lexical_order = "+".join(
        "CASE WHEN instr(lower(d.text),lower(?))>0 THEN 1 ELSE 0 END" for _ in tokens
    )
    ordering = lexical_order + " DESC," if lexical_order else ""
    cursor = await engine.db_connection.execute(
        f"SELECT d.id,d.text,{meta},0 FROM documents d WHERE {' AND '.join(filters)} "
        f"ORDER BY {ordering}COALESCE(CAST(json_extract(d.metadata,'$.create_time') AS REAL),0) DESC,d.id DESC LIMIT ?",
        [*params, *tokens, 200],
    )
    for row in await cursor.fetchall():
        hits = sum(token.casefold() in str(row[1]).casefold() for token in tokens)
        score = 0.15 + min(hits, 8) * 0.08
        previous = selected.get(int(row[0]))
        if previous is None or score > previous[1]:
            selected[int(row[0])] = (row, score)
    results = []
    pending = []
    checked = 0
    batch_size = min(20, max(5, k * 2))
    for row, score in sorted(selected.values(), key=lambda item: item[1], reverse=True):
        metadata = json.loads(row[2])
        if not allowed_memory(metadata, origins, personas):
            continue
        candidate = HybridResult(
            int(row[0]),
            score,
            0,
            score,
            None,
            str(row[1]),
            metadata,
            {"narrative_local": score},
        )
        # Both policies preserve order. Count only candidates admitted by the
        # retrieval policy, matching the previous results[:200] ceiling.
        pending.extend(engine._filter_by_retrieval_policy([candidate]))
        if len(pending) >= batch_size or checked + len(pending) >= 200:
            results.extend(await engine._apply_atom_policy(pending))
            checked += len(pending)
            pending = []
            if len(results) >= k or checked >= 200:
                return results[:k]
    if pending:
        results.extend(await engine._apply_atom_policy(pending))
    return results[:k]
