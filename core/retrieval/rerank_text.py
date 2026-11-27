"""Conservative excerpts and atom-focused text builders for rerank scoring only.

Canonical recall text and metadata stay intact for evidence verification.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_WORDS = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
_QUALIFIERS = re.compile(
    r"不|没|未|否|以前|后来|现在|曾经|更正|补充|改为|取消|引用|转述|承诺|答应|"
    r"\b(?:not|never|no|before|after|now|correction|instead|cancelled|quoted|promised)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")


def scoring_excerpt(query: str, content: str, target_chars: int = 2400) -> str:
    """Keep complete fact lines, context neighbours and all qualification lines.

    The target is soft: if preserving context is too expensive or the query
    provides no discriminating signal, retain the original rather than cut it.
    Never split a single long fact or invent a summarized claim.
    """
    if len(content) <= target_chars or "\n" not in content:
        return content
    lines = content.splitlines()
    terms = set()
    for word in _WORDS.findall(query.casefold()[:1000]):
        if re.search(r"[\u4e00-\u9fff]", word):
            terms.update(word[i : i + 2] for i in range(len(word) - 1))
        elif len(word) >= 2:
            terms.add(word)
    terms -= {"什么", "怎么", "是否", "一下", "我们", "你们", "告诉", "事情"}
    if not terms:
        return content
    scores = [sum(term in line.casefold() for term in terms) for line in lines]
    maximum = max(scores, default=0)
    if not maximum or min(scores) == maximum:
        return content
    mandatory = {0}
    mandatory.update(i for i, line in enumerate(lines) if _QUALIFIERS.search(line))

    # Preserve the neighbours of global qualifications too: their subject or
    # antecedent may be in the immediately preceding/following fact line.
    def neighbours(indices):
        return {j for i in indices for j in (i - 1, i, i + 1) if 0 <= j < len(lines)}

    selected = neighbours(mandatory)
    # Every equally strong match is retained, including a corrected fact at
    # the end. If ties do not fit, no arbitrary prefix wins.
    selected |= neighbours(i for i, score in enumerate(scores) if score == maximum)

    def size(indices):
        return sum(len(lines[i]) + 1 for i in indices)

    if size(selected) > target_chars:
        return content
    for score in sorted(set(scores) - {0, maximum}, reverse=True):
        group = neighbours(i for i, value in enumerate(scores) if value == score)
        if size(selected | group) > target_chars:
            break
        selected |= group
    # Mark omissions explicitly, without merging distant claims into a sentence.
    output, previous = [], -1
    for index in sorted(selected):
        if index > previous + 1:
            output.append("[…]")
        output.append(lines[index])
        previous = index
    if previous < len(lines) - 1:
        output.append("[…]")
    excerpt = "\n".join(output)
    return excerpt if len(excerpt) < len(content) else content


def _extract_query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for word in _WORDS.findall(str(query or "").casefold()[:1000]):
        if re.search(r"[\u4e00-\u9fff]", word):
            if len(word) == 1:
                terms.add(word)
            else:
                terms.update(word[i : i + 2] for i in range(len(word) - 1))
        elif len(word) >= 2:
            terms.add(word)
    terms -= {"什么", "怎么", "是否", "一下", "我们", "你们", "告诉", "事情"}
    return terms


def _concise_qualifier_excerpt(query: str, content: str, budget: int) -> str:
    """Build a bounded excerpt (<= budget) preserving query hits and qualifiers."""
    text = str(content or "").strip()
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text

    # Try line-level conservative excerpt first when multiline
    if "\n" in text:
        candidate = scoring_excerpt(query, text, target_chars=budget)
        if len(candidate) <= budget:
            return candidate

    # Sentence-level selection preserving query-matching & qualifier sentences
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 1:
        return text[: max(1, budget - 1)].rstrip() + "…"

    terms = _extract_query_terms(query)
    scored: list[tuple[int, int, int, str]] = []
    for idx, sent in enumerate(sentences):
        low = sent.casefold()
        hit = sum(1 for term in terms if term in low) if terms else 0
        qual = 1 if _QUALIFIERS.search(sent) else 0
        # Prioritize query hits, then qualifiers, then earlier position
        scored.append((hit, qual, -idx, sent))

    selected_indices: set[int] = set()
    used = 0
    for _hit, _qual, neg_idx, sent in sorted(scored, reverse=True):
        idx = -neg_idx
        extra = len(sent) + (1 if selected_indices else 0)
        if used + extra > budget:
            if not selected_indices:
                return sent[: max(1, budget - 1)].rstrip() + "…"
            continue
        selected_indices.add(idx)
        used += extra

    if not selected_indices:
        return text[: max(1, budget - 1)].rstrip() + "…"

    ordered = [sentences[i] for i in sorted(selected_indices)]
    joined = " ".join(ordered)
    if len(joined) > budget:
        return joined[: max(1, budget - 1)].rstrip() + "…"
    return joined


def _normalize_atom_texts(matched_atoms: Sequence[Any] | None) -> list[str]:
    if not matched_atoms:
        return []
    texts: list[str] = []
    seen: set[str] = set()
    for atom in matched_atoms:
        if isinstance(atom, dict):
            raw = str(atom.get("content") or "").strip()
        elif isinstance(atom, str):
            raw = atom.strip()
        else:
            raw = str(getattr(atom, "content", "") or "").strip()
        compact = re.sub(r"\s+", " ", raw)
        if compact and compact not in seen:
            seen.add(compact)
            texts.append(compact)
        if len(texts) >= 3:
            break
    return texts


def build_rerank_document(
    query: str,
    content: str,
    *,
    matched_atoms: Sequence[Any] | None = None,
    max_chars: int = 350,
    fallback_target_chars: int | None = None,
) -> str:
    """Construct concise rerank input text eliminating multi-topic summary dilution.

    - When ``matched_atoms`` are present: leads with ``[核心事实]`` followed by a
      qualifier-preserving ``[上下文摘录]`` bounded to ``<= max_chars`` (default 350).
    - Otherwise: returns a qualifier-preserving excerpt bounded to
      ``fallback_target_chars`` (defaults to ``max_chars``).
    """
    raw_content = str(content or "").strip()
    atom_texts = _normalize_atom_texts(matched_atoms)
    if atom_texts:
        cap = max(120, int(max_chars))
        facts_header = "[核心事实]\n"
        atom_lines: list[str] = []
        used = len(facts_header)
        for text in atom_texts:
            line = f"- {text}"
            if used + len(line) + 1 > cap:
                remaining = cap - used - 3
                if not atom_lines and remaining > 16:
                    atom_lines.append(f"- {text[:remaining].rstrip()}…")
                break
            atom_lines.append(line)
            used += len(line) + 1
        facts_block = facts_header + "\n".join(atom_lines)

        context_header = "\n[上下文摘录]\n"
        remaining_budget = cap - len(facts_block) - len(context_header)
        if remaining_budget >= 40 and raw_content:
            excerpt = _concise_qualifier_excerpt(query, raw_content, remaining_budget)
            if excerpt:
                return f"{facts_block}{context_header}{excerpt}"[:cap]
        return facts_block[:cap]

    target = max(
        120,
        int(fallback_target_chars if fallback_target_chars is not None else max_chars),
    )
    if len(raw_content) <= target:
        return raw_content
    excerpt = scoring_excerpt(query, raw_content, target_chars=target)
    if len(excerpt) <= target:
        return excerpt
    return _concise_qualifier_excerpt(query, excerpt, target)


__all__ = ["build_rerank_document", "scoring_excerpt"]
