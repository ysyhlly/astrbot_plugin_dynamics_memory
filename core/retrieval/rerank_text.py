"""Conservative excerpts for scoring only; canonical recall text stays intact."""

from __future__ import annotations

import re

_WORDS = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
_QUALIFIERS = re.compile(
    r"不|没|未|否|以前|后来|现在|曾经|更正|补充|改为|取消|引用|转述|承诺|答应|"
    r"\b(?:not|never|no|before|after|now|correction|instead|cancelled|quoted|promised)\b",
    re.IGNORECASE,
)


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
