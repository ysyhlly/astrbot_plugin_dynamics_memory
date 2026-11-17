"""Retained parser paths keep real content and reject absent summaries."""

from types import SimpleNamespace as NS

import pytest

from astrbot_plugin_dynamics_memory.core.base.constants import (
    MEMORY_INJECTION_FOOTER,
    MEMORY_INJECTION_HEADER,
)
from astrbot_plugin_dynamics_memory.core.processors.memory_processor import (
    MemoryProcessor,
)
from astrbot_plugin_dynamics_memory.core.processors.memory_processor_parse import (
    extract_by_regex,
)
from astrbot_plugin_dynamics_memory.core.prompts import prompt_manager
from astrbot_plugin_dynamics_memory.core.utils.formatting import (
    format_memories_for_injection,
)


def test_regex_recovers_actual_fields_from_truncated_json():
    data = extract_by_regex(
        'prefix {broken} "summary":"actual preference", "importance":0.8, "sentiment":"positive", "topics":["coffee"], "key_facts":["likes coffee"]',
        False,
    )
    assert data["summary"] == "actual preference" and data["importance"] == 0.8
    assert data["topics"] == ["coffee"] and data["key_facts"] == ["likes coffee"]
    assert data["sentiment"] == "positive"
    assert (
        extract_by_regex('prefix {"summary":"actual"} suffix', True)["summary"]
        == "actual"
    )
    assert extract_by_regex("no source information", False) is None
    assert extract_by_regex('"summary":"actual", "importance":1.2.3', False) is None


def test_merge_parser_handles_wrappers_but_requires_object():
    processor = MemoryProcessor.__new__(MemoryProcessor)
    for text in ('{"summary":"coffee"}', '```json\n{"summary":"coffee"}\n```'):
        assert processor._parse_merge_response(text) == {"summary": "coffee"}
    for text in ("[]", "not json"):
        with pytest.raises(RuntimeError):
            processor._parse_merge_response(text)


@pytest.mark.parametrize("manager_mode", ["missing", "broken", "custom"])
def test_formatter_preserves_source_boundary_and_budget(monkeypatch, manager_mode):
    def manager():
        if manager_mode == "broken":
            raise RuntimeError("prompt manager unavailable")
        if manager_mode == "custom":
            return NS(get_prompt=lambda name: "custom " + name)
        return None

    monkeypatch.setattr(prompt_manager, "get_prompt_manager", manager)
    rows = [
        NS(
            content="first fact",
            score=0.8,
            timestamp="bad",
            metadata='{"importance":0.8,"topics":["coffee"]}',
        ),
        {
            "content": "second fact " * 200,
            "score": 0.7,
            "metadata": {"importance": 0.5, "create_time": 1700000000},
        },
        {"content": "invalid entry", "metadata": None},
    ]
    result = format_memories_for_injection(rows, max_chars=750)
    assert result.startswith(MEMORY_INJECTION_HEADER) and result.endswith(
        MEMORY_INJECTION_FOOTER
    )
    assert (
        "first fact" in result and "invalid entry" not in result and len(result) <= 750
    )
    assert format_memories_for_injection([{"content": "bad", "metadata": None}]) == ""
