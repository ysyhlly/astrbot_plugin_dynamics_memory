"""Regressions for JSON repair (#278) and summary prefix caching (#277)."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.models.conversation_models import Message
from astrbot_plugin_dynamics_memory.core.processors import (
    memory_processor as processor_module,
)
from astrbot_plugin_dynamics_memory.core.processors.memory_processor import (
    MemoryProcessor,
)
from astrbot_plugin_dynamics_memory.core.prompts import prompt_manager


@pytest.fixture
def processor(monkeypatch, tmp_path):
    manager = prompt_manager.PromptManager(str(tmp_path))
    monkeypatch.setattr(prompt_manager, "get_prompt_manager", lambda: manager)
    provider = SimpleNamespace(
        text_chat=AsyncMock(
            return_value=SimpleNamespace(
                completion_text=(
                    '{"summary":"测试记忆总结的完整内容","key_facts":["测试事实"],'
                    '"topics":[],"importance":0.5}'
                )
            )
        )
    )
    return MemoryProcessor(llm_provider=provider)


@pytest.mark.parametrize("fenced", [False, True])
def test_merge_preserves_multiline_json_and_string_punctuation(processor, fenced):
    expected = {
        "summary": '原文含 ,}、,]、[{ 和转义引号 "，不能改写',
        "key_facts": ["第一行\n第二行\t制表符", "反斜杠\\"],
        "topics": [],
        "importance": 0.5,
    }
    text = json.dumps(expected, ensure_ascii=False, indent=2)
    assert processor._try_fix_json(text) == text
    response = f"```json\n{text}\n```" if fenced else text
    assert processor._parse_merge_response(response) == expected


def test_merge_tries_valid_original_before_repair(processor, monkeypatch):
    def reject_repair(text):
        raise AssertionError("合法 JSON 不应先修复")

    monkeypatch.setattr(processor, "_try_fix_json", reject_repair)
    assert processor._parse_merge_response('{\n "summary": "正常"\n}') == {
        "summary": "正常"
    }


def test_repair_only_escapes_controls_inside_strings(processor):
    response = '{\n\t"summary": "首行\n次行\r\t末尾",\n "key_facts": [],\n}'
    repaired = processor._try_fix_json(response)
    assert repaired.startswith('{\n\t"summary"')
    assert processor._parse_merge_response(response)["summary"] == "首行\n次行\r\t末尾"


def test_repairable_json_still_parses_end_to_end():
    """可修复的 JSON 仍应正常解析，不会被当作解析失败。"""
    from astrbot_plugin_dynamics_memory.core.processors.memory_processor_parse import (
        parse_llm_response,
    )

    data = parse_llm_response(
        '{"summary": "修复后仍可使用", "key_facts": ["事实"], "importance": 0.5,}',
        is_group_chat=False,
    )

    assert data["summary"] == "修复后仍可使用"
    assert data["key_facts"] == ["事实"]


def test_repair_closes_nested_structures_in_order(processor):
    assert json.loads(processor._try_fix_json('{"facts": [{"text": "含 [{ 字符')) == {
        "facts": [{"text": "含 [{ 字符"}]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("with_persona", [False, True])
@pytest.mark.parametrize("is_group", [False, True])
async def test_summary_keeps_fixed_prefix_across_dates(
    processor, monkeypatch, with_persona, is_group
):
    if with_persona:
        processor.context = SimpleNamespace(
            persona_manager=SimpleNamespace(
                get_persona=AsyncMock(
                    return_value=SimpleNamespace(system_prompt="你是专业助手")
                )
            )
        )
    messages = [
        Message(
            id=1,
            session_id="s",
            role="user",
            content="明天开会",
            sender_id="u",
            sender_name="张三",
            group_id="g" if is_group else None,
            platform="test",
            timestamp=datetime(2026, 9, 20, 23, 50).timestamp(),
            metadata={},
        )
    ]
    calls = []
    for day in (1, 2):

        class FrozenDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 10, day, 12, 0)

        monkeypatch.setattr(processor_module, "datetime", FrozenDateTime)
        await processor.process_conversation(
            messages, is_group, "p" if with_persona else None
        )
        calls.append(processor._llm_provider.text_chat.call_args.kwargs)

    assert calls[0]["system_prompt"] == calls[1]["system_prompt"]
    first_prompt, second_prompt = (call["prompt"] for call in calls)
    fixed_prefix = first_prompt.split("# 本次请求")[0]
    assert fixed_prefix == second_prompt.split("# 本次请求")[0]
    assert "# 输出格式" in fixed_prefix and "# 示例" in fixed_prefix
    assert "2026-10-01 12:00" in first_prompt
    assert "2026-10-02 12:00" in second_prompt
    assert "2026-09-20 23:50" in first_prompt
    # DM-1: {current_date} must sit at the end after the conversation block so the
    # static instructions + conversation prefix stays byte-identical across retries/minutes.
    assert first_prompt.index("2026-09-20") < first_prompt.index("2026-10-01")
    assert (
        first_prompt.split("当前日期时间:")[0]
        == second_prompt.split("当前日期时间:")[0]
    )
    assert "消息的发送时间" in calls[0]["system_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_persona", [False, True])
@pytest.mark.parametrize("is_group", [False, True])
async def test_reflection_system_prompt_byte_identical_across_normal_and_retry(
    processor, monkeypatch, with_persona, is_group
):
    """DM-1: reflection system_prompt stays byte-identical on quality retry;
    current_date and quality_feedback both sit at the end of user prompt.
    """
    if with_persona:
        processor.context = SimpleNamespace(
            persona_manager=SimpleNamespace(
                get_persona=AsyncMock(
                    return_value=SimpleNamespace(system_prompt="你是严谨助手")
                )
            )
        )
    messages = [
        Message(
            id=1,
            session_id="s",
            role="user",
            content="下周三提交版本发布说明",
            sender_id="u1",
            sender_name="李四",
            group_id="g1" if is_group else None,
            platform="test",
            timestamp=datetime(2026, 11, 19, 10, 0).timestamp(),
            metadata={"source_event_id": "evt_1"},
        )
    ]

    class FirstMinute(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 19, 10, 5)

    class NextMinute(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 19, 10, 6)

    monkeypatch.setattr(processor_module, "datetime", FirstMinute)
    await processor.process_conversation(
        messages, is_group, "p1" if with_persona else None
    )
    normal_call = dict(processor._llm_provider.text_chat.call_args.kwargs)

    monkeypatch.setattr(processor_module, "datetime", NextMinute)
    await processor.process_conversation(
        messages,
        is_group,
        "p1" if with_persona else None,
        quality_feedback="summary_generic_reference",
    )
    retry_call = dict(processor._llm_provider.text_chat.call_args.kwargs)

    # 1. system_prompt is strictly byte-identical across normal and retry calls
    assert normal_call["system_prompt"] == retry_call["system_prompt"]
    assert "上次摘要" not in retry_call["system_prompt"]
    assert "2026-11-19 10:05" not in normal_call["system_prompt"]
    assert "2026-11-19 10:06" not in retry_call["system_prompt"]

    # 2. Leading user prompt prefix (instructions + conversation + static source constraint)
    #    stays byte-identical across normal and retry attempts
    normal_prefix, normal_date_tail = normal_call["prompt"].rsplit("当前日期时间:", 1)
    retry_prefix, retry_date_tail = retry_call["prompt"].rsplit("当前日期时间:", 1)
    assert normal_prefix == retry_prefix
    assert "下周三提交版本发布说明" in normal_prefix
    assert "来源约束" in normal_prefix

    # 3. current_date and quality_feedback are both placed at the tail of user prompt
    assert normal_date_tail.strip() == "2026-11-19 10:05"
    assert "2026-11-19 10:06" in retry_date_tail
    assert "上次摘要" in retry_date_tail
    assert retry_date_tail.index("2026-11-19 10:06") < retry_date_tail.index("上次摘要")
    assert retry_call["prompt"].endswith("并在输出前逐项自检。")


@pytest.mark.asyncio
async def test_legacy_custom_template_relocates_current_date_to_tail(
    processor, monkeypatch
):
    """Legacy user templates with leading current_date still get tail placement."""
    manager = prompt_manager.get_prompt_manager()
    manager.update_prompt(
        "private_chat_prompt",
        "# 规则\n只输出 JSON。\n\n# 本次请求\n当前日期时间: {current_date}\n\n# 对话历史\n{conversation}\n",
    )
    messages = [
        Message(
            id=1,
            session_id="s",
            role="user",
            content="记录一下明天的安排",
            sender_id="u1",
            sender_name="王五",
            group_id=None,
            platform="test",
            timestamp=datetime(2026, 11, 19, 9, 0).timestamp(),
            metadata={},
        )
    ]

    class FrozenNow(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 19, 14, 20)

    monkeypatch.setattr(processor_module, "datetime", FrozenNow)
    await processor.process_conversation(messages, is_group_chat=False, persona_id=None)
    sent_prompt = processor._llm_provider.text_chat.call_args.kwargs["prompt"]
    assert "{current_date}" not in sent_prompt
    assert sent_prompt.index("记录一下明天的安排") < sent_prompt.index(
        "2026-11-19 14:20"
    )
    assert sent_prompt.rstrip().endswith("当前日期时间: 2026-11-19 14:20")


@pytest.mark.asyncio
async def test_custom_system_template_keeps_date_variable_support(processor):
    manager = prompt_manager.get_prompt_manager()
    manager.update_prompt("memory_system_prompt_base", "自定义 {current_date}")
    system_prompt = await processor._build_system_prompt_with_persona(None)
    assert datetime.now().strftime("%Y-%m-%d") in system_prompt
    assert "{current_date}" not in system_prompt


def test_system_prompt_fallbacks_are_static():
    base = MemoryProcessor._build_base_prompt_fallback("2026-10-01")
    assert base == MemoryProcessor._build_base_prompt_fallback("2026-10-02")
    assert MemoryProcessor._build_enhanced_prompt_fallback(
        base, "专业助手", "2026-10-01"
    ) == (
        MemoryProcessor._build_enhanced_prompt_fallback(base, "专业助手", "2026-10-02")
    )
