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
    assert "# 输出格式" in calls[0]["system_prompt"]
    assert "# 示例" in calls[0]["system_prompt"]
    assert "# 本次请求" not in calls[0]["system_prompt"]
    first_prompt, second_prompt = (call["prompt"] for call in calls)
    assert first_prompt.startswith("# 对话历史")
    assert second_prompt.startswith("# 对话历史")
    assert "# 输出格式" not in first_prompt and "# 示例" not in first_prompt
    assert "2026-10-01 12:00" in first_prompt
    assert "2026-10-02 12:00" in second_prompt
    assert "2026-09-20 23:50" in first_prompt
    # DM-1 / Phase 5: static template lives in system_prompt and {current_date} sits
    # at the end after the conversation block so prefixes stay byte-identical.
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
    assert processor._current_date_str()[:10] in system_prompt
    assert "{current_date}" not in system_prompt


def test_system_prompt_fallbacks_are_static():
    base = MemoryProcessor._build_base_prompt_fallback("2026-10-01")
    assert base == MemoryProcessor._build_base_prompt_fallback("2026-10-02")
    assert MemoryProcessor._build_enhanced_prompt_fallback(
        base, "专业助手", "2026-10-01"
    ) == (
        MemoryProcessor._build_enhanced_prompt_fallback(base, "专业助手", "2026-10-02")
    )


@pytest.mark.asyncio
async def test_phase4_cache_anchor_contexts_across_prompts_streams_and_strict_stubs():
    """Phase 4: deterministic 2-message static contexts anchor keyed on system_prompt hash."""
    import hashlib

    recorded: list[dict] = []

    class ContextAwareProvider:
        async def text_chat(self, prompt=None, system_prompt=None, contexts=None):
            recorded.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "contexts": contexts,
                }
            )
            return SimpleNamespace(completion_text='{"summary":"ok"}')

        async def text_chat_stream(
            self, prompt=None, system_prompt=None, contexts=None
        ):
            recorded.append(
                {
                    "stream": True,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "contexts": contexts,
                }
            )
            yield SimpleNamespace(is_chunk=False, completion_text='{"summary":"ok"}')

    proc = MemoryProcessor(llm_provider=ContextAwareProvider())
    sys_a = "固定系统提示词 A"
    sys_b = "固定系统提示词 B"
    sha_a = hashlib.sha256(sys_a.encode("utf-8")).hexdigest()[:12]
    sha_b = hashlib.sha256(sys_b.encode("utf-8")).hexdigest()[:12]

    await proc._call_llm_with_retry("动态对话 1", sys_a)
    await proc._call_llm_with_retry("动态对话 2", sys_a)
    await proc._call_llm_with_retry("动态对话 3", sys_b)
    await proc._call_llm_with_retry("动态对话 4", "")

    expected_a = [
        {
            "role": "user",
            "content": f"[cache_anchor task=dynamics_memory prefix={sha_a}]",
        },
        {"role": "assistant", "content": "OK"},
    ]
    expected_b = [
        {
            "role": "user",
            "content": f"[cache_anchor task=dynamics_memory prefix={sha_b}]",
        },
        {"role": "assistant", "content": "OK"},
    ]
    assert recorded[0]["contexts"] == expected_a
    assert recorded[1]["contexts"] == expected_a
    assert recorded[2]["contexts"] == expected_b
    assert recorded[3]["contexts"] is None

    # Stream mode also passes the exact same static 2-message anchor
    stream_proc = MemoryProcessor(
        llm_provider=ContextAwareProvider(), config={"llm_stream_mode": "always"}
    )
    await stream_proc._call_llm_with_retry("流式对话", sys_a)
    assert recorded[4]["stream"] is True
    assert recorded[4]["contexts"] == expected_a

    # Strict mock without `contexts` parameter works without TypeError
    strict_calls: list[tuple[str, str]] = []

    class StrictStubProvider:
        async def text_chat(self, prompt, system_prompt):
            strict_calls.append((prompt, system_prompt))
            return SimpleNamespace(completion_text='{"summary":"strict"}')

    strict_proc = MemoryProcessor(llm_provider=StrictStubProvider())
    assert await strict_proc._call_llm_with_retry("p", sys_a) == '{"summary":"strict"}'
    assert strict_calls == [("p", sys_a)]


@pytest.mark.asyncio
async def test_phase4_evidence_derivation_passes_static_cache_anchor_and_handles_strict_stub():
    """Phase 4: PluginHelpersMixin._derive_evidence_materials attaches static anchor safely."""
    import hashlib

    from astrbot_plugin_dynamics_memory.core.plugin_helpers_mixin import (
        PluginHelpersMixin,
    )

    seen_contexts: list[tuple[str, list | None]] = []

    class ContextAwareProvider:
        async def text_chat(self, prompt=None, system_prompt=None, contexts=None):
            seen_contexts.append((system_prompt, contexts))
            payload = json.loads(prompt)
            out = {
                "materials": [
                    {"material_id": m["material_id"], "atoms": []}
                    for m in payload["materials"]
                ]
            }
            return SimpleNamespace(completion_text=json.dumps(out))

    class DummyPlugin(PluginHelpersMixin):
        def __init__(self, provider):
            self.initializer = SimpleNamespace(
                is_initialized=True,
                _providers_ready=True,
                memory_processor=SimpleNamespace(
                    _get_current_llm_provider=lambda: provider
                ),
            )

    plugin = DummyPlugin(ContextAwareProvider())
    mat1 = [
        {
            "material_id": "m1",
            "source_status": "recorded",
            "role": "user",
            "text": "第一段原始对话",
        }
    ]
    mat2 = [
        {
            "material_id": "m2",
            "source_status": "recorded",
            "role": "user",
            "text": "第二段完全不同的原始对话",
        }
    ]
    await plugin._derive_evidence_materials(mat1)
    await plugin._derive_evidence_materials(mat2)

    assert len(seen_contexts) == 2
    assert seen_contexts[0][0] == seen_contexts[1][0]
    assert seen_contexts[0][1] == seen_contexts[1][1]
    sha12 = hashlib.sha256(seen_contexts[0][0].encode("utf-8")).hexdigest()[:12]
    assert seen_contexts[0][1] == [
        {
            "role": "user",
            "content": f"[cache_anchor task=dynamics_memory prefix={sha12}]",
        },
        {"role": "assistant", "content": "OK"},
    ]

    # Strict stub without `contexts` parameter succeeds without TypeError
    class StrictStubProvider:
        async def text_chat(self, prompt, system_prompt):
            return SimpleNamespace(
                completion_text='{"materials":[{"material_id":"m1","atoms":[]}]}'
            )

    strict_plugin = DummyPlugin(StrictStubProvider())
    assert await strict_plugin._derive_evidence_materials(mat1) == {"m1": []}


@pytest.mark.asyncio
async def test_phase5_reflection_static_prefix_in_system_prompt_across_groups_and_retries(
    monkeypatch, tmp_path
):
    """Phase 5: static reflection template moves into system_prompt; byte-identical across groups & retries."""
    manager = prompt_manager.PromptManager(str(tmp_path))
    monkeypatch.setattr(prompt_manager, "get_prompt_manager", lambda: manager)

    recorded: list[dict] = []

    class ContextAwareProvider:
        async def text_chat(self, prompt=None, system_prompt=None, contexts=None):
            recorded.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "contexts": contexts,
                }
            )
            return SimpleNamespace(
                completion_text=(
                    '{"summary":"张三和李四确认了下周排期计划","key_facts":["张三确认排期"],'
                    '"topics":["排期"],"participants":["张三","李四"],"sentiment":"neutral","importance":0.6}'
                )
            )

    proc = MemoryProcessor(
        context=SimpleNamespace(
            persona_manager=SimpleNamespace(
                get_persona=AsyncMock(
                    return_value=SimpleNamespace(system_prompt="你是夏目安安" * 50)
                )
            )
        ),
        llm_provider=ContextAwareProvider(),
    )

    group1_msgs = [
        Message(
            id=1,
            session_id="aiocqhttp:GroupMessage:10001",
            role="user",
            content="第一组讨论周三发版",
            sender_id="u1",
            sender_name="张三",
            group_id="10001",
            platform="aiocqhttp",
            timestamp=datetime(2026, 11, 21, 9, 0).timestamp(),
            metadata={"source_event_id": "evt_g1"},
        )
    ]
    group2_msgs = [
        Message(
            id=2,
            session_id="aiocqhttp:GroupMessage:20002",
            role="user",
            content="第二组讨论周五团建滑雪",
            sender_id="u2",
            sender_name="李四",
            group_id="20002",
            platform="aiocqhttp",
            timestamp=datetime(2026, 11, 21, 9, 15).timestamp(),
            metadata={},
        )
    ]

    class T1(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 21, 9, 30)

    class T2(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 21, 9, 45)

    monkeypatch.setattr(processor_module, "datetime", T1)
    await proc.process_conversation(group1_msgs, is_group_chat=True, persona_id="anan")

    monkeypatch.setattr(processor_module, "datetime", T2)
    await proc.process_conversation(group2_msgs, is_group_chat=True, persona_id="anan")
    await proc.process_conversation(
        group1_msgs,
        is_group_chat=True,
        persona_id="anan",
        quality_feedback="summary_generic_reference",
    )

    assert len(recorded) == 3
    g1_call, g2_call, g1_retry_call = recorded

    # system_prompt and contexts[0..1] are 100% byte-identical across groups & retries
    assert g1_call["system_prompt"] == g2_call["system_prompt"]
    assert g1_call["system_prompt"] == g1_retry_call["system_prompt"]
    assert g1_call["contexts"] == g2_call["contexts"]
    assert g1_call["contexts"] == g1_retry_call["contexts"]
    assert g1_call["contexts"] is not None

    # Static reflection template sections live in system_prompt (not in user prompt)
    sys_p = g1_call["system_prompt"]
    for marker in ("# 任务", "# 消息格式", "# 规则", "# 输出格式", "# 示例"):
        assert marker in sys_p
        assert marker not in g1_call["prompt"]
        assert marker not in g2_call["prompt"]
        assert marker not in g1_retry_call["prompt"]
    assert "# 本次请求" not in sys_p
    assert "# 对话历史" not in sys_p
    assert "第一组讨论周三发版" not in sys_p
    assert "2026-11-21" not in sys_p
    assert "上次摘要" not in sys_p

    # user_prompt starts with # 对话历史 and places dynamic date + quality_feedback at tail
    assert g1_call["prompt"].startswith("# 对话历史\n")
    assert g2_call["prompt"].startswith("# 对话历史\n")
    assert g1_retry_call["prompt"].startswith("# 对话历史\n")
    assert "来源约束" in g1_call["prompt"]
    assert "当前日期时间: 2026-11-21 09:30" in g1_call["prompt"]
    assert "当前日期时间: 2026-11-21 09:45" in g1_retry_call["prompt"]
    assert g1_retry_call["prompt"].index("2026-11-21 09:45") < g1_retry_call[
        "prompt"
    ].index("上次摘要")


@pytest.mark.asyncio
async def test_phase5_custom_templates_fall_back_cleanly(processor, monkeypatch):
    """Phase 5: custom templates without # 对话历史 or with inline date in prefix fall back safely."""
    manager = prompt_manager.get_prompt_manager()
    messages = [
        Message(
            id=1,
            session_id="s1",
            role="user",
            content="自定义模板测试消息内容",
            sender_id="u1",
            sender_name="张三",
            group_id=None,
            platform="test",
            timestamp=datetime(2026, 11, 21, 8, 0).timestamp(),
            metadata={},
        )
    ]

    class Frozen(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 21, 9, 50)

    monkeypatch.setattr(processor_module, "datetime", Frozen)
    base_sys = await processor._build_system_prompt_with_persona(None)

    # 1. Custom template without `# 对话历史` marker
    manager.update_prompt(
        "private_chat_prompt",
        "自定义规则：直接总结如下对话。\n{conversation}\n当前日期时间: {current_date}",
    )
    await processor.process_conversation(messages, is_group_chat=False, persona_id=None)
    call_no_marker = dict(processor._llm_provider.text_chat.call_args.kwargs)
    assert call_no_marker["system_prompt"] == base_sys
    assert "自定义规则：直接总结如下对话。" in call_no_marker["prompt"]
    assert "自定义模板测试消息内容" in call_no_marker["prompt"]
    assert "2026-11-21 09:50" in call_no_marker["prompt"]
    assert "2026-11-21 09:50" not in call_no_marker["system_prompt"]

    # 2. Custom template with inline `{current_date}` in prefix before `# 对话历史`
    manager.update_prompt(
        "private_chat_prompt",
        "今天是 {current_date}，请按规则总结。\n# 对话历史\n{conversation}\n",
    )
    await processor.process_conversation(messages, is_group_chat=False, persona_id=None)
    call_inline_date = dict(processor._llm_provider.text_chat.call_args.kwargs)
    assert call_inline_date["system_prompt"] == base_sys
    assert "2026-11-21 09:50" not in call_inline_date["system_prompt"]
    assert "今天是 2026-11-21 09:50" in call_inline_date["prompt"]
    assert "自定义模板测试消息内容" in call_inline_date["prompt"]


@pytest.mark.asyncio
async def test_phase5_merge_memories_and_evidence_derivation_keep_schema_in_system_prompt():
    """Phase 5: merge_memories and _derive_evidence_materials keep static JSON schema in system_prompt."""
    from astrbot_plugin_dynamics_memory.core.plugin_helpers_mixin import (
        PluginHelpersMixin,
    )

    merge_calls: list[dict] = []

    class MergeProvider:
        async def text_chat(self, prompt=None, system_prompt=None, contexts=None):
            merge_calls.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "contexts": contexts,
                }
            )
            return SimpleNamespace(
                completion_text='{"summary":"合并后的精炼记忆内容","key_facts":["事实A"],"topics":["主题"],"importance":0.6}'
            )

    proc = MemoryProcessor(llm_provider=MergeProvider())
    await proc.merge_memories([{"content": "记忆一", "metadata": {}}])
    await proc.merge_memories(
        [{"content": "记忆二", "metadata": {}}, {"content": "记忆三", "metadata": {}}]
    )

    assert len(merge_calls) == 2
    assert merge_calls[0]["system_prompt"] == merge_calls[1]["system_prompt"]
    assert merge_calls[0]["contexts"] == merge_calls[1]["contexts"]
    assert '"key_facts"' in merge_calls[0]["system_prompt"]
    assert '"importance"' in merge_calls[0]["system_prompt"]
    assert "按如下 JSON 格式输出" not in merge_calls[0]["prompt"]
    assert "记忆一" in merge_calls[0]["prompt"]
    assert "记忆三" in merge_calls[1]["prompt"]

    evidence_calls: list[dict] = []

    class EvidenceProvider:
        async def text_chat(self, prompt=None, system_prompt=None, contexts=None):
            evidence_calls.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "contexts": contexts,
                }
            )
            payload = json.loads(prompt)
            return SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "materials": [
                            {"material_id": m["material_id"], "atoms": []}
                            for m in payload["materials"]
                        ]
                    }
                )
            )

    class DummyPlugin(PluginHelpersMixin):
        def __init__(self, provider):
            self.initializer = SimpleNamespace(
                is_initialized=True,
                _providers_ready=True,
                memory_processor=SimpleNamespace(
                    _get_current_llm_provider=lambda: provider
                ),
            )

    plugin = DummyPlugin(EvidenceProvider())
    await plugin._derive_evidence_materials(
        [
            {
                "material_id": "m1",
                "source_status": "recorded",
                "role": "user",
                "text": "A",
            }
        ]
    )
    await plugin._derive_evidence_materials(
        [
            {
                "material_id": "m2",
                "source_status": "recorded",
                "role": "user",
                "text": "B",
            }
        ]
    )

    assert len(evidence_calls) == 2
    assert evidence_calls[0]["system_prompt"] == evidence_calls[1]["system_prompt"]
    assert evidence_calls[0]["contexts"] == evidence_calls[1]["contexts"]
    assert "required_output" in evidence_calls[0]["system_prompt"]
    assert "utterance_fact|reported_claim" in evidence_calls[0]["system_prompt"]
    parsed_user = json.loads(evidence_calls[0]["prompt"])
    assert set(parsed_user.keys()) == {"schema_version", "materials"}
    assert "required_output" not in parsed_user
