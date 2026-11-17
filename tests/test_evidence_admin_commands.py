"""Admin commands enforce authorization and pass revision-bound corrections."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_dynamics_memory.main import LivingMemoryPlugin


@pytest.mark.asyncio
async def test_non_admin_cannot_list_or_correct_even_with_direct_method_call():
    owner = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    owner.list_evidence_insights = AsyncMock()
    owner.correct_insight = AsyncMock()
    event = NS(is_admin=lambda: False, plain_result=lambda text: text)
    for command in ("list", "atom|0|withdraw|reason"):
        replies = [r async for r in owner.evidence_admin(event, command)]
        assert len(replies) == 1 and "管理员" in replies[0]
    owner.list_evidence_insights.assert_not_awaited()
    owner.correct_insight.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_is_redacted_and_empty_list_is_actionable():
    owner = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    owner.list_evidence_insights = AsyncMock(return_value={"items": []})
    event = NS(is_admin=lambda: True, plain_result=lambda text: text)
    replies = [r async for r in owner.evidence_admin(event, "list")]
    assert "没有" in replies[0]
    owner.list_evidence_insights.return_value = {
        "items": [
            {
                "atom_id": "a",
                "kind": "reported_claim",
                "status": "active",
                "correction_revision": 2,
                "text": "DO NOT PRINT RAW SOURCE",
            }
        ]
    }
    replies = [r async for r in owner.evidence_admin(event, "list")]
    assert "a" in replies[0] and "DO NOT PRINT RAW SOURCE" not in replies[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["", "too|few", "id|not-a-number|withdraw|reason", "id|2|supersede|replacement"],
)
async def test_invalid_commands_never_write(command):
    owner = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    owner.correct_insight = AsyncMock(side_effect=ValueError("invalid"))
    event = NS(is_admin=lambda: True, plain_result=lambda text: text)
    replies = [r async for r in owner.evidence_admin(event, command)]
    assert len(replies) == 1
    if "not-a-number" in command or command.count("|") < 3:
        owner.correct_insight.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["withdraw", "supersede"])
async def test_correction_keeps_revision_and_reports_concurrent_conflict(action):
    owner = LivingMemoryPlugin.__new__(LivingMemoryPlugin)
    owner.correct_insight = AsyncMock(
        return_value={"action": action, "target_id": "atom", "correction_revision": 3}
    )
    event = NS(is_admin=lambda: True, plain_result=lambda text: text)
    command = (
        "atom|2|withdraw|admin reason"
        if action == "withdraw"
        else "atom|2|supersede|replacement claim|admin reason"
    )
    replies = [r async for r in owner.evidence_admin(event, command)]
    assert "revision=3" in replies[0]
    assert owner.correct_insight.await_args.args == ("atom", 2, action)
    owner.correct_insight.side_effect = RuntimeError("revision changed")
    replies = [r async for r in owner.evidence_admin(event, command)]
    assert "未写入" in replies[0]
