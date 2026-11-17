from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from astrbot_plugin_dynamics_memory.core import utils
from astrbot_plugin_dynamics_memory.core.utils import (
    get_persona_id,
    resolve_persona_alias,
    set_persona_aliases,
)


@pytest.fixture(autouse=True)
def _reset_aliases():
    set_persona_aliases("")
    yield
    set_persona_aliases("")


def _context(conversation_persona=None, default_persona="default"):
    conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(
            return_value="cid" if conversation_persona else None
        ),
        get_conversation=AsyncMock(
            return_value=SimpleNamespace(persona_id=conversation_persona)
        ),
    )
    persona_manager = SimpleNamespace(
        get_default_persona_v3=AsyncMock(
            return_value={"name": default_persona} if default_persona else None
        ),
    )
    return SimpleNamespace(
        conversation_manager=conversation_manager, persona_manager=persona_manager
    )


def _event():
    return SimpleNamespace(unified_msg_origin="bot:FriendMessage:1")


def test_set_persona_aliases_parses_lines_and_lists():
    assert set_persona_aliases(
        "default=wb\n bad line \n=x\ny=\nsame=same\n a = b "
    ) == {
        "default": "wb",
        "a": "b",
    }
    assert set_persona_aliases(["default=wb"]) == {"default": "wb"}
    assert resolve_persona_alias("default") == "wb"
    assert resolve_persona_alias("other") == "other"
    assert resolve_persona_alias(None) is None


@pytest.mark.asyncio
async def test_get_persona_id_maps_default_persona_to_carrier():
    set_persona_aliases("default=wb_natsuume_anan")
    with patch.object(utils.sp, "get_async", new=AsyncMock(return_value={})):
        assert await get_persona_id(_context(), _event()) == "wb_natsuume_anan"
        assert await get_persona_id(_context("other"), _event()) == "other"
        assert await get_persona_id(_context("default"), _event()) == "wb_natsuume_anan"


@pytest.mark.asyncio
async def test_get_persona_id_maps_session_persona_and_keeps_explicit_none():
    set_persona_aliases("default=wb_natsuume_anan")
    with patch.object(
        utils.sp, "get_async", new=AsyncMock(return_value={"persona_id": "default"})
    ):
        assert await get_persona_id(_context(), _event()) == "wb_natsuume_anan"
    with patch.object(utils.sp, "get_async", new=AsyncMock(return_value={})):
        assert await get_persona_id(_context("[%None]"), _event()) is None


@pytest.mark.asyncio
async def test_get_persona_id_without_aliases_is_unchanged():
    with patch.object(utils.sp, "get_async", new=AsyncMock(return_value={})):
        assert await get_persona_id(_context(), _event()) == "default"
