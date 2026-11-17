import hashlib
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock,Mock,patch
import pytest

from .test_event_handler import handler,memory_engine,memory_processor,conversation_manager,_make_event


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['rejected','timeout'])
async def test_rejected_summary_does_not_leave_a_live_source_validator(handler,memory_engine,failure):
    event=_make_event(group=False)
    req=NS(prompt='hello',system_prompt='',contexts=[],extra_user_content_parts=[])
    memory_engine.search_memories=AsyncMock(return_value=[Mock(doc_id=99,content='mem1',final_score=.7,
        metadata={'importance':.9,'session_id':'test:private:sid-1'})])
    async def reject():
        if failure=='timeout':raise TimeoutError()
        return False
    async def guard(event,req,sources,**kwargs):
        req._lm_private_summary_sources_v1={'segment_sha256':hashlib.sha256(kwargs['segment_text'].encode()).hexdigest()}
        req._lm_private_summary_validate_v1=reject
        return True
    handler._memory_recall.summary_guard=guard
    with patch('astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall.get_persona_id',new_callable=AsyncMock,return_value='persona_1'):
        await handler.handle_memory_recall(event,req)
    assert not req.extra_user_content_parts
    assert getattr(req,'_lm_private_summary_sources_v1',None) is None
    assert getattr(req,'_lm_private_summary_validate_v1',None) is None
    assert not getattr(req,'_lm_private_segment_refs_v1',{})


def test_cleanup_restores_prior_guard_and_preserves_other_sources():
    from astrbot_plugin_dynamics_memory.core.recall_runtime import discard_private_summary_segment
    old,other,rejected=NS(text='approved'),NS(text='other source'),NS(text='unconfirmed')
    old_sources={'segment_sha256':hashlib.sha256(old.text.encode()).hexdigest()}
    async def valid():return True
    async def invalid():return False
    native={'verified':'unchanged'}
    req=NS(extra_user_content_parts=[old,other,rejected],_lm_native_memory_v1=native,
        _lm_private_segment_refs_v1={id(old):{'source_level':'legacy_summary'},id(other):{'source_level':'verified_quote'},id(rejected):{'source_level':'legacy_summary'}},
        _lm_private_summary_sources_v1={'segment_sha256':hashlib.sha256(rejected.text.encode()).hexdigest()},
        _lm_private_summary_validate_v1=invalid)
    discard_private_summary_segment(req,rejected,previous_sources=old_sources,previous_validator=valid)
    assert req.extra_user_content_parts==[old,other]
    assert req._lm_private_summary_sources_v1 is old_sources
    assert req._lm_private_summary_validate_v1 is valid
    assert req._lm_native_memory_v1 is native
    assert set(req._lm_private_segment_refs_v1)=={id(old),id(other)}


def test_cleanup_cannot_clear_a_different_summary_receipt():
    from astrbot_plugin_dynamics_memory.core.recall_runtime import discard_private_summary_segment
    rejected,kept=NS(text='unconfirmed'),NS(text='selected source')
    source={'segment_sha256':hashlib.sha256(kept.text.encode()).hexdigest()}
    async def validator():return True
    req=NS(extra_user_content_parts=[rejected,kept],_lm_private_summary_sources_v1=source,
        _lm_private_summary_validate_v1=validator)
    discard_private_summary_segment(req,rejected)
    assert req.extra_user_content_parts==[kept]
    assert req._lm_private_summary_sources_v1 is source
    assert req._lm_private_summary_validate_v1 is validator
