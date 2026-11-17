"""Contract tests for retained compatibility helpers using real document rows."""

import copy
import time
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.managers.memory_engine_crud import (
    MemoryEngineCrudMixin,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine_recall import (
    MemoryEngineRecallMixin,
)
from astrbot_plugin_dynamics_memory.core.models.memory_atom import AtomType, MemoryAtom
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import HybridResult
from tests.integration.test_real_db_end_to_end import real_db_stack as real_db_stack


def policy_owner(engine):
    owner = NS(config=dict(engine.config), db_connection=engine.db_connection)
    for name in (
        "_filter_by_retrieval_policy",
        "_get_recent_memory_results",
        "_merge_recent_memories",
    ):
        setattr(owner, name, MethodType(getattr(MemoryEngineRecallMixin, name), owner))
    return owner


@pytest.mark.asyncio
async def test_recent_policy_keeps_scope_persona_visibility_and_uniqueness(
    real_db_stack,
):
    engine = real_db_stack["memory_engine"]
    wanted = await engine.add_memory(
        "recent group source",
        session_id="fixture:GroupMessage:a",
        persona_id="p",
        importance=0.8,
    )
    second = await engine.add_memory(
        "second group source",
        session_id="fixture:GroupMessage:a",
        persona_id="p",
        importance=0.8,
    )
    await engine.add_memory(
        "other group",
        session_id="fixture:GroupMessage:b",
        persona_id="p",
        importance=0.8,
    )
    await engine.add_memory(
        "private source",
        session_id="fixture:FriendMessage:u",
        persona_id="p",
        importance=0.8,
    )
    owner = policy_owner(engine)
    owner.config.update(
        recent_memory_count=2,
        min_importance_for_retrieval=0,
        min_similarity_for_retrieval=0,
    )
    rows = await owner._get_recent_memory_results(
        5,
        "fixture:GroupMessage:a",
        "p",
        interaction_type="group_chat",
        source_platform="fixture",
        source_session_id="fixture:GroupMessage:a",
    )
    assert {r.doc_id for r in rows} == {wanted, second}
    merged = await owner._merge_recent_memories(
        rows[:1],
        2,
        "fixture:GroupMessage:a",
        "p",
        interaction_type="group_chat",
        source_platform="fixture",
        source_session_id="fixture:GroupMessage:a",
    )
    assert len(merged) == 2
    assert {r.doc_id for r in merged} == {wanted, second}
    assert await owner._get_recent_memory_results(0, None, None) == []
    assert (
        await owner._get_recent_memory_results(5, "fixture:GroupMessage:a", "other")
        == []
    )
    owner.config["recent_memory_count"] = 0
    assert await owner._merge_recent_memories(rows[:1], 1, None, None) == rows[:1]
    owner.config.update(recent_memory_count=2, recent_memory_max_age_hours=0)
    assert await owner._merge_recent_memories(rows[:1], 2, "missing", "p") == rows[:1]


@pytest.mark.parametrize(
    "metadata,score,accepted",
    [
        ({"status": "archived", "importance": 1}, 1, False),
        ({"importance": 0.1}, 1, False),
        ({"importance": 0.8}, 0.1, False),
        ({"importance": 0.8, "atom_types": ["opinion"]}, 0.9, False),
        ({"importance": 0.8, "atom_types": ["factual"]}, 0.9, True),
        ({"importance": 0.8}, None, True),
    ],
)
def test_policy_thresholds_do_not_promote_archived_or_irrelevant_memory(
    metadata, score, accepted
):
    owner = NS(
        config={
            "min_importance_for_retrieval": 0.5,
            "min_similarity_for_retrieval": 0.5,
            "memory_type_filter": "event_only",
        }
    )
    row = HybridResult(1, 0.8, 0.1, None, score, "source", metadata)
    before = copy.deepcopy(metadata)
    result = MemoryEngineRecallMixin._filter_by_retrieval_policy(owner, [row])
    assert bool(result) == accepted and metadata == before


@pytest.mark.parametrize(
    "create_time,preserve", [(1234, True), ("invalid", True), (1234, False)]
)
def test_metadata_builder_preserves_requested_history_position(create_time, preserve):
    old = {"create_time": create_time, "custom": "retained"}
    atom = MemoryAtom(parent_memory_id=0, atom_type=AtomType.FACTUAL, content="fact")
    before = time.time()
    metadata = MemoryEngineCrudMixin._build_memory_metadata(
        NS(),
        session_id="scope",
        persona_id="persona",
        importance=2,
        metadata=old,
        atoms=[atom, atom],
        preserve_create_time=preserve,
        source_messages=[{"content": "source"}],
    )
    assert metadata["importance"] == 1 and metadata["custom"] == "retained"
    assert metadata["has_source"] and metadata["source_message_count"] == 1
    assert metadata["atom_types"] == ["factual"]
    assert (
        metadata["create_time"] == 1234
        if preserve and isinstance(create_time, int)
        else metadata["create_time"] >= before
    )
    assert old == {"create_time": create_time, "custom": "retained"}


@pytest.mark.asyncio
async def test_source_persist_failure_rolls_back_new_document(real_db_stack):
    engine = real_db_stack["memory_engine"]
    memory = await engine.add_memory(
        "must not survive missing source",
        session_id="fixture:GroupMessage:a",
        persona_id="p",
    )
    engine.save_memory_source = AsyncMock(side_effect=OSError("fixture disk full"))
    with pytest.raises(OSError):
        await engine._persist_memory_source(
            memory, [{"role": "user", "content": "source"}]
        )
    assert await engine.get_memory(memory) is None


@pytest.mark.asyncio
async def test_atom_append_is_idempotent_and_preserves_owner(real_db_stack):
    engine = real_db_stack["memory_engine"]
    memory = await engine.add_memory(
        "source", session_id="fixture:GroupMessage:a", persona_id="p"
    )
    metadata = {
        "session_id": "fixture:GroupMessage:a",
        "persona_id": "p",
        "adapter_id": "fixture",
    }
    atoms = [
        MemoryAtom(parent_memory_id=0, atom_type=AtomType.FACTUAL, content="one fact"),
        MemoryAtom(parent_memory_id=0, atom_type=AtomType.FACTUAL, content="one fact"),
        MemoryAtom(parent_memory_id=0, atom_type=AtomType.FACTUAL, content=""),
    ]
    await engine._append_atoms_to_memory(memory, atoms, metadata)
    await engine._append_atoms_to_memory(memory, atoms, metadata)
    rows = await (
        await engine.db_connection.execute(
            "SELECT content,session_id,persona_id,metadata FROM memory_atoms WHERE parent_memory_id=?",
            (memory,),
        )
    ).fetchall()
    assert len(rows) == 1 and rows[0][0] == "one fact"
    assert rows[0][1:3] == ("fixture:GroupMessage:a", "p")
    another = MemoryAtom(
        parent_memory_id=0, atom_type=AtomType.FACTUAL, content="second fact"
    )
    await engine._write_memory_atoms(
        memory, metadata["session_id"], "p", [another], adapter_id="fixture"
    )
    assert (
        another.parent_memory_id == memory
        and another.metadata["adapter_id"] == "fixture"
    )
