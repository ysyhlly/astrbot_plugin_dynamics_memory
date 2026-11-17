import asyncio
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from astrbot.api.platform import MessageType
from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB
from astrbot_plugin_dynamics_memory.core.managers.conversation_manager import ConversationManager
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.series import SeriesMixin
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore

from .test_real_db_end_to_end import _DeterministicEmbeddingProvider


class ActivityMemory(SeriesMixin):
    async def _memory_scope_allowed(self, scope, actor_id=""):
        return actor_id != "blocked"


class GroupEvent:
    def __init__(self, umo, uid, message_id, timestamp):
        self.unified_msg_origin = umo
        self.uid = uid
        self.message_obj = SimpleNamespace(message_id=message_id, timestamp=timestamp)

    def get_sender_id(self):
        return self.uid

    def get_sender_name(self):
        return self.uid

    def get_message_type(self):
        return MessageType.GROUP_MESSAGE

    def get_platform_name(self):
        return "aiocqhttp"


@pytest.mark.asyncio
async def test_games_experience_persists_original_evidence_and_deduplicates(tmp_path):
    db_path = str(tmp_path / "memory.sqlite3")
    store = ConversationStore(str(tmp_path / "conversation.sqlite3"))
    await store.initialize()
    manager = ConversationManager(store=store)
    faiss = FaissVecDB(
        doc_store_path=db_path, index_store_path=str(tmp_path / "memory.index"),
        embedding_provider=_DeterministicEmbeddingProvider(),
    )
    await faiss.initialize()
    engine = MemoryEngine(db_path, faiss, config={"graph_memory_enabled": False})
    await engine.initialize()
    initialized = SimpleNamespace(is_initialized=True, memory_engine=engine,
                                  conversation_manager=manager)
    owner = ActivityMemory()
    owner.initializer = initialized

    umo = "qq-alt:GroupMessage:10"
    other_umo = "qq-other:GroupMessage:10"
    platform = "qq-alt"
    event_scope = {"platform": platform, "umo": umo, "persona_id": "a", "instance_id": "a"}
    timestamp = time.time() - 100
    native = await manager.add_message_from_event(
        GroupEvent(umo, "a", 12121, timestamp), "user", "完整平台原文"
    )
    await manager.add_message_from_event(
        GroupEvent(other_umo, "a", 99999, timestamp), "user", "另一个实例"
    )
    envelope_id = "in:" + hashlib.sha256(f"{umo}|34343".encode()).hexdigest()
    assert await store.ingest_narrative_event({
        "schema_version": 1, "type": "message.received", "event_id": envelope_id,
        "scope": event_scope, "actor_id": "b", "text": "完整托管原文",
        "occurred_at": timestamp + 6, "source_ids": ["34343"],
    })
    occurred_at = timestamp + 10
    refs = [
        {"message_id": "12121", "uid": "a", "ts": timestamp + 1, "excerpt": "任意片段"},
        {"message_id": "34343", "uid": "b", "ts": timestamp + 7},
    ]
    payload = {
        "schema_version": 1, "scope": {"umo": umo, "platform": platform},
        "activity_id": "G101", "title": "读书会", "summary": "已完成共读",
        "occurred_at": occurred_at, "participants": [{"uid": "a"}, {"uid": "b"}],
        "source_refs": refs,
    }
    try:
        receipts = await asyncio.gather(*(owner.record_activity_experience(payload) for _ in range(2)))
        first = next(receipt for receipt in receipts if receipt["recorded"])
        duplicate = next(receipt for receipt in receipts if not receipt["recorded"])
        doc_id = first["doc_id"]
        assert duplicate == {"schema_version": 1, "recorded": False, "reason": "duplicate", "doc_id": doc_id}
        result = await engine.get_memory(doc_id)
        metadata = result["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert metadata["status"] == "active"
        assert metadata["create_time"] == metadata["occurred_at"] == occurred_at
        sources = await engine.get_memory_source(doc_id)
        assert [source["content"] for source in sources] == ["完整平台原文", "完整托管原文"]
        assert [source["timestamp"] for source in sources] == [timestamp, timestamp + 6]
        assert sources[0]["id"] == native.id
        assert sources[1]["metadata"]["source_event_id"] == envelope_id
        assert [source["metadata"]["platform_message_id"] for source in sources] == ["12121", "34343"]

        # The replacement facade has no process-local receipt cache.
        resumed = ActivityMemory()
        resumed.initializer = initialized
        assert await resumed.record_activity_experience(payload) == duplicate
        wrong = [
            ("source_unresolved", {"source_refs": [{**refs[0], "message_id": "99999"}]}),
            ("source_unresolved", {"source_refs": [{**refs[0], "message_id": str(native.id)}]}),
            ("source_identity_mismatch", {"source_refs": [{**refs[0], "uid": "b"}]}),
            ("source_identity_mismatch", {"source_refs": [{**refs[0], "ts": timestamp - 600}]}),
            ("scope_not_allowed", {"participants": [{"uid": "a"}, {"uid": "b"}, {"uid": "blocked"}]}),
            ("invalid_occurred_at", {"occurred_at": float("nan")}),
        ]
        for index, (reason, changes) in enumerate(wrong):
            rejected = await resumed.record_activity_experience({
                **payload, "activity_id": f"G{102 + index}", **changes,
            })
            assert rejected == {"schema_version": 1, "recorded": False, "reason": reason}
        assert await store.delete_session_messages(umo) == 1
        assert await resumed.record_activity_experience(payload) == duplicate
        assert faiss.embedding_storage.index.ntotal == 1
    finally:
        await engine.close()
        await faiss.close()
        await store.close()
