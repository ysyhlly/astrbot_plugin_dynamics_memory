"""Clean install/reopen with real SDK, SQLite and FAISS, no external model calls."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from astrbot.api.provider import LLMResponse
from astrbot.core.provider.provider import EmbeddingProvider, Provider
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.plugin_initializer import PluginInitializer


class Embeddings(EmbeddingProvider):
    def __init__(self):
        super().__init__({"id": "fixture-embedding", "type": "fixture"}, {})
        self.calls = 0

    def get_dim(self):
        return 8

    async def get_embedding(self, text):
        self.calls += 1
        vector = [0.01] * 8
        for index, char in enumerate(text):
            vector[index % 8] += (ord(char) % 31 + 1) / 31
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector]

    async def get_embeddings(self, texts):
        return [await self.get_embedding(text) for text in texts]


class TextModel(Provider):
    def __init__(self):
        super().__init__({"id": "fixture-text", "type": "fixture"}, {})
        self.calls = 0

    def get_current_key(self):
        return "fixture-only"

    def set_key(self, key):
        pass

    def get_model(self):
        return "fixture-model"

    async def get_models(self):
        return ["fixture-model"]

    async def text_chat(self, *args, **kwargs):
        self.calls += 1
        return LLMResponse(
            role="assistant",
            completion_text=json.dumps(
                {
                    "summary": "I remember the user prefers coffee while reading books.",
                    "topics": ["reading", "coffee"],
                    "key_facts": ["user prefers coffee"],
                    "importance": 0.8,
                    "sentiment": "positive",
                }
            ),
        )


async def close(initializer):
    await initializer.stop_background_tasks()
    await initializer.stop_scheduler()
    await initializer.shutdown_index_persisters()
    if initializer.conversation_manager:
        await initializer.conversation_manager.store.close()
    if initializer.memory_engine:
        await initializer.memory_engine.close()
    for db in (initializer.db, initializer.graph_db):
        if db:
            await db.close()


def setup(path, graph, *, with_providers=True):
    embedding, model = Embeddings(), TextModel()
    providers = {"fixture-embedding": embedding, "fixture-text": model}
    context = NS(
        get_provider_by_id=providers.get,
        provider_manager=NS(inst_map=providers),
        get_all_providers=lambda: [model],
        get_all_embedding_providers=lambda: [embedding],
        get_config=lambda: {},
        get_using_provider=lambda **kw: model,
        platform_manager=NS(get_insts=lambda: []),
    )
    config = ConfigManager(
        {
            "graph_memory": {"enabled": graph},
            "provider_settings": {
                "embedding_provider_id": "fixture-embedding",
                "llm_provider_id": "fixture-text",
            },
            "index_rebuild_settings": {
                "batch_delay": 0,
                "request_delay": 0,
                "retry_base_delay": 0,
            },
        }
    )
    instance = PluginInitializer(context, config, str(path))
    if with_providers:
        instance.embedding_provider = embedding
        instance.llm_provider = model
    return instance, embedding, model


@pytest.mark.asyncio
@pytest.mark.parametrize("graph", [False, True])
async def test_clean_initialize_store_reopen_and_shutdown(tmp_path, graph):
    instance, _embedding, model = setup(tmp_path, graph)
    try:
        await instance._complete_initialization()
        assert instance.is_initialized
        assert await instance.ensure_initialized(timeout=0.05)
        memory_id = await instance.memory_engine.add_memory(
            "coffee and books",
            session_id="fixture:GroupMessage:room",
            persona_id="reader",
            importance=0.8,
            metadata={"topics": ["coffee"], "participants": ["reader"]},
            source_messages=[
                {
                    "id": 1,
                    "session_id": "fixture:GroupMessage:room",
                    "role": "user",
                    "content": "coffee and books",
                    "sender_id": "u",
                    "timestamp": 1234,
                    "metadata": {},
                }
            ],
        )
        assert memory_id > 0
        assert (await instance.memory_engine.get_memory_source(memory_id))[0][
            "content"
        ] == "coffee and books"
        results = await instance.memory_engine.search_memories(
            "coffee",
            k=2,
            session_id="fixture:GroupMessage:room",
            persona_id="reader",
            touch_access=False,
        )
        assert memory_id in {row.doc_id for row in results}
        assert model.calls == 0
    finally:
        await close(instance)
    reopened, _, _ = setup(tmp_path, graph)
    try:
        await reopened._complete_initialization()
        assert (await reopened.memory_engine.get_memory(memory_id))[
            "text"
        ] == "coffee and books"
        assert (await reopened.memory_engine.get_memory_source(memory_id))[0][
            "sender_id"
        ] == "u"
    finally:
        await close(reopened)


@pytest.mark.asyncio
async def test_missing_explicit_provider_does_not_call_llm(tmp_path):
    instance, _, model = setup(tmp_path, False, with_providers=False)
    instance.context = NS(
        get_provider_by_id=lambda _id: None,
        provider_manager=NS(inst_map={}),
        get_all_providers=lambda: [],
        get_all_embedding_providers=lambda: [],
        get_config=lambda: {},
        platform_manager=NS(get_insts=lambda: []),
    )
    instance.config_manager = ConfigManager(
        {
            "provider_settings": {
                "llm_provider_id": "missing",
                "embedding_provider_id": "missing",
            }
        }
    )
    try:
        await instance._initialize_providers(silent=True)
        assert instance.llm_provider is None and instance.embedding_provider is None
        assert model.calls == 0
    finally:
        await close(instance)


@pytest.mark.asyncio
async def test_reopen_rebuilds_vector_index_when_fingerprint_changes(
    tmp_path, monkeypatch
):
    instance, _, _ = setup(tmp_path, True)
    try:
        await instance._complete_initialization()
        memory_id = await instance.memory_engine.add_memory(
            "coffee and books",
            session_id="fixture:GroupMessage:room",
            persona_id="reader",
            importance=0.9,
            metadata={"topics": ["coffee"], "participants": ["reader"]},
        )
    finally:
        await close(instance)
    for name in ("livingmemory.index", "livingmemory_graph.index"):
        path = tmp_path / name
        if path.exists():
            path.unlink()
    reopened, _, model = setup(tmp_path, True)
    reopened.config_manager = ConfigManager(
        {
            "graph_memory": {"enabled": True},
            "index_rebuild_settings": {
                "batch_delay": 0,
                "request_delay": 0,
                "retry_base_delay": 0,
            },
        }
    )
    try:
        await reopened._complete_initialization()
        if reopened._index_maintenance_task is not None:
            await asyncio.wait_for(asyncio.shield(reopened._index_maintenance_task), 15)
        monkeypatch.setattr(
            reopened.index_validator,
            "provider_fingerprint_changed",
            lambda: asyncio.sleep(0, result=True),
        )
        await reopened._run_index_maintenance()
        assert (await reopened.memory_engine.get_memory(memory_id))[
            "text"
        ] == "coffee and books"
        assert reopened.db.embedding_storage.index.ntotal == 1
        assert model.calls == 0
    finally:
        await close(reopened)
