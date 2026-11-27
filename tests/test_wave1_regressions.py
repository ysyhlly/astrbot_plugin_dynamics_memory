"""Wave 1 regression tests for Dynamics Memory.

Covers:
1. WebUI config update 500 crash fix (DualRouteRetriever.graph_route_timeout hot-refresh)
2. Graph route timeout default tuning (3.5s), graceful fallback, and unsummarized high-watermark summary
3. /lmem forget safe soft-archive + --purge and /lmem help i18n completeness
4. Series scope writable contract (set_series_scope_enabled with optimistic lock + activate_series_runtime)
5. Shared config inheritance (embedding, rerank, light_analysis, persona, private_shared_group_sources str/list)
6. Cross-plugin privacy mute (collection & recall filtering via zero-coupling dynamic lookup)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import EmbeddingProvider, Provider
from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.event_handler_modules.group_capture import (
    GroupCapture,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_recall import (
    MemoryRecall,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.memory_reflection import (
    MemoryReflection,
)
from astrbot_plugin_dynamics_memory.core.event_handler_modules.message_utils import (
    MessageUtils,
)
from astrbot_plugin_dynamics_memory.core.managers.memory_engine import MemoryEngine
from astrbot_plugin_dynamics_memory.core.memory_scope import (
    is_event_memory_allowed,
    is_event_privacy_muted,
    is_member_privacy_muted,
    resolve_memory_access_policy,
    resolve_verified_group_sessions,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.settings_handler import (
    SettingsHandler,
)
from astrbot_plugin_dynamics_memory.core.page_api_modules.utils import PageApiUtils
from astrbot_plugin_dynamics_memory.core.plugin_initializer import PluginInitializer
from astrbot_plugin_dynamics_memory.core.processors.memory_processor import (
    MemoryProcessor,
)
from astrbot_plugin_dynamics_memory.core.retrieval.dual_route_retriever import (
    DualRouteRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.hybrid_retriever import (
    HybridResult,
    HybridRetriever,
)
from astrbot_plugin_dynamics_memory.core.retrieval.route_execution import (
    _failure_log_state,
    search_route,
)
from astrbot_plugin_dynamics_memory.core.retrieval.rrf_fusion import RRFFusion
from astrbot_plugin_dynamics_memory.core.series import SeriesMixin
from astrbot_plugin_dynamics_memory.core.utils import _get_native_persona_id
from astrbot_plugin_dynamics_memory.core.utils.injection_adapter import InjectionAdapter
from quart import Quart


class _WritableConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved = 0

    def save_config(self):
        self.saved += 1


def _make_chat_dynamics_ctx(
    *,
    shared_config: dict | None = None,
    muted_map: dict[tuple[str, str], bool] | None = None,
    providers: dict | None = None,
    embedding_providers: list | None = None,
):
    muted_lookup = dict(muted_map or {})
    shared = dict(shared_config or {})

    class _ChatOwner:
        def get_series_shared_config(self):
            return dict(shared)

        def get_series_privacy_muted(self, umo: str, member_id: str, scope: str = ""):
            muted = bool(muted_lookup.get((umo, member_id), False))
            return {
                "muted": muted,
                "scopes": ["global"] if muted else [],
                "updated_at": "2026-11-27T00:00:00Z",
                "by": "member" if muted else "",
            }

    owner = _ChatOwner()
    provider_map = dict(providers or {})
    emb_list = list(embedding_providers or [])
    return SimpleNamespace(
        get_all_stars=lambda: {
            "chat": SimpleNamespace(
                name="astrbot_plugin_chat_dynamics",
                activated=True,
                star_cls=owner,
            )
        },
        get_provider_by_id=lambda pid: provider_map.get(pid),
        get_all_providers=lambda: list(provider_map.values()),
        get_all_embedding_providers=lambda: emb_list,
        get_using_provider=lambda *_a, **_kw: None,
        get_config=lambda: {"platform": []},
        platform_manager=SimpleNamespace(get_insts=lambda: []),
    )


# ==================== Task 1: Config Save 500 Hot-Refresh Regression ====================


@pytest.mark.asyncio
async def test_settings_update_hot_refreshes_dual_route_timeout_without_500(tmp_path):
    raw = _WritableConfig({})
    config_manager = ConfigManager(raw)

    engine = MemoryEngine(str(tmp_path / "m.db"), faiss_db=Mock(), config={})
    engine.rrf_fusion = RRFFusion(k=60)
    engine.hybrid_retriever = HybridRetriever(
        bm25_retriever=Mock(),
        vector_retriever=Mock(),
        rrf_fusion=engine.rrf_fusion,
        config={},
    )
    engine.dual_route_retriever = DualRouteRetriever(
        document_retriever=engine.hybrid_retriever,
        graph_retriever=Mock(),
        memory_loader=AsyncMock(return_value=None),
        config={},
    )
    assert engine.dual_route_retriever.graph_route_timeout == pytest.approx(3.5)

    plugin = SimpleNamespace(
        config_manager=config_manager,
        initializer=SimpleNamespace(
            memory_engine=engine,
            memory_processor=SimpleNamespace(config={}),
            is_initialized=True,
        ),
        context=SimpleNamespace(
            get_config=lambda: {"platform": []},
            platform_manager=SimpleNamespace(get_insts=lambda: []),
        ),
    )
    handler = SettingsHandler(PageApiUtils(), plugin)

    app = Quart(__name__)
    async with app.test_request_context(
        "/config/update",
        method="POST",
        json={
            "values": {
                "graph_memory.graph_route_timeout_seconds": 4.2,
                "recall_engine.top_k": 7,
            }
        },
    ):
        resp = await handler.update_config()

    assert resp["status"] == "ok"
    assert raw.saved == 1
    assert engine.dual_route_retriever.graph_route_timeout == pytest.approx(4.2)
    assert config_manager.get(
        "graph_memory.graph_route_timeout_seconds"
    ) == pytest.approx(4.2)


# ==================== Task 2: Graph Timeout Default & High-Watermark Summary ====================


@pytest.mark.asyncio
async def test_graph_route_timeout_default_and_graceful_fallback(caplog):
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert (
        schema["graph_memory"]["items"]["graph_route_timeout_seconds"]["default"] == 3.5
    )
    assert ConfigManager({}).get(
        "graph_memory.graph_route_timeout_seconds"
    ) == pytest.approx(3.5)

    doc_hit = HybridResult(
        doc_id=1,
        final_score=0.9,
        rrf_score=0.8,
        bm25_score=0.7,
        vector_score=0.9,
        content="document result",
        metadata={"status": "active"},
    )
    doc_retriever = SimpleNamespace(search=AsyncMock(return_value=[doc_hit]))

    async def slow_graph(*_args, **_kwargs):
        await asyncio.sleep(0.5)
        return []

    graph_retriever = SimpleNamespace(search=slow_graph)
    dual = DualRouteRetriever(
        document_retriever=doc_retriever,
        graph_retriever=graph_retriever,
        memory_loader=AsyncMock(return_value=None),
        config={"graph_route_timeout_seconds": 0.05},
    )
    _failure_log_state.clear()
    diagnostics = {}
    with caplog.at_level(logging.WARNING):
        results = await dual.search("who is alice", k=5, diagnostics=diagnostics)

    assert len(results) == 1
    assert results[0].doc_id == 1
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    # Direct search_route also never raises on graph timeout
    res, err = await search_route("graph", slow_graph(), timeout=0.02, safe_errors=True)
    assert res == []
    assert isinstance(err, TimeoutError)


@pytest.mark.asyncio
async def test_enforce_message_limit_triggers_high_watermark_summary_before_hard_delete():
    cfg = ConfigManager(
        {
            "session_manager": {
                "max_messages_per_session": 100,
                "hard_max_messages_per_session": 200,
                "cleanup_batch_size": 20,
            }
        }
    )
    store = SimpleNamespace(
        connection=object(),
        get_message_count=AsyncMock(
            return_value=165
        ),  # >= 80% of 200 (160), < hard_max 200
    )
    conv = SimpleNamespace(
        store=store,
        get_session_metadata=AsyncMock(return_value=0),
        trim_session_messages=AsyncMock(return_value=0),
        invalidate_cache=AsyncMock(),
    )
    triggered: list[str] = []

    async def on_high_watermark(session_id: str):
        triggered.append(session_id)
        return True

    utils = MessageUtils(cfg, conv, high_watermark_summary_callback=on_high_watermark)
    await utils.enforce_message_limit("qq:GroupMessage:888")
    # Second immediate call should be throttled by cooldown
    await utils.enforce_message_limit("qq:GroupMessage:888")

    assert triggered == ["qq:GroupMessage:888"]
    assert conv.trim_session_messages.await_count == 0


# ==================== Task 3: i18n Help Completeness ====================


@pytest.mark.parametrize("lang", ["zh", "en", "ja", "ru"])
def test_i18n_help_contains_evidence_summarize_retry_and_purge(lang: str):
    path = Path(__file__).resolve().parents[1] / "core" / "i18n" / f"{lang}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    help_text = data["help"]["text"]
    assert "/lmem evidence" in help_text
    assert "/lmem summarize_retry" in help_text
    assert "--purge" in help_text
    assert "archived_success" in data["forget"]
    assert "purge_success" in data["forget"]


# ==================== Task 5: Scope Contract, Runtime Activation, Shared Config & Privacy Mute ====================


@pytest.mark.asyncio
async def test_series_scope_writable_contract_and_optimistic_lock(tmp_path):
    raw = _WritableConfig({"series_runtime_enabled": True})
    cfg = ConfigManager(raw)

    class _Plugin(SeriesMixin):
        def __init__(self):
            self.context = _make_chat_dynamics_ctx()
            self._data_dir = str(tmp_path)
            self._series_enabled = True
            self._terminating = False
            self.config_manager = cfg
            self.initializer = SimpleNamespace(
                is_initialized=True, data_dir=str(tmp_path)
            )

    plugin = _Plugin()
    umo = "qq:GroupMessage:12345"

    st0 = plugin.get_series_scope_status(umo)
    assert st0["umo"] == umo
    assert st0["enabled"] is True
    assert st0["writable"] is True
    assert st0["supported"] is True

    # Version mismatch returns version_conflict without saving
    conflict = await plugin.set_series_scope_enabled(
        umo, False, expected_version="stale-v"
    )
    assert conflict["ok"] is False
    assert conflict["reason"] == "version_conflict"
    assert raw.saved == 0

    # Disable in open mode writes disabled_ids and disables event access
    res_off = await plugin.set_series_scope_enabled(
        umo, False, expected_version=st0["version"]
    )
    assert res_off["ok"] is True
    assert res_off["enabled"] is False
    assert raw.saved == 1
    ev = SimpleNamespace(
        unified_msg_origin=umo,
        get_group_id=lambda: "12345",
        get_sender_id=lambda: "u1",
        get_platform_name=lambda: "qq",
    )
    assert is_event_memory_allowed(cfg, ev) is False

    # Re-enable with new version restores access
    res_on = await plugin.set_series_scope_enabled(
        umo, True, expected_version=res_off["version"]
    )
    assert res_on["ok"] is True
    assert res_on["enabled"] is True
    assert is_event_memory_allowed(cfg, ev) is True


@pytest.mark.asyncio
async def test_activate_series_runtime_enables_and_persists(tmp_path):
    raw = _WritableConfig({"series_runtime_enabled": False})
    cfg = ConfigManager(raw)
    init_started = []

    class _Plugin(SeriesMixin):
        def __init__(self):
            self.context = _make_chat_dynamics_ctx()
            self._data_dir = str(tmp_path)
            self._series_enabled = False
            self._terminating = False
            self.config_manager = cfg
            self.initializer = SimpleNamespace(
                is_initialized=False, data_dir=str(tmp_path)
            )

        def _create_tracked_task(self, coro):
            init_started.append(True)
            coro.close()
            return Mock()

        async def _initialize_plugin(self):
            return None

    plugin = _Plugin()
    res = await plugin.activate_series_runtime()
    assert res["ok"] is True
    assert res["enabled"] is True
    assert plugin._series_enabled is True
    assert raw["series_runtime_enabled"] is True
    assert init_started == [True]


@pytest.mark.asyncio
async def test_shared_config_inheritance_models_persona_and_private_group_sources(
    tmp_path,
):
    class _FakeEmb(EmbeddingProvider):
        def __init__(self, pid):
            self.provider_config = {"id": pid}

        async def get_embedding(self, text: str):
            return [0.1]

        async def get_embeddings(self, texts: list[str]):
            return [[0.1] for _ in texts]

        def get_dim(self) -> int:
            return 1

    class _FakeLLM(Provider):
        def __init__(self, pid):
            self.provider_config = {"id": pid}
            self.is_systemone_provider = False
            self._api_key = "fake-key"

        async def text_chat(self, *args, **kwargs):
            return SimpleNamespace(completion_text="{}")

        def get_current_key(self) -> str:
            return self._api_key

        def get_models(self) -> list[str]:
            return ["fake-model"]

        def set_key(self, key: str) -> None:
            self._api_key = key

    emb_shared = _FakeEmb("shared-emb")
    emb_other = _FakeEmb("other-emb")
    light_llm = _FakeLLM("shared-light")
    reply_llm = _FakeLLM("shared-reply")
    rerank_p = SimpleNamespace(provider_config={"id": "shared-rerank"})

    group_umo = "qq:GroupMessage:777"
    ctx = _make_chat_dynamics_ctx(
        shared_config={
            "reply_provider": "shared-reply",
            "light_analysis_provider": "shared-light",
            "embedding_provider": "shared-emb",
            "rerank_provider": "shared-rerank",
            "persona_id": "shared-persona",
            "private_shared_group_sources": [group_umo],
        },
        providers={
            "shared-emb": emb_shared,
            "shared-light": light_llm,
            "shared-reply": reply_llm,
            "shared-rerank": rerank_p,
        },
        embedding_providers=[emb_other, emb_shared],
    )
    ctx.conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value=None)
    )
    ctx.persona_manager = SimpleNamespace(
        get_default_persona_v3=AsyncMock(return_value=None)
    )

    # 1. Embedding provider inherits shared embedding_provider instead of first fallback
    cfg = ConfigManager({})
    initializer = PluginInitializer(ctx, cfg, str(tmp_path))
    await initializer._initialize_providers(silent=True)
    assert initializer.embedding_provider is emb_shared

    # 2. MemoryProcessor inherits light_analysis_provider when explicit config is blank
    proc = MemoryProcessor(context=ctx, config={})
    assert await proc._get_current_llm_provider() is light_llm

    # 3. Persona falls back to shared persona_id when host has no persona configured
    private_event = SimpleNamespace(
        unified_msg_origin="qq:FriendMessage:u1",
        get_platform_id=lambda: "qq",
        get_platform_name=lambda: "qq",
        get_sender_id=lambda: "u1",
    )
    assert await _get_native_persona_id(ctx, private_event) == "shared-persona"

    # 4. private_shared_group_sources accepts list in ConfigManager AND inherits from shared config
    cfg_list = ConfigManager(
        {
            "filtering_settings": {
                "private_shared_group_sources": [group_umo, "qq:GroupMessage:888"]
            }
        }
    )
    assert group_umo in cfg_list.get("filtering_settings.private_shared_group_sources")

    cfg_inherit = ConfigManager(
        {"filtering_settings": {"enable_group_to_private_recall": True}}
    )
    conv_mgr = SimpleNamespace(
        get_session_info=AsyncMock(return_value=SimpleNamespace(platform="qq"))
    )
    verified = await resolve_verified_group_sessions(
        cfg_inherit, private_event, conv_mgr, context=ctx
    )
    assert verified == (group_umo,)
    policy = resolve_memory_access_policy(
        cfg_inherit,
        private_event,
        persona_id="shared-persona",
        verified_group_sessions=verified,
        context=ctx,
    )
    assert any(scope.source_session_id == group_umo for scope in policy.read_scopes)


@pytest.mark.asyncio
async def test_privacy_mute_skips_group_capture_recall_and_reflection():
    umo = "qq:GroupMessage:555"
    ctx = _make_chat_dynamics_ctx(muted_map={(umo, "muted-user"): True})

    assert is_member_privacy_muted(ctx, umo, "muted-user", "memory") is True
    assert is_member_privacy_muted(ctx, umo, "normal-user", "memory") is False

    muted_event = SimpleNamespace(
        unified_msg_origin=umo,
        get_message_type=lambda: MessageType.GROUP_MESSAGE,
        get_sender_id=lambda: "muted-user",
        get_self_id=lambda: "bot-self",
        get_sender_name=lambda: "MutedUser",
        get_group_id=lambda: "555",
        get_platform_name=lambda: "qq",
    )
    assert is_event_privacy_muted(ctx, muted_event, "memory") is True

    cfg = ConfigManager({})
    conv = SimpleNamespace(add_message_from_event=AsyncMock())
    msg_utils = SimpleNamespace(
        extract_message_content=AsyncMock(return_value="secret text"),
        build_dedup_key=AsyncMock(return_value="k1"),
        is_duplicate_message=AsyncMock(return_value=False),
        mark_message_processed=AsyncMock(),
        enforce_message_limit=AsyncMock(),
        resolve_original_input=AsyncMock(),
    )

    # 1. GroupCapture skips muted member
    capture = GroupCapture(cfg, conv, msg_utils, context=ctx)
    await capture.handle_all_group_messages(muted_event)
    assert conv.add_message_from_event.await_count == 0

    # 2. MemoryRecall skips muted member and marks privacy_muted outcome
    recall = MemoryRecall(
        context=ctx,
        config_manager=cfg,
        memory_engine=Mock(),
        conversation_manager=conv,
        message_utils=msg_utils,
        injection_adapter=InjectionAdapter(),
    )
    req = ProviderRequest(prompt="hi")
    await recall.handle_memory_recall(muted_event, req)
    assert getattr(req, "_lm_private_hook_outcome_v1", None) == "privacy_muted"
    assert msg_utils.resolve_original_input.await_count == 0

    # 3. MemoryReflection skips muted member
    reflection = MemoryReflection(
        context=ctx,
        config_manager=cfg,
        memory_engine=Mock(),
        memory_processor=Mock(),
        conversation_manager=conv,
        message_utils=msg_utils,
        storage_tasks=set(),
        storage_sessions_inflight=set(),
        storage_state_lock=asyncio.Lock(),
    )
    resp = LLMResponse(role="assistant", completion_text="hello")
    assert await reflection._reflection_precheck(muted_event, resp) is None
