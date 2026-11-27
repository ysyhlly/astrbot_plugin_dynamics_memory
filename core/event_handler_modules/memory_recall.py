"""
记忆召回模块
负责长期记忆的检索和注入
"""

import asyncio
import contextvars
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

from ..memory_scope import (
    classify_umo,
    current_adapter_access,
    fetch_group_culture_context,
    is_event_excluded_by_series,
    is_event_memory_allowed,
    is_event_privacy_muted,
    is_member_privacy_muted,
    is_suspicious_session_id,
    resolve_memory_access_policy,
    resolve_verified_group_sessions,
)
from ..recall_runtime import (
    BoundedRecallSearchContext,
    RecallDeadline,
    RecallOperationContext,
    discard_private_summary_segment,
    remember_private_segment,
)
from ..retrieval.vector_search import QueryEmbeddingCache
from ..summary_source import AtomProjectionProof, summary_metadata_sha256, text_sha256
from ..utils import (
    format_memories_for_injection,
    get_persona_id,
)
from ..utils.injection_adapter import InjectionAdapter
from .memory_reflection import cancel_session_storage_task

if TYPE_CHECKING:
    from ..base.config_manager import ConfigManager
    from ..managers.conversation_manager import ConversationManager
    from ..managers.memory_engine import MemoryEngine
    from .message_utils import MessageUtils


_GROUP_COMMAND_PREFIXES = ("/", "!", ".", "！")
_MEANINGFUL_CHAR_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def should_skip_group_recall_query(
    event: AstrMessageEvent, actual_query: str
) -> str | None:
    """Return a skip reason when a group message should bypass recall/embedding.

    Skips when:
    - Raw message components contain ``At`` targeting only other bots/users (not ``self_id``);
    - ``actual_query`` starts with a command prefix (``/``, ``!``, ``.``, ``！``);
    - ``actual_query`` has fewer than 2 meaningful alphanumeric/CJK characters.
    """
    get_msg_type = getattr(event, "get_message_type", None)
    try:
        is_group = (
            callable(get_msg_type) and get_msg_type() == MessageType.GROUP_MESSAGE
        )
    except Exception:
        is_group = False
    if not is_group:
        return None

    # 1. Inspect raw message components for At targeting only other bots/users
    raw_components = None
    msg_obj = getattr(event, "message_obj", None)
    if msg_obj is not None:
        raw_components = getattr(msg_obj, "message", None)
    if not isinstance(raw_components, (list, tuple)):
        get_messages = getattr(event, "get_messages", None)
        if callable(get_messages):
            try:
                raw_components = get_messages()
            except Exception:
                raw_components = None
    if isinstance(raw_components, (list, tuple)):
        at_targets: set[str] = set()
        mentions_all = False
        for comp in raw_components:
            if type(comp).__name__ != "At":
                continue
            target = str(
                getattr(comp, "qq", None)
                or getattr(comp, "target", None)
                or getattr(comp, "user_id", None)
                or ""
            ).strip()
            if not target:
                continue
            if target.lower() == "all":
                mentions_all = True
                break
            at_targets.add(target)
        if at_targets and not mentions_all:
            get_self_id = getattr(event, "get_self_id", None)
            try:
                self_id = (
                    str(get_self_id() or "").strip()
                    if callable(get_self_id)
                    else str(getattr(event, "self_id", "") or "").strip()
                )
            except Exception:
                self_id = ""
            if self_id and self_id not in at_targets:
                return "at_other"

    # 2. Skip command-prefixed messages (e.g. /今日运势, !ping, .help, ！菜单)
    stripped = (actual_query or "").strip()
    if not stripped:
        return "empty"
    if stripped[0] in _GROUP_COMMAND_PREFIXES:
        return "command_prefix"

    # 3. Skip trivially short messages (<2 meaningful chars)
    if len(_MEANINGFUL_CHAR_RE.findall(stripped)) < 2:
        return "trivial_short"
    return None


class MemoryRecall:
    """记忆召回类"""

    def __init__(
        self,
        context,
        config_manager: "ConfigManager",
        memory_engine: "MemoryEngine",
        conversation_manager: "ConversationManager",
        message_utils: "MessageUtils",
        injection_adapter: "InjectionAdapter",
        evidence_filter=None,
        recall_runtime=None,
        evidence_revision_provider=None,
        summary_guard=None,
        storage_session_tasks: dict[str, asyncio.Task] | None = None,
    ):
        """
        初始化记忆召回模块

        Args:
            context: AstrBot上下文
            config_manager: 配置管理器
            memory_engine: 记忆引擎
            conversation_manager: 会话管理器
            message_utils: 消息处理工具
            storage_session_tasks: 会话 -> 进行中总结任务映射（共享状态），
                用于在检测到 WebUI 编辑/重试回滚后取消过期总结任务
        """
        self.context = context
        self.config_manager = config_manager
        self.memory_engine = memory_engine
        self.conversation_manager = conversation_manager
        self.message_utils = message_utils
        self.injection_adapter = injection_adapter
        self.evidence_filter = evidence_filter
        self.recall_runtime = recall_runtime
        self.evidence_revision_provider = evidence_revision_provider
        self.summary_guard = summary_guard
        self._message_limit_tasks: dict[str, asyncio.Task] = {}
        self._message_limit_task_limit = 16
        self._scope_tasks: set[asyncio.Task] = set()
        self._storage_session_tasks: dict[str, asyncio.Task] = (
            storage_session_tasks if storage_session_tasks is not None else {}
        )

    @staticmethod
    def _message_timestamp_seconds(value) -> float | None:
        if isinstance(value, (int, float)):
            timestamp = float(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                timestamp = float(stripped)
            except ValueError:
                try:
                    timestamp = datetime.fromisoformat(
                        stripped.replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    return None
        else:
            return None

        if timestamp > 100_000_000_000:
            timestamp /= 1000.0
        return timestamp if timestamp > 0 else None

    async def handle_memory_recall(
        self, event: AstrMessageEvent, req: ProviderRequest, *, budget=None
    ):
        """Query and inject long-term memory before LLM request"""
        try:
            if is_event_excluded_by_series(
                getattr(self, "context", None), event
            ) or not is_event_memory_allowed(self.config_manager, event):
                logger.debug("[LMRecall] route=policy outcome=denied")
                return
            if is_event_privacy_muted(getattr(self, "context", None), event, "memory"):
                logger.debug("[LMRecall] route=privacy_mute outcome=skipped")
                self._set_recall_outcome(req, "privacy_muted")
                return
            session_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
            deadline = (
                budget if isinstance(budget, RecallDeadline) else RecallDeadline.start()
            )

            # 检测异常session_id
            if is_suspicious_session_id(session_id):
                logger.warning("[LMRecall] route=scope outcome=suspicious")

            # Never use ProviderRequest.prompt: CD may have rewritten it into a
            # planning envelope. The authenticated sidecar resolver is the only
            # input for owned clones; ordinary host events use direct Plain parts.
            cached_original = getattr(req, "_lm_resolved_original_input_v1", None)
            if cached_original is not None:
                original = cached_original
            else:
                try:
                    original = await asyncio.wait_for(
                        self.message_utils.resolve_original_input(event),
                        timeout=deadline.search_remaining(),
                    )
                except asyncio.TimeoutError:
                    self._set_recall_outcome(req, "deadline_exceeded")
                    return
            if original.status == "unavailable":
                self._set_recall_outcome(req, "original_input_unavailable")
                return
            self._remove_agent_recall_policy_from_context(req)
            self._maybe_inject_agent_recall_policy(req, session_id)
            if (
                original.status == "not_owned"
                and event.get_message_type() != MessageType.GROUP_MESSAGE
            ):
                try:
                    async with asyncio.timeout(deadline.search_remaining()):
                        await self._reconcile_with_llm_context(event, session_id, req)
                except TimeoutError:
                    self._set_recall_outcome(req, "deadline_exceeded")
                    return
            actual_query = original.query_text.strip()
            if deadline.remaining() <= 0:
                self._set_recall_outcome(req, "deadline_exceeded")
                return

            async with RecallOperationContext("memory_recall"):
                # 自动删除旧的注入记忆
                if self.config_manager.get("recall_engine.auto_remove_injected", True):
                    removed = self._remove_injected_memories_from_context(
                        req, session_id
                    )
                    removed += self._remove_fake_tool_call_from_context(req, session_id)
                    if removed > 0:
                        logger.debug(
                            "[LMRecall] route=context_cleanup removed=%d", removed
                        )

                if not actual_query:
                    self._set_recall_outcome(req, "empty")
                    return

                skip_reason = should_skip_group_recall_query(event, actual_query)
                if skip_reason is not None:
                    logger.debug(
                        "[LMRecall] route=noise_filter outcome=skipped reason=%s",
                        skip_reason,
                    )
                    self._set_recall_outcome(req, f"skipped_{skip_reason}")
                    return

                # Store only real original private user text. Owned CD DTOs are
                # archived idempotently with their authentic per-message identity;
                # quote/media parts remain typed metadata and are not text.
                is_group = event.get_message_type() == MessageType.GROUP_MESSAGE
                source_store_partial = False
                if not is_group and actual_query and original.status == "valid":
                    try:
                        source_store_partial = (
                            not await self._store_owned_original_messages(
                                event, req, original, deadline
                            )
                        )
                        setattr(
                            req,
                            "_lm_private_original_store_status_v1",
                            "unavailable" if source_store_partial else "stored",
                        )
                    except asyncio.TimeoutError:
                        setattr(
                            req, "_lm_private_original_store_status_v1", "unavailable"
                        )
                        self._set_recall_outcome(req, "deadline_exceeded")
                        return
                elif not is_group and actual_query and original.status == "not_owned":
                    dedup_key = await self.message_utils.build_dedup_key(
                        event, session_id, actual_query
                    )
                    if not await self.message_utils.is_duplicate_message(dedup_key):
                        remaining = deadline.search_remaining()
                        if remaining <= 0:
                            self._set_recall_outcome(req, "deadline_exceeded")
                            return
                        try:
                            await asyncio.wait_for(
                                self.conversation_manager.add_message_from_event(
                                    event=event, role="user", content=actual_query
                                ),
                                timeout=remaining,
                            )
                        except asyncio.TimeoutError:
                            self._set_recall_outcome(req, "deadline_exceeded")
                            return
                        await self.message_utils.mark_message_processed(dedup_key)
                if not is_group and original.status != "unavailable":
                    if classify_umo(session_id) == "private":
                        self._schedule_message_limit(session_id)

                # Beta 自主回忆规则：独立于自动召回是否命中；即使 top_k=0
                # 也可为当前请求可用的主动工具提供规则说明。

                # 若 top_k <= 0，跳过记忆检索和注入，但上述清理和消息存储已执行
                top_k = self.config_manager.get("recall_engine.top_k", 5)
                if top_k <= 0:
                    logger.debug("[LMRecall] route=search outcome=disabled")
                    self._set_recall_outcome(
                        req, "partial" if source_store_partial else "empty"
                    )
                    return

                # on_llm_request 钩子在 _ensure_persona_and_skills 之前触发，
                # 因此人格需按三级优先级自行解析；召回与写入共用同一解析入口。

                # 获取 persona_id，与 AstrBot 主流程保持一致的三级优先级：
                # 1. session_service_config（最高）
                # 2. req.conversation.persona_id（会话级）
                # 3. 全局默认人格（最低）
                # 注意：on_llm_request 钩子在 _ensure_persona_and_skills 之前触发，
                # 因此不能直接依赖 req.system_prompt 已注入人格，需自行走完整优先级。
                try:
                    persona_id = await asyncio.wait_for(
                        get_persona_id(self.context, event),
                        timeout=deadline.search_remaining(),
                    )
                except asyncio.TimeoutError:
                    self._set_recall_outcome(req, "deadline_exceeded")
                    return

                try:
                    verified_groups = await asyncio.wait_for(
                        self._verified_shared_group_sessions(event),
                        timeout=deadline.search_remaining(),
                    )
                except asyncio.TimeoutError:
                    self._set_recall_outcome(req, "deadline_exceeded")
                    return
                access_policy = resolve_memory_access_policy(
                    self.config_manager,
                    event,
                    persona_id=persona_id,
                    verified_group_sessions=verified_groups,
                    context=getattr(self, "context", None),
                )
                recall_persona_id = (
                    persona_id
                    if self.config_manager.get(
                        "filtering_settings.use_persona_filtering", True
                    )
                    else None
                )

                # Use the trusted original query, current event first. Do not
                # concatenate stored assistant turns or rewritten request text.
                query_for_search = actual_query

                # 执行记忆召回
                logger.info(
                    "[LMRecall] route=search outcome=start query_chars=%d",
                    len(query_for_search),
                )

                recall_k = self.config_manager.get("recall_engine.top_k", 5)
                recalled_by_id = {}
                own_recalled_ids: set[int] = set()
                source_scope_by_id = {}
                read_only_by_id = {}
                # The source guard needs the authenticated persona even when
                # the retriever is configured to skip persona filtering.
                search_context = self._make_search_context(
                    session_id=session_id,
                    query=query_for_search,
                    persona_id=persona_id,
                    budget=deadline,
                    req=req,
                )
                phase_started = time.monotonic()
                scope_jobs = []
                for scope in access_policy.read_scopes:
                    scope_session_id = (
                        scope.session_ids[0]
                        if scope.session_ids and len(scope.session_ids) == 1
                        else None
                    )
                    scope_persona_id = (
                        persona_id if scope.require_persona else recall_persona_id
                    )
                    interaction_type = (
                        scope.interaction_types[0]
                        if scope.interaction_types and len(scope.interaction_types) == 1
                        else scope.interaction_types
                    )
                    scope_jobs.append(
                        (scope, scope_session_id, scope_persona_id, interaction_type)
                    )
                scope_results_list = await self._run_scope_searches(
                    scope_jobs,
                    deadline,
                    query_for_search,
                    recall_k,
                    search_context,
                    req,
                    session_id,
                    persona_id,
                )
                phases = getattr(req, "_lm_private_phase_ms_v1", {})
                phases["scope_searches"] = round(
                    (time.monotonic() - phase_started) * 1000, 1
                )
                req._lm_private_phase_ms_v1 = phases
                search_partial = any(
                    outcome["state"] != "completed" for outcome in scope_results_list
                ) or bool(search_context and search_context.route_failures)
                for (scope, scope_session_id, _, _), outcome in zip(
                    scope_jobs, scope_results_list
                ):
                    scope_results = outcome["value"]
                    for result in scope_results:
                        existing = recalled_by_id.get(result.doc_id)
                        if (
                            existing is None
                            or result.final_score > existing.final_score
                        ):
                            recalled_by_id[result.doc_id] = result
                            source_scope_by_id[int(result.doc_id)] = (
                                self._search_scope_identity(
                                    req,
                                    session_id,
                                    scope_session_id,
                                    persona_id,
                                    scope.source_session_id,
                                )
                            )
                            read_only_by_id[int(result.doc_id)] = bool(scope.read_only)
                        if not scope.read_only:
                            own_recalled_ids.add(int(result.doc_id))
                recalled_memories = sorted(
                    recalled_by_id.values(),
                    key=lambda result: result.final_score,
                    reverse=True,
                )[:recall_k]
                if recalled_memories and getattr(self, "context", None) is not None:
                    filtered_recalled = []
                    for mem in recalled_memories:
                        meta = getattr(mem, "metadata", None)
                        mem_sender = (
                            str(meta.get("sender_id") or "").strip()
                            if isinstance(meta, dict)
                            else ""
                        )
                        mem_umo = (
                            str(
                                meta.get("source_session_id")
                                or meta.get("session_id")
                                or session_id
                            ).strip()
                            if isinstance(meta, dict)
                            else session_id
                        )
                        if mem_sender and is_member_privacy_muted(
                            self.context, mem_umo, mem_sender, "memory"
                        ):
                            continue
                        filtered_recalled.append(mem)
                    recalled_memories = filtered_recalled
                for scope in access_policy.read_scopes:
                    if (
                        not scope.read_only
                        and scope.session_ids
                        and len(scope.session_ids) == 1
                        and scope.session_ids[0] == session_id
                    ):
                        migrate = getattr(
                            self.memory_engine,
                            "schedule_legacy_session_migration",
                            None,
                        )
                        if callable(migrate):
                            migrate(session_id)

                if recalled_memories:
                    logger.debug(
                        "[LMRecall] route=search outcome=found item_count=%d",
                        len(recalled_memories),
                    )

                    # 格式化并注入记忆
                    memory_list = [
                        {
                            "id": getattr(mem, "doc_id", None),
                            "content": mem.content,
                            "score": mem.final_score,
                            "metadata": mem.metadata,
                            "timestamp": mem.metadata.get("create_time"),
                        }
                        for mem in recalled_memories
                    ]

                    if self.evidence_filter:
                        try:
                            projected = await asyncio.wait_for(
                                self.evidence_filter(
                                    event, req, memory_list, actual_query, persona_id
                                ),
                                timeout=deadline.remaining(),
                            )
                        except asyncio.TimeoutError:
                            self._set_recall_outcome(req, "deadline_exceeded")
                            return
                        if projected is not None:
                            memory_list = projected
                            if not memory_list:
                                logger.debug(
                                    "证据视图尚未就绪，本轮不注入未解析的混合摘要"
                                )
                                self._set_recall_outcome(req, "partial")
                                return

                    # 仅记录统计信息，不输出真实记忆或群友原文。
                    for i, mem in enumerate(memory_list, 1):
                        logger.debug(
                            f"[LMRecall] item={i} score={float(mem.get('score', 0)):.3f}, "
                            f"provenance={str((mem.get('metadata') or {}).get('provenance_kind', 'unknown'))[:32]}"
                        )

                    # 注入策略适配层负责把历史配置值映射到唯一的 add-only
                    # 注入方式，废弃模式会返回回退原因供日志提示。
                    configured_method = self.config_manager.get(
                        "recall_engine.injection_method", "extra_user_content"
                    )
                    injection_method, fallback_reason = self.injection_adapter.resolve(
                        configured_method
                    )
                    if fallback_reason:
                        logger.warning(
                            f"[LMRecall] 注入方式从 {configured_method} 回退为 "
                            f"{injection_method}: {fallback_reason}"
                        )

                    culture_items: list[dict[str, str]] | None = None
                    if is_group and deadline.remaining() > 0.12:
                        try:
                            culture_items = await fetch_group_culture_context(
                                getattr(self, "context", None),
                                session_id,
                                query=actual_query,
                                limit=4,
                                timeout=min(0.4, deadline.remaining() * 0.25),
                            )
                        except Exception:
                            culture_items = None

                    memory_str = format_memories_for_injection(
                        memory_list, culture_items=culture_items or None
                    )

                    source_records = []
                    if len(recalled_memories) > 20:
                        self._set_recall_outcome(req, "busy")
                        return
                    for memory in recalled_memories:
                        memory_id = getattr(memory, "doc_id", None)
                        content = getattr(memory, "content", None)
                        metadata = getattr(memory, "metadata", None)
                        if (
                            isinstance(memory_id, bool)
                            or not isinstance(memory_id, int)
                            or not isinstance(content, str)
                            or not isinstance(metadata, dict)
                        ):
                            continue
                        source_scope = dict(source_scope_by_id.get(memory_id) or {})
                        source_umo = metadata.get("source_session_id") or metadata.get(
                            "session_id"
                        )
                        if isinstance(source_umo, str) and source_umo.strip():
                            source_scope["umo"] = source_umo
                        if (
                            isinstance(metadata.get("persona_id"), str)
                            and metadata["persona_id"].strip()
                        ):
                            source_scope["persona_id"] = metadata["persona_id"]
                        projection = getattr(memory, "atom_projection", None)
                        if not isinstance(projection, AtomProjectionProof):
                            projection = None
                        content_hash = text_sha256(content)
                        source_records.append(
                            {
                                "document_id": memory_id,
                                "document_text_sha256": (
                                    projection.document_sha256
                                    if projection
                                    else content_hash
                                ),
                                "projected_text_sha256": content_hash,
                                "atom_proof": [list(atom) for atom in projection.atoms]
                                if projection
                                else [],
                                "summary_metadata_sha256": summary_metadata_sha256(
                                    metadata
                                ),
                                "source_scope": source_scope,
                                "source_session_id": source_umo,
                                "source_platform": metadata.get("source_platform"),
                                "interaction_type": metadata.get("interaction_type"),
                                "persona_id": metadata.get("persona_id"),
                                "memory_visibility": metadata.get("memory_visibility"),
                                "status": metadata.get("status", "active"),
                                "read_only": read_only_by_id.get(
                                    memory_id, memory_id not in own_recalled_ids
                                ),
                            }
                        )
                    if deadline.remaining() <= 0:
                        self._set_recall_outcome(req, "deadline_exceeded")
                        return
                    # Attach the summary as a temporary part once its source
                    # records are ready for the guard.
                    previous_summary_sources = getattr(
                        req, "_lm_private_summary_sources_v1", None
                    )
                    previous_summary_validator = getattr(
                        req, "_lm_private_summary_validate_v1", None
                    )
                    injected_part = TextPart(text=memory_str).mark_as_temp()
                    req.extra_user_content_parts.append(injected_part)
                    remember_private_segment(
                        req,
                        injected_part,
                        source_level="legacy_summary",
                        item_count=len(memory_list),
                    )
                    guarded = False
                    if (
                        callable(self.summary_guard)
                        and source_records
                        and len(source_records) == len(recalled_memories)
                    ):
                        card = getattr(req, "_cd_memory_feedback_v1", None)
                        try:
                            guard_result = self.summary_guard(
                                event,
                                req,
                                source_records,
                                target_persona_id=persona_id,
                                card_version=(
                                    str(card.get("card_version", ""))
                                    if isinstance(card, dict)
                                    else ""
                                ),
                                segment_text=getattr(injected_part, "text", ""),
                                item_count=len(memory_list),
                            )
                            if asyncio.iscoroutine(guard_result) or hasattr(
                                guard_result, "__await__"
                            ):
                                guard_result = await asyncio.wait_for(
                                    guard_result, timeout=deadline.remaining()
                                )
                            validator = getattr(
                                req, "_lm_private_summary_validate_v1", None
                            )
                            guarded = guard_result is True and callable(validator)
                            if guarded:
                                guarded = (
                                    await asyncio.wait_for(
                                        validator(), timeout=deadline.remaining()
                                    )
                                    is True
                                )
                            if deadline.remaining() <= 0:
                                raise asyncio.TimeoutError
                        except asyncio.TimeoutError:
                            guarded = False
                            self._set_recall_outcome(req, "deadline_exceeded")
                        except asyncio.CancelledError:
                            discard_private_summary_segment(
                                req,
                                injected_part,
                                previous_sources=previous_summary_sources,
                                previous_validator=previous_summary_validator,
                            )
                            raise
                        except Exception:
                            guarded = False
                    if not guarded:
                        discard_private_summary_segment(
                            req,
                            injected_part,
                            previous_sources=previous_summary_sources,
                            previous_validator=previous_summary_validator,
                        )
                        if (
                            getattr(req, "_lm_private_hook_outcome_v1", None)
                            != "deadline_exceeded"
                        ):
                            self._set_recall_outcome(req, "partial")
                        return
                    own_ids = [
                        int(mem.doc_id)
                        for mem in recalled_memories
                        if isinstance(getattr(mem, "doc_id", None), int)
                        and int(mem.doc_id) in own_recalled_ids
                    ]
                    atoms = [
                        int(atom_id)
                        for mem in recalled_memories
                        if isinstance(getattr(mem, "doc_id", None), int)
                        and int(mem.doc_id) in own_recalled_ids
                        for atom_id in (mem.metadata or {}).get(
                            "retrieved_atom_ids", []
                        )
                        if isinstance(atom_id, int) and not isinstance(atom_id, bool)
                    ]
                    if own_ids:
                        touch = getattr(
                            self.memory_engine, "schedule_recall_access_touch", None
                        )
                        if callable(touch):
                            touch(own_ids, atoms)
                    self._set_recall_outcome(
                        req,
                        "partial"
                        if search_partial or source_store_partial
                        else "ready",
                    )
                    logger.info(
                        "[LMRecall] route=surface outcome=appended item_count=%d",
                        len(memory_list),
                    )
                else:
                    logger.info("[LMRecall] route=search outcome=empty")
                    self._set_recall_outcome(
                        req,
                        "partial"
                        if search_partial or source_store_partial
                        else "empty",
                    )

        except asyncio.CancelledError:
            self._set_recall_outcome(req, "cancelled")
            raise
        except Exception as e:
            self._set_recall_outcome(req, "error")
            logger.error(
                "[LMRecall] route=handler outcome=error exception_type=%s",
                type(e).__name__,
            )

    def _remove_agent_recall_policy_from_context(self, req: ProviderRequest) -> int:
        """移除本插件上一轮注入的自主回忆规则临时片段（重复钩子去重）。

        只识别带本插件专用边界标记的临时片段，不动记忆注入片段、
        真实工具结果或其他插件的临时内容。
        """
        from ..base.constants import (
            AGENT_RECALL_POLICY_FOOTER,
            AGENT_RECALL_POLICY_HEADER,
        )

        parts = getattr(req, "extra_user_content_parts", None)
        if not parts:
            return 0
        kept = [
            part
            for part in parts
            if not self._is_agent_recall_policy_part(
                part, AGENT_RECALL_POLICY_HEADER, AGENT_RECALL_POLICY_FOOTER
            )
        ]
        removed = len(parts) - len(kept)
        if removed:
            req.extra_user_content_parts = kept
        return removed

    @staticmethod
    def _is_agent_recall_policy_part(part, header: str, footer: str) -> bool:
        text = getattr(part, "text", "")
        return (
            getattr(part, "_no_save", False)
            and isinstance(text, str)
            and header in text
            and footer in text
        )

    def _load_agent_recall_policy(self) -> str | None:
        """按 自定义覆盖 > 内置默认文件 顺序读取自主回忆规则。"""
        try:
            from ..prompts.prompt_manager import get_prompt_manager

            manager = get_prompt_manager()
            if manager is not None:
                content = manager.get_prompt("agent_recall_policy")
                if content and content.strip():
                    return content
        except Exception:
            pass
        try:
            from pathlib import Path

            builtin = (
                Path(__file__).resolve().parents[1]
                / "prompts"
                / "agent_recall_policy.txt"
            )
            if builtin.exists():
                content = builtin.read_text(encoding="utf-8")
                if content.strip():
                    return content
        except Exception:
            pass
        return None

    def _maybe_inject_agent_recall_policy(
        self, req: ProviderRequest, session_id: str
    ) -> None:
        """Beta：按需注入自主回忆规则（add-only 临时片段，先清后加可去重）。

        仅在主动回忆工具与 Beta 开关同时开启、插件就绪、事件已放行、且当前
        请求实际提供 recall_long_term_memory 工具时注入；能力说明严格以
        当前请求的 func_tool 集合为准。
        """
        from ..base.constants import (
            AGENT_RECALL_POLICY_FOOTER,
            AGENT_RECALL_POLICY_HEADER,
        )

        try:
            if not self.config_manager.get("agent_tools.enable_recall_tool", True):
                return
            if not self.config_manager.get(
                "agent_tools.enable_agentic_recall_beta", False
            ):
                return
            if self.memory_engine is None:
                return
            func_tool = getattr(req, "func_tool", None)
            get_names = getattr(func_tool, "names", None)
            names: list[str] = []
            if callable(get_names):
                try:
                    names = list(get_names())
                except Exception:
                    names = []
            if "recall_long_term_memory" not in names:
                return
            policy = self._load_agent_recall_policy()
            if not policy:
                return
            available = [
                "- recall_long_term_memory: search long-term memory by topic "
                "keywords, or browse the current speaker's past experiences with "
                'target="current_speaker" (query may be empty) when the question '
                "has no usable keywords; refine follow-up searches with "
                "exclude_ids / start_time / end_time / search_offset."
            ]
            if "read_memory_evidence" in names:
                available.append(
                    "- read_memory_evidence: deep-read one memory by its integer "
                    "memory_id for exact facts or retained original messages "
                    "(permission and lifecycle re-checked); pass the returned "
                    "next_cursor unchanged as cursor to continue long evidence. "
                    "For original-message pages, keep include_source=true."
                )
            policy_text = policy.replace("{available_tools}", "\n".join(available))
            wrapped = (
                f"{AGENT_RECALL_POLICY_HEADER}\n"
                f"{policy_text.strip()}\n"
                f"{AGENT_RECALL_POLICY_FOOTER}"
            )
            req.extra_user_content_parts.append(TextPart(text=wrapped).mark_as_temp())
            logger.info(f"[{session_id}] 已注入自主回忆规则（Beta）")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[{session_id}] 注入自主回忆规则失败: {e}")

    async def _reconcile_with_llm_context(
        self, event: AstrMessageEvent, session_id: str, req: ProviderRequest
    ) -> None:
        """以 LLM 请求上下文为锚点，对账插件会话库的尾部轮次。

        WebUI 的「编辑上一条消息并重新请求」「换模型重试上一条对话」只作用于
        AstrBot 侧会话历史，插件无感知；此处从 req.contexts 提取 ``_checkpoint``
        段并清理会话库中已被撤销的轮次（见 core/utils/history_alignment.py）。

        前置门槛：仅当事件携带 webchat 的 ``llm_checkpoint_id`` extra 时才执行
        （非 webchat 平台与旧版 AstrBot 无此标记，库内也不会有 checkpoint 行，
        直接跳过，避免每个私聊请求都读取全量会话库）。任何异常都不影响召回
        主流程。
        """
        checkpoint_id = None
        get_extra = getattr(event, "get_extra", None)
        if callable(get_extra):
            try:
                checkpoint_id = get_extra("llm_checkpoint_id")
            except Exception:
                checkpoint_id = None
        if not (isinstance(checkpoint_id, str) and checkpoint_id):
            return

        contexts = getattr(req, "contexts", None)
        if not isinstance(contexts, list):
            return
        ctx_checkpoints: list[str] = []
        for message in contexts:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "_checkpoint":
                continue
            content = message.get("content")
            checkpoint_id = content.get("id") if isinstance(content, dict) else None
            if isinstance(checkpoint_id, str) and checkpoint_id:
                ctx_checkpoints.append(checkpoint_id)
        try:
            deleted = await self.conversation_manager.reconcile_session_tail(
                session_id, ctx_checkpoints
            )
        except Exception as exc:
            logger.warning(f"[{session_id}] 对话历史对账调用失败（已跳过）: {exc}")
            return
        if deleted > 0:
            # 已确认发生回滚：取消进行中的总结任务，避免它继续用旧范围调用 LLM
            # 并写入过期内容（修订号校验是最终防线，取消负责尽早止损）
            cancel_session_storage_task(self._storage_session_tasks, session_id)

    def _remove_injected_memories_from_context(
        self, req: ProviderRequest, session_id: str
    ) -> int:
        """从请求上下文中移除临时注入的记忆片段"""
        import re

        from ..base.constants import MEMORY_INJECTION_FOOTER, MEMORY_INJECTION_HEADER

        removed = 0

        # 清理 system_prompt（兼容旧版本注入残留）
        if hasattr(req, "system_prompt") and req.system_prompt:
            if isinstance(req.system_prompt, str):
                original_prompt = req.system_prompt
                if (
                    MEMORY_INJECTION_HEADER in original_prompt
                    and MEMORY_INJECTION_FOOTER in original_prompt
                ):
                    # 使用正则清理记忆片段
                    pattern = re.compile(
                        re.escape(MEMORY_INJECTION_HEADER)
                        + r".*?"
                        + re.escape(MEMORY_INJECTION_FOOTER),
                        re.DOTALL,
                    )
                    cleaned_prompt = pattern.sub("", original_prompt)
                    cleaned_prompt = re.sub(r"\n{3,}", "\n\n", cleaned_prompt).strip()
                    req.system_prompt = cleaned_prompt
                    if cleaned_prompt != original_prompt:
                        removed += 1

        # 清理 extra_user_content_parts（通过 mark_as_temp/_no_save 标记）
        for part in getattr(req, "extra_user_content_parts", []):
            if self._is_livingmemory_temp_part(part):
                discard_private_summary_segment(req, part)
                removed += 1

        return removed

    def _is_livingmemory_temp_part(self, part) -> bool:
        """判断是否为 LivingMemory 本轮临时注入的 extra_user_content part"""
        from ..base.constants import MEMORY_INJECTION_FOOTER, MEMORY_INJECTION_HEADER

        text = getattr(part, "text", "")
        return (
            getattr(part, "_no_save", False)
            and isinstance(text, str)
            and MEMORY_INJECTION_HEADER in text
            and MEMORY_INJECTION_FOOTER in text
        )

    def _remove_fake_tool_call_from_context(
        self, req: ProviderRequest, session_id: str
    ) -> int:
        """从请求上下文中移除伪造的工具调用记忆（legacy 兼容清理）

        老版本 fake_tool_call 注入方式会把伪造的
        assistant(tool_calls) + tool(result) 消息对持久化进 AstrBot 历史。
        该注入方式已废弃、不再产生新消息，此方法仅在请求视图里识别并
        移除以 FAKE_TOOL_CALL_ID_PREFIX 为 ID 前缀的历史残留消息对，
        不改写用户自身的任何历史消息。
        """
        from ..base.constants import FAKE_TOOL_CALL_ID_PREFIX

        if not hasattr(req, "contexts") or not req.contexts:
            return 0

        removed = 0
        indices_to_remove: set[int] = set()
        fake_call_ids: set[str] = set()

        try:
            # 单轮扫描：同时收集伪造 assistant(tool_calls) 和对应 tool(result) 消息
            for i, msg in enumerate(req.contexts):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tc_id = (
                            tc.get("id", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "id", "")
                        )
                        if tc_id.startswith(FAKE_TOOL_CALL_ID_PREFIX):
                            fake_call_ids.add(tc_id)
                            indices_to_remove.add(i)
                elif role == "tool":
                    tc_id = msg.get("tool_call_id", "")
                    if tc_id in fake_call_ids:
                        indices_to_remove.add(i)

            # 从后往前删除，避免索引偏移
            for i in sorted(indices_to_remove, reverse=True):
                req.contexts.pop(i)
                removed += 1

        except Exception:
            pass

        return removed

    async def _verified_shared_group_sessions(self, event) -> tuple[str, ...]:
        return await resolve_verified_group_sessions(
            self.config_manager,
            event,
            self.conversation_manager,
            context=getattr(self, "context", None),
        )

    @staticmethod
    def _is_valid_recall_outcome(outcome: Any) -> bool:
        if not isinstance(outcome, str) or not outcome:
            return False
        if outcome in {
            "ready",
            "partial",
            "empty",
            "deadline_exceeded",
            "cancelled",
            "busy",
            "error",
            "original_input_unavailable",
            "disabled",
            "privacy_muted",
            "policy_denied",
        }:
            return True
        return (
            outcome.startswith("skipped_")
            and len(outcome) <= 64
            and all(ch.isalnum() or ch == "_" for ch in outcome)
        )

    @classmethod
    def _set_recall_outcome(cls, req, outcome: str) -> None:
        if not cls._is_valid_recall_outcome(outcome):
            outcome = "error"
        try:
            setattr(req, "_lm_private_hook_outcome_v1", outcome)
        except Exception:
            pass

    async def _run_scope_searches(
        self, jobs, deadline, query, k, context, req, session_id, persona_id
    ):
        """Collect completed scopes by the search cutoff, with three reads at a time."""
        outcomes = [{"state": "timeout", "value": []} for _ in jobs]
        jobs = iter(enumerate(jobs))
        active = {}

        def fill_slots():
            while len(active) < 3 and deadline.search_remaining() > 0:
                job = next(jobs, None)
                if job is None:
                    break
                index, (scope, selected_session, search_persona, interaction) = job
                identity = self._search_scope_identity(
                    req,
                    session_id,
                    selected_session,
                    persona_id,
                    scope.source_session_id,
                )
                scoped_context = (
                    context.for_scope(identity, self._scope_revision(identity))
                    if context
                    else None
                )
                extra_kwargs = (
                    {"excluded_session_ids": scope.excluded_session_ids}
                    if scope.excluded_session_ids
                    else {}
                )
                task = self._submit_scope_search(
                    query=query,
                    k=k,
                    session_id=selected_session,
                    persona_id=search_persona,
                    interaction_type=interaction,
                    source_platform=scope.source_platform,
                    source_session_id=scope.source_session_id,
                    touch_access=False,
                    cache_results=False,
                    allow_legacy_session_migration=False,
                    execution_context=scoped_context,
                    **extra_kwargs,
                )
                if task is None:
                    outcomes[index]["state"] = "busy"
                else:
                    active[task] = index

        try:
            fill_slots()
            while active:
                done, _ = await asyncio.wait(
                    active,
                    timeout=deadline.search_remaining(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    break
                for task in done:
                    index = active.pop(task)
                    if task.cancelled():
                        continue
                    try:
                        finished, value = task.result()
                    except Exception as exc:
                        outcomes[index]["state"] = "error"
                        logger.debug(
                            "[LMRecall] route=memory_scope outcome=error exception_type=%s",
                            type(exc).__name__,
                        )
                    else:
                        if finished <= deadline.search_deadline:
                            outcomes[index] = {"state": "completed", "value": value}
                fill_slots()
        finally:
            # Store reads have no request reference. The owner holds their slots
            # until cleanup finishes, without spending the validation reserve.
            for task in active:
                task.cancel()
        return outcomes

    def _submit_scope_search(self, **search_kwargs):
        search = self.memory_engine.search_memories

        async def read():
            result = await search(**search_kwargs)
            return time.monotonic(), result

        task_context = contextvars.Context()
        task_context.run(current_adapter_access.set, current_adapter_access.get())
        if self.recall_runtime is not None:
            return self.recall_runtime.submit_scope(read, context=task_context)
        if len(self._scope_tasks) >= 16:
            return None
        task = asyncio.create_task(read(), context=task_context)
        self._scope_tasks.add(task)

        def completed(done):
            self._scope_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed, context=contextvars.Context())
        return task

    def _scope_revision(self, scope: dict):
        provider = self.evidence_revision_provider
        if not callable(provider) or not isinstance(scope, dict):
            return None
        try:
            revision = provider(scope)
            return (
                revision.get("correction_revision")
                if isinstance(revision, dict)
                else None
            )
        except Exception:
            return None

    def _make_search_context(self, *, session_id, query, persona_id, budget, req):
        if self.recall_runtime is None:
            return None
        scope = self._search_scope_identity(
            req, session_id, session_id, persona_id, None
        )
        receipt = getattr(req, "_cd_memory_feedback_v1", None)
        # Leave time for keyword reads, canonical hydration and atom projection.
        remote_deadline = budget.search_deadline - min(
            1.0, budget.search_remaining() * 0.35
        )
        return BoundedRecallSearchContext(
            self.recall_runtime,
            budget,
            scope=scope,
            query=query,
            purpose="reply",
            card_version=str(receipt.get("card_version", ""))[:256]
            if isinstance(receipt, dict)
            else "",
            correction_revision=self._scope_revision(scope),
            persona_ids=(persona_id,) if persona_id else (),
            origin="private-native",
            max_parallel=3,
            hard_deadline=True,
            remote_deadline=remote_deadline,
            embedding_cache=QueryEmbeddingCache(
                remote_deadline, owner=self.recall_runtime
            ),
            local_only=getattr(req, "_cd_local_recall_only_v1", False) is True,
        )

    @staticmethod
    def _search_scope_identity(
        req, session_id, selected_session, persona_id, shared_session
    ):
        receipt = getattr(req, "_cd_memory_feedback_v1", None)
        base = receipt.get("scope") if isinstance(receipt, dict) else None
        if isinstance(base, dict):
            scope = {
                key: base.get(key)
                for key in ("platform", "umo", "persona_id", "instance_id")
            }
        else:
            scope = {
                "platform": str(session_id).split(":", 1)[0],
                "umo": session_id,
                "persona_id": persona_id or "",
                "instance_id": persona_id or "",
            }
        target = shared_session or selected_session or session_id
        scope["umo"] = target
        return scope

    def _schedule_message_limit(self, session_id: str) -> str:
        active = self._message_limit_tasks.get(session_id)
        if active is not None and not active.done():
            return "coalesced"
        if active is not None:
            self._message_limit_tasks.pop(session_id, None)
        for key, task in tuple(self._message_limit_tasks.items()):
            if task.done():
                self._message_limit_tasks.pop(key, None)
        if len(self._message_limit_tasks) >= self._message_limit_task_limit:
            return "busy"
        create = getattr(self.memory_engine, "_create_tracked_task", None)
        if not callable(create):
            return "unavailable"
        task = create(
            self.message_utils.enforce_message_limit(session_id), clean_context=True
        )
        self._message_limit_tasks[session_id] = task

        def release(completed, key=session_id):
            if self._message_limit_tasks.get(key) is completed:
                self._message_limit_tasks.pop(key, None)

        task.add_done_callback(release, context=contextvars.Context())
        return "scheduled"

    async def _store_owned_original_messages(
        self, event, req, original, budget
    ) -> bool:
        if original.status != "valid" or not isinstance(original.scope, dict):
            return False
        if classify_umo(original.scope.get("umo")) != "private":
            return False
        request_receipt = getattr(req, "_cd_memory_feedback_v1", None)
        receipt_scope = (
            request_receipt.get("scope") if isinstance(request_receipt, dict) else None
        )
        if not isinstance(receipt_scope, dict):
            getter = getattr(event, "get_extra", None)
            context = (
                getter("_autonomous_blocking_context_v1", None)
                if callable(getter)
                else None
            )
            receipt_scope = context.get("scope") if isinstance(context, dict) else None
        if not isinstance(receipt_scope, dict):
            return False
        scope = {
            key: receipt_scope.get(key)
            for key in ("platform", "umo", "persona_id", "instance_id")
        }
        if (
            scope.get("umo") != original.scope.get("umo")
            or scope.get("platform") != str(scope.get("umo") or "").rsplit(":", 2)[0]
            or not all(
                isinstance(scope.get(key), str) and scope[key].strip()
                for key in ("platform", "umo", "persona_id", "instance_id")
            )
        ):
            return False
        self_id_getter = getattr(event, "get_self_id", None)
        try:
            self_id = self_id_getter() if callable(self_id_getter) else None
        except Exception:
            self_id = None
        if isinstance(self_id, str):
            self_id = self_id.strip()
        complete = True
        for item in original.messages:
            if item.author_id == self_id or not item.plain_text:
                continue
            if item.occurred_at is None or not item.event_key or not item.author_id:
                complete = False
                continue
            envelope = {
                "schema_version": 1,
                "type": "message.received",
                "event_id": "cd-original-input-v1:" + item.event_key,
                "scope": dict(scope),
                "actor_id": item.author_id,
                "occurred_at": item.occurred_at,
                "text": item.plain_text,
                "original_parts": [dict(part) for part in item.parts],
                "platform_message_id": item.platform_message_id,
            }
            remaining = (
                budget.search_remaining() if isinstance(budget, RecallDeadline) else 0.0
            )
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(
                self.conversation_manager.ingest_owned_private_original_event(
                    envelope, source_platform=original.scope["platform"]
                ),
                timeout=remaining,
            )
        return complete
