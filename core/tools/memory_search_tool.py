"""供 Agent 主动调用的长期记忆回忆工具。"""

import asyncio
import copy
import inspect
import time
from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass

from astrbot.api import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from ..base.config_manager import ConfigManager
from ..event_handler_modules.message_utils import discard_unexpected_awaitable
from ..evidence_projection import conversation_row_hash
from ..memory_scope import (
    adapter_access,
    classify_umo,
    is_event_memory_allowed,
    resolve_memory_access_policy,
    resolve_verified_group_sessions,
)
from ..narrative_api import is_managed_event, scope_key
from ..passive_group_capture import get_active_plugin
from ..recall_runtime import (
    BoundedRecallSearchContext,
    BoundedRecallTaskPool,
    RecallDeadline,
)
from ..retrieval.vector_search import QueryEmbeddingCache
from ..summary_source import AtomProjectionProof, summary_metadata_sha256, text_sha256
from ..tool_recall_guard import (
    make_managed_evidence_validator,
    make_memory_source_validator,
)
from ..utils import get_persona_id
from .agentic_recall_common import json_result as _raw_json_result
from .memory_evidence_output import fitting_prefix

DEFAULT_RESULT_CHAR_BUDGET = 6000
SOURCE_RESULT_CHAR_BUDGET = 12000


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[: max(0, limit)]
    return text[: limit - 1].rstrip() + "…"


def _json_result(
    payload: dict[str, Any], *, max_chars: int | None = DEFAULT_RESULT_CHAR_BUDGET
) -> str:
    """序列化工具返回并强制执行字符预算：优先保留高分记忆并截断超长文本。"""
    raw = _raw_json_result(payload)
    if max_chars is None or max_chars <= 0 or len(raw) <= max_chars:
        return raw

    bounded = copy.deepcopy(payload)
    diagnostics = bounded.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics["truncated"] = True
        diagnostics["char_budget"] = max_chars

    results = bounded.get("results")
    if isinstance(results, list) and results:
        original_count = len(results)

        # 1. 先按档位截断超长正文与原文片段，尽量保留更多高分条目
        for content_cap, source_cap, max_sources in (
            (1200, 400, 4),
            (600, 240, 3),
            (320, 160, 2),
        ):
            for item in results:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("content"), str):
                    item["content"] = _truncate_text(item["content"], content_cap)
                sources = item.get("source_messages")
                if isinstance(sources, list):
                    trimmed_sources = []
                    for msg in sources[:max_sources]:
                        if isinstance(msg, dict) and isinstance(
                            msg.get("content"), str
                        ):
                            msg["content"] = _truncate_text(msg["content"], source_cap)
                        trimmed_sources.append(msg)
                    item["source_messages"] = trimmed_sources
            encoded = _raw_json_result(bounded)
            if len(encoded) <= max_chars:
                return encoded

        # 2. 若仍超出预算，从尾部（低分记忆）开始丢弃，优先保留头部高分记忆
        while len(results) > 1 and len(_raw_json_result(bounded)) > max_chars:
            results.pop()
            bounded["count"] = len(results)
            if isinstance(diagnostics, dict):
                diagnostics["omitted_count"] = original_count - len(results)

        encoded = _raw_json_result(bounded)
        if len(encoded) <= max_chars:
            return encoded

        # 3. 即使只剩最高分 1 条仍超限：先裁剪其 source_messages，再二分截断正文
        top_item = results[0]
        if isinstance(top_item, dict):
            if isinstance(top_item.get("source_messages"), list):
                while (
                    top_item["source_messages"]
                    and len(_raw_json_result(bounded)) > max_chars
                ):
                    top_item["source_messages"].pop()
                if not top_item["source_messages"]:
                    top_item.pop("source_messages", None)
            if len(_raw_json_result(bounded)) > max_chars and isinstance(
                top_item.get("content"), str
            ):
                original_text = top_item["content"]

                def _fits_content(candidate: str) -> bool:
                    top_item["content"] = candidate
                    return len(_raw_json_result(bounded)) <= max_chars

                top_item["content"] = fitting_prefix(original_text, _fits_content)

        encoded = _raw_json_result(bounded)
        if len(encoded) <= max_chars:
            return encoded

    # 4. 兜底：若 query 过长导致超限，截断 query
    if (
        isinstance(bounded.get("query"), str)
        and len(_raw_json_result(bounded)) > max_chars
    ):
        query_text = bounded["query"]

        def _fits_query(candidate: str) -> bool:
            bounded["query"] = candidate
            return len(_raw_json_result(bounded)) <= max_chars

        bounded["query"] = fitting_prefix(query_text, _fits_query)

    return _raw_json_result(bounded)


@dataclass
class MemorySearchTool(FunctionTool[AstrAgentContext]):
    """长期记忆主动回忆工具。"""

    __pydantic_config__ = {"arbitrary_types_allowed": True}

    context: Any = None
    config_manager: ConfigManager | None = None
    memory_engine: Any = None

    name: str = "recall_long_term_memory"
    description: str = (
        "Recall long-term memory when the current context is insufficient. "
        "Use concise, focused recall keywords instead of copying the full user message. "
        "Call this when the user asks you to recall prior facts, preferences, agreements, or older context, "
        "or when resolving ambiguous references requires checking memory. "
        "Prefer short topic phrases, named entities, preferences, commitments, or past events as recall keywords. "
        "If the first recall is not enough, refine the keywords and recall again."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concise recall keywords for long-term memory. Prefer key entities, topics, preferences, commitments, or past events instead of copying the full user message.",
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum number of memory items to return for one recall. Keep this small unless more evidence is needed.",
                    "default": 5,
                },
                "include_source": {
                    "type": "boolean",
                    "description": "Include retained original messages when exact details are needed. Use only after normal recall is insufficient.",
                    "default": False,
                },
            },
            "required": ["query"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        query: str,
        k: int = 5,
        include_source: bool = False,
    ) -> ToolExecResult:
        guard = getattr(self, "_series_owner_guard", None)
        if callable(guard) and not guard():
            return "群间记忆当前不可用：记忆提供者冲突或已停用。"
        with adapter_access(self.config_manager, context.context.event):
            """Execute one bounded, readonly recall over all authorized scopes."""
            call_started = time.monotonic()
            deadline = RecallDeadline.start(
                total_seconds=6.0, final_reserve_seconds=1.0
            )
            cleaned_query = (query or "").strip()
            event_hint = getattr(getattr(context, "context", None), "event", None)
            try:
                route_kind = "managed" if is_managed_event(event_hint) else "ordinary"
            except Exception:
                route_kind = "ordinary"
            logger.info("[LMToolRecall] route=%s phase=start", route_kind)
            runtime = self.recall_runtime
            route_states: dict[str, str] = {}
            timeout_seen = False
            busy_seen = False
            closed_seen = False
            partial_seen = False

            char_budget = (
                SOURCE_RESULT_CHAR_BUDGET
                if include_source
                else DEFAULT_RESULT_CHAR_BUDGET
            )

            def reply(status: str, results=None, *, error=None, applied_filters=None):
                result_count = len(results or [])
                logger.info(
                    "[LMToolRecall] route=%s phase=finish status=%s elapsed_ms=%d count=%d",
                    route_kind,
                    status,
                    int((time.monotonic() - call_started) * 1000),
                    result_count,
                )
                payload = {
                    "status": status,
                    "query": cleaned_query,
                    "count": result_count,
                    "results": results or [],
                    "diagnostics": {
                        "elapsed_ms": int((time.monotonic() - call_started) * 1000),
                        "route_states": dict(route_states),
                        "inflight_count": (
                            runtime.stats().get("inflight_count", 0)
                            if callable(getattr(runtime, "stats", None))
                            else 0
                        ),
                    },
                }
                if error:
                    payload["error"] = error
                if applied_filters is not None:
                    payload["applied_filters"] = applied_filters
                return _json_result(payload, max_chars=char_budget)

            if not cleaned_query:
                return reply("empty", error="query is empty")
            if (
                self.config_manager is None
                or self.memory_engine is None
                or self.context is None
                or runtime is None
            ):
                return reply(
                    "unavailable", error="memory search tool is not initialized"
                )

            try:
                event = context.context.event
                session_id = str(getattr(event, "unified_msg_origin", "") or "").strip()
                if not session_id or not is_event_memory_allowed(
                    self.config_manager, event
                ):
                    return reply("unavailable", error="memory access is not allowed")

                owned_private = (
                    getattr(event, "_chat_dynamics_owned_request", None) is True
                    and getattr(event, "_cd_owned_original_input_v1", None) is True
                    and classify_umo(session_id) == "private"
                )
                extra_getter = getattr(event, "get_extra", None)
                cd_receipt = (
                    extra_getter("_cd_memory_feedback_v1", None)
                    if callable(extra_getter)
                    else None
                )
                receipt_scope = (
                    cd_receipt.get("scope") if isinstance(cd_receipt, dict) else None
                )
                scope_keys = ("platform", "umo", "persona_id", "instance_id")
                if (
                    isinstance(receipt_scope, dict)
                    and all(
                        isinstance(receipt_scope.get(key), str)
                        and receipt_scope[key].strip()
                        for key in scope_keys
                    )
                    and receipt_scope.get("umo") == session_id
                    and receipt_scope.get("platform") == session_id.rsplit(":", 2)[0]
                ):
                    request_scope = {key: receipt_scope[key] for key in scope_keys}
                else:
                    request_scope = None

                plugin = get_active_plugin()
                if (
                    plugin is not None
                    and getattr(
                        getattr(plugin, "initializer", None), "memory_engine", None
                    )
                    is not self.memory_engine
                ):
                    plugin = None
                filtering_config = getattr(
                    self.config_manager, "filtering_settings", {}
                )
                use_persona_filtering = bool(
                    filtering_config.get("use_persona_filtering", True)
                )
                group_sharing = bool(
                    filtering_config.get("enable_group_to_private_recall", False)
                )
                core_persona_id = None
                persona_id = None
                # Source scopes need the current identity even when retrieval
                # is allowed to search memories from multiple personas.
                if (
                    use_persona_filtering
                    or group_sharing
                    or owned_private
                    or plugin is not None
                ):
                    remaining = deadline.search_remaining()
                    if remaining <= 0:
                        return reply(
                            "deadline_exceeded", error="recall deadline exceeded"
                        )
                    core_persona_id = await asyncio.wait_for(
                        get_persona_id(self.context, event), timeout=remaining
                    )
                    if (
                        not isinstance(core_persona_id, str)
                        or not core_persona_id.strip()
                    ):
                        return reply("unavailable", error="persona unavailable")
                    if use_persona_filtering or group_sharing:
                        persona_id = core_persona_id
                if request_scope is None:
                    if owned_private:
                        return reply("unavailable", error="memory scope unavailable")
                    base_persona = core_persona_id or persona_id or ""
                    request_scope = {
                        "platform": session_id.rsplit(":", 2)[0],
                        "umo": session_id,
                        "persona_id": base_persona,
                        "instance_id": base_persona,
                    }

                if is_managed_event(event):
                    route_kind = "managed"
                    managed = (
                        extra_getter("_narrative_managed_v1", None)
                        if callable(extra_getter)
                        else None
                    )
                    scope = (
                        managed.get("scope", {}) if isinstance(managed, dict) else {}
                    )
                    if scope.get("umo") != session_id:
                        return reply(
                            "unavailable", error="managed memory scope unavailable"
                        )
                    try:
                        scope_key(scope)
                    except Exception:
                        return reply(
                            "unavailable", error="managed memory scope unavailable"
                        )
                    recall = getattr(plugin, "recall_context", None)
                    if not callable(recall):
                        return reply(
                            "unavailable", error="managed memory recall unavailable"
                        )
                    limited_k = max(
                        1,
                        min(
                            int(
                                k
                                if k is not None
                                else self.config_manager.get("recall_engine.top_k", 5)
                            ),
                            int(self.config_manager.get("recall_engine.max_k", 10)),
                        ),
                    )
                    remaining = min(
                        deadline.remaining(), deadline.search_remaining() + 0.25
                    )
                    if remaining <= 0:
                        return reply(
                            "deadline_exceeded", error="recall deadline exceeded"
                        )
                    response = await asyncio.wait_for(
                        recall(
                            scope,
                            cleaned_query,
                            purpose="reply",
                            card_version=str(cd_receipt.get("card_version", ""))
                            if isinstance(cd_receipt, dict)
                            else "",
                        ),
                        timeout=remaining,
                    )
                    if (
                        not isinstance(response, dict)
                        or response.get("available") is not True
                    ):
                        status = (
                            "deadline_exceeded"
                            if (
                                isinstance(response, dict)
                                and response.get("evidence_status")
                                in {"timeout", "deadline_exceeded"}
                            )
                            else "unavailable"
                        )
                        return reply(status, error="managed memory recall unavailable")
                    results = []
                    managed_guard_items = []
                    for row in response.get("items", [])[:limited_k]:
                        if not isinstance(row, dict):
                            continue
                        managed_guard_items.append(
                            {
                                "source_id": row.get("source_id"),
                                "text": row.get("text", ""),
                                "fact_kind": row.get("fact_kind"),
                                "scope_key": row.get("scope_key"),
                                "correction_revision": row.get("correction_revision"),
                                "source_refs": row.get("source_refs", []),
                                "root_evidence_ids": row.get("root_evidence_ids", []),
                                "root_material_ids": row.get("root_material_ids", []),
                            }
                        )
                        results.append(
                            {
                                "id": row.get("source_id"),
                                "content": row.get("text", ""),
                                "session_id": scope["umo"],
                                "persona_id": row.get("source_persona_id"),
                                "fact_kind": row.get("fact_kind"),
                                "unknown": row.get("unknown", False),
                                "provenance_kind": row.get("provenance_kind"),
                                "source_umo": row.get("source_umo"),
                                "create_time": row.get("occurred_at"),
                                "source_refs": row.get("source_refs", []),
                                "root_evidence_ids": row.get("root_evidence_ids", []),
                                "root_material_ids": row.get("root_material_ids", []),
                                "scope_key": row.get("scope_key"),
                                "correction_revision": row.get("correction_revision"),
                            }
                        )
                    if include_source and results:
                        partial_seen = True
                    status = response.get("evidence_status", "ready")
                    status = (
                        "partial"
                        if partial_seen or status == "partial"
                        else ("empty" if not results else "ok")
                    )
                    registrar = getattr(
                        event, "_cd_tool_memory_sources_register_v1", None
                    )
                    if results:
                        if (
                            plugin is None
                            or not core_persona_id
                            or not callable(getattr(plugin, "get_evidence_view", None))
                        ):
                            return reply(
                                "unavailable", error="source validation unavailable"
                            )
                        if owned_private and not callable(registrar):
                            return reply(
                                "unavailable", error="source validation unavailable"
                            )
                        validator = make_managed_evidence_validator(
                            plugin=plugin,
                            event=event,
                            scope=scope,
                            card_version=str(cd_receipt.get("card_version", ""))
                            if isinstance(cd_receipt, dict)
                            else "",
                            items=managed_guard_items,
                        )
                        remaining = deadline.remaining()
                        if remaining <= 0:
                            return reply(
                                "deadline_exceeded",
                                error="source validation deadline exceeded",
                            )
                        try:
                            valid = await asyncio.wait_for(
                                validator(), timeout=remaining
                            )
                        except asyncio.CancelledError:
                            raise
                        except asyncio.TimeoutError:
                            return reply(
                                "deadline_exceeded",
                                error="source validation deadline exceeded",
                            )
                        if valid is not True:
                            return reply(
                                "unavailable", error="source validation failed"
                            )
                        if callable(registrar):
                            registered = registrar(validator, tool=self)
                            if inspect.isawaitable(registered):
                                discard_unexpected_awaitable(registered)
                                registered = False
                            if registered is not True:
                                return reply(
                                    "unavailable", error="source validation unavailable"
                                )
                    return reply(
                        status,
                        results,
                        applied_filters={
                            "session_filtered": True,
                            "persona_filtered": True,
                            "narrative_managed": True,
                            "unknown_persona_allowed": True,
                        },
                    )

                if (use_persona_filtering or group_sharing) and persona_id is None:
                    return reply("unavailable", error="persona unavailable")
                default_k = int(self.config_manager.get("recall_engine.top_k", 5))
                max_k = int(self.config_manager.get("recall_engine.max_k", 10))
                try:
                    limited_k = max(1, min(int(default_k if k is None else k), max_k))
                except (TypeError, ValueError):
                    limited_k = max(1, min(default_k, max_k))

                remaining = deadline.search_remaining()
                if remaining <= 0:
                    return reply("deadline_exceeded", error="recall deadline exceeded")
                verified_groups = await asyncio.wait_for(
                    resolve_verified_group_sessions(
                        self.config_manager, event, self.conversation_manager
                    ),
                    timeout=remaining,
                )
                policy = resolve_memory_access_policy(
                    self.config_manager,
                    event,
                    persona_id=persona_id,
                    verified_group_sessions=verified_groups,
                )
                guard_persona = core_persona_id
                if plugin is not None and callable(
                    getattr(plugin, "peek_evidence_revision", None)
                ):
                    revision = plugin.peek_evidence_revision(request_scope)
                    correction_revision = (
                        revision.get("correction_revision")
                        if isinstance(revision, dict) and revision.get("available")
                        else None
                    )
                else:
                    correction_revision = None
                card_version = (
                    str(cd_receipt.get("card_version", ""))[:256]
                    if isinstance(cd_receipt, dict)
                    else ""
                )
                remote_deadline = deadline.search_deadline - min(
                    1.0, deadline.search_remaining() * 0.35
                )
                embedding_cache = QueryEmbeddingCache(remote_deadline, owner=runtime)
                search_context = BoundedRecallSearchContext(
                    runtime,
                    deadline,
                    scope=request_scope,
                    query=cleaned_query,
                    purpose="reply",
                    card_version=card_version,
                    correction_revision=correction_revision,
                    persona_ids=(guard_persona,) if guard_persona else (),
                    origin="memory-search-tool",
                    max_parallel=3,
                    hard_deadline=True,
                    remote_deadline=remote_deadline,
                    embedding_cache=embedding_cache,
                )

                memories_by_id = {}
                source_scope_by_id = {}
                search_context_by_id = {}
                readonly_by_id = {}

                scope_jobs = []
                for read_scope in policy.read_scopes:
                    scope_session_id = (
                        read_scope.session_ids[0]
                        if read_scope.session_ids and len(read_scope.session_ids) == 1
                        else None
                    )
                    source_umo = (
                        read_scope.source_session_id or scope_session_id or session_id
                    )
                    scope_persona = (
                        persona_id
                        if read_scope.require_persona
                        else (persona_id if use_persona_filtering else None)
                    )
                    interaction_type = (
                        read_scope.interaction_types[0]
                        if read_scope.interaction_types
                        and len(read_scope.interaction_types) == 1
                        else read_scope.interaction_types
                    )
                    if plugin is not None and callable(
                        getattr(plugin, "_native_scope_from_request", None)
                    ):
                        source_scope = plugin._native_scope_from_request(
                            request_scope, source_umo
                        )
                    else:
                        source_scope = {
                            "platform": source_umo.rsplit(":", 2)[0],
                            "umo": source_umo,
                            "persona_id": request_scope.get("persona_id", ""),
                            "instance_id": request_scope.get("instance_id", ""),
                        }
                    if not isinstance(source_scope, dict):
                        partial_seen = True
                        continue
                    scope_label = f"scope_{len(scope_jobs) + 1}"
                    revision_info = (
                        plugin.peek_evidence_revision(source_scope)
                        if (
                            plugin is not None
                            and callable(
                                getattr(plugin, "peek_evidence_revision", None)
                            )
                        )
                        else None
                    )
                    # 每个 scope 拥有独立的 metrics 字典，同时共享 embedding_cache、remote_deadline 与并发信号量
                    scope_context = BoundedRecallSearchContext(
                        runtime,
                        deadline,
                        scope=source_scope,
                        query=cleaned_query,
                        purpose="reply",
                        card_version=card_version,
                        correction_revision=(
                            revision_info.get("correction_revision")
                            if isinstance(revision_info, dict)
                            and revision_info.get("available")
                            else None
                        ),
                        persona_ids=search_context.persona_ids,
                        origin=search_context.origin,
                        _parallel=search_context._parallel,
                        _metrics=({}, {}, {}),
                        hard_deadline=True,
                        embedding_cache=embedding_cache,
                        local_only=search_context.local_only,
                        adapter_access_snapshot=search_context.adapter_access_snapshot,
                        remote_deadline=remote_deadline,
                        _remote_parallel=search_context._remote_parallel,
                    )
                    search_kwargs = {
                        "query": cleaned_query,
                        "k": limited_k,
                        "session_id": scope_session_id,
                        "persona_id": scope_persona,
                        "touch_access": False,
                        "cache_results": False,
                        "allow_legacy_session_migration": False,
                        "execution_context": scope_context,
                    }
                    if interaction_type is not None:
                        search_kwargs["interaction_type"] = interaction_type
                    if read_scope.source_platform is not None:
                        search_kwargs["source_platform"] = read_scope.source_platform
                    if read_scope.source_session_id is not None:
                        search_kwargs["source_session_id"] = (
                            read_scope.source_session_id
                        )
                    if read_scope.excluded_session_ids:
                        search_kwargs["excluded_session_ids"] = (
                            read_scope.excluded_session_ids
                        )
                    scope_jobs.append(
                        (
                            scope_label,
                            read_scope,
                            source_scope,
                            scope_context,
                            search_kwargs,
                        )
                    )

                if scope_jobs:
                    if deadline.search_remaining() <= 0:
                        timeout_seen = True
                        route_states[scope_jobs[0][0]] = "deadline_exceeded"
                    else:
                        scope_sem = asyncio.Semaphore(min(3, len(scope_jobs)))

                        async def _run_one_scope(job):
                            (
                                s_label,
                                r_scope,
                                s_scope,
                                s_ctx,
                                s_kwargs,
                            ) = job
                            async with scope_sem:
                                rem = min(
                                    deadline.remaining(),
                                    max(0.0, deadline.search_remaining() + 0.25),
                                )
                                if rem <= 0:
                                    return (
                                        s_label,
                                        r_scope,
                                        s_scope,
                                        s_ctx,
                                        "deadline_exceeded",
                                        [],
                                    )
                                try:
                                    found = await asyncio.wait_for(
                                        self.memory_engine.search_memories(**s_kwargs),
                                        timeout=rem,
                                    )
                                except asyncio.CancelledError:
                                    raise
                                except asyncio.TimeoutError:
                                    return (
                                        s_label,
                                        r_scope,
                                        s_scope,
                                        s_ctx,
                                        "deadline_exceeded",
                                        [],
                                    )
                                except Exception as exc:
                                    logger.info(
                                        "[LMToolRecall] route=search phase=scope outcome=error elapsed_ms=%d exception_class=%s",
                                        int((time.monotonic() - call_started) * 1000),
                                        type(exc).__name__,
                                    )
                                    return (
                                        s_label,
                                        r_scope,
                                        s_scope,
                                        s_ctx,
                                        "error",
                                        [],
                                    )
                                return (
                                    s_label,
                                    r_scope,
                                    s_scope,
                                    s_ctx,
                                    "completed",
                                    found if isinstance(found, list) else [],
                                )

                        scope_tasks = [
                            asyncio.create_task(_run_one_scope(job))
                            for job in scope_jobs
                        ]
                        outer_timeout = min(
                            deadline.remaining(),
                            max(0.0, deadline.search_remaining() + 0.25),
                        )
                        try:
                            done_tasks, pending_tasks = await asyncio.wait(
                                scope_tasks, timeout=outer_timeout
                            )
                        finally:
                            for task in scope_tasks:
                                if not task.done():
                                    task.cancel()
                            pending_cleanup = [t for t in scope_tasks if not t.done()]
                            if pending_cleanup:
                                await asyncio.gather(
                                    *pending_cleanup, return_exceptions=True
                                )

                        if pending_tasks:
                            timeout_seen = True

                        for job, task in zip(scope_jobs, scope_tasks):
                            s_label = job[0]
                            if task not in done_tasks:
                                timeout_seen = True
                                route_states[s_label] = "deadline_exceeded"
                                continue
                            if task.cancelled():
                                raise asyncio.CancelledError
                            try:
                                (
                                    s_label,
                                    r_scope,
                                    s_scope,
                                    s_ctx,
                                    outcome_state,
                                    found,
                                ) = task.result()
                            except Exception:
                                route_states[s_label] = "error"
                                continue

                            if outcome_state == "deadline_exceeded":
                                timeout_seen = True
                                route_states[s_label] = "deadline_exceeded"
                                continue
                            if outcome_state == "error":
                                route_states[s_label] = "error"
                                continue

                            failures = dict(s_ctx.route_failures)
                            if failures:
                                route_states[s_label] = next(iter(failures.values()))
                                timeout_seen = timeout_seen or any(
                                    value == "timeout" for value in failures.values()
                                )
                                busy_seen = busy_seen or any(
                                    value == "busy" for value in failures.values()
                                )
                                closed_seen = closed_seen or any(
                                    value == "closed" for value in failures.values()
                                )
                                partial_seen = True
                            else:
                                route_states[s_label] = "completed"

                            for memory in found:
                                memory_id = getattr(memory, "doc_id", None)
                                if isinstance(memory_id, bool) or not isinstance(
                                    memory_id, int
                                ):
                                    continue
                                previous = memories_by_id.get(memory_id)
                                if (
                                    previous is None
                                    or memory.final_score > previous.final_score
                                ):
                                    memories_by_id[memory_id] = memory
                                    source_scope_by_id[memory_id] = dict(s_scope)
                                    search_context_by_id[memory_id] = s_ctx
                                    readonly_by_id[memory_id] = bool(r_scope.read_only)

                memories = sorted(
                    memories_by_id.values(),
                    key=lambda memory: memory.final_score,
                    reverse=True,
                )[:limited_k]
                plugin_filtering = bool(
                    filtering_config.get("use_persona_filtering", True)
                )
                document_snapshots = []
                source_row_snapshots = []
                serialized_results = []
                valid_entries: list[tuple[Any, dict, dict, str, dict]] = []
                for memory in memories:
                    metadata = (
                        memory.metadata if isinstance(memory.metadata, dict) else {}
                    )
                    source_scope = source_scope_by_id.get(memory.doc_id)
                    source_umo = metadata.get("source_session_id") or metadata.get(
                        "session_id"
                    )
                    if (
                        not isinstance(source_scope, dict)
                        or not isinstance(source_umo, str)
                        or source_scope.get("umo") != source_umo
                    ):
                        partial_seen = True
                        continue
                    content = memory.content if isinstance(memory.content, str) else ""
                    item = {
                        "id": memory.doc_id,
                        "content": content,
                        "score": memory.final_score,
                        "importance": metadata.get("importance"),
                        "session_id": metadata.get("session_id"),
                        "persona_id": metadata.get("persona_id"),
                        "create_time": metadata.get("create_time"),
                        "last_access_time": metadata.get("last_access_time"),
                    }
                    if "interaction_type" in metadata:
                        item["interaction_type"] = metadata.get("interaction_type")
                    if "source_session_id" in metadata:
                        item["source_session_id"] = metadata.get("source_session_id")
                    revision_info = (
                        plugin.peek_evidence_revision(source_scope)
                        if (
                            plugin is not None
                            and callable(
                                getattr(plugin, "peek_evidence_revision", None)
                            )
                        )
                        else None
                    )
                    projection = getattr(memory, "atom_projection", None)
                    if not isinstance(projection, AtomProjectionProof):
                        projection = None
                    content_hash = text_sha256(content)
                    document_snapshots.append(
                        {
                            "document_id": memory.doc_id,
                            "document_text_sha256": projection.document_sha256
                            if projection
                            else content_hash,
                            "projected_text_sha256": content_hash,
                            "atom_proof": projection.atoms if projection else (),
                            "summary_metadata_sha256": summary_metadata_sha256(
                                metadata
                            ),
                            "source_scope": dict(source_scope),
                            "source_session_id": source_umo,
                            "source_platform": metadata.get("source_platform"),
                            "interaction_type": metadata.get("interaction_type"),
                            "persona_id": metadata.get("persona_id"),
                            "memory_visibility": metadata.get("memory_visibility"),
                            "status": metadata.get("status", "active"),
                            "read_only": readonly_by_id.get(memory.doc_id, False),
                            "scope_key": scope_key(source_scope),
                            "correction_revision": (
                                revision_info.get("correction_revision")
                                if isinstance(revision_info, dict)
                                and revision_info.get("available")
                                else None
                            ),
                        }
                    )
                    valid_entries.append(
                        (memory, metadata, source_scope, source_umo, item)
                    )
                    serialized_results.append(item)

                if include_source and valid_entries:
                    can_load_sources = (
                        plugin is not None
                        and callable(
                            getattr(plugin, "_native_memory_source_row_ids", None)
                        )
                        and callable(
                            getattr(plugin, "_native_verified_source_row", None)
                        )
                    )
                    if not can_load_sources:
                        partial_seen = True
                    else:
                        source_sem = asyncio.Semaphore(4)

                        async def _fetch_source_ids(entry_idx, entry):
                            mem, meta, _s_scope, s_umo, _item = entry
                            async with source_sem:
                                if deadline.search_remaining() <= 0:
                                    return entry_idx, "timeout", []
                                source_context = search_context_by_id.get(mem.doc_id)
                                if source_context is None:
                                    return entry_idx, "timeout", []
                                try:
                                    ids_outcome = await source_context.run(
                                        "tool-source-ids",
                                        plugin._native_memory_source_row_ids,
                                        self.memory_engine,
                                        mem.doc_id,
                                        dict(meta),
                                        s_umo,
                                        discriminator=mem.doc_id,
                                    )
                                except asyncio.CancelledError:
                                    raise
                                except asyncio.TimeoutError:
                                    return entry_idx, "timeout", []
                                except Exception:
                                    return entry_idx, "error", []
                                state = str(ids_outcome.get("state", "error"))
                                if state != "completed":
                                    return entry_idx, state, []
                                return (
                                    entry_idx,
                                    "completed",
                                    list(ids_outcome.get("value") or []),
                                )

                        id_results = await asyncio.gather(
                            *(
                                _fetch_source_ids(idx, entry)
                                for idx, entry in enumerate(valid_entries)
                            )
                        )

                        # 按高分记忆顺序分配全局最多 32 条原文行（每条记忆最多 4 条），再受控并发批量拉取
                        row_jobs: list[tuple[int, Any, dict, Any]] = []
                        planned_total = 0
                        for entry_idx, state, source_ids in id_results:
                            if state != "completed":
                                route_states["source_ids"] = state
                                timeout_seen |= state == "timeout"
                                busy_seen |= state == "busy"
                                closed_seen |= state == "closed"
                                partial_seen = True
                                continue
                            route_states["source_ids"] = "completed"
                            if not source_ids:
                                partial_seen = True
                                continue
                            mem, _meta, s_scope, _s_umo, _item = valid_entries[
                                entry_idx
                            ]
                            allowance = max(0, min(4, 32 - planned_total))
                            for row_id in source_ids[:allowance]:
                                row_jobs.append((entry_idx, mem, s_scope, row_id))
                                planned_total += 1

                        async def _fetch_verified_row(job):
                            entry_idx, mem, s_scope, row_id = job
                            async with source_sem:
                                if deadline.search_remaining() <= 0:
                                    return (
                                        entry_idx,
                                        mem,
                                        s_scope,
                                        row_id,
                                        "timeout",
                                        None,
                                    )
                                source_context = search_context_by_id.get(mem.doc_id)
                                if source_context is None:
                                    return (
                                        entry_idx,
                                        mem,
                                        s_scope,
                                        row_id,
                                        "timeout",
                                        None,
                                    )
                                try:
                                    row_outcome = await source_context.run(
                                        "tool-source-row",
                                        plugin._native_verified_source_row,
                                        s_scope,
                                        mem.doc_id,
                                        row_id,
                                        persona_id=guard_persona,
                                        persona_filtering=plugin_filtering,
                                        discriminator=(mem.doc_id, row_id),
                                    )
                                except asyncio.CancelledError:
                                    raise
                                except asyncio.TimeoutError:
                                    return (
                                        entry_idx,
                                        mem,
                                        s_scope,
                                        row_id,
                                        "timeout",
                                        None,
                                    )
                                except Exception:
                                    return (
                                        entry_idx,
                                        mem,
                                        s_scope,
                                        row_id,
                                        "error",
                                        None,
                                    )
                                state = str(row_outcome.get("state", "error"))
                                if state != "completed":
                                    return entry_idx, mem, s_scope, row_id, state, None
                                return (
                                    entry_idx,
                                    mem,
                                    s_scope,
                                    row_id,
                                    "completed",
                                    row_outcome.get("value"),
                                )

                        if row_jobs:
                            fetched_rows = await asyncio.gather(
                                *(_fetch_verified_row(job) for job in row_jobs)
                            )
                            source_messages_by_entry: dict[int, list[dict]] = {}
                            for (
                                entry_idx,
                                mem,
                                s_scope,
                                row_id,
                                state,
                                row,
                            ) in fetched_rows:
                                if state != "completed":
                                    route_states["source_rows"] = state
                                    timeout_seen |= state == "timeout"
                                    busy_seen |= state == "busy"
                                    closed_seen |= state == "closed"
                                    partial_seen = True
                                    continue
                                route_states["source_rows"] = "completed"
                                if (
                                    not isinstance(row, dict)
                                    or row.get("role") != "user"
                                ):
                                    partial_seen = True
                                    continue
                                try:
                                    row_hash = conversation_row_hash(row)
                                except Exception:
                                    partial_seen = True
                                    continue
                                source_row_snapshots.append(
                                    {
                                        "document_id": mem.doc_id,
                                        "scope": dict(s_scope),
                                        "scope_key": scope_key(s_scope),
                                        "row_id": row_id,
                                        "row_hash": row_hash,
                                        "read_only": readonly_by_id.get(
                                            mem.doc_id, False
                                        ),
                                    }
                                )
                                source_messages_by_entry.setdefault(
                                    entry_idx, []
                                ).append(
                                    {
                                        key: row.get(key)
                                        for key in (
                                            "role",
                                            "content",
                                            "session_id",
                                            "sender_id",
                                            "sender_name",
                                            "platform",
                                            "timestamp",
                                        )
                                    }
                                )
                            for entry_idx, msgs in source_messages_by_entry.items():
                                if msgs:
                                    valid_entries[entry_idx][4]["source_messages"] = (
                                        msgs
                                    )

                registrar = getattr(event, "_cd_tool_memory_sources_register_v1", None)
                if serialized_results:
                    if (
                        plugin is None
                        or not isinstance(guard_persona, str)
                        or not guard_persona.strip()
                        or not document_snapshots
                        or not callable(getattr(plugin, "_native_read_scopes", None))
                        or not callable(
                            getattr(plugin, "_native_verified_source_row", None)
                        )
                    ):
                        return reply(
                            "unavailable", error="source validation unavailable"
                        )
                    if owned_private and not callable(registrar):
                        return reply(
                            "unavailable", error="source validation unavailable"
                        )
                    validator = make_memory_source_validator(
                        plugin=plugin,
                        event=event,
                        engine=self.memory_engine,
                        request_scope=request_scope,
                        core_persona_id=guard_persona,
                        persona_filtering=plugin_filtering,
                        document_snapshots=document_snapshots,
                        source_row_snapshots=source_row_snapshots,
                        owned_private=owned_private,
                        cd_receipt=cd_receipt,
                    )
                    remaining = deadline.remaining()
                    if remaining <= 0:
                        return reply(
                            "deadline_exceeded",
                            error="source validation deadline exceeded",
                        )
                    try:
                        valid = await asyncio.wait_for(validator(), timeout=remaining)
                    except asyncio.CancelledError:
                        raise
                    except asyncio.TimeoutError:
                        return reply(
                            "deadline_exceeded",
                            error="source validation deadline exceeded",
                        )
                    if valid is not True:
                        return reply("unavailable", error="source validation failed")
                    if callable(registrar):
                        try:
                            registered = registrar(validator, tool=self)
                        except Exception:
                            registered = False
                        if inspect.isawaitable(registered):
                            discard_unexpected_awaitable(registered)
                            registered = False
                        if registered is not True:
                            return reply(
                                "unavailable", error="source validation unavailable"
                            )

                if timeout_seen:
                    status = "partial" if serialized_results else "deadline_exceeded"
                elif closed_seen:
                    status = "partial" if serialized_results else "unavailable"
                elif busy_seen:
                    status = "partial" if serialized_results else "busy"
                elif partial_seen or any(
                    state == "error" for state in route_states.values()
                ):
                    status = "partial" if serialized_results else "unavailable"
                else:
                    status = "ok" if serialized_results else "empty"
                return reply(
                    status,
                    serialized_results,
                    applied_filters={
                        "session_filtered": any(
                            session_id in (scope.session_ids or ())
                            or scope.source_session_id
                            for scope in policy.read_scopes
                        ),
                        "persona_filtered": use_persona_filtering,
                    },
                )
            except asyncio.CancelledError:
                logger.info(
                    "[LMToolRecall] route=%s phase=finish status=cancelled elapsed_ms=%d exception_class=CancelledError",
                    route_kind,
                    int((time.monotonic() - call_started) * 1000),
                )
                raise
            except asyncio.TimeoutError:
                return reply("deadline_exceeded", error="recall deadline exceeded")
            except Exception as exc:
                logger.info(
                    "[LMToolRecall] route=%s phase=execute status=error elapsed_ms=%d exception_class=%s",
                    route_kind,
                    int((time.monotonic() - call_started) * 1000),
                    type(exc).__name__,
                )
                return reply("unavailable", error="internal_error")

    conversation_manager: Any = None

    recall_runtime: Any = field(
        default_factory=lambda: BoundedRecallTaskPool(max_inflight=16, max_cache=32)
    )
