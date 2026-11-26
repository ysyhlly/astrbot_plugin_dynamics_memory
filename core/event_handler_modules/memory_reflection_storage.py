"""记忆反思的后台存储任务：LLM 总结、游标提交、回滚补偿与修订号校验。"""

import asyncio
import inspect
import time
from contextlib import nullcontext
from typing import Any

from astrbot.api import logger

from ...storage.conversation_summary_progress import (
    SUMMARY_COOLDOWN_SECONDS,
    SUMMARY_RETRY_LIMIT,
    TERMINAL_SUMMARY_STAGES,
)
from ..base.exceptions import LowQualityMemoryError, MemoryParseError
from ..memory_scope import classify_umo
from ..memory_source import (
    build_source_row_refs_v1,
    serialize_source_messages,
    summary_window_fingerprint,
)
from ..models.conversation_models import Message

_DEFAULT_MEMORY_SCOPE = object()
MANAGED_WINDOW_REFUSED = object()
_SUMMARY_RETRY_BACKOFF_SECONDS = (60, 300, 900)


class MemoryReflectionStorageMixin:
    """后台总结任务与提交补偿；宿主状态由 MemoryReflection 持有。"""

    async def _persist_summary_memory(
        self,
        *,
        session_id: str,
        memory_scope: str,
        persona_id: Any,
        content: str,
        metadata: dict[str, Any],
        importance: float,
        atoms: list,
        source_messages: list[dict[str, Any]] | None,
        source_validator=None,
        source_publish_lease=None,
    ) -> tuple[int | None, bool, float | None]:
        """Store a summary memory, merging into a near-duplicate when configured.

        Args:
            session_id: Raw conversation session id (for logging/compensation).
            memory_scope: Resolved memory scope stored in metadata.
            persona_id: Persona id for the summary.
            content: Generated summary text.
            metadata: Structured summary metadata.
            importance: Summary importance.
            atoms: Classified memory atoms.
            source_messages: Optional retained source messages.

        Returns:
            ``(memory_id, created, similarity)``. ``created`` is False when the
            summary was merged into an existing memory, so rollback compensation
            must not delete that pre-existing memory.
        """
        lock = (
            self.consolidation_manager.write_lock
            if self.consolidation_manager is not None
            else nullcontext()
        )
        async with lock:
            engine = self.memory_engine
            if engine is None:
                return None, False, None

            enabled = True
            threshold = 0.92
            limit = 5
            try:
                enabled = bool(self.config_manager.get("memory_dedup.enabled", True))
                threshold = float(
                    self.config_manager.get("memory_dedup.similarity_threshold", 0.92)
                    or 0.92
                )
                limit = int(
                    self.config_manager.get("memory_dedup.max_candidates", 5) or 5
                )
            except (TypeError, ValueError):
                enabled, threshold, limit = True, 0.92, 5

            # 去重检测算过的查询向量直接复用给写入路径，避免同一文本二次 embedding
            embedding: list[float] | None = None
            if enabled:
                try:
                    similar = await engine.find_similar_memory(
                        content,
                        session_id=memory_scope,
                        persona_id=persona_id,
                        threshold=threshold,
                        limit=limit,
                        return_embedding=True,
                    )
                    # 兼容不支持 return_embedding 的旧引擎/测试替身
                    duplicate, embedding = (
                        similar if isinstance(similar, tuple) else (similar, None)
                    )
                    if duplicate is not None:
                        merge = (
                            engine.merge_duplicate_memory
                            if source_validator is None
                            else engine.merge_guarded_summary_memory
                        )
                        merged_id = await merge(
                            int(duplicate["id"]),
                            content=content,
                            importance=importance,
                            metadata=metadata,
                            atoms=atoms,
                            source_messages=source_messages,
                            **(
                                {
                                    "source_validator": source_validator,
                                    "source_publish_lease": source_publish_lease,
                                    "session_id": memory_scope,
                                    "persona_id": persona_id,
                                }
                                if source_validator is not None
                                else {}
                            ),
                        )
                        if type(merged_id) is int and merged_id > 0:
                            return (
                                int(merged_id),
                                False,
                                float(duplicate.get("similarity") or 0.0),
                            )
                        logger.warning(
                            f"[{session_id}] 近重复记忆合并失败，改为新增记忆"
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        f"[{session_id}] 近重复检测不可用，按新记忆写入", exc_info=True
                    )

            memory_id = await engine.add_memory(
                content=content,
                session_id=memory_scope,
                persona_id=persona_id,
                importance=importance,
                metadata=metadata,
                atoms=atoms,
                source_messages=source_messages,
                embedding=embedding,
                **(
                    {
                        "source_validator": source_validator,
                        "source_publish_lease": source_publish_lease,
                        "source_session_id": session_id,
                    }
                    if source_validator is not None
                    else {}
                ),
            )
            return memory_id, True, None

    async def _read_session_revision(self, session_id: str) -> int | None:
        """读取会话修订号；不可用（老版本/Mock）时返回 None，退化为不校验。"""
        manager = self.conversation_manager
        reader = getattr(manager, "get_session_revision", None) if manager else None
        if not callable(reader):
            return None
        try:
            value = await reader(session_id)
        except Exception:
            return None
        return value if isinstance(value, int) else None

    async def _revision_still_valid(
        self, session_id: str, expected_revision: int | None
    ) -> bool:
        """本任务是否仍然有效（会话未被回滚/清空）。"""
        if expected_revision is None:
            return True
        current = await self._read_session_revision(session_id)
        return current is not None and current == expected_revision

    async def _compensate_rolled_back_memory(
        self,
        session_id: str,
        memory_id: int,
        *,
        merged_snapshot: str | None = None,
    ) -> None:
        """回滚后补偿删除刚落库的记忆（关闭"校验通过 -> 写入完成"之间的窗口）。

        游标更新会被修订号校验正确拒绝，但内容此时已经 durable；仅靠"写入前校验"
        无法撤回，因此这里做补偿删除，避免已撤销内容留在长期记忆里。
        """
        lock = (
            self.consolidation_manager.write_lock
            if self.consolidation_manager is not None
            else nullcontext()
        )
        async with lock:
            if not self.memory_engine:
                return
            try:
                if merged_snapshot is None:
                    removed = await self.memory_engine.delete_memory(memory_id)
                else:
                    removed = await self.memory_engine.rollback_guarded_summary_merge(
                        memory_id,
                        merged_snapshot,
                    )
            except Exception as exc:
                logger.error(
                    f"[{session_id}] 回滚补偿删除记忆 {memory_id} 失败: {exc}",
                    exc_info=True,
                )
                return
            if not removed:
                logger.error(
                    f"[{session_id}] 回滚补偿删除记忆 {memory_id} 未生效"
                    f"（removed={removed}），已撤销内容可能仍可被检索"
                )
                return
            logger.warning(
                f"[{session_id}] 会话已回滚，已补偿删除刚落库的记忆 {memory_id}"
                f"，避免已撤销内容污染长期记忆"
            )

    async def _storage_task(
        self,
        session_id: str,
        history_messages: list[dict],
        persona_id: str,
        start_index: int,
        end_index: int,
        retry_count: int,
        memory_scope: str | None | object = _DEFAULT_MEMORY_SCOPE,
        expected_revision: int | None = None,
        *,
        retry_refused: bool = False,
    ) -> bool:
        """总结并提交一个窗口片段；返回是否成功推进总结游标。"""
        from ..utils import OperationContext

        if not history_messages:
            return False
        window_last_message = history_messages[-1]
        if (
            type(getattr(window_last_message, "id", None)) is not int
            or window_last_message.id <= 0
        ):
            return False
        window_fingerprint = summary_window_fingerprint(history_messages)
        window_messages = history_messages
        managed_scope = session_id.startswith("narrative:v1:")

        if memory_scope is _DEFAULT_MEMORY_SCOPE:
            memory_scope = session_id

        # 会话修订号：WebUI 编辑/重试回滚或清空会话会推进修订号，使本任务
        # （持有回滚前的 history_messages 与总结范围）失效。
        # 调用方应在读取总结范围前捕获并传入；未传入时在此处补读。
        # 修订号不可用时（老版本会话管理器/测试替身）退化为不校验。
        if expected_revision is None:
            expected_revision = await self._read_session_revision(session_id)

        async def commit_progress():
            if managed_scope:
                return await self.conversation_manager.write_managed_summary_progress(
                    session_id,
                    window_messages,
                    None,
                    expected_revision,
                    retry_refused=retry_refused,
                )
            return await self.conversation_manager.commit_summary_cursor(
                session_id,
                window_last_message,
                expected_revision,
                window_messages=window_messages,
            )

        async def record_failure(*, failure_stage="model", error_code=None):
            return await self._record_pending_summary(
                session_id,
                start_index,
                end_index,
                retry_count,
                expected_revision,
                failure_stage=failure_stage,
                error_code=error_code,
                window_messages=window_messages,
                retry_refused=retry_refused,
            )

        # 记忆 id 在 try 外初始化：取消可能发生在赋值之前，异常分支需要它
        memory_id: int | None = None
        memory_created = False
        async with OperationContext("记忆存储", session_id):
            try:
                # 如果其他任务已经推进了总结进度，本任务可能已过期，直接跳过
                current_summarized = (
                    await self.conversation_manager.get_session_metadata(
                        session_id, "last_summarized_index", 0
                    )
                )
                try:
                    summarized_index = int(current_summarized)
                except (TypeError, ValueError):
                    summarized_index = 0

                if not managed_scope and summarized_index >= end_index:
                    logger.info(
                        f"[{session_id}] 检测到过期总结任务，跳过: "
                        f"current={summarized_index}, target_end={end_index}"
                    )
                    return
                if not await self._revision_still_valid(session_id, expected_revision):
                    return False
                finder = getattr(self.memory_engine, "stored_summary_window", None)
                if inspect.iscoroutinefunction(finder):
                    existing = await finder(
                        session_id,
                        start_index,
                        end_index,
                        snapshot_fingerprint=window_fingerprint,
                    )
                    if type(existing) is int and existing > 0:
                        committed = await commit_progress()
                        logger.info(
                            "Summary snapshot already stored; reused without another API call"
                        )
                        return committed

                # A previously generated assistant row is eligible only while
                # its original request-bound source capability remains in this
                # process-local, bounded lineage map. Keep the historical row
                # in SQLite, but do not feed an unverifiable draft to a later
                # reflection attempt after reload or LRU eviction.
                history_messages = self._eligible_reflection_messages(history_messages)
                if not history_messages:
                    return False if managed_scope else await commit_progress()
                if not await self._history_rows_current(
                    session_id, history_messages
                ) or not await self._owned_source_lineages_current(history_messages):
                    logger.warning(
                        "[LMReflection] route=source_revalidate outcome=skip"
                    )
                    await record_failure(
                        failure_stage="source_unavailable", error_code="source_changed"
                    )
                    return

                # 判断是否为群聊
                is_group_chat = bool(
                    history_messages[0].group_id if history_messages else False
                )
                # 备用判断：从 session_id 解析（防御性编程）
                if not is_group_chat and "GroupMessage" in session_id:
                    is_group_chat = True

                logger.info(
                    f"[{session_id}] 开始处理记忆，类型={'群聊' if is_group_chat else '私聊'}, "
                    f"范围=[{start_index}:{end_index}], 重试次数={retry_count}, "
                    f"当前人格={persona_id or '未设置'}"
                )

                # 使用 MemoryProcessor 处理对话历史
                if not self.memory_processor:
                    logger.error(f"[{session_id}] MemoryProcessor 未初始化，记录待重试")
                    await record_failure()
                    return False

                try:
                    logger.info(
                        f"[{session_id}] 调用 MemoryProcessor 处理 {len(history_messages)} 条消息"
                    )
                    # Carry only a known rejection code from the exact pending
                    # window. Never treat a rejected draft as source evidence.
                    quality_options = {}
                    pending = await self.conversation_manager.get_session_metadata(
                        session_id, "pending_summary", None
                    )
                    if (
                        isinstance(pending, dict)
                        and pending.get("start_index") == start_index
                        and pending.get("end_index") == end_index
                        and pending.get("error_code") == "summary_generic_reference"
                    ):
                        quality_options["quality_feedback"] = (
                            "summary_generic_reference"
                        )
                    prev_exhausted_cycles = (
                        int(pending.get("exhausted_cycles", 0) or 0)
                        if isinstance(pending, dict)
                        and pending.get("start_index") == start_index
                        and pending.get("end_index") == end_index
                        else 0
                    )
                    if (
                        retry_count >= SUMMARY_RETRY_LIMIT - 1
                        or prev_exhausted_cycles >= 1
                    ):
                        proc_fn = getattr(
                            self.memory_processor, "process_conversation", None
                        )
                        try:
                            sig_params = inspect.signature(proc_fn).parameters
                            accepts_degraded = (
                                "allow_degraded_quality" in sig_params
                                or any(
                                    p.kind == inspect.Parameter.VAR_KEYWORD
                                    for p in sig_params.values()
                                )
                            )
                        except (TypeError, ValueError):
                            accepts_degraded = False
                        if accepts_degraded:
                            quality_options["allow_degraded_quality"] = True
                    (
                        content,
                        metadata,
                        importance,
                    ) = await self.memory_processor.process_conversation(
                        messages=history_messages,
                        is_group_chat=is_group_chat,
                        persona_id=persona_id,
                        **quality_options,
                    )

                    # Visibility is derived from the trusted originating UMO,
                    # never from model-produced summary metadata. Private-derived
                    # summaries stay private even when their text cites a group.
                    source_kind = classify_umo(session_id)
                    metadata["interaction_type"] = {
                        "group": "group_chat",
                        "private": "private_chat",
                    }.get(source_kind, "unknown")
                    metadata["memory_visibility"] = (
                        "group" if source_kind == "group" else "private"
                    )
                    metadata["source_platform"] = (
                        session_id.split(":", 1)[0]
                        if source_kind == "group" and ":" in session_id
                        else ""
                    )

                    atoms = self.memory_processor.classify_atoms_from_metadata(
                        metadata=metadata,
                        parent_importance=importance,
                        session_id=memory_scope,
                        persona_id=persona_id,
                    )

                    # 补充 source_window 元数据，记录本次总结的消息范围
                    metadata["source_window"] = {
                        "session_id": session_id,
                        "start_index": start_index,
                        "end_index": end_index,
                        "message_count": end_index - start_index,
                        "snapshot_fingerprint": window_fingerprint,
                        "end_message_id": window_last_message.id,
                        "end_message_timestamp": window_last_message.timestamp,
                    }
                    metadata["source_session_id"] = session_id
                    if session_id.startswith("narrative:v1:"):
                        # Preserve evidence types through compression. A model
                        # summary remains an interpretation, never a new fact.
                        # Cap provenance list size so huge windows cannot bloat documents.metadata.
                        provenance_rows = [
                            {
                                "source_id": message.metadata.get("source_event_id"),
                                "scope": message.metadata.get("scope"),
                                "fact_kind": message.metadata.get("fact_kind"),
                                "occurred_at": message.timestamp,
                            }
                            for message in history_messages
                        ]
                        if len(provenance_rows) > 64:
                            provenance_rows = provenance_rows[-64:]
                        metadata["narrative_provenance"] = provenance_rows
                        metadata["fact_kind"] = "subjective"
                        metadata["unknown"] = True

                    # Persist body-free locators for the exact private user
                    # rows selected above. This is derived from the stored
                    # inputs to this summary, never from LLM-produced metadata.
                    if source_kind == "private":
                        metadata["source_row_refs_v1"] = build_source_row_refs_v1(
                            history_messages, session_id
                        )
                    else:
                        metadata.pop("source_row_refs_v1", None)

                    logger.info(
                        f"[{session_id}] 已使用LLM生成结构化记忆, "
                        f"主题={metadata.get('topics', [])}, "
                        f"重要性={importance:.2f}"
                    )

                except (MemoryParseError, LowQualityMemoryError) as e:
                    # 解析失败/低质量总结：不落库，保留窗口按现有 pending retry
                    # 机制重试；达到上限后暂缓并保留同一个窗口。
                    logger.warning(
                        f"[{session_id}] 记忆总结被拒绝（可重试）: "
                        f"reason={e.error_code}, "
                        f"range=[{start_index}:{end_index}], "
                        f"retry={retry_count + 1}/3"
                    )
                    await record_failure(error_code=e.error_code)
                    return False
                except Exception as e:
                    # LLM处理失败，记录待重试信息
                    logger.error(
                        f"[{session_id}] LLM处理失败 (重试 {retry_count + 1}/3): {e}",
                        exc_info=True,
                    )
                    from ..utils.api_failure import terminal_provider_failure

                    terminal_code = terminal_provider_failure(e)
                    recorded = await record_failure(
                        failure_stage="provider_terminal" if terminal_code else "model",
                        error_code=terminal_code or type(e).__name__,
                    )
                    if (
                        managed_scope
                        and terminal_code
                        and recorded
                        and not retry_refused
                    ):
                        return MANAGED_WINDOW_REFUSED
                    return False

                # 回滚会推进会话修订号：等待 LLM 期间若发生编辑/重试回滚，
                # 本任务的消息范围已失效，不得把被撤销内容写入长期记忆
                if not await self._revision_still_valid(session_id, expected_revision):
                    logger.info(
                        f"[{session_id}] 会话已回滚（修订号变化），放弃写入过期总结 "
                        f"范围=[{start_index}:{end_index}]"
                    )
                    return False

                # 正常流程：写入记忆引擎；近重复检测命中时合并到已有记忆
                memory_id: int | None = None
                if self.memory_engine:
                    # The model may finish after the captured source rows have
                    # changed. Re-read each exact stored row immediately before
                    # committing derived memory; missing or changed input cannot
                    # be replaced by prompt text or a new timestamp.
                    if not await self._history_rows_current(
                        session_id, history_messages
                    ) or not await self._owned_source_lineages_current(
                        history_messages
                    ):
                        logger.warning(
                            "[LMReflection] route=source_revalidate outcome=skip"
                        )
                        await record_failure(
                            failure_stage="source_unavailable",
                            error_code="source_changed",
                        )
                        return
                    source_threshold = float(
                        self.config_manager.get(
                            "reflection_engine.source_retention_importance_threshold",
                            0.8,
                        )
                    )
                    source_messages = (
                        serialize_source_messages(history_messages)
                        if importance >= source_threshold
                        or session_id.startswith("narrative:v1:")
                        else None
                    )

                    async def source_snapshot_current_at_commit():
                        # The cursor covers every stored row in the window.
                        # Only messages sent to the model require live lineage.
                        if not await self._revision_still_valid(
                            session_id, expected_revision
                        ) or not await self._history_rows_current(
                            session_id, window_messages
                        ):
                            return False
                        if not managed_scope:
                            store = self.conversation_manager.store
                            session = await store.get_session(session_id)
                            if (
                                session is None
                                or await store.native_summary_window_start(
                                    session_id,
                                    window_messages,
                                    session.metadata,
                                )
                                is None
                            ):
                                return False
                        return await self._owned_source_lineages_current(
                            history_messages
                        )

                    (
                        memory_id,
                        memory_created,
                        dedup_similarity,
                    ) = await self._persist_summary_memory(
                        session_id=session_id,
                        memory_scope=memory_scope,
                        persona_id=persona_id,
                        content=content,
                        metadata=metadata,
                        importance=importance,
                        atoms=atoms,
                        source_messages=source_messages,
                        source_validator=source_snapshot_current_at_commit,
                        source_publish_lease=lambda: (
                            self.conversation_manager.source_publication_lease(
                                source_snapshot_current_at_commit,
                                session_id=session_id,
                            )
                        ),
                    )
                    if memory_id is not None and not memory_created:
                        logger.info(
                            f"[{session_id}] 检测到近重复记忆，已合并到 #{memory_id}"
                            f"（相似度={dedup_similarity:.3f}），未新增记忆"
                        )
                    else:
                        logger.info(
                            f"[{session_id}] 成功存储对话记忆（{len(history_messages)}条消息，重要性={importance:.2f}）"
                        )

                # 成功：更新已总结的位置，清除待处理记录（原子校验修订号，
                # 避免把游标写回回滚前的旧位置）
                if self.conversation_manager is None or memory_id is None:
                    await record_failure(
                        failure_stage="storage", error_code="no_memory_receipt"
                    )
                    return False

                try:
                    committed = await commit_progress()
                    if committed:
                        logger.info(
                            f"[{session_id}] 已提交总结范围 [{start_index}:{end_index}]"
                        )
                        return True
                    if memory_id is not None:
                        await self._compensate_rolled_back_memory(
                            session_id,
                            memory_id,
                            **(
                                {}
                                if memory_created
                                else {"merged_snapshot": window_fingerprint}
                            ),
                        )
                    return False
                except Exception as meta_err:
                    logger.error(
                        f"[{session_id}] 记忆已存储但元数据更新失败: {meta_err}。"
                        "会保留该范围以便重试确认。",
                        exc_info=True,
                    )
                    # Advance the index anyway to prevent re-processing the
                    # same message range (memory is already stored durably).
                    try:
                        committed = await commit_progress()
                        if committed:
                            return True
                    except Exception:
                        logger.error(
                            f"[{session_id}] 重试元数据更新仍然失败，"
                            "可能出现重复总结。",
                            exc_info=True,
                        )
                    return False

            except asyncio.CancelledError as cancellation:
                # A valid publication survives ordinary task cancellation.
                # Keep its snapshot for restart, or compensate if the source
                # was revoked between publication and the progress commit.
                published_id = memory_id or getattr(
                    cancellation, "published_memory_id", None
                )
                if published_id is not None and not await self._revision_still_valid(
                    session_id, expected_revision
                ):
                    was_merge = (
                        not memory_created
                        if memory_id is not None
                        else bool(getattr(cancellation, "published_as_merge", False))
                    )
                    await asyncio.shield(
                        self._compensate_rolled_back_memory(
                            session_id,
                            published_id,
                            **(
                                {"merged_snapshot": window_fingerprint}
                                if was_merge
                                else {}
                            ),
                        )
                    )
                raise
            except Exception as e:
                logger.error(f"[{session_id}] 存储记忆失败: {e}", exc_info=True)
                await record_failure(
                    failure_stage="storage",
                    error_code="sqlite_busy"
                    if "database is locked" in str(e).lower()
                    else type(e).__name__,
                )
                return False

    async def _record_pending_summary(
        self,
        session_id: str,
        start_index: int,
        end_index: int,
        current_retry_count: int,
        expected_revision: int | None = None,
        *,
        failure_stage: str = "model",
        error_code: str | None = None,
        window_messages: list[Message],
        retry_refused: bool = False,
    ):
        """记录待处理的失败总结信息（回滚后不得复活旧范围）。"""
        if not self.conversation_manager:
            return False

        new_retry_count = current_retry_count + 1
        existing_pending = await self.conversation_manager.get_session_metadata(
            session_id, "pending_summary", None
        )
        managed_scope = session_id.startswith("narrative:v1:")
        window_fp = (
            summary_window_fingerprint(window_messages)
            if managed_scope and window_messages
            else None
        )
        same_window = isinstance(existing_pending, dict) and (
            (
                existing_pending.get("start_index") == start_index
                and existing_pending.get("end_index") == end_index
            )
            or (
                window_fp is not None
                and existing_pending.get("fingerprint") == window_fp
            )
        )
        prev_exhausted_cycles = (
            int(existing_pending.get("exhausted_cycles", 0) or 0) if same_window else 0
        )
        now_ts = time.time()
        backoff_idx = min(
            max(new_retry_count, 1) - 1, len(_SUMMARY_RETRY_BACKOFF_SECONDS) - 1
        )
        retry_delay = _SUMMARY_RETRY_BACKOFF_SECONDS[backoff_idx]
        pending_summary = {
            "start_index": start_index,
            "end_index": end_index,
            "retry_count": new_retry_count,
            "failure_stage": failure_stage,
            "error_code": error_code,
        }
        if prev_exhausted_cycles > 0:
            pending_summary["exhausted_cycles"] = prev_exhausted_cycles
        if not retry_refused and failure_stage not in TERMINAL_SUMMARY_STAGES:
            pending_summary["retry_after"] = now_ts + retry_delay
            if managed_scope:
                pending_summary["recovery_after"] = pending_summary["retry_after"]
        if failure_stage == "retry_exhausted" or (
            new_retry_count >= SUMMARY_RETRY_LIMIT
            and failure_stage not in TERMINAL_SUMMARY_STAGES
            and not retry_refused
        ):
            next_exhausted_cycles = prev_exhausted_cycles + 1
            # After repeated exhaustion cycles (first cycle cools down and
            # retries; a second exhausted cycle proves the window is
            # persistently poisoned), circuit-break the poisoned window for
            # BOTH native sessions and managed scopes so it never loops LLM
            # calls every 15 minutes indefinitely.
            if next_exhausted_cycles >= 2 and window_messages:
                if not managed_scope:
                    last_msg = window_messages[-1]
                    advanced = await self.conversation_manager.commit_summary_cursor(
                        session_id,
                        last_msg,
                        expected_revision,
                        window_messages=window_messages,
                    )
                    if advanced:
                        logger.warning(
                            f"[{session_id}] 总结窗口连续 {next_exhausted_cycles} 轮重试耗尽，"
                            f"已跳过毒化范围 [{start_index}:{end_index}] 并推进游标以解除死锁 "
                            f"(error_code={error_code})"
                        )
                        return True
                else:
                    pending_summary.pop("retry_after", None)
                    pending_summary.pop("recovery_after", None)
                    pending_summary["exhausted_retry_count"] = new_retry_count
                    pending_summary["exhausted_cycles"] = next_exhausted_cycles
                    pending_summary["retry_count"] = 0
                    pending_summary["failure_stage"] = "poisoned_window"
                    committed = (
                        await self.conversation_manager.write_managed_summary_progress(
                            session_id,
                            window_messages,
                            pending_summary,
                            expected_revision,
                            retry_refused=retry_refused,
                        )
                    )
                    if committed:
                        logger.warning(
                            f"[{session_id}] 托管总结窗口连续 {next_exhausted_cycles} 轮重试耗尽，"
                            f"已熔断隔离毒化范围 [{start_index}:{end_index}] 并停止自动恢复 "
                            f"(error_code={error_code})"
                        )
                    return committed
            # Retain the failed segment and retry after cooldown on first exhaustion.
            cooldown = max(SUMMARY_COOLDOWN_SECONDS, _SUMMARY_RETRY_BACKOFF_SECONDS[-1])
            pending_summary["retry_after"] = now_ts + cooldown
            if managed_scope:
                pending_summary["recovery_after"] = pending_summary["retry_after"]
            pending_summary["exhausted_retry_count"] = new_retry_count
            pending_summary["exhausted_cycles"] = next_exhausted_cycles
            # Older code ignores the cooldown marker and skips a whole backlog
            # at retry_count >= 3. A source rollback must resume, never skip it.
            pending_summary["retry_count"] = 0
            pending_summary["failure_stage"] = "retry_exhausted"

        if managed_scope:
            committed = await self.conversation_manager.write_managed_summary_progress(
                session_id,
                window_messages,
                pending_summary,
                expected_revision,
                retry_refused=retry_refused,
            )
        else:
            committed = await self.conversation_manager.write_native_pending_summary(
                session_id,
                window_messages,
                pending_summary,
                expected_revision,
            )
        if not committed:
            # 会话已被回滚/清空：该范围的消息已不存在，记录待重试会造成
            # 下一次触发按旧范围总结已撤销内容
            logger.info(
                f"[{session_id}] 会话已回滚（修订号变化），跳过记录待重试总结 "
                f"范围=[{start_index}:{end_index}]"
            )
            return

        logger.warning(
            f"[{session_id}] 记录待重试总结: 范围=[{start_index}:{end_index}], "
            f"重试次数={new_retry_count}/3"
        )
        return committed

    def _on_storage_task_done(self, task: asyncio.Task, session_id: str):
        """存储任务完成回调"""
        self._storage_tasks.discard(task)
        self._storage_sessions_inflight.discard(session_id)
        self._recorded_scope_wakes.discard(session_id)
        if self._storage_session_tasks.get(session_id) is task:
            self._storage_session_tasks.pop(session_id, None)

        if task.cancelled():
            logger.info(f"[{session_id}] 存储任务已取消")
            return

        exc = task.exception()
        if exc:
            logger.error(f"[{session_id}] 存储任务异常: {exc}", exc_info=exc)
