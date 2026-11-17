"""
记忆反思模块
负责检查是否需要记忆总结和存储
"""

import hashlib
import inspect
import time
import secrets
from collections import OrderedDict
from astrbot.api.platform import MessageType
from ..memory_scope import classify_umo, resolve_memory_write_scope
from ..models.conversation_models import Message
from ..utils import get_persona_id


import asyncio
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import LLMResponse

from ..memory_scope import (
    is_event_memory_allowed,
    is_suspicious_session_id,
)
from .memory_reflection_storage import (
    MANAGED_WINDOW_REFUSED,
    MemoryReflectionStorageMixin,
)
from ...storage.conversation_summary_progress import (
    SUMMARY_RETRY_LIMIT,
    TERMINAL_SUMMARY_STAGES,
)
from ..narrative_api import scope_key


_OWNED_MEMORY_SOURCE_VALIDATE_ATTR = "_cd_owned_memory_sources_validate_v1"


_OWNED_DRAFT_LINEAGE_KEY = "_lm_owned_model_output_lineage_v1"


_OWNED_LINEAGE_LIMIT = 64


_OWNED_VALIDATION_TIMEOUT_SECONDS = 1.0


_DEFAULT_MEMORY_SCOPE = object()

# 后台反思任务的全局并发上限：避免多会话同时爆发时齐刷刷调用 LLM/写嵌入。
# 仅限制反思总结（LLM + 记忆写入）阶段，不限制消息入库。
_MAX_CONCURRENT_REFLECTIONS = 2

# 长空闲补总结阈值：距上一条消息超过该时长时，下一轮对话立即补一次总结，
# 不再等 summary_trigger_rounds 轮（避免长时间不聊导致这次对话永久漏总结）。
_IDLE_SUMMARY_GAP_SECONDS = 60 * 60


def _message_timestamp(message: Any) -> float | None:
    """读取已入库消息的时间戳；缺失或非法时返回 None（该相邻对不参与切分）。"""
    value = getattr(message, "timestamp", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp <= 0:
        return None
    return timestamp


def cancel_session_storage_task(
    session_tasks: dict[str, asyncio.Task], session_id: str
) -> bool:
    """取消指定会话进行中的后台记忆总结任务。

    会话历史被回滚（WebUI 编辑/重试）时调用：任务持有的消息范围已失效，
    取消可避免它继续调用 LLM 并写入过期内容。修订号校验仍是最终防线
    （取消在 LLM 调用期间可能延迟生效）。
    """
    task = session_tasks.get(session_id)
    if task is None or task.done():
        return False
    task.cancel()
    logger.info(f"[{session_id}] 会话历史已回滚，取消进行中的记忆总结任务")
    return True


async def _mark_idle_summary_if_needed(manager: Any, session_id: str) -> bool:
    """长空闲后标记“下一轮补总结”，返回是否写入标记。

    Args:
        manager: ConversationManager；会话元数据与最近消息的读取方。
        session_id: 会话 ID。

    Returns:
        bool: 本次是否新写入 ``idle_summary_pending=True``。
    """
    try:
        messages = await manager.get_messages(session_id, limit=2)
    except Exception as exc:
        logger.debug(f"[{session_id}] 读取最近消息失败，跳过空闲检测: {exc}")
        return False
    if len(messages) < 2:
        return False
    previous_timestamp = _message_timestamp(messages[-2])
    current_timestamp = _message_timestamp(messages[-1])
    if previous_timestamp is None or current_timestamp is None:
        return False
    gap_seconds = current_timestamp - previous_timestamp
    if gap_seconds < _IDLE_SUMMARY_GAP_SECONDS:
        return False
    await manager.update_session_metadata(session_id, "idle_summary_pending", True)
    logger.info(
        f"[{session_id}] 距上一条消息 {gap_seconds / 60:.0f} 分钟，已标记下一轮补总结"
    )
    return True


@dataclass(frozen=True, slots=True)
class _ReflectionPlan:
    """一次反思总结的不可变计划，供后台存储任务消费。"""

    session_id: str
    history_messages: list[Any]
    persona_id: Any
    start_index: int
    end_index: int
    retry_count: int
    memory_scope: str
    expected_revision: int | None
    retry_refused: bool = False


if TYPE_CHECKING:
    from ..base.config_manager import ConfigManager
    from ..managers.conversation_manager import ConversationManager
    from ..managers.memory_engine import MemoryEngine
    from ..processors.memory_processor import MemoryProcessor
    from .message_utils import MessageUtils


class MemoryReflection(MemoryReflectionStorageMixin):
    """记忆反思类"""

    def __init__(
        self,
        context: Any,
        config_manager: "ConfigManager",
        memory_engine: "MemoryEngine",
        memory_processor: "MemoryProcessor",
        conversation_manager: "ConversationManager",
        message_utils: "MessageUtils",
        storage_tasks: set[asyncio.Task],
        storage_sessions_inflight: set[str],
        storage_state_lock: asyncio.Lock,
        consolidation_manager=None,
        storage_session_tasks: dict[str, asyncio.Task] | None = None,
    ):
        """
        初始化记忆反思模块

        Args:
            context: AstrBot上下文
            config_manager: 配置管理器
            memory_engine: 记忆引擎
            memory_processor: 记忆处理器
            conversation_manager: 会话管理器
            message_utils: 消息处理工具
            storage_tasks: 后台存储任务集合（共享状态）
            storage_sessions_inflight: 正在处理的会话集合（共享状态）
            storage_state_lock: 存储状态锁（共享状态）
            storage_session_tasks: 会话 -> 进行中任务映射（共享状态，用于回滚取消）
            consolidation_manager: 记忆整合管理器（用于反思触发）
        """
        self.context = context
        self.config_manager = config_manager
        self.memory_engine = memory_engine
        self.memory_processor = memory_processor
        self.conversation_manager = conversation_manager
        self.message_utils = message_utils
        self._storage_tasks = storage_tasks
        self._storage_sessions_inflight = storage_sessions_inflight
        self._storage_state_lock = storage_state_lock
        # 会话 -> 进行中的总结任务：会话回滚时按会话精准取消（共享状态）
        self._storage_session_tasks: dict[str, asyncio.Task] = (
            storage_session_tasks if storage_session_tasks is not None else {}
        )
        self._recorded_scope_wakes: set[str] = set()
        self.consolidation_manager = consolidation_manager
        self._shutting_down = False
        self._owned_source_lineages: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._reflection_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REFLECTIONS)

    def _schedule_consolidation(self) -> None:
        """反思触发时在后台顺带检查记忆整合（带冷却，不会频繁执行）。"""
        if self.consolidation_manager is None or self._shutting_down:
            return
        task = asyncio.create_task(self.consolidation_manager.maybe_run("reflection"))
        self._storage_tasks.add(task)
        task.add_done_callback(self._on_auxiliary_task_done)

    def _on_auxiliary_task_done(self, task: asyncio.Task) -> None:
        """取回后台任务异常并记录，避免静默失败。"""
        self._storage_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"[MemoryReflection] Background task failed: {exc}", exc_info=exc
            )

    @staticmethod
    def _find_window_split_boundary(
        messages: list[Any],
        *,
        split_by_day: bool,
        gap_minutes: int,
        min_segment_messages: int = 2,
    ) -> tuple[int, str] | None:
        """在候选消息中寻找最早的总结窗口切分边界（本地日期变化或空闲间隔）。

        Args:
            messages: ``[start_index, end_index)`` 范围内的消息，按入库顺序排列。
            split_by_day: 相邻消息的本地日历日期不同时切分。
            gap_minutes: 相邻消息时间差超过该分钟数时切分；``<= 0`` 表示禁用。
            min_segment_messages: 边界前最老片段至少需要的消息数。更早的边界会被
                跳过，否则保留的“不足一轮跳过”逻辑会让该会话永远无法推进。

        Returns:
            ``(最老片段的条数, 原因)``，原因为 ``"day"`` 或 ``"gap"``；
            没有可用边界时返回 ``None``（保持原有整段总结行为）。
        """
        if len(messages) <= min_segment_messages:
            return None

        timestamps = [_message_timestamp(message) for message in messages]
        gap_seconds = gap_minutes * 60 if gap_minutes > 0 else 0
        for index in range(min_segment_messages, len(messages)):
            previous = timestamps[index - 1]
            current = timestamps[index]
            # 时间戳缺失或乱序（后一条更早）时保守不切分
            if previous is None or current is None or current < previous:
                continue
            if split_by_day and (
                datetime.fromtimestamp(previous).date()
                != datetime.fromtimestamp(current).date()
            ):
                return index, "day"
            if gap_seconds and current - previous > gap_seconds:
                return index, "gap"
        return None

    async def handle_memory_reflection(
        self, event: AstrMessageEvent, resp: LLMResponse
    ):
        """Check if reflection and memory storage is needed after LLM response."""
        logger.debug(
            f"[DEBUG-Reflection] 进入 handle_memory_reflection，resp.role={resp.role}"
        )
        try:
            prepared = await self._reflection_precheck(event, resp)
            if prepared is None:
                return
            session_id, response_text = prepared
            plan = await self._reflection_plan(event, session_id, response_text)
            if plan is None:
                return
            await self._schedule_reflection(plan)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"处理记忆反思时发生错误: {e}", exc_info=True)

    async def _reflection_precheck(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> tuple[str, str] | None:
        """过滤不需要反思的响应；返回 (session_id, response_text) 表示可以继续。

        工具调用轮次的助手文字由本方法单独写入会话库（不触发后台总结），
        因此这类轮次也返回 None。
        """
        if not is_event_memory_allowed(self.config_manager, event):
            logger.debug("当前事件不在记忆白名单中，跳过记忆反思")
            return None

        # 流式增量只是完整回复的片段，不能作为一轮对话入库；
        # Mock 对象没有 is_chunk 属性，按 False 处理，仅真正为 True 时跳过
        if getattr(resp, "is_chunk", False) is True:
            return None

        # 工具调用轮次的 role 会被 OpenAI 兼容 Provider 改写为 "tool"，
        # 这些轮次的助手文字仍需入库；其他 role（如 err）不记录
        if resp.role not in ("assistant", "tool"):
            return None

        session_id = event.unified_msg_origin
        logger.debug(f"[DEBUG-Reflection] 获取到 unified_msg_origin: {session_id}")
        if not session_id:
            logger.warning("[DEBUG-Reflection] session_id 为空，跳过反思")
            return None

        if is_suspicious_session_id(session_id):
            logger.warning(
                f"[{session_id}] 检测到异常的session_id，这可能导致记忆总结异常。"
            )

        # 检查响应内容是否有效（过滤空回复和错误）
        response_text = resp.completion_text
        if not response_text or not response_text.strip():
            logger.debug(f"[{session_id}] 模型返回空回复，跳过记录")
            return None

        error_indicators = [
            "api error",
            "request failed",
            "rate limit",
            "timeout",
            "connection error",
            "服务暂时不可用",
            "请求失败",
            "接口错误",
        ]
        response_lower = response_text.lower()
        if any(indicator in response_lower for indicator in error_indicators):
            logger.debug(
                f"[{session_id}] 检测到错误响应，跳过记录: {response_text[:50]}..."
            )
            return None

        if await self._is_superseded_response(event, session_id):
            return None

        # 长空闲后的下一轮：标记补总结，_reflection_plan 会把触发阈值降为 1
        await _mark_idle_summary_if_needed(self.conversation_manager, session_id)

        # 工具轮：tool loop 尚未结束，总结会截断在工具调用中间，
        # 因此只把助手文字写入会话库，最终回复轮才触发后台总结。
        if resp.role == "tool" or resp.tools_call_name or resp.tools_call_extra_content:
            logger.debug(f"[{session_id}] 工具调用轮次仅记录助手消息，不触发记忆总结")
            await self.conversation_manager.add_message_from_event(
                event=event,
                role="assistant",
                content=response_text,
            )
            await self.message_utils.enforce_message_limit(session_id)
            return None

        return session_id, response_text

    async def _is_superseded_response(
        self, event: AstrMessageEvent, session_id: str
    ) -> bool:
        """WebUI 编辑/重试防护：本回复属于已被取代或重复的轮次时返回 True。

        库尾为 assistant 且 checkpoint 不同的情况（如纯图片轮次的回复紧跟在前
        一轮回复之后）不属于被取代轮次；真正迟到的孤立回复由 on_llm_request
        对账在下次请求时清理。
        """
        checkpoint_id = None
        get_extra = getattr(event, "get_extra", None)
        if callable(get_extra):
            try:
                checkpoint_id = get_extra("llm_checkpoint_id")
            except Exception as exc:
                checkpoint_id = None
                logger.debug(f"Failed to read llm_checkpoint_id from event: {exc}")
        if not (isinstance(checkpoint_id, str) and checkpoint_id):
            return False

        try:
            last_message = await self.conversation_manager.get_last_message(session_id)
        except Exception as exc:
            logger.debug(f"[{session_id}] 读取最后一条消息失败，跳过守卫: {exc}")
            return False
        if last_message is None:
            return False

        last_metadata = last_message.metadata or {}
        last_checkpoint = last_metadata.get("llm_checkpoint_id")
        if not (isinstance(last_checkpoint, str) and last_checkpoint):
            return False

        if last_message.role == "user" and last_checkpoint != checkpoint_id:
            # 被取代的回复会被丢弃而不入库：用 warning 暴露 checkpoint 漂移
            # （旧->新），避免平台时间戳异常时助手消息被静默丢掉（#289）
            logger.warning(
                f"[{session_id}] 检测到被取代的回复"
                f"（checkpoint {last_checkpoint} -> {checkpoint_id}），跳过存储"
            )
            return True
        if last_message.role == "assistant" and last_checkpoint == checkpoint_id:
            logger.debug(
                f"[{session_id}] 检测到重复回复（checkpoint {checkpoint_id}），跳过存储"
            )
            return True
        return False

    async def _reflection_plan(
        self, event: AstrMessageEvent, session_id: str, response_text: str
    ) -> _ReflectionPlan | None:
        """写入助手消息、判断触发条件并计算本次总结窗口。"""

        owned_private = self._is_owned_private_event(event)
        lineage_validator = getattr(event, _OWNED_MEMORY_SOURCE_VALIDATE_ATTR, None)
        lineage_token = None
        if owned_private or callable(lineage_validator):
            if not callable(lineage_validator):
                logger.warning("[LMReflection] route=owned_source outcome=unavailable")
                return
            if not await self._run_owned_source_validator(event, lineage_validator):
                logger.warning("[LMReflection] route=owned_source outcome=unavailable")
                return
            if owned_private:
                try:
                    exact_current_input = await asyncio.wait_for(
                        self._owned_current_input_row_current(event, session_id),
                        timeout=_OWNED_VALIDATION_TIMEOUT_SECONDS,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    exact_current_input = False
                if (
                    exact_current_input is not True
                    or not await self._run_owned_source_validator(
                        event, lineage_validator
                    )
                ):
                    logger.warning(
                        "[LMReflection] route=owned_source outcome=unavailable"
                    )
                    return
            lineage_token = secrets.token_hex(24)

        # 添加助手响应
        append_kwargs = {
            "event": event,
            "role": "assistant",
            "content": response_text,
        }
        if lineage_token is not None:
            append_kwargs["internal_metadata"] = {
                _OWNED_DRAFT_LINEAGE_KEY: lineage_token,
            }
        assistant_message = await self.conversation_manager.add_message_from_event(
            **append_kwargs
        )
        if lineage_token is not None:
            self._remember_owned_source_lineage(
                lineage_token, event, lineage_validator, assistant_message
            )
        logger.debug(f"[DEBUG-Reflection] [{session_id}] 已添加助手响应消息")

        # 私聊：助手消息写入后也执行消息数量上限控制
        is_group = event.get_message_type() == MessageType.GROUP_MESSAGE
        if not is_group:
            await self.message_utils.enforce_message_limit(session_id)

        # Read the retained history length used by the reflection cursor.
        actual_message_count = await self.conversation_manager.store.get_message_count(
            session_id
        )

        # 在读取总结范围之前捕获会话修订号：范围计算与后台任务启动之间
        # 若发生 WebUI 编辑/重试回滚，任务必须整体失效（否则会用回滚前的
        # 消息范围写入已撤销内容）
        expected_revision = await self._read_session_revision(session_id)

        total_messages = actual_message_count
        trigger_rounds = self.config_manager.get(
            "reflection_engine.summary_trigger_rounds", 10
        )
        # 长空闲后的补总结标记：本次只要有未总结轮次就触发，不改变正常阈值逻辑
        idle_summary_pending = bool(
            await self.conversation_manager.get_session_metadata(
                session_id, "idle_summary_pending", False
            )
        )
        if idle_summary_pending:
            trigger_rounds = 1
        last_summarized_index = await self.conversation_manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        )

        # 消息被删除后游标可能超出实际消息数；调整为当前总数，而非归零
        # （避免重复处理已总结的内容）
        if last_summarized_index > total_messages:
            logger.warning(
                f"[DEBUG-Reflection] [{session_id}] last_summarized_index({last_summarized_index}) "
                f"> 实际消息数({total_messages})，调整为当前消息总数"
            )
            last_summarized_index = total_messages
            await self.conversation_manager.update_session_metadata(
                session_id, "last_summarized_index", total_messages
            )

        unsummarized_rounds = (total_messages - last_summarized_index) // 2
        pending_summary = await self.conversation_manager.get_session_metadata(
            session_id, "pending_summary", None
        )
        logger.info(
            f"[DEBUG-Reflection] [{session_id}] 总消息数: {total_messages}, "
            f"上次总结位置: {last_summarized_index}, "
            f"未总结轮数: {unsummarized_rounds}, "
            f"触发阈值: {trigger_rounds}轮, "
            f"空闲补总结: {idle_summary_pending}, "
            f"待处理失败总结: {pending_summary is not None}"
        )
        if unsummarized_rounds < trigger_rounds:
            return None

        logger.info(
            f"[{session_id}] 未总结轮数达到 {unsummarized_rounds} 轮，启动记忆反思任务"
        )
        return await self._build_reflection_window(
            event,
            session_id,
            total_messages,
            last_summarized_index,
            pending_summary,
            expected_revision,
        )

    async def _build_reflection_window(
        self,
        event: AstrMessageEvent,
        session_id: str,
        total_messages: int,
        last_summarized_index: int,
        pending_summary: dict | None,
        expected_revision: int | None,
    ) -> _ReflectionPlan | None:
        """首次触发总结：解析人格/作用域后构造最老片段计划。"""
        persona_id = await get_persona_id(self.context, event)
        memory_scope = (
            resolve_memory_write_scope(self.config_manager, event) or session_id
        )
        return await self._build_reflection_plan_for_cursor(
            session_id,
            total_messages,
            last_summarized_index,
            pending_summary,
            expected_revision,
            persona_id,
            memory_scope,
        )

    async def _build_reflection_plan_for_cursor(
        self,
        session_id: str,
        total_messages: int,
        last_summarized_index: int,
        pending_summary: dict | None,
        expected_revision: int | None,
        persona_id: Any,
        memory_scope: str,
    ) -> _ReflectionPlan | None:
        """处理待重试范围、窗口切分并取回总结输入消息。"""
        if (
            pending_summary
            and pending_summary.get("failure_stage") in TERMINAL_SUMMARY_STAGES
        ):
            logger.warning("Summary paused; original input retained")
            return None
        if (
            pending_summary
            and pending_summary.get("failure_stage") == "retry_exhausted"
        ):
            if time.time() < float(pending_summary.get("retry_after", 0) or 0):
                return None
            pending_summary = {
                **pending_summary,
                "failure_stage": "model",
                "retry_count": 0,
            }
        # Older pending offsets may point past the first unsummarized row.
        # Use a bounded segment from the cursor before allowing later ranges.
        exact_pending = bool(
            pending_summary
            and type(pending_summary.get("end_index")) is int
            and type(pending_summary.get("start_index")) is int
            and pending_summary["start_index"] <= last_summarized_index
        )
        (
            start_index,
            candidate_messages,
        ) = await self.conversation_manager.get_summary_window(
            session_id,
            self._max_messages_per_segment(),
            exact_pending=exact_pending,
        )
        if len(candidate_messages) < 2:
            return None
        if not exact_pending:
            candidate_messages = self._split_reflection_window(
                session_id, start_index, candidate_messages
            )
        retry_count = (
            int(pending_summary.get("retry_count", 0)) if pending_summary else 0
        )
        if retry_count >= SUMMARY_RETRY_LIMIT:
            await self._record_pending_summary(
                session_id,
                start_index,
                start_index + len(candidate_messages),
                retry_count - 1,
                expected_revision,
                failure_stage="retry_exhausted",
                error_code="segment_retries_exhausted",
                window_messages=candidate_messages,
            )
            return None
        return _ReflectionPlan(
            session_id,
            candidate_messages,
            persona_id,
            start_index,
            start_index + len(candidate_messages),
            retry_count,
            memory_scope,
            expected_revision,
        )

    def _max_messages_per_segment(self) -> int:
        try:
            maximum = int(
                self.config_manager.get(
                    "reflection_engine.max_messages_per_segment", 120
                )
            )
        except (TypeError, ValueError):
            maximum = 120
        return max(2, min(maximum, 1000))

    def _split_reflection_window(self, session_id, start_index, candidate_messages):
        # 窗口切分配置：跨天或空闲间隔超过阈值时从最老片段开始逐段总结，
        # 避免一整段跨天/跨长空闲的对话被合并进同一条记忆。
        split_by_day = bool(
            self.config_manager.get("reflection_engine.window_split_by_day", True)
        )
        try:
            gap_minutes = int(
                self.config_manager.get(
                    "reflection_engine.window_split_gap_minutes", 180
                )
            )
        except (TypeError, ValueError):
            gap_minutes = 180

        split = self._find_window_split_boundary(
            candidate_messages,
            split_by_day=split_by_day,
            gap_minutes=gap_minutes,
        )
        if split is not None:
            split_offset, split_reason = split
            reason_text = "日期变化" if split_reason == "day" else "空闲间隔"
            logger.info(
                f"[{session_id}] 总结窗口按{reason_text}在消息 {start_index + split_offset} 处切分"
            )
            return candidate_messages[:split_offset]
        return candidate_messages

    async def _build_recorded_scope_plan(
        self,
        session_id,
        persona_id,
        expected_revision,
        threshold,
        *,
        retry_refused=False,
    ) -> _ReflectionPlan | None:
        (
            start,
            messages,
            pending,
        ) = await self.conversation_manager.store.get_managed_summary_window(
            session_id,
            self._max_messages_per_segment(),
            threshold,
            retry_refused=retry_refused,
        )
        if not messages or not await self._revision_still_valid(
            session_id, expected_revision
        ):
            return None
        retries = int(pending.get("retry_count", 0) or 0) if pending else 0
        if (
            not retry_refused
            and pending
            and pending.get("failure_stage") == "retry_exhausted"
        ):
            retries = 0
        if not retry_refused and retries >= SUMMARY_RETRY_LIMIT:
            await self._record_pending_summary(
                session_id,
                start,
                start + len(messages),
                retries - 1,
                expected_revision,
                failure_stage="retry_exhausted",
                error_code="segment_retries_exhausted",
                window_messages=messages,
            )
            return None
        if pending is None:
            messages = self._split_reflection_window(session_id, start, messages)
            if not await self.conversation_manager.write_managed_summary_progress(
                session_id,
                messages,
                {
                    "start_index": start,
                    "end_index": start + len(messages),
                    "retry_count": 0,
                    "failure_stage": "planned",
                },
                expected_revision,
            ):
                return None
        return _ReflectionPlan(
            session_id,
            messages,
            persona_id,
            start,
            start + len(messages),
            retries,
            session_id,
            expected_revision,
            retry_refused,
        )

    async def schedule_recorded_scope(self, scope, *, retry_refused=False) -> bool:
        session_id = scope_key(scope)
        async with self._storage_state_lock:
            if self._shutting_down:
                return False
            if session_id in self._storage_sessions_inflight:
                if not retry_refused:
                    self._recorded_scope_wakes.add(session_id)
                return False
            self._storage_sessions_inflight.add(session_id)
        requested_retry = retry_refused
        try:
            while True:
                revision = await self._read_session_revision(session_id)
                plan = await self._build_recorded_scope_plan(
                    session_id,
                    scope["persona_id"],
                    revision,
                    self._recorded_threshold(),
                    retry_refused=retry_refused,
                )
                if plan is not None:
                    self._queue_reflection_task(plan)
                    return not requested_retry or plan.retry_refused
                if session_id not in self._recorded_scope_wakes or self._shutting_down:
                    self._storage_sessions_inflight.discard(session_id)
                    return False
                self._recorded_scope_wakes.discard(session_id)
                if retry_refused:
                    # A delivery arrived while checking for an explicit retry.
                    retry_refused = False
        except BaseException:
            self._storage_sessions_inflight.discard(session_id)
            self._recorded_scope_wakes.discard(session_id)
            raise

    def _recorded_threshold(self) -> int:
        try:
            rounds = int(
                self.config_manager.get("reflection_engine.summary_trigger_rounds", 10)
            )
        except (TypeError, ValueError):
            rounds = 10
        return max(1, rounds) * 2

    async def _build_next_reflection_plan(
        self,
        session_id: str,
        persona_id: Any,
        memory_scope: str,
        expected_revision: int | None,
    ) -> _ReflectionPlan | None:
        """读取成功提交后的最新游标，为积压消费构造下一段计划。"""
        if not await self._revision_still_valid(session_id, expected_revision):
            return None
        if session_id.startswith("narrative:v1:"):
            return await self._build_recorded_scope_plan(
                session_id, persona_id, expected_revision, 1
            )

        total_messages = await self.conversation_manager.store.get_message_count(
            session_id
        )
        last_summarized_index = await self.conversation_manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        )
        pending_summary = await self.conversation_manager.get_session_metadata(
            session_id, "pending_summary", None
        )
        if pending_summary is not None:
            return None
        if total_messages - last_summarized_index < 2:
            return None

        return await self._build_reflection_plan_for_cursor(
            session_id,
            total_messages,
            last_summarized_index,
            pending_summary,
            expected_revision,
            persona_id,
            memory_scope,
        )

    async def _run_reflection_task(self, plan: _ReflectionPlan) -> None:
        """获取全局并发额度后执行反思任务。

        信号量用 ``async with`` 管理，任务被取消或总结抛异常时都会自动释放
        额度，不会出现额度泄漏导致后续反思永久排队。
        """
        while True:
            async with self._reflection_semaphore:
                batch_has_more = await self._consume_reflection_segments(plan)
            session_id = plan.session_id
            next_plan = None
            if (
                batch_has_more
                and session_id.startswith("narrative:v1:")
                and not self._shutting_down
            ):
                await asyncio.sleep(1)
                if not self._shutting_down:
                    next_plan = await self._build_recorded_scope_plan(
                        session_id,
                        plan.persona_id,
                        plan.expected_revision,
                        self._recorded_threshold(),
                    )
            while (
                next_plan is None
                and session_id in self._recorded_scope_wakes
                and not self._shutting_down
            ):
                self._recorded_scope_wakes.discard(session_id)
                revision = await self._read_session_revision(session_id)
                next_plan = await self._build_recorded_scope_plan(
                    session_id,
                    plan.persona_id,
                    revision,
                    self._recorded_threshold(),
                )
            if next_plan is None:
                return
            plan = next_plan

    async def _consume_reflection_segments(self, plan: _ReflectionPlan) -> bool:
        """在同一后台任务内逐段消费积压，直到清空或达到单次片段预算。"""
        committed_any = False
        batch_has_more = False
        max_segments = self._max_segments_per_run()
        for segment_index in range(max_segments):
            outcome = await self._storage_task(
                plan.session_id,
                plan.history_messages,
                plan.persona_id,
                plan.start_index,
                plan.end_index,
                plan.retry_count,
                memory_scope=plan.memory_scope,
                expected_revision=plan.expected_revision,
                **({"retry_refused": True} if plan.retry_refused else {}),
            )
            if not outcome:
                break
            if outcome is not MANAGED_WINDOW_REFUSED:
                committed_any = True
                if await self.conversation_manager.get_session_metadata(
                    plan.session_id, "idle_summary_pending", False
                ):
                    await self.conversation_manager.update_session_metadata(
                        plan.session_id, "idle_summary_pending", False
                    )
            if self._shutting_down or plan.retry_refused:
                break

            if segment_index + 1 >= max_segments:
                logger.debug(
                    f"[{plan.session_id}] 达到单次反思片段上限 {max_segments}，"
                    "停止继续消费"
                )
                batch_has_more = True
                break

            next_plan = await self._build_next_reflection_plan(
                plan.session_id,
                plan.persona_id,
                plan.memory_scope,
                plan.expected_revision,
            )
            if next_plan is None:
                break
            previous_anchor = (
                plan.history_messages[-1].timestamp,
                plan.history_messages[-1].id,
            )
            next_anchor = (
                next_plan.history_messages[-1].timestamp,
                next_plan.history_messages[-1].id,
            )
            if (
                not plan.session_id.startswith("narrative:v1:")
                and next_anchor <= previous_anchor
            ):
                logger.warning("Summary anchor did not advance; backlog retained")
                break
            plan = next_plan
        if committed_any:
            self._schedule_consolidation()
        return batch_has_more

    async def _schedule_reflection(self, plan: _ReflectionPlan) -> bool:
        """创建并跟踪后台总结任务；同一会话已有任务在跑时跳过。"""
        if self._shutting_down:
            return False
        async with self._storage_state_lock:
            if plan.session_id in self._storage_sessions_inflight:
                logger.info(f"[{plan.session_id}] 已有记忆反思任务在执行，跳过本次触发")
                return False
            self._storage_sessions_inflight.add(plan.session_id)

        self._queue_reflection_task(plan)
        return True

    def _queue_reflection_task(self, plan: _ReflectionPlan) -> None:
        try:
            task = asyncio.create_task(self._run_reflection_task(plan))
        except Exception:
            self._storage_sessions_inflight.discard(plan.session_id)
            raise

        self._storage_tasks.add(task)
        self._storage_session_tasks[plan.session_id] = task
        task.add_done_callback(
            lambda t, sid=plan.session_id: self._on_storage_task_done(t, sid)
        )

    def set_shutting_down(self, value: bool):
        """设置关闭状态"""
        self._shutting_down = value

    async def _history_rows_current(self, session_id: str, history_messages) -> bool:
        """Require each captured message to still match its stored row."""
        store = getattr(self.conversation_manager, "store", None)
        raw_batch_loader = getattr(store, "get_messages_by_ids", None)
        batch_loader = (
            raw_batch_loader if asyncio.iscoroutinefunction(raw_batch_loader) else None
        )
        loader = getattr(store, "get_message_by_id", None)
        if (
            (batch_loader is None and not callable(loader))
            or not isinstance(session_id, str)
            or not session_id.strip()
        ):
            return False
        if not isinstance(history_messages, (list, tuple)) or not history_messages:
            return False
        sources: list[dict] = []
        message_ids: list[int] = []
        for message in history_messages:
            if isinstance(message, dict):
                source = message
            elif callable(getattr(message, "to_dict", None)):
                source = message.to_dict()
            else:
                source = {
                    key: getattr(message, key, None)
                    for key in (
                        "id",
                        "session_id",
                        "role",
                        "content",
                        "sender_id",
                        "sender_name",
                        "group_id",
                        "platform",
                        "timestamp",
                        "metadata",
                    )
                }
            message_id = source.get("id")
            if (
                isinstance(message_id, bool)
                or not isinstance(message_id, int)
                or message_id <= 0
                or source.get("session_id") != session_id
            ):
                return False
            sources.append(source)
            message_ids.append(message_id)
        rows_by_id: dict[int, dict] | None = None
        if callable(batch_loader):
            try:
                loaded = await batch_loader(session_id, message_ids)
                if isinstance(loaded, dict):
                    rows_by_id = loaded
            except asyncio.CancelledError:
                raise
            except Exception:
                return False
        for source, message_id in zip(sources, message_ids):
            if rows_by_id is not None:
                current = rows_by_id.get(message_id)
            else:
                try:
                    current = await loader(session_id, message_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return False
            if not isinstance(current, dict):
                return False
            if (
                current.get("id") != message_id
                or current.get("session_id") != session_id
                or current.get("role") != source.get("role")
                or current.get("content")
                != Message.content_to_text(source.get("content"))
                or current.get("sender_id") != source.get("sender_id")
                or current.get("sender_name") != source.get("sender_name")
                or current.get("group_id") != source.get("group_id")
                or current.get("platform") != source.get("platform")
            ):
                return False
            try:
                if float(current.get("timestamp")) != float(source.get("timestamp")):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
            current_metadata = current.get("metadata", {})
            if isinstance(current_metadata, str):
                try:
                    import json

                    current_metadata = json.loads(current_metadata)
                except (TypeError, ValueError):
                    return False
            expected_metadata = source.get("metadata", {})
            if not isinstance(current_metadata, dict) or not isinstance(
                expected_metadata, dict
            ):
                return False
            identity_keys = (
                "schema_version",
                "source_event_id",
                "scope",
                "fact_kind",
                "is_bot_message",
                _OWNED_DRAFT_LINEAGE_KEY,
            )
            if any(
                current_metadata.get(key) != expected_metadata.get(key)
                for key in identity_keys
            ):
                return False
        return True

    @staticmethod
    def _is_owned_private_event(event) -> bool:
        owned = (
            getattr(event, "_cd_owned_original_input_v1", None) is True
            or getattr(event, "_chat_dynamics_owned_request", None) is True
        )
        if not owned:
            return False
        return classify_umo(getattr(event, "unified_msg_origin", None)) == "private"

    async def _run_owned_source_validator(self, event, validator) -> bool:
        if (
            not callable(validator)
            or getattr(event, _OWNED_MEMORY_SOURCE_VALIDATE_ATTR, None) is not validator
        ):
            return False
        try:
            pending = validator(event)
            if not inspect.isawaitable(pending):
                return False
            result = await asyncio.wait_for(
                pending, timeout=_OWNED_VALIDATION_TIMEOUT_SECONDS
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return False
        return (
            result is True
            and getattr(event, _OWNED_MEMORY_SOURCE_VALIDATE_ATTR, None) is validator
        )

    async def _owned_current_input_row_current(self, event, session_id: str) -> bool:
        """Bind an owned reply to the exact persisted raw row for its source event."""
        resolver = getattr(self.message_utils, "resolve_original_input", None)
        if not callable(resolver):
            return False
        original = await resolver(event)
        if (
            getattr(original, "status", None) != "valid"
            or not isinstance(getattr(original, "scope", None), dict)
            or original.scope.get("umo") != session_id
        ):
            return False
        current_key = getattr(original, "current_event_key", None)
        if not isinstance(current_key, str) or not current_key:
            return False
        current = [
            item
            for item in getattr(original, "messages", ())
            if getattr(item, "event_key", None) == current_key
        ]
        if len(current) != 1:
            return False
        source = current[0]
        if (
            not isinstance(getattr(source, "author_id", None), str)
            or not source.author_id.strip()
            or not isinstance(getattr(source, "plain_text", None), str)
            or not source.plain_text
            or getattr(source, "occurred_at", None) is None
        ):
            return False
        store = getattr(self.conversation_manager, "store", None)
        loader = getattr(store, "get_messages", None)
        if not callable(loader):
            return False
        rows = await loader(session_id, limit=256)
        source_event_id = "cd-original-input-v1:" + current_key
        matches = []
        for row in rows if isinstance(rows, (list, tuple)) else ():
            metadata = getattr(row, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            if metadata.get("source_event_id") == source_event_id:
                matches.append((row, metadata))
        if len(matches) != 1:
            return False
        row, metadata = matches[0]
        scope = metadata.get("scope")
        try:
            row_timestamp = float(getattr(row, "timestamp", None))
            source_timestamp = float(source.occurred_at)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            getattr(row, "session_id", None) == session_id
            and getattr(row, "role", None) == "user"
            and getattr(row, "content", None) == source.plain_text
            and getattr(row, "sender_id", None) == source.author_id
            and row_timestamp == source_timestamp
            and getattr(row, "platform", None) == original.scope.get("platform")
            and isinstance(scope, dict)
            and scope.get("umo") == session_id
            and metadata.get("is_bot_message") is False
        )

    @staticmethod
    def _lineage_message_identity(message):
        if isinstance(message, dict):
            source = message
        elif callable(getattr(message, "to_dict", None)):
            source = message.to_dict()
        else:
            source = {
                key: getattr(message, key, None)
                for key in ("id", "session_id", "role", "content")
            }
        message_id = source.get("id")
        session_id = source.get("session_id")
        role = source.get("role")
        if (
            isinstance(message_id, bool)
            or not isinstance(message_id, int)
            or message_id <= 0
            or not isinstance(session_id, str)
            or role != "assistant"
        ):
            return None
        content = Message.content_to_text(source.get("content"))
        return (
            session_id,
            message_id,
            role,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    def _remember_owned_source_lineage(
        self, token: str, event, validator, assistant_message
    ) -> None:
        if not isinstance(token, str) or not token or not callable(validator):
            return
        identity = self._lineage_message_identity(assistant_message)
        if identity is None:
            return
        lineages = getattr(self, "_owned_source_lineages", None)
        if not isinstance(lineages, OrderedDict):
            lineages = OrderedDict()
            self._owned_source_lineages = lineages
        lineages[token] = (event, validator, identity)
        lineages.move_to_end(token)
        while len(lineages) > _OWNED_LINEAGE_LIMIT:
            lineages.popitem(last=False)

    def _forget_owned_source_pair(self, event, validator) -> None:
        """Evict only rows bound to this exact source capability pair."""
        lineages = getattr(self, "_owned_source_lineages", None)
        if not isinstance(lineages, OrderedDict):
            return
        for token, pair in tuple(lineages.items()):
            if (
                isinstance(pair, tuple)
                and len(pair) == 3
                and pair[0] is event
                and pair[1] is validator
            ):
                lineages.pop(token, None)

    @staticmethod
    def _message_role_metadata(message):
        if isinstance(message, dict):
            role = message.get("role")
            metadata = message.get("metadata", {})
        else:
            role = getattr(message, "role", None)
            metadata = getattr(message, "metadata", {})
        return role, metadata if isinstance(metadata, dict) else {}

    def _eligible_reflection_messages(self, messages):
        lineages = getattr(self, "_owned_source_lineages", None)
        if not isinstance(lineages, OrderedDict):
            lineages = OrderedDict()
            self._owned_source_lineages = lineages
        eligible = []
        for message in messages or ():
            role, metadata = self._message_role_metadata(message)
            if _OWNED_DRAFT_LINEAGE_KEY in metadata:
                token = metadata.get(_OWNED_DRAFT_LINEAGE_KEY)
                lineage = lineages.get(token) if isinstance(token, str) else None
                if (
                    role != "assistant"
                    or not isinstance(lineage, tuple)
                    or len(lineage) != 3
                    or self._lineage_message_identity(message) != lineage[2]
                ):
                    continue
                lineages.move_to_end(token)
            eligible.append(message)
        return eligible

    async def _owned_source_lineages_current(self, history_messages) -> bool:
        lineages = getattr(self, "_owned_source_lineages", None)
        unique = {}
        expected_by_token = {}
        for message in history_messages or ():
            role, metadata = self._message_role_metadata(message)
            if _OWNED_DRAFT_LINEAGE_KEY not in metadata:
                continue
            token = metadata.get(_OWNED_DRAFT_LINEAGE_KEY)
            if (
                role != "assistant"
                or not isinstance(token, str)
                or not isinstance(lineages, OrderedDict)
            ):
                return False
            pair = lineages.get(token)
            if (
                not isinstance(pair, tuple)
                or len(pair) != 3
                or self._lineage_message_identity(message) != pair[2]
            ):
                return False
            event, validator, _identity = pair
            if (
                not callable(validator)
                or getattr(event, _OWNED_MEMORY_SOURCE_VALIDATE_ATTR, None)
                is not validator
            ):
                self._forget_owned_source_pair(event, validator)
                return False
            unique[(id(event), id(validator))] = (event, validator)
            expected_by_token[token] = pair
        if not unique:
            return True
        try:
            outcomes = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        self._run_owned_source_validator(event, validator)
                        for event, validator in unique.values()
                    ),
                    return_exceptions=True,
                ),
                timeout=_OWNED_VALIDATION_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            for event, validator in unique.values():
                self._forget_owned_source_pair(event, validator)
            return False
        except Exception:
            for event, validator in unique.values():
                self._forget_owned_source_pair(event, validator)
            return False
        denied = False
        for (event_id, validator_id), outcome in zip(unique, outcomes):
            if outcome is not True:
                event, validator = unique[(event_id, validator_id)]
                self._forget_owned_source_pair(event, validator)
                denied = True
        if denied:
            return False
        current_lineages = getattr(self, "_owned_source_lineages", None)
        if not isinstance(current_lineages, OrderedDict):
            return False
        for token, expected in expected_by_token.items():
            current = current_lineages.get(token)
            if (
                not isinstance(current, tuple)
                or len(current) != 3
                or current[0] is not expected[0]
                or current[1] is not expected[1]
                or current[2] != expected[2]
            ):
                return False
        return True

    def _max_segments_per_run(self) -> int:
        try:
            value = int(
                self.config_manager.get("reflection_engine.max_segments_per_run", 10)
            )
        except (TypeError, ValueError):
            value = 10
        return max(1, min(value, 50))
