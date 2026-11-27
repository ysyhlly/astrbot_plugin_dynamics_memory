"""
群聊消息捕获模块
负责捕获和存储群聊中的所有消息
"""

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.platform import MessageType

from ..memory_scope import (
    is_event_excluded_by_series,
    is_event_memory_allowed,
    is_event_privacy_muted,
    is_suspicious_session_id,
)

if TYPE_CHECKING:
    from ..base.config_manager import ConfigManager
    from ..managers.conversation_manager import ConversationManager
    from .message_utils import MessageUtils


class GroupCapture:
    """群聊消息捕获类"""

    _LIMIT_CHECK_INTERVAL = 20
    _LIMIT_COUNTER_MAX_SESSIONS = 1024

    def __init__(
        self,
        config_manager: "ConfigManager",
        conversation_manager: "ConversationManager",
        message_utils: "MessageUtils",
        context=None,
    ):
        """
        初始化群聊消息捕获模块

        Args:
            config_manager: 配置管理器
            conversation_manager: 会话管理器
            message_utils: 消息处理工具
            context: 可选 AstrBot 上下文（用于检查主控 exclude_groups）
        """
        self.config_manager = config_manager
        self.conversation_manager = conversation_manager
        self.message_utils = message_utils
        self.context = context
        self._limit_check_counters: OrderedDict[str, int] = OrderedDict()

    def _should_check_message_limit(self, session_id: str) -> bool:
        """每 20 条群聊消息触发一次上限检查，避免逐条 SELECT COUNT(*)。"""
        interval = max(1, int(getattr(self, "_LIMIT_CHECK_INTERVAL", 20)))
        count = self._limit_check_counters.get(session_id, 0) + 1
        if count >= interval:
            self._limit_check_counters[session_id] = 0
            self._limit_check_counters.move_to_end(session_id)
            return True
        self._limit_check_counters[session_id] = count
        self._limit_check_counters.move_to_end(session_id)
        while len(self._limit_check_counters) > self._LIMIT_COUNTER_MAX_SESSIONS:
            self._limit_check_counters.popitem(last=False)
        return False

    async def handle_all_group_messages(self, event: AstrMessageEvent):
        """Capture all group messages for memory storage"""
        # 检查配置
        if not self.config_manager.get(
            "session_manager.enable_full_group_capture", True
        ):
            return

        # 只处理群聊消息
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        if (
            is_event_excluded_by_series(getattr(self, "context", None), event)
            or not is_event_memory_allowed(self.config_manager, event)
            or is_event_privacy_muted(getattr(self, "context", None), event, "memory")
        ):
            return

        # 群聊中 Bot 自己的消息由 handle_memory_reflection 负责写入，此处跳过
        # 避免 platform echo 导致 assistant 响应被写入两次
        if event.get_sender_id() == event.get_self_id():
            return

        try:
            session_id = event.unified_msg_origin

            # 检测异常session_id
            if is_suspicious_session_id(session_id):
                logger.warning(
                    f"检测到异常的session_id: {session_id}。"
                    f"这可能是平台适配器初始化问题，建议检查平台配置。"
                )

            # 获取消息内容
            content = await self.message_utils.extract_message_content(event)
            dedup_key = await self.message_utils.build_dedup_key(
                event, session_id, content
            )

            # 消息去重
            if dedup_key and await self.message_utils.is_duplicate_message(dedup_key):
                logger.debug(f"[{session_id}] 消息已存在,跳过: dedup_key={dedup_key}")
                return

            # 存储消息到数据库（群聊用户消息，role 固定为 user）
            await self.conversation_manager.add_message_from_event(
                event=event,
                role="user",
                content=content,
            )
            if dedup_key:
                await self.message_utils.mark_message_processed(dedup_key)

            # 每 20 条群聊消息执行一次消息数量上限控制，避免每条消息都查 SELECT COUNT(*)
            if self._should_check_message_limit(session_id):
                await self.message_utils.enforce_message_limit(session_id)

            logger.debug(
                f"[{session_id}] 捕获群聊消息: "
                f"sender={event.get_sender_name()}({event.get_sender_id()}), "
                f"content={content[:50]}..."
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"处理群聊全量消息时发生错误: {e}", exc_info=True)
