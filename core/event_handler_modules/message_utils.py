"""
消息处理工具模块
负责消息内容提取、去重、限制等操作
"""

import asyncio
import hashlib
import inspect
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

if TYPE_CHECKING:
    from ..base.config_manager import ConfigManager
    from ..managers.conversation_manager import ConversationManager


@dataclass(frozen=True)
class OriginalInputMessage:
    """One immutable original message from the trusted CD handoff."""

    event_key: str
    platform_message_id: str | None
    author_id: str
    occurred_at: float | None
    parts: tuple[dict, ...]
    plain_text: str


@dataclass(frozen=True)
class OriginalInput:
    """Validated source for a private recall query and optional persistence."""

    status: str
    scope: dict | None
    current_event_key: str | None
    messages: tuple[OriginalInputMessage, ...]
    query_text: str
    query_truncated: bool = False


def discard_unexpected_awaitable(value) -> None:
    """Close an accidental coroutine without awaiting or cancelling caller tasks."""
    if inspect.iscoroutine(value):
        try:
            value.close()
        except Exception:
            pass
        return
    if isinstance(value, asyncio.Future):
        # A Future/Task may belong to the caller; never cancel or wait for it.
        return
    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


ORIGINAL_PART_KINDS = frozenset(
    {
        "text",
        "quote",
        "mention",
        "image",
        "audio",
        "video",
        "file",
        "think",
        "other",
    }
)


MAX_ORIGINAL_MESSAGES = 32


MAX_ORIGINAL_PARTS = 128


MAX_ORIGINAL_TOTAL_PARTS = MAX_ORIGINAL_MESSAGES * MAX_ORIGINAL_PARTS


MAX_ORIGINAL_TEXT_PER_MESSAGE = 64_000


MAX_ORIGINAL_TEXT_TOTAL = 256_000


MAX_RECALL_QUERY_CHARS = 2_000


def _copy_original_json(value, *, depth: int = 0):
    """Copy frozen DTO values to JSON data without invoking arbitrary objects."""
    if depth > 12:
        raise ValueError("original_input_depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("original_input_number")
        return value
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("original_input_key")
            copied[key] = _copy_original_json(item, depth=depth + 1)
        return copied
    if isinstance(value, tuple):
        return [_copy_original_json(item, depth=depth + 1) for item in value]
    raise ValueError("original_input_value")


def validate_owned_original_input(
    dto,
    *,
    expected_scope: Mapping,
    trigger_author_id: str,
    platform_message_id: str | None = None,
    self_id: str,
) -> OriginalInput | None:
    """Validate the CD-owned v1 source sidecar; never inspect request.prompt."""
    schema_version = dto.get("schema_version") if isinstance(dto, Mapping) else None
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        return None
    source_scope = dto.get("scope")
    if not isinstance(source_scope, Mapping) or not isinstance(expected_scope, Mapping):
        return None
    scope_keys = ("umo", "platform", "self_id")
    canonical_scope = {}
    for key in scope_keys:
        value = source_scope.get(key)
        expected = expected_scope.get(key)
        if (
            not isinstance(value, str)
            or not value.strip()
            or not isinstance(expected, str)
            or not expected.strip()
            or value != expected
        ):
            return None
        canonical_scope[key] = value
    if (
        not isinstance(trigger_author_id, str)
        or not trigger_author_id.strip()
        or not isinstance(self_id, str)
        or not self_id.strip()
        or trigger_author_id == self_id
    ):
        return None

    current_key = dto.get("current_event_key")
    raw_messages = dto.get("messages")
    if (
        not isinstance(current_key, str)
        or not current_key.strip()
        or len(current_key) > 512
        or not isinstance(raw_messages, tuple)
        or not raw_messages
        or len(raw_messages) > MAX_ORIGINAL_MESSAGES
    ):
        return None

    messages: list[OriginalInputMessage] = []
    event_keys: set[str] = set()
    total_parts = 0
    total_text = 0
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            return None
        event_key = raw.get("event_key")
        author_id = raw.get("author_id")
        platform_id = raw.get("platform_message_id")
        occurred = raw.get("occurred_at")
        raw_parts = raw.get("parts")
        if (
            not isinstance(event_key, str)
            or not event_key.strip()
            or len(event_key) > 512
            or event_key in event_keys
            or not isinstance(author_id, str)
            or not author_id.strip()
            or (
                platform_id is not None
                and (not isinstance(platform_id, str) or not platform_id.strip())
            )
            or (
                occurred is not None
                and (
                    isinstance(occurred, bool)
                    or not isinstance(occurred, (int, float))
                    or not math.isfinite(float(occurred))
                )
            )
            or not isinstance(raw_parts, tuple)
        ):
            return None
        event_keys.add(event_key)
        parts: list[dict] = []
        plain_text: list[str] = []
        message_text = 0
        if len(raw_parts) > MAX_ORIGINAL_PARTS:
            return None
        total_parts += len(raw_parts)
        if total_parts > MAX_ORIGINAL_TOTAL_PARTS:
            return None
        try:
            for raw_part in raw_parts:
                if not isinstance(raw_part, Mapping):
                    return None
                kind = raw_part.get("kind")
                if not isinstance(kind, str) or kind not in ORIGINAL_PART_KINDS:
                    return None
                part = _copy_original_json(raw_part)
                if not isinstance(part, dict):
                    return None
                part_text = part.get("text")
                if part_text is not None and not isinstance(part_text, str):
                    return None
                if kind == "text" and isinstance(part_text, str):
                    message_text += len(part_text)
                    total_text += len(part_text)
                    if (
                        message_text > MAX_ORIGINAL_TEXT_PER_MESSAGE
                        or total_text > MAX_ORIGINAL_TEXT_TOTAL
                    ):
                        return None
                if kind == "text":
                    if not isinstance(part_text, str):
                        return None
                    if part_text:
                        plain_text.append(part_text)
                parts.append(part)
        except (TypeError, ValueError):
            return None
        timestamp = float(occurred) if occurred is not None else None
        messages.append(
            OriginalInputMessage(
                event_key=event_key,
                platform_message_id=platform_id,
                author_id=author_id,
                occurred_at=timestamp,
                parts=tuple(parts),
                plain_text="".join(plain_text),
            )
        )

    current_messages = [
        message for message in messages if message.event_key == current_key
    ]
    if (
        len(current_messages) != 1
        or current_messages[0].author_id != trigger_author_id
        or current_messages[0].author_id == self_id
    ):
        return None
    if (
        platform_message_id
        and current_messages[0].platform_message_id
        and current_messages[0].platform_message_id != platform_message_id
    ):
        return None

    ordered = [
        current_messages[0],
        *(message for message in messages if message.event_key != current_key),
    ]
    query_parts: list[str] = []
    remaining = MAX_RECALL_QUERY_CHARS
    truncated = False
    for message in ordered:
        if message.author_id == self_id or not message.plain_text:
            continue
        separator = " " if query_parts else ""
        if remaining <= len(separator):
            truncated = True
            break
        if separator:
            remaining -= len(separator)
        text = message.plain_text
        if len(text) > remaining:
            query_parts.append(text[:remaining])
            truncated = True
            remaining = 0
            break
        query_parts.append(text)
        remaining -= len(text)
        if remaining <= 0:
            truncated = True
            break
    return OriginalInput(
        status="valid",
        scope=canonical_scope,
        current_event_key=current_key,
        messages=tuple(messages),
        query_text=" ".join(part for part in query_parts if part),
        query_truncated=truncated,
    )


class MessageUtils:
    """消息处理工具类"""

    def __init__(
        self,
        config_manager: "ConfigManager",
        conversation_manager: "ConversationManager | None",
    ):
        """
        初始化消息处理工具

        Args:
            config_manager: 配置管理器
            conversation_manager: 会话管理器
        """
        self.config_manager = config_manager
        self.conversation_manager = conversation_manager

        # 消息去重缓存
        self._message_dedup_cache: dict[str, float] = {}
        self._dedup_cache_max_size = 1000
        self._dedup_cache_ttl = 300

    async def build_dedup_key(
        self, event: AstrMessageEvent, session_id: str, content: str
    ) -> str | None:
        """构建去重键：优先使用 message_id，缺失时退化为消息内容指纹。"""
        raw_message_id = getattr(
            getattr(event, "message_obj", None), "message_id", None
        )
        if raw_message_id is not None:
            message_id = str(raw_message_id).strip()
            if message_id:
                return f"id:{message_id}"

        sender_id = event.get_sender_id() if hasattr(event, "get_sender_id") else ""
        timestamp = getattr(getattr(event, "message_obj", None), "timestamp", 0)
        fingerprint = f"{session_id}|{sender_id}|{timestamp}|{content}"
        digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
        return f"fallback:{digest}"

    async def is_duplicate_message(self, dedup_key: str | None) -> bool:
        """检查消息是否已经处理过（惰性过期 + 溢出时逐条淘汰）"""
        if not dedup_key:
            return False

        result = dedup_key in self._message_dedup_cache
        if not result:
            return False

        # 惰性过期检查：命中时若已过期则视为未命中
        if time.time() - self._message_dedup_cache[dedup_key] > self._dedup_cache_ttl:
            del self._message_dedup_cache[dedup_key]
            return False

        return True

    async def mark_message_processed(self, dedup_key: str | None):
        """标记消息已处理（超限时淘汰最早插入的条目）"""
        if not dedup_key:
            return
        cache = self._message_dedup_cache
        if len(cache) >= self._dedup_cache_max_size:
            # 淘汰最早插入的条目（O(n) 但仅超限时触发，n≤1000）
            oldest_key = min(cache.items(), key=lambda x: x[1])[0]
            del cache[oldest_key]
        cache[dedup_key] = time.time()

    async def extract_message_content(
        self, event: AstrMessageEvent, req: ProviderRequest | None = None
    ) -> str:
        """提取消息内容，按组件原始顺序拼接，保留文字与图片的相对位置。
        若 AstrBot 已完成图片转述（req.extra_user_content_parts 中含 <image_caption> 标签），
        则按图片出现顺序依次替换，不会重复消费同一条转述。
        """
        from astrbot.core.message.components import (
            At,
            AtAll,
            Face,
            File,
            Forward,
            Image,
            Plain,
            Record,
            Reply,
            Video,
        )

        # 预先提取所有图片转述（按 extra_user_content_parts 中的出现顺序）
        # AstrBot 按消息链中图片的顺序依次追加转述，与 get_messages() 中 Image 的顺序一一对应
        caption_queue: list[str] = []
        if req is not None:
            for part in getattr(req, "extra_user_content_parts", []):
                text = getattr(part, "text", "")
                if not text:
                    continue
                for m in re.findall(
                    r"<image_caption>(.*?)</image_caption>", text, re.DOTALL
                ):
                    m = m.strip()
                    if m:
                        caption_queue.append(m)

        parts = []
        caption_idx = 0

        # 按组件原始顺序遍历，保留文字与图片的相对位置
        for component in event.get_messages():
            if isinstance(component, Plain):
                text = component.text.strip() if component.text else ""
                if text:
                    parts.append(text)
            elif isinstance(component, Image):
                if caption_idx < len(caption_queue):
                    parts.append(f"[图片: {caption_queue[caption_idx]}]")
                    caption_idx += 1
                else:
                    parts.append("[图片]")
            elif isinstance(component, Record):
                parts.append("[语音]")
            elif isinstance(component, Video):
                parts.append("[视频]")
            elif isinstance(component, File):
                file_name = component.name or "未知文件"
                parts.append(f"[文件: {file_name}]")
            elif isinstance(component, Face):
                parts.append(f"[表情:{component.id}]")
            elif isinstance(component, At):
                if isinstance(component, AtAll):
                    parts.append("[At:全体成员]")
                else:
                    parts.append(f"[At:{component.qq}]")
            elif isinstance(component, Forward):
                parts.append("[转发消息]")
            elif isinstance(component, Reply):
                if component.message_str:
                    parts.append(f"[引用: {component.message_str[:30]}]")
                else:
                    parts.append("[引用消息]")
            else:
                component_type = getattr(
                    component,
                    "type",
                    component.__class__.__name__,
                )
                logger.debug(f"跳过未知消息组件: {component_type}")

        return " ".join(parts).strip()

    async def get_event_message_str(self, event: AstrMessageEvent) -> str:
        """Get normalized raw message text from event."""
        get_message_str = getattr(event, "get_message_str", None)
        raw_message = ""

        if callable(get_message_str):
            raw_message = get_message_str()
            if asyncio.iscoroutine(raw_message):
                raw_message = await raw_message
        else:
            raw_message = getattr(event, "message_str", "")

        if not isinstance(raw_message, str):
            return ""

        return raw_message.strip()

    async def enforce_message_limit(self, session_id: str):
        """执行消息数量上限控制，只删除已被总结的消息"""
        if not self.conversation_manager:
            return

        max_messages = self.config_manager.get(
            "session_manager.max_messages_per_session", 1000
        )
        cleanup_batch_size = self.config_manager.get(
            "session_manager.cleanup_batch_size", 50
        )
        try:
            cleanup_batch_size = int(cleanup_batch_size)
        except (TypeError, ValueError):
            cleanup_batch_size = 50
        cleanup_batch_size = max(1, cleanup_batch_size)

        if (
            not self.conversation_manager.store
            or not self.conversation_manager.store.connection
        ):
            return

        try:
            actual_count = await self.conversation_manager.store.get_message_count(
                session_id
            )

            if actual_count <= max_messages:
                return

            # 获取已总结的消息位置
            last_summarized_index = (
                await self.conversation_manager.get_session_metadata(
                    session_id, "last_summarized_index", 0
                )
            )

            # 计算需要删除的数量；超过上限时按批量清理，减少每轮只删 1 条的抖动。
            overflow_count = actual_count - max_messages
            target_delete = max(overflow_count, cleanup_batch_size)

            # 只能删除已总结的消息，不能删除未总结的
            safe_to_delete = min(target_delete, last_summarized_index)

            if safe_to_delete <= 0:
                logger.debug(
                    f"[{session_id}] 无可删除消息: "
                    f"溢出={overflow_count}, 批量={cleanup_batch_size}, "
                    f"目标删除={target_delete}, 已总结={last_summarized_index}"
                )
                return

            logger.info(
                f"[{session_id}] 开始清理已总结消息: "
                f"总数={actual_count}, 上限={max_messages}, "
                f"溢出={overflow_count}, 批量={cleanup_batch_size}, "
                f"目标删除={target_delete}, 已总结={last_summarized_index}, "
                f"实际删除={safe_to_delete}"
            )

            actually_deleted = await self.conversation_manager.trim_session_messages(
                session_id,
                safe_to_delete,
            )

            new_actual_count = max(0, actual_count - actually_deleted)
            new_summarized_index = await self.conversation_manager.get_session_metadata(
                session_id,
                "last_summarized_index",
                max(0, last_summarized_index - actually_deleted),
            )

            # 清除缓存（使用公共接口）
            await self.conversation_manager.invalidate_cache(session_id)

            logger.info(
                f"[{session_id}] 消息清理完成: "
                f"删除={actually_deleted}条, 剩余={new_actual_count}条, "
                f"总结索引: {last_summarized_index} -> {new_summarized_index}"
            )

        except Exception as e:
            logger.error(f"[{session_id}] 删除旧消息失败: {e}", exc_info=True)

    async def resolve_original_input(self, event: AstrMessageEvent) -> OriginalInput:
        """Read only authentic source text; a CD-owned clone never falls back to prompt text."""
        owned = getattr(event, "_cd_owned_original_input_v1", None)
        if (
            getattr(event, "_chat_dynamics_owned_request", None) is True
            and owned is not True
        ):
            return OriginalInput("unavailable", None, None, (), "")
        if owned is not None and owned is not False and owned is not True:
            return OriginalInput("unavailable", None, None, (), "")
        if owned is True:
            resolver = getattr(event, "_cd_original_input_resolver_v1", None)
            if not callable(resolver):
                return OriginalInput("unavailable", None, None, (), "")
            try:
                # Contract is synchronous and tied by CD to this exact clone.
                dto = resolver(event)
                if inspect.isawaitable(dto):
                    discard_unexpected_awaitable(dto)
                    return OriginalInput("unavailable", None, None, (), "")
                umo = getattr(event, "unified_msg_origin", None)
                if not isinstance(umo, str) or not umo.strip():
                    return OriginalInput("unavailable", None, None, (), "")
                umo = umo.strip()
                platform_name_getter = getattr(event, "get_platform_name", None)
                platform = (
                    platform_name_getter() if callable(platform_name_getter) else None
                )
                platform = (
                    platform.strip().lower()[:64] if isinstance(platform, str) else ""
                )
                self_id_getter = getattr(event, "get_self_id", None)
                sender_getter = getattr(event, "get_sender_id", None)
                self_id = self_id_getter() if callable(self_id_getter) else None
                author_id = sender_getter() if callable(sender_getter) else None
                self_id = self_id.strip() if isinstance(self_id, str) else self_id
                author_id = (
                    author_id.strip() if isinstance(author_id, str) else author_id
                )
                message_obj = getattr(event, "message_obj", None)
                raw_platform_id = getattr(message_obj, "message_id", None)
                platform_id = (
                    str(raw_platform_id).strip()
                    if raw_platform_id is not None
                    and not isinstance(raw_platform_id, bool)
                    else None
                )
                validated = validate_owned_original_input(
                    dto,
                    expected_scope={
                        "umo": umo,
                        "platform": platform,
                        "self_id": self_id,
                    },
                    trigger_author_id=author_id,
                    platform_message_id=platform_id,
                    self_id=self_id,
                )
                return validated or OriginalInput("unavailable", None, None, (), "")
            except Exception:
                # The caller records only the fixed outcome, never exception text.
                return OriginalInput("unavailable", None, None, (), "")
        if owned is not None and owned is not False:
            return OriginalInput("unavailable", None, None, (), "")

        # Unowned events are the unmodified host event path. Only direct Plain
        # components become query text; quote/media parts are not flattened.
        get_messages = getattr(event, "get_messages", None)
        if callable(get_messages):
            try:
                from astrbot.core.message.components import Plain

                components = get_messages()
                if asyncio.iscoroutine(components) or hasattr(components, "__await__"):
                    components = await components
                pieces = [
                    part.text.strip()
                    for part in components
                    if isinstance(part, Plain)
                    and isinstance(getattr(part, "text", None), str)
                    and part.text.strip()
                ]
                query = " ".join(pieces)[:MAX_RECALL_QUERY_CHARS]
                return OriginalInput(
                    "not_owned",
                    None,
                    None,
                    (),
                    query,
                    query_truncated=sum(map(len, pieces)) > MAX_RECALL_QUERY_CHARS,
                )
            except Exception:
                return OriginalInput("not_owned", None, None, (), "")
        try:
            query = await self.get_event_message_str(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            query = ""
        return OriginalInput(
            "not_owned",
            None,
            None,
            (),
            query[:MAX_RECALL_QUERY_CHARS],
            query_truncated=len(query) > MAX_RECALL_QUERY_CHARS,
        )
