"""供 Agent 主动调用的长期记忆写入工具。"""

import asyncio
from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.platform import MessageType
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from ..memory_scope import (
    adapter_access,
    classify_umo,
    is_event_memory_allowed,
    resolve_memory_write_scope,
)
from ..narrative_api import scope_key
from ..utils import get_persona_id
from .agentic_recall_common import json_result as _json_result


def _normalize_list(value: Any, limit: int = 5) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:limit]
    if isinstance(value, str) and value.strip():
        return [value.strip()][:limit]
    return []


@dataclass
class MemoryMemorizeTool(FunctionTool[AstrAgentContext]):
    """长期记忆主动写入工具。"""

    __pydantic_config__ = {"arbitrary_types_allowed": True}

    context: Any = None
    config_manager: Any = None
    memory_engine: Any = None
    memory_processor: Any = None

    name: str = "memorize_long_term_memory"
    description: str = (
        "Memorize durable long-term memory when the user explicitly asks to remember something, "
        "or when stable preferences, identity details, agreements, or project context appear. "
        "Write concise factual memory, not the full conversation."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "Concise factual long-term memory to save. Do not copy the full conversation.",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional short topic tags for this memory, up to 5.",
                    "default": [],
                },
                "key_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional key facts supporting the memory, up to 5.",
                    "default": [],
                },
                "sentiment": {
                    "type": "string",
                    "description": "Sentiment of the memory: positive, neutral, or negative.",
                    "default": "neutral",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance from 0.0 to 1.0. Use higher values for durable preferences, commitments, or identity facts.",
                    "default": 0.7,
                },
                "reason": {
                    "type": "string",
                    "description": "Optional short reason why this information should be remembered.",
                    "default": "",
                },
            },
            "required": ["memory"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        memory: str,
        topics: list[str] | None = None,
        key_facts: list[str] | None = None,
        sentiment: str = "neutral",
        importance: float = 0.7,
        reason: str = "",
    ) -> ToolExecResult:
        guard = getattr(self, "_series_owner_guard", None)
        if callable(guard) and not guard():
            return "群间记忆当前不可用：记忆提供者冲突或已停用。"
        with adapter_access(self.config_manager, context.context.event, write=True):
            """执行长期记忆写入。"""
            cleaned_memory = (memory or "").strip()
            if not cleaned_memory:
                return _json_result({"memorized": False, "error": "memory is empty"})

            normalized_sentiment = str(sentiment or "neutral").strip().lower()
            if normalized_sentiment not in {"positive", "neutral", "negative"}:
                normalized_sentiment = "neutral"

            if (
                self.context is None
                or self.memory_engine is None
                or self.memory_processor is None
            ):
                return _json_result(
                    {
                        "memorized": False,
                        "error": "memory memorize tool is not initialized",
                    }
                )

            try:
                event = context.context.event
                if not is_event_memory_allowed(self.config_manager, event):
                    return _json_result(
                        {"memorized": False, "error": "memory access is not allowed"}
                    )
                session_id = event.unified_msg_origin
                memory_scope = resolve_memory_write_scope(self.config_manager, event)
                managed = getattr(event, "_extras", {}).get("_narrative_managed_v1")
                provenance = []
                if isinstance(managed, dict):
                    scope = managed.get("scope", {})
                    if scope.get("umo") != session_id or managed.get("internal"):
                        return _json_result(
                            {
                                "memorized": False,
                                "error": "managed_scope_or_authorization_missing",
                            }
                        )
                    memory_scope = scope_key(scope)
                    persona_id = scope["persona_id"]
                    for source in managed.get("events", []):
                        if all(
                            source.get("scope", {}).get(k) == scope.get(k)
                            for k in ("platform", "umo", "persona_id", "instance_id")
                        ) and source.get("event_id"):
                            provenance.append(
                                {
                                    "source_id": source["event_id"],
                                    "scope": dict(scope),
                                    "fact_kind": source.get("fact_kind", "user_claim"),
                                    "occurred_at": source.get("occurred_at"),
                                }
                            )
                    if not provenance:
                        return _json_result(
                            {"memorized": False, "error": "source_evidence_missing"}
                        )
                    # A model-written summary is an interpretation even if it sounds
                    # factual. Saving it creates no promise, reminder or execution.
                    cleaned_memory = (
                        "[角色主观记忆；待核实，不代表约定或执行已完成] "
                        + cleaned_memory[:2000]
                    )
                else:
                    persona_id = await get_persona_id(self.context, event)
                is_group_chat = event.get_message_type() == MessageType.GROUP_MESSAGE

                structured_data = {
                    "summary": cleaned_memory,
                    "topics": _normalize_list(topics),
                    "key_facts": [] if provenance else _normalize_list(key_facts),
                    "sentiment": normalized_sentiment,
                    "importance": importance,
                }

                content, metadata, normalized_importance = (
                    self.memory_processor.build_memory_from_structured_data(
                        structured_data=structured_data,
                        is_group_chat=is_group_chat,
                        fallback_excerpt=cleaned_memory,
                    )
                )
                metadata["source_window"] = {
                    "session_id": session_id,
                    "triggered_by": "agent_tool",
                    "tool_name": self.name,
                }
                metadata["memory_origin"] = "agent_memorize_tool"
                metadata["source_session_id"] = session_id
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
                if provenance:
                    metadata.update(
                        narrative_provenance=provenance,
                        narrative_scope=dict(scope),
                        fact_kind="subjective",
                        unknown=True,
                        memory_origin="narrative_agent_interpretation",
                        narrative_turn_id=managed.get("turn_id"),
                    )
                cleaned_reason = (reason or "").strip()
                if cleaned_reason:
                    metadata["memorize_reason"] = cleaned_reason

                memory_id = await self.memory_engine.add_memory(
                    content=content,
                    session_id=memory_scope,
                    persona_id=persona_id,
                    importance=normalized_importance,
                    metadata=metadata,
                )

                return _json_result(
                    {
                        "memorized": True,
                        "id": memory_id,
                        "content": content,
                        "importance": normalized_importance,
                        "session_id": memory_scope,
                        "persona_id": persona_id,
                    }
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"记忆工具写入失败: {e}", exc_info=True)
                return _json_result({"memorized": False, "error": "internal_error"})
