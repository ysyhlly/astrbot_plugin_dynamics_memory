"""Conditional fast-LLM query rewriting with parallel raw-query fallback.

Only applies to passive hook recall (``memory_recall.py``) and narrative
context preparation. LLM-initiated ``memory_search_tool`` queries are already
explicit retrieval terms and MUST bypass this rewriter.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from astrbot.api import logger

_EFFECTIVE_CHAR_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# Coreference / pronoun / ellipsis markers that benefit from recent-turn resolution.
_COREFERENCE_RE = re.compile(
    r"(?:他|她|它|他们|她们|它们|那个|这个|那些|这些|那家|这家|那儿|那事|那次|"
    r"那个谁|那个人|那件事|那个地方|前面说的|刚才说的|上面说的|之前说的|还记得|"
    r"同上|后来呢|然后呢|怎么样了|咋样了|叫啥来着|是什么来着|在哪来着|"
    r"上次说的|之前提过|当时|原本|刚才|"
    r"\b(?:he|she|it|they|them|that one|this one|those|earlier|aforementioned)\b)",
    re.IGNORECASE,
)

# Relative / recent temporal expressions.
_RECENT_TEMPORAL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"刚才|刚刚|方才|just now", re.IGNORECASE), 1),
    (re.compile(r"昨晚|今天早|今早|今天|today|tonight", re.IGNORECASE), 1),
    (re.compile(r"昨天|昨日|yesterday", re.IGNORECASE), 2),
    (re.compile(r"前天|前两天", re.IGNORECASE), 3),
    (
        re.compile(
            r"前几天|这几天|这周|本周|上周|上星期|last week|this week", re.IGNORECASE
        ),
        7,
    ),
    (re.compile(r"上次|最近|近期|前阵子|recently|last time", re.IGNORECASE), 14),
)

_SPECIFIC_DATE_RE = re.compile(
    r"(?:\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*[日号])?|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]|"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"去年|前年|上个月|上月)"
)

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)

_REWRITE_SYSTEM_PROMPT = (
    "你是聊天记忆检索查询改写器。根据最近几轮对话、引用消息（如有）和当前发言，"
    "把含指代/省略/时间词的口语消息改写为清晰、独立的记忆检索查询。\n"
    "要求：\n"
    "1. 只输出合法 JSON，不要任何解释或多余文本。\n"
    "2. 补全代词指代的人名、对象、事件，不要编造上下文未提及的事实。\n"
    "3. 格式："
    '{"standalone_query":"独立完整检索句","sub_queries":["可选1-2个关键实体或话题短语"],'
    '"temporal_intent":"recent|specific_date|none","time_hint_days":null}\n'
    "4. temporal_intent 仅在询问近期事件（昨天/上次/最近/刚才/上周等）时为 recent，"
    "询问明确日期时为 specific_date，常规事实/偏好/设定为 none。"
)


@dataclass(slots=True)
class RewriteResult:
    """Structured output of conditional query rewriting."""

    standalone_query: str
    sub_queries: list[str] = field(default_factory=list)
    temporal_intent: str = "none"  # "recent" | "specific_date" | "none"
    time_hint_days: int | None = None
    triggered: bool = False
    rewritten: bool = False
    trigger_reason: str = "none"
    fallback_reason: str | None = None
    elapsed_ms: float = 0.0

    @property
    def all_queries(self) -> list[str]:
        """Deduplicated ordered list ``[standalone_query, *sub_queries]``."""
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in (self.standalone_query, *self.sub_queries):
            text = str(candidate or "").strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)
        return ordered


def detect_temporal_intent(query: str) -> tuple[str, int | None]:
    """Detect temporal intent and soft day-hint using lightweight regex (<0.2ms)."""
    text = str(query or "").strip()
    if not text:
        return "none", None
    for pattern, hint_days in _RECENT_TEMPORAL_PATTERNS:
        if pattern.search(text):
            return "recent", hint_days
    if _SPECIFIC_DATE_RE.search(text):
        return "specific_date", None
    return "none", None


def should_rewrite_query(
    query: str,
    *,
    recent_messages: Sequence[Any] = (),
    reply_quote: str | None = None,
    short_threshold: int = 15,
) -> tuple[bool, str, str, int | None]:
    """Fast (<1ms) heuristic gate deciding whether to invoke fast-LLM rewrite.

    Triggers ONLY on:
    1. Present QQ / platform reply quote;
    2. Relative temporal expressions (昨天/前天/上周/这周/前几天/最近/上次/刚才);
    3. Pronouns / coreference / elliptical follow-up markers;
    4. Short query (<15 effective chars) when recent same-session context exists.

    Returns:
        ``(should_trigger, reason, heuristic_temporal_intent, time_hint_days)``
    """
    cleaned = str(query or "").strip()
    temporal_intent, time_hint_days = detect_temporal_intent(cleaned)
    if not cleaned:
        return False, "empty", "none", None

    if reply_quote and str(reply_quote).strip():
        return True, "reply_quote", temporal_intent, time_hint_days

    if temporal_intent != "none":
        return True, "temporal", temporal_intent, time_hint_days

    if _COREFERENCE_RE.search(cleaned):
        return True, "coreference", temporal_intent, time_hint_days

    effective_len = len(_EFFECTIVE_CHAR_RE.findall(cleaned))
    has_context = bool(recent_messages)
    if effective_len < short_threshold and has_context:
        return True, "short_with_context", temporal_intent, time_hint_days

    return False, "none", temporal_intent, time_hint_days


def extract_reply_quote(event: Any) -> str | None:
    """Extract quoted reply text from an AstrMessageEvent if present."""
    if event is None:
        return None
    raw_components = None
    msg_obj = getattr(event, "message_obj", None)
    if msg_obj is not None:
        raw_components = getattr(msg_obj, "message", None)
    if not isinstance(raw_components, (list, tuple)):
        get_messages = getattr(event, "get_messages", None)
        if callable(get_messages):
            try:
                maybe = get_messages()
                if isinstance(maybe, (list, tuple)):
                    raw_components = maybe
            except Exception:
                raw_components = None
    if not isinstance(raw_components, (list, tuple)):
        return None
    for comp in raw_components:
        if type(comp).__name__ != "Reply":
            continue
        for attr in ("message_str", "text", "content"):
            val = getattr(comp, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()[:240]
        chain = getattr(comp, "chain", None)
        if isinstance(chain, (list, tuple)):
            pieces = [
                str(getattr(part, "text", "") or "").strip()
                for part in chain
                if getattr(part, "text", None)
            ]
            joined = " ".join(p for p in pieces if p).strip()
            if joined:
                return joined[:240]
    return None


def _format_recent_turn(msg: Any) -> str:
    """Render a recent message (Message object or dict) concisely for rewrite prompt."""
    if isinstance(msg, dict):
        role = str(msg.get("role") or "user")
        sender = str(msg.get("sender_name") or msg.get("sender_id") or role).strip()
        content = str(msg.get("content") or "").strip()
    else:
        role = str(getattr(msg, "role", "user") or "user")
        sender = str(
            getattr(msg, "sender_name", None) or getattr(msg, "sender_id", None) or role
        ).strip()
        content = str(getattr(msg, "content", "") or "").strip()
    if not content:
        return ""
    if role == "assistant":
        sender = "助手"
    compact = re.sub(r"\s+", " ", content)[:180]
    return f"{sender}: {compact}"


def _parse_rewrite_json(
    raw_text: str,
    fallback_query: str,
    heuristic_temporal: str,
    heuristic_days: int | None,
) -> tuple[str, list[str], str, int | None] | None:
    """Parse structured JSON from fast LLM output; returns None on invalid JSON."""
    text = str(raw_text or "").strip()
    if not text:
        return None
    match = _JSON_FENCE_RE.search(text)
    if match:
        candidate = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None

    standalone = str(parsed.get("standalone_query") or "").strip()
    if not standalone:
        standalone = fallback_query
    standalone = re.sub(r"\s+", " ", standalone)[:300]

    raw_subs = parsed.get("sub_queries")
    sub_queries: list[str] = []
    if isinstance(raw_subs, list):
        for item in raw_subs:
            sub = re.sub(r"\s+", " ", str(item or "")).strip()[:120]
            if sub and sub != standalone and sub not in sub_queries:
                sub_queries.append(sub)
            if len(sub_queries) >= 2:
                break

    temporal_raw = str(parsed.get("temporal_intent") or "").strip().lower()
    if temporal_raw in {"recent", "specific_date", "none"}:
        temporal_intent = temporal_raw
    else:
        temporal_intent = heuristic_temporal

    time_hint_raw = parsed.get("time_hint_days")
    time_hint_days: int | None = heuristic_days
    if time_hint_raw is not None and not isinstance(time_hint_raw, bool):
        try:
            days_val = int(float(time_hint_raw))
            if days_val > 0:
                time_hint_days = min(days_val, 3650)
        except (TypeError, ValueError):
            pass
    elif temporal_intent == "none":
        time_hint_days = None

    return standalone, sub_queries, temporal_intent, time_hint_days


def _extract_llm_text(resp: Any) -> str:
    """Extract completion text from AstrBot LLMResponse or raw string."""
    if isinstance(resp, str):
        return resp
    for attr in ("completion_text", "text", "content"):
        val = getattr(resp, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    raw = getattr(resp, "raw_completion", None)
    if raw is not None:
        choices = getattr(raw, "choices", None)
        if isinstance(choices, list) and choices:
            msg = getattr(choices[0], "message", None)
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content
    return ""


class QueryRewriter:
    """Conditional fast-LLM query rewriter with strict timeout and zero-penalty fallback."""

    def __init__(
        self,
        context: Any = None,
        config_manager: Any = None,
        *,
        provider_resolver: Callable[[], Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.context = context
        self.config_manager = config_manager
        self._provider_resolver = provider_resolver
        self._config = config or {}

    def _get_cfg(self, dotted_key: str, flat_key: str, default: Any) -> Any:
        if self.config_manager is not None:
            getter = getattr(self.config_manager, "get", None)
            if callable(getter):
                val = getter(dotted_key, None)
                if val is not None:
                    return val
        if flat_key in self._config:
            return self._config[flat_key]
        return default

    @property
    def enabled(self) -> bool:
        return bool(
            self._get_cfg(
                "recall_engine.query_rewrite_enabled", "query_rewrite_enabled", True
            )
        )

    @property
    def timeout_seconds(self) -> float:
        raw = self._get_cfg(
            "recall_engine.query_rewrite_timeout_seconds",
            "query_rewrite_timeout_seconds",
            0.55,
        )
        try:
            return max(0.05, min(float(raw), 5.0))
        except (TypeError, ValueError):
            return 0.55

    def _resolve_provider(self) -> Any:
        if callable(self._provider_resolver):
            try:
                return self._provider_resolver()
            except Exception:
                return None

        if self.context is None:
            return None

        provider_id = str(
            self._get_cfg(
                "recall_engine.query_rewrite_provider_id",
                "query_rewrite_provider_id",
                "",
            )
            or ""
        ).strip()

        candidates: list[str] = []
        if provider_id:
            candidates.append(provider_id)
        else:
            bg_id = str(
                self._get_cfg(
                    "provider_settings.background_llm_provider_id",
                    "background_llm_provider_id",
                    "",
                )
                or ""
            ).strip()
            if bg_id:
                candidates.append(bg_id)
            candidates.append("cmdcode/deepseek/deepseek-v4.1-flash")
            llm_id = str(
                self._get_cfg(
                    "provider_settings.llm_provider_id",
                    "llm_provider_id",
                    "",
                )
                or ""
            ).strip()
            if llm_id:
                candidates.append(llm_id)

        provider_manager = getattr(self.context, "provider_manager", None)
        inst_map = getattr(provider_manager, "inst_map", None)
        getter = getattr(self.context, "get_provider_by_id", None)
        for pid in candidates:
            if isinstance(inst_map, dict) and pid in inst_map:
                found = inst_map.get(pid)
                if found is not None:
                    return found
            if callable(getter):
                try:
                    found = getter(pid)
                    if found is not None:
                        return found
                except Exception:
                    continue
        return None

    async def _invoke_provider(self, provider: Any, prompt: str) -> str:
        """Call fast LLM with thinking disabled, temperature=0.0, max_tokens=160."""
        text_chat = getattr(provider, "text_chat", None)
        if not callable(text_chat):
            raise RuntimeError("rewrite provider missing text_chat")

        extra_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": 160,
            "extra_body": {
                "enable_thinking": False,
                "thinking": {"type": "disabled"},
            },
        }
        try:
            sig = inspect.signature(text_chat)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "system_prompt": _REWRITE_SYSTEM_PROMPT,
            }
            if accepts_var_kw:
                call_kwargs.update(extra_kwargs)
            else:
                for k, v in extra_kwargs.items():
                    if k in sig.parameters:
                        call_kwargs[k] = v
            resp = await text_chat(**call_kwargs)
        except TypeError:
            resp = await text_chat(
                prompt=prompt,
                system_prompt=_REWRITE_SYSTEM_PROMPT,
            )
        return _extract_llm_text(resp)

    def _build_prompt(
        self,
        query: str,
        *,
        sender_name: str | None = None,
        sender_id: str | None = None,
        recent_messages: Sequence[Any] = (),
        reply_quote: str | None = None,
        current_date: str | None = None,
    ) -> str:
        lines: list[str] = []
        date_str = current_date or datetime.now().strftime("%Y-%m-%d")
        lines.append(f"当前日期: {date_str}")
        speaker = str(sender_name or sender_id or "").strip()
        if speaker:
            lines.append(f"当前发言人: {speaker}")
        if reply_quote and str(reply_quote).strip():
            lines.append(f"引用消息: {str(reply_quote).strip()[:240]}")
        formatted_turns = [
            line
            for line in (_format_recent_turn(m) for m in list(recent_messages)[-6:])
            if line
        ]
        if formatted_turns:
            lines.append("最近对话:")
            lines.extend(formatted_turns)
        lines.append(f"当前消息: {query.strip()}")
        lines.append("请输出改写 JSON:")
        return "\n".join(lines)

    async def rewrite(
        self,
        query: str,
        *,
        recent_messages: Sequence[Any] = (),
        reply_quote: str | None = None,
        sender_name: str | None = None,
        sender_id: str | None = None,
        timeout_seconds: float | None = None,
        current_date: str | None = None,
    ) -> RewriteResult:
        """Conditionally rewrite ``query`` within a strict timeout budget."""
        started = time.monotonic()
        raw_query = str(query or "").strip()
        should_trigger, reason, heuristic_temporal, heuristic_days = (
            should_rewrite_query(
                raw_query,
                recent_messages=recent_messages,
                reply_quote=reply_quote,
            )
        )
        if not self.enabled:
            return RewriteResult(
                standalone_query=raw_query,
                temporal_intent=heuristic_temporal,
                time_hint_days=heuristic_days,
                triggered=False,
                rewritten=False,
                trigger_reason=reason,
                fallback_reason="disabled",
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            )
        if not should_trigger:
            return RewriteResult(
                standalone_query=raw_query,
                temporal_intent=heuristic_temporal,
                time_hint_days=heuristic_days,
                triggered=False,
                rewritten=False,
                trigger_reason="none",
                fallback_reason=None,
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            )

        provider = self._resolve_provider()
        if provider is None:
            return RewriteResult(
                standalone_query=raw_query,
                temporal_intent=heuristic_temporal,
                time_hint_days=heuristic_days,
                triggered=True,
                rewritten=False,
                trigger_reason=reason,
                fallback_reason="no_provider",
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            )

        effective_timeout = (
            max(0.05, float(timeout_seconds))
            if timeout_seconds is not None
            else self.timeout_seconds
        )
        prompt = self._build_prompt(
            raw_query,
            sender_name=sender_name,
            sender_id=sender_id,
            recent_messages=recent_messages,
            reply_quote=reply_quote,
            current_date=current_date,
        )
        try:
            raw_response = await asyncio.wait_for(
                self._invoke_provider(provider, prompt),
                timeout=effective_timeout,
            )
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, TimeoutError):
            elapsed_ms = round((time.monotonic() - started) * 1000, 1)
            logger.debug(
                "[LMRecall] route=rewrite outcome=timeout elapsed_ms=%.1f reason=%s",
                elapsed_ms,
                reason,
            )
            return RewriteResult(
                standalone_query=raw_query,
                temporal_intent=heuristic_temporal,
                time_hint_days=heuristic_days,
                triggered=True,
                rewritten=False,
                trigger_reason=reason,
                fallback_reason="timeout",
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 1)
            logger.debug(
                "[LMRecall] route=rewrite outcome=error exception_type=%s elapsed_ms=%.1f",
                type(exc).__name__,
                elapsed_ms,
            )
            return RewriteResult(
                standalone_query=raw_query,
                temporal_intent=heuristic_temporal,
                time_hint_days=heuristic_days,
                triggered=True,
                rewritten=False,
                trigger_reason=reason,
                fallback_reason="error",
                elapsed_ms=elapsed_ms,
            )

        parsed = _parse_rewrite_json(
            raw_response, raw_query, heuristic_temporal, heuristic_days
        )
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        if parsed is None:
            return RewriteResult(
                standalone_query=raw_query,
                temporal_intent=heuristic_temporal,
                time_hint_days=heuristic_days,
                triggered=True,
                rewritten=False,
                trigger_reason=reason,
                fallback_reason="parse_error",
                elapsed_ms=elapsed_ms,
            )
        standalone, sub_queries, temporal_intent, time_hint_days = parsed
        return RewriteResult(
            standalone_query=standalone,
            sub_queries=sub_queries,
            temporal_intent=temporal_intent,
            time_hint_days=time_hint_days,
            triggered=True,
            rewritten=True,
            trigger_reason=reason,
            fallback_reason=None,
            elapsed_ms=elapsed_ms,
        )

    async def rewrite_with_raw_fallback(
        self,
        query: str,
        raw_search_fn: Callable[[], Awaitable[Any]],
        *,
        recent_messages: Sequence[Any] = (),
        reply_quote: str | None = None,
        sender_name: str | None = None,
        sender_id: str | None = None,
        timeout_seconds: float | None = None,
        current_date: str | None = None,
    ) -> tuple[RewriteResult, Any | None]:
        """Run conditional rewrite in parallel with a raw-query prefetch fallback.

        When rewrite is NOT triggered: returns ``(RewriteResult(triggered=False), None)``
        immediately without launching ``raw_search_fn`` (caller runs the normal
        direct path once).

        When rewrite IS triggered: launches ``self.rewrite(...)`` and
        ``raw_search_fn()`` concurrently.
        - If rewrite succeeds within timeout and produces a refined query (or
          sub_queries): cancels the raw fallback task and returns ``(result, None)``
          so the caller executes retrieval on the rewritten query bundle.
        - If rewrite times out, errors, or leaves the query unchanged: awaits
          the already-running ``raw_search_fn`` task and returns
          ``(result, raw_output)`` with zero tail-latency penalty!
        """
        raw_query = str(query or "").strip()
        should_trigger, reason, heuristic_temporal, heuristic_days = (
            should_rewrite_query(
                raw_query,
                recent_messages=recent_messages,
                reply_quote=reply_quote,
            )
        )
        if not self.enabled or not should_trigger:
            return (
                RewriteResult(
                    standalone_query=raw_query,
                    temporal_intent=heuristic_temporal,
                    time_hint_days=heuristic_days,
                    triggered=False,
                    rewritten=False,
                    trigger_reason=reason if not self.enabled else "none",
                    fallback_reason="disabled" if not self.enabled else None,
                    elapsed_ms=0.0,
                ),
                None,
            )

        raw_task = asyncio.create_task(raw_search_fn())
        try:
            rewrite_result = await self.rewrite(
                raw_query,
                recent_messages=recent_messages,
                reply_quote=reply_quote,
                sender_name=sender_name,
                sender_id=sender_id,
                timeout_seconds=timeout_seconds,
                current_date=current_date,
            )
        except BaseException:
            if not raw_task.done():
                raw_task.cancel()
            await asyncio.gather(raw_task, return_exceptions=True)
            raise

        changed = rewrite_result.rewritten and (
            rewrite_result.standalone_query.strip() != raw_query
            or bool(rewrite_result.sub_queries)
        )
        if changed:
            if not raw_task.done():
                raw_task.cancel()
                await asyncio.gather(raw_task, return_exceptions=True)
            else:
                # Consume any exception cleanly
                if not raw_task.cancelled():
                    raw_task.exception()
            return rewrite_result, None

        # Rewrite timed out / failed / unchanged -> immediately use parallel raw result
        raw_output = await raw_task
        return rewrite_result, raw_output


__all__ = [
    "QueryRewriter",
    "RewriteResult",
    "detect_temporal_intent",
    "extract_reply_quote",
    "should_rewrite_query",
]
