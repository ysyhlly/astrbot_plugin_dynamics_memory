"""Bound retrieval latency while preserving independent successful routes."""

import asyncio
import time

from astrbot.api import logger


async def search_route(name, awaitable, timeout=10.0, *, safe_errors=False):
    """Execute one route without allowing an ordinary failure to escape.

    Args:
        name: Route name used for diagnostics.
        awaitable: Retrieval coroutine.
        timeout: Maximum duration in seconds.

    Returns:
        A pair of results and an optional exception. Cancellation propagates.
    """
    if _vector_route_cooling_down() and "vector" in str(name):
        # 供应商已确认不可用：跳过一次注定失败的请求，并关闭未等待的协程。
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return [], RuntimeError("vector_route_suppressed_provider_unavailable")
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout), None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _log_route_failure(name, exc, safe_errors)
        if "vector" in str(name) and _is_provider_hard_error(exc):
            _enter_vector_cooldown()
        return [], exc


_FAILURE_LOG_WINDOW_SECONDS = 300.0

_VECTOR_COOLDOWN_SECONDS = 300.0

_HARD_PROVIDER_ERROR_MARKERS = (
    "arrearage",
    "access denied",
    "invalid api key",
    "unauthorized",
    "401",
    "403",
)

_failure_log_state: dict[str, tuple[float, int]] = {}

_vector_cooldown_until = 0.0


def _is_provider_hard_error(exc: Exception) -> bool:
    """供应商侧拒绝（欠费/额度/鉴权）而不是查询本身出错。"""
    text = str(exc).lower()
    return any(marker in text for marker in _HARD_PROVIDER_ERROR_MARKERS)


def _vector_route_cooling_down() -> bool:
    return time.monotonic() < _vector_cooldown_until


def _enter_vector_cooldown() -> None:
    global _vector_cooldown_until
    _vector_cooldown_until = time.monotonic() + _VECTOR_COOLDOWN_SECONDS


def _log_route_failure(name, exc, safe_errors=False) -> None:
    """同类失败在窗口期内只记录一条 WARN，其余降为 DEBUG 并累计抑制次数。"""
    key = f"{name}:{type(exc).__name__}"
    now = time.monotonic()
    last_ts, suppressed = _failure_log_state.get(key, (0.0, 0))
    if last_ts and now - last_ts < _FAILURE_LOG_WINDOW_SECONDS:
        _failure_log_state[key] = (last_ts, suppressed + 1)
        logger.debug(
            "Retrieval route %s failed (suppressed %d): %s", name, suppressed + 1, exc
        )
        return
    _failure_log_state[key] = (now, 0)
    suffix = f"（已抑制 {suppressed} 次同类告警）" if suppressed else ""
    if safe_errors:
        logger.warning(
            "[LMRecall] route=%s outcome=error exception_type=%s%s",
            name,
            type(exc).__name__,
            suffix,
        )
    else:
        logger.warning("Retrieval route %s failed: %s%s", name, exc, suffix)
    if _is_provider_hard_error(exc):
        logger.warning(
            "Embedding 供应商拒绝请求（欠费/额度不足/鉴权失败），向量路由暂停 %d 秒，"
            "检索自动回退关键词路由；供应商恢复后自动继续。",
            int(_VECTOR_COOLDOWN_SECONDS),
        )
