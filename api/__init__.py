"""LivingMemory 正式对外门面包。

其它插件应只从本包导入，例如::

    from astrbot_plugin_dynamics_memory.api.public import ALIASES, find_plugin

`core.*` 下的模块（含 `core.passive_group_capture`）都是**内部实现**，
不保证跨版本稳定。
"""

from .public import (
    ALIASES,
    API_VERSION,
    GROUP_SESSION_LIKE,
    HISTORY_COLUMNS,
    SCHEMA_VERSION,
    Unavailable,
    ensure_ready,
    find_plugin,
    get_active_plugin,
    get_conversation_manager,
    get_history_reader,
    get_memory_engine,
    query_history_bounds,
    query_history_page,
    query_history_sessions,
    require_engine,
    require_plugin,
    unavailable_reason,
    unavailable_stage,
)

__all__ = [
    "ALIASES",
    "API_VERSION",
    "GROUP_SESSION_LIKE",
    "HISTORY_COLUMNS",
    "SCHEMA_VERSION",
    "Unavailable",
    "ensure_ready",
    "find_plugin",
    "get_active_plugin",
    "get_conversation_manager",
    "get_history_reader",
    "get_memory_engine",
    "query_history_bounds",
    "query_history_page",
    "query_history_sessions",
    "require_engine",
    "require_plugin",
    "unavailable_reason",
    "unavailable_stage",
]
