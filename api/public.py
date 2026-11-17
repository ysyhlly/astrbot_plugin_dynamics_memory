"""LivingMemory 的对外门面（Public API）。

本模块是**其它插件访问 LivingMemory 的唯一推荐入口**。它的存在是为了把此前
分散在各消费方（group_chronicle / truth_game / group_chat_plus / self_learning）
里的三件私有知识收敛回插件内部：

1. 插件身份别名（原 `core/passive_group_capture.SESSION_PLUGIN_NAMES`，曾被抄成
   4 份且已经分叉）；
2. "如何找到活着的插件实例"（weakref → `get_registered_star` → `get_all_stars`）；
3. "如何从实例上取到引擎 / 会话管理器"（原先由消费方直接 `getattr` 私有属性链
   `plugin.initializer.memory_engine`）。

稳定性承诺（自 `API_VERSION = 1` 起冻结）
------------------------------------------
以下符号是**公共契约**。可以只做向后兼容的扩展（新增参数必须带默认值）；
删除、改名或改变既有语义时必须提升 `API_VERSION`：

- `ALIASES`：插件注册身份的唯一真源（`tuple[str, ...]`）。
- `API_VERSION`：本门面契约版本（int）。
- `SCHEMA_VERSION`：LivingMemory 数据库 schema 版本（int；不可读时为 0）。
- `find_plugin(context)`：三级解析，返回插件实例或 `None`，**永不抛异常**。
- `get_active_plugin()`：只读 LivingMemory 自发布的 weakref，**永不抛异常**。
- `ensure_ready(plugin, timeout=30.0, require=("memory_engine",))`：就绪判定，
  返回 bool，**永不抛异常**。
- `get_memory_engine(plugin)` / `get_conversation_manager(plugin)`：访问器，
  不可用时返回 `None`，**永不抛异常** —— 消费方不要再直接读私有属性链。
- `get_history_reader(context)`：只读历史读取器（见下），不可用时返回 `None`。
- `unavailable_stage(context=None, plugin=None)` / `unavailable_reason(...)`：
  不可用阶段与中文原因。
- `Unavailable`：显式异常，仅由 `require_plugin` / `require_engine` 抛出。

**不承诺**（内部实现，消费方不应依赖）
--------------------------------------
- `plugin.initializer.<任意属性>`、`plugin._terminating` 等私有字段的具体属性链。
- `ConversationStore.connection` 的具体类型与生命周期；请改用 `get_history_reader`。
- `messages` 表的表名与列名**不属本契约**：历史表 schema 仍可能演进。当前只读
  约定列为 `id, session_id, role, content, sender_id, sender_name, timestamp,
  metadata`；读取方应在 `SCHEMA_VERSION` 变化时重新核对。
"""

from __future__ import annotations

from typing import Any

from ..core.passive_group_capture import (
    get_active_plugin as _get_active_plugin_impl,
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
    "record_activity_experience",
    "require_engine",
    "require_plugin",
    "unavailable_reason",
    "unavailable_stage",
]

#: 插件注册身份的唯一真源。消费方不要再各自硬编码别名元组。
ALIASES: tuple[str, ...] = (
    "astrbot_plugin_dynamics_memory",
    "DynamicsMemory",
    "群间记忆",
)

#: 本门面契约版本。破坏性变更时必须 +1。
API_VERSION: int = 1

#: 不可用阶段取值（与 `unavailable_stage` 的返回值一致）。
STAGE_NOT_LOADED = "not_loaded"
STAGE_NOT_DETECTED = "not_detected"
STAGE_NOT_READY = "not_ready"

_REASON_BY_STAGE = {
    STAGE_NOT_LOADED: "LivingMemory 插件未加载",
    STAGE_NOT_DETECTED: "未检测到 LivingMemory",
    STAGE_NOT_READY: "LivingMemory 尚未初始化",
}


def _read_schema_version() -> int:
    """尽力读取 LivingMemory 的数据库 schema 版本；不可读时返回 0（未知）。"""
    try:
        from ..storage.db_migration import DBMigration

        return int(getattr(DBMigration, "CURRENT_VERSION", 0) or 0)
    except Exception:  # 迁移模块缺失 / 依赖不可用都不应影响门面导入
        return 0


#: LivingMemory 数据库 schema 版本（0 = 未知）。`messages` 表读取方应据此复核列名。
SCHEMA_VERSION: int = _read_schema_version()


class Unavailable(RuntimeError):
    """门面显式不可用异常。

    Attributes:
        stage: `"not_loaded"` / `"not_detected"` / `"not_ready"`。
        reason: 面向用户的中文原因。
    """

    def __init__(self, stage: str, reason: str = "") -> None:
        self.stage = stage
        self.reason = reason or _REASON_BY_STAGE.get(stage, stage)
        super().__init__(self.reason)


def _initializer_of(plugin: Any) -> Any:
    """Guard 化的 initializer 读取；任何异常都返回 None。"""
    try:
        guard = getattr(plugin, "series_runtime_allowed", None)
        if callable(guard) and not guard():
            return None
        return getattr(plugin, "initializer", None)
    except Exception:
        return None


def _is_initialized(plugin: Any) -> bool:
    """就绪标记判定：同时认 `is_initialized` 与旧版 `_initialization_complete`。"""
    initializer = _initializer_of(plugin)
    if initializer is None:
        return False
    try:
        return bool(
            getattr(initializer, "is_initialized", False)
            or getattr(initializer, "_initialization_complete", False)
        )
    except Exception:
        return False


def _get_attr(obj: Any, name: str) -> Any:
    """Guard 化的属性读取；任何异常都返回 None。"""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def get_active_plugin() -> Any:
    """返回 LivingMemory 自发布的插件实例 weakref 目标，未发布时为 None。

    Returns:
        LivingMemoryPlugin 实例，或 None。**永不抛异常。**

    Notes:
        这是三级解析里的第一级（PRIMARY）。它只反映"本插件自己是否已发布
        weakref"，不涉及 context；完整解析请用 `find_plugin`。
    """
    try:
        return _get_active_plugin_impl()
    except Exception:
        return None


def _star_is_active(star: Any) -> bool:
    """注册星是否可用：未被显式停用，且带插件实例。"""
    if star is None:
        return False
    if _get_attr(star, "activated") is False:
        return False
    return _get_attr(star, "star_cls") is not None


def _star_identity(star: Any) -> set[str]:
    """注册星的全部身份字段（小写、去空白）。未知字段不参与。"""
    names: set[str] = set()
    for field in ("name", "display_name", "root_dir_name", "module_path"):
        value = _get_attr(star, field)
        if value is None:
            continue
        text = str(value).strip().casefold()
        if text:
            names.add(text)
    return names


def _find_plugin_via_context(context: Any) -> Any | None:
    """context 侧解析：`get_registered_star` 逐别名 → `get_all_stars` 元数据扫描。"""
    try:
        getter = getattr(context, "get_registered_star", None)
    except Exception:
        getter = None
    if callable(getter):
        for alias in ALIASES:
            try:
                star = getter(alias)
            except Exception:
                star = None
            if _star_is_active(star):
                return _get_attr(star, "star_cls")

    try:
        scanner = getattr(context, "get_all_stars", None)
    except Exception:
        scanner = None
    if not callable(scanner):
        return None
    try:
        stars = scanner() or []
    except Exception:
        return None
    if isinstance(stars, dict):
        stars = stars.values()

    wanted = {alias.casefold() for alias in ALIASES}
    for star in stars:
        if _star_is_active(star) and (_star_identity(star) & wanted):
            return _get_attr(star, "star_cls")
    return None


def find_plugin(context: Any) -> Any | None:
    """按固定顺序解析 LivingMemory 插件实例。

    FIXED resolution order:

    1. PRIMARY — `get_active_plugin()` 发布的 weakref，直接返回插件实例；
    2. FALLBACK — `context.get_registered_star(alias)`，逐个尝试 `ALIASES`；
    3. LAST RESORT — 扫描 `context.get_all_stars()`，用 star 的
       `name/display_name/root_dir_name/module_path` 与 `ALIASES` 做小写匹配。

    Args:
        context: AstrBot Context（可为 None 或任意残缺对象）。

    Returns:
        插件实例（`star.star_cls`），或 None。**永不抛异常。**

    Notes:
        返回的是 `star_cls`（插件实例），不是 star 元对象。
    """
    plugin = get_active_plugin()
    if plugin is not None:
        return plugin
    if context is None:
        return None
    try:
        return _find_plugin_via_context(context)
    except Exception:
        return None


async def ensure_ready(
    plugin: Any,
    timeout: float = 30.0,
    require: tuple[str, ...] = ("memory_engine",),
) -> bool:
    """判定插件实例是否已就绪可安全使用。

    Args:
        plugin: `find_plugin` 返回的插件实例（可为 None / 任意对象）。
        timeout: 未初始化时 `ensure_initialized` 的等待上限（秒）。
        require: 就绪必须存在的 initializer 属性名。默认只要求 `memory_engine`；
            需要会话历史的调用方传 `("memory_engine", "conversation_manager")`。

    Returns:
        True 表示可安全调用引擎/会话管理器；否则 False。**永不抛异常。**

    Notes:
        只通过 `getattr` 访问，老版本/半初始化实现会安全降级为 False。
    """
    initializer = _initializer_of(plugin)
    if initializer is None:
        return False
    try:
        if not getattr(initializer, "is_initialized", False):
            ensure = getattr(initializer, "ensure_initialized", None)
            if not callable(ensure):
                return False
            try:
                ok = await ensure(timeout=timeout)
            except Exception:
                ok = False
            if not ok:
                return False
        return all(getattr(initializer, name, None) is not None for name in require)
    except Exception:
        return False


def get_memory_engine(plugin: Any) -> Any | None:
    """返回 `plugin.initializer.memory_engine`，不可用时 None。

    Notes:
        消费方应使用本访问器，不要直接读 `plugin.initializer.memory_engine`：
        属性链属内部实现，可能随版本调整。
    """
    return _get_attr(_initializer_of(plugin), "memory_engine")


def get_conversation_manager(plugin: Any) -> Any | None:
    """返回 `plugin.initializer.conversation_manager`，不可用时 None。"""
    return _get_attr(_initializer_of(plugin), "conversation_manager")


def get_history_reader(context: Any) -> Any | None:
    """返回 LivingMemory 会话历史的**只读**读取器（当前是 store 的 SQLite 连接）。

    Args:
        context: AstrBot Context。

    Returns:
        保证带 `execute(...)` 的对象；不可用时 None。**永不抛异常。**

    Notes:
        读取方只应执行 SELECT，且只读 `messages` 的约定列
        （`id, session_id, role, content, sender_id, sender_name, timestamp,
        metadata`）。表名/列名不是本门面契约，`SCHEMA_VERSION` 变化时需复核。
        返回对象当前与 LivingMemory 自身共用同一连接，因此读取方必须避免长事务。
    """
    plugin = find_plugin(context)
    manager = get_conversation_manager(plugin) if plugin is not None else None
    store = _get_attr(manager, "store")
    connection = _get_attr(store, "connection")
    if callable(_get_attr(connection, "execute")):
        return connection
    return None


#: `messages` 只读约定列。表名/列名不是门面契约，`SCHEMA_VERSION` 变化时需复核。
HISTORY_COLUMNS: tuple[str, ...] = (
    "id",
    "session_id",
    "role",
    "content",
    "sender_id",
    "sender_name",
    "timestamp",
    "metadata",
)
#: group session_id 的匹配模式（原 group_chronicle 内的私有约定，收归 LM）。
GROUP_SESSION_LIKE = "%:GroupMessage:%"

_HISTORY_SESSIONS_SQL = (
    "SELECT DISTINCT session_id FROM messages WHERE session_id LIKE ?"
)
_HISTORY_BOUNDS_SQL = (
    "SELECT COUNT(*),COALESCE(MAX(id),0),MIN(timestamp),MAX(timestamp) "
    "FROM messages WHERE session_id=?"
)
_HISTORY_PAGE_SQL = (
    "SELECT id,role,content,sender_id,sender_name,timestamp,metadata FROM messages "
    "WHERE session_id=? AND id>? AND id<=? ORDER BY id ASC LIMIT ?"
)


def _require_history_reader(connection: Any) -> Any:
    """校验只读读取器；缺失时抛 `Unavailable`（RuntimeError 子类）。"""
    if connection is None or not callable(_get_attr(connection, "execute")):
        raise Unavailable(STAGE_NOT_READY, "LivingMemory 历史连接尚未就绪")
    return connection


async def query_history_sessions(connection: Any) -> list[str]:
    """列出已有 group 会话 id（只读，单条 SELECT）。

    Args:
        connection: `get_history_reader(context)` 返回的读取器。

    Returns:
        会话 id 字符串列表（无数据时 []）。

    Raises:
        Unavailable: 读取器不可用。

    Notes:
        `messages` 表名与列名由本门面承担；消费方不要再自己拼这段 SQL。
    """
    reader = _require_history_reader(connection)
    async with reader.execute(_HISTORY_SESSIONS_SQL, (GROUP_SESSION_LIKE,)) as cursor:
        return [str(row[0]) for row in await cursor.fetchall()]


async def query_history_bounds(connection: Any, session_id: str) -> dict[str, Any]:
    """取某会话的消息边界（只读，单条 SELECT）。

    Returns:
        `{"count": int, "high": int, "first": <ts|None>, "last": <ts|None>}`。

    Raises:
        Unavailable: 读取器不可用。
    """
    reader = _require_history_reader(connection)
    async with reader.execute(_HISTORY_BOUNDS_SQL, (session_id,)) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return {"count": 0, "high": 0, "first": None, "last": None}
    return {
        "count": int(row[0]),
        "high": int(row[1]),
        "first": row[2],
        "last": row[3],
    }


async def query_history_page(
    connection: Any,
    session_id: str,
    after: Any,
    high: Any,
    limit: Any,
) -> list[dict[str, Any]]:
    """按 id keyset 读取一页消息（只读，单条 SELECT）。

    Args:
        connection: 只读读取器。
        session_id: 会话 id。
        after: 排他下界（已处理到的 id）。
        high: 含入上界（快照水位）。
        limit: 最多返回条数（至少 1）。

    Returns:
        与既有消费方一致的字典列表：`id / role / content / uid / name / ts /
        metadata`（metadata 已解码为 dict，时间戳已转 float）。

    Raises:
        Unavailable: 读取器不可用。

    Notes:
        表名、列名、metadata 解码与时间戳强制转换都由门面负责。
    """
    import json as _json

    reader = _require_history_reader(connection)
    async with reader.execute(
        _HISTORY_PAGE_SQL,
        (session_id, int(after), int(high), max(1, int(limit))),
    ) as cursor:
        rows = await cursor.fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            metadata = _json.loads(row[6]) if isinstance(row[6], str) else row[6]
        except (ValueError, TypeError):
            metadata = {}
        try:
            timestamp = float(row[5] or 0)
        except (ValueError, TypeError, OverflowError):
            timestamp = 0
        result.append(
            {
                "id": int(row[0]),
                "role": str(row[1]),
                "content": str(row[2] or ""),
                "uid": str(row[3] or ""),
                "name": str(row[4] or row[3] or ""),
                "ts": timestamp,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )
    return result


def unavailable_stage(context: Any = None, plugin: Any = None) -> str:
    """返回不可用阶段：`""` / `"not_detected"` / `"not_ready"`。

    Notes:
        门面本身能被 import 就说明 LivingMemory 包已加载，因此 `"not_loaded"`
        只能由调用方在**门面 import 失败**时自行判定（消费方通常已用
        guarded import 处理这一情形）。
    """
    if plugin is None:
        plugin = find_plugin(context)
    if plugin is None:
        return STAGE_NOT_DETECTED
    if not _is_initialized(plugin):
        return STAGE_NOT_READY
    return ""


def unavailable_reason(context: Any = None, plugin: Any = None) -> str:
    """返回面向用户的中文不可用原因；可用时返回空字符串。

    Notes:
        `"not_loaded"` 同样需由调用方在门面 import 失败时传入
        （`unavailable_reason(stage="not_loaded")` 形式的兼容请用 `Unavailable`）。
    """
    stage = unavailable_stage(context, plugin)
    return _REASON_BY_STAGE.get(stage, "") if stage else ""


def require_plugin(context: Any) -> Any:
    """返回插件实例；不可用时抛 `Unavailable`。

    Raises:
        Unavailable: stage 为 `"not_detected"`。
    """
    plugin = find_plugin(context)
    if plugin is None:
        raise Unavailable(STAGE_NOT_DETECTED)
    return plugin


def require_engine(context: Any) -> Any:
    """返回 memory_engine；不可用时抛 `Unavailable`。

    Raises:
        Unavailable: 插件未检测到（`"not_detected"`）或引擎缺失（`"not_ready"`）。
    """
    plugin = require_plugin(context)
    engine = get_memory_engine(plugin)
    if engine is None:
        raise Unavailable(STAGE_NOT_READY)
    return engine


async def record_activity_experience(
    context: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    """向群间记忆写入经过核实的群间游戏协作活动经历（幂等）。"""
    plugin = find_plugin(context)
    if plugin is None:
        return {"schema_version": 1, "recorded": False, "reason": STAGE_NOT_DETECTED}
    fn = _get_attr(plugin, "record_activity_experience")
    if not callable(fn):
        return {"schema_version": 1, "recorded": False, "reason": "unsupported"}
    return await fn(payload)
