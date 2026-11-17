"""Passive group capture helpers for LivingMemory."""

import sys
from .narrative_api import is_managed_event


import weakref
from typing import Any

from astrbot.api import logger, sp
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import CustomFilter
from astrbot.api.platform import MessageType

from .memory_scope import is_event_memory_allowed

SESSION_PLUGIN_NAMES = ("群间记忆", "DynamicsMemory", "astrbot_plugin_dynamics_memory", "LivingMemory", "astrbot_plugin_livingmemory")
_ACTIVE_PLUGIN_REF: weakref.ReferenceType | None = None


def set_active_plugin(plugin: Any) -> None:
    """Track the active plugin instance for passive filter side effects.

    INTERNAL: 由 `main.py` 在插件生命周期内调用，不是对外契约。
    同时写入模块级全局与稳定 holder（见 `_plugin_ref_holder`）。
    """
    global _ACTIVE_PLUGIN_REF
    _ACTIVE_PLUGIN_REF = weakref.ref(plugin) if plugin is not None else None
    _publish_plugin_ref(_ACTIVE_PLUGIN_REF)


def get_active_plugin() -> Any:
    """Return the plugin instance published by LivingMemory, else None.

    INTERNAL —— 外部插件请使用 `astrbot_plugin_dynamics_memory.api.public`
    的 `get_active_plugin()` / `find_plugin(context)`；本函数仅作为
    门面的实现细节与旧调用方的兼容出口保留，行为不会改变。

    B6：优先读稳定 holder（热重载后**旧模块对象里的本函数**也能读到新实例），
    holder 不存在时退回模块级全局。
    """
    holder = _plugin_ref_holder()
    if holder is not None and hasattr(holder, _PLUGIN_REF_ATTR):
        ref = getattr(holder, _PLUGIN_REF_ATTR, None)
        return ref() if ref is not None else None
    if _ACTIVE_PLUGIN_REF is None:
        return None
    return _ACTIVE_PLUGIN_REF()


async def is_session_enabled(session_id: str) -> bool:
    """Mirror AstrBot's session-level shutdown check for passive capture."""
    try:
        session_services = await sp.get_async(
            scope="umo",
            scope_id=session_id,
            key="session_service_config",
            default={},
        )
    except Exception as exc:
        logger.debug(f"[{session_id}] 读取会话总开关失败，默认允许捕获: {exc}")
        return True

    if not isinstance(session_services, dict):
        return True
    session_enabled = session_services.get("session_enabled")
    return True if session_enabled is None else bool(session_enabled)


async def is_plugin_enabled_for_session(session_id: str) -> bool:
    """Mirror AstrBot session-level plugin disable checks for passive capture."""
    try:
        session_plugin_config = await sp.get_async(
            scope="umo",
            scope_id=session_id,
            key="session_plugin_config",
            default={},
        )
    except Exception as exc:
        logger.debug(f"[{session_id}] 读取会话插件开关失败，默认允许捕获: {exc}")
        return True

    if not isinstance(session_plugin_config, dict):
        return True
    session_config = session_plugin_config.get(session_id, {})
    if not isinstance(session_config, dict):
        return True
    disabled_plugins = session_config.get("disabled_plugins", [])
    if not isinstance(disabled_plugins, list):
        return True
    return not any(name in disabled_plugins for name in SESSION_PLUGIN_NAMES)


class PassiveGroupCaptureFilter(CustomFilter):
    """Schedule group-message capture without waking AstrBot's message pipeline."""

    def __init__(self, raise_error: bool = True, plugin=None, **kwargs) -> None:
        if not isinstance(raise_error, bool) and plugin is None:
            plugin = raise_error
            raise_error = True
        super().__init__(raise_error=raise_error, **kwargs)
        self._plugin_ref = weakref.ref(plugin) if plugin is not None else None

    def _get_plugin(self):
        if self._plugin_ref is not None:
            return self._plugin_ref()
        return get_active_plugin()

    @staticmethod
    def _passes_global_whitelist(event: AstrMessageEvent, cfg) -> bool:
        platform_settings = (
            cfg.get("platform_settings", {}) if isinstance(cfg, dict) else {}
        )
        if not platform_settings.get("enable_id_white_list", False):
            return True

        whitelist = [
            str(item).strip()
            for item in platform_settings.get("id_whitelist", [])
            if str(item).strip()
        ]
        if not whitelist or event.get_platform_name() == "webchat":
            return True

        if platform_settings.get("wl_ignore_admin_on_group", False):
            try:
                if (
                    getattr(event, "role", None) == "admin"
                    and event.get_message_type() == MessageType.GROUP_MESSAGE
                ):
                    return True
            except Exception:
                pass

        try:
            group_id = str(event.get_group_id()).strip()
        except Exception:
            group_id = ""

        return event.unified_msg_origin in whitelist or group_id in whitelist

    def filter(self, event: AstrMessageEvent, cfg) -> bool:
        if is_managed_event(event):
            return False
        plugin = self._get_plugin()
        if not plugin or getattr(plugin, "_terminating", False) is True:
            return False
        if not plugin.initializer.is_initialized:
            return False
        if not plugin.config_manager.get(
            "session_manager.enable_full_group_capture", True
        ):
            return False
        try:
            if event.get_message_type() != MessageType.GROUP_MESSAGE:
                return False
        except Exception as exc:
            logger.debug(f"LivingMemory 被动群消息捕获类型检查失败: {exc}")
            return False

        if not self._passes_global_whitelist(event, cfg):
            return False

        if not is_event_memory_allowed(plugin.config_manager, event):
            return False

        plugin._schedule_passive_group_capture(event)
        return False


_PLUGIN_REF_ATTR = "_lm_active_plugin_ref"

def _plugin_ref_holder() -> Any:
    """返回热重载后仍然稳定的 weakref holder；取不到时返回 None。

    背景（B6）：`reload_bootstrap.ensure_current` 在源码指纹变化时会把
    `<package>.*` 子模块从 `sys.modules` 摘除并 delattr
    （keep = {main_module, reload_bootstrap}）。于是**带状态的模块级全局会被换成
    新字典**：消费方在重载前绑定的 `get_active_plugin` 读到的是旧字典里的
    `_ACTIVE_PLUGIN_REF`，永远看不到重载后新实例发布的 weakref，外部因此
    "永久失联"。

    解决：把同一个 weakref 也写到不会被 pop 的模块对象上 —— 优先
    `<package>.main`（始终在 keep 集合里），退化到包对象 `<package>`
    （purge 只匹配 `<package>.` 前缀，裸包名不会命中）。两者都不在
    `sys.modules` 时返回 None（独立导入 / 单测环境），调用方退回模块级全局，
    行为与改动前完全一致。
    """
    # 去掉最后一段（.core），得到包根，例如
    # "data.plugins.astrbot_plugin_dynamics_memory"。用 rsplit 而不是 split[0]：
    # 生产布局下 split[0] 会得到顶级命名空间 "data"（48 个插件共用的模块），
    # 既把运行态写到共享模块上，也让测试的 bare 布局（__package__ 恰为包根）
    # 掩盖该问题。
    package = (__package__ or "").rsplit(".", 1)[0]
    if not package:
        return None
    for name in (package + ".main", package):
        module = sys.modules.get(name)
        if module is not None:
            return module
    return None

def _publish_plugin_ref(ref: "weakref.ReferenceType | None") -> None:
    """把 weakref 副本写到稳定 holder 上（写不进去也不影响主路径）。"""
    holder = _plugin_ref_holder()
    if holder is None:
        return
    try:
        setattr(holder, _PLUGIN_REF_ATTR, ref)
    except Exception:
        pass
