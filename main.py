# ruff: noqa: E402
from .reload_bootstrap import ensure_current

ensure_current(
    __package__,
    __name__,
    __file__,
    checks=[
        ("storage.evidence_sidecar", "EvidenceSidecar", "_bounded_queue_claim_v1"),
        ("core.recall_runtime", "BoundedRecallTaskPool", "_reply_scope_parallel_v1"),
        ("core.managers.memory_engine", "MemoryEngine", "repair_retained_sources"),
        ("core.evidence_view", "EvidenceViewService", "defer_policy_wakeup"),
        ("core.recall_runtime", "BoundedRecallTaskPool", "_summary_cleanup_release"),
        ("core.plugin_helpers_mixin", "PluginHelpersMixin", "_native_memory_recall_v1"),
    ],
)

"""
main.py - LivingMemory 插件主文件
负责插件注册、初始化和生命周期管理
"""

import copy
import inspect
import json
import time
from pathlib import Path
from .core.blocking_policy import BlockingPolicyGate
from .core.evidence_projection import MAX_MATERIALS, PURPOSES
from .core.evidence_view import EvidenceViewService
from .core.recall_runtime import (
    BoundedRecallSearchContext,
    BoundedRecallTaskPool,
    RecallDeadline,
    recall_task_key,
    trace_recall_hook,
    timed_recall_phase,
)
from .core.narrative_api import is_managed_event, scope_key
from .core.retrieval.narrative_history import persona_names, source_origin
from .core.utils import _get_native_persona_id, set_persona_aliases
from .series_panels import get_series_defaults
from .storage.evidence_sidecar import EvidenceSidecar


import asyncio
import re
from collections.abc import AsyncGenerator
from importlib import metadata as importlib_metadata
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools

from .core.base.config_manager import ConfigManager
from .core.command_handler import CommandHandler
from .core.event_handler import EventHandler
from .core.i18n_backend import init as i18n_init
from .core.i18n_backend import t
from .core.managers.backup_manager import BackupManager
from .core.memory_scope import adapter_access
from .core.page_api import PluginPageApi
from .core.passive_group_capture import (
    PassiveGroupCaptureFilter,
    get_active_plugin,
    is_plugin_enabled_for_session,
    is_session_enabled,
    set_active_plugin,
)
from .core.plugin_initializer import PluginInitializer
from .core.tools import (
    AgenticMemorySearchTool,
    MemoryForgetTool,
    MemoryMemorizeTool,
    MemoryReadTool,
    MemorySearchTool,
)


from .core.plugin_helpers_mixin import PluginHelpersMixin, _SCOPE_RECALL_TIMINGS
from .core.series import active_entries, find_role, manifest


_HISTORY_LOOKUP = re.compile(
    r"还记得|记不记得|你记得|回忆|回顾|上次|当时|以前|之前|前几天|昨天|原话|原文|"
    r"\b(?:remember|recall|last time|previously|yesterday|verbatim)\b",
    re.IGNORECASE,
)
_MIN_ASTRBOT_VERSION = "4.29.4"
_ASTRBOT_DISTRIBUTION_NAMES = ("AstrBot", "astrbot")


def _parse_version(v: str) -> tuple[int, ...]:
    m = re.match(r"v?(\d+(?:\.\d+)*)", v.strip(), re.IGNORECASE)
    if not m:
        return ()
    return tuple(int(x) for x in m.group(1).split("."))


def _version_lt(current: str, minimum: str) -> bool:
    current_parts = _parse_version(current)
    minimum_parts = _parse_version(minimum)
    if not current_parts or not minimum_parts:
        return False
    width = max(len(current_parts), len(minimum_parts))
    return current_parts + (0,) * (width - len(current_parts)) < minimum_parts + (
        0,
    ) * (width - len(minimum_parts))


def _detect_astrbot_version() -> str | None:
    for distribution_name in _ASTRBOT_DISTRIBUTION_NAMES:
        try:
            return importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception as exc:
            logger.debug(f"读取 AstrBot 分发版本失败 ({distribution_name}): {exc}")

    for module_name in ("astrbot.core.config.default", "astrbot.core.config"):
        try:
            module = __import__(module_name, fromlist=["VERSION"])
            version_value = getattr(module, "VERSION", None)
        except Exception as exc:
            logger.debug(f"读取 AstrBot 模块版本失败 ({module_name}): {exc}")
            continue
        if version_value:
            return str(version_value)

    return None


_CURRENT_ASTRBOT_VERSION = _detect_astrbot_version()

if _CURRENT_ASTRBOT_VERSION is None:
    logger.debug("未能检测到 AstrBot 版本，跳过 LivingMemory 版本兼容提示")
elif _version_lt(_CURRENT_ASTRBOT_VERSION, _MIN_ASTRBOT_VERSION):
    logger.warning(
        f"AstrBot 版本 {_CURRENT_ASTRBOT_VERSION} 低于推荐版本 {_MIN_ASTRBOT_VERSION}。"
        f"插件视图 / WebUI 功能可能不可用。建议升级 AstrBot 以获得完整体验。"
    )


# 插件元数据由 metadata.yaml 提供。AstrBot 3.5.19+ 会自动识别继承 Star 的插件类，
# 已废弃的 @register 装饰器不再使用，避免宿主未来移除它时插件无法加载。
from .core.series import SeriesMixin


class LivingMemoryPlugin(PluginHelpersMixin, SeriesMixin, Star):
    """LivingMemory 插件主类"""

    def __init__(self, context: Context, config: dict[str, Any]):
        super().__init__(context)
        self.context = context
        self._series_enabled = config.get("series_runtime_enabled", False) is True
        self._blocking_policy_gate = BlockingPolicyGate(context)

        # 获取插件数据目录
        data_dir = str(StarTools.get_data_dir("astrbot_plugin_dynamics_memory"))
        self._data_dir = data_dir

        # 版本变更时自动备份数据（延迟到异步初始化阶段执行，避免 __init__ 中同步 I/O 阻塞）
        self._backup_manager = BackupManager(data_dir)

        # 初始化配置管理器
        self.config_manager = ConfigManager(config)
        persona_aliases = set_persona_aliases(
            (config.get("filtering_settings") or {}).get("persona_aliases", "")
        )
        if persona_aliases:
            logger.info(f"[LivingMemory] 记忆人格别名: {persona_aliases}")

        # 初始化后端 i18n
        raw_language = str(config.get("bot_language", "") or "").strip()
        if not raw_language:
            shared_lang = str(
                get_series_defaults(context).get("ui_language", "") or ""
            ).strip()
            normalized = shared_lang.replace("_", "-").split("-", 1)[0].lower()
            raw_language = (
                normalized if normalized in {"zh", "en", "ja", "ru"} else "zh"
            )
        i18n_init(raw_language)

        # 初始化插件初始化器
        self.initializer = PluginInitializer(context, self.config_manager, data_dir)

        # 事件处理器和命令处理器（初始化后创建）
        self.event_handler: EventHandler | None = None
        self.command_handler: CommandHandler | None = None

        # 后台任务跟踪集合
        self._background_tasks: set[asyncio.Task] = set()
        self._recall_runtime = BoundedRecallTaskPool(
            max_inflight=16, max_cache=32, cache_ttl_seconds=2.0
        )
        self._component_init_lock = asyncio.Lock()
        self._llm_tools_registered = False
        self._terminating = False
        self.evidence_service: EvidenceViewService | None = None

        self.page_api = PluginPageApi(self)
        self.page_api.register_routes()
        self.register_series_routes()
        from importlib import reload
        from . import series_panels

        reload(series_panels).register_series_panels(
            self, "astrbot_plugin_dynamics_memory"
        )
        set_active_plugin(self)

        # 启动非阻塞的初始化任务
        self._create_tracked_task(self._initialize_plugin())

    def _create_tracked_task(self, coro) -> asyncio.Task:
        """创建并跟踪后台任务"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        """取回后台任务异常并记录，避免静默失败。"""
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Background task failed: {exc}", exc_info=exc)

    async def _initialize_plugin(self):
        """初始化插件"""
        if not self.series_runtime_allowed():
            return
        try:
            # 版本变更时自动备份数据（在任何数据库操作之前，通过线程池避免阻塞事件循环）
            await self._backup_manager.backup_if_needed_async()

            # 启动时清理历史备份（版本/适配器/作用域迁移/每日备份）；失败不阻断初始化
            try:
                await self._backup_manager.prune_async(
                    keep_days=self.config_manager.get("backup_settings.keep_days", 7)
                )
            except Exception as e:
                logger.warning(f"[BackupManager] 备份清理失败: {e}")

            # 执行初始化
            success = await self.initializer.initialize()

            if success:
                await self._ensure_runtime_components()

        except Exception as e:
            logger.error(f"插件初始化失败: {e}", exc_info=True)

    async def _ensure_runtime_components(self) -> bool:
        """确保运行期组件（事件/命令处理器、WebUI）已就绪"""
        if not self.series_runtime_allowed():
            return False
        if self._terminating:
            return False
        if not self.initializer.is_initialized:
            return False

        async with self._component_init_lock:
            if self._terminating:
                return False
            # 检查必要组件是否初始化成功
            if not all(
                [
                    self.initializer.memory_engine,
                    self.initializer.memory_processor,
                    self.initializer.conversation_manager,
                ]
            ):
                logger.error("插件初始化不完整：部分核心组件未能初始化")
                return False

            # 创建事件处理器（幂等）
            if not self.event_handler:
                self.event_handler = EventHandler(
                    context=self.context,
                    config_manager=self.config_manager,
                    memory_engine=self.initializer.memory_engine,  # type: ignore[arg-type]
                    memory_processor=self.initializer.memory_processor,  # type: ignore[arg-type]
                    conversation_manager=self.initializer.conversation_manager,  # type: ignore[arg-type]
                    consolidation_manager=self.initializer.consolidation_manager,
                    evidence_filter=self._native_memory_evidence_filter,
                    recall_runtime=getattr(self, "_recall_runtime", None),
                    evidence_revision_provider=self.peek_evidence_revision,
                    summary_guard=self._install_private_summary_guard,
                )
                self._create_tracked_task(self._recover_managed_reflections())

            # 创建命令处理器（幂等）
            if not self.command_handler:
                self.command_handler = CommandHandler(
                    context=self.context,
                    config_manager=self.config_manager,
                    memory_engine=self.initializer.memory_engine,
                    conversation_manager=self.initializer.conversation_manager,
                    index_validator=self.initializer.index_validator,
                    memory_processor=self.initializer.memory_processor,
                    initialization_status_callback=self._get_initialization_status_message,
                )

            self._register_agent_tools_if_needed()

            data_dir = getattr(self, "_data_dir", None)
            if getattr(self, "evidence_service", None) is None and data_dir:
                try:
                    sidecar = EvidenceSidecar(
                        Path(data_dir) / "evidence_sidecar.sqlite3"
                    )
                    service = EvidenceViewService(
                        sidecar=sidecar,
                        scope_key_fn=scope_key,
                        derive_callback=self._derive_evidence_materials,
                        queue_size=128,
                        per_scope_hourly_limit=24,
                        material_loader=self._load_evidence_material,
                        conversation_row_loader=self._load_conversation_row_evidence,
                        event_loader=self._load_archived_evidence_event,
                        material_policy=self._evidence_material_policy,
                        owner_readiness=self._blocking_policy_gate.owner_ready,
                    )
                    await service.initialize()
                    self.evidence_service = service
                    self._blocking_policy_gate.wakeup = service.defer_policy_wakeup
                except Exception as exc:
                    # Evidence sidecar failure must not disable the existing memory path.
                    logger.warning(
                        "Evidence sidecar unavailable; cached projections are disabled (%s)",
                        type(exc).__name__,
                    )

        return True

    async def _recover_managed_reflections(self) -> None:
        """Resume due interrupted or failed windows with current scope access."""
        after_id = 0
        store = self.initializer.conversation_manager.store
        while not self._terminating:
            await asyncio.sleep(60)
            try:
                after_id = await self._recover_managed_reflection_page(store, after_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Managed summary recovery pass failed (%s)", type(exc).__name__
                )

    async def _recover_managed_reflection_page(self, store, after_id: int) -> int:
        page_size = 4
        rows = await store.managed_summary_recovery_candidates(
            after_id, time.time(), page_size
        )
        for row in rows:
            try:
                envelope = json.loads(row["envelope"] or "{}")
                scope, actor_id = envelope.get("scope"), envelope.get("actor_id")
                if (
                    not isinstance(actor_id, str)
                    or not actor_id.strip()
                    or scope_key(scope) != row["session_id"]
                ):
                    continue
                if not await self._narrative_scope_allowed(scope, actor_id):
                    continue
                await self.event_handler.reflect_recorded_scope(scope)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Managed summary recovery scope unavailable (%s)",
                    type(exc).__name__,
                )
        return rows[-1]["id"] if len(rows) == page_size else 0

    def _register_agent_tools_if_needed(self) -> None:
        """在核心组件就绪后注册 Agent 工具（回忆/写入）。

        Beta 自主多轮回忆开启时，用增强版搜索工具替代基线工具（同名
        recall_long_term_memory，注册时二选一），并额外注册深读工具；
        Beta 关闭时保持 f054c82 基线的工具集合与行为不变。开关经宿主
        重载插件后生效。
        """
        if not self.series_runtime_allowed():
            return
        if self._llm_tools_registered:
            return
        if not self.initializer.memory_engine or not self.initializer.memory_processor:
            return

        recall_enabled = bool(
            self.config_manager.get("agent_tools.enable_recall_tool", True)
        )
        beta_enabled = recall_enabled and bool(
            self.config_manager.get("agent_tools.enable_agentic_recall_beta", False)
        )

        tools = []
        if recall_enabled:
            search_tool_cls = (
                AgenticMemorySearchTool if beta_enabled else MemorySearchTool
            )
            tools.append(
                search_tool_cls(
                    context=self.context,
                    config_manager=self.config_manager,
                    memory_engine=self.initializer.memory_engine,
                    **(
                        {
                            "conversation_manager": self.initializer.conversation_manager,
                            "recall_runtime": getattr(self, "_recall_runtime", None),
                        }
                        if not beta_enabled
                        else {}
                    ),
                )
            )
        if beta_enabled:
            tools.append(
                MemoryReadTool(
                    context=self.context,
                    config_manager=self.config_manager,
                    memory_engine=self.initializer.memory_engine,
                )
            )
        if self.config_manager.get("agent_tools.enable_memorize_tool", False):
            tools.append(
                MemoryMemorizeTool(
                    context=self.context,
                    config_manager=self.config_manager,
                    memory_engine=self.initializer.memory_engine,
                    memory_processor=self.initializer.memory_processor,
                )
            )
        if self.config_manager.get("agent_tools.enable_forget_tool", False):
            tools.append(
                MemoryForgetTool(
                    context=self.context,
                    config_manager=self.config_manager,
                    memory_engine=self.initializer.memory_engine,
                )
            )

        if tools:
            for tool in tools:
                tool._series_owner_guard = self.series_runtime_allowed
            self.context.add_llm_tools(*tools)
        # 标记注册流程完成，后续不再重复检查。
        # 若用户中途修改 agent_tools 开关，需要重载插件才能生效。
        self._llm_tools_registered = True

    def _schedule_passive_group_capture(self, event: AstrMessageEvent) -> None:
        """Schedule full group capture from a filter without waking the message."""
        if not self.series_runtime_allowed():
            return
        if self._terminating or not self.initializer.is_initialized:
            return
        self._create_tracked_task(self._remember_blocking_owner(event))
        self._create_tracked_task(self._run_passive_group_capture(event))

    async def _run_passive_group_capture(self, event: AstrMessageEvent) -> None:
        if not self.series_runtime_allowed():
            return
        try:
            if not await is_session_enabled(event.unified_msg_origin):
                logger.debug(
                    f"[{event.unified_msg_origin}] 当前会话已关闭，跳过被动群聊消息捕获"
                )
                return
            if not await is_plugin_enabled_for_session(event.unified_msg_origin):
                logger.debug(
                    f"[{event.unified_msg_origin}] LivingMemory 已在当前会话禁用，"
                    "跳过被动群聊消息捕获"
                )
                return
            if not await self._ensure_runtime_components():
                logger.debug("插件组件未就绪，跳过被动群聊消息捕获")
                return
            if not self.event_handler:
                return
            await self.event_handler.handle_all_group_messages(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"被动群聊消息捕获失败: {e}", exc_info=True)

    async def _ensure_plugin_ready(self, *, wait: bool = True) -> tuple[bool, str]:
        """确保插件已完成初始化并且运行期组件可用。

        Args:
            wait: True 时等待后台初始化完成（命令/WebUI 请求）；False 时只检查
                当前状态，用于 LLM 请求钩子，避免初始化期间每个请求轮询等待。
        """
        if not self.series_runtime_allowed():
            return (
                False,
                "群间记忆插件未启用，或存在其他活动记忆提供者；调整配置后重载。",
            )
        if wait:
            if not await self.initializer.ensure_initialized():
                return False, self._get_initialization_status_message()
        elif not self.initializer.is_initialized:
            return False, self._get_initialization_status_message()

        if not await self._ensure_runtime_components():
            return (
                False,
                t("command.core_not_ready"),
            )

        return True, ""

    def _get_initialization_status_message(self) -> str:
        """获取初始化状态的用户友好消息"""
        if self.initializer.is_initialized:
            return t("init.ready")
        elif self.initializer.is_failed:
            return t(
                "init.failed",
                error=self.initializer.error_message or t("common.unknown_error"),
            )
        else:
            return t(
                "init.in_progress",
                attempts=self.initializer._provider_check_attempts,
            )

    @staticmethod
    def _command_handler_not_ready_message() -> str:
        """命令处理器未就绪时的提示"""
        return t("command.not_ready")

    async def _run_command(
        self,
        event: AstrMessageEvent,
        handler_name: str,
        *args,
        write: bool = False,
        **kwargs,
    ) -> AsyncGenerator[MessageEventResult, None]:
        """统一命令入口：就绪检查 + 委托 command_handler。

        命令函数自身的参数签名保持不变（AstrBot 依赖签名解析命令参数），
        这里只收敛重复的门禁与迭代样板。

        Args:
            event: 命令事件。
            handler_name: CommandHandler 上的处理方法名。
            *args: 传给处理方法的额外位置参数。
            write: 命令是否会写入记忆，决定适配器访问作用域的写权限。
            **kwargs: 传给处理方法的额外关键字参数。

        Yields:
            处理方法产生的消息结果；未就绪时产生提示消息。
        """
        ready, message = await self._ensure_plugin_ready()
        if not ready:
            yield event.plain_result(message)
            return
        if not self.command_handler:
            yield event.plain_result(self._command_handler_not_ready_message())
            return
        handler = getattr(self.command_handler, handler_name)
        messages = handler(event, *args, **kwargs)
        try:
            while True:
                with adapter_access(self.config_manager, event, write=write):
                    try:
                        item = await anext(messages)
                    except StopAsyncIteration:
                        return
                yield item
        finally:
            with adapter_access(self.config_manager, event, write=write):
                await messages.aclose()

    # ==================== 事件钩子 ====================

    @filter.custom_filter(PassiveGroupCaptureFilter, False)
    async def handle_all_group_messages(self, event: AstrMessageEvent):
        """[Passive Filter Hook] Capture group messages without waking AstrBot."""
        # PassiveGroupCaptureFilter schedules the capture task and always returns
        # False, so AstrBot will not invoke this handler or mark the event as wake.
        return

    @filter.on_llm_request(priority=3)  # 顺序确定化：原 0 与其余 6 个注入方并列
    @trace_recall_hook
    async def handle_memory_recall(self, event: AstrMessageEvent, req: ProviderRequest):
        """[Event Hook] Query and inject long-term memory before LLM request"""
        if not self.series_runtime_allowed():
            return
        if is_managed_event(event):
            await self._remember_blocking_owner(event)
            return
        if getattr(event, "_cd_retry_without_memory_v1", False) is True:
            # Chat Dynamics retries a reply after its native memory evidence was
            # invalidated mid-request; the retry must not re-inject that memory.
            try:
                setattr(req, "_lm_private_hook_outcome_v1", "retry_without_memory")
            except Exception:
                pass
            return
        # Keep the same total limit; reserve fresh policy and source checks
        # instead of letting an optional remote search consume that time.
        budget = RecallDeadline.start(total_seconds=6.0, final_reserve_seconds=1.5)
        self._install_private_injection_getter(event, req)
        if (
            not getattr(self.initializer, "is_initialized", False)
            or not self.event_handler
        ):
            setattr(req, "_lm_private_hook_outcome_v1", "disabled")
            return
        try:
            receipt = await timed_recall_phase(
                req,
                "projection",
                asyncio.wait_for(
                    self._prepare_native_memory_projection(event, req),
                    timeout=budget.search_remaining(),
                ),
            )
            owner_budget = min(0.5, budget.search_remaining())
            if owner_budget > 0:
                await asyncio.wait_for(
                    self._remember_blocking_owner(event, receipt), timeout=owner_budget
                )
        except asyncio.CancelledError:
            setattr(req, "_lm_private_hook_outcome_v1", "cancelled")
            raise
        except asyncio.TimeoutError:
            setattr(req, "_lm_private_hook_outcome_v1", "deadline_exceeded")
            return
        if budget.remaining() <= 0:
            try:
                setattr(req, "_lm_private_hook_outcome_v1", "deadline_exceeded")
            except Exception:
                pass
            return
        get_extra = getattr(event, "get_extra", None)
        blocking_context = (
            get_extra("_autonomous_blocking_context_v1", None)
            if callable(get_extra)
            else None
        )
        if (
            isinstance(blocking_context, dict)
            and blocking_context.get("schema_version") == 1
            and blocking_context.get("projection_restricted") is True
        ):
            # Stage canonical READY sources before the optional summary search.
            # This is read-only and does not append the request surface yet.
            try:
                await timed_recall_phase(
                    req,
                    "native_ready",
                    asyncio.wait_for(
                        self._native_memory_recall_v1(
                            event, req, collect_only=True, budget=budget
                        ),
                        timeout=budget.search_remaining(),
                    ),
                )
            except asyncio.CancelledError:
                setattr(req, "_lm_private_hook_outcome_v1", "cancelled")
                raise
            except asyncio.TimeoutError:
                self._native_clear_request(req)
                setattr(req, "_lm_private_hook_outcome_v1", "deadline_exceeded")
                return
            if budget.search_remaining() <= 0:
                self._native_clear_request(req)
                setattr(req, "_lm_private_hook_outcome_v1", "deadline_exceeded")
                return
            # Keep MemoryRecall's real event parsing, access policy and search.
            # Its evidence callback stages exact source views on this request;
            # the native surface is appended only after the fresh Blocking check.
            search_outcome = "returned"
            try:
                await timed_recall_phase(
                    req,
                    "summary_recall",
                    self.event_handler.handle_memory_recall(event, req, budget=budget),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                search_outcome = f"error:{type(exc).__name__}"
            if (
                search_outcome == "returned"
                and getattr(req, "_lm_native_memory_search_hit_v1", False) is not True
            ):
                search_outcome = "returned_no_hit_or_handler_error"
            elif search_outcome == "returned":
                search_outcome = "returned_with_hit"
            setattr(req, "_lm_native_search_outcome_v1", search_outcome)
            try:
                await timed_recall_phase(
                    req,
                    "native_surface",
                    asyncio.wait_for(
                        self._native_memory_recall_v1(event, req, budget=budget),
                        timeout=budget.remaining(),
                    ),
                )
            except asyncio.CancelledError:
                setattr(req, "_lm_private_hook_outcome_v1", "cancelled")
                raise
            except asyncio.TimeoutError:
                self._native_clear_request(req)
                setattr(req, "_lm_private_hook_outcome_v1", "deadline_exceeded")
            return
        ready, _ = await self._ensure_plugin_ready(wait=False)
        if not ready:
            logger.debug("插件未完成初始化，跳过记忆召回")
            return

        if not self.event_handler:
            return

        await timed_recall_phase(
            req,
            "summary_recall",
            self.event_handler.handle_memory_recall(event, req, budget=budget),
        )

    @filter.on_llm_response()
    async def handle_memory_reflection(
        self, event: AstrMessageEvent, resp: LLMResponse
    ):
        """[Event Hook] Check if reflection and memory storage is needed after LLM response"""
        if not self.series_runtime_allowed():
            return
        await self._remember_blocking_owner(event)
        if is_managed_event(event):
            return
        ready, _ = await self._ensure_plugin_ready(wait=False)
        if not ready:
            logger.debug("插件未完成初始化，跳过记忆反思")
            return

        if not self.event_handler:
            return

        await self.event_handler.handle_memory_reflection(event, resp)

    @filter.after_message_sent()
    async def handle_session_reset(self, event: AstrMessageEvent, *_args):
        """[Event Hook] After message sent, check if plugin session context needs clearing (/reset or /new)"""
        if not self.series_runtime_allowed():
            return
        if not (
            event.get_extra("_clean_group_context_session", False)
            or event.get_extra("_clean_ltm_session", False)
        ):
            return

        ready, _ = await self._ensure_plugin_ready(wait=False)
        if not ready:
            return

        if not self.event_handler:
            return

        await self.event_handler.handle_session_reset(event)

    # ==================== 命令处理 ====================

    @filter.command_group("lmem")
    def lmem(self):
        """Long-term memory management command group /lmem"""
        pass

    @permission_type(PermissionType.ADMIN)
    @lmem.command("status", priority=10)
    async def status(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Show memory system status"""
        async for message in self._run_command(event, "handle_status"):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("diag")
    async def diag(
        self, event: AstrMessageEvent, mode: str = ""
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Show data diagnostics; `compact` runs VACUUM"""
        async for message in self._run_command(event, "handle_diag", mode):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("search", priority=10)
    async def search(
        self, event: AstrMessageEvent, query: str, k: int = 5
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Search memories"""
        async for message in self._run_command(event, "handle_search", query, k):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("forget")
    async def forget(
        self, event: AstrMessageEvent, doc_id: int
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Delete specified memory"""
        async for message in self._run_command(event, "handle_forget", doc_id):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("rebuild-index")
    async def rebuild_index(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Manually rebuild index"""
        async for message in self._run_command(event, "handle_rebuild_index"):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("rebuild-graph")
    async def rebuild_graph(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Manually rebuild graph memory index"""
        async for message in self._run_command(event, "handle_rebuild_graph"):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("migrate-scope")
    async def migrate_scope(
        self,
        event: AstrMessageEvent,
        target: str = "",
        action: str = "",
        source: str = "",
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Migrate existing memories to the given memory scope"""
        async for message in self._run_command(
            event, "handle_migrate_scope", target, action, source
        ):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("webui")
    async def webui(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Show WebUI access information"""
        async for message in self._run_command(event, "handle_webui"):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("summarize")
    async def summarize(
        self, event: AstrMessageEvent, message_count: int | None = None
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Immediately trigger memory summarization for current session"""
        async for message in self._run_command(
            event,
            "handle_summarize",
            message_count,
            write=True,
        ):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("summarize_retry")
    async def summarize_retry(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[MessageEventResult, None]:
        """Retry the oldest refused summary for the current managed conversation."""
        if not event.is_admin() or not self.event_handler:
            return
        persona = await _get_native_persona_id(self.context, event)
        scope = self._native_scope_from_request(
            {"persona_id": persona, "instance_id": persona},
            event.unified_msg_origin,
        )
        world = find_role(self.context, "character")
        world_claimed = world is not None or any(
            bool(data.get("enabled"))
            and (data.get("capabilities") or {}).get("character") == 1
            for _, owner in active_entries(self.context)
            if (data := manifest(owner))
        )
        if world_claimed:
            effective_scope = None
            try:
                context = (
                    await asyncio.wait_for(
                        world.get_character_context(scope, event), 1.0
                    )
                    if scope and world
                    else None
                )
                if (
                    isinstance(context, dict)
                    and context.get("schema_version") == 1
                    and context.get("available", True)
                    and not context.get("disabled")
                    and find_role(self.context, "character") is world
                ):
                    effective = context.get("scope")
                    if not isinstance(effective, dict):
                        effective = {
                            **scope,
                            "persona_id": context.get("persona_id"),
                            "instance_id": context.get(
                                "instance_id", context.get("persona_id")
                            ),
                        }
                    if all(
                        effective.get(key) == scope[key] for key in ("platform", "umo")
                    ):
                        effective_scope = self._native_scope_from_request(
                            effective, event.unified_msg_origin
                        )
            except (TimeoutError, ValueError, RuntimeError):
                pass
            if effective_scope is None:
                yield event.plain_result("当前无法确认世界角色，暂未安排重试。")
                return
            scope = effective_scope
        if scope is None or not await self._narrative_scope_allowed(
            scope, event.get_sender_id()
        ):
            yield event.plain_result(t("summarize.access_denied"))
            return
        scheduled = await self.event_handler.retry_refused_scope(scope)
        yield event.plain_result(
            "已安排重试原始拒绝窗口。"
            if scheduled
            else "当前角色没有可重试的拒绝窗口，或已有反思任务在运行。"
        )

    @permission_type(PermissionType.ADMIN)
    @lmem.command("reset")
    async def reset(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Reset long-term memory context for current session"""
        async for message in self._run_command(event, "handle_reset"):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("cleanup")
    async def cleanup(
        self, event: AstrMessageEvent, mode: str = "preview"
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Clean up memory injection fragments from historical messages

        Args:
            mode: Execution mode, "preview" (default) for rehearsal, "exec" for actual cleanup
        """
        # 判断是否为执行模式
        dry_run = mode.lower() != "exec"

        async for message in self._run_command(
            event, "handle_cleanup", dry_run=dry_run
        ):
            yield message

    @permission_type(PermissionType.ADMIN)
    @lmem.command("help")
    async def help(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        """[Admin] Show help information"""
        async for message in self._run_command(event, "handle_help"):
            yield message

    # ==================== 生命周期管理 ====================

    async def terminate(self):
        panel = getattr(self, "_series_panels", None)
        if panel:
            panel.close()
        """Cleanup logic when plugin stops"""
        logger.info("LivingMemory 插件正在停止...")
        self._terminating = True
        recall_runtime = getattr(self, "_recall_runtime", None)
        if recall_runtime:
            # Drain recall work before closing the stores and engine it may use.
            await recall_runtime.close()
        if get_active_plugin() is self:
            set_active_plugin(None)

        evidence_service = getattr(self, "evidence_service", None)
        if evidence_service:
            await evidence_service.close()
            self.evidence_service = None

        # 取消所有后台任务
        if self._background_tasks:
            logger.info(f"正在取消 {len(self._background_tasks)} 个后台任务...")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # 停止初始化后台任务（如Provider重试）
        await self.initializer.stop_background_tasks()

        # 通知EventHandler停止（如果有正在运行的存储任务）
        if self.event_handler:
            await self.event_handler.shutdown()

        # 停止衰减调度器
        await self.initializer.stop_scheduler()

        # 关闭 ConversationManager
        if (
            self.initializer.conversation_manager
            and self.initializer.conversation_manager.store
        ):
            await self.initializer.conversation_manager.store.close()
            logger.info("ConversationManager 已关闭")

        # 关闭 FaissVecDB 之前，先把索引的未落盘变更写掉
        await self.initializer.shutdown_index_persisters()

        # 关闭 MemoryEngine
        if self.initializer.memory_engine:
            await self.initializer.memory_engine.close()
            logger.info("MemoryEngine 已关闭")

        # 关闭 FaissVecDB
        if self.initializer.db:
            await self.initializer.db.close()
            logger.info("FaissVecDB 已关闭")

        logger.info("LivingMemory 插件已成功停止。")

    async def get_evidence_view(
        self,
        scope: dict,
        source_refs: list[dict] | None = None,
        purpose: str = "reply",
        card_version: str = "",
        conversation_id: str | None = None,
        snapshot_fingerprint: str | None = None,
        materials: list[dict] | None = None,
        typed_items: list[dict] | None = None,
        allow_derivation: bool = True,
    ) -> dict:
        request_started = time.perf_counter()
        if self._terminating:
            return EvidenceViewService._empty_view("terminating", purpose, card_version)
        if purpose not in PURPOSES:
            return EvidenceViewService._empty_view(
                "invalid_purpose", purpose, card_version
            )
        if not await self._narrative_scope_allowed(scope):
            return EvidenceViewService._empty_view(
                "scope_not_allowed", purpose, card_version
            )
        auth_done = time.perf_counter()
        if not self.initializer.is_initialized or not self.evidence_service:
            return EvidenceViewService._empty_view(
                "service_unavailable", purpose, card_version
            )
        combined_materials = []
        combined_typed_items = [
            row for row in (typed_items or []) if isinstance(row, dict)
        ]
        for row in [*(source_refs or []), *(materials or [])]:
            if not isinstance(row, dict):
                continue
            if row.get("item_kind") == "evidence_atom":
                combined_typed_items.append(row)
            else:
                combined_materials.append(row)
        combined_materials = combined_materials[:MAX_MATERIALS]
        combined_typed_items = combined_typed_items[:MAX_MATERIALS]
        resolved_refs, rejected = await self._resolve_evidence_source_refs(
            scope, combined_materials
        )
        refs_done = time.perf_counter()
        view = await self.evidence_service.get_evidence_view(
            scope,
            source_refs=resolved_refs,
            purpose=purpose,
            card_version=card_version,
            conversation_id=conversation_id,
            snapshot_fingerprint=snapshot_fingerprint,
            typed_items=combined_typed_items,
            allow_derivation=allow_derivation,
        )
        if rejected:
            diagnostics = dict(view.get("diagnostics", {}))
            diagnostics["source_refs_rejected"] = (
                int(diagnostics.get("source_refs_rejected", 0)) + rejected
            )
            view = {**view, "diagnostics": diagnostics}
            view["evidence_status"] = EvidenceViewService._evidence_status(view)
        diagnostics = dict(view.get("diagnostics", {}))
        diagnostics["request_stage_ms"] = {
            "scope_auth": round((auth_done - request_started) * 1000, 2),
            "source_resolve": round((refs_done - auth_done) * 1000, 2),
            "cached_view": round((time.perf_counter() - refs_done) * 1000, 2),
        }
        view = {**view, "diagnostics": diagnostics}
        return view

    async def queue_material_derivation(
        self,
        materials: list[dict],
        scope: dict | None = None,
        purpose: str = "reply",
        card_version: str = "",
    ) -> dict:
        if (
            self._terminating
            or not self.evidence_service
            or not isinstance(scope, dict)
            or purpose not in PURPOSES
            or not await self._narrative_scope_allowed(scope)
        ):
            return {
                "schema_version": 1,
                "accepted": False,
                "queued": 0,
                "duplicates": 0,
                "skipped": len(materials) if isinstance(materials, list) else 0,
                "material_ids": [],
            }
        resolved, rejected = await self._resolve_evidence_source_refs(scope, materials)
        result = await self.evidence_service.queue_material_derivation(
            resolved, scope=scope, purpose=purpose, card_version=card_version
        )
        if rejected:
            result["skipped"] = int(result.get("skipped", 0)) + rejected
        return result

    def peek_evidence_revision(self, scope: dict) -> dict:
        try:
            key = scope_key(scope)
        except Exception:
            return {
                "schema_version": 1,
                "available": False,
                "scope_key": "",
                "correction_revision": None,
                "revocations": [],
            }
        if not self.evidence_service or self._terminating:
            return {
                "schema_version": 1,
                "available": False,
                "scope_key": key,
                "correction_revision": None,
                "revocations": [],
            }
        return self.evidence_service.peek_evidence_revision(scope)

    async def correct_insight(
        self,
        target_id: str,
        expected_revision: int,
        action: str,
        replacement: str = "",
        reason: str = "",
        action_id: str = "",
    ) -> dict:
        if not self.evidence_service or self._terminating:
            raise RuntimeError("evidence store unavailable")
        return await self.evidence_service.correct_insight(
            target_id, expected_revision, action, replacement, reason, action_id
        )

    async def list_evidence_insights(self, limit: int = 50) -> dict:
        if not self.evidence_service or self._terminating:
            return {"schema_version": 1, "available": False, "items": []}
        return await self.evidence_service.list_admin_insights_global(limit)

    async def plan_graph_vector_prune(self):
        """Build a read-only, exact-ID graph vector prune plan; never apply on startup."""
        if self._terminating or not self.initializer.is_initialized:
            raise RuntimeError("LivingMemory is not ready")
        engine = getattr(self.initializer, "memory_engine", None)
        planner = getattr(engine, "plan_graph_vector_prune", None) if engine else None
        if not callable(planner):
            raise RuntimeError("graph_vector_prune_unavailable")
        return await planner()

    async def apply_graph_vector_prune(self, plan):
        """Apply only an explicit plan returned by plan_graph_vector_prune."""
        if self._terminating or not self.initializer.is_initialized:
            raise RuntimeError("LivingMemory is not ready")
        if plan is None:
            raise ValueError("an explicit graph vector prune plan is required")
        engine = getattr(self.initializer, "memory_engine", None)
        apply_plan = (
            getattr(engine, "apply_graph_vector_prune", None) if engine else None
        )
        if not callable(apply_plan):
            raise RuntimeError("graph_vector_prune_unavailable")
        return await apply_plan(plan)

    async def recall_context(
        self, scope: dict, query: str, purpose: str = "reply", card_version: str = ""
    ) -> dict:
        """Deliver a local history baseline even when optional remote recall stalls."""
        loop = asyncio.get_running_loop()
        call_started = loop.time()
        budget_raw = self.config_manager.get(
            "recall_engine.narrative_total_timeout_seconds", 6.0
        )
        try:
            request_budget = max(0.1, min(float(budget_raw), 7.0))
        except (TypeError, ValueError):
            request_budget = 6.0
        deadline = call_started + request_budget
        finalize_reserve = min(1.0, max(0.02, request_budget * 0.15))
        finalize_deadline = deadline - finalize_reserve
        candidate_reserve = min(finalize_reserve * 0.5, max(0.01, request_budget * 0.1))
        retrieval_deadline = finalize_deadline - candidate_reserve
        route_failures: dict[str, str] = {}
        route_elapsed_ms: dict[str, int] = {}
        runtime = getattr(self, "_recall_runtime", None)
        if runtime is None:
            runtime = BoundedRecallTaskPool(
                max_inflight=16, max_cache=32, cache_ttl_seconds=2.0
            )
            self._recall_runtime = runtime

        def empty_reply(reason: str, status: str, available: bool = False) -> dict:
            view = EvidenceViewService._empty_view(reason, purpose, card_version)
            view["available"] = available
            view["evidence_status"] = status
            return {
                "schema_version": 1,
                "items": [],
                "available": available,
                "evidence_view": view,
                "evidence_status": status,
                "diagnostics": {
                    "timeout_count": sum(
                        value == "timeout" for value in route_failures.values()
                    ),
                    "route_error_count": sum(
                        value in {"error", "busy", "closed"}
                        for value in route_failures.values()
                    ),
                    "route_failures": dict(route_failures),
                    "route_elapsed_ms": dict(route_elapsed_ms),
                    "elapsed_ms": int((loop.time() - call_started) * 1000),
                    "inflight_count": runtime.stats().get("inflight_count", 0),
                    "source_text_logged": False,
                },
            }

        if self._terminating:
            return empty_reply("scope_not_allowed", "unavailable")
        if not self.initializer.is_initialized:
            return empty_reply("service_unavailable", "unavailable")
        auth_started = loop.time()
        auth_personas = persona_names(scope)
        revision_started = loop.time()
        auth_revision = self.peek_evidence_revision(scope)
        route_elapsed_ms["revision_peek"] = int((loop.time() - revision_started) * 1000)
        auth_key = recall_task_key(
            route="scope_auth",
            scope=scope,
            query=query,
            purpose=purpose,
            card_version=card_version,
            correction_revision=(
                auth_revision.get("correction_revision")
                if auth_revision.get("available")
                else None
            ),
            persona_ids=auth_personas,
            origin=str(scope.get("umo", "")),
        )
        auth_details = {}

        async def authorize():
            token = _SCOPE_RECALL_TIMINGS.set(auth_details)
            try:
                return await self._narrative_scope_allowed(scope)
            finally:
                _SCOPE_RECALL_TIMINGS.reset(token)

        auth_handle = runtime.submit(auth_key, authorize, cache_result=False)
        auth_outcome = await runtime.wait(auth_handle, deadline=deadline)
        route_elapsed_ms.update(auth_details)
        route_elapsed_ms["scope_auth"] = int((loop.time() - auth_started) * 1000)
        if auth_outcome.get("state") != "completed":
            route_failures["scope_auth"] = auth_outcome.get("state", "error")
            return empty_reply(
                "recall_deadline"
                if route_failures["scope_auth"] == "timeout"
                else "scope_not_allowed",
                "timeout"
                if route_failures["scope_auth"] == "timeout"
                else "unavailable",
            )
        if auth_outcome.get("value") is not True:
            return empty_reply("scope_not_allowed", "unavailable")
        top_k = max(0, min(int(self.config_manager.get("recall_engine.top_k", 5)), 20))
        if not top_k:
            return empty_reply("recall_disabled", "empty", available=True)
        if loop.time() >= deadline:
            route_failures["request"] = "timeout"
            return empty_reply("recall_deadline", "timeout")

        manager = self.initializer.conversation_manager
        engine = self.initializer.memory_engine
        personas = persona_names(scope)
        managed_origin = scope_key(scope)
        local_budget = max(
            0.1,
            min(
                float(
                    self.config_manager.get(
                        "recall_engine.narrative_local_timeout_seconds", 1.0
                    )
                ),
                5.0,
            ),
        )
        semantic_budget = max(
            0.0,
            min(
                float(
                    self.config_manager.get(
                        "recall_engine.narrative_hybrid_timeout_seconds", 0.75
                    )
                ),
                10.0,
            ),
        )
        revision_info = self.peek_evidence_revision(scope)
        correction_revision = (
            revision_info.get("correction_revision")
            if revision_info.get("available")
            else None
        )

        def task_key(route: str, origin: str, discriminator=None):
            return recall_task_key(
                route=route,
                scope=scope,
                query=query,
                purpose=purpose,
                card_version=card_version,
                correction_revision=correction_revision,
                persona_ids=personas,
                origin=origin,
                discriminator={"top_k": top_k, "route": discriminator},
            )

        async def run_route(
            method,
            *args,
            budget=None,
            route="unknown",
            origin="",
            discriminator=None,
            cache_result=False,
            use_total_deadline=False,
        ):
            if not callable(method):
                route_failures[route] = "unavailable"
                return {"state": "error", "value": None}
            started = loop.time()
            route_deadline = deadline if use_total_deadline else finalize_deadline
            if budget is not None:
                route_deadline = min(deadline, started + max(0.0, float(budget)))
            if route_deadline <= loop.time():
                route_failures[route] = "timeout"
                route_elapsed_ms.setdefault(route, 0)
                return {"state": "timeout", "value": None}
            key = task_key(route, origin, discriminator)
            handle = runtime.submit(
                key, lambda: method(*args), cache_result=cache_result
            )
            outcome = await runtime.wait(handle, deadline=route_deadline)
            route_elapsed_ms[route] = route_elapsed_ms.get(route, 0) + int(
                (loop.time() - started) * 1000
            )
            state = outcome.get("state", "error")
            if state != "completed":
                route_failures[route] = state
                if state in {"timeout", "busy", "closed", "error"}:
                    logger.debug(
                        "Narrative recall partial route=%s state=%s elapsed_ms=%d",
                        route,
                        state,
                        route_elapsed_ms[route],
                    )
            return outcome

        async def safe(
            method,
            *args,
            budget=None,
            route="unknown",
            origin=None,
            discriminator=None,
            cache_result=False,
        ):
            outcome = await run_route(
                method,
                *args,
                budget=budget,
                route=route,
                origin=(
                    str(origin)
                    if origin is not None
                    else "|".join(str(value) for value in args[2:3])
                ),
                discriminator=discriminator,
                cache_result=cache_result,
            )
            result = (
                outcome.get("value") if outcome.get("state") == "completed" else None
            )
            return result if isinstance(result, list) else []

        local = getattr(engine, "search_narrative_history", None)
        semantic = getattr(engine, "search_narrative_semantic", None)
        remote_routes = ("semantic_legacy", "semantic_managed")
        semantic_deadline = min(retrieval_deadline, loop.time() + semantic_budget)
        local_waiters = [
            asyncio.create_task(
                safe(
                    local,
                    query,
                    top_k,
                    (scope["umo"],),
                    personas,
                    budget=min(
                        local_budget, max(0.0, retrieval_deadline - loop.time())
                    ),
                    route="history_legacy",
                )
            ),
            asyncio.create_task(
                safe(
                    local,
                    query,
                    top_k,
                    (managed_origin,),
                    personas,
                    budget=min(
                        local_budget, max(0.0, retrieval_deadline - loop.time())
                    ),
                    route="history_managed",
                )
            ),
            asyncio.create_task(
                safe(
                    manager.store.recall_narrative_events,
                    scope,
                    query,
                    2,
                    budget=min(
                        local_budget, max(0.0, retrieval_deadline - loop.time())
                    ),
                    route="narrative_events",
                )
            ),
        ]
        ready_waiter = asyncio.create_task(
            run_route(
                lambda: self._ready_narrative_baseline(
                    scope, query, purpose, card_version, correction_revision
                ),
                budget=max(0.0, retrieval_deadline - loop.time()),
                route="ready_baseline",
                origin=managed_origin,
                discriminator=correction_revision,
                cache_result=False,
            )
        )
        semantic_waiters = []
        semantic_contexts = []
        if semantic_budget and str(query).strip():
            semantic_deadline = min(retrieval_deadline, loop.time() + semantic_budget)
            from .core.retrieval.vector_search import QueryEmbeddingCache

            embedding_cache = QueryEmbeddingCache(semantic_deadline)
            semantic_slots = asyncio.Semaphore(4)
            try:
                accepts_context = (
                    callable(semantic)
                    and "execution_context" in inspect.signature(semantic).parameters
                )
            except (TypeError, ValueError):
                accepts_context = False
            for index, origin in enumerate((scope["umo"], managed_origin)):
                search_context = BoundedRecallSearchContext(
                    runtime,
                    RecallDeadline(loop.time(), semantic_deadline, 0.0),
                    scope=scope,
                    query=query,
                    purpose=purpose,
                    card_version=card_version,
                    correction_revision=correction_revision,
                    persona_ids=personas,
                    origin=origin,
                    _parallel=semantic_slots,
                    hard_deadline=True,
                    embedding_cache=embedding_cache,
                )
                semantic_contexts.append(search_context)

                async def search(
                    q, count, source_origin, role_ids, context=search_context
                ):
                    if accepts_context:
                        return await semantic(
                            q, count, source_origin, role_ids, execution_context=context
                        )
                    return await semantic(q, count, source_origin, role_ids)

                semantic_waiters.append(
                    asyncio.create_task(
                        safe(
                            search,
                            query,
                            top_k,
                            origin,
                            personas,
                            budget=max(0.0, semantic_deadline - loop.time()),
                            route=remote_routes[index],
                            origin=origin,
                            cache_result=True,
                        )
                    )
                )
        try:
            historical, managed, evidence, ready_outcome = await asyncio.gather(
                *local_waiters, ready_waiter
            )
            ready_baseline = (
                ready_outcome.get("value")
                if ready_outcome.get("state") == "completed"
                and isinstance(ready_outcome.get("value"), dict)
                else {"source_refs": [], "typed_items": [], "diagnostic": "partial"}
            )
            local_reply_fast_path = (
                purpose == "reply"
                and not _HISTORY_LOOKUP.search(str(query))
                and ready_baseline.get("query_matched") is True
                and bool(ready_baseline.get("evidence_view"))
                and any(not waiter.done() for waiter in semantic_waiters)
            )
            if local_reply_fast_path:
                # The common ready-view path rechecks scope, permissions and revision
                # after the waiters have drained and before publishing the local result.
                historical, managed, evidence = [], [], []
            elif semantic_waiters:
                semantic_values = await asyncio.gather(*semantic_waiters)
                for index, context in enumerate(semantic_contexts):
                    prefix = remote_routes[index] + "_"
                    route_elapsed_ms.update(
                        {
                            prefix + key: value
                            for key, value in context.route_elapsed_ms.items()
                        }
                    )
                    route_failures.update(
                        {
                            prefix + key: value
                            for key, value in context.route_failures.items()
                        }
                    )
                for index, baseline in enumerate((historical, managed)):
                    enriched = (
                        semantic_values[index] if index < len(semantic_values) else []
                    )
                    baseline_origins = (
                        (scope["umo"],) if index == 0 else (managed_origin,)
                    )
                    revalidated = await self._revalidate_narrative_candidates(
                        [*enriched, *baseline],
                        baseline_origins,
                        personas,
                        run_route,
                    )
                    baseline[:] = revalidated[:top_k]
            else:
                historical, managed = await asyncio.gather(
                    self._revalidate_narrative_candidates(
                        historical, (scope["umo"],), personas, run_route
                    ),
                    self._revalidate_narrative_candidates(
                        managed, (managed_origin,), personas, run_route
                    ),
                )
        finally:
            # These are only bounded waiters. The owned pool keeps any late
            # provider/search task and never lets it mutate this request view.
            for waiter in [*local_waiters, ready_waiter, *semantic_waiters]:
                if not waiter.done():
                    waiter.cancel()
            pending_waiters = [
                waiter
                for waiter in [*local_waiters, ready_waiter, *semantic_waiters]
                if not waiter.done()
            ]
            if pending_waiters:
                await asyncio.gather(*pending_waiters, return_exceptions=True)
        items = []
        for rows, legacy in ((historical, True), (managed, False)):
            for memory in rows:
                metadata = getattr(memory, "metadata", {}) or {}
                unknown_role = metadata.get("persona_id") in (
                    None,
                    "",
                    "default",
                    "[%None]",
                )
                prefix = (
                    "当时角色不明的历史群聊材料（不代表当前角色承诺或人生事实）：\n"
                    if legacy and unknown_role
                    else ""
                )
                items.append(
                    {
                        "text": prefix + memory.content,
                        "source_id": str(memory.doc_id),
                        "occurred_at": metadata.get("create_time"),
                        "scope": dict(scope),
                        "fact_kind": "subjective",
                        "unknown": True,
                        "provenance_kind": "historical_unknown_character"
                        if legacy and unknown_role
                        else "historical"
                        if legacy
                        else "managed_summary",
                        "source_persona_id": metadata.get("persona_id"),
                        "source_umo": source_origin(metadata),
                        "evidence": metadata.get("narrative_provenance", []),
                        "_source_metadata": metadata,
                    }
                )
        routes = {
            "historical": len(historical),
            "managed": len(managed),
            "evidence": len(evidence),
        }
        if purpose == "history_quote":
            # Explicit history query is the only route that returns legacy summary text.
            view = EvidenceViewService._empty_view(
                "history_quote_uses_source_route", purpose, card_version
            )
            view["available"] = True
            view["evidence_status"] = "empty"
            return {
                "schema_version": 1,
                "items": [*items, *evidence][: top_k * 2],
                "available": True,
                "routes": routes,
                "evidence_view": view,
                "evidence_status": "empty",
                "diagnostics": {
                    **self._narrative_runtime_diagnostics(
                        call_started, route_failures, route_elapsed_ms, runtime
                    ),
                    "ready_baseline_status": str(
                        ready_baseline.get("diagnostic", "partial")
                    )[:64],
                    "timeout_count": sum(
                        value == "timeout" for value in route_failures.values()
                    ),
                    "route_error_count": sum(
                        value in {"error", "unavailable", "busy", "closed"}
                        for value in route_failures.values()
                    ),
                },
            }
        if purpose not in PURPOSES:
            unavailable = empty_reply("invalid_purpose", "invalid")
            unavailable["routes"] = routes
            unavailable["diagnostics"] = {
                "reason": "invalid_purpose",
                "raw_legacy_summaries_returned": False,
                **self._narrative_runtime_diagnostics(
                    call_started, route_failures, route_elapsed_ms, runtime
                ),
            }
            return unavailable
        if not self.evidence_service:
            unavailable = empty_reply("service_unavailable", "unavailable")
            unavailable["routes"] = routes
            unavailable["diagnostics"] = {
                "reason": "evidence_view_unavailable",
                "raw_legacy_summaries_returned": False,
                **self._narrative_runtime_diagnostics(
                    call_started, route_failures, route_elapsed_ms, runtime
                ),
            }
            return unavailable

        materials = []
        source_plans = []
        for item in items:
            provenance = item.get("evidence") or []
            if isinstance(provenance, list) and provenance:
                metadata = item.pop("_source_metadata", {})
                source_plans.append((item, metadata))
            else:
                item.pop("_source_metadata", None)
                materials.append(
                    {
                        "source_kind": "legacy_summary",
                        "document_id": item.get("source_id"),
                        "source_umo": item.get("source_umo"),
                        "text": item.get("text", ""),
                    }
                )

        source_semaphore = asyncio.Semaphore(4)

        async def load_source_plan(item, metadata):
            async with source_semaphore:
                origin = str(
                    metadata.get("source_session_id")
                    or metadata.get("session_id")
                    or ""
                )
                doc_id = item.get("source_id", "")
                return await run_route(
                    manager.store.load_narrative_sources,
                    metadata,
                    budget=max(0.0, finalize_deadline - loop.time()),
                    route="narrative_sources",
                    origin=origin,
                    discriminator=doc_id,
                    cache_result=False,
                )

        source_outcomes = []
        batch_sources = getattr(manager.store, "load_narrative_sources_many", None)
        if source_plans and loop.time() < deadline and callable(batch_sources):
            batch_outcome = await run_route(
                batch_sources,
                [metadata for _, metadata in source_plans],
                budget=max(0.0, finalize_deadline - loop.time()),
                route="narrative_sources",
                origin=managed_origin,
                discriminator=[item.get("source_id") for item, _ in source_plans],
            )
            values = batch_outcome.get("value")
            if (
                batch_outcome.get("state") == "completed"
                and isinstance(values, list)
                and len(values) == len(source_plans)
            ):
                source_outcomes = [
                    {"state": "completed", "value": value} for value in values
                ]
            else:
                if batch_outcome.get("state") == "completed":
                    batch_outcome = {"state": "error", "value": None}
                    route_failures["narrative_sources"] = "error"
                source_outcomes = [
                    {**batch_outcome, "value": None} for _ in source_plans
                ]
        elif source_plans and loop.time() < deadline:
            source_outcomes = await asyncio.gather(
                *(load_source_plan(item, metadata) for item, metadata in source_plans)
            )
        for (item, _metadata), outcome in zip(source_plans, source_outcomes):
            if outcome.get("state") != "completed":
                # A summary without completed source proof cannot substitute for
                # a timed-out source read.
                continue
            source_rows = outcome.get("value")
            source_rows = source_rows if isinstance(source_rows, list) else []
            for source in source_rows[:32]:
                metadata = source.get("metadata", {})
                fact_kind = metadata.get("fact_kind")
                source_scope = metadata.get("scope")
                event_id = metadata.get("source_event_id")
                if not isinstance(source_scope, dict) or not event_id:
                    continue
                materials.append(
                    {
                        "source_kind": "event",
                        "event_id": event_id,
                        "event_type": "delivery.success"
                        if fact_kind == "receipt"
                        else "message.received",
                        "scope": source_scope,
                        "text": source.get("content", ""),
                        "actor_id": source.get("sender_id"),
                        "occurred_at": source.get("timestamp"),
                    }
                )
            if source_rows:
                continue
            item.pop("_source_metadata", None)
            materials.append(
                {
                    "source_kind": "legacy_summary",
                    "document_id": item.get("source_id"),
                    "source_umo": item.get("source_umo"),
                    "text": item.get("text", ""),
                }
            )
        for source in evidence:
            if not isinstance(source, dict):
                continue
            fact_kind = source.get("fact_kind")
            text = str(source.get("text", ""))
            if text.startswith("用户发言："):
                text = text[len("用户发言：") :]
            elif text.startswith("已发送："):
                text = text[len("已发送：") :]
            materials.append(
                {
                    "source_kind": "event",
                    "event_id": source.get("source_id"),
                    "event_type": "delivery.success"
                    if fact_kind == "receipt"
                    else "message.received",
                    "scope": source.get("scope"),
                    "text": text,
                    "occurred_at": source.get("occurred_at"),
                }
            )
        ready_refs = (
            ready_baseline.get("source_refs", [])
            if isinstance(ready_baseline, dict)
            else []
        )
        ready_typed = (
            ready_baseline.get("typed_items", [])
            if isinstance(ready_baseline, dict)
            else []
        )
        # Keep proof for already verified atoms before optional summary expansion.
        # Deduplicate complete identities, not text (same text can have different authors).
        view_refs, seen_refs = [], set()
        for ref in [*ready_refs, *materials]:
            eid = ref.get("source_event_id", ref.get("event_id"))
            identity = (
                ("event", str(eid))
                if eid
                else ("material", json.dumps(ref, sort_keys=True, ensure_ascii=False))
            )
            if identity not in seen_refs:
                seen_refs.add(identity)
                view_refs.append(ref)
        if not materials and ready_baseline.get("evidence_view"):
            # All useful material is already rendered. Recheck identity/policy
            # instead of sending the same ready atoms through SQLite again.
            view_outcome = await run_route(
                self._current_ready_narrative_view,
                scope,
                ready_baseline,
                purpose,
                card_version,
                route="baseline_recheck",
                origin=managed_origin,
                use_total_deadline=True,
            )
        else:
            view_outcome = await run_route(
                lambda: self.get_evidence_view(
                    scope,
                    source_refs=view_refs[:MAX_MATERIALS],
                    typed_items=ready_typed[:MAX_MATERIALS],
                    purpose=purpose,
                    card_version=card_version,
                    allow_derivation=bool(materials),
                ),
                route="evidence_view",
                origin=scope.get("umo", ""),
                discriminator=revision_info.get("correction_revision"),
                cache_result=False,
                # Keep the final reserve for checking/publishing a valid local
                # baseline. Optional enrichment cannot consume that reserve.
                use_total_deadline=False,
            )
        baseline_preserved = False
        if (
            view_outcome.get("state") != "completed"
            or not isinstance(view_outcome.get("value"), dict)
            or view_outcome["value"].get("available") is not True
            or not view_outcome["value"].get("atoms")
        ):
            baseline_outcome = (
                await run_route(
                    self._current_ready_narrative_view,
                    scope,
                    ready_baseline,
                    purpose,
                    card_version,
                    route="baseline_recheck",
                    origin=managed_origin,
                    use_total_deadline=True,
                )
                if ready_baseline.get("evidence_view")
                else {}
            )
            if baseline_outcome.get("state") == "completed" and isinstance(
                baseline_outcome.get("value"), dict
            ):
                view_outcome = baseline_outcome
                baseline_preserved = True
        if (
            view_outcome.get("state") != "completed"
            or not isinstance(view_outcome.get("value"), dict)
            or view_outcome["value"].get("available") is not True
        ):
            value = view_outcome.get("value")
            reported_status = (
                value.get("evidence_status") if isinstance(value, dict) else None
            )
            status = (
                "timeout"
                if view_outcome.get("state") == "timeout"
                else reported_status
                if reported_status
                in {"invalid", "unavailable", "pending", "deferred", "timeout"}
                else "unavailable"
            )
            unavailable = empty_reply(
                "recall_deadline"
                if status == "timeout"
                else "evidence_view_unavailable",
                status,
            )
            unavailable["routes"] = routes
            return unavailable
        view = copy.deepcopy(view_outcome["value"])
        if route_failures:
            diagnostics = dict(view.get("diagnostics", {}))
            diagnostics["timeout_count"] = sum(
                value == "timeout" for value in route_failures.values()
            )
            diagnostics["route_error_count"] = sum(
                value in {"error", "unavailable", "busy", "closed"}
                for value in route_failures.values()
            )
            diagnostics["timed_out_routes"] = sorted(
                key for key, value in route_failures.items() if value == "timeout"
            )
            view = {**view, "diagnostics": diagnostics}
            view["evidence_status"] = EvidenceViewService._evidence_status(view)
        if baseline_preserved or (route_failures and view.get("atoms")):
            view["evidence_status"] = "partial"
        safe_items = []
        for slot_name in (
            "current_evidence",
            "commitments_and_boundaries",
            "recent",
            "historical_claims",
        ):
            for atom in view.get("slots", {}).get(slot_name, []):
                safe_items.append(
                    {
                        "item_kind": "evidence_atom",
                        "evidence_schema_version": 1,
                        "text": atom.get("claim", ""),
                        "source_id": atom.get("atom_id"),
                        "scope": dict(scope),
                        "fact_kind": atom.get("kind"),
                        "provenance_kind": atom.get("source_status"),
                        "status": atom.get("status"),
                        "scope_key": view.get("scope_key"),
                        "correction_revision": view.get("correction_revision"),
                        "source_refs": atom.get("source_refs", []),
                        "root_evidence_ids": atom.get("root_evidence_ids", []),
                        "root_material_ids": atom.get("root_material_ids", []),
                    }
                )
        return {
            "schema_version": 1,
            "items": safe_items[: top_k * 4],
            "available": bool(view.get("available")),
            "routes": routes,
            "evidence_view": view,
            "evidence_status": view.get("evidence_status", "unavailable"),
            "diagnostics": {
                "accepted": int(
                    view.get("diagnostics", {}).get("typed_items_accepted", 0)
                ),
                "resolved": int(view.get("diagnostics", {}).get("materials_cached", 0)),
                "deferred": int(
                    view.get("diagnostics", {}).get("materials_deferred", 0)
                ),
                "rejected": int(
                    view.get("diagnostics", {}).get("source_refs_rejected", 0)
                )
                + int(view.get("diagnostics", {}).get("typed_items_rejected", 0)),
                "timeout": int(view.get("diagnostics", {}).get("timeout_count", 0)),
                "rerank": self._effective_rerank_status(),
                "raw_legacy_summaries_returned": False,
                "source_text_logged": False,
                "ready_baseline_status": str(
                    ready_baseline.get("diagnostic", "partial")
                )[:64],
                "baseline_preserved": baseline_preserved,
                "local_reply_fast_path": local_reply_fast_path,
                **self._narrative_runtime_diagnostics(
                    call_started, route_failures, route_elapsed_ms, runtime
                ),
            },
        }

    async def ingest_events(self, events: list[dict]) -> dict:
        """Persist received messages and confirmed segments; never model drafts."""
        if not self.series_runtime_allowed():
            raise RuntimeError("memory_owner_unavailable")
        if self._terminating or not self.initializer.is_initialized:
            raise RuntimeError("LivingMemory is not ready")
        manager = self.initializer.conversation_manager
        accepted = 0
        for event in events:
            if event.get("type") not in {"message.received", "delivery.success"}:
                continue
            scope = event.get("scope", {})
            if not await self._narrative_scope_allowed(
                scope, str(event.get("actor_id", ""))
            ):
                continue
            if await manager.store.ingest_narrative_event(event):
                accepted += 1
                await manager.invalidate_cache(scope_key(scope))
                if event["type"] == "delivery.success" and self.event_handler:
                    await self.event_handler.reflect_recorded_scope(scope)
        return {"schema_version": 1, "accepted": accepted}

    @permission_type(PermissionType.ADMIN)
    @lmem.command("evidence")
    async def evidence_admin(self, event: AstrMessageEvent, args: str = ""):
        """Append an admin correction without rewriting source messages or affinity."""
        is_admin = getattr(event, "is_admin", None)
        if not callable(is_admin) or not is_admin():
            yield event.plain_result("此操作仅管理员可用。")
            return
        if args.strip().lower() == "list":
            result = await self.list_evidence_insights(40)
            rows = result.get("items", [])
            if not rows:
                yield event.plain_result("当前没有可纠正的派生见解。")
                return
            lines = ["派生见解（仅显示ID与来源元数据，不显示群友原文）："]
            lines.extend(
                f"{row['atom_id']} | {row['kind']} | {row['status']} | correction rev {row['correction_revision']}"
                for row in rows
            )
            yield event.plain_result("\n".join(lines))
            return
        parts = [part.strip() for part in args.split("|", 4)]
        if len(parts) not in {4, 5}:
            yield event.plain_result(
                "用法：/lmem evidence list；或 /lmem evidence <ID>|<revision>|withdraw|<原因>；"
                "supersede格式：<ID>|<revision>|supersede|<替代见解>|<原因>"
            )
            return
        target_id = parts[0]
        try:
            expected_revision = int(parts[1])
        except (TypeError, ValueError):
            yield event.plain_result("revision必须是整数。")
            return
        action = parts[2].lower()
        replacement = parts[3] if action == "supersede" and len(parts) == 5 else ""
        reason = parts[4] if action == "supersede" and len(parts) == 5 else parts[3]
        try:
            result = await self.correct_insight(
                target_id,
                expected_revision,
                action,
                replacement=replacement,
                reason=reason,
            )
        except (ValueError, RuntimeError):
            yield event.plain_result(
                "纠正未写入：目标不存在、revision已变化或参数无效。请重新查看ID和revision。"
            )
            return
        yield event.plain_result(
            f"纠正已追加：{result['action']} {result['target_id']}；"
            f"revision={result['correction_revision']}；依赖项已失效。"
        )
