"""
记忆重要性衰减调度器
每日自动对记忆重要性进行衰减处理，并定期备份数据库
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from astrbot.api import logger

from ...storage.conversation_store import ConversationStore
from ...storage.evidence_sidecar import EvidenceSidecar
from ..managers.backup_manager import BackupManager

if TYPE_CHECKING:
    from ...storage.db_migration import DBMigration
    from ..managers.memory_engine import MemoryEngine


class DecayScheduler:
    """
    记忆重要性衰减调度器

    功能：
    1. 每日凌晨自动执行衰减
    2. 启动时检查并补偿错过的衰减
    3. 防止同一天重复执行
    4. 定期自动备份数据库
    """

    def __init__(
        self,
        memory_engine: "MemoryEngine",
        decay_rate: float,
        data_dir: str,
        check_hour: int = 0,
        check_minute: int = 5,
        db_migration: "DBMigration | None" = None,
        backup_enabled: bool = True,
        backup_keep_days: int = 7,
        consolidation_manager=None,
        evidence_sidecar=None,
        evidence_retention_days: int = 30,
    ):
        """
        初始化衰减调度器

        Args:
            memory_engine: 记忆引擎实例
            decay_rate: 每日衰减率 (0-1)
            data_dir: 数据目录，用于存储状态文件
            check_hour: 每日执行时间（小时）
            check_minute: 每日执行时间（分钟）
            db_migration: 数据库迁移管理器（用于备份）
            backup_enabled: 是否启用每日自动备份
            backup_keep_days: 备份保留天数，超期自动删除
            consolidation_manager: 记忆整合管理器（用于每日整合触发）
            evidence_sidecar: 可选 EvidenceSidecar 实例（用于每日过期清理与 WAL 维护）
            evidence_retention_days: sidecar 过期失效数据保留天数
        """
        self.memory_engine = memory_engine
        self.decay_rate = decay_rate
        self.data_dir = Path(data_dir)
        self.check_hour = check_hour
        self.check_minute = check_minute
        self.db_migration = db_migration
        self.backup_enabled = backup_enabled
        self.backup_keep_days = backup_keep_days
        self.consolidation_manager = consolidation_manager
        self.evidence_sidecar = evidence_sidecar
        self.evidence_retention_days = max(1, int(evidence_retention_days))

        self._state_file = self.data_dir / "decay_state.json"
        self._task: asyncio.Task | None = None
        self._running = False

    async def _load_state(self) -> dict:
        """加载状态文件"""
        if not self._state_file.exists():
            return {}
        try:
            try:
                import aiofiles
            except ImportError:
                content = await asyncio.to_thread(
                    self._state_file.read_text,
                    encoding="utf-8",
                )
            else:
                async with aiofiles.open(self._state_file, encoding="utf-8") as f:
                    content = await f.read()
            return json.loads(content)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[衰减调度] 加载状态文件失败: {e}")
            return {}

    async def _save_state(self, state: dict) -> None:
        """保存状态文件"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            content = json.dumps(state, ensure_ascii=False)
            try:
                import aiofiles
            except ImportError:
                await asyncio.to_thread(
                    self._state_file.write_text,
                    content,
                    encoding="utf-8",
                )
            else:
                async with aiofiles.open(self._state_file, "w", encoding="utf-8") as f:
                    await f.write(content)
        except OSError as e:
            logger.error(f"[衰减调度] 保存状态文件失败: {e}")

    async def _get_last_decay_date(self) -> str | None:
        """获取上次衰减日期 (格式: YYYY-MM-DD)"""
        state = await self._load_state()
        return state.get("last_decay_date")

    async def _set_last_decay_date(self, date_str: str) -> None:
        """设置上次衰减日期"""
        state = await self._load_state()
        state["last_decay_date"] = date_str
        state["last_decay_timestamp"] = time.time()
        await self._save_state(state)

    def _get_today_str(self) -> str:
        """获取今天日期字符串"""
        return datetime.now().strftime("%Y-%m-%d")

    async def _calculate_missed_days(self) -> int:
        """计算错过的衰减天数"""
        last_date_str = await self._get_last_decay_date()
        if not last_date_str:
            return 0

        try:
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            today = datetime.now().date()
            delta = (today - last_date).days
            return max(0, delta - 1)
        except ValueError:
            return 0

    async def _execute_decay(self, days: int = 1) -> bool:
        """
        执行衰减操作

        Args:
            days: 衰减天数（用于补偿错过的天数）

        Returns:
            是否执行成功
        """
        try:
            if self.decay_rate > 0:
                affected = await self.memory_engine.apply_daily_decay(
                    self.decay_rate, days
                )
                logger.info(
                    f"[衰减调度] 衰减完成，影响 {affected} 条记忆，衰减天数: {days}"
                )
            else:
                logger.info("[衰减调度] 衰减率为0，跳过衰减")

            # 每日衰减后可选执行一次旧记忆清理
            if self.memory_engine.config.get("auto_cleanup_enabled", True):
                try:
                    cleanup_days = self.memory_engine.config.get(
                        "cleanup_days_threshold", 30
                    )
                    cleanup_importance = self.memory_engine.config.get(
                        "cleanup_importance_threshold", 0.3
                    )
                    deleted = await self.memory_engine.cleanup_old_memories(
                        days_threshold=cleanup_days,
                        importance_threshold=cleanup_importance,
                    )
                    action = (
                        "归档"
                        if self.memory_engine.config.get("auto_archived_enabled", False)
                        else "删除"
                    )
                    logger.info(f"[衰减调度] 自动清理完成，{action} {deleted} 条旧记忆")
                except Exception as cleanup_err:
                    logger.error(
                        f"[衰减调度] 自动清理失败: {cleanup_err}", exc_info=True
                    )

            # 每日清理超期归档文档及其图谱/原子关联行
            purge_archived = getattr(
                self.memory_engine, "purge_expired_archived_memories", None
            )
            if callable(purge_archived):
                try:
                    purged = await purge_archived()
                    if purged:
                        logger.info(
                            f"[衰减调度] 超期归档记忆清除完成，彻底删除 {purged} 条"
                        )
                except Exception as purge_err:
                    logger.warning(
                        f"[衰减调度] 超期归档记忆清除失败: {purge_err}",
                        exc_info=True,
                    )

            await self._set_last_decay_date(self._get_today_str())

            # 每日执行记忆整合
            if self.consolidation_manager is not None:
                try:
                    result = await self.consolidation_manager.maybe_run("daily")
                    if not result.get("skipped"):
                        logger.info(
                            f"[衰减调度] 记忆整合完成: 候选={result.get('candidates', 0)}, "
                            f"组数={result.get('groups', 0)}, "
                            f"合并={result.get('merged', 0)}"
                        )
                except Exception as consolidation_err:
                    logger.error(
                        f"[衰减调度] 记忆整合失败: {consolidation_err}", exc_info=True
                    )

            # 每日执行备份
            if self.backup_enabled and self.db_migration:
                await self._run_backup()

            maintained_companions: set[str] = set()
            try:
                maintenance_result = await self.memory_engine.maintain_storage()
                if isinstance(maintenance_result, dict) and maintenance_result.get(
                    "success"
                ):
                    reclaimed = int(maintenance_result.get("bytes_reclaimed", 0))
                    companion_map = maintenance_result.get("companion_dbs")
                    if isinstance(companion_map, dict):
                        maintained_companions.update(
                            name
                            for name, detail in companion_map.items()
                            if isinstance(detail, dict) and detail.get("success")
                        )
                    logger.info(
                        f"[衰减调度] 存储维护完成，释放 {reclaimed / 1024 / 1024:.2f} MB"
                    )
                else:
                    err = (
                        maintenance_result.get("error")
                        if isinstance(maintenance_result, dict)
                        else "unknown"
                    )
                    logger.warning(f"[衰减调度] 存储维护失败: {err}")
            except Exception as maintenance_err:
                logger.warning(
                    f"[衰减调度] 存储维护异常: {maintenance_err}",
                    exc_info=True,
                )

            await self._maintain_companion_databases(
                already_checkpointed=maintained_companions
            )

            return True
        except Exception as e:
            logger.error(f"[衰减调度] 执行衰减失败: {e}", exc_info=True)
            return False

    @staticmethod
    def _checkpoint_sqlite_file_sync(db_file: Path) -> bool:
        """对单个 SQLite 库执行安全的 WAL 截断与增量回收（不影响现有数据）。"""
        if not db_file.is_file():
            return False
        conn = sqlite3.connect(str(db_file), timeout=5.0, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            conn.execute("PRAGMA incremental_vacuum").fetchall()
            return True
        finally:
            conn.close()

    async def _maintain_companion_databases(
        self, *, already_checkpointed: set[str] | None = None
    ) -> dict[str, dict]:
        """每日伴生库安全维护：清理 evidence_sidecar 过期数据并对所有伴生库执行 WAL 截断。"""
        done_set = set(already_checkpointed or ())
        report: dict[str, dict] = {}

        # 1. evidence_sidecar.sqlite3 清理过期失效记录并执行 WAL 截断
        sidecar_path = self.data_dir / "evidence_sidecar.sqlite3"
        try:
            if self.evidence_sidecar is not None:
                prune_stats = await self.evidence_sidecar.prune_old(
                    retention_days=self.evidence_retention_days
                )
                report["evidence_sidecar.sqlite3"] = {
                    "success": True,
                    "pruned": prune_stats,
                }
                done_set.add("evidence_sidecar.sqlite3")
            elif sidecar_path.is_file():
                prune_stats = await asyncio.to_thread(
                    EvidenceSidecar.prune_file_sync,
                    sidecar_path,
                    self.evidence_retention_days,
                )
                report["evidence_sidecar.sqlite3"] = {
                    "success": True,
                    "pruned": prune_stats,
                }
                done_set.add("evidence_sidecar.sqlite3")
        except Exception as exc:
            logger.warning(f"[衰减调度] evidence_sidecar 维护失败: {exc}")
            report["evidence_sidecar.sqlite3"] = {"success": False, "error": str(exc)}

        # 2. conversations.db 清理过期/超量 narrative_events
        conv_path = self.data_dir / "conversations.db"
        if conv_path.is_file():
            try:
                narrative_pruned = await asyncio.to_thread(
                    ConversationStore.prune_narrative_events_file_sync,
                    conv_path,
                    retention_days=self.evidence_retention_days,
                )
                report["conversations.db"] = {
                    "success": True,
                    "narrative_events_pruned": narrative_pruned,
                }
            except Exception as exc:
                logger.warning(f"[衰减调度] conversations.db narrative_events 维护失败: {exc}")

        # 3. 对 conversations.db / livingmemory_graph_documents.db / evidence_sidecar.sqlite3 兜底 WAL 截断
        for db_name in (
            "conversations.db",
            "livingmemory_graph_documents.db",
            "evidence_sidecar.sqlite3",
        ):
            if db_name in done_set:
                continue
            db_file = self.data_dir / db_name
            if not db_file.is_file():
                continue
            try:
                ok = await asyncio.to_thread(self._checkpoint_sqlite_file_sync, db_file)
                prev = report.get(db_name, {})
                report[db_name] = {**prev, "success": ok}
            except Exception as exc:
                logger.warning(f"[衰减调度] 伴生库 {db_name} WAL 维护失败: {exc}")
                report[db_name] = {"success": False, "error": str(exc)}

        return report

    async def _check_and_execute(self) -> None:
        """检查并执行衰减（启动时调用）"""
        today_str = self._get_today_str()
        last_date_str = await self._get_last_decay_date()

        if last_date_str == today_str:
            logger.debug("[衰减调度] 今日已执行过衰减，跳过")
            return

        missed_days = await self._calculate_missed_days()
        total_days = missed_days + 1

        if missed_days > 0:
            logger.info(f"[衰减调度] 检测到错过 {missed_days} 天衰减，执行补偿")

        await self._execute_decay(total_days)

    async def _run_backup(self) -> None:
        """执行数据库备份并清理过期备份（含总量上限裁剪与临时文件清理）"""
        if not self.db_migration:
            return
        try:
            backup_path = await self.db_migration.create_backup()
            if backup_path:
                logger.info(f"[衰减调度] 每日备份完成: {backup_path}")
                await self._cleanup_old_backups()
            else:
                logger.warning("[衰减调度] 每日备份失败")
            try:
                await BackupManager(str(self.data_dir)).prune_async(
                    keep_days=self.backup_keep_days
                )
            except Exception as prune_err:
                logger.warning(f"[衰减调度] BackupManager.prune 失败: {prune_err}")
        except Exception as e:
            logger.error(f"[衰减调度] 备份异常: {e}", exc_info=True)

    async def _cleanup_old_backups(self) -> None:
        """删除超过保留天数的旧备份文件（兼容 .db 与 .db.gz）及遗留临时文件"""
        if not self.db_migration:
            return
        try:
            db_path = Path(self.db_migration.db_path)
            backup_dir = db_path.parent / "backups"
            if not backup_dir.exists():
                return

            cutoff = datetime.now().timestamp() - self.backup_keep_days * 86400
            removed = 0
            for pattern in (
                f"{db_path.stem}_backup_*.db",
                f"{db_path.stem}_backup_*.db.gz",
            ):
                for f in backup_dir.glob(pattern):
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1

            if removed:
                logger.info(
                    f"[衰减调度] 清理过期备份 {removed} 个（保留 {self.backup_keep_days} 天）"
                )
        except Exception as e:
            logger.warning(f"[衰减调度] 清理旧备份失败: {e}")

    def _seconds_until_next_run(self) -> float:
        """计算距离下次执行的秒数"""
        now = datetime.now()
        target = now.replace(
            hour=self.check_hour,
            minute=self.check_minute,
            second=0,
            microsecond=0,
        )

        if now >= target:
            target += timedelta(days=1)

        return (target - now).total_seconds()

    async def _scheduler_loop(self) -> None:
        """调度器主循环"""
        # 启动时异步让出控制权再执行首次检查，避免阻塞插件初始化流程
        try:
            await asyncio.sleep(0)
            if self._running:
                await self._check_and_execute()
        except asyncio.CancelledError:
            logger.info("[衰减调度] 调度器被取消")
            return
        except Exception as e:
            logger.error(f"[衰减调度] 首次检查异常: {e}", exc_info=True)

        while self._running:
            try:
                wait_seconds = self._seconds_until_next_run()
                logger.debug(f"[衰减调度] 下次执行在 {wait_seconds / 3600:.1f} 小时后")

                await asyncio.sleep(wait_seconds)

                if not self._running:
                    break

                await self._execute_decay(1)

            except asyncio.CancelledError:
                logger.info("[衰减调度] 调度器被取消")
                break
            except Exception as e:
                logger.error(f"[衰减调度] 循环异常: {e}", exc_info=True)
                await asyncio.sleep(3600)

    async def start(self) -> None:
        """启动调度器"""
        if self._running:
            logger.warning("[衰减调度] 调度器已在运行")
            return

        self._running = True

        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            f"[衰减调度] 调度器已启动 (衰减率: {self.decay_rate}, "
            f"执行时间: {self.check_hour:02d}:{self.check_minute:02d})"
        )

    async def stop(self) -> None:
        """停止调度器"""
        self._running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._task = None
        logger.info("[衰减调度] 调度器已停止")
