# astrbot_plugin_dynamics_memory 0.2.0-rc.9

## [0.2.0-rc.9] - 2026-11-25

- 安全加固（Batch 1）：修复跨群隔离边界、消除仪表盘 XSS 风险并接入全系列统一管理鉴权。
- 数据与存储修复（Batch 2）：保护共享知识图谱边数据、统一 SQLite 写锁并遏制数据库膨胀。
- 性能与检索优化（Batch 3）：为受管域接入熔断器、修复 MMR 与相关性打分及仪表盘缺陷，完善配置热重载。
- 工程与发布治理（Batch 4）：改为从 `metadata.yaml` 动态读取版本号，加固发布打包与 CI 工作流并同步系列共享资产。

## [0.2.0-rc.8] - 2026-11-24

- 全面落地存储生命周期治理：全部 SQLite 启用 `PRAGMA auto_vacuum=INCREMENTAL` + 定期增量收缩与 WAL 截断；每日备份改为 `.db.gz` 压缩并调用 `BackupManager.prune()`；`evidence_sidecar` 按 `(scope_key, conversation_id, text_hash)` 去重并对滞留队列/材料按 TTL 与单群上限裁剪；`conversations.db` 强制消息上限与 `narrative_events` TTL 裁剪；`livingmemory.db` 定期剥离老旧 `memory_write_ops` payload 与清除过期归档。
- 修复反思质检对「有人」习语误杀并增加指数退避与游标脱死锁机制；前置便宜的权限/命令/`@他人` 过滤以减少无效 Projection 与 Embedding。
- `GraphStore` 升级 4 连接 WAL 并发读池、索引前缀/FTS5 三元组与 `1.5s` 超时优雅降级；同步最新 `series_panels.py`。

## [0.2.0-rc.7] - 2026-11-22

- 接入全系列共享 `ProviderBudget` 并发池与遥测统计（覆盖 `memory_processor` 非流式/流式调用及 `plugin_helpers_mixin` 证据提炼），遇队列繁忙可安全降级。
- 下沉主控共享配置（SSOT）：今日手账与反思日边界统一解析共享时区；模型选择支持主控 `reply_provider` 兜底；捕获、召回、反思与总控矩阵全面联动 `exclude_groups` 排除护栏。
- 同步按语言拆包的 `series-messages.js` 轻加载器（`series-messages-{en,ja,ru}.js` 按需懒加载）及统一 `series_panels.py`，大幅降低仪表盘首屏体积。

## [0.2.0-rc.6] - 2026-11-21

- 合并线上 phase5 提示词缓存优化：将静态反思模板、证据提取 `required_output` 与记忆合并 JSON schema 上提至 `system_prompt`，并为流式与非流式模型调用附带确定性 2 消息静态缓存锚点（`contexts`）。
- 升级群间系列 v2 面板联动：支持联邦搜索（`series_search_items`）、群控矩阵只读状态（`get_series_scope_status`）、世界今日生活手账（`get_journal_events`）与共享 UI Kit 跨插件跳转定位。
- 强化手账与联邦查询的角色隔离与日期时区边界：统一以 `UTC+8` 解析自然日并支持显式 `[start_ts, end_ts)` 区间透传，严格按角色过滤且禁止指定群未绑定角色时放宽为全角色查询。

## 0.2.0-rc.5 — 2026-11-20

- 优化反思（reflection）链路前缀缓存（DM-1）：保持 `system_prompt` 完全静态，将 `current_date` 与重试反馈移至用户提示尾部。

## 0.2.0-rc.4

支持在原版 AstrBot 4.29.4 的插件区域直接切换已更新的群间面板。入口和目标均需更新，左侧菜单保留入口。

补全 VitePress 主题入口，文档构建会加载现有的自定义样式。

普通回复命中经过来源校验的相关本地记忆时，可提前结束对语义检索的等待。

整合长期记忆、经历、人物与主题检索、群间请求注入及管理面板；包含检索超时、写入锁和来源守卫修复。保留数据库与原有许可声明。

这是发布候选。需要 AstrBot 4.29.4 或兼容的 4.x 版本。安装前保存旧插件目录、配置及对应 plugin_data；停止目标插件后安装，保留数据库与配置，再启用并检查加载日志。

如升级失败，先停用新版本并保留故障现场与新产生的数据；恢复升级前的代码与相匹配的数据备份。不要直接把旧数据库覆盖到运行中的插件，也不要假设较旧版本能够读取新版本写入的数据。

本候选的测试结果见发布批次验证报告。自动化检查不等于真实 QQ 收发或真实模型对话验收。
