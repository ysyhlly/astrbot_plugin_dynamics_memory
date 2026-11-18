# 群间记忆 · Dynamics Memory

> 发布候选：`0.2.0-rc.3`。升级说明与验证边界见 [RELEASE_NOTES.md](RELEASE_NOTES.md)。

![群间记忆](logo.png)

Chat Dynamics 系列的独立长期记忆插件。基于当前部署的 LivingMemory **2.10.0+local.3**，保留本地修改、数据结构、原文证据、混合召回、2D/3D 图谱和公开接口。无需安装原 LivingMemory。

## 功能与分工

- **群间记忆**保存对话经历、来源证据、衍生见解与历史变化，提供召回和原文查看。
- **群间世界**维护角色设定、世界规则、生活、故事与约定；记忆不复制它的数据库，也不将生活计划当成已经发生的事实。
- **群间学习**维护已审查的表达与行为。学习策略不是聊天事实；通过同一来源引用读取证据，不把学习结果重新灌入原始消息。
- 同一会话、人格、角色实例和来源 ID 的重复写入幂等；内容发生变化则保留原记录，另记来源冲突。来源审查不会直接覆盖原文。
- 回复证据视图合并完全相同条件下的重复陈述；不同成员、时间、极性、条件以及实际执行回执不合并。所有原始证据仍保留。
- 语义整合、衰减、图谱、重建索引、来源修复与人工纠正继续沿用原实现。语义相似不等于事实相同，未经确认的矛盾不自动判真伪。

## 安装与接入

需要 AstrBot `>=4.29.4,<5`。安装本包及交付目录中的 Chat Dynamics 接入补丁。主插件必须支持 `memory: 1` 能力及新插件工具身份，单独改插件名称不能完成原生联动。

新装默认关闭“接入聊天与记忆记录”。先迁移或配置模型，再停用原 LivingMemory，开启配置并重载。本插件发现其他活动记忆提供者时暂停自身接入，避免双重采集和同名工具竞争。不要同时启用两套记忆。

插件使用独立目录 `data/plugin_data/astrbot_plugin_dynamics_memory/`。原配置值与 Provider ID 可原样迁入；本插件不修改全局模型、角色卡或旧数据。管理页保留原记忆、图谱、召回测试、提示词及设置，并新增“系列接入与来源冲突”。

## 迁移

迁移工具默认只预览；正式导入前停用源和目标插件。只向不存在或完全为空的目标目录导入，已初始化的目标拒绝覆盖。目标已有数据时应先另存备份，选择新的导入目标；不要直接删除已有记忆。

```bash
python migration.py --source /AstrBot/data/plugin_data/astrbot_plugin_livingmemory --config /AstrBot/data/config/astrbot_plugin_livingmemory_config.json
python migration.py --source /AstrBot/data/plugin_data/astrbot_plugin_livingmemory --config /AstrBot/data/config/astrbot_plugin_livingmemory_config.json --destination /AstrBot/data/plugin_data/astrbot_plugin_dynamics_memory --apply-fingerprint <预览指纹> --offline
```

复制活动数据库、索引、衰减状态、自定义素材与配置；SQLite 使用只读连接和 backup API 包含已提交 WAL，校验数据库完整性和表行数。文件摘要、配置摘要在导入前后复核。旧备份、损坏索引和临时重建文件留在源目录，不作为活动数据导入。

导入后将目标目录的 `migration-settings.json` 作为**新插件**配置（`data/config/astrbot_plugin_dynamics_memory_config.json`），再开启接入并重载。迁移文件含原配置，应按原配置的权限保管。迁移工具不自动启用插件。

回滚：停用群间记忆，恢复此前 Chat Dynamics，再启用旧 LivingMemory。旧数据从未被改写；切换后新增数据留在新目录，不自动反向覆盖旧库。

## 接口兼容

保留 `get_evidence_view`、`recall_context`、`ingest_events`、`correct_insight` 和 `api.public` 门面、`/lmem` 管理命令及原工具名。请求标记和来源验证器保持兼容。**Python 导入前缀和页面 URL 改为新插件 ID**；旧插件硬编码导入路径的消费者需迁到公开能力发现，不能声称任意第三方无需调整。

新增 `get_dynamics_capabilities()` 声明 `memory` 与 `evidence` v1。
`get_shared_evidence(scope, source_refs, purpose="reply", card_version="")` 只读取已验证来源，不产生二次提取任务。
`get_series_context(...)` 可只读组合世界、记忆与学习；各自结果分开返回、不落地复制、不自行注入提示词。学习调用者需传入当前作用域、人格指纹、成员、策略基线和宿主版本，不能猜测这些值。

来源冲突审查采用版本校验；原有衍生见解纠正仍通过 `correct_insight` 的 withdraw / dispute / supersede 流程进行，不会修改用户原话或把亲近度当作事实权限。

## 验证与来源

新增测试覆盖幂等、来源内容碰撞、并发、重启、跨成员/作用域隔离、来源审查版本、世界重载、只读学习联动及 WAL 迁移。

```bash
PYTHONPATH=/AstrBot:<本插件父目录> python -m pytest -q tests/test_dynamics_series.py
```

交付目录 `VALIDATION.md` 记录真实检查结果及基线问题。离线测试不等于真实 QQ 和 LLM 验收。

基线作者 `lxfight`；保留 [LICENSE](LICENSE)、[CHANGELOG.md](CHANGELOG.md) 和 [NOTICE.md](NOTICE.md)。原项目资料：[原说明](README_LIVINGMEMORY.md)、[English](README_en.md)、[Русский](README_ru.md)。
