# 持续叙事接入（schema_version=1）

`recall_context(scope, query)` 返回 `items/available`；`ingest_events(events)` 返回本次新增 `accepted`。Scope 必须包含 platform、完整 umo、persona_id、instance_id。实际消息与确认成功的发送段在 ConversationStore 内和去重记录一起提交；失败、不确定、草稿不会成为聊天记忆。

受管事件设置 `_narrative_managed_v1`，原召回、群收集和模型响应反思跳过，防止组合提示词与未发送草稿入库。已存在的长期摘要以 subjective/unknown 与 provenance_kind=historical 返回，不可直接当作约定或执行证据。受管原始记录按完整 Scope 分区，用户文字以“用户发言”前缀返回，不能把用户声称的事当成已确认事实。确认的发送记录使用统一事实类别 receipt。

新表自动创建，无需删除旧数据。撤回到旧插件版本保留 narrative_events 表但停止读取；备份与回退时同时保留原 conversation SQLite 数据文件。

`2.7.0-rc.2` 的历史底座在本地 SQLite/FTS 完成，不依赖 embedding/rerank 成功。原始来源严格限定为当前完整 UMO；source_session_id、source_window.session_id 相互矛盾的记录，以及全局或无法确定原会话的旧资料不注入。允许当前角色 ID、世界书提供的有限名称别名，以及 default/空人格的同会话旧材料；其他命名角色排除。旧 default/空人格记录明确标为“当时角色不明的历史群聊材料”，不能成为当前角色承诺或人生事实。

旧摘要占据输出前部，近期输入不能挤掉全部历史。可选语义补充与本地检索并行且有独立预算；超时保留本地结果。可在 recall_engine 配置 narrative_local_timeout_seconds（默认 1 秒）和 narrative_hybrid_timeout_seconds（默认 0.75 秒，0 可关闭）；上层调用预算需更长。受管检索不沿用全库缓存或自动迁移未知旧 ID。

受管 `recall_long_term_memory` 工具使用同一个公开 `recall_context`，保留来源、未知角色和主观标注；要求原文时只返回同完整 UMO 或当前受管分区的来源消息。非受管工具路径保持原行为，访问白名单仍在查询前检查。
