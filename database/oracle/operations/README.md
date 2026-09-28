# Oracle DBA 操作资产

本目录保存必须由 DBA 显式确认后执行的重置、导出等 Oracle SQL。它们不属于空 Schema
初始化序列，也不会由应用启动或部署入口自动执行。

每个操作必须在文件头说明影响范围、前置条件、是否包含敏感数据以及失败后的恢复方式。

`apply_aiops_schema_20.sql` 用于已有测试数据不能重建时，将 AIOps Schema 19 原地升级到
Schema 20。执行前必须停止 AIOps API、Worker 和 DB Executor，并完成 Schema 备份；脚本保留
历史数据，但会关闭历史 Agent–Target 的受控执行，升级后需要在 Agent 页面重新明确授权。

`apply_aiops_schema_22.sql` 用于将 AIOps Schema 21 原地升级到 Schema 22，扩展附件受控
检索所需的工具调用分类约束。执行前必须停止 AIOps API、Worker、Scheduler 和 DB Executor，
并完成 Schema 备份；脚本不删除表、不删除业务行，也不修改既有诊断或审计数据。

`apply_aiops_schema_23.sql` 用于将 AIOps Schema 22 原地升级到 Schema 23，增加 Oracle
CDB/PDB 预期范围与连接实测字段。脚本保留全部业务行、Endpoint 和凭据引用；由于 Service
Name 不能可靠判断容器类型，既有 Oracle 直连 Target 会被安全停用为仅监控模式，升级后须在
Target 页面明确选择 `CDB Root`、`PDB` 或 `Non-CDB`，重新测试连接后再启用。

`apply_aiops_schema_26.sql` 用于将 AIOps Schema 23 / `aiops-oracle-v13` 或 Schema 24 /
`aiops-oracle-v14` 原地升级到 Schema 26 / `aiops-oracle-v16`。脚本给巡检计划补
`SELECTED_CHECKS_JSON`，并把缺少检查项快照的计划回填为当前 Check Catalog 的 READY
默认项；同时新增 `KBOT_OPS_TARGET_FACT`。执行前必须停止 AIOps API、Worker、Scheduler
和 DB Executor，并完成 Schema 备份。若当前仍是 Schema 23，先执行
`converge_core_authorization.sql` 删除废弃的 `KBOT_OPS_AGENT_GRANT`。脚本不删除业务行，
也不改写历史运行、诊断、报告、附件或审计数据。

`apply_aiops_schema_27.sql` 用于将 AIOps Schema 26 / `aiops-oracle-v16` 原地升级到
Schema 27 / `aiops-oracle-v17`。脚本仅扩展回答块类型约束以允许
`IMPLEMENTATION_RUNBOOK`，并更新 Schema 版本视图；不删除或改写任何业务行。

`apply_aiops_schema_28.sql` 用于将 AIOps Schema 27 / `aiops-oracle-v17` 原地升级到
Schema 28 / `aiops-oracle-v18`。脚本移除全部 `KBOT_OPS_%` 表上的用户命名 `CHECK`
约束，把枚举、状态和范围规则收回应用层合同；不删除表，也不删除或改写业务行。执行前
必须停止 AIOps API、Worker、Scheduler 和 DB Executor，并完成 Schema 备份。

`apply_knowledge_retrieval_and_media_studio.sql` 用于既有 Schema 补齐知识检索与多媒体创作
工作台表结构：把 `KBOT_KR_AGENT_VERSION` 从 `ENABLED_CAPABILITIES_JSON` 收敛为
`KNOWLEDGE_CORE_ID` + `DATA_MODEL_IDS_JSON`，新增 X Search 运行表，并创建多媒体绑定、
Run、事件、提示词版本和资产表。脚本会删除已退役的智能工作台 `KBOT_ASST_%` 表，不迁移
历史运行或图片元数据。表结构完成后依次执行
`conda run -n kbot4 python scripts/db/apply_oracle_schema.py --foundation-only`、
`database/oracle/bootstrap/knowledge_retrieval/initial_admin.sql` 和
`database/oracle/bootstrap/media_studio/initial_admin.sql`，以收敛权限目录并写入各 App
初始管理员。

`enable_model_serving_capabilities.sql` 用于为已有模型目录新增图片生成能力标志。
X Search 属于 Grok 模型能力，不再写入模型表。脚本不会把任何模型标记为可用；新增列默认
关闭，必须由受控 Canary 写入验收结果后，业务入口才可使用对应能力。

## 权限模型收敛

既有 KBot 4.0 Schema 升级到统一核心权限规则时，先由 Schema Owner 执行
`converge_core_authorization.sql` 删除废弃的人类 Agent Grant 表，再运行：

```bash
conda run -n kbot4 python scripts/db/apply_oracle_schema.py --foundation-only
```

第二步会精确收敛权限目录和系统内置角色映射；自定义角色保留，但其废弃权限映射会被清理。
