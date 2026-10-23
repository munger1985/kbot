# AIOps Agent Oracle Schema

本目录当前拥有 AIOps 服务的 76 张表和 10 个只读视图，按以下顺序在空 Schema
一次性执行：

1. `001_ops_roots.sql`：Target、Policy、Agent Binding、Monitor Source；
2. `002_ops_runtime.sql`：Event、Alert、Run、Task、Artifact、Run Event；
3. `003_ops_change.sql`：Proposal、HITL、Approval Token、Execution；
4. `004_ops_inspection.sql`：Inspection Plan/Fire、Report；
5. `005_ops_messaging.sql`：Inbox、Outbox；
6. `006_ops_fks_views.sql`：循环外键、函数唯一索引和 APEX 投影；
7. `007_ops_agents.sql`：私有 Agent、版本和资源绑定；
8. `008_ops_conversations_reports.sql`：Turn、Skill调用、证据、回答块、对话和报告模板；
9. `009_ops_workload.sql`：MySQL/PostgreSQL工作负载快照、指标和活动采样；
10. `010_ops_knowledge.sql`：运维知识资产、版本、受控标签、适用范围、来源、索引引用和审核记录；
11. `011_ops_recovery.sql`：数据库组件恢复目标、数据库类型化恢复坐标和人工恢复演练记录。
12. `012_ops_work_items.sql`：DBA 工作项、发生记录、资源关联和审计活动。

需要丢弃旧 AIOps 数据并重新部署时，停止 AIOps API、Worker、Scheduler，备份
需要保留的数据，然后用 KBot Schema 所有者执行
`../generated/aiops_agent/rebuild_aiops_schema.sql`。它会
直接删除 `KBOT_OPS_%` 表与 `KBOT_V_OPS_%` 视图，再内嵌执行上述十二份规范 DDL，避免
维护第二份建表定义。

在 SQL Developer 中打开该文件，确认当前连接是目标 KBot Schema，然后使用
Run Script（F5）执行。不要使用 Run Statement（Ctrl+Enter）。重建文件已经内嵌
全部十二份规范 DDL，不依赖 SQL Developer 的工作目录或其他 SQL 文件。脚本会先确认
共享的 `KBOT_PLATFORM_DOMAIN(DOMAIN_ID)` 与
`KBOT_MANAGED_CREDENTIAL(CREDENTIAL_ID, DOMAIN_ID)` 父键可用，再删除旧 AIOps
对象；结束时会按 Manifest 精确核对对象名称和 Schema 50 关键完整性合同。

此前由 `initialize_aiops.py` 创建的 `aiopsadmin`、`aiops_portal` Domain、AIOps
权限/角色/成员关系以及 `diagnosis-cases` 固定 KC Collection 位于共享平台/KC 表；用户创建的运维知识库各自拥有独立 KC Collection，
不会被本脚本删除，重建后无需再次初始化。不要无意中重复执行初始化脚本，因为它
会把 `aiopsadmin` 恢复成代码内置的初始密码。重建成功后确认
`KBOT_V_OPS_SCHEMA_VERSION` 返回 `AIOPS / 50 / aiops-oracle-v50-unified-agent-only`，再启动 AIOps
服务并检查 `/ready`。

`schema_manifest.json` 是部署与步骤 2 Entity 对齐的机器可读契约。应用就绪检查会同时
校验 `KBOT_V_OPS_SCHEMA_VERSION`、Schema 50 关键列以及业务 `CHECK` 已清零，不得执行 DDL、补列或调用
`create_all()`。APEX 只能读取 `KBOT_V_OPS_*`，所有状态迁移仍通过 API Command 完成。

从 Schema 45 升级时执行 `../operations/apply_aiops_schema_46.sql`。该脚本创建受控标签表，
为知识版本增加标签分析结果，并把知识范围关联到稳定标签 ID；执行前必须先备份并停止 AIOps 写入服务。

从 Schema 46 升级时执行 `../operations/apply_aiops_schema_47.sql`。该脚本创建 Domain 级运维
知识库和 Agent 多选绑定表，并把既有运维知识、诊断 Agent 版本及固定知识问答 Agent 迁移到
默认运维知识库；执行前必须先备份并停止 AIOps 写入服务。

从 Schema 47 升级时执行 `../operations/apply_aiops_schema_48.sql`。该脚本增加知识问答 Agent
回答模型绑定列；升级后需要在管理页完成模型配置。

从 Schema 48 升级时先执行 `../operations/apply_aiops_schema_49.sql`，再执行
`../operations/apply_aiops_schema_50.sql`。已处于 Schema 49 的环境只执行后者。Schema 50
仅删除旧固定知识问答 Agent 的 Domain 绑定与知识库关联表，不读取、迁移或覆盖
`KBOT_OPS_AGENT_VERSION_KNOW_BASE` 中统一 Agent 版本的知识库绑定。

升级失败但必须保留已配置的监控源和运维目标时，改用
`../generated/aiops_agent/rebuild_aiops_preserve_sources_targets.sql`。该文件同样在
SQL Developer 中使用 Run Script（F5）执行，会保留下列配置数据并重建全部 AIOps
对象：

- `KBOT_OPS_TARGET`：运维目标及其 Managed Credential 引用；
- `KBOT_OPS_TARGET_FACT`：人工确认或发现的目标事实；
- `KBOT_OPS_DIAGNOSTIC_SOURCE`：监控源、Webhook 和能力配置；
- `KBOT_OPS_TARGET_SOURCE_BINDING`：目标与监控源的定位和能力绑定；
- `KBOT_OPS_BUSINESS_SYSTEM`、`KBOT_OPS_MANAGED_RESOURCE` 及资源拓扑表；
- `KBOT_OPS_RECOVERY_PROFILE`：数据库组件当前及历史恢复目标版本。

恢复后，脚本会为仍未归属组件的历史 Target 各自补建一个业务系统和数据库组件；
已经属于 RAC、Data Guard 或复制拓扑的 Target 保持原归属，不会重复生成组件。

策略、Agent、巡检计划、运行、任务、报告、对话、证据、消息和审计等其他 AIOps 数据
不会恢复。脚本在删除旧表前创建 `KBOT_KEEP_AIOPS_%` 临时备份表，按显式列名恢复并核对
行数；只有 Schema 和数据边界全部验证通过后才删除临时备份表。若执行中途失败，不要删除
这些临时表，应先根据 SQL Developer 报错修复或人工恢复。
