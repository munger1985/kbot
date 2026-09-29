# AIOps Agent Oracle Schema

本目录当前拥有 AIOps 服务的50张表和10个只读视图，按以下顺序在空Schema
一次性执行：

1. `001_ops_roots.sql`：Target、Policy、Agent Binding、Monitor Source；
2. `002_ops_runtime.sql`：Event、Alert、Run、Task、Artifact、Run Event；
3. `003_ops_change.sql`：Proposal、HITL、Approval Token、Execution；
4. `004_ops_inspection.sql`：Inspection Plan/Fire、Report；
5. `005_ops_messaging.sql`：Inbox、Outbox；
6. `006_ops_fks_views.sql`：循环外键、函数唯一索引和 APEX 投影；
7. `007_ops_agents.sql`：私有 Agent、版本和资源绑定；
8. `008_ops_conversations_reports.sql`：Turn、Skill调用、证据、回答块、对话和报告模板；
9. `009_ops_workload.sql`：MySQL/PostgreSQL工作负载快照、指标和活动采样。

需要丢弃旧 AIOps 数据并重新部署时，停止 AIOps API、Worker、Scheduler，备份
需要保留的数据，然后用 KBot Schema 所有者执行
`../generated/aiops_agent/rebuild_aiops_schema.sql`。它会
直接删除 `KBOT_OPS_%` 表与 `KBOT_V_OPS_%` 视图，再内嵌执行上述九份规范 DDL，避免
维护第二份建表定义。

在 SQL Developer 中打开该文件，确认当前连接是目标 KBot Schema，然后使用
Run Script（F5）执行。不要使用 Run Statement（Ctrl+Enter）。重建文件已经内嵌
全部八段规范 DDL，不依赖 SQL Developer 的工作目录或其他 SQL 文件。脚本会先确认
共享的 `KBOT_PLATFORM_DOMAIN(DOMAIN_ID)` 与
`KBOT_MANAGED_CREDENTIAL(CREDENTIAL_ID, DOMAIN_ID)` 父键可用，再删除旧 AIOps
对象；结束时会按 Manifest 精确核对对象名称和 Schema 31 关键完整性合同。

此前由 `initialize_aiops.py` 创建的 `aiopsadmin`、`aiops_portal` Domain、AIOps
权限/角色/成员关系和 `operations-manuals` KC Collection 位于共享平台/KC 表，
不会被本脚本删除，重建后无需再次初始化。不要无意中重复执行初始化脚本，因为它
会把 `aiopsadmin` 恢复成代码内置的初始密码。重建成功后确认
`KBOT_V_OPS_SCHEMA_VERSION` 返回 `AIOPS / 31 / aiops-oracle-v21`，再启动 AIOps
服务并检查 `/ready`。

`schema_manifest.json` 是部署与步骤 2 Entity 对齐的机器可读契约。应用就绪检查会同时
校验 `KBOT_V_OPS_SCHEMA_VERSION`、Schema 31 关键列以及业务 `CHECK` 已清零，不得执行 DDL、补列或调用
`create_all()`。APEX 只能读取 `KBOT_V_OPS_*`，所有状态迁移仍通过 API Command 完成。

升级失败但必须保留已配置的监控源和运维目标时，改用
`../generated/aiops_agent/rebuild_aiops_preserve_sources_targets.sql`。该文件同样在
SQL Developer 中使用 Run Script（F5）执行，会保留以下四张表的数据并重建全部 AIOps
对象：

- `KBOT_OPS_TARGET`：运维目标及其 Managed Credential 引用；
- `KBOT_OPS_TARGET_FACT`：人工确认或发现的目标事实；
- `KBOT_OPS_DIAGNOSTIC_SOURCE`：监控源、Webhook 和能力配置；
- `KBOT_OPS_TARGET_SOURCE_BINDING`：目标与监控源的定位和能力绑定。

策略、Agent、巡检计划、运行、任务、报告、对话、证据、消息和审计等其他 AIOps 数据
不会恢复。脚本在删除旧表前创建 `KBOT_KEEP_AIOPS_%` 临时备份表，按显式列名恢复并核对
行数；只有 Schema 和数据边界全部验证通过后才删除临时备份表。若执行中途失败，不要删除
这些临时表，应先根据 SQL Developer 报错修复或人工恢复。
