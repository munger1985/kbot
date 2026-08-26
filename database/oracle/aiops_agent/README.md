# AIOps Agent Oracle Schema

本目录拥有 AIOps 服务的 34 张表和 10 个只读视图，按以下顺序在空 Schema
一次性执行：

1. `001_ops_roots.sql`：Target、Policy、Agent Binding、Monitor Source；
2. `002_ops_runtime.sql`：Event、Alert、Run、Task、Artifact、Run Event；
3. `003_ops_change.sql`：Proposal、HITL、Approval Token、Execution；
4. `004_ops_inspection.sql`：Inspection Plan/Target/Fire、Report；
5. `005_ops_messaging.sql`：Inbox、Outbox；
6. `006_ops_fks_views.sql`：循环外键、函数唯一索引和 APEX 投影；
7. `007_ops_agents.sql`：私有 Agent、版本和 Domain 授权；
8. `008_ops_conversations_reports.sql`：对话、报告模板和多模态证据。

需要丢弃旧 AIOps 数据并重新部署时，停止 AIOps API、Worker、Scheduler，备份
需要保留的数据，然后用 KBot Schema 所有者执行 `rebuild_aiops_schema.sql`。该脚本
会校验预期 PDB/Schema、列出待删除对象和行数、拒绝跨服务外键，并要求两次显式
确认。它只删除 `KBOT_OPS_%` 表与 `KBOT_V_OPS_%` 视图，然后直接调用上述八份
规范 DDL，避免维护第二份建表定义。

从本目录启动 SQLcl/SQL*Plus、连接到 KBot Schema 所有者后执行：

```sql
@rebuild_aiops_schema.sql
```

按提示输入预期 PDB、Schema、`STOPPED` 和最终的 `REBUILD_AIOPS`。不要把数据库
密码写进脚本或命令历史。脚本成功后先确认 `KBOT_V_OPS_SCHEMA_VERSION` 返回
`AIOPS / 10 / aiops-oracle-v2`，再启动 AIOps 服务并检查 `/ready`。

`schema_manifest.json` 是部署与步骤 2 Entity 对齐的机器可读契约。应用启动时
只检查 `KBOT_V_OPS_SCHEMA_VERSION`，不得执行 DDL、补列或调用 `create_all()`。
APEX 只能读取 `KBOT_V_OPS_*`，所有状态迁移仍通过 API Command 完成。
