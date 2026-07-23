# AIOps Agent Oracle Schema

本目录拥有 AIOps 服务的 21 张表和 10 个只读视图，按以下顺序在空 Schema
一次性执行：

1. `001_ops_roots.sql`：Target、Policy、Agent Binding、Monitor Source；
2. `002_ops_runtime.sql`：Event、Alert、Run、Task、Artifact、Run Event；
3. `003_ops_change.sql`：Proposal、HITL、Approval Token、Execution；
4. `004_ops_inspection.sql`：Inspection Plan/Target/Fire、Report；
5. `005_ops_messaging.sql`：Inbox、Outbox；
6. `006_ops_fks_views.sql`：循环外键、函数唯一索引和 APEX 投影。

`schema_manifest.json` 是部署与步骤 2 Entity 对齐的机器可读契约。应用启动时
只检查 `KBOT_V_OPS_SCHEMA_VERSION`，不得执行 DDL、补列或调用 `create_all()`。
APEX 只能读取 `KBOT_V_OPS_*`，所有状态迁移仍通过 API Command 完成。
