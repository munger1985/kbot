# AIOps Agent Oracle Schema

本目录当前拥有 AIOps 服务的44张表和10个只读视图，按以下顺序在空Schema
一次性执行：

1. `001_ops_roots.sql`：Target、Policy、Agent Binding、Monitor Source；
2. `002_ops_runtime.sql`：Event、Alert、Run、Task、Artifact、Run Event；
3. `003_ops_change.sql`：Proposal、HITL、Approval Token、Execution；
4. `004_ops_inspection.sql`：Inspection Plan/Target/Fire、Report；
5. `005_ops_messaging.sql`：Inbox、Outbox；
6. `006_ops_fks_views.sql`：循环外键、函数唯一索引和 APEX 投影；
7. `007_ops_agents.sql`：私有 Agent、版本和 Domain 授权；
8. `008_ops_conversations_reports.sql`：Turn、Skill调用、证据、回答块、对话和报告模板。

需要丢弃旧 AIOps 数据并重新部署时，停止 AIOps API、Worker、Scheduler，备份
需要保留的数据，然后用 KBot Schema 所有者执行 `rebuild_aiops_schema.sql`。它会
直接删除 `KBOT_OPS_%` 表与 `KBOT_V_OPS_%` 视图，再调用上述八份规范 DDL，避免
维护第二份建表定义。

在 SQL Developer 中打开该文件，确认当前连接是目标 KBot Schema，然后使用
Run Script（F5）执行。不要使用 Run Statement（Ctrl+Enter）。重建文件已经内嵌
全部八段规范 DDL，不依赖 SQL Developer 的工作目录或其他 SQL 文件。

此前由 `initialize_aiops.py` 创建的 `aiopsadmin`、`aiops_portal` Domain、AIOps
权限/角色/成员关系和 `operations-manuals` KC Collection 位于共享平台/KC 表，
不会被本脚本删除，重建后无需再次初始化。不要无意中重复执行初始化脚本，因为它
会把 `aiopsadmin` 恢复成代码内置的初始密码。重建成功后确认
`KBOT_V_OPS_SCHEMA_VERSION` 返回 `AIOPS / 15 / aiops-oracle-v5`，再启动 AIOps
服务并检查 `/ready`。

`schema_manifest.json` 是部署与步骤 2 Entity 对齐的机器可读契约。应用启动时
只检查 `KBOT_V_OPS_SCHEMA_VERSION`，不得执行 DDL、补列或调用 `create_all()`。
APEX 只能读取 `KBOT_V_OPS_*`，所有状态迁移仍通过 API Command 完成。

## 保留现有 AIOps 数据升级

已经运行 Schema 13 且需要保留 Target、Agent、监控源、Webhook Key/Secret、会话和
诊断数据时，不要执行全量重建文件。停止 AIOps API、Worker 和 Scheduler 并完成备份后，
使用 Schema Owner 在 SQL Developer 中以 Run Script（F5）执行：

```text
upgrade_13_to_15_preserve_data.sql
```

该脚本只接受 `AIOPS / 13 / aiops-oracle-v3`，会在任何 DDL 前检查当前 Agent 与历史
会话是否能够唯一确定逻辑 Target；无法确定时直接终止，不猜测映射。升级完成后应返回
`AIOPS / 15 / aiops-oracle-v5`。脚本不会删除或重建
`KBOT_OPS_DIAGNOSTIC_SOURCE`、`KBOT_MANAGED_CREDENTIAL` 和
`KBOT_OPS_TARGET_SOURCE_BINDING`，因此不需要因为本次数据库升级重新生成 Webhook 密钥
或重新运行监控 Docker。Schema 13 的 Skill 调用会先逐条复制为 Playbook 与 Tool 审计
记录并核对数量，确认完整后才删除已经被替代的旧 Skill 表。
