# 智能运维功能入口技术设计

本页描述当前功能入口；SQL 结果行的结构化追问和诊断分析目标见
[SQL 诊断详细设计（待实施）](../proposals/aiops-sql-diagnosis-detailed-design.md)。

2026-10-28 目标设计复审明确：Top SQL 入口先完成排名与负载解读，不因历史对话切入
旧 SQL 的调查；选择某行后按结构化对象进入多源诊断，单 SQL 视图不重复整张排名表。
本页既有确定性入口编译不等于上述选择链、假设驱动调查及方案验证已实施。

## 权威目录

功能定义保存在版本化 JSON：

```text
services/aiops_agent/src/aiops_agent/application/conversation_starters/catalog.json
```

目录不新增数据库表。每次选择通过 `starter_id + catalog_version + parameters` 提交，服务端
重新读取目录并校验，不信任浏览器提交内部 Profile、Tool 或 Playbook。

实施文档入口的 `input_schema` 不在 JSON 中重复维护，而是按 `ImplementationProfile` 从服务端参数
目录动态生成。这样功能菜单、聊天参数白名单和编译器共享同一字段集合，避免展示字段与可执行字段漂移。

## API 与冻结链路

```text
GET /api/v1/apps/aiops/conversation-starters?agent_id=...
  → Main API
  → GET /internal/v1/aiops/conversation-starters
  → 返回当前 Agent 版本授权的业务系统及其全部数据库 Target

POST Conversation {context_hint: {business_system_id, resource_ids}, starter: {...}}
  → 由数据库资源解析 Target 并再次校验 Agent 授权和业务系统归属
  → 服务端校验参数并生成可读 User Message
  → Outbox.execution_context.conversation_starter
  → Run.plan_snapshot_json.client_metadata.conversation_starter
  → TurnPlanningContext.conversation_starter
```

选择结果进入现有 Message、Turn 和 Run JSON 快照，不改变 Oracle Schema，也不引入业务
`CHECK`、`UNIQUE` 或唯一索引。

## 确定性规划

`TurnPlanningService` 在巡检模板之后、模型 Planner 之前识别已冻结功能入口：

- 实施文档直接映射 `ImplementationProfile` 和固定前置 Tool/Playbook；
- 单 SQL 直接映射 `SINGLE_SQL_PERFORMANCE`；
- AWR/AWR Diff 使用现有 snapshots discovery binding 完成时间到快照的绑定；
- 通用诊断按目录编译固定 Tool DAG；
- 空间趋势固定历史窗口并请求监控快照。
- 备份与恢复使用公共 `database.backup-recovery.status`：按 Target 数据库类型选择 Oracle RMAN、
  PostgreSQL WAL/Archiver 或 MySQL Binlog/Replication 工具，并同时消费
  `TARGET_RECOVERY_ASSURANCE.v1` 中的 RPO/RTO 与演练证据。

该分支不调用模型做意图分类。后续回答可能由确定性摘要直接生成或由模型结合证据组织；
不能因此推断单 SQL 的证据关联与深入分析已经完成。

## 安全边界

- 浏览器只接收展示字段、输入 Schema 和可用状态，不接收内部 planning 定义；
- 浏览器只提交业务系统 ID 和资源 ID；`target_id` 由服务端根据数据库资源解析；
- 一个业务系统存在多个数据库 Target 时全部列出，由用户明确单选后执行；
- 快捷入口始终使用范围化 Conversation，不在 Turn 级覆盖已有会话的可信上下文；
- 目录版本变化后旧选择被拒绝，要求用户重新打开菜单；
- 参数白名单、类型、时间顺序和 AWR 等长窗口由服务端校验；
- 实施文档参数全部可选，空值不进入冻结快照；解析优先级为用户输入、数据库事实、Target 运维事实、
  确定性默认值；
- 生成结果固化入口版本和用户参数；调整参数会创建新 Turn，不修改历史文档；
- 密码、密钥、Wallet 和自由命令不属于任何入口参数；
- 实施入口保持 `ActionIntent.NONE`，不会绕过审批进入受控变更。
- WAL、Binlog 或备份元数据不能单独证明实际可恢复；外部备份 Provider 未接入时必须输出数据缺口。
