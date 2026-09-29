# 智能运维功能入口技术设计

## 权威目录

功能定义保存在版本化 JSON：

```text
services/aiops_agent/src/aiops_agent/application/conversation_starters/catalog.json
```

目录不新增数据库表。每次选择通过 `starter_id + catalog_version + parameters` 提交，服务端
重新读取目录并校验，不信任浏览器提交内部 Profile、Tool 或 Playbook。

## API 与冻结链路

```text
GET /api/v1/apps/aiops/conversation-starters?target_id=...&agent_id=...
  → Main API
  → GET /internal/v1/aiops/conversation-starters
  → 校验 Domain、当前 Agent 版本绑定和 Target 状态

POST Conversation / Turn {starter: {...}}
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

该分支不调用模型做意图分类。模型只在后续基于真实证据组织诊断结论或文档内容。

## 安全边界

- 浏览器只接收展示字段、输入 Schema 和可用状态，不接收内部 planning 定义；
- 目录版本变化后旧选择被拒绝，要求用户重新打开菜单；
- 参数白名单、类型、时间顺序和 AWR 等长窗口由服务端校验；
- 实施入口保持 `ActionIntent.NONE`，不会绕过审批进入受控变更。
