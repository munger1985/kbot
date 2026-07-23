# 4.0 AIOps API 与跨服务契约

## 契约边界

AIOps 使用三套明确分离的契约，不能共用 Controller 或信任模型：

```text
Browser / APEX / Client
        │ /api/v1/ops/*
        ▼
Main API / BFF ── Management Client ──► /internal/v1/aiops/*
        │                               │
        └─ monitoring webhook ──────────┘
                                        │ signed executor contract
                                        ▼
                              AIOps DB Executor
```

- `/api/v1/ops/*` 是唯一面向用户的 AIOps API，由 Main API 发布；
- `/api/v1/integrations/monitoring/*` 是唯一面向监控系统的接入 API；
- `/internal/v1/aiops/*` 只接受 Main API、Root Agent 和受信 Worker 的 Service Identity；
- `/internal/v1/db-executor/*` 只接受 AIOps Worker，不对用户、APEX 或 Root Agent 开放。

公开与内部 `v1` 都是接口契约的首个版本，不代表 KBot 1.x，也不随产品 4.0 使用 `v4`。未来 AIOps 内部契约可以独立升级，不改变外部 `/api/v1`。

## 外部资源 API

### 配置资源

| 方法与路径 | 用途 |
| --- | --- |
| `POST/GET /api/v1/ops/targets` | 创建 Target、按授权范围分页查询 |
| `GET/PATCH /api/v1/ops/targets/{target_id}` | 查询或修改 Target 配置 |
| `POST /api/v1/ops/targets/{target_id}/disable` | 停用 Target；已产生的历史 Run 不删除 |
| `POST /api/v1/ops/targets/{target_id}/agent-bindings` | 创建 Agent Binding |
| `PATCH /api/v1/ops/targets/{target_id}/agent-bindings/{binding_id}` | 修改 Agent Binding；撤权使用 `status=REVOKED` |
| `POST/GET /api/v1/ops/monitor-sources` | 创建或查询 Monitor Source |
| `GET/PATCH /api/v1/ops/monitor-sources/{source_id}` | 查询或修改 Monitor Source |
| `POST /api/v1/ops/monitor-sources/{source_id}/webhook-key:rotate` | 轮换只显示一次的 Webhook 路由 Key |
| `POST /api/v1/ops/targets/{target_id}/monitor-bindings` | 绑定监控对象、优先级和指标范围 |
| `PATCH /api/v1/ops/targets/{target_id}/monitor-bindings/{binding_id}` | 修改或停用 Monitor Binding |
| `POST /api/v1/ops/monitor-sources/{source_id}/health-checks` | 异步触发连接检查 |
| `POST/GET /api/v1/ops/policies` | 创建不可变 Policy 版本、分页查询 |
| `GET /api/v1/ops/policies/{policy_id}` | 查询 Policy 版本 |
| `POST /api/v1/ops/policies/{policy_id}/activate` | 原子替换同 Key 的 Active Policy |
| `POST /api/v1/ops/policies/{policy_id}/retire` | 退役 Policy 版本 |
| `POST/GET /api/v1/ops/inspection-plans` | 创建或查询巡检计划 |
| `GET/PATCH /api/v1/ops/inspection-plans/{plan_id}` | 查询、暂停、恢复或修改巡检计划 |

Target、Source、Binding、Policy 和 Plan 均不提供在线物理删除 API。`PATCH`、激活、停用、撤权、暂停和恢复必须携带 `If-Match: "rv-{row_version}"`；版本过期返回 `412 PRECONDITION_FAILED`。列表使用不透明 Cursor，不使用会在并发写入下漂移的页码。

配置 API 的完整状态、权限、SecretRef、Key Rotation、ETag 和 Cursor 规则见 [32_aiops_step3_configuration_and_authorization_api.md](32_aiops_step3_configuration_and_authorization_api.md)。

创建 Target 的请求不接受 `domain_id`、`app_id`、用户 ID 或明文密码：

```json
{
  "target_key": "erp-prod",
  "display_name": "ERP Production",
  "db_type": "ORACLE",
  "version_code": "19c",
  "environment": "PROD",
  "db_role": "PRIMARY",
  "endpoint": {"host": "db.internal", "port": 1521, "service": "ERP"},
  "diagnostic_secret_ref": "vault://kbot/erp/readonly",
  "execution_secret_ref": null,
  "execution_mode": "ADVISORY",
  "security_level": 3
}
```

`domain_id/app_id/created_by` 从 `AuthContext` 和平台配置派生。响应和日志均不得回显 Secret 内容。

### Run、交互与报告

| 方法与路径 | 用途 |
| --- | --- |
| `POST /api/v1/ops/runs` | 直接发起 AIOps Chat/API Run |
| `GET /api/v1/ops/runs/{ops_run_id}` | 查询状态、根因等级和最终 ArtifactRef |
| `GET /api/v1/ops/runs/{ops_run_id}/events` | SSE 订阅，支持 `Last-Event-ID` |
| `POST /api/v1/ops/runs/{ops_run_id}/cancel` | 协作式取消未终止 Run |
| `GET /api/v1/ops/runs/{ops_run_id}/pending-input` | 恢复当前待回复的 Chat HITL |
| `GET /api/v1/ops/hitl/{hitl_id}` | 授权读取完整人工输入请求 |
| `POST /api/v1/ops/hitl/{hitl_id}/responses` | 回贴人工诊断结果或补充数据 |
| `POST /api/v1/ops/hitl/{hitl_id}/skip` | 放弃当前补证并基于现有证据收敛 |
| `POST /api/v1/ops/hitl/{hitl_id}/uploads` | 创建绑定 Query 的受限上传会话 |
| `POST /api/v1/ops/hitl/{hitl_id}/uploads/{upload_id}/complete` | 完成上传、内容检查和 Hash 固化 |
| `GET /api/v1/ops/proposals/{proposal_id}` | 查询命令、影响、证据、回滚和验证方案 |
| `POST /api/v1/ops/proposals/{proposal_id}/approve` | 显式批准一条命令 |
| `POST /api/v1/ops/proposals/{proposal_id}/reject` | 拒绝一条命令 |
| `POST /api/v1/ops/proposals/{proposal_id}/manual-result` | 回填 Advisory 人工执行结果 |
| `GET /api/v1/ops/approvals?status=PENDING` | 查询当前用户有权处理的待审项 |
| `GET /api/v1/ops/inspection-fires`、`GET /api/v1/ops/inspection-fires/{fire_id}` | 查询计划时点、Target 展开计数和终态 |
| `GET /api/v1/ops/reports`、`GET /api/v1/ops/reports/{report_id}` | 查询当前报告列表、摘要和内容引用 |
| `GET /api/v1/ops/reports/{report_id}/versions` | 查询同一 Run/Report Key 的更正历史 |

创建 Run 和所有 Command API 必须携带 `Idempotency-Key`。直接创建 AIOps Run 的最小请求为：

```json
{
  "agent_id": "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11",
  "target_id": "019c03b6-10df-7869-9d0f-28a973389bd7",
  "input": "分析过去一小时 ERP 数据库响应变慢的原因",
  "session_id": "chat-20260723-001",
  "client_metadata": {"channel": "apex"}
}
```

服务端验证 `用户范围 ∩ Agent/Target Binding ∩ Target 状态`；客户端提交的 `agent_id/target_id` 只能缩小范围。成功返回 `202`：

```json
{
  "ops_run_id": "019c03b7-2e18-78a1-b07c-a12c84e93f44",
  "status": "CREATED",
  "row_version": 1,
  "event_cursor": 0,
  "events_url": "/api/v1/ops/runs/019c03b7-2e18-78a1-b07c-a12c84e93f44/events"
}
```

经 Root Agent 路由的请求先创建通用 Agent Run/Delegation，再由 Agent Runtime 调用内部 AIOps 契约；此时 `PARENT_AGENT_RUN_ID/PARENT_DELEGATION_ID` 非空。直接 AIOps Chat 的两个字段均为空。两类 Run 使用相同 AIOps 状态机，但 Root SSE 只读取父 Run 中已持久化的安全投影，不复制全部内部诊断事件。

报告列表默认只返回 `IS_CURRENT=1`，可按 Target、Report Type 和周期过滤；`include_history` 不作为列表开关，历史必须从单 Report 的 Versions 资源读取。Fire 和 Report 的版本、发布与 APEX 投影规则见 [39_aiops_step10_inspection_reporting_and_comparison.md](39_aiops_step10_inspection_reporting_and_comparison.md)。

## HITL 与审批 DTO

人工诊断回复不接受任意 SQL 指令，只接受对当前 `MANUAL_DIAGNOSTIC_SQL` 请求的结果：

```json
{
  "expected_row_version": 4,
  "responses": [
    {
      "query_id": "oracle.session_waits.v2:q1",
      "status": "SUCCEEDED",
      "format": "CSV",
      "upload_id": "upl_01J...",
      "error": null
    }
  ],
  "note": "在主库执行，采样时间 10:30"
}
```

小结果可使用受大小限制的 `inline_data`，大结果必须引用一次性 `upload_id`。`query_id` 必须属于该 HITL，文件 Hash、MIME、大小和安全级别必须与上传会话一致。`TRIGGER_TYPE` 不是 `CHAT` 时，服务端拒绝 `MANUAL_DIAGNOSTIC_SQL` 回复。

HITL SSE 事件只携带 ID、类型、过期时间和请求 ArtifactRef，不携带 SQL 或结果 Schema；完整请求由授权用户通过 GET 获取。普通聊天文本不能替代带 `hitl_id/row_version/idempotency_key` 的回复 Command。上传、解析、锁竞争和恢复契约见 [37_aiops_step8_chat_manual_diagnosis_hitl.md](37_aiops_step8_chat_manual_diagnosis_hitl.md)。

批准接口仅接受 `expected_row_version`、用户实际查看的 `expected_proposal_hash` 和可选备注。前端不能提交或看到 Approval Authorization/Mutation Grant；AIOps 在事务内创建授权记录和 Execution，DB Executor 随后通过 Claim 获取绑定实例的短期 Grant。聊天文本中的“同意”不调用批准接口。重复同一批准请求返回原结果，参数、Target、Policy 或 Proposal Hash 已变化则返回冲突并要求重新审阅。

## 监控事件接入

监控系统调用：

```text
POST /api/v1/integrations/monitoring/{webhook_key}/events
```

`webhook_key` 是随机、不透明的路由标识，不是 `source_key` 或凭据。Main API 只进行请求大小、速率、Content-Type 和 Trace 检查；AIOps Provider Adapter 使用原始请求字节、时间戳、签名头和 `SECRET_REF` 验证来源，再解析 Provider Payload。请求体中的 domain、Target ID、严重等级和用户字段都不可信，Target 只能由已验证的 Monitor Source 与 `TARGET_MONITOR` Mapping 解析。

只有 Route Key 命中、原始字节验签成功且 Inbox/Event 事务持久化后才返回 `202 event_receipt`。无效认证返回 `401/403` 且不落 Inbox；Secret Store 或 AIOps 暂不可用返回 `503` 让 Provider 重试。Provider 重试同一事件时，根据 `MONITOR_SOURCE_ID + SOURCE_EVENT_KEY` 返回已有 `event_id`，不重复创建 Alert 或 Run。经验证的正文超过数据库 Inline 阈值但未超过请求上限时先存对象存储，数据库只保存 URI 与 Hash。

## 内部 AIOps 契约

`AIOpsManagementClient` 与 `AIOpsDelegationClient` 包装同一服务的两组最小权限契约。关键接口为：

| 方法与路径 | 调用者 | 作用 |
| --- | --- | --- |
| `POST /internal/v1/aiops/runs` | Main API | 创建 Direct Chat/API Run |
| `GET /internal/v1/aiops/runs/{id}` | Main API | 查询授权后的 Run 摘要 |
| `GET /internal/v1/aiops/runs/{id}/events` | Main API | 读取 Direct Run 用户事件 |
| `POST /internal/v1/aiops/runs/{id}/commands` | Main API | 取消、回复、批准等类型化 Command |
| `POST /internal/v1/aiops/delegations` | Agent Runtime | 以 Delegation ID 幂等创建子 Run |
| `GET /internal/v1/aiops/delegations/{id}/events` | Agent Runtime | 按 Child Sequence 分页读取安全事件 |
| `GET /internal/v1/aiops/delegations/{id}/result` | Agent Runtime | 获取终态 Result Envelope |
| `POST /internal/v1/aiops/delegations/{id}/cancel` | Agent Runtime | 请求取消精确关联的子 Run |
| `POST /internal/v1/aiops/intake/monitor-events` | Main API Integration Adapter | 提交已限流但未解析的原始事件 |
| `POST /internal/v1/aiops/executions/{id}/claim` | DB Executor | 原子消费审批令牌并获取一次执行许可 |
| `POST /internal/v1/aiops/executor-events` | DB Executor | 幂等回传执行状态和结果引用 |

配置、HITL、Proposal 和 Report 使用相同资源名挂在 `/internal/v1/aiops/` 下，由 Management Client 将 Public DTO 映射为 Internal DTO；Main API 不能简单透传 URL、Header 或原始响应。

内部请求同时包含 Service Identity 和签名、短期、限定 audience 的 `AuthContext`。Agent Runtime 只能创建 `caller_mode=ROOT_DELEGATION` 且带 `parent_agent_run_id/parent_delegation_id` 的 Run，并通过精确 Delegation 资源读取事件、结果或取消；不能调用配置、审批、HITL Response 或 Executor API。Main API 才能代理用户 Command。`trigger_type=ALERT/SCHEDULE` 只能由已登记的 Integration Adapter/Scheduler 身份创建，不能由请求体伪造。完整父子协议见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

DB Executor 契约独立为：

```text
POST /internal/v1/db-executor/diagnostics
POST /internal/v1/db-executor/executions
GET  /internal/v1/db-executor/executions/{executor_request_id}
```

只读诊断提交短期签名 `DiagnosticExecutionGrant`、类型化参数和幂等键，Grant 绑定 Task Lease、Target 版本、连接 Profile、SecretRef、模板版本/Hash 和执行上限。变更 Dispatcher 只向 DB Executor 提交 `execution_id + executor_request_id`；Executor 使用实例身份调用 AIOps Claim，原子消费 Approval Authorization 并取得 `MutationExecutionGrant`。两者均不接受自然语言、任意 SQL、数据库密码、Run 状态或用户 Bearer Token。重复、过期、跨实例或绑定字段不一致时不得执行。回调可能重复或乱序，AIOps 通过 Inbox Key 与 `status_version` 对账后推进状态。只读诊断见 [35_aiops_step6_readonly_database_diagnostics.md](35_aiops_step6_readonly_database_diagnostics.md)，Mutation 见 [38_aiops_step9_advisory_approval_and_execution.md](38_aiops_step9_advisory_approval_and_execution.md)。

## SSE 事件契约

SSE `id` 等于 `KBOT_OPS_RUN_EVENT.SEQUENCE_NO`，`Last-Event-ID` 从下一序号恢复。`report.ready` 仅携带 Report ID/Key/Type/Version/Status 和安全摘要，完整内容始终重新经授权 GET 获取。稳定事件类型包括：

```text
run.status
task.status
diagnostic.progress
diagnostic.input_required
proposal.pending_approval
execution.status
report.ready
run.completed | run.failed | run.cancelled | run.expired
```

每个 `data` 至少包含 `schema_version`、`ops_run_id`、`sequence_no`、`occurred_at`、`status`、`trace_id` 和可选 Artifact/Resource Ref。用户事件不包含内部 Prompt、Policy 全量快照、Secret、数据库连接信息或未脱敏原始结果。终态事件后发送 `done` 并关闭连接。

## 身份与授权

请求体永远不能覆盖 `AuthContext`。当前阶段由 Main API 校验门户 API Key，将 Portal 声明的 `domain_id` 和 `user_id` 写入短期内部 JWT；AIOps 只接受该内部身份，不接受门户 Key。KBot 4.0 暂不实现 Scope、角色或 Target ACL，但所有读取和写入仍必须限制在 AuthContext 的 Domain 内。

Agent/Target Binding、Target 状态、Policy、单命令审批、Assignee 和一次性执行授权属于业务及执行安全约束，不是可省略的 RBAC。审批操作必须由当前待审的 `asserted_user_id` 完成并留痕。未来加入细粒度权限时，可在 AuthContext 中增加版本化 Scope，不因产品版本升级改变现有 URL。

## HTTP、幂等与错误

- 配置创建返回 `201`；异步 Run、健康检查和事件接入返回 `202`；读取返回 `200`；
- `Idempotency-Key` 的作用域为调用主体、Domain、操作和资源；相同 Key 不同指纹返回 `409 IDEMPOTENCY_CONFLICT`；
- 乐观锁失败返回 `412 ROW_VERSION_CHANGED`；非法状态迁移返回 `409 OPS_STATE_CONFLICT`；
- 限流返回 `429`，暂时依赖不可用返回 `503`，但已经接收并持久化的任务仍返回 `202`；
- 错误使用 `application/problem+json`，稳定字段为 `type/title/status/code/detail/request_id/trace_id/retryable/field_errors`，不返回堆栈、SQL 或 Provider 原始错误。

所有跨服务 Command、Webhook、Executor 回调和上传完成通知都使用 Inbox/Outbox 保证至少一次传递下的业务幂等。`GET` 可以安全重试；未经幂等保护的变更请求不得由 Client 自动重试。

## 代码归属与验收

稳定 DTO 建议放置：

```text
platform_core/contracts/aiops/
  public.py       # Main API 对外请求/响应
  internal.py     # Root/Main 到 AIOps
  delegation.py   # Agent Runtime 的子 Run、事件页和 Result Envelope
  executor.py     # AIOps 到 DB Executor
  events.py       # SSE/Event DTO
  errors.py       # 稳定错误码
platform_clients/aiops_management.py
platform_clients/aiops_delegation.py
```

FastAPI Schema 只能映射这些 DTO，不能直接返回 Entity。OpenAPI 分别生成 Public、AIOps Internal 和 Executor 三份文档；Internal 文档不发布到外网。契约测试至少覆盖 API Key 与内部 JWT、Domain 隔离、ETag、幂等冲突、SSE 断点续传、Webhook 重放、HITL 非 Chat 拒绝、重复审批、过期 Token、Executor 回调乱序和跨 Domain 隐藏。

4.0 不复用 3.x Ops Controller 中的请求体用户/Domain 字段、请求内 Agent 实例、明文数据库密码或长连接内执行逻辑，也不提供对应兼容路由。
