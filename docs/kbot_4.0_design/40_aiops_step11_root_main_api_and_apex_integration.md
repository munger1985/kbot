# 4.0 AIOps 步骤 11：Root Agent、Main API 与 APEX 集成

## 当前实施进度

阶段 11A/11B 已完成 AIOps 侧边界与 Agent Runtime 可恢复委派：

- `POST /internal/v1/aiops/delegations` 使用稳定 Delegation ID 幂等创建
  `diagnosis.root-cause` 子 Run，并冻结 `PARENT_AGENT_RUN_ID` 与
  `PARENT_DELEGATION_ID`；
- Events、Result 和 Cancel 都先按 `delegation_id + APP_ID + DOMAIN_ID`
  精确解析子 Run，并要求独立 `aiops.delegate` Service Scope；
- Events 只投影白名单字段，完整命令、原始 SQL 结果、内部 Task 图和未知 Payload
  字段不会穿透到 Root；
- Result 仅返回终态、安全摘要和不可变 Artifact 引用；诊断摘要只使用
  `supporting_fact_refs` 指向的 `fact_summary`，不复制原始行；
- Cancel 复用 AIOps Run 的版本围栏和权威状态机，不伪造子任务已停止；
- Root Planner 已支持配置驱动的 AIOps 单路由，生成
  `DELEGATION → COMPOSE` 两任务 DAG；
- Worker 在本地事务中创建 `SUBMITTING` Delegation、将 Task 转为
  `WAITING_EXTERNAL` 并释放租约；独立 Reconciler 使用有限租约完成提交、事件分页、
  终态结果和取消联动；
- Reconciler 使用稳定幂等键、持久化 Child Cursor 和父事件键恢复，不在数据库事务
  内执行 HTTP；
- Response Composer 已能把受限结果映射为 `O1` 类型化引用，并只公开安全摘要。

AIOps 丰富 Result Envelope、多来源并行 Composer、Main API/APEX 的完整交互体验
仍属于后续 11C。

## 目标与入口

本步骤将 AIOps 接入 Root Agent、Main API/BFF 和 APEX，同时保持 AIOps 对诊断、HITL、审批、执行和报告的完整所有权。

```text
Root 对话：POST /api/v1/runs
  Main API → Agent Runtime → AIOps Delegation → Root Composer

直接运维：POST /api/v1/ops/runs
  Main API → AIOps API

自动触发：POST /api/v1/integrations/monitoring/{webhook_key}/events
  Main API → AIOps Intake

定时巡检：AIOps Scheduler → Inspection Fire → Ops Run
```

Root 对话只订阅 `/api/v1/runs/{run_id}/events`；直接运维只订阅 `/api/v1/ops/runs/{ops_run_id}/events`。前端不需要同时拼接两条 SSE。

## 对 3.5 过渡实现的处理

以下代码只作为行为和测试样本，不进入 4.0 Runtime：

- `agent/agent/root_agent_v2.py` 在请求内直接执行并生成 SSE，没有 Run/Task、租约或恢复；
- 3.5 Agent V2 Router 使用进程级 Agent 实例，并接受客户端提供的 Domain/Security 范围；
- `knowledge_core/application/answer_generation.py`、`grounding.py` 和 `sse_v2.py` 把最终回答职责放进了 KC 包。

可复用的是 Citation Label、回答后引用校验和 Document Card 投影规则。实现迁入 `agent_runtime/specialists/response_composer/`；Knowledge Core 只保留 Citation/Evidence 契约，不 import Agent、回答模型或 SSE。

## 一等 Delegation 模型

跨服务子 Agent 不能只保存在 Python Future 或 JSON Artifact 中。新增 Agent Runtime 表 `KBOT_AGENT_DELEGATION`：

```text
DELEGATION_ID, PARENT_RUN_ID, PARENT_TASK_ID
TARGET_SERVICE, TARGET_CAPABILITY
CHILD_RUN_ID
IDEMPOTENCY_KEY
STATUS: CREATED | SUBMITTING | RUNNING |
        WAITING_INPUT | WAITING_APPROVAL |
        COMPLETED | DEGRADED | FAILED |
        CANCEL_REQUESTED | CANCELLED | EXPIRED
LAST_CHILD_EVENT_SEQUENCE, NEXT_POLL_AT
RESULT_ARTIFACT_ID
ATTEMPT_COUNT, MAX_ATTEMPTS
LEASE_OWNER, LEASE_TOKEN, LEASE_UNTIL
ERROR_CODE, ERROR_MESSAGE, ROW_VERSION
CREATED_AT, UPDATED_AT, COMPLETED_AT
```

约束：

- `PARENT_TASK_ID` 唯一，一个 Delegate Task 只创建一个子 Run；
- `(TARGET_SERVICE, IDEMPOTENCY_KEY)` 唯一；
- 非空 `(TARGET_SERVICE, CHILD_RUN_ID)` 唯一；
- Parent Run/Task/Result Artifact 建 Agent Runtime 内部外键；
- Child Run 是跨服务 UUIDv7 引用，不建数据库外键。

`KBOT_AGENT_TASK` 增加 `WAITING_EXTERNAL`。提交子 Run 后释放 Task 租约并进入该状态，不能让 Worker 持有长租约等待 AIOps。Delegation Reconciler 使用自己的有限租约和 `NEXT_POLL_AT` 继续推进。

## AIOps 侧父子关联

`KBOT_OPS_RUN` 增加可空、非空时唯一的 `PARENT_DELEGATION_ID`。Root 创建子 Run 时提交：

```text
delegation_id
parent_agent_run_id
caller_mode: ROOT_DELEGATION
agent_id, target_id, input
deadline_at, budget
```

`PARENT_AGENT_RUN_ID` 用于会话级追踪，`PARENT_DELEGATION_ID` 精确区分同一 Root Run 的多个子 Run。两者都是跨服务 UUID，不建外键。

Agent Runtime 使用窄 `AIOpsDelegationClient`，只能：

```text
create_delegated_run
read_delegation_events
read_delegation_result
request_delegation_cancel
```

Main API 使用独立 `AIOpsManagementClient` 代理 Target、Run、HITL、Proposal、Report 等用户 API。Root 的 Client 类型中不提供审批、人工结果、配置或 Executor 方法，从代码层阻止越权调用。

## 委派事务与不确定结果

Root 委派流程：

1. Route Task 产生不可变 `ROUTE_DECISION.v1`；
2. Planner 在 Agent UoW 中创建 Delegate Task；
3. Worker 领取 Task，在同一事务创建 `SUBMITTING` Delegation、将 Task 置为
   `WAITING_EXTERNAL` 并释放 Worker 租约；
4. Reconciler 使用独立有限租约，在事务外以稳定幂等键调用 AIOps；
5. 接收 `ops_run_id` 后，在一个事务中保存 Child Run ID 和游标，并将 Delegation
   置为 `RUNNING`；
6. Reconciler 分页读取子事件并更新持久化游标；
7. 子 Run 终止后获取最终 Result Envelope，写入 Root Artifact，完成 Delegate Task
   并释放 Compose Task。

HTTP 超时并不代表创建失败。Delegation 保持 `SUBMITTING`，使用相同幂等键重试或查询接收结果，不能创建第二个子 Run。远程调用、事件读取和结果读取都在数据库事务外；游标与状态更新使用 Lease Token + Row Version fencing。

## 子事件投影，而非 SSE 转发

AIOps 提供面向 Delegation 的内部分页接口：

```text
GET /internal/v1/aiops/delegations/{delegation_id}/events
    ?after_sequence=<n>&limit=<bounded>
GET /internal/v1/aiops/delegations/{delegation_id}/result
POST /internal/v1/aiops/delegations/{delegation_id}/cancel
```

AIOps 必须验证 `delegation_id + child_run_id + Root Service Identity` 的精确关联。后台读取不依赖已经过期的用户 Access Token，但只能获得 `VISIBILITY=USER` 的安全投影；用户通过 Main API 查看资源时仍重新执行当前权限校验。

Agent Runtime 不保持到 AIOps 的长 SSE，也不复用子事件序号作为父事件序号。每个子事件按：

```text
event_key = "delegation:<delegation_id>:child-event:<child_sequence>"
```

幂等追加到 `KBOT_AGENT_RUN_EVENT`，再推进 `LAST_CHILD_EVENT_SEQUENCE`。重复页、进程崩溃和多 Reconciler 不会产生重复父事件。

只投影：

- `delegation.started/status/completed/failed`；
- 阶段级、已脱敏 `delegation.progress`；
- `interaction.required`：HITL ID、类型、过期时间和授权资源 URL；
- `approval.required`：Proposal/HITL ID、风险、过期时间和授权资源 URL；
- `report.ready`：Report ID、类型、版本和安全摘要。

不投影 AIOps 内部 Task 图、Prompt、原始 SQL 结果、完整命令、Policy 快照、监控 Payload 或内部错误。

## 父子状态映射

| AIOps 子状态 | Delegation | Root Run |
| --- | --- | --- |
| Created/Observe/Diagnose/Execute/Verify | `RUNNING` | `RUNNING` |
| `WAITING_INPUT` | `WAITING_INPUT` | `WAITING_INPUT` |
| `WAITING_APPROVAL` | `WAITING_APPROVAL` | `WAITING_APPROVAL` |
| `COMPLETED` | `COMPLETED` | 等待其他必要分支或 Compose |
| `DEGRADED` | `DEGRADED` | 允许 Compose，但必须披露缺口 |
| Failed/Cancelled/Expired | 同名终态 | 按分支必要性决定失败或部分回答 |

Route Target 增加 `completion_requirement: REQUIRED | OPTIONAL`。唯一必要分支失败时 Root Run 失败；混合请求中的可选或可降级分支失败时，可以生成 `PARTIAL` Grounded Answer，但不能隐藏缺失来源。Root 状态只反映用户体验，AIOps 状态仍是权威事实。

父状态由 Runtime 在锁定 Run 后聚合计算，而不是直接复制最后一个子事件：仍有 `READY/RUNNING` Task 时保持 `RUNNING`；没有可运行 Task 且存在待补证时为 `WAITING_INPUT`；否则存在待审批时为 `WAITING_APPROVAL`。任一等待解除后重新计算，避免并行分支互相覆盖状态。

## HITL、审批与 SSE

Root 投影事件只通知“存在待处理资源”，不承接命令：

- SQL 补证：先 `GET /api/v1/ops/hitl/{hitl_id}` 获取完整 SQL/Schema，再调用 Responses/Skip API；
- 变更审批：先 `GET /api/v1/ops/proposals/{proposal_id}` 查看精确命令、风险和 Hash，再调用 Approve/Reject；
- Advisory 人工结果：调用 Proposal 的 `manual-result` API。

Root 的普通聊天文本、`POST /api/v1/runs/{id}/resume` 或通用 Runtime Approval API 都不能替代这些 AIOps Command。Root Agent、Composer 和 LLM 没有批准权限。用户提交后由 AIOps 继续子 Run，Reconciler 再把恢复状态投影到父 Run。

这也意味着 SSE 中必须让用户知道待执行 SQL/命令存在，但正文只通过授权 GET 返回，不能进入 SSE、APEX View 或日志。

## 取消、截止时间与权限变化

Root 创建子 Run 时将子 Deadline 限制为不晚于父 Deadline。父取消执行：

1. 原子记录 Root Cancel Request，取消尚未开始的本地 Task；
2. 将 Delegation 置为 `CANCEL_REQUESTED` 并写 Outbox；
3. 提交后调用 AIOps Cancel；
4. 等待子 Run 确认终态，再收敛父 Run。

正在执行的数据库命令可能无法安全中断；取消只阻止后继 Task，不能伪造“命令未执行”。若 AIOps 暂不可达，Reconciler 继续有限重试并保持可见的 Cancel Pending 状态。

创建时冻结的授权快照支持后台完成，但所有 HITL、审批、报告、Artifact 和 SSE 读取仍检查当前 AuthContext。用户或 Target Binding 被撤销后，前端统一得到 `404 *_NOT_FOUND_OR_DENIED`；Root Service 的后台事件读取权不能转授给用户。

## AIOps Result Envelope

子 Run 终态后，AIOps 返回受限、版本化 `AIOPS_DELEGATION_RESULT.v1`：

```text
ops_run_id, status, target_ref
root_cause_level, diagnosis_summary
verified_facts[]: fact_label, text, evidence_refs[]
data_gaps[], confidence
proposal_refs[]: id, display_name, risk, status
execution_refs[]: id, status, verification_result
comparison_refs[]: report_id, result
final_report_ref
security_level, content_hash, generated_at
```

Envelope 不包含完整命令、凭据、未脱敏结果或对象存储 URI。Agent Runtime 将其保存为不可变 `DELEGATED_AIOPS_RESULT.v1` Artifact，Provenance 记录 Child Run、远端内容 Hash 和事件终点；不把 `KBOT_OPS_ARTIFACT` 外键直接写入 Agent 表。

## Response Composer

`Response Composer` 是此前讨论中 `Conversation Composer` 的正式名称，避免误解为它只处理闲聊。它是 Agent Runtime 中的确定性输入整理、受限 LLM 叙述和 Grounding 组合，不是独立服务或可调用工具的 Agent。

```text
COMPOSITION_INPUT.v1
  original_input
  route_decision_ref
  source_envelopes[]
    DOCUMENT: CITATION_PACK
    DATA: QUERY_RESULT
    AIOPS: DELEGATED_AIOPS_RESULT

GROUNDED_ANSWER.v1
  answer_markdown
  claims[]: claim_id, text, source_labels[]
  references[]: typed reference card
  source_status[]
  grounding_status: VERIFIED | PARTIAL | INSUFFICIENT
```

不同来源使用独立标签命名空间，例如 Document `D1`、Data `Q1`、AIOps Fact `O1`。Verifier 分别校验：

- `D*` 必须属于 Citation Pack 且引用实际采用的 Evidence；
- `Q*` 必须定位到 Query Result 的列、行或聚合结果；
- `O*` 必须属于 AIOps Envelope 的 Verified Fact、Execution 或 Report 引用。

Composer 可以改善表达，但不能改变 AIOps 的根因等级、Target/环境、风险、命令/审批/执行状态、Verification 或 Comparison 结果。精确命令永远不进入合成 Prompt。模型输出无效时，确定性 Presenter 仍能生成来源分区的安全摘要。

4.0 最终 `references[]` 是 `DOCUMENT/DATA/AIOPS` 的判别联合，不再输出 `doc_results` 或 `doc_results_v2`。只有回答实际使用且验证通过的来源才进入 Reference Card。

## Main API/BFF

Main API 是外部唯一入口，但不是透传代理：

- 校验门户后端 API Key，并把门户声明的 Domain 和操作人写入短期内部 AuthContext JWT；
- 根据路径调用 Agent Runtime、AIOps Management 或 Monitoring Intake Client；
- 将 Public DTO 映射为 Internal DTO，重新生成授权资源 URL；
- 删除客户端伪造的内部身份头，且不把门户 API Key 转发到下游；
- 统一限流、请求大小、Idempotency Key、ETag、Trace 和 Problem Details；
- 不读取下游 Repository，不在请求线程执行 LLM、监控查询或 SQL。

Direct Ops 与 Root Run 使用不同资源 ID 和路径，不提供含糊的自动识别接口。Main API 不允许通过 Root Run Command 修改 AIOps HITL/Proposal。

Monitoring 路由保留原始请求字节和签名头，在请求大小/速率/路由 Key 检查后交给 AIOps 验签、映射和 Inbox 持久化。Main API 和反向代理均不得解析后重新序列化 Payload，也不得记录原始 Body。只有 AIOps 持久化成功才返回 `202`。

## SSE 传输规则

Main API 从所属服务的持久化 Event 表分页读取并输出：

```text
id: <run-local sequence>
event: <versioned public event type>
data: <public event dto>
```

- `Last-Event-ID` 只在当前 Run 内解释；大于当前游标返回 `400 EVENT_CURSOR_INVALID`；
- 游标早于可恢复保留窗口返回 `410 EVENT_CURSOR_EXPIRED`，客户端改用 Run GET 获取当前/终态；
- 无业务事件时发送不带 ID 的注释心跳，不能伪造进度事件；
- 每页有界读取并应用写超时，慢客户端断开后靠游标恢复；
- 终态持久化事件发送后再发送传输级 `done` 并关闭；
- Token 不进入 Query String；APEX 使用同源安全 Cookie，其他客户端使用支持 Header 的 Fetch Stream。

Main API 不缓存完整事件流，也不把 AIOps 子 SSE 直接管道转发到 Root SSE。SSE 断开不会取消 Run。

## APEX 页面边界

| 页面 | 只读列表来源 | 详情与写操作 |
| --- | --- | --- |
| Target/Monitor | `KBOT_V_OPS_TARGET/MONITOR_SOURCE` | Main API 配置与 Health Check |
| 运维 Run | `KBOT_V_OPS_RUN` | Run GET、SSE、Cancel |
| 待补证 | `KBOT_V_OPS_CHAT_PENDING` | HITL GET、Upload、Response/Skip |
| 待审批 | `KBOT_V_OPS_PENDING_APPROVAL` | Proposal GET、Approve/Reject |
| 巡检 | Plan/Fire View | Plan Command、Fire GET |
| 报告 | `KBOT_V_OPS_REPORT` | Report GET、Versions |

View 始终包含 `APP_ID/DOMAIN_ID` 过滤字段，只提供安全摘要，不授予 DML。完整 SQL、命令、Evidence、配置 JSON 和 Report 内容走 API。APEX 写请求必须携带 CSRF、Idempotency Key，并在配置/状态变更时携带 `If-Match`。

## 推荐代码布局

```text
agent_runtime/
  application/delegations/
    create.py
    reconcile.py
    cancel.py
    event_projection.py
  specialists/response_composer/
    contracts.py
    prompt_builder.py
    grounding.py
    presenter.py
  entities/delegation.py
  repositories/delegations.py
  clients/aiops_delegation.py

main_api/
  controllers/runs.py
  controllers/ops/
  controllers/integrations/monitoring.py
  application/sse_proxy.py
  mappers/aiops.py

platform_core/contracts/
  agent/delegation.py
  agent/composition.py
  aiops/delegation.py
platform_clients/
  aiops_delegation.py
  aiops_management.py
```

## 验收场景

- Root 创建 AIOps 子 Run 时 HTTP 超时/重试仍只有一个 Child Run；
- 同一 Root Run 可并行委派多个子 Run，并以 Delegation ID 精确关联；
- Runtime/AIOps/Main API 任意重启后，从持久化 Child Cursor 恢复且不重复父事件；
- Root SSE 只使用父序号，Direct Ops SSE 只使用子序号；
- AIOps 等待 SQL/审批时 Root 正确展示资源，但聊天文本和 Root API 不能完成操作；
- 审批、补证后无需重建父 Run即可继续并最终 Compose；
- 父取消、子不可达、命令不可中断和 Deadline 到期均保留真实状态；
- Document/Data/AIOps 混合回答使用独立引用类型，不发生来源冒充；
- Composer 不能改变根因、风险、审批、执行和 Comparison 结论；
- 用户撤权后不能读取 SSE/HITL/Proposal/Report，Root 后台权限不泄露；
- APEX View 不含敏感正文，所有写操作经过 CSRF、ETag 和幂等校验；
- Monitoring Payload 不被重序列化或写入普通日志；
- 不 import 3.5 RootAgentV2、旧 Controller、旧 SSE 或 KC 中的回答生成实现。

## 完成定义

步骤 11 完成时，Direct AIOps Chat、Root 委派、Alert 和 Schedule 四类入口使用同一 AIOps 内核；Root 委派可跨进程恢复，SSE 只有一个权威游标；HITL/审批仍由 AIOps 掌控；最终混合回答和引用经过类型化 Grounding；APEX 只读投影和 API Command 边界清晰。
