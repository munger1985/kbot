# 4.0 AIOps 步骤 4：确定性 Run 执行内核

## 实施结果

步骤 4 已完成。当前生产 Registry 固定
`kernel.observe-report@1`，按 `SCOPE → OBSERVE → REPORT` 生成
`SCOPE_RESULT.v1`、`OBSERVATION_SET.v1` 和
`KERNEL_TEST_REPORT.v1`。Handler 只接收不可变 `TaskLease` 与 Artifact DTO，
不持有 Session，也不调用 LLM、监控、Knowledge Core、Secret Store 或目标数据库。

运行内核已接入 AIOps Internal API、Main API
`/api/v1/ops/runs`、取消命令和支持 `Last-Event-ID` 的 SSE。AIOps Worker
实际启动多并发 Task Worker、Reconciler 和 Outbox Dispatcher，不再是探针骨架。
真实 Oracle Smoke 已验证三任务完成、双 Worker 单租约、运行中取消收敛、租约过期
接管、旧 Token 回写拒绝、并发幂等创建、连续事件和测试数据清理。

实现时额外修正了两个数据库边界：

- 所有 Oracle Session 固定为 `+00:00`，`UniversalTimestamp` 只接受 aware
  datetime 并统一按 UTC 绑定和读取；DDL 默认时间使用
  `CURRENT_TIMESTAMP`，避免数据库主机时区改变 Deadline 与租约语义。
- `(INSPECTION_FIRE_ID, TARGET_ID)` 改为仅在 Fire ID 非空时生效的函数唯一
  索引，普通 Chat/API Run 可在同一 Target 上并存。

## 目标与非目标

本步骤实现可多副本运行、可崩溃恢复的 `OpsRun → OpsTask → Artifact → RunEvent` 内核。只使用确定性 Blueprint 和测试 Handler，不调用 LLM、Monitor、Knowledge Core、Secret Store 或目标数据库；步骤 5 以后只新增 Handler/编排，不允许绕过本内核自行维护状态。

本内核不是通用 BPMN，也不是事件溯源系统。Run/Task 表保存权威当前状态，Run Event 是只追加的审计与 SSE 流；恢复读取当前状态和不可变 Artifact，不通过重放事件重新执行任务。

## 最小确定性 Blueprint

首个 Blueprint 固定为：

```text
kernel.observe-report.v1

SCOPE ──→ OBSERVE ──→ REPORT
```

三个 Handler 只对类型化输入做确定性转换：

- `SCOPE`：冻结 Target、Trigger、Policy/Config Snapshot，输出 `SCOPE_RESULT.v1`；
- `OBSERVE`：消费 Scope Artifact，输出无外部数据的 `OBSERVATION_SET.v1`；
- `REPORT`：消费前两项，输出 `KERNEL_TEST_REPORT.v1`。

Blueprint Registry 使用固定 `(blueprint_id, version) → TaskSpec[]` 映射。Application Service 先在 Target Scope 内校验 Active Agent Binding，并把 `agent_id`、Binding/Policy/Target 版本写入不可变 Snapshot；再校验 Task Key 唯一、依赖存在、无环、输入输出 Schema 匹配、Task 数量/并行度/截止时间合法，在一个事务内写 Run、全部 Task、初始事件和 Outbox。只有无依赖 Task 初始为 `READY`，其余为 `PENDING`。

每个持久化 Task 冻结 `handler_id/handler_version/input_schema_version/output_schema_version/timeout_seconds/max_attempts`。Worker 只能从 Registry 精确解析该版本，不允许按当前最新实现隐式升级历史 Task。

Chat/API/Root Delegation 使用请求上下文中已经授权的 `agent_id`；Alert/Schedule 没有终端用户选择，必须使用服务配置中的 `system_aiops_agent_id` 并验证其 Active Target Binding。Webhook 或 Job Payload 不能指定 Agent。首版一个部署只有一个 System AIOps Agent；未来多 AIOps Profile 通过显式 Routing Policy 扩展，不做“任选一个 Binding”。

步骤 4 只支持 `ALL_SUCCEEDED` 依赖条件。可降级 Provider 必须在后续 Handler 内生成 Gap/Partial Artifact 并成功结束，而不是让 Runtime 猜测业务失败是否可忽略。

## Run 状态

终态为 `COMPLETED/DEGRADED/REJECTED/FAILED/CANCELLED/EXPIRED`，不可再次迁移。步骤 4 使用以下子集：

```text
CREATED --claim SCOPE--> SCOPING
SCOPING --scope success--> OBSERVING
OBSERVING --required tasks success--> COMPLETED

CREATED/SCOPING/OBSERVING --fatal failure--> FAILED
任意非终态 --deadline--> EXPIRED
任意非终态 --cancel requested--> draining --> CANCELLED
```

`draining` 不是数据库 Status。Run 写入 `CANCEL_REQUESTED_AT/BY` 后停止释放和领取新 Task；待当前租约结束、Worker 确认取消或 Reconciler 接管后进入 `CANCELLED`。

Run Status 是用户可见阶段投影，不要求与每个 Task 一一对应。后续 Task Type 到阶段的固定映射为：

| Task Type | Run 阶段 |
| --- | --- |
| `SCOPE` | `SCOPING` |
| `OBSERVE` | `OBSERVING` |
| `DIAGNOSE` | `DIAGNOSING` |
| `PROPOSE` | `PROPOSING` |
| `APPROVE` | `WAITING_APPROVAL` |
| `EXECUTE` | `EXECUTING` |
| `VERIFY/COMPARE` | `VERIFYING` |
| `REPORT` | 保持当前阶段，完成时决定终态 |

阶段只能由 Runtime 根据已验证 Task 类型推进，Handler 不提交 Run Status。

## Task 状态与依赖

```text
PENDING --all dependencies succeeded--> READY
READY --claim--> RUNNING
RUNNING --success--> SUCCEEDED
RUNNING --retryable failure--> RETRY_WAIT --due--> READY
RUNNING --terminal failure--> FAILED
PENDING/READY/RETRY_WAIT --upstream terminal failure--> BLOCKED
非终态 --cancel/deadline--> CANCELLED/EXPIRED
```

步骤 4 中 `BLOCKED` 对当前 Blueprint Version 是终态；重新规划必须创建新 Task/Plan Version，不能原地修改依赖。后继释放在完成 Task 的同一事务内执行，查询其全部前置 Task；禁止依赖 Python 内存计数。

当 required Task 进入 `FAILED/BLOCKED/EXPIRED`，Runtime 原子阻断所有不可达后继，并将 Run 终止。`CANCELLED` 由用户取消产生，不伪装成失败。

## Claim 与租约 fencing

所有可领取表增加 `LEASE_TOKEN RAW(16)`。Token 在每次领取时重新生成，用于隔离同一 Worker ID 重启和迟到结果；`LEASE_OWNER` 只用于观测，不能单独证明所有权。

Task Claim 使用 Oracle 数据库时间：

1. 只读选择少量 `READY AND AVAILABLE_AT <= SYSTIMESTAMP` 候选；
2. 按候选逐个 `FOR UPDATE SKIP LOCKED` 锁 Run，再锁 Task 并重新校验状态、取消和 Deadline；
3. 写 `RUNNING`、新的 Lease Token、Owner/Until/Heartbeat、`ATTEMPT_COUNT+1`、`ROW_VERSION+1`；
4. 写 `task.status` Event 后 Commit；
5. 返回包含 `task_id/lease_token/attempt/deadline/handler_and_schema_versions/input_artifact_refs` 的 `TaskLease`。

普通 Claim 不接管过期 `RUNNING`。Reconciler 使用同样的 `SKIP LOCKED` 扫描过期租约，清除旧 Token，并根据 Attempt/错误策略转为 `RETRY_WAIT` 或 `FAILED`；之后才能再次领取。

Heartbeat 条件为：

```text
TASK_ID + STATUS=RUNNING + LEASE_OWNER + LEASE_TOKEN
+ LEASE_UNTIL >= SYSTIMESTAMP
```

Heartbeat 只延长租约和更新时间，不推进业务状态。延长上限不能超过 Task Timeout、Run Deadline 或部署配置的最大租约。所有完成/失败命令必须携带 Lease Token；不接受只有 Worker ID 的写回。

Inspection Plan 和 Outbox 同样增加 Lease Token，使用相同 fencing 规则；实现可复用 SQL/Protocol，但各 Repository 仍归自己的聚合。

## 锁顺序

任何同时修改 Run 与 Task 的事务统一按：

```text
Run → Task（多个时按 TASK_ID 排序）→ Artifact/Outbox Insert → Run Event
```

Claim 的候选查询不持有锁，真正领取时按上述顺序重查；Completion、Failure、Cancel 和 Reconciler 也先锁 Run。这样避免 Claim 使用 `Task → Run`、Cancel 使用 `Run → Task` 造成循环等待。领取事务虽然会短暂串行化同一 Run，但 Commit 后不同 Task 可并行执行。

Plan Scheduler 使用 `Inspection Plan → Inspection Fire → 新 Run/Task Insert`；Outbox Dispatcher 只锁 Outbox，不反向锁领域聚合。Repository 方法不能私自改变锁顺序。Oracle Deadlock/Lock Timeout 映射为可重试基础设施错误，并保留限次退避，不能把它误报为业务失败。

## Worker 执行协议

```text
claim → load immutable TaskLease/Input Artifact
      → validate Handler Registry + schema
      → execute deterministic Handler outside transaction
      → complete_task / fail_task in a new UoW
```

Worker 不把 SQLAlchemy Entity、Session 或 UoW 交给 Handler。`TaskExecutionContext` 仅包含 UUID、已冻结 Scope/Policy/Config、Deadline、Attempt、Trace 和取消探针。

取消探针通过短查询或心跳响应读取，不依赖进程内 Event。Handler 必须在开始、长循环边界和输出提交前检查；超过 Deadline 本地停止，但最终状态仍由数据库条件更新决定。

Handler 异常映射为稳定 Error：

```text
HANDLER_NOT_FOUND
INPUT_SCHEMA_INVALID
OUTPUT_SCHEMA_INVALID
HANDLER_TIMEOUT
HANDLER_RETRYABLE_FAILURE
HANDLER_TERMINAL_FAILURE
```

原始异常和堆栈只进入受控日志，不写 Event/Error Message。

## Artifact 原子提交与幂等

每个 Task 输出使用稳定 `artifact_key`：

```text
task:{task_id}:output:{output_name}:v{schema_version}
```

`complete_task` 在一个 UoW 中：

1. 依次锁定并校验 Run、Task、有效 Lease Token、Deadline 和取消标记；
2. 校验 Artifact Type/Schema/Producer 与 Handler Manifest；
3. 按规范序列化计算 SHA-256；
4. 插入不可变 Artifact；
5. Task 条件更新为 `SUCCEEDED`，清除租约；
6. 释放满足依赖的后继 Task；
7. 推进 Run 阶段或终态；
8. 追加 Artifact/Task/Run Event 和必要 Outbox；
9. Commit 后返回 Artifact Receipt。

唯一 `(ops_run_id, artifact_key)` 防止重试生成重复结果。网络丢失后重复 `complete_task`：

- 已存在相同 Artifact Key + Hash + Producer：返回原 Receipt；
- Key 相同但 Hash/Schema 不同：`ARTIFACT_IDEMPOTENCY_CONFLICT`；
- Task 已被新 Lease 接管且尚未成功：`STALE_LEASE`。

Artifact 内容不可更新。步骤 4 只允许受限小 JSON；对象存储暂用 Fake Port，后续大内容必须先写临时对象并在事务提交后发布/清理。

## 失败、重试与 Backoff

Handler 只能报告结构化 Error Code；是否可重试由 `TaskPolicy + ErrorCatalog` 共同决定，不能完全相信 Handler 的布尔值。

```text
delay = min(base * 2^(attempt-1), max_delay) + deterministic_jitter
```

Jitter 由 `task_id + attempt` 计算，保证测试和恢复可复现。`AVAILABLE_AT` 使用数据库时间计算。只有 Manifest 声明幂等、Error Catalog 允许且 Attempt 未达上限时进入 `RETRY_WAIT`；否则进入 `FAILED`。

`ATTEMPT_COUNT` 在 Claim 时增加，所以 Worker 崩溃也消耗一次 Attempt。错误历史通过 Run Event/日志保存；Task 行只保存最近稳定 Error Code 和脱敏摘要。

## 取消与截止时间

`request_cancel` 在一个事务内：

1. 校验 AuthContext、Run Scope 和 ETag/幂等键；
2. 写 `CANCEL_REQUESTED_AT/BY`；
3. 将 `PENDING/READY/RETRY_WAIT` Task 改为 `CANCELLED`；
4. 为 Running Task 追加取消请求 Event；
5. 若无 Running Task，立即将 Run 改为 `CANCELLED`。

取消请求与 Task 完成竞争时，以先获得行锁并满足条件的一方为准：完成先提交则保留成功 Artifact，取消继续处理剩余 Task；取消先提交则 Completion 返回 `RUN_CANCEL_REQUESTED`，不得写新 Artifact。

Deadline Reconciler 使用相同逻辑将 Run/Task 置为 `EXPIRED`。应用时间只能用于展示，所有租约、Retry Due 和 Deadline 比较使用数据库 `SYSTIMESTAMP`。

## Run Event 与 SSE

Run Event 使用复合主键 `(OPS_RUN_ID, SEQUENCE_NO)`，增加可空 `EVENT_KEY`。有幂等要求的事件使用唯一 `(OPS_RUN_ID, EVENT_KEY)`：

```text
task:{task_id}:claimed:{attempt}
task:{task_id}:progress:{progress_sequence}
task:{task_id}:completed
run:{run_id}:terminal
```

写事件时先锁定 Run，再取该 Run 当前最大 Sequence + 1；同 Run 严格有序，不同 Run 可并行。Event 与状态/Artifact 在同一事务提交，事件不会描述未提交事实。

SSE 接口：

```text
GET /internal/v1/aiops/runs/{run_id}/events?after={sequence}
GET /api/v1/ops/runs/{run_id}/events
Last-Event-ID: <sequence>
```

- Main API 建连前和 AIOps 查询时都校验 Domain/Target/Agent Binding；
- 只输出 `VISIBILITY=USER` 的版本化 DTO；
- 先回放 `sequence > cursor`，再短轮询新事件；SSE 连接不是任务触发器；
- Cursor 大于当前最大值返回 `409 OPS_EVENT_CURSOR_INVALID`；
- 已归档 Cursor 返回 `410 OPS_EVENT_CURSOR_EXPIRED` 并提示读取 Run Snapshot；
- 终态 Event 后发送 `done`，重连仍能重复读该终态 Event。

Progress 事件限频、限大小，只包含阶段、百分比/计数和 ArtifactRef；不包含 Prompt、原始结果、Secret、内部错误或 Task Lease。

## Outbox 与进程模型

状态事务只写 Outbox，不在 Commit 前唤醒远端。步骤 4 的 Dispatcher 对 Fake Sink 验证至少一次投递、Lease Token、Backoff 和重复消费；Worker 本身可以轮询 READY Task，因此 Outbox 丢失不会成为唯一调度信号。

进程职责：

```text
AIOps API       Create/Get/Cancel Run，读取 Event
AIOps Worker    Claim/Heartbeat/Execute/Complete/Fail
Reconciler      Lease、Retry、Deadline、Cancellation 收敛
Dispatcher      Outbox 至少一次交付
```

Reconciler 可运行在 Worker 进程的独立循环中，后续可拆部署；任何循环都必须有批量上限、抖动和数据库错误退避。

## Internal Command

```text
CreateOpsRunCommand
ClaimOpsTaskCommand
HeartbeatOpsTaskCommand
CompleteOpsTaskCommand
FailOpsTaskCommand
CancelOpsRunCommand
ReconcileExpiredLeaseCommand
ReconcileDeadlineCommand
```

Command 使用类型化 DTO，不接受任意状态字符串。Create/Cancel 等跨服务命令带 AuthContext 和 Idempotency Key；Worker 命令带 Service Identity、Lease Token 和 Trace。Repository 只提供条件查询/更新原语，不暴露 `set_status(value)`。

## 代码布局

```text
aiops_agent/domain/operations/
  run.py
  task.py
  transitions.py
  errors.py
aiops_agent/application/runtime/
  service.py
aiops_agent/orchestration/blueprints.py
aiops_agent/workers/{task_worker,reconciliation,outbox_dispatcher}.py
aiops_agent/contracts/artifacts/kernel.py
tests/test_aiops_runtime_kernel.py
scripts/smoke_aiops_runtime.py
```

状态迁移表由 Domain 单点定义；API Schema、Worker 和 Repository 不复制允许迁移
集合。Application Service 作为事务 Facade 保持统一入口，后续业务 Handler 增加时
不扩张其状态迁移职责。Logging Outbox Sink 仅用于当前开发部署；步骤 5 接入的业务
Adapter 仍通过同一 Outbox Port 交付。

## 故障注入与验收矩阵

- 两个 Worker 并发 Claim 同一 Task，仅一个获得 Lease；
- Claim Commit 后进程崩溃，租约到期后只产生一次有效接管；
- 旧 Worker 在新 Lease 后回写，返回 `STALE_LEASE` 且无 Artifact/Event；
- Artifact Insert、Task Success、后继 Ready、Run/Event 任一步失败时全部回滚；
- Completion 响应丢失后重试返回同一 Artifact；
- 相同 Artifact Key 不同 Hash 被拒绝；
- Heartbeat、Reconciler 和 Completion 竞争不产生双终态；
- Cancel 与 Completion 两种锁顺序均收敛且无后继 Task 被错误释放；
- Deadline、Max Attempts、Retry Backoff 使用数据库时间并可确定复现；
- SSE 断线续传无缺口，重复连接最多重复已提交事件；
- Dispatcher 重复投递不产生重复业务动作；
- API/Worker/Reconciler/Dispatcher 分别重启后 Run 仍可完成。

## 完成定义

- 固定 Blueprint 可从 `CREATED` 运行到 `COMPLETED/FAILED/CANCELLED/EXPIRED`；
- 多副本 Worker 不重复提交 Artifact，迟到结果受 Lease Token fencing；
- Run/Task/Artifact/Event/Outbox 在事务失败和进程崩溃时保持一致；
- SSE 可按 Sequence 恢复且不泄露内部事件；
- Handler 不持有 Session、不修改状态、不创建后继 Task；
- 本步骤的生产代码没有 LLM、Monitor、KC、Secret Store 或目标数据库调用。
