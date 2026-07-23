# 4.0 Agent Runtime API 与状态迁移

本文定义 Run/Task 执行模型的命令接口和状态迁移边界。外部 API 只提交用户意图、查询结果或发出取消/审批命令；Task 状态和 Artifact 只能由 Agent Runtime、Scheduler 和 Worker 通过内部接口修改。

## 公开 API

| 方法 | 路径 | 作用 | 返回 |
| --- | --- | --- | --- |
| `POST` | `/api/v1/runs` | 创建一次 Agent Run | `202` + `run_id`、当前状态和事件游标 |
| `GET` | `/api/v1/runs/{run_id}` | 查询 Run 汇总和最终 Artifact 摘要 | Run DTO |
| `GET` | `/api/v1/runs/{run_id}/events` | 订阅事件流，支持 `Last-Event-ID` | SSE |
| `POST` | `/api/v1/runs/{run_id}/cancel` | 请求协作式取消 | Run 状态 |
| `POST` | `/api/v1/runs/{run_id}/approvals` | 处理 Agent Runtime 自有 Proposal | 审批结果和新事件游标 |
| `POST` | `/api/v1/runs/{run_id}/resume` | 从可恢复状态继续执行 | Run 状态 |

`POST /api/v1/runs` 必须携带 `Idempotency-Key`。相同 `domain_id + asserted_user_id + key` 且请求指纹一致时返回原 Run；指纹不同返回 `409 IDEMPOTENCY_CONFLICT`。API 使用 `202 Accepted`，不等待 LLM、KC 或外部系统完成。

API 响应只暴露 DTO 和 Artifact 引用，不暴露 SQLAlchemy Entity、内部租约、Worker 标识或完整策略快照。所有资源查询先由 `AuthContext` 限定 domain，再应用 Agent 与 Collection Binding；停用 Collection 不得被新 Run 使用。

创建请求体只提交用户意图和资源提示。门户通过受信请求头声明 `domain_id` 和 `user_id`，Main API 在 API Key 校验后将其写入 `AuthContext`；权限系统当前不实现，运行策略由服务端配置派生：

```json
{
  "agent_id": "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11",
  "input": "列出关于某案例的资产，并说明相关附件内容",
  "collection_ids": [
    "019c03b8-037c-76a1-827a-a11e628d6913",
    "019c03b8-5fac-7e39-8e75-9bd9f7b6cd18"
  ],
  "security_level": 3,
  "client_metadata": {"channel": "portal"}
}
```

服务端返回：

```json
{
  "run_id": "…",
  "status": "CREATED",
  "event_cursor": 0,
  "events_url": "/api/v1/runs/…/events"
}
```

客户端提供的 `collection_ids` 只是候选范围，最终范围必须与 Agent Binding、Collection 状态和用户授权求交集；不能通过请求体扩大权限。

AIOps 子 Run 的补证、审批和人工结果不走通用 Run Approval/Resume API。Root SSE 只返回 AIOps 资源引用，用户必须调用 `/api/v1/ops/hitl/*` 或 `/api/v1/ops/proposals/*` 的权威 Command；详见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

## 内部 Runtime 接口

Runtime 以命令接口接收状态变化，禁止 Controller 或 Skill 直接更新状态：

```python
class AgentRuntime:
    async def create_run(self, command: CreateRunCommand) -> RunReceipt: ...
    async def claim_task(self, command: ClaimTaskCommand) -> TaskLease | None: ...
    async def record_progress(self, command: ProgressCommand) -> None: ...
    async def complete_task(self, command: CompleteTaskCommand) -> ArtifactReceipt: ...
    async def fail_task(self, command: FailTaskCommand) -> None: ...
    async def request_cancel(self, command: CancelRunCommand) -> None: ...
    async def approve(self, command: ApprovalCommand) -> None: ...
    async def resume(self, command: ResumeRunCommand) -> RunReceipt: ...
```

每个 Command 都包含 `run_id`/`task_id`、调用者身份、`expected_row_version`、幂等键和 Trace 信息。Worker 完成/失败命令还必须携带本次领取生成的 `lease_token`。返回值只包含新状态、版本号、事件序号和 Artifact ID。

## 状态迁移规则

### Run

```text
CREATED        --start--> RUNNING
RUNNING        --wait--> WAITING_INPUT / WAITING_APPROVAL
WAITING_*      --resume/approve--> RUNNING
RUNNING        --complete--> COMPLETED
RUNNING        --fail--> FAILED
CREATED/RUNNING/WAITING_*
               --cancel--> CANCELLED
RUNNING        --deadline--> EXPIRED
```

### Task

```text
PENDING        --dependencies_ready--> READY
READY          --claim--> RUNNING
RUNNING        --success--> SUCCEEDED
RUNNING        --child_accepted--> WAITING_EXTERNAL
WAITING_EXTERNAL --child_success--> SUCCEEDED
WAITING_EXTERNAL --child_failure--> FAILED
RUNNING        --retryable_error--> RETRY_WAIT
RETRY_WAIT     --backoff_elapsed--> READY
RUNNING        --terminal_error--> FAILED
PENDING/READY/RUNNING/RETRY_WAIT
               --cancel--> CANCELLED
```

`SUCCEEDED`、`FAILED`、`CANCELLED` 和 `EXPIRED` 是终态。`WAITING_EXTERNAL` 只允许 Delegation Application Service 根据持久化子状态推进，普通 Worker 不能领取。`BLOCKED` 表示依赖或策略未满足，可由 Runtime 在修复依赖后重新变为 `READY`，但不能由 Worker 直接绕过策略进入 `RUNNING`。

## 并发控制和租约

状态更新使用 `WHERE id = :id AND row_version = :expected_version` 的条件更新；影响行数为 0 时返回 `409 STATE_VERSION_CONFLICT`，调用方必须重新读取，不允许盲目覆盖。

领取 Task 时只检查 `STATUS = 'READY'`，并原子写入 `RUNNING`、`LEASE_OWNER`、新的 `LEASE_TOKEN`、`LEASE_UNTIL` 和 `row_version`。普通 Claim 不直接窃取过期 `RUNNING`；Reconciler 先将其收敛到 `RETRY_WAIT/FAILED` 并清除旧 Token。Worker 心跳和完成必须同时匹配 Owner、Token 和有效期；旧 Worker 回写返回 `409 STALE_LEASE`，不得写入 Artifact 或终态。

进度事件可以重复发送，但必须使用 `(task_id, progress_sequence)` 幂等；完成、失败和取消命令必须使用逻辑幂等键，保证重试请求不会生成重复 Artifact 或重复后继 Task。

## 完成 Task 的原子操作

`complete_task` 在一个 Agent UoW 中执行：

1. 校验 Task 仍由当前 Worker 持有且处于 `RUNNING`；
2. 校验 Artifact 类型和 schema 与 Skill Manifest 匹配；
3. 插入不可变 Artifact；
4. 将 Task 更新为 `SUCCEEDED`；
5. 根据依赖条件把后继 Task 更新为 `READY`；
6. 写入 `ARTIFACT_CREATED` 和 `TASK_COMPLETED` 事件；
7. 若所有必要 Task 完成，推进 Run 并写入 `RUN_COMPLETED`。

任何一步失败都回滚，不发送成功事件。外部通知、SSE 推送和下一次 HTTP 调用在提交后异步执行。

## 错误和恢复语义

| 错误 | HTTP/内部码 | 处理 |
| --- | --- | --- |
| 无权限访问 Run | `404 RUN_NOT_FOUND_OR_DENIED` | 不泄露 Run 是否存在 |
| 幂等键请求体不同 | `409 IDEMPOTENCY_CONFLICT` | 要求调用方生成新键 |
| 状态版本过期 | `409 STATE_VERSION_CONFLICT` | 重新读取并重新规划 |
| Worker 租约过期 | `409 STALE_LEASE` | 丢弃本次结果，等待接管 |
| Skill 输出不符合 schema | `422 ARTIFACT_SCHEMA_INVALID` | Task 终止或按策略重试 |
| 超过预算/截止时间 | `429 RUN_BUDGET_EXCEEDED` / `408 RUN_DEADLINE_EXCEEDED` | 取消剩余 Task，生成错误 Artifact |

恢复只重放未完成的 Task；已经提交的 Artifact 作为输入继续使用。事件重放用于构建状态视图，不重新执行 Skill。

## 非目标

- 外部调用方不能直接创建或修改 Task；
- Skill 不能自行改变 Run 状态或发起任意后继 Task；
- 不将内部数据库事件直接暴露为稳定 API 字段，SSE 通过版本化事件 DTO 输出；
- 不为旧 `/api/kb`、旧 Agent SSE 或旧 `doc_results` 提供兼容映射。
