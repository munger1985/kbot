# 4.0 AIOps Agent 领域模型、流程与 DB Executor

## 设计目标

AIOps Agent 面向数据库监控与运维，覆盖“感知 → 诊断 → 决策 → 审批 → 执行 → 验证 → 报告”。它拥有独立 `KBOT_OPS_*` 表和流程状态，不复用 3.x 的完整 Context 快照、动态 SkillRuntime 或在实例表中保存数据库密码。

关键规则：

- 诊断和变更使用不同权限与 Service Identity；
- LLM 只能生成结构化诊断和 Action Draft；ChangeProposal 由确定性 Catalog/Policy/Hash 流程构建，不能直接执行自然语言或任意 SQL；
- 凭据只保存 Secret 引用，由 DB Executor 在执行时解析；
- HITL 只保存请求、schema 和 Artifact 引用，不保存整份可变 Agent Context；
- Proposal、Approval、Execution 和 Verification 分离并可独立审计；
- 一条变更命令对应一个 Proposal、一次审批和一个 Execution，不需要多人会签；
- 验证失败不自动回滚；任何回滚都创建新的 Proposal 并重新显式审批。

## 聚合与表

本文定义领域语义；完整物理字段、外键、唯一约束、索引、APEX 视图和保留策略以 [26_aiops_physical_data_model.md](26_aiops_physical_data_model.md) 为准。

### `KBOT_OPS_TARGET`

数据库实例或集群的运维资产根对象。

| 字段 | 说明 |
| --- | --- |
| `TARGET_ID` | UUIDv7 领域主键；Oracle 使用 `RAW(16)` |
| `APP_ID` / `DOMAIN_ID` | APEX 过滤和强制 domain 边界 |
| `TARGET_KEY` / `DISPLAY_NAME` | domain 内唯一键和展示名 |
| `ENVIRONMENT` | `PROD`、`STG`、`DEV` |
| `DB_TYPE` / `VERSION_CODE` | Oracle、PostgreSQL、MySQL 及版本 |
| `ENDPOINT_JSON` | host、port、service/database；不含密码，未配置时为纯监控模式 |
| `DIAGNOSTIC_SECRET_REF` | 可选只读诊断凭据引用 |
| `EXECUTION_SECRET_REF` | 可选变更执行凭据引用，与诊断账号分离 |
| `EXECUTION_MODE` | `ADVISORY` 或 `AGENT_EXECUTE` |
| `CAPABILITIES_JSON` | 允许的诊断和动作类别 |
| `SECURITY_LEVEL` / `STATUS` | 安全等级；管理状态为 `ACTIVE/MAINTENANCE/DISABLED` |
| `HEALTH_STATUS` | 独立的运行健康状态：`UNKNOWN/HEALTHY/DEGRADED/UNREACHABLE` |
| `ROW_VERSION` / 审计时间 | 乐观锁和审计字段 |

唯一约束为 `(APP_ID, DOMAIN_ID, TARGET_KEY)`。Target 停用后不得创建新 Ops Run，但历史记录保留。

同一 Target 可同时绑定 Prometheus、Zabbix 和 OEM 等多个监控源，通过 `KBOT_OPS_MONITOR_SOURCE` 和 `KBOT_OPS_TARGET_MONITOR` 保存连接引用、外部目标标识、主/补充角色、优先级、指标覆盖和单 Target 覆盖配置。不再在 Target 上使用单值 `monitor_type`。

### `KBOT_OPS_TARGET_BINDING`

Agent 与 Target 的多对多权限绑定。

```text
ACCESS_MODE: OBSERVE | DIAGNOSE | PROPOSE | EXECUTE
REQUIRE_APPROVAL
POLICY_ID
MAX_DAILY_EXECUTIONS
ALLOWED_ACTIONS_JSON
CHANGE_WINDOW_JSON
STATUS
```

请求权限取 AuthContext、Target Binding、Policy 和请求目标的交集。`EXECUTE` 不代表绕过审批，只表示该 Agent 具备申请执行的资格。

### `KBOT_OPS_EVENT` 与 `KBOT_OPS_ALERT`

Event 保存标准化监控事件，Alert 是去重/关联后的告警聚合。

Event 重点字段：`MONITOR_SOURCE_ID`、`SOURCE_EVENT_KEY`、`TARGET_ID`、`EVENT_TYPE`、`SEVERITY`、`OCCURRED_AT`、`RECEIVED_AT`、`FINGERPRINT`、`PAYLOAD_JSON/URI`、`PAYLOAD_HASH`、`STATUS`。唯一约束为 `(MONITOR_SOURCE_ID, SOURCE_EVENT_KEY)`。

Alert 重点字段：`ALERT_ID`、`TARGET_ID`、`FINGERPRINT`、`STATUS`、`SEVERITY`、`FIRST_SEEN_AT`、`LAST_SEEN_AT`、`EVENT_COUNT`、`SUMMARY`、`CORRELATION_JSON`。状态为 `OPEN/ACKNOWLEDGED/SUPPRESSED/RESOLVED/CLOSED`。当前 Run 通过 `RUN.TRIGGER_ALERT_ID` 查询，不保存循环指针。

### `KBOT_OPS_RUN` 与 `KBOT_OPS_TASK`

Ops Run 是一次数据库诊断或运维闭环，可由用户、Webhook、告警或计划任务触发。Root 委派时保存 `PARENT_AGENT_RUN_ID/PARENT_DELEGATION_ID`，但流程完全由 AIOps Agent 管理。

Run 重点字段：

```text
OPS_RUN_ID, PARENT_AGENT_RUN_ID, PARENT_DELEGATION_ID, TARGET_ID, AGENT_ID
TRIGGER_TYPE, TRIGGER_EVENT_ID, TRIGGER_ALERT_ID, INSPECTION_FIRE_ID
ACTOR_ID, ORIGINAL_REQUEST
STATUS, PLAN_SNAPSHOT_JSON, POLICY_SNAPSHOT_JSON
DEADLINE_AT, CANCEL_REQUESTED_AT, CANCEL_REQUESTED_BY
FINAL_ARTIFACT_ID
ERROR_CODE, ERROR_MESSAGE, ROW_VERSION
CREATED_AT, STARTED_AT, COMPLETED_AT
```

Task 重点字段：

```text
OPS_TASK_ID, OPS_RUN_ID, TASK_KEY, TASK_TYPE
HANDLER_ID, HANDLER_VERSION, INPUT_SCHEMA_VERSION, OUTPUT_SCHEMA_VERSION
DEPENDS_ON_JSON, INPUT_ARTIFACTS_JSON, OUTPUT_ARTIFACT_ID
STATUS, ATTEMPT_COUNT, MAX_ATTEMPTS, TIMEOUT_SECONDS
LEASE_OWNER, LEASE_TOKEN, LEASE_UNTIL, AVAILABLE_AT
ERROR_CODE, ERROR_MESSAGE, ROW_VERSION
```

`TASK_TYPE` 限定为 `SCOPE/OBSERVE/DIAGNOSE/REQUEST_INPUT/PROPOSE/APPROVE/EXECUTE/VERIFY/ROLLBACK/COMPARE/REPORT`，禁止 Planner 自造执行类型。每次领取生成新的 Lease Token；完成、失败和心跳必须匹配本次 Token。

### `KBOT_OPS_ARTIFACT`

不可变运维产物，类型包括：

```text
METRIC_OBSERVATION
LOG_OBSERVATION
DB_DIAGNOSTIC
DIAGNOSTIC_GAP
MANUAL_SQL_REQUEST
USER_PROVIDED_DB_RESULT
SOP_CITATION_PACK
DIAGNOSIS
APPROVAL_CONTEXT
EXECUTION_RESULT
VERIFICATION_REPORT
ROLLBACK_RESULT
OPS_EXECUTION_REPORT
INSPECTION_REPORT
COMPARISON_PLAN
COMPARISON_RESULT
COMPARISON_REPORT
REPORT_CONTENT
```

字段包含 `ARTIFACT_ID`、`OPS_RUN_ID`、`OPS_TASK_ID`、`ARTIFACT_TYPE`、`SCHEMA_VERSION`、`PAYLOAD_JSON/URI`、`CONTENT_HASH`、`PROVENANCE_JSON`、`SECURITY_LEVEL`、`CREATED_AT`。原 Artifact 不覆盖，修订生成新 Artifact。

### `KBOT_OPS_CHANGE_PROPOSAL`

ChangeProposal 是不可变的候选变更，不是执行命令。

```text
PROPOSAL_ID, OPS_RUN_ID, TARGET_ID, PROPOSAL_VERSION
ACTION_TYPE, ACTION_TEMPLATE_ID, PARAMETERS_JSON
RATIONALE, IMPACT_SCOPE_JSON, RISK_LEVEL
PRECONDITIONS_JSON, ROLLBACK_PLAN_JSON, VERIFICATION_PLAN_JSON
EVIDENCE_ARTIFACTS_JSON, PROPOSAL_HASH
STATUS, EXPIRES_AT, CREATED_BY_TASK_ID, CREATED_AT
```

状态：`DRAFT → PENDING_APPROVAL → APPROVED | REJECTED | EXPIRED | SUPERSEDED`；批准后提交执行时变为 `CONSUMED`。修改参数、目标或回滚方案必须创建新版本并重新审批。

### `KBOT_OPS_HITL`

统一处理补充数据、人工诊断结果、人工处理结果和变更审批。

```text
HITL_ID, OPS_RUN_ID, OPS_TASK_ID, PROPOSAL_ID
REQUEST_TYPE: DATA_REQUIRED | MANUAL_DIAGNOSTIC_SQL |
              MANUAL_ACTION_RESULT | CHANGE_APPROVAL
PROMPT_TEXT, RESPONSE_SCHEMA_JSON
INPUT_ARTIFACTS_JSON, RESPONSE_JSON
STATUS, REQUESTED_BY, ASSIGNEE_USER_ID, RESPONDED_BY
REQUESTED_AT, RESPONDED_AT, EXPIRES_AT, ROW_VERSION
```

状态：`PENDING → ANSWERED | APPROVED | REJECTED | EXPIRED | CANCELLED`。恢复执行时根据 Task 和 Artifact 重建上下文，不反序列化旧 Python Context、Prompt 对象或执行历史。

仅当 `TRIGGER_TYPE=CHAT`，自动只读诊断不可用或缺少必要目录能力，且现有证据无法支撑根因判断时，才创建 `MANUAL_DIAGNOSTIC_SQL` HITL，并将 `ASSIGNEE_USER_ID` 固定为当前对话用户。用户回贴结果后恢复原 Ops Run，可继续产生下一轮最小必要查询。Alert、Schedule 等自动 Run 不进入该交互流程。总体边界见 [21_aiops_interactive_diagnosis.md](21_aiops_interactive_diagnosis.md)，实施级状态机见 [37_aiops_step8_chat_manual_diagnosis_hitl.md](37_aiops_step8_chat_manual_diagnosis_hitl.md)。

### `KBOT_OPS_EXECUTION`

记录一次 DB Executor 调用及结果。

```text
EXECUTION_ID, PROPOSAL_ID, OPS_RUN_ID, TARGET_ID
IDEMPOTENCY_KEY, EXECUTOR_REQUEST_ID
PROPOSAL_HASH, ACTION_TYPE, ACTION_TEMPLATE_ID
STATUS, APPROVAL_TOKEN_ID, STARTED_AT, COMPLETED_AT
RESULT_ARTIFACT_ID, ERROR_CODE, ERROR_MESSAGE
ROLLBACK_OF_EXECUTION_ID, ROW_VERSION
```

状态：`CREATED → SUBMITTED → RUNNING → SUCCEEDED | FAILED | TIMED_OUT | CANCELLED`。需要回滚时创建新的 Execution，并通过 `ROLLBACK_OF_EXECUTION_ID` 关联原执行，不覆盖原记录。

### `KBOT_OPS_RUN_EVENT`、`KBOT_OPS_APPROVAL_TOKEN` 与可靠消息

`RUN_EVENT` 是只追加的 Run/SSE 状态流，与外部监控 `EVENT` 分离。`APPROVAL_TOKEN` 保存一次性令牌 Hash、绑定字段和消费状态，不落明文 Token。`INBOX/OUTBOX` 分别保证外部回调去重和提交后可靠交付。

### `KBOT_OPS_POLICY`

保存版本化运维策略：环境、Target、动作类别、风险等级、是否需要一次审批、时间窗口、频次上限、是否允许预授权回滚等。Run 创建时冻结策略快照，执行前再次检查当前策略；若策略收紧，旧批准失效。

精确的 Policy 输入/输出、四类人工节点、单命令审批、Advisory 和审批 API 见 [23_aiops_policy_hitl_and_command_lifecycle.md](23_aiops_policy_hitl_and_command_lifecycle.md)。

## Ops Run 状态机

```text
CREATED → SCOPING → OBSERVING → DIAGNOSING
                                  ├→ WAITING_INPUT → DIAGNOSING
                                  ├→ DIAGNOSED → COMPLETED
                                  └→ PROPOSING → WAITING_APPROVAL
                                                        ├→ REJECTED
                                                        └→ EXECUTING
                                                              ↓
                                                           VERIFYING
                                                    ├→ COMPLETED
                                                    ├→ DEGRADED
                                                    └→ ROLLBACK_PENDING
```

任何非终态都可进入 `FAILED/CANCELLED/EXPIRED`。`DEGRADED` 表示执行完成但验证未达到目标，需要人工处理；不能为了显示成功自动改写诊断或验证结果。

诊断阶段的时间窗口、Evidence Pack、假设/反证、根因级别和触发模式详见 [24_aiops_diagnosis_orchestration_and_evidence.md](24_aiops_diagnosis_orchestration_and_evidence.md)。

## 监控事件到诊断

```text
Webhook/Polling/Scheduler
        ↓ Inbox 幂等校验
KBOT_OPS_EVENT
        ↓ fingerprint + time window
KBOT_OPS_ALERT
        ↓ routing/policy
KBOT_OPS_RUN
        ↓
Observe Tasks（Metrics/Logs/DB，可并行）
        ↓
Diagnosis Artifact
```

接入响应只完成验签、去重和持久化，不同步运行 Agent。相同外部事件不得重复创建 Run；同一 Alert 已存在活跃 Run 时，新增 Event 关联到当前 Run，除非策略要求开启新诊断。

## DB Executor 请求契约

AIOps Agent 先在本地事务中写入 Execution 和 Outbox，提交后由 Dispatcher 调用 DB Executor：

```text
ExecutorDispatchRequest {
  execution_id
  executor_request_id
  trace_id
}
```

DB Executor 随后使用独立实例身份向 AIOps Claim。Claim 原子消费审批授权并返回短期 `MutationExecutionGrant`，其中绑定 Target/Proposal/Template/Parameter/Command/Policy Hash、SecretRef、期限和 `max_database_attempts=1`。Dispatcher 不发送 SQL、参数、密码或明文 Token。

DB Executor 必须：

1. 校验调用服务身份、domain、Target 状态和请求 audience；
2. 通过 `SECRET_REF` 获取短期凭据，不接受请求传入密码；
3. 只解析登记过的诊断/动作模板和参数 schema，诊断请求只接受 `tool_id + parameters`，不接受 SQL 文本；
4. Mutation 通过 Claim 校验 Proposal Hash、一次性审批授权、风险等级和变更窗口；
5. 使用 `IDEMPOTENCY_KEY` 防止重复执行；
6. 返回 `EXECUTOR_REQUEST_ID`，执行结果通过查询或签名回调进入 Inbox；
7. 输出脱敏结果和 Result Hash，不把凭据或完整敏感结果写入日志。

第一版不接受 LLM 或用户提交的任意 SQL。确需临时 SQL 时只能进入人工管理通道，不属于自动 AIOps 流程。

直连诊断目录、Oracle/MySQL Dialect、SQL 模板版本和输出 Artifact 契约详见 [22_aiops_database_diagnostic_catalog.md](22_aiops_database_diagnostic_catalog.md)。Advisory、Action Catalog、Claim、Grant 和 Mutation 的实施级契约见 [38_aiops_step9_advisory_approval_and_execution.md](38_aiops_step9_advisory_approval_and_execution.md)。

## 审批令牌

审批授权绑定：`proposal_id + proposal_hash + target_id/version + template/command/parameters_hash + policy_decision_hash + expires_at + approver_id`。每条命令只需一位有权用户批准一次，授权在 Executor Claim 时一次性消费；不提供整个计划的批量批准。Proposal 任何字段变化、Target 版本变化、策略收紧或超时都会使授权失效。DB Executor 是最终强制校验点，AIOps Agent 的“已批准”状态不能替代 Grant 验证。

## 执行、验证与回滚

执行成功只表示动作完成，不代表问题已解决。AIOps Agent 必须按 Proposal 中的 Verification Plan 创建只读 Verify Task，对比前后指标、数据库健康检查和告警状态，生成 `VERIFICATION_REPORT`。

验证结果：`VERIFIED/DEGRADED/FAILED/INCONCLUSIVE`。只有 `VERIFIED` 才可将 Run 标记为成功完成。回滚始终是新的 Proposal、审批授权和 Mutation Execution，不允许原审批覆盖，也不自动触发；回滚后仍需再次验证。

变更或人工处理完成后必须使用同一指标定义、聚合方式和可比较时间窗口生成 `COMPARISON_REPORT`。它单独记录处理前基线、处理后观测、绝对/相对变化、告警恢复和最终结论；数据不足时必须标记 `INCONCLUSIVE`。

监控、巡检和报告的概念模型见 [20_aiops_monitoring_inspection_and_reporting.md](20_aiops_monitoring_inspection_and_reporting.md)，多副本调度、Inspection Fire、报告版本和 Comparison 判级见 [39_aiops_step10_inspection_reporting_and_comparison.md](39_aiops_step10_inspection_reporting_and_comparison.md)。

## 事务与恢复

- Event/Alert/Run 创建通过 Inbox/Outbox 和幂等键保证一致性；
- Worker 在外部调用前释放数据库事务和连接；
- Task 完成时原子写 Artifact、Task 状态和 Ops Event；
- Executor 回调使用 `(EXECUTOR_REQUEST_ID, STATUS_VERSION)` 去重；
- Worker 崩溃后通过租约恢复，不重复提交已经 `SUBMITTED` 的 Execution；
- HITL 恢复只重放未完成 Task，已完成 Observation/Diagnosis Artifact 继续复用。

## 从 3.x 迁移时明确废弃

- `encrypted_password`：替换为 `SECRET_REF`；
- `kbot_ops_pending_request` 的完整 Context/Plan 快照：替换为 HITL + Artifact 引用；
- `entry_id/session_id` 作为运维流程主键：替换为 `OPS_RUN_ID`；
- LLM 生成 `sql_to_run` 后由用户或系统直接执行：替换为动作模板 + 参数 schema；
- 验证失败默认自动执行 rollback SQL：替换为独立、可审批的 Rollback Execution；
- 单张执行报告表承载全部过程：替换为不可变 Artifact + Execution 记录 + APEX 只读投影视图。
