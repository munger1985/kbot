# AIOps 步骤 9：Advisory、逐命令审批与受控执行

## 当前实施进度

步骤 9 按安全阶段门拆分实施。阶段 9A 已于 2026-07-24 建立 Advisory 基线：

- 新增随代码发布的 Oracle/MySQL Action Catalog、严格参数 Schema、模板 Hash、
  Catalog Hash 和 `strict-template.v1` Renderer；
- 首期 Catalog 只登记 `db.session.terminate@1.0.0`，Oracle 精确渲染单会话
  `DISCONNECT SESSION`，MySQL 精确渲染单连接 `KILL CONNECTION`；
- Action 参数只能来自当前 Target 的 `SOURCE_VERIFIED` 阻塞链/活动会话 Fact，
  `USER_PROVIDED`、KC Citation 和模型文本不能提供可执行参数；
- 诊断 DAG 增加 `ACTION_PLAN.v1` 和 `PROPOSAL_OUTCOME.v1`，运行内核在完成
  Proposal Task 时原子创建 `ADVISORY_READY` 行和不可变 Snapshot 引用；
- Public/Internal API 支持查看权威命令预览、驳回 Advisory、回填受限人工结果；
  人工声明保存为 `USER_PROVIDED_ACTION_RESULT.v1`，不会被标记为系统执行；
- `action_execution_enabled` 在 9A 注册时固定为 `false`。即使 Target、Binding 和
  Policy 允许执行，也只能生成 Advisory，不创建 Approval Token 或 Execution。

阶段 9B 已完成 Proposal Expiry 和人工结果后的独立 Verify：

- `EXECUTED` 人工结果与 Outbox 在同一事务提交；`FAILED/CANCELLED` 仅留痕，
  不创建无意义的效果验证；
- 领域 Sink 以 Proposal 为幂等键创建 `change.advisory-verify@1` 独立 Run，
  Run 显式记录来源 Proposal 和人工结果 Artifact；
- Outbox 只携带来源 ID 与路由上下文；Runtime 必须回读并核对 Proposal
  Snapshot、人工结果、原 Run、Target、Agent 和 Actor，不能信任消息中的动作参数；
- 验证 Run 重新解析当前 Target、Binding、Policy 与 Diagnostic Catalog，只执行
  Identity、Active Session 和 Blocking Chain 等只读工具；
- 最终生成 `ACTION_VERIFICATION.v1`，结论限定为 `VERIFIED`、
  `NOT_ACHIEVED`、`ADVERSE` 或 `INCONCLUSIVE`；用户声明本身不能证明成功；
- Reconciler 将到期的 `ADVISORY_READY/PENDING_APPROVAL` Proposal 收敛为
  `EXPIRED`，回填接口也在事务内拒绝已到期 Proposal。

阶段 9C 再实现逐命令审批、一次性授权、Claim、Mutation Driver、回调对账和
UNKNOWN 收敛。在这些安全门完成前不得把 Bootstrap 中的执行开关改为配置值。

## 目标与安全边界

本步骤将 `CONFIRMED/PROBABLE` 诊断转化为可审计的处理建议，并在 Target 明确配置为 `AGENT_EXECUTE` 时，允许一位有权用户逐条批准后由 DB Executor 执行 Oracle/MySQL 变更。

“Agent Execute”表示 Agent 可以在批准后投递命令，不表示 LLM 自主执行。LLM 只能建议 `action_template_id + parameters`；确定性代码负责模板版本、参数来源、Policy、风险、审批、Executor Grant、执行和验证。任何路径都不接受 LLM、用户或 API 传入的任意 SQL。

首期不实现 Shell、SSH、OEM Job、Zabbix Remote Command、Kubernetes 操作或批量批准。Mutation 部署级 Kill Switch 默认关闭。

## 现状审计

3.x `ops-heal-skill` 和旧 DB Executor 存在以下不可迁移设计：

- Prompt 根据 `is_mutation_allowed` 决定是否追加自动执行步骤；
- `sql_to_run` 在 Agent Context、HITL 和接口间传递；
- Executor 通过字符串包含判断放行 `ALTER SYSTEM/DBMS_/KILL/FLUSH/TRUNCATE` 等大范围命令；
- 请求携带原始 SQL、连接信息和密码，Driver 暴露通用 `execute_non_query()`；
- 网络错误后无法判断命令是否已经到达数据库，也缺少可靠的逐命令 fencing。

旧命令只能作为 Action Catalog 评审输入。4.0 不 import 旧 Heal Skill、Mutation 白名单或通用非查询执行接口。

## 两种处理模式

最终能力是以下交集，任何一层只能降级：

```text
Template Capability
∩ Target.execution_mode
∩ Deployment Kill Switch
∩ Current Policy
∩ Actor Permission
∩ Root Cause Level
∩ Fresh Evidence / Preconditions
∩ Execution Secret Capability
```

### `ADVISORY`

- 生成精确、版本化的处理步骤或命令预览；
- 系统不创建 Approval Token 或 Execution；
- 用户自行处理后可提交 `MANUAL_ACTION_RESULT`；
- 人工声明只按 `USER_PROVIDED` 保存，仍需监控/只读诊断验证。

### `AGENT_EXECUTE`

- 仅支持模板声明为 `EXECUTABLE_AFTER_APPROVAL` 的动作；
- 每条命令各自创建 Proposal、HITL、Approval Authorization 和 Execution；
- 每条命令只允许一次数据库投递；失败、超时或未知结果不得自动重试；
- 多命令方案严格串行，上一条完成验证后才创建下一条 Proposal。

Policy 可将 `AGENT_EXECUTE` 降为 `ADVISORY`，不能把 `ADVISORY_ONLY` 模板提升为可执行。

## Action Template Catalog

Action Catalog 是随代码发布、只读、版本化的部署资产，与 Diagnostic Catalog 物理分目录：

```text
aiops_agent/actions/
  contracts.py
  registry.py
  validation.py
  dialects/
    oracle/{manifest.yaml,commands/,preconditions/,verification/}
    mysql/{manifest.yaml,commands/,preconditions/,verification/}
```

每个模板至少包含：

```text
action_template_id, version, template_sha256, renderer_version
db_type, supported_versions, required_capabilities
required_privileges, required_entitlements
execution_capability: ADVISORY_ONLY | EXECUTABLE_AFTER_APPROVAL
risk_level, environment_allowlist
parameter_schema, parameter_evidence_rules
command_ref, command_sha256
precondition_tool_refs[], precondition_predicates[]
expected_effects[], verification_tool_refs[]
rollback_template_ref?, rollback_constraints
statement_timeout_seconds, observation_delay_seconds
idempotency_class: IDEMPOTENT | CHECK_THEN_ACT | NON_RETRYABLE
concurrency_key, sensitivity, status
```

参数值必须来自已验证 FactRef、服务端常量或显式用户输入，并保存逐参数来源。`AGENT_EXECUTE` 要求影响对象由近期 `SOURCE_VERIFIED` Evidence 重新确认；只依赖 SOP、模型推断或用户回贴时强制降为 Advisory。

不能 bind 的标识符只能由模板根据枚举或已验证对象 ID 渲染。Action Renderer 产生一条确定命令及 `command_sha256`；Proposal Builder 与 Executor 安装相同 Catalog，并分别渲染、比对 Hash。模板禁止多语句、动态 SQL、自由后缀和自然语言片段。

## 首期动作范围

首批可在批准后执行的动作应极窄，并且必须可检查目标是否仍存在：

- Oracle：终止/断开一个已验证会话等单对象动作；
- MySQL：终止一个已验证连接或查询等单对象动作；
- 经真实环境评测后新增的同等级、范围明确动作。

以下类别首期只允许 Advisory，部分甚至只描述步骤而不渲染命令：

- 参数、实例角色、归档、复制拓扑和集群成员变更；
- 表空间、数据文件、对象 DDL 和容量调整；
- `DBMS_*` 管理包、统计信息、优化器和计划管理；
- `FLUSH/PURGE/OPTIMIZE/ANALYZE` 等影响范围不稳定的操作；
- `DROP/TRUNCATE/GRANT/REVOKE`、业务 DML、文件系统、重启和操作系统命令。

不能因为一条命令出现在旧字符串白名单中就进入新目录。每个模板必须通过 DBA、安全、版本、权限、回滚和故障注入评审。

## Action Plan 与 Proposal Snapshot

步骤 7 的 `SOLUTION_DRAFT` 只提供候选 Action Template。`ActionPlanValidator` 校验根因、参数来源、依赖、风险和 Policy，形成不可变 `ACTION_PLAN.v1`：

```text
solution_group_key
actions[] {
  ordinal, action_template_id
  canonical_parameters, parameter_fact_refs
  rationale, expected_effects
  preconditions, verification_plan
  rollback_candidate
}
```

多动作计划不一次性生成所有待审 Proposal。系统只为当前 ordinal 创建 `CHANGE_PROPOSAL_SNAPSHOT.v1`，其内容包括：

```text
proposal_id, target_id, target_version
template_id/version/hash, renderer_version
canonical_parameters, parameters_hash
rendered_command, command_hash
risk, impact, rationale
preconditions, rollback_plan, verification_plan
evidence_refs, policy_decision_hash
expires_at, proposal_hash
```

`proposal_hash` 对上述规范化字段整体计算。Snapshot 是 Proposal Task 的唯一输出 Artifact，使用 Target 安全等级，不进入 SSE/APEX 视图。Proposal 行保存 Snapshot Artifact ID、Template Hash 和 Command Hash，便于直接定位权威版本。

Proposal 创建后内容不可修改，只能迁移状态。Target、参数、Template、Evidence、验证或回滚任一变化都创建新 Proposal Version，旧版本标记 `SUPERSEDED`。

## Advisory 流程

```text
ACTION_PLAN
  → PolicyDecision(ADVISORY_ONLY)
  → Proposal(ADVISORY_READY) + Snapshot
  → 用户 GET 查看命令/步骤
  → 用户自行执行或放弃
  → MANUAL_ACTION_RESULT
  → Verify / Compare
```

`MANUAL_ACTION_RESULT` 接受 `EXECUTED/FAILED/CANCELLED`、处理时间、备注和受限输出。它不产生 Approval Token，也不会把 Proposal 改成系统已执行。规范化结果保存为 `USER_PROVIDED_ACTION_RESULT.v1`。

若用户不回填，Advisory Proposal 保持可查看直到过期，Run 可以先以“建议已生成”结束，不永久占用 Worker。用户之后回填时创建独立 Verify Run 或关联的后续 Run。

人工回填与验证采用“事实先落库、命令后投递”：

```text
MANUAL_ACTION_RESULT(EXECUTED)
  ├─ USER_PROVIDED_ACTION_RESULT.v1
  └─ OPS_ADVISORY_RESULT_RECORDED (Outbox)
       → change.advisory-verify@1
          → fresh read-only diagnostics
          → ACTION_VERIFICATION.v1
```

验证 Run 不复用原诊断证据，不包含 `PROPOSE/EXECUTE` Task，也不会递归生成新
Proposal。目标会话同时从活动会话和阻塞链消失才可判定 `VERIFIED`；目标仍存在
为 `NOT_ACHIEVED`；连接、权限、版本或证据缺失统一为 `INCONCLUSIVE`。
`ADVERSE` 为后续监控指标比较保留，不能在缺少负面观测时推断。

## 审批展示与用户操作

SSE 只发送：

```text
proposal.pending_approval {
  proposal_id, risk_level, expires_at, proposal_hash
}
```

完整命令必须通过授权 GET 获取：

```text
GET /api/v1/ops/proposals/{proposal_id}
```

前端必须展示精确命令、Target/环境、参数来源、影响、风险、前置条件、Evidence、回滚和验证计划，以及 Proposal Hash/版本。SQL/命令不写入 SSE、日志或 APEX 待审视图。

批准接口：

```json
{
  "expected_row_version": 3,
  "expected_proposal_hash": "sha256...",
  "note": "已确认终止该阻塞会话"
}
```

批准请求必须来自已认证门户 API Key，AuthContext Domain 与 Proposal 一致，`asserted_user_id` 匹配待审记录，并通过当前 Policy。首期只需一位批准人，不强制发起人与批准人分离。普通聊天中的“同意”、批量勾选、Root Agent、LLM 或监控系统均不能构成审批；细粒度审批 Scope 留待后续权限阶段实现。

## Approval Authorization

`KBOT_OPS_APPROVAL_TOKEN` 表示一次不可转移的审批授权记录，不是返回给浏览器的 Bearer Token。`TOKEN_HASH` 对以下规范化 Approval Claims 和随机 Nonce 计算：

```text
proposal_id, proposal_hash
target_id, target_version
template_id/version/hash, command_hash
parameters_hash, policy_decision_hash
approver_id, issued_at, expires_at, nonce
```

批准事务按 Run → Proposal → HITL → Approval Token → Execution 加锁并完成：

1. 重新检查 Proposal/Pending HITL、Row Version、Hash、Assignee/权限；
2. 使用当前 Target/Policy/Kill Switch 重新评估，不只相信 Run Snapshot；
3. 创建 `APPROVAL_DECISION.v1`、`APPROVAL_TOKEN(ISSUED)`；
4. 创建唯一 `EXECUTION(CREATED)` 和 Outbox，不调用 Executor；
5. Proposal/HITL 进入 `APPROVED`，Run 进入 `EXECUTING`；
6. 提交后 Dispatcher 只发送 `execution_id + executor_request_id`。

前端永远看不到 Token 或 Mutation Grant。审批重放同一 Idempotency Key 返回原结果；内容或 Hash 不同返回冲突。

## Executor Claim 与 Mutation Grant

DB Executor 收到最小通知后，以启动时生成且不复用的 `executor_instance_id` 调用：

```text
POST /internal/v1/aiops/executions/{execution_id}/claim
{
  executor_request_id,
  executor_instance_id,
  supported_catalog_version
}
```

Claim 在一个 UoW 中再次校验：

- Execution 仍为 `CREATED`，Approval Token 为 `ISSUED` 且未过期；
- Proposal/Target/Template/Parameters/Command Hash 全部匹配；
- Target 版本、状态、Execution Mode、当前 Policy、时间窗口和 Kill Switch 仍允许；
- Executor Service Identity、实例、Catalog 版本和 Action Capability 合法；
- 同一 Target/Concurrency Key 没有冲突执行。

成功时 Token 原子变为 `CONSUMED`，Proposal 变为 `CONSUMED`，Execution 变为 `SUBMITTED` 并绑定 `EXECUTOR_INSTANCE_ID/CLAIMED_AT/GRANT_JTI_HASH`。响应为短期签名 `MutationExecutionGrant`：

```text
grant_id/jti, audience, executor_instance_id
execution_id, executor_request_id, expires_at
initial_status_version
target_id/version, db_type, connection_profile, execution_secret_ref
template_id/version/hash, renderer_version
parameters, parameters_hash, command_hash
precondition_refs/predicates, timeout
proposal_hash, policy_decision_hash, approver_id
max_database_attempts: 1
```

Grant 不含原始任意 SQL；Executor 根据本地 Template 渲染并校验 Command Hash。Claim 对相同 Execution/Request/Instance 在命令尚未开始前可幂等返回等价 Grant；不同实例、不同请求 ID 或已进入 `RUNNING` 后拒绝。审批授权一经 Claim 即被消耗，即使 Executor 随后崩溃也不会自动再投递。

## Mutation Executor

Mutation 能力与只读诊断使用独立并发池、Secret namespace 和 Kill Switch。高权限连接使用短连接，不进入共享池。Executor 固定顺序为：

1. 校验 Service Identity、Grant 签名/audience/instance/expiry；
2. 校验本地 Catalog、Template/Renderer/Parameter/Command Hash；
3. 从 Secret Store 解析 `execution_secret_ref`，不回退到诊断凭据；
4. 建立 TLS 短连接，设置 Client Identifier、超时和审计标签；
5. 执行模板内只读 Precondition，确认目标身份、版本和对象状态仍匹配；
6. 通过签名回调将 Execution 置为 `RUNNING`；
7. 仅一次调用方言专用 `execute_action()`；
8. 收集有界结果、数据库错误码、耗时和 Result Hash；
9. 立即关闭连接并回调终态。

Driver 不对外暴露 `execute_non_query(sql)`；只有 Registry 已解析的 `RenderedAction` 能进入 `execute_action()`。一次 Execution 只能包含一条语句。Precondition 不满足时返回 `PRECONDITION_CHANGED`，不执行 Mutation。

Executor 日志不记录命令正文、参数、账号、连接 Profile 或数据库原始错误。执行凭据必须满足模板声明的最小权限；Secret 元数据中的 Action Capability 也必须覆盖模板。

## At-most-once 与未知结果

数据库管理命令通常无法提供真正的分布式 exactly-once。4.0 采用“单 Claim、单数据库投递、未知即停”的 at-most-once 策略：

- Executor 在调用 Driver 前记录 `RUNNING`，之后绝不自动再次调用；
- HTTP/进程/数据库连接在投递后中断时标记 `UNKNOWN`，不能推断失败；
- Dispatcher 仅在 Execution 仍为 `CREATED` 时重试通知；
- `SUBMITTED/RUNNING/UNKNOWN` 只允许对账和只读验证，不重新投递；
- 即使 Template 声明 `IDEMPOTENT/CHECK_THEN_ACT`，第二次数据库调用也需要新 Proposal 和新审批；
- `PROPOSAL_ID` 对 Execution 唯一，保证一个审批版本最多对应一次执行。

`idempotency_class` 用于决定如何验证未知结果，不用于绕过新的审批。

## Executor Event 与对账

Executor 使用 Service Identity 和签名事件回传：

```text
executor_request_id, execution_id
status, status_version, occurred_at
grant_jti_hash, executor_instance_id
result_schema, bounded_result, result_hash
error_code, retryable=false
```

AIOps 先写 Inbox，再按 `status_version` 和允许迁移对账；重复/乱序事件不重复推进。`EXECUTION_RESULT.v1` 保存有界、脱敏结果和 Template/Command/Grant Hash，不保存 Secret。

若 `SUBMITTED/RUNNING` 超过 Deadline 且没有终态，Reconciler 标记 `UNKNOWN` 并创建 Verify Task。Executor 不能直接修改 Run/Task/Proposal 表。

## 验证、后续命令与回滚

`SUCCEEDED` 只表示数据库接受了命令，不表示故障恢复。每个动作完成后按 Template 的 Observation Delay 使用监控或只读诊断凭据执行：

```text
ACTION_EXECUTION_RESULT
  → VERIFY_PRECONDITION_EFFECT
  → ACTION_VERIFICATION.v1
  → VERIFIED | NOT_ACHIEVED | ADVERSE | INCONCLUSIVE
```

验证使用与基线兼容的指标、窗口、单位和工具版本。`UNKNOWN` 也先验证实际状态，再决定是否需要新 Proposal。只有 `VERIFIED` 才可创建下一 ordinal；`NOT_ACHIEVED/ADVERSE/INCONCLUSIVE` 停止后续命令并生成报告。

回滚永远是新的 Action Plan/Proposal/Approval/Execution，不允许原审批覆盖回滚，也不自动执行。不可逆动作必须在 Template、Proposal 和 UI 显著标识；没有可信回滚与验证路径的动作默认 Advisory。

完整的处理前后 `COMPARISON_REPORT` 在步骤 10 实现，本步骤先保存 Verification 所需基线引用、执行时间和 After Window 计划。

## 取消和策略变化

- Proposal `PENDING_APPROVAL` 可拒绝或过期；
- 已批准但仍为 `CREATED` 的 Execution 在 Run 取消、Target 停用或 Policy 收紧时撤销 Token并取消 Execution；
- 已 `SUBMITTED/RUNNING` 时取消只能阻止后续动作，不能声称数据库命令已停止；
- Target Version、Template、Parameters、Command Hash 或 Policy 变化都会使旧 Proposal/Token 失效；
- Agent/Target Binding 停用立即阻止新审批和 Claim。

所有竞争事务统一锁顺序：

```text
Run → Proposal → HITL → Approval Token → Execution → Target concurrency rows
```

## API 契约

```text
GET  /api/v1/ops/proposals/{proposal_id}
POST /api/v1/ops/proposals/{proposal_id}/approve
POST /api/v1/ops/proposals/{proposal_id}/reject
POST /api/v1/ops/proposals/{proposal_id}/manual-result

POST /internal/v1/aiops/executions/{execution_id}/claim
POST /internal/v1/aiops/executor-events

POST /internal/v1/db-executor/executions
```

审批/拒绝/人工结果都需要 Idempotency Key 和 Row Version。Approve 额外提交用户实际看到的 `expected_proposal_hash`。跨 Domain 或无 Target 权限统一 `404`；Hash/版本变化返回 `409/412`；已过期返回 `410`；Kill Switch/Policy 阻止返回稳定 `OPS_EXECUTION_DISABLED`，不返回模板正文或内部策略。

## 数据模型补充

为支持权威预览和 Executor fencing，步骤 1 DDL 增加：

```text
KBOT_OPS_CHANGE_PROPOSAL:
  SNAPSHOT_ARTIFACT_ID
  ACTION_TEMPLATE_HASH
  RENDERER_VERSION
  COMMAND_HASH

KBOT_OPS_EXECUTION:
  ACTION_TEMPLATE_HASH
  PARAMETERS_HASH
  COMMAND_HASH
  EXECUTOR_INSTANCE_ID
  CLAIMED_AT
  GRANT_JTI_HASH
  STATUS_VERSION
```

`APPROVAL_TOKEN_ID` 在 Execution `CREATED` 起即必填；`TOKEN_HASH` 表示 Approval Claims Hash，不是可重放明文 Bearer。`SNAPSHOT_ARTIFACT_ID/RESULT_ARTIFACT_ID` 外键延后创建。`STATUS_VERSION` 从 1 单调增加并参与回调去重。

## 代码布局

```text
aiops_agent/
  actions/{contracts,registry,validation,rendering}.py
  domain/change/{proposal,approval,execution,transitions,policy}.py
  application/change/
    build_action_plan.py
    create_proposal.py
    approve.py
    reject.py
    claim_execution.py
    apply_executor_event.py
    reconcile.py
  orchestration/change/{handlers,blueprints}.py
  tests/change/

apps/aiops_db_executor/
  executor/{grant_verifier,mutation_service}.py
  drivers/{oracle_actions,mysql_actions}.py
```

Action Registry/Renderer 可由 AIOps Worker 和 Executor 打包复用，但 Proposal、Policy 和执行状态只属于 `aiops_agent`。Executor 不持有 KBot Schema Session。

## 实施顺序

1. 完成 DDL 增量、Proposal/Approval/Execution Entity 与状态迁移；
2. 建立 Action Catalog、离线 Validator、Renderer 和 Proposal Hash；
3. 先实现完整 Advisory/Manual Result/Verify，证明没有自动执行路径；
4. 实现 Approval API、SSE 通知、权威 GET 和单审批并发测试；
5. 实现 Execution/Outbox、Claim、Mutation Grant 和 Executor Event Inbox；
6. 实现 Oracle/MySQL 独立 Mutation Driver、Precondition 和窄首批模板；
7. 实现 Unknown Reconciliation、Verify、串行下一动作和独立 Rollback Proposal；
8. 默认 Kill Switch 关闭，完成真实数据库故障注入后才按环境启用。

## 验收门槛

- LLM/用户/API 无法提交 SQL、模板版本、Command Hash 或自定义执行限制；
- Advisory 不创建 Token/Execution，Agent Execute 每条命令恰好一个 Proposal 和一次审批；
- 用户看到的 Proposal Hash/命令与 Executor 本地渲染完全一致；
- 聊天“同意”、Root Agent、批量请求和重复点击不能批准命令；
- Target/Policy/Binding/Template/参数变化会在审批或 Claim 阶段阻断；
- Token/Grant 过期、重放、跨实例 Claim 和伪造回调全部失败；
- 网络超时、Executor 崩溃和回调丢失不会自动二次投递数据库命令；
- Precondition 改变时不执行；`UNKNOWN` 先只读验证；
- 回滚始终创建新 Proposal 并重新审批；
- SQL、Secret、连接 Profile 和敏感结果不进入 SSE、APEX 视图或普通日志；
- Mutation Kill Switch 关闭时系统仍可完整运行 Advisory；
- Oracle/MySQL 首批模板通过权限最小化、并发、超时、故障注入和真实效果验证。
