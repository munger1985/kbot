# 4.0 AIOps Policy、HITL 与命令生命周期

步骤 8 的 Chat 人工补证、Manual SQL、上传和恢复事务见 [37_aiops_step8_chat_manual_diagnosis_hitl.md](37_aiops_step8_chat_manual_diagnosis_hitl.md)。
步骤 9 的 Action Catalog、Advisory、单命令审批、Executor Claim 与执行验证见 [38_aiops_step9_advisory_approval_and_execution.md](38_aiops_step9_advisory_approval_and_execution.md)。

## 决策边界

Policy 用于强制 Target、用户、Agent、动作和时间窗口权限；HITL 用于记录一次人工决定。它们都不代替 Executor 的最终校验。

首期规则：

- 版本化只读诊断工具自动执行，不需要审批；
- Chat 人工诊断 SQL 由用户自己执行，不是变更审批；
- `ADVISORY` 只生成解决命令，系统不执行；
- `AGENT_EXECUTE` 下每条变更命令必须由一位有权用户批准一次；
- 不要求多人会签，不允许批量批准整个执行计划；
- LLM、Agent Service Identity 和监控系统不能成为批准人。

## 四类人工节点

| 类型 | 适用场景 | 是否发放审批令牌 |
| --- | --- | --- |
| `DATA_REQUIRED` | Chat 中要求用户补充上下文 | 否 |
| `MANUAL_DIAGNOSTIC_SQL` | Chat 中用户执行只读 SQL 并回贴 | 否 |
| `MANUAL_ACTION_RESULT` | Advisory 模式回填人工处理结果 | 否 |
| `CHANGE_APPROVAL` | 系统将执行一条变更命令 | 是，一次性 |

“自动 Run 不交互”仅指它不进入人工诊断 SQL 循环。Alert/Schedule Run 在产生变更提案后，仍可进入 `WAITING_APPROVAL`，由前端待审列表展示；超时后结束为 `EXPIRED`，不会自动执行。

## Policy 输入与结果

Policy Gate 输入：

```text
PolicyInput {
  domain_id, app_id
  actor_id, actor_permissions
  agent_id, service_identity
  target_id, target_version, environment
  trigger_type, execution_mode
  action_template_id, action_version, parameters_hash
  risk_level, requested_at
}
```

评估顺序固定为：Domain 边界 → Agent/Target Binding → Target 状态 → Execution Mode → Action Allowlist → 用户权限 → 环境/时间窗口 → 频次/并发上限 → Proposal/Target 版本。

```text
PolicyDecision {
  decision: ALLOW_READ | REQUIRE_APPROVAL | ADVISORY_ONLY | DENY
  policy_id, policy_version, decision_hash
  reason_codes, constraints
  expires_at
}
```

Policy 可将 `AGENT_EXECUTE` 降级为 `ADVISORY_ONLY`，不得反向提升 Target 配置的执行权限。Run 保存 Policy Snapshot，执行前再用当前策略评估一次；任一次为 `DENY` 都禁止执行。

## 每条命令的聚合

一个解决方案可包含多条顺序命令，但每条命令独立拥有：

```text
ChangeProposal 1:1 ApprovalRequest 1:0..1 ApprovalToken 1:0..1 Execution
```

Proposal 保存动作模板、参数、影响、风险、前置条件、验证和回滚方案。用户批准的是该 Proposal Hash，不是 LLM 的自然语言摘要。修改任一参数、目标、模板版本或回滚方案均创建新 Proposal 并重新审批。

多命令执行时严格串行：下一条命令只有在上一条执行成功并完成必要验证后才进入待审批。这样用户批准时看到的是当前真实状态，而不是执行前已经过期的整体计划。

## 命令状态机

```text
Action Draft → Policy DENY（不创建 Proposal）
                    └→ Proposal.ADVISORY_READY → Manual Result → Verify
                    └→ Proposal.PENDING_APPROVAL
                              ├→ REJECTED
                              ├→ EXPIRED
                              └→ APPROVED → CONSUMED
                                      Execution.CREATED → SUBMITTED → RUNNING
                                                             ├→ SUCCEEDED → VERIFYING
                                                             ├→ FAILED
                                                             ├→ TIMED_OUT
                                                             └→ UNKNOWN
```

`SUCCEEDED` 只表示 Executor 完成命令；处理结果只有在 Verify/Comparison 后才能标记为 `IMPROVED/UNCHANGED/DEGRADED/INCONCLUSIVE`。

## 审批权限与令牌

当前阶段不实现 Scope 或 Target ACL。批准请求必须来自已认证门户 API Key，AuthContext Domain 必须与 Proposal 一致，`asserted_user_id` 必须匹配当前待审操作人。首期不强制发起人与批准人分离；同一用户可批准自己发起的 Proposal，但仍必须进行一次显式批准并留痕。

一次性 Approval Authorization 绑定：

```text
proposal_id, proposal_hash, target_id, target_version
action_template_id, action_version, parameters_hash
approver_id, policy_decision_hash, issued_at, expires_at, nonce
```

`KBOT_OPS_APPROVAL_TOKEN` 是审批授权记录，不是返回给浏览器的 Bearer Token。它只能由对应 Execution 在 Claim 时消费一次；拒绝、取消、超时、Target/参数/策略变化后均不可复用。过期或撤销后如需再次执行，必须创建新的 Proposal Version 并重新审批。Executor 获得的是另行签发、绑定实例和短期 audience 的 `MutationExecutionGrant`。数据库中只保存 Claims/Nonce/Grant Hash 和消费事实，不保存可重放的明文凭据。

## Advisory 与人工处理

`ADVISORY` 模式同样使用版本化动作模板生成命令，但不创建 Approval Token 或 Execution。用户可回填 `EXECUTED/FAILED/CANCELLED`、执行时间、输出和备注，形成 `MANUAL_ACTION_RESULT` Artifact。

回填结果仅证明用户声称已处理，不直接证明故障已解决。AIOps Agent 仍必须用监控源或只读数据库执行 Verify/Comparison；无可验证数据时结论为 `INCONCLUSIVE`。

## API 与审计

- `GET /api/v1/ops/approvals?status=PENDING` 查询当前用户有权审批的命令；
- `GET /api/v1/ops/proposals/{proposal_id}` 返回完整命令、参数、影响、证据、回滚和验证方案；
- `POST /api/v1/ops/proposals/{proposal_id}/approve` 显式批准一条命令；
- `POST /api/v1/ops/proposals/{proposal_id}/reject` 拒绝并记录可选原因；
- `POST /api/v1/ops/proposals/{proposal_id}/manual-result` 回填 Advisory 执行结果。

审计事件包含 actor/service identity、AuthContext、Target、Proposal/Policy/Template Hash、前后状态、时间、IP/Client 和 trace ID。审批 API 是独立 Command，不使用聊天自然语言中的“同意”作为批准信号。
完整 API、DTO 和幂等规则见 [27_aiops_api_and_contracts.md](27_aiops_api_and_contracts.md)。

## 失败与恢复

- 重复批准返回已有结果，不发放第二个 Token；
- 批准后 Dispatcher 崩溃由 Outbox 恢复，Executor 使用幂等键防止重复执行；
- Execution 状态不明时先向 Executor 对账，不盲目重试变更命令；
- 验证失败只创建新的回滚 Proposal；首期任何回滚都必须重新显式审批；
- 部分执行的多命令计划停在失败命令，未开始命令的旧 Proposal 标记 `SUPERSEDED`。

## 验收

- 只读诊断、Chat 人工 SQL、Advisory 和 Agent Execute 不能相互越权；
- 一个 Approval 只对应一条命令和一个 Proposal Hash；
- Alert/Schedule 无数据库证据时不进入人工 SQL 循环，但可将结构化变更提案送入待审列表；
- 任何参数、Target、策略或模板版本变化都使原审批失效；
- 重试、重放、超时、并发审批和 Worker 崩溃都不会导致重复变更。
