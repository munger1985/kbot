# AIOps 步骤 7：诊断编排、证据判定与 LLM 接入

## 目标与非目标

本步骤把监控、只读数据库诊断和 Knowledge Core SOP 转化为可追溯的候选假设、补证计划、根因等级和解决方案草稿。AIOps Agent 仍是一个独立领域服务；“假设生成器、证据规划器、验证器”是 Worker 内的版本化 Handler/Prompt Role，不拆成多个微服务或可自由互调的 Agent。

本步骤不实现人工 SQL 回贴、Change Proposal、审批或命令执行。LLM 不能创建权限、改变 Target/时间窗口/预算、提交 SQL、调用工具或推进 Run 状态。

## 设计原则

LLM 擅长提出解释和理解非结构化上下文，但不是事实源、授权源或状态机。完整边界为：

```text
Deterministic Orchestrator
  ├─ Scope/Window/Budget/Capability Snapshot
  ├─ Evidence Normalizer + Index
  ├─ Prompt/Input Pack Builder
  └─ LLM: hypotheses + evidence requests
          ↓ strict schema
      Plan Validator
          ↓ approved read-only Tasks
  Monitor / DB Executor / KC
          ↓ immutable Artifacts
      Evidence Assessment + Grounding Verifier
          ↓
      RootCauseGradePolicy
          ↓ effective level
      Solution Draft + Report Draft
```

任何通过基础 Schema 校验的模型输出都先保存为 `MODEL_INFERENCE`，再经引用、权限、预算和根因等级校验后成为下一阶段输入；无效原文只记录 Hash 和拒绝原因。Planner 的建议不等于可执行计划，模型置信度也不等于根因置信度。

## 对现有模型调用方式的取舍

当前通用 `AIModelClient.get_llm_json()` 依赖 Prompt 注入 JSON 指令、从自由文本提取 JSON、自动补全截断括号，并可能记录原始响应预览。它适合旧流程兼容，不适合作为 AIOps 状态迁移边界。

新增窄接口 `AIOpsModelPort`：

```text
generate_structured(
  purpose, schema_id,
  model_snapshot, prompt_ref,
  input_artifact_refs, input_payload,
  max_output_tokens, deadline,
  idempotency_key
) -> StructuredModelResult
```

- 只调用 Model Serving API，不在 AIOps Worker 加载模型；
- 模型配置通过 Agent Runtime/Model Management Client 解析，AIOps 不查询模型 Entity；
- Run 创建时冻结 `model_id/revision/technical_name/capabilities`，Prompt 使用部署资产的版本与 SHA-256；
- 优先使用 Provider 原生 Structured Output/JSON Schema；无原生支持时仍必须通过严格 Pydantic/JSON Schema；
- 禁止本地截断修复、平衡括号提取或接受额外字段；
- Schema 失败最多进行一次受预算约束的模型修正调用，仍失败则生成结构化 Model Gap；
- 使用进程级连接池、Service Identity、有限重试和 Deadline，不为每次调用创建全新 HTTP Session；
- 不记录 Prompt、模型原文、隐藏推理或证据正文，只记录 Hash、稳定错误码和使用量。

`MODEL_INVOCATION_RECEIPT.v1` 保存模型/Prompt/Input/Output Hash、Provider Request ID、Token、耗时、finish reason 和修正次数。系统只保存简洁的因果解释，不请求或持久化 Chain of Thought。

## 可恢复诊断状态机

步骤 6 的确定性 Baseline 作为第 0 轮输入。步骤 7 在现有 Run/Task 内核上增加以下 Handler，不增加新的状态机表：

```text
SCOPE
  → COLLECT_BASELINE
  → BUILD_EVIDENCE_INDEX
  → DIAGNOSIS_ROUND_DRAFT[r1]
  → VALIDATE_EVIDENCE_REQUESTS[r1]
  → COLLECT_APPROVED_EVIDENCE[r1,*]
  → ASSESS_DIAGNOSIS_ROUND[r1]
  → ROUND_DECISION
       ├─ CONTINUE → r2 ... rN
       ├─ WAIT_FOR_CHAT_INPUT → 步骤 8
       └─ FINALIZE
  → ROOT_CAUSE_ASSESSMENT
  → GROUNDING_VERIFICATION
  → SOLUTION_DRAFT
  → REPORT_DRAFT
```

Task 仍使用 `OBSERVE/DIAGNOSE/REPORT` 等粗粒度 `TASK_TYPE`，具体职责由锁定版本的 `HANDLER_ID` 表示。动态轮次只能通过 `ExpandDiagnosisRoundCommand` 在一个 UoW 中创建，Task Key 固定为：

```text
diagnosis:r{round}:draft
diagnosis:r{round}:validate
diagnosis:r{round}:collect:{request_key}
diagnosis:r{round}:assess
diagnosis:finalize
diagnosis:verify
diagnosis:solution
diagnosis:report
```

初始 `PLAN_SNAPSHOT_JSON` 不被覆盖；每次扩展生成不可变 `DIAGNOSIS_PLAN_REVISION.v1` Artifact，记录前置 Artifact、创建的 Task、预算变化和决策原因。命令校验 Run/Lease、轮次、唯一 Task Key、依赖无环和剩余预算后原子创建 Task/Event。进程崩溃后 Runtime 只恢复未完成 Task。

## Scope 与输入冻结

`DIAGNOSIS_SCOPE.v1` 由确定性代码建立：

```text
run_id, target_id, agent_id, trigger_type
symptom_codes, user_question_summary
window_start, window_end, baseline_start, baseline_end, timezone
db_type/version, environment, target_capabilities
monitor_coverage, allowed_collection_ids
policy_snapshot_hash, budget_snapshot, security_level
```

Alert/Schedule 的窗口由规则或 Inspection Template 固定；Chat 可从用户文本提取候选时间，但最终窗口由服务端规则裁剪并使用 UTC。LLM 可补充症状语义标签，不能扩大 Target、Domain、Collection、窗口或权限。

## Evidence Index 与事实模型

模型不直接读取无上限的时序点、SQL 行集或文档。每类 Adapter 提供确定性 Normalizer，将每个不可变来源 Artifact 分别转换为 `EVIDENCE_INDEX.v1`：

```text
EvidenceFact {
  fact_id
  source_artifact_id, source_json_pointer
  source_type, source_group_id, trust_level
  target_id, observed_subject
  metric_or_fact_type, value, unit, dimensions
  window_start, window_end, captured_at
  quality_flags, security_level
  fact_summary
}
```

`fact_id` 由来源 Artifact、JSON Pointer 和规范化内容 Hash 派生。`source_group_id` 表示共同来源血缘，例如同一 Prometheus 查询、同一 DB Diagnostic 或同一用户上传，防止把重复加工后的数据算成多个独立证据。

每个 Evidence Index 继承其单一来源 Artifact 的 `TRUST_LEVEL`，不把不同信任等级混装在一个 Artifact 中。模型输入的 Evidence Pack 只是一组 Index/FactRef 和规范化摘要，不制造新的“派生事实”或改变来源信任等级。

证据角色严格区分：

| Trust Level | 可证明当前 Target 状态 | 用途 |
| --- | --- | --- |
| `SOURCE_VERIFIED` | 是 | 监控和 DB Executor 事实 |
| `USER_PROVIDED` | 有限 | 步骤 8 的人工结果，必须带质量警告 |
| `KNOWLEDGE_CITATION` | 否 | SOP、产品机制、历史案例和处理依据 |
| `MODEL_INFERENCE` | 否 | 假设、解释、总结和方案 |

Knowledge Citation 和历史案例不能把根因从 `POSSIBLE` 提升到 `PROBABLE/CONFIRMED`。模型生成的摘要也不能作为下一轮的新事实重复计数。缺失、截断、采样延迟、时钟偏差、单位不兼容和脱敏影响均显式保留在 `quality_flags`。

## 每轮模型契约

### 诊断草稿

`DIAGNOSIS_ROUND_DRAFT.v1`：

```text
round_no
hypotheses[] {
  hypothesis_key, existing_hypothesis_id?
  statement, mechanism, causal_role
  explained_symptom_codes[]
  supporting_fact_refs[], counter_fact_refs[]
  unresolved_questions[]
}
evidence_requests[] {
  request_key, tool_id, parameters
  hypothesis_keys[]
  diagnostic_question
  supports_if, contradicts_if
  priority_reason
}
stop_recommendation, stop_reason
```

模型不能填写 Target、工具版本、SQL、查询语句、Credential、Timeout、Cost 或并发。新假设由服务端分配 UUIDv7；已有假设只能按 ID 修订状态，旧 Artifact 不覆盖。

### 轮次评估

`DIAGNOSIS_ROUND_ASSESSMENT.v1`：

```text
round_no, new_fact_refs[]
hypothesis_assessments[] {
  hypothesis_id
  status: SUPPORTED | WEAKENED | REJECTED | UNTESTED
  causal_role: ROOT | CONTRIBUTOR | SYMPTOM | COINCIDENTAL
  supporting_fact_refs[], counter_fact_refs[]
  test_results[] {
    request_key
    outcome: SUPPORTS | CONTRADICTS | NEUTRAL | UNAVAILABLE
    strength: DIRECT | CORRELATED | CONTEXTUAL
    fact_refs[]
  }
  remaining_gaps[]
}
recommended_next_step, rationale_summary
```

同一模型可以提出解释，但不能自行声明最终根因等级。对可结构化判定的指标阈值、空结果和状态枚举，优先使用工具 Manifest 的确定性 Evaluator；LLM 只解释无法通过规则表达的组合关系。

## Evidence Request Validator

模型建议先转换为 `EvidenceRequestDraft`，由 Validator 决定接受、合并或拒绝：

1. `tool_id` 存在于本 Run 的 Capability Snapshot，且运行模式为只读；
2. Monitor Metric、DB Diagnostic 或 KC Skill 版本由 Registry 锁定，模型不能覆盖；
3. 参数严格通过 Schema，窗口在 Scope 内，Target/Domain/Collection 不可变；
4. DB 工具满足版本、权限、entitlement 和 Cost Level；
5. 请求明确关联假设并声明可区分的预期结果；
6. 工具调用数、并发、Token、字节和 Deadline 未超预算；
7. 相同 `tool_id + parameters + window + source` 指纹不重复执行，除非输入明确声明新时间窗口；
8. 请求不能创建 Mutation、人工 SQL、Shell、远程命令或未知 Adapter。

被拒绝项记录稳定原因，例如 `TOOL_NOT_AVAILABLE`、`PARAMETER_INVALID`、`ENTITLEMENT_REQUIRED`、`DUPLICATE_REQUEST`、`BUDGET_EXCEEDED`。可接受请求由 Application Service 创建 Task；LLM 输出本身从不直接成为 Task DAG。

SOP 检索使用服务端授权的 AIOps Collection Binding。模型只能给出检索意图和关键词，Knowledge Core Client 始终携带冻结的 Domain/Collection 范围，并返回 `CitationPack`。

## 根因等级的确定性上限

模型输出建议等级后，`RootCauseGradePolicy` 根据已验证证据计算 `eligible_ceiling`，最终等级只能取更低者。系统不使用不可解释的浮点分数。

| 等级上限 | 最低条件 |
| --- | --- |
| `CONFIRMED` | 存在当前 Target 的直接致因条件观测和时间一致的症状证据；关键替代假设已被反证；无未解决强反证或关键质量缺口 |
| `PROBABLE` | 至少两个独立的当前状态证据组支持同一机制，其中至少一个区分性测试为支持；主要替代假设已弱化；仍缺一项直接闭环验证 |
| `POSSIBLE` | 有当前状态事实支持部分症状，但机制主要依赖推断、独立性不足或强替代假设尚未排除 |
| `INCONCLUSIVE` | 没有稳定假设，证据冲突/质量不足，或无法区分关键假设 |

额外降级规则：

- 只有 SOP、案例或模型解释时固定为 `INCONCLUSIVE`；
- 只有单组用户回贴证据时不能达到 `CONFIRMED`；
- 强反证未解释、关键窗口错位、单位不兼容或关键结果截断时最高为 `POSSIBLE`；
- 证据来自同一原始来源的多个派生 Artifact 只算一个独立组；
- “相关”不能自动转换为“因果”，必须在 causal chain 中标为限制。

`ROOT_CAUSE_ASSESSMENT.v1` 保存模型建议等级、Policy 上限、最终等级、主因、贡献因素、因果链 Claim、支持/反证 FactRef、未解决 Gap 和降级原因。只有该 Artifact 成功提交后，Application Service 才更新 `RUN.ROOT_CAUSE_LEVEL` 和 `STATUS=DIAGNOSED`。

## Grounding Verifier

最终诊断使用“模型检查 + 确定性 Gate”，而不是让另一个 LLM 单独裁决：

- 确定性检查每个事实性 Claim 均有存在且可见的 FactRef；
- 检查 FactRef 的 Target、窗口、单位、Trust、质量和来源独立性；
- 检查解决建议是否与根因等级相符，是否包含未登记命令或“已经执行”等虚假状态；
- 独立 Verifier Prompt 检查引用是否语义支持 Claim、是否忽略反证或混淆相关与因果；
- Verifier 只能提出 `PASS/REVISE/BLOCK` 和问题列表，不能提升根因等级或授权工具；
- 最多一次受限修订；仍为 `BLOCK` 时输出模板化 `INCONCLUSIVE` 报告并保留所有 Gap。

Verifier 可以与生成器使用同一模型服务，但使用独立 Prompt、Schema 和调用记录；生产可配置不同模型。任何情况下，LLM-as-a-judge 都不是权限或执行安全边界。

## 终止、降级与无进展检测

每次 `ROUND_DECISION` 由确定性代码选择：

- `FINALIZE`：达到证据充分条件，或再采集不会改变等级；
- `CONTINUE`：存在有效、可执行且有区分度的新请求；
- `WAIT_FOR_CHAT_INPUT`：仅 Chat 且需要用户补证，按 [37_aiops_step8_chat_manual_diagnosis_hitl.md](37_aiops_step8_chat_manual_diagnosis_hitl.md) 在步骤 8 启用；
- `STOP_INCONCLUSIVE`：达到轮次/工具/Token/Deadline，数据源不可用，或无进展。

默认预算建议从 `max_rounds=3`、`max_tool_calls=12`、`max_schema_repairs=1` 起步，并由 Policy/环境配置调整。预算只可缩小，不可由模型扩大。

无进展满足任一条件即终止：

- 新旧 Evidence Index Hash 相同；
- 所有请求均被拒绝、重复或不可用；
- 连续两轮 Hypothesis 状态和有效请求指纹未变化；
- 剩余时间不足以完成最小下一轮。

监控/数据库证据缺失但模型正常完成时，Run 可 `COMPLETED` 且报告标记 `PARTIAL/INCONCLUSIVE`；模型服务持续不可用或输出始终无效时，系统用已验证事实生成模板化报告并将 Run 标为 `DEGRADED`。只有状态机、契约或持久化损坏才进入 `FAILED`。

## Solution 与报告草稿

`SOLUTION_DRAFT.v1` 只能包含：

```text
immediate_mitigations[]
long_term_remediations[]
candidate_action_template_refs[]
risks[], prerequisites[]
verification_plan[]
knowledge_citations[]
limitations[]
```

`CONFIRMED/PROBABLE` 可给出定向方案和候选 Action Template Ref；`POSSIBLE/INCONCLUSIVE` 只能给出低风险缓解、补证方法和限制。本步骤不创建 `KBOT_OPS_CHANGE_PROPOSAL`，不渲染最终命令，也不把建议表述为已执行。步骤 9 才进行 Policy Gate、参数冻结、逐命令审批和执行。

`DIAGNOSIS_REPORT_DRAFT.v1` 固定包含 Scope、时间线、事实、假设、反证、根因等级、Gap、方案、Citation 和 Model/Prompt Provenance。Conversation Composer 后续只能改变表达，不能改变 FactRef、根因等级、命令状态或审批状态。

## Prompt 与数据安全

- System Prompt 来自受控 Platform Prompt Registry：运行时数据库 Active
  Version 优先、统一文件兜底，并冻结版本与 SHA-256；APEX、用户请求以及
  目标数据库中的业务文本不能修改或覆盖 Prompt。统一注册和初始化规则见
  [46_versioned_prompt_registry.md](46_versioned_prompt_registry.md)；
- 用户输入、告警标签、数据库 SQL Text、日志、SOP 和案例全部标记为不可信数据，并以结构化字段传入；
- Evidence 中出现“忽略规则”“执行命令”等文字只作为数据，不进入指令层；
- 输入按 Security Level、字段白名单、Token 和字节上限裁剪；Secret、DSN、账号、内部 Token、Lease Token 永不进入模型；
- Tool Cards 由服务端 Registry 生成，只暴露允许工具的用途和参数 Schema；
- 输出设置 `extra=forbid`，拒绝未知工具、未知 Artifact、伪造 FactRef 和自然语言动作；
- 用户可见 SSE 只报告阶段、轮次、有效请求数、Gap 和 ArtifactRef，不发送 Prompt、模型原文或内部验证细节。

## 代码布局

```text
aiops_agent/
  contracts/diagnosis/
    scope.py
    evidence.py
    hypothesis.py
    root_cause.py
    solution.py
  domain/diagnosis/
    grade_policy.py
    round_decision.py
    validators.py
  application/diagnosis/
    build_evidence_index.py
    expand_round.py
    finalize_diagnosis.py
  orchestration/diagnosis/
    handlers.py
    blueprints.py
    prompt_registry.py
  ports/model.py
  adapters/model_serving.py
  tests/diagnosis/
```

Normalizer 与 Validator 不依赖 HTTP、Repository 或模型 Client。Model Adapter 不创建 Task、不访问数据库 Session。所有外部调用发生在 UoW 外，Artifact/Task/Event 仍由 Runtime Application Service 原子提交。

## 实施顺序

1. 固化 Scope、Fact、Hypothesis、Request、Assessment、RootCause 和 Solution Schema；
2. 为步骤 5/6 Artifact 实现确定性 Evidence Normalizer 与来源血缘；
3. 实现 Prompt Registry、AIOpsModelPort、Model Snapshot 和 Invocation Receipt；
4. 实现 Round Draft/Assessment Handler 及严格输出校验；
5. 实现 Evidence Request Validator、去重、预算和动态 Task 扩展命令；
6. 接入 Monitor、DB Diagnostic、KC Citation 三类只读证据；
7. 实现 Grade Policy、Grounding Verifier、无进展终止和模板化降级报告；
8. 建立 Oracle/MySQL 故障/性能 Golden Cases，完成离线回放和模型版本门禁。

## 验收门槛

- 模型不能提交 SQL、Mutation、未知工具、越权 Target/Collection 或扩大预算；
- 所有事实 Claim 可回链至不可变 Artifact 和 JSON Pointer，伪造引用会被拒绝；
- 同源派生数据不会被误算为独立证据，SOP/案例不能证明当前故障；
- 强反证、窗口错位、截断或质量缺口会确定性降低根因等级；
- 同样 Evidence Pack 可按 Model/Prompt 版本回放并比较，不承诺生成文本逐字一致；
- Prompt Injection、恶意 SQL Text/SOP、超长输入、Schema 污染和无效 JSON 均不能改变执行计划；
- LLM 超时、限流、无效输出和 Worker 崩溃可恢复且不会重复写 Artifact；
- 无进展循环在预算内终止，Alert/Schedule 不进入人工等待；
- `POSSIBLE/INCONCLUSIVE` 不创建 Proposal，步骤 7 的任何路径都不能调用 Mutation Executor；
- Golden Cases 评测工具计划有效率、引用正确率、反证保留率、根因等级校准、误报率、轮次、时延和 Token 成本。

## 当前实现结果

步骤 7 已实现 `diagnosis.root-cause@1` 有界 Blueprint：同一 Run 内先采集监控、Oracle/MySQL 只读基线和授权范围内的 KC Citation，再生成稳定 `EVIDENCE_INDEX.v1`。最多三轮 Draft → Request Validator → Catalog Collection → Assessment 使用预分配 Task 槽位；上一轮终止或无新 Evidence 时后续槽位确定性短路。该实现不在运行中修改 `PLAN_SNAPSHOT_JSON`，同时保留 Task 级租约、恢复和预算围栏。

模型通过新的 `AIOpsModelPort` 调用 Model Serving 原生 JSON Schema，Prompt 以版本和 SHA-256 冻结。模型只能输出假设、FactRef 与 `tool_id + parameters`；目录版本、Target、Secret、Grant、超时和执行权限仍由服务端决定。模型调用收据随推断 Artifact 保存，模型推断使用 `MODEL_INFERENCE` Trust Level，监控和 DB Executor 事实保持 `SOURCE_VERIFIED`，KC SOP 保持 `KNOWLEDGE_CITATION`。

确定性 `RootCauseGradePolicy` 会按来源独立性、直接测试、反证和质量标志计算等级上限；知识引用不能证明当前 Target 状态。模型或 KC 不可用时链路生成 `DEGRADED/INCONCLUSIVE` 报告，不调用 Mutation Executor。当前自动化覆盖结构污染、未知工具、参数 Schema、同源血缘、知识引用降级、三轮上限和 Prompt Hash；真实模型 Golden Cases 与多版本模型质量门禁留在统一验收阶段。
