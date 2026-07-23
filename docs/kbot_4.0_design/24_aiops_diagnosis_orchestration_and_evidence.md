# 4.0 AIOps 诊断编排与根因证据

步骤 7 的可实施级状态机、Artifact、模型端口、等级策略和验收设计见 [36_aiops_step7_diagnosis_orchestration_and_llm.md](36_aiops_step7_diagnosis_orchestration_and_llm.md)。

## 设计目标

诊断内核负责将告警、指标、数据库观测和 SOP 转化为可回溯的根因结论。它不是让 LLM 自由生成一串 Skill，而是在确定性状态机中让 LLM 完成语义判断、假设生成和证据综合。

```text
Trigger Context
      ↓
Scope + Time Window
      ↓
Initial Observation Plan
      ↓
Monitor / DB / Log / SOP Collectors
      ↓
Normalized Evidence Pack
      ↓
Hypothesis Generation
      ↓
Evidence Gap + Hypothesis Tests
      ├─ 补充预置工具 ──┐
      ├─ Chat 人工 SQL ─────┤
      └─ 无法补充 → INCONCLUSIVE │
                   ↑──────────────┘
      ↓
Root Cause Decision
      ↓
Solution / Proposal / Report
```

## 确定性与 LLM 边界

| 环节 | 责任方 | 约束 |
| --- | --- | --- |
| Target/权限/触发解析 | 确定性代码 | 不由 Prompt 决定 domain 和 Target |
| 时间窗口、时区、指标映射 | 确定性代码 | 记录映射版本和采集时间 |
| 初始诊断意图和工具选择 | LLM + Validator | 只选已注册 `tool_id` |
| 并行采集、超时、重试、脱敏 | 确定性 Worker | 不允许 LLM 绕过 Adapter |
| 假设、因果机制和缺口分析 | LLM | 输出强类型 Artifact |
| 证据引用、反证检查、置信级别 | Validator + LLM | 引用必须指向存在的 Fact；确定性策略限制等级上限 |
| Policy、审批和执行 | 确定性代码 | 根因置信度不等于执行权限 |

LLM 不直接读取无上限的原始时序、日志或 SQL 结果。Adapter 先完成聚合、单位归一、截断、脱敏和 schema 校验，再交给诊断模型。

## 诊断输入与时间窗口

`DiagnosisScope` 包含 Target、数据库类型/版本、环境、触发类型、症状、影响范围、时区、分析窗口和基线窗口。

- Chat 优先从问题提取时间，缺失时使用场景默认窗口；
- Alert 以 `starts_at/first_seen_at` 为中心，包含告警前基线和告警后窗口；
- Inspection 由模板固定窗口，日报/周报不允许 LLM 任意改变统计区间；
- 所有时间在契约中使用 UTC，展示层再转换为用户时区。

不得将不同窗口或不同采样粒度的峰值直接对比。当时钟偏差、缺失点或采样延迟超过阈值时，Evidence 标记质量警告。

## Evidence Pack

Evidence Pack 只引用不可变 Artifact：

```text
EvidenceRef {
  artifact_id, artifact_type
  source_type, source_id
  captured_at, window_start, window_end
  schema_version, content_hash
  trust_level, quality_flags
  fact_summary
}
```

`trust_level` 至少区分：

- `SOURCE_VERIFIED`：由监控 Adapter 或只读 DB Executor 直接获得；
- `USER_PROVIDED`：Chat 中用户回贴的查询结果；
- `KNOWLEDGE_CITATION`：Knowledge Core 返回的 SOP/案例 Evidence；
- `MODEL_INFERENCE`：假设、摘要或结论，不能冒充观测事实。

SOP 用于解释机制、选择诊断步骤和生成解决思路，不能证明当前 Target 正在发生某个问题。

## 假设与证伪

LLM 生成 `DiagnosisHypothesis[]`：

```text
hypothesis_id, statement, mechanism
explains_symptoms[], supporting_evidence[]
counter_evidence[], missing_evidence[]
test_actions[], status, confidence_level
```

`test_actions` 只能引用监控指标、预置诊断 `tool_id` 或 Chat 人工 SQL 请求。每轮优先执行能区分多个假设、开销最低的测试，不为每个假设无限采集数据。

每轮结束时必须对每个假设标记 `SUPPORTED/WEAKENED/REJECTED/UNTESTED`，并显式保留反证。新 Evidence 与原结论冲突时修订 Diagnosis Artifact，不改写旧 Artifact。

## 根因判定

根因级别不使用一个无法解释的浮点分数：

| 级别 | 判定条件 |
| --- | --- |
| `CONFIRMED` | 直接观测、时间顺序和机制一致，且关键替代假设已被反证 |
| `PROBABLE` | 有多项一致证据和可解释机制，但仍缺少一项直接验证 |
| `POSSIBLE` | 仅能解释部分症状，证据不足或存在强替代假设 |
| `INCONCLUSIVE` | 无法区分关键假设、数据质量不足或观测相互冲突 |

只有 `CONFIRMED/PROBABLE` 可生成定向解决方案。`POSSIBLE/INCONCLUSIVE` 只输出候选原因、缺口、风险较低的缓解建议和后续诊断方法，不自动生成变更 Proposal。

## 三种触发模式

### Chat

可以进行多轮澄清和 `MANUAL_DIAGNOSTIC_SQL` HITL。对话文本仅提供意图和补充信息，诊断状态始终来自 Ops Run/Task/Artifact。

### Alert

使用告警规则、事件指纹和预置 Playbook 建立初始范围。不等待人工诊断输入；证据不足时产出 `PARTIAL/INCONCLUSIVE` 报告。若已有 `CONFIRMED/PROBABLE` 根因和登记动作模板，本步骤只输出候选 Action Template Ref；步骤 9 才可创建待审批 Proposal。

### Inspection

使用版本化 Inspection Template 固定必检项、阈值、基线和报告 schema。LLM 用于解释异常联系和总结，不得删除失败检查项或改写阈值。

## 预算、并发与终止

Run 固定下列预算：最大诊断轮次、最大工具数、每 Target 并发采集数、最大原始结果大小、LLM Token 和总截止时间。

同一轮中无依赖的 Monitor/DB/SOP Task 可并行；同一 Target 的重型 SQL 按 `cost_level` 限流。达到预算、用户取消、Target 进入维护/停用或数据源持续失败时终止新采集，保留已完成 Artifact 并生成部分报告。

## 报告与解决方案

Diagnosis Artifact 的固定结构：

1. 问题范围和时间线；
2. 已验证事实及 EvidenceRef；
3. 候选假设、支持证据和反证；
4. 根因级别和因果机制；
5. 影响范围和已知局限；
6. 短期缓解、长期修复和验证方法；
7. 可选的候选 Action Template 引用；ChangeProposal 由步骤 9 独立生成。

解决方案必须区分“建议”、“待批准命令”、“已执行事实”和“验证结果”，Response Composer 不得把任一建议表述为已执行。

## 从 3.x 迁移

- 保留已有监控 Provider、指标映射和 DBA 诊断意图作为评审输入；
- 不迁移整份 `OpsContextMemory`、线性动态 Skill Plan 或 Prompt 内的权限决策；
- SOP 检索统一经 Knowledge Core Client 获取 CitationPack，AIOps Planner 不调用 Document Orchestrator 内部对象；
- 报告从“一次 LLM 总结”改为基于不可变 Artifact 和可验证引用的结构化产物。

## 验收

- 每个根因结论都能回链至监控、数据库、用户回贴或 KC Evidence；
- 时间窗口、指标单位、采样粒度或数据质量异常会降低结论级别；
- 有强反证的假设不能标记 `CONFIRMED`；
- `POSSIBLE/INCONCLUSIVE` 不能生成自动变更 Proposal；
- Chat、Alert 和 Inspection 使用同一诊断 Artifact schema，但只有 Chat 能进入人工 SQL 循环；
- 同样的 Evidence Pack 和 Prompt/Model 版本可重放并对比诊断结果。
