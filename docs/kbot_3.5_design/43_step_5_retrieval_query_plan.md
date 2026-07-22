# 步骤 5 详细设计：Retrieval QueryPlan

## 定位

Root Agent 已负责会话指代消解、生成 `standalone_query`，并把任务路由为问文、问数或混合任务。KC 的 `RetrievalQueryPlanner` 不重复这些职责，只把已确定的知识检索请求编译为 Discovery/Evidence 策略。KC 不读取完整对话历史；`DocumentAgentV2` 仅传入上轮已验证 Citation identity，KC 重新校验权限、当前 Revision 和可见状态。

QueryPlan 使用正交维度而不是单一 intent：

```text
task_mode: DISCOVER | ANSWER | SUMMARIZE | COMPARE
target_level: BUNDLE | DOCUMENT | EVIDENCE
coverage_mode: BREADTH | DEPTH | BALANCED | STRUCTURAL
evidence_preferences: TEXT | TABLE | TABLE_ROW | IMAGE | SHEET | CELL_RANGE
```

例如，“列出零售 AIOps 案例”为 `DISCOVER+BUNDLE+BREADTH`；“这个方案如何告警降噪”为 `ANSWER+EVIDENCE+DEPTH`；附件总结使用 `SUMMARIZE+DOCUMENT+STRUCTURAL`，不能退化为语义 Top-K。QueryPlan 不因 Bundle 只有一个 Document 而改变目标层次；Discovery 会使用 `SINGLE_MEMBER` 快速路径自动确定 Evidence Scope。

## 输入与输出

输入至少包含原始问题、`standalone_query`、认证 Domain/Agent、安全上下文、可选 Collection 收窄条件、显式 Facet 和历史有效 Citation references。输出为版本化 QueryPlan：

```text
query_plan_id/version, task/target/coverage dimensions,
lexical { exact_phrases, identifiers, required/optional/excluded_terms },
semantic_query, hard_filters, soft_facets,
resolved_references, discovery_policy_key, evidence_policy_key,
confidence, warnings, plan_status
```

完整 `standalone_query` 供向量召回和 LLM Selector/Judge 使用。Oracle Text 使用分别保存的精确短语、业务 ID 和可选扩展词，不再把全部词清洗后拼成单个 ACCUM 字符串。首期只有一个主语义查询；多查询改写和 HyDE 不作为默认能力。

## Facet 与 Scope

Domain、ACTIVE Agent Binding、Collection 状态、安全等级、当前 Revision、READY Member 和 ACTIVE View/Evidence 是强制 Scope，不是 LLM 输出。

业务 Facet 只有同时满足以下条件才可进入 `hard_filters`：调用方显式传入或用户表达明确字段条件；Facet Schema 声明为可过滤；值可映射到受控字典；字段完整度满足策略门槛。LLM 从自然语言推断的行业、产品、场景、模糊时间和地域默认进入 `soft_facets`，只能参与画像召回、解释或排序。LLM 可以建议 Facet，但不能扩大 Collection Scope 或自行产生硬过滤。

## 规划流程与失败

1. 确定性提取显式 API 条件、引号短语、ID、文件名、时间范围和历史 Citation。
2. 通用轻量 LLM 以严格 Schema 输出四个维度、软 Facet 和 Evidence 类型偏好。
3. KC 使用 Collection Facet Schema、策略注册表和权限上下文校验结果。
4. LLM 不直接生成 Top-K；它选择版本化 Policy，具体候选数、批大小和 token 预算由 Policy 决定。

模型失败、非法枚举或 Facet 校验失败时，丢弃不可信字段并使用 `ANSWER+EVIDENCE+BALANCED`，返回显式 `plan_status=DEGRADED_DEFAULT`。涉及多个历史对象且无法唯一解析“这个/该附件”时返回 `CLARIFICATION_REQUIRED`，由 `DocumentAgentV2` 决定澄清，不能任选一个对象。

## 验收

- 相同输入、Facet Schema、Policy 和模型版本可重放；QueryPlan 写入 `retrieval_run_id` 追踪。
- 自然语言推断 Facet 不会成为静默硬过滤；非法 Collection 或跨 Domain reference 在规划前被拒绝。
- BREADTH、DEPTH、BALANCED、STRUCTURAL 分别用对象覆盖、证据深度、比较覆盖和章节覆盖指标验收。
- QueryPlan 不包含 V1 `kb_id/search_top_k/tool_weight/reranker_flag`，也不读取旧 File/Chunk 表。

当前代码已提供 `RetrievalQueryPlanner` 和版本化 QueryPlan DTO，能稳定提取短语、业务标识和默认降级状态；自然语言 Facet 推断及模型托管接入仍待评测后启用。
