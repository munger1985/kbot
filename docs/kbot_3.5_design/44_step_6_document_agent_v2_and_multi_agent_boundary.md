# 步骤 6 详细设计：DocumentAgentV2 与多 Agent 边界

## 调用方向

V1 的调用方向是 AskDoc Skill 实例化 DocAgent，再经过 DocOrchestrator 和 DocService 访问 Chunk 检索。V2 不复制这组倒置且职责重叠的层次，调用方向固定为：

```text
Root Agent / future Agent Router
  → DocumentAgentV2
      → KnowledgeRetrievalSkillV2
          → KnowledgeCoreClient
              → KC QueryPlan → Discovery Selector → Evidence Support Judge
```

Agent 调用 Skill，Skill 不能创建或持有 Agent。`DocumentAgentV2` 通过 Skill Registry/依赖注入获得能力；`KnowledgeRetrievalSkillV2` 是无状态、可复用的受控工具适配器；KC 是检索规则、版本、Scope 和证据事实的唯一所有者。V2 不再建立等价于旧 `DocService/TxtBaseSearch` 的本地检索实现。

## 职责

| 组件 | 负责 | 不负责 |
| --- | --- | --- |
| Agent Router/Root | 顶层问文/问数/混合路由、并行委派、最终综合 | KC 候选排序和 Evidence 查询 |
| `DocumentAgentV2` | 知识任务状态、历史引用约束、澄清、有限重试、调用 Skill、返回知识任务结果 | SQL、KC 表访问、RRF/Selector/Judge 算法 |
| `KnowledgeRetrievalSkillV2` | 校验工具输入、调用 KC、组装 Citation Pack、覆盖检查 | 会话记忆、候选自由选择、最终自然语言回答 |
| Knowledge Core | QueryPlan、Binding/Scope、Discovery、Evidence、LLM Selector/Judge、稳定定位 | Agent 规划、跨领域综合、最终 SSE |

## Agent 边界 DTO

未来多 Agent 不共享可变 `ContextMemory` 作为隐式协议，而使用版本化任务 DTO：

```text
KnowledgeTask {
  task_id, parent_run_id, domain_id, agent_id,
  original_query, standalone_query, response_goal,
  explicit_collection_keys?, explicit_facets?,
  prior_citations?, security_context, deadline
}

KnowledgeTaskResult {
  task_id, status,
  citation_pack?, grounded_findings?,
  coverage_gaps[], clarification?,
  retrieval_run_id, warnings[], diagnostics_ref?
}
```

`prior_citations` 只能来自前序已验证结果，不能接受模型自由生成的 Bundle/Document ID。`grounded_findings` 是带 Citation Group label 的知识结论，不是跨 Agent 最终答案；未来 Root/Synthesis Agent 可将它与 `DataQueryResult.query_result_id` 并行综合，且分别保留来源。

## 3.5 落地与 4.0 演进

3.5 同步实现 V2 Skill 与 `DocumentAgentV2`，但只要求 Root 的知识任务能够委派给该 Agent；不在本期重构全部 Agent。纯问文和未来多 Agent 使用同一 `KnowledgeTask/KnowledgeTaskResult` 契约，避免 4.0 再拆接口。

4.0 的 Agent Router 可并行委派 Document Agent、Data Agent 或其他领域 Agent。Document Agent 仍只通过 Skill 调 KC；其他 Agent 如需检索能力，可通过 Skill Registry 复用 `KnowledgeRetrievalSkillV2`，但必须携带自己的授权执行上下文，不能借用 Document Agent 身份扩权。

链路统一传播 `parent_run_id/task_id/retrieval_run_id`，设置 deadline、取消和有限重试。KC 的 `CANDIDATE_STALE` 最多重跑一次；证据不足由 Document Agent 澄清或如实返回，不能回退 V1、扩大 Collection 或让 Root 用常识补齐。

## 验收

- 代码依赖中不存在 Skill import/instantiate Agent；Agent 仅通过注册能力调用 Skill。
- `DocumentAgentV2` 的单元测试使用 mock Skill，Skill 的契约测试使用 mock KC Client，KC 测试不 import Agent/Skill。
- Root 与 Document Agent 之间只传版本化 DTO；替换 Agent 实现不影响 KC API。
- 混合任务能并行保留 Citation Pack 和 `query_result_id`，最终答案不得丢失任一来源的审计链。
