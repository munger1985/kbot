# 步骤 6 详细设计：RootAgentV2 显式路由

## 路由隔离

新增 `/api/agent/v2/knowledge` 和 `/api/agent/v2/knowledge/streaming`。请求显式携带 `domain_id`、已授权 `collection_ids`、Agent 和安全等级；该入口只构造 `KnowledgeTask → RootAgentV2 → DocumentAgentV2 → KnowledgeRetrievalSkillV2`，不会调用 V1 RootAgent、AskDocSkill、DocService 或旧表。

V1 `/api/agent/chat-*` 路由保持不变。V2 SSE 先发送 `knowledge_task_result` 和
`citation_candidates_v2`，随后由回答模型生成结构化 Draft，再由
`AnswerGroundingVerifier` 校验，最终发送 `grounded_answer`、`doc_results_v2`、
`grounding_status` 和 `done`。最终引用不复用 V1 `doc_results`。

## 安全边界

路由层只接受上层已验证的 Domain/Collection Scope；KC 仍会重新校验 Collection 状态、当前 Revision、Evidence 状态和安全等级。V2 失败返回 `INSUFFICIENT_EVIDENCE` 或受控错误，不自动回退 V1。

## 已落地

- `RootAgentV2` 独立流式/非流式编排器。
- V2 Agent API Router 和请求 DTO。
- V2-only SSE 事件序列化测试。
- 回答模型适配器、Citation Pack 水合和最终 `doc_results_v2` 投影。

回答模型不可用时仍返回检索任务结果，但 grounding 状态为 `INSUFFICIENT`，不会把
未验证的候选片段伪装成前端引用。
