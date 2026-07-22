# 步骤 6 详细设计：DocumentAgentV2 与 KnowledgeRetrievalSkillV2 接入

## 调用边界

```text
Root Agent / Agent Router
  → DocumentAgentV2
      → KnowledgeRetrievalSkillV2
          → KnowledgeCoreClient
              → Discovery → Evidence → Citation Pack
```

`DocumentAgentV2` 只负责版本化 `KnowledgeTask` 的校验、有限重试和结果封装；Skill 是无状态能力适配器，不创建或持有任何 Agent，也不访问 KC 数据库。V1 `AskDocSkill`、`DocAgent`、`DocService` 和 File/Chunk 链路保持不变。

## DTO 与结果

`KnowledgeTask` 携带 `task_id/parent_run_id/domain_id/agent_id`、原始/消歧后的问题、已授权 Collection、权限等级和历史有效引用。Skill 返回 `KnowledgeTaskResult`，其中 `citation_pack` 只包含 KC Verified Citation，证据不足时返回明确 `INSUFFICIENT_EVIDENCE` 和安全缺口说明。

Skill 通过 `KnowledgeCoreClient` 调用 Discovery 和 Evidence API；它不本地根据 rank 选择 Bundle，不扩大 Collection，也不回退 V1。多 Agent 场景可复用同一 Skill，但必须传入各自的授权 Scope 和 task identity。

## 已落地内容

- `knowledge_core/client.py`：带内部认证的 KC V2 HTTP Client。
- `knowledge_core/application/task_dto.py`：版本化 Agent↔Skill DTO。
- `skills/knowledge_retrieval_v2.py`：无状态 V2 Skill。
- `agent/agent/document_agent_v2.py`：只依赖 Skill 的 V2 Document Agent。
- `skills/skill_libs/knowledge_retrieval_v2/`：纳入现有 Skill Registry 的独立包。

Skill 输出 `knowledge_task_result` 和 Citation Pack 数据；`knowledge_core/application/sse_v2.py`
定义 GroundingResult 到 `citations_v2/doc_results_v2/grounding_status` 的独立序列化边界，
由 RootAgentV2 在回答模型完成后统一执行。该链路不改变 V1 Skill、DocAgent 或 SSE。
