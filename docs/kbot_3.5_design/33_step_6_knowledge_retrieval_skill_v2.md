# 步骤 6 详细设计：KnowledgeRetrievalSkillV2

## 定位与隔离

`KnowledgeRetrievalSkillV2` 是 V2 问文的检索编排器，不是 KC 的数据库访问层，也不是最终回答模型。它只调用 KC 的 Discovery/Evidence API，构造 Citation Pack 并执行回答前证据覆盖校验。

V1 `TxtBaseSearch`、`DocService`、旧 ask-doc Skill 和 `TxtBaseSearchResult` 保持独立。路由在 Agent/App 配置中显式选择 `knowledge_retrieval_v1` 或 `knowledge_retrieval_v2`；V2 发生空结果、超时或服务错误时绝不读取 V1 File/Chunk 表或自动降级。

## 输入与输出

```text
输入：question, agent_id, authenticated domain_id,
      explicit collection_keys?, conversation retrieval constraints?

输出：CitationPack | RetrievalInsufficientEvidence
```

Skill 不接受 `app_id`、`kb_id`、任意 Bundle/Document ID 作为未验证权限参数。KC 根据 Agent Binding 解析允许 Collection；用户选择 Collection 时，Skill 仅把选择作为收窄条件传递。

`CitationPack` 至少包括：

```text
question, retrieval_run_id, policy_key,
selected_collections, discovery_summary,
evidence[] { content, citation, rank, match_reason, context_role },
coverage { candidate_count, document_count, evidence_count, gaps[] },
warnings[]
```

其中 `citation` 固定包含 Collection、Bundle、Bundle Revision、Revision Member、Document、Document Version、Parse View、Evidence 和 `locator_json`。`context_role` 标记 `PRIMARY/NEIGHBOR/RELATION_CONTEXT`，防止回答模型把邻接内容误当直接命中。

## 编排流程

```text
1. 校验 Domain/Agent/显式 Collection 输入
2. 调用 Discovery
3. 依据问题、预算和多样性选择 candidate
4. 调用 Evidence（受 candidate 范围限制）
5. 组装、去重、预算截断和覆盖校验
6. 返回 Citation Pack 或明确的证据不足结果
```

### Candidate 选择

Discovery 返回 Bundle 或 Document Member。Skill 首期不使用 LLM 在候选间自由猜测：按 KC rank、明确 Facet/标题命中、Bundle 多样性和最大附件数形成确定性 candidate plan。问题出现“该文件/这个资产”等会话指代时，只能使用会话内已有且仍通过 KC 当前 Revision 校验的引用；否则重新 Discovery。

### Evidence 与上下文

Evidence 预算按 token、最大 Evidence 数、每 Document 最大条数和最小跨附件覆盖控制。Skill 保留每个直接命中及其必要标题/页/表格邻接内容；不因分数高而无限选择同一长文档。表格问题优先保留表头、相关行/单元格范围和其定位，图片问题保留带 provenance 的图像 Evidence 与预览引用。

### 覆盖校验

在把 Citation Pack 交给回答模型前，Skill 检查：

- 至少一个 PRIMARY Evidence，且所有 Evidence 都属于当前 Revision；
- 问题中明确的对象/时间/附件约束是否有命中或被明确标为缺失；
- 多部分问题是否每个子问题至少有证据、或在 `gaps` 中说明无法支持；
- 引用定位是否完整（文档页/坐标或 Spreadsheet Sheet/cell range）。

失败时返回 `RetrievalInsufficientEvidence`，带可安全显示的原因，如“找到 Asset，但附件不可用”或“没有可引用的相关内容”。上层 Agent 应据此澄清、缩小问题或诚实说明不足；不得让模型用 Discovery 摘要、对话记忆或常识伪造答案。

## 与 Agent、SSE 的接缝

V2 Root Agent/Doc Orchestrator 将 Citation Pack 作为受控工具结果传给回答模型，并在 SSE 末尾输出独立的 `citations_v2` 结构。旧 `references`/Chunk 格式不能复用同一字段静默混装；客户端通过路由版本识别并渲染 V1 或 V2。

Skill 返回的 Retrieval 诊断仅面向 Agent/审计：Collection Scope、候选数、耗时、错误码和覆盖缺口。SSE 对终端用户只显示经授权的标题、引用和安全摘要，不显示 storage URI、Job、内部 score 或其他 Collection 的存在性。前端 `doc_results_v2` 必须由回答后有效使用的 Evidence 上卷生成，不能直接使用 Discovery/Evidence Top-K；完整规则见[回答溯源与 doc_results](34_step_6_answer_grounding_and_doc_results.md)。

## 失败与测试

- KC `403/404`：按无权限/不可见处理，不尝试其他 Collection 或 V1。
- `409 CANDIDATE_STALE`：最多重新 Discovery 一次；仍冲突则返回可重试的证据不足。
- KC 短暂错误：返回受控暂时不可用，不自动改走旧检索。
- 空 Discovery、空 Evidence、PARTIAL Revision、表格/图片、多 Collection 平权和会话指代均须有 mock/integration 测试。

验收时，对已路由到 V2 的请求检查调用链只包含 KC Client，运行 SQL/日志中不存在 V1 File/Chunk Repository；每个最终回答句的引用来自 Citation Pack，并可回到稳定定位。
