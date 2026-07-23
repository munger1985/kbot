# 4.0 Document Agent 边界与检索契约

## 定位

Document Agent 是 Knowledge 领域的查询 Agent，当前先作为 Agent Runtime 内的独立模块运行，未来可按同一契约提取为 `apps/document_agent` 服务。它不拥有知识库表，不负责文件生命周期，也不替代 Knowledge Core。

```text
Root Agent / Supervisor
        ↓ DocumentQueryTask
Document Agent
        ↓ KnowledgeCoreClient
Knowledge Core
        ↓
CitationPack / RetrievalReport
        ↓
Root Grounding / Response Composer
```

职责边界：

- 解析用户意图和检索目标；
- 生成查询计划和候选范围；
- 调用 KC Discovery/Evidence API；
- 在 Bundle、Document 和 Evidence 层级做聚合、去重和选择；
- 生成可引用的 `CITATION_PACK` 和覆盖度报告；
- 识别证据不足、范围不匹配和需要澄清的情况。

Document Agent 不负责：

- 创建或解析 Document Version、Parse View 和 Evidence；
- 直接访问 KC Repository 或数据库；
- 直接调用 Embedding 表或模型 Entity；
- 把候选 Chunk 直接作为最终引用返回；
- 生成未经 Grounding 验证的最终回答；
- 访问 Data Query、Ops 表或跨 domain 数据。

## 输入 DTO

```text
DocumentQueryTask {
  task_id
  run_id
  original_query
  standalone_query
  domain_id
  agent_id
  authorized_collection_ids
  max_security_level
  intent: LIST | ANSWER | SUMMARIZE | COMPARE | LOCATE
  scope_hint: BUNDLE | DOCUMENT | EVIDENCE | AUTO
  answer_language
  max_bundles
  max_documents
  max_citations
  citation_required
  deadline_at
}
```

`domain_id` 和授权 Collection 范围由 AuthContext 派生；请求中的 Collection ID 只能缩小范围。`intent` 和 `scope_hint` 是检索目标，不是数据库过滤条件。用户说“列出关于某案例的资产”时使用 `LIST + BUNDLE`；询问附件细节时使用 `ANSWER/SUMMARIZE + DOCUMENT/EVIDENCE`。

## 输出 Artifact

```text
DocumentRetrievalResult {
  status: READY | INSUFFICIENT_EVIDENCE | NEEDS_CLARIFICATION | FAILED
  citation_pack: CitationPack.v2
  retrieval_report: RetrievalReport.v1
  coverage_gaps
  warnings
  kc_request_ids
}
```

`CITATION_PACK` 至少包含：

```text
question
query_plan
bundle_candidates
document_candidates
citations[] {
  citation_label
  collection_id
  bundle_id
  document_id
  document_version_id
  evidence_id
  title
  excerpt
  locator        # page/section/sheet/cell/coordinate
  relevance_reason
  source_hash
}
coverage {
  candidate_bundle_count
  selected_document_count
  selected_evidence_count
  uncovered_aspects
}
```

Citation 的最小稳定单位是 Document/Evidence 组合，而不是裸 Chunk。Bundle 列表类问题必须返回 Bundle/Document 级引用；只有需要解释具体内容时才增加 Evidence 片段。最终 `DOCUMENT` Reference Card 只能由 Grounding 阶段根据回答实际使用的 `citation_label` 生成；4.0 不再使用 `doc_results_v2` 作为跨领域引用契约。

其中 `agent_id`、`collection_id`、`bundle_id`、`document_id`、`document_version_id` 和 `evidence_id` 都是 UUIDv7 领域主键；Oracle 以 `RAW(16)` 保存，契约序列化为规范 UUID 字符串，不存在另一套内部数字 PK。完整标识策略见 [31_aiops_step2_persistence_and_identity.md](31_aiops_step2_persistence_and_identity.md)。

## 检索编排

```text
1. Query Normalization
2. Intent + Scope Classification
3. RetrievalPlan
4. KC Discovery（Bundle/Document 候选）
5. Candidate Aggregation + Deduplication
6. KC Evidence Retrieval（限定候选范围）
7. Document/Evidence Group Selection
8. Coverage and Citation Pack
9. Grounding Agent / Response Composer
```

### 查询理解

查询改写只补充必要的上下文，不改变 domain、Collection 和安全边界。计划至少声明 `target_level`、`answer_mode`、`breadth/depth`、过滤 Facet、候选数量和证据要求。

### 两阶段检索

- Discovery 负责识别相关 Bundle/Document 和 Facet，不直接作为事实引用；
- Evidence 只在候选对象和授权范围内检索；
- 一个 Bundle 只有一个 Document 时可走单成员快速路径，但仍保留 Bundle 与 Document 层级；
- 多文档 Bundle 先通过 Bundle Manifest/Discovery Profile 建立整体语义，再按问题选择具体 Document；
- 不使用纯数值 reranker 对裸 Chunk 做最终排序。

### LLM 选择器

当候选 Bundle/Document 较多时，可以用 LLM 对候选对象或 Evidence Group 做结构化判断：

```text
CandidateJudge {
  candidate_id
  supports_question: true | false
  support_aspects[]
  contradiction: true | false
  evidence_labels[]
  confidence
  reason
}
```

LLM 只能在已授权、已召回的候选集合内选择，不能扩展检索范围；判断结果必须通过 schema 校验，并保留原始 Evidence 引用。

## 与 Root Agent 的协作

Root Agent 负责会话目标、跨领域路由和最终回答；Document Agent 只返回检索 Artifact。典型流程：

```text
Root Run
  └─ Document Task
       ├─ CitationPack
       └─ RetrievalReport
            ↓
       Answer Generation
            ↓
       Grounding Verification
            ↓
       GROUNDED_ANSWER + final citations
```

如果 Root Agent 同时委派 MCP 问数 Adapter 或 AIOps Agent，各领域结果保持独立 Artifact。当前不存在独立 Data Agent；Document Agent 不读取 MCP QueryResult，不把运维诊断当成文档证据。

## 失败、幂等和可观测性

- 同一 `run_id + task_id + query_plan_hash` 的重试不得生成重复 CitationPack；
- KC 暂时不可用时返回可解释的 `FAILED` 或 `INSUFFICIENT_EVIDENCE`，不能回退到旧 KB/TxtChunk 表；
- 候选为空、Evidence 为空、权限范围为空和引用定位缺失使用不同错误/覆盖度状态；
- 每次 KC 调用记录 query hash、collection scope、request ID、延迟和候选数量，不记录未脱敏正文；
- `RetrievalReport` 保存计划版本、KC API 版本、Embedding 模型版本和选择器版本，便于评测和复现。

## 服务化演进

第一阶段，`DocumentAgentV2` 演进为实现 `DocumentQueryTask → DocumentRetrievalResult` 的 Runtime 模块；Root Agent 通过接口调用，不依赖具体类。第二阶段，当需要独立扩缩、Portal/MCP 直接调用或检索任务与 Root Runtime 解耦时，复制同一 Application/Client 契约为 `document_agent` 服务。服务化只改变传输和部署，不改变 KC 表所有权、CitationPack schema 或 Grounding 规则。
