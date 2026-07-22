# 步骤 5 详细设计：Discovery 与 Evidence 检索 API

## 调用边界

V2 问文 Skill 通过受认证的内部 API 调用 KC；浏览/管理页面可使用相同语义的外部只读 API，但不能绕过 Domain 与 Collection Binding 校验。所有请求以 Domain 为路径 Scope，不接受跨 Domain 的 Collection、Bundle 或 Document ID。

```text
POST /internal/v2/knowledge/domains/{domain_id}/retrieval/discovery
POST /internal/v2/knowledge/domains/{domain_id}/retrieval/evidence
```

请求中的 `agent_id` 用于 KC 读取 ACTIVE `AGENT` Binding。调用方未显式选 Collection 时，KC 使用该 Agent 的全部 ACTIVE、且 Collection 未 DISABLED 的 Binding；显式 `collection_keys` 只能收窄这一范围，不能扩权。页面调用不带 Agent 时必须由用户权限服务传入等价的已校验 Collection Scope。

## Discovery 请求与响应

```json
{
  "agent_id": "42",
  "query": "有哪些零售行业的 AIOps 方案？",
  "collection_keys": ["assets"],
  "facets": {"industry": ["Retail"]},
  "top_k_per_collection": 20,
  "max_results": 30
}
```

KC 对每个 Collection 独立执行 Revision 当前值过滤、权限/安全等级过滤、Facet 过滤、Oracle Text 与向量召回并做 Collection 内融合；跨 Collection 使用校准分数或 RRF 融合和多样性约束，不预设业务权重。

每个 `DiscoveryHit` 返回：

```text
collection_id/key, bundle_id, bundle_revision_id,
object_type(BUNDLE/DOCUMENT), bundle_revision_document_id?,
document_id?/document_version_id?, display_title, matched_facets,
match_reasons, coverage_summary, rank
```

`match_reasons` 只能是高亮词、Facet 命中或受控画像摘要片段；不能作为最终回答证据。`coverage_summary` 显示附件数量、可用/失败数量和部分可用说明，不泄漏失败堆栈、对象 URI 或无权限文件名称。

## Evidence 请求与响应

```json
{
  "agent_id": "42",
  "query": "该方案如何处理告警降噪？",
  "candidates": [
    {"bundle_id": 101, "bundle_revision_id": 301},
    {"bundle_revision_document_id": 401}
  ],
  "max_evidence": 12,
  "context_token_budget": 6000,
  "include_neighbors": true,
  "include_relations": false
}
```

KC 重新验证 candidate 属于请求 Domain、允许 Collection、Bundle 当前 Revision 和 READY Member；不能仅因为调用方曾得到一个 ID 就跳过安全校验。随后只检索这些 Member 对应的 `ACTIVE Parse View + ACTIVE Evidence`，执行关键词/向量混合召回、视图去重、章节/页/单元格邻接扩展和可选的一跳 ACTIVE Relation 扩展。

每个 `EvidenceHit` 返回：

```text
collection_id/key, bundle_id, bundle_revision_id,
bundle_revision_document_id, document_id, document_version_id, parse_view_id,
evidence_id, evidence_type, content, locator_json,
heading_path, payload_reference?, rank, score_components?,
neighbor_of_evidence_id?, relation_support?
```

`payload_reference` 是经 KC 授权的短期预览引用，不是持久存储 URI。`score_components` 仅在内部调试/评测开关下返回；正常 Agent 调用只需要排序和命中理由。任何扩展结果都要标记其邻接/关系来源，Citation Pack 只能把真正命中或经上下文策略采纳的 Evidence 作为引用。

## 检索可见性与错误

- 只查询 Bundle `current_revision_id` 指向的 `READY/PARTIAL` Revision；旧 Revision、`STAGED/DELETING` Evidence 和 DISABLED Collection 永不返回。
- `PARTIAL` Revision 的 Manifest 和 READY 附件可正常检索；失败 Member 仅进入覆盖摘要。
- `403 COLLECTION_SCOPE_DENIED`：Agent/用户不拥有请求 Collection；`404` 不暴露其他 Domain 对象是否存在。
- `409 CANDIDATE_STALE`：candidate 已非当前 Revision 或在请求期间被切换；Skill 可重新 Discovery 一次，不能回退 V1。
- `422 RETRIEVAL_REQUEST_INVALID`：预算、Facet、候选组合或显式 Collection 参数非法。

## Citation Pack 与 Skill 接缝

Skill 将 Evidence API 的结果转换为 Citation Pack，按 `document_version_id + parse_view_id + evidence_id` 去重并控制上下文预算。每个最终引用至少包含 Collection、Bundle/Revision、Document/Version、Parse View、Evidence 和 `locator_json`。若 Discovery 有候选但 Evidence 没有足够可引用内容，Skill 必须明确报告证据不足；禁止使用 Discovery profile 或模型常识补成事实回答。

V2 API 不返回 `TxtBaseSearchResult`，也不读取 V1 File/Chunk 表。离线评测记录 Discovery Recall@K、Evidence Recall@K、当前 Revision 正确率、引用定位准确率、跨附件覆盖率和“Discovery 命中但 Evidence 缺失”比例。
