# 步骤 5 详细设计：Discovery 查询与候选聚合

## 查询入口

`POST /api/v2/knowledge/discovery/search` 接受 `query`、已通过 Agent Binding/Domain 校验的 `collection_ids`，以及按 Collection 分组的可选 query vector。KC 不接受 `primary_collection` 或业务权重；输入 Collection 只做去重和稳定排序。

每个 Collection 独立执行 Oracle Text 和可选 Oracle Vector 初筛。Vector 查询必须由调用方按 Collection 的唯一模型生成，KC 不允许跨模型复用 query vector。结果只包含 `DISCOVERY_STATUS=ACTIVE`、当前 Bundle Revision、`READY/PARTIAL` Revision 和安全范围内的画像。

## Bundle 聚合

Bundle Profile 和多个 Document Profile 命中先在 Collection 内折叠为一个 `BundleCandidate`。Document 命中保留 `matched_members`，但不会按附件数量线性增加排名贡献。Text/Vector 各取同一 Bundle 的最佳 rank 后执行 Bundle 级 RRF，输出 `SINGLE_MEMBER`、`MATCHED_MEMBERS` 或 `BUNDLE_ALL` scope。

跨 Collection 时使用相同局部预算按 Collection key 稳定交错，保持平权；不设置 Primary，也不为无命中 Collection 生成占位结果。LLM Candidate Selector 尚未接入，当前响应只提供确定性候选和信号，后续 Selector 只能在该候选池内选择，不能自行扩展 Scope。

## 已落地内容

- `DiscoveryRepository.search_text/search_vector`：当前版本和 Active Profile 过滤。
- `KnowledgeCoreDiscoveryService`：Text/Vector 通道合并及 Bundle 级候选聚合。
- V2 Discovery Search API 与候选 DTO。
- 单文档快速路径和多文档 matched member 信息均保留在候选契约中。

下一步实现候选范围内的 Evidence 查询、证据组装和引用单位；该阶段不把 Discovery `profile_text` 当作最终回答来源。
