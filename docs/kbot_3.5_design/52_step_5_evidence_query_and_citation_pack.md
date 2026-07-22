# 步骤 5 详细设计：Evidence 查询与 Citation Pack

## 查询边界

`POST /api/v2/knowledge/retrieval/evidence` 只接受 Discovery 阶段产生的 Bundle/Revision/Document Version Scope。KC 重新按当前 Revision、Active Evidence、Embedding 就绪和安全等级校验 Scope；调用方不能用历史或跨 Domain ID 扩大查询范围。

Evidence 查询同时支持 Oracle Text 与按 Collection 分组的 Vector 查询。Text/Vector 命中先去重，再按同一 Document Version/Parse View 组装 Evidence Group；Group 不跨版本、视图或文档。

## Group 与引用

直接召回项标记为 `ANCHOR/PRIMARY`，同章节标题、父级或必要邻接项标记为 `STRUCTURAL_CONTEXT`。当前实现不自动把上下文当事实来源。每个至少含 PRIMARY 的 Group 获得请求级 `C1/C2/...` Citation Label；Citation Pack 保存完整 Evidence identity、locator、source span、provenance 和角色。

Discovery Profile、Bundle 摘要和 context-only Evidence 不会生成 Citation Label。后续 Support Judge 可以在预分配 Group 内提升/降级角色，但不能跨 Scope 拼接 Group。

## 已落地内容

- Evidence Text/Vector 查询及安全/当前 Revision 过滤。
- Evidence Group 组装、同版本边界、上下文预算和去重。
- Citation Pack DTO 与 `/api/v2/knowledge/retrieval/evidence` 接口。

回答生成后的 `used_citation_labels` 校验和 `doc_results_v2` 投影属于下一步 Skill/Answer Grounding 改造；当前 API 不直接把原子 Evidence Top-K 当作前端引用列表。
