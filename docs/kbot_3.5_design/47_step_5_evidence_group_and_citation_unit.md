# 步骤 5 详细设计：Evidence Group 与引用单位

## 定位

`KBOT_KC_EVIDENCE` 保存 Parser 产生的最小、可定位事实单元；Evidence Group 是 KC 在一次检索请求中按问题和 Retrieval Policy 组装的运行时 DTO，不新增事实表。回答模型引用 Group 级 `citation_label`，KC/Skill 在内部保留组成它的全部 Evidence identity 和定位。

```text
Evidence rows → 去重 → 结构上下文组装 → Evidence Group
  → LLMEvidenceSupportJudge → Verified Citation Group
```

Group 不跨 Document Version/Parse View。多个文档共同支持一个结论时，Citation Pack 保存多个 Citation Group，由回答 Claim 同时引用；不能把不同来源拼成一个不可定位引用。

## 组装前后角色

召回后、判断前的输入角色为：

```text
ANCHOR              Text/Vector/RRF 直接召回
STRUCTURAL_CONTEXT  父标题、表头、Sheet、单位、Caption
NEIGHBOR_CONTEXT    同章节或同来源 Atom 的必要邻接内容
```

`LLMEvidenceSupportJudge` 输出后才形成最终角色：

```text
PRIMARY              直接支持问题或 Claim
STRUCTURAL_CONTEXT   理解 PRIMARY 必需但不能独立支撑
NEIGHBOR              仅补全语义
```

Judge 可以把上下文项提升为 PRIMARY，但必须记录 `promoted_from_context`；也可以把 ANCHOR 降为 context/no-support。最终 Group 至少有一个 PRIMARY，否则不能产生 citation label。

## Group DTO

```text
group_label, bundle/revision/member/document/version/view identity,
items[] { item_label, evidence_id, input_role, final_role?,
          evidence_type, content, heading_path, locator_json,
          source_spans, provenance, quality_score },
anchor_evidence_ids[], token_count, assembly_trace,
support_grade?, answerable_aspects[], unsupported_aspects[]
```

Verified Group 获得请求内短标签 `C1/C2/...`：

```text
CitationGroup {
  citation_label,
  primary_evidence_ids[],
  structural_context_ids[],
  neighbor_evidence_ids[],
  citation identity + locators
}
```

模型只能返回预分配 `citation_label`，不能引用 item label、数据库 ID 或 context-only Group。

## 结构规则

- 段落：ANCHOR 加父标题路径；只在同 `section_key`、来源 span 或 fragment 链表明语义未完整时附加邻接，不使用机械 `ordinal ± 1`。
- 长段落 Fragment：同 `source_item_ref + fragment_index` 可重组为一组，仍保留每个定位。
- 表格：TABLE_ROW/CELL_RANGE 加表格标题、多级表头、行标签、单位和必要脚注；不默认加入无关前后行。
- Spreadsheet：保留 Sheet、子表、表头、合并单元格上下文和精确 cell range；VLM 描述不能替代可定位单元格。
- 图片：IMAGE 加 Caption、图号、所属章节和必要解释段；OCR/VLM 内容必须保留 provenance，纯生成描述不能单独成为高置信 PRIMARY。

DOCUMENT/SECTION 级生成摘要、Bundle Discovery Profile 和其他无原始 source span 的生成文本只能用于召回或上下文，不得成为 PRIMARY。

## 去重与预算

依次按 Evidence ID、内容 hash、`source_item_ref/source_spans`、多视图同区域和结构重叠去重。表格不同数据行不能仅因文本相似而合并。多视图重复优先 ACTIVE、质量更高、locator 更精确、生成比例更低的表示。

预算在 Group 层执行：BREADTH 每 Bundle 保留少量直接支持 Group；DEPTH 允许同 Document 多组但受上限；COMPARE 先保证各对象覆盖；STRUCTURAL 按章节/Sheet 覆盖。初始 Policy 可设置每 Document、Bundle、section 的 Group 上限及 context token 比例，具体数值通过评测版本化，不写入 Agent Binding。

## 验收

- 每个 citation label 至少有一个当前、授权、可定位 PRIMARY Evidence，且全部成员属于同一 Document Version/View。
- 删除或失效任一 PRIMARY 后重新验证；Group 无剩余 PRIMARY 时 citation 必须消失。
- 标题、表头和邻接内容不会以 context-only 身份单独进入回答引用或 `doc_results_v2`。
- 分别报告 Group 支持准确率、上下文冗余率、PRIMARY 提升/降级率、定位准确率和每 token 有效支撑率。
