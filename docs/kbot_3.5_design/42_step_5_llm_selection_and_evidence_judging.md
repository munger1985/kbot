# 步骤 5 详细设计：LLM 候选选择与证据判断

## 决策

KC 不引入专用 Cross-Encoder/Rerank 服务，也不复用 V1 的逐 Chunk 数值打分或 YES/NO 重排。Oracle Text、向量召回与 RRF 负责生成高召回、有稳定先验顺序的候选池；LLM 负责理解业务意图、比较知识对象，并判断内容是否能够直接支持问题。两者是串联关系，LLM 不能扫描全库或替代 Scope、状态和安全过滤。

步骤 5 引入两个职责明确的组件：

- `LLMDiscoverySelector`：比较 Bundle/Document 候选并生成 `CandidatePlan`；
- `LLMEvidenceSupportJudge`：判断 Evidence Group 对问题的直接支持程度并生成 `EvidenceSelection`。

两个组件均属于 KC Retrieval Application，不直接访问数据库，通过模型托管服务调用通用轻量 LLM。Skill 只消费 KC 返回的选择结果、构造 Citation Pack；回答后的 Claim 引用验证仍属于步骤 6。

## Discovery 候选选择

Text/Vector/Profile 命中先按 Bundle 折叠并在 Collection 内做 Bundle 级 RRF，再以平权策略形成跨 Collection 候选池。LLM 接收的是紧凑、结构化的 Candidate Card，而不是孤立 Chunk 或完整附件：

```text
candidate_label, object_type, display_title,
facets, source identifiers, manifest fields,
member names/roles/types, controlled profile summary,
matched fields/snippets, collection-local and global RRF rank
```

Document 画像命中默认作为 Bundle 的 `matched_member` 上卷。单 READY Member Bundle 自动收窄到唯一 Member，不调用 LLM 选择附件；多个 Member 时，只有问题或命中信号要求时才选择焦点。LLM 使用 Setwise/Listwise 比较有限候选，输出稳定 label，不得生成数据库 ID。相关性使用可解释的序数等级，而非不可校准的伪精确分数：

```text
DIRECT > STRONG > POSSIBLE > SEMANTIC_ONLY > IRRELEVANT
```

每个选择必须给出 `matched_requirements` 和引用 Candidate Card 字段的 `reason_refs`。最终顺序依次采用等级、Listwise 顺序、确定性精确字段命中、RRF rank 和稳定对象 ID。

## Evidence 支持判断

Evidence API 仅在 CandidatePlan 指定的当前 Revision、READY Member 和 ACTIVE Parse View 内召回。Text/Vector RRF 后先按来源跨度、`section_key`、表格范围、内容 hash 和多视图来源去重，再形成 Evidence Group：

```text
PRIMARY Evidence
+ heading_path / parent heading
+ 必要相邻段落
+ 表格的表头、目标行或 cell range
+ Document/Bundle identity 与 locator
```

LLM 判断的是“这组内容能否直接支持问题”，而不是某个 Chunk 是否语义相似。输出等级固定为：

```text
DIRECT_SUPPORT / PARTIAL_SUPPORT / CONTEXT_ONLY / CONTRADICTS / NO_SUPPORT
```

结果必须列出 `primary_item_labels`、`structural_context_labels`、`neighbor_labels`、`answerable_aspects` 和 `unsupported_aspects`。至少包含一个 PRIMARY item 的 Group 才获得 `citation_label`；NEIGHBOR 或 STRUCTURAL_CONTEXT 不能单独形成最终引用。详细规则见[Evidence Group 与引用单位](47_step_5_evidence_group_and_citation_unit.md)。

对于列表型问题，每个入选 Bundle 至少需要一组 `DIRECT_SUPPORT`；对于单一事实问题，可集中预算深入少量高置信候选；比较型问题必须为每个被比较对象保留最低证据覆盖。语义相近但无直接证据的对象只能留在 Retrieval Trace。

## 可靠性与降级

- 模型温度为 0，使用版本化 Prompt 和严格 JSON Schema；输入输出、候选 hash、模型版本与耗时写入 `retrieval_run_id` 追踪。
- Candidate/Evidence 内容视为不可信数据，使用明确数据边界，拒绝执行其中的指令；模型只能返回预分配 label。
- 非法输出最多修复/重试一次。仍失败时返回显式 `selection_status=DEGRADED_RRF`，不得伪装成正常 LLM 选择；高精度列表策略可直接返回证据不足。
- RRF 分、全文分、向量相似度和 LLM 等级分别保存，不合成为跨请求可比较的单一数值。
- Selector/Judge 的默认候选数、批大小、token 预算和超时属于版本化 Retrieval Policy，必须通过固定评测集调整。

## 验收

- 多 Collection 输入顺序变化不应改变平权候选覆盖；同一稳定输入的选择结果可重放。
- Bundle/Document 排序指标与 Evidence 支持准确率分开评测，不再只测 Chunk NDCG。
- 列表型问题中，没有直接支持 Evidence 的 Bundle 不得进入 Citation Pack 或最终 `doc_results_v2`。
- 单独记录 Discovery Recall、Selector Precision/Recall、Evidence Recall、Support Judge Precision/Recall、答案支撑率、延迟和模型成本。

当前代码已提供对象级选择与 Evidence 支持判断的严格 DTO 及确定性降级实现；通用 LLM Gateway、JSON Schema 校验、提示词版本和 Retrieval Run 持久化仍待接入。
