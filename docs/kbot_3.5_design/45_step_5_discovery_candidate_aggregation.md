# 步骤 5 详细设计：Discovery 候选聚合与 Collection 平权

## 候选单位

KC 内部同时检索 Bundle Profile 与 Document Member Profile，但对外统一输出 `BundleCandidate`。Document Profile 命中后上卷所属 Bundle，并作为 `matched_members` 保留文件焦点；同一 Bundle 不以 Bundle、附件一、附件二等多条顶层候选重复占位。

```text
BundleCandidate {
  collection_id, bundle_id, bundle_revision_id,
  member_count, matched_members[], candidate_scope,
  match_signals[], local_rank, selection_grade?
}
```

Document Profile 仍用于发现附件内容，Bundle Profile 负责来源标题、Facet、Manifest 和附件目录等业务画像。两者不可合表：Bundle/Revision 决定来源身份与当前快照，Document/Version/View/Evidence 决定不可变内容和引用定位。

## 常见单文档快速路径

这里的 Document 是 KC 逻辑文档，不只表示上传附件。来源主信息如果需要参与事实回答和最终引用，Ingestion Adapter 必须把规范化主信息生成一个 `PRIMARY` Document/Version/Member；Bundle Profile 只负责发现，不能代替 Evidence。因而“没有附件”的来源对象通常仍有一个 PRIMARY Document，附件作为额外 Member。

真实数据以每个 Bundle 一个逻辑 Document 为主，少量 Bundle 才有多个 Document。运行时按 READY Member 数量选择路径：

- `member_count=1`：自动将唯一 Member 作为 Evidence Scope；不调用 LLM 做附件选择，也不执行多附件预算分配。Candidate Card 合并 Bundle 画像与该 Member 的命中摘要。
- `member_count>1`：根据 QueryPlan 和命中画像选择 `MATCHED_MEMBERS/EXPLICIT_MEMBERS/BUNDLE_ALL`，由 `LLMDiscoverySelector` 只能从预分配 Member label 中选择焦点。
- `member_count=0`：只允许作为接收失败或尚未就绪状态存在，不能将 Revision 标记为可完整检索；可以在内部覆盖状态中呈现，但不得进入最终答案。

单文档快速路径是执行优化，不改变 API identity 和数据模型。不能因为当前大多数为一对一，就把 Bundle 与 Document 合并，否则来源修订、无附件对象、多附件对象和后续新增 Member 都会重新引入特殊分支。

## Collection 内聚合

每条召回通道先按 Bundle 折叠，再执行 Bundle 级 RRF：

```text
Text profile hits   → first/best hit per Bundle
Vector profile hits → first/best hit per Bundle
Exact/Facet hits    → first/best hit per Bundle
                    → Bundle-level RRF
```

同一 Bundle 的其他 Profile 命中只写入 `matched_members/match_signals`，不重复贡献完整 rank 分。可以把“多个不同 Member 均有信号”作为 Candidate Card 的解释字段，但不得按附件数线性加分，避免大 Bundle 天然占优。

候选只有在精确 ID/标题、明确 Facet、达到版本化 Text/Vector 门槛或多路共同命中之一成立时才进入局部池。Collection 平权不代表无关 Collection 必须返回占位结果。

## 跨 Collection 候选池

每个允许 Collection 使用相同局部预算和质量门槛。合格结果按稳定 Collection key 与局部 rank 公平交错进入全局池；每个 Collection 有相同初始上限，空余名额可重新分配。绑定查询顺序、Collection 大小和文档数量不能改变候选机会，不设置 Primary、业务权重或最低强制占位。

`LLMDiscoverySelector` 对全局 Candidate Card 做 Setwise/Listwise 选择。每张卡使用相同 token 上限，默认最多展示三个 matched Member；Collection 名称不作为业务优先级信号。Selector 输出 `DIRECT/STRONG/POSSIBLE/SEMANTIC_ONLY/IRRELEVANT` 及预分配 Bundle/Member label，只有前三类可进入 CandidatePlan。

## Candidate Scope

```text
SINGLE_MEMBER    唯一 READY Member，自动聚焦
MATCHED_MEMBERS  普通问答，优先画像命中的 Member
EXPLICIT_MEMBERS 用户明确指定文件或历史有效 Citation
BUNDLE_ALL       整体总结/比较，需要覆盖全部合格 Member
```

列表型问题默认使用主信息与 matched Member 做低深度证据验证；事实问答先深入 matched Member，证据不足时只能按 Policy 扩展一次；结构总结按章节/Member 覆盖，不使用相关度 Top-K 代替完整性。

## 验收

- 单 Document Bundle 只产生一个 BundleCandidate，且 Evidence Scope 自动落到唯一 Member。
- 多 Document Bundle 不因附件数量增加 RRF 贡献；Document 命中可追溯到对应 matched Member。
- 交换 Binding/Collection 输入顺序不改变局部候选和公平池；无合格信号的 Collection 不占位。
- 分别报告单/多 Document Bundle Recall、Bundle size bias、每 Collection Candidate Recall、Selector Precision 和最终 Evidence 支持率。
