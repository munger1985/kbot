# 步骤 5 详细设计：检索基线与投影生成

## 原则

本设计交付一条可复现的基线，而非宣称它是最终最优检索器。所有生成规则和检索参数都带版本；后续 Docling、Query 改写、重排或表格专用策略只替换对应阶段，并与基线在同一评测集对比。

## 两类投影的生成

### Evidence 投影

KC 在 Parser DTO 写入后确定性构造：

```text
retrieval_text =
  MIME/type label
  + heading_path
  + stable structure label (table columns / sheet / cell range when available)
  + evidence content
```

它不包含 Bundle 标题、Revision Facet、Member 声明名称、角色或 LLM 文档摘要，因为这些字段可随 Revision 改变。`INDEX` Job 使用所属 Collection 绑定模型对该文本生成 embedding，记录 `embedding_model_id/key`、配置 fingerprint、`retrieval_profile_version` 和输入 hash。重跑相同 Version/View/规则/hash 幂等；Collection 更换模型时只重建本 Collection，且切换期间不能让新旧模型向量共同参与查询。

### Discovery 投影

`PROFILE` Job 为当前候选 Bundle Revision 建立 Bundle 与 Document Member 画像。基线 `profile_text` 确定性拼接：Revision 标题、Facet、Manifest 中的 Asset/来源字段、Member 名称/角色/MIME、Document 的受控摘要/关键词、标题/表格概要以及覆盖摘要。它可以使用质量受控的摘要作为召回信号，但不作为最终引用。

画像输入和其依赖的 Member/Version/View 列表写入 `coverage_json`；`profile_hash` 覆盖所有规范化输入。Evidence/Manifest 更新、Member 补传或新 Revision 都使对应 Profile 失效并重新构建。旧 Profile 仅在旧 Revision 仍为 Bundle current 时保持可见。

## 基线查询流程

```text
Query
  → 规范化（空白、语言、受控 Facet）
  → Discovery：每 Collection 的 Text + Vector → RRF → candidate objects
  → Candidate Card → LLMDiscoverySelector → CandidatePlan
  → Evidence：受限范围内的 Text + Vector → RRF → 去重/邻接
  → Evidence Group → LLMEvidenceSupportJudge
  → Citation Pack
```

### Discovery

1. 先以 Domain、允许 Collection、Collection 状态、安全等级、Bundle current Revision 和 `discovery_status=ACTIVE` 过滤。
2. 在每个 Collection 内分别取得 `profile_text` 全文候选和 profile embedding 向量候选。
3. 使用 Reciprocal Rank Fusion：`score = Σ 1 / (rrf_k + rank)`；首期 `rrf_k=60` 仅作为可配置基线值。
4. 每条召回通道先按 Bundle 折叠 Profile hit，再做 Bundle 级 RRF，避免多附件 Bundle 重复占位或按附件数获益。
5. 每个 Collection 使用同一候选预算和最低相关性阈值，随后以稳定公平交错形成跨 Collection 候选池；未达阈值的 Collection 不强制占位。
6. 将有限 BundleCandidate 构造成 Candidate Card，由 `LLMDiscoverySelector` 做 Setwise/Listwise 比较；单 READY Member 自动聚焦，多个 Member 才选择附件范围；输出 CandidatePlan，不直接输出 Evidence。

### Evidence

1. 重新验证 candidate 的 Collection Scope、当前 Bundle Revision、READY Member、Document Version、Active Parse View 和 `evidence_status=ACTIVE`。
2. 在受限 Evidence 范围内分别执行全文和向量候选；表格/Spreadsheet 可额外以表头、Sheet 和 cell range 作为关键词字段，但不另建绕过 Scope 的索引。
3. 同样用版本化 RRF 融合；先按 `document_version_id + parse_view_id + source_key/section_key` 去重，再按问题预算扩展相邻页、同章节或同表格范围。
4. 将 ANCHOR 命中及其必要结构上下文组成不跨 Document Version/View 的 Evidence Group，由 `LLMEvidenceSupportJudge` 输出直接支持、部分支持、仅上下文、矛盾或不支持，并最终分配 PRIMARY/STRUCTURAL_CONTEXT/NEIGHBOR；不对孤立 Chunk 生成数值 rerank 分。
5. Relation 仍是后续受评测开关；命中理由、各路分数、RRF rank、LLM 序数等级和扩展来源分别保存，不折算成单一总分。

## 参数、版本与可观测性

Collection 可保存受控 `retrieval_policy_key`，指向版本化策略配置：全文/vector 候选数、RRF 常数、最低阈值、邻接窗口、每 Document 上限、总 token 预算和是否启用特定 ViewType。Policy 不得覆盖应用级 Embedding 模型、维度或距离度量；Agent Binding 不保存这些参数，保持 Collection 平权和策略集中治理。

Policy 还必须固定 Selector/Judge 的模型、Prompt 版本、候选批大小、超时、结构化输出 Schema 和降级方式。完整契约见[LLM 候选选择与证据判断](42_step_5_llm_selection_and_evidence_judging.md)。

每次请求记录不可逆的 `retrieval_run_id` 和：policy key、embedding model、Query 规范化结果、每阶段候选数、各过滤原因、RRF rank、耗时、最终 Citation 数。在线日志只保存受控摘要；完整评测可在隔离数据集保存 question ID 和标注结果。

## 基线验收

- 同一数据/模型/策略/查询在索引稳定后产生确定性候选顺序（允许近似向量检索的声明误差范围）。
- 禁用任意 Collection、切换 Revision、隔离 Version 或替换 Parse View 后，旧对象不能进入结果。
- 每个 Evidence 命中都能追溯到 Discovery candidate、当前 Revision 和定位；任何跨 Domain/未绑定 Collection 访问被拒绝。
- 报告按文件类型、问题类型、单/多附件、表格/非表格和 Collection 维度拆分的 Discovery Recall@K、Evidence Recall@K、NDCG、定位准确率、时延与 token 成本。
