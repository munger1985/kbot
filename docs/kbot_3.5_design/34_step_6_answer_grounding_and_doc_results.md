# 步骤 6 详细设计：回答溯源与前端 `doc_results`

## 问题与原则

V1 把检索 Top-K Chunk 直接作为前端 `doc_results/reference` 返回。它混淆了“语义候选”“模型阅读上下文”和“模型实际引用的来源”：得分高的 Chunk 可能只是在语义上接近，最终答案没有使用它，尤其在“列出关于 XX 的案例/Asset”场景会向用户展示噪音文档。

V2 将三者严格分离：

```text
Retrieval Trace（内部候选，允许噪音）
  → Citation Pack（供回答模型阅读的可引用 Evidence）
    → Answer Citation Usage（模型声明实际采用的 Evidence）
      → doc_results（前端仅显示已验证采用的 Bundle/Document 卡片）
```

前端不能再把 Discovery/Evidence Top-K 原样展示为 reference。一个来源只有同时满足“模型声明使用”和“KC 验证可引用”才进入 `doc_results`。

## 回答模型输出契约

Citation Pack 为每条 Evidence 分配不可伪造的短标签，如 `E1`、`E2`，并包含稳定内部 identity。回答模型使用结构化输出：

```json
{
  "answer_markdown": "……[E1]……",
  "claims": [
    {"claim_id": "c1", "text": "……", "evidence_labels": ["E1"]}
  ],
  "used_evidence_labels": ["E1", "E3"],
  "selected_bundle_ids": [101]
}
```

`used_evidence_labels` 必须是 Citation Pack 的子集；每个事实性 claim 至少引用一个 Evidence。对于纯澄清、拒答或无事实回答，允许无引用，但不得生成 `doc_results`。模型不得提交 Bundle/Document ID 或自造标签；KC/Skill 只接受预分配 label。

## 回答后验证与投影

`AnswerGroundingVerifier` 在发送 SSE 前执行：

1. 校验所有 label 存在于本次 Citation Pack，去除未使用、无效或邻接-only 的伪引用。
2. 校验每个被引用 Evidence 仍属于本请求的当前 Revision、授权 Collection 和可见 Parse View；切换期间失效则重新检索或降级为证据不足。
3. 校验每个事实性 claim 有至少一个 PRIMARY Evidence。首期做结构/定位/范围校验；后续可加入轻量 NLI 或 LLM claim-evidence verifier，但不能仅依赖模型自报。
4. 从有效 `used_evidence_labels` 向上投影为唯一的 Bundle/Revision/Document 卡片，生成前端 `doc_results_v2`。同一 Asset 多条 Evidence 只生成一张 Asset 卡片，附件命中作为其子项/定位摘要。

`doc_results_v2` 卡片包括 Bundle 标题（Asset 标题）、Collection、当前 Revision、命中的 Manifest/附件名称、采用的 Evidence 数量和安全定位摘要。它不包含 Chunk 正文、向量得分、未采用候选、其他 Asset 的存在性或持久对象 URI。用户展开卡片时才请求授权的 Evidence 定位/预览。

## “列出关于 XX 的案例/Asset”专用门槛

该意图的主要交付是 Asset 列表，不是 Chunk 列表。流程为：

1. Discovery 广召回候选 Asset；允许语义相关噪音，只作内部 candidate。
2. Evidence 在每个候选 Asset 内寻找直接支持“XX”的 Manifest 或附件 Evidence。
3. 回答模型只选择存在直接支持 Evidence 的 Asset，并为每个列出的 Asset 提供至少一个 PRIMARY Evidence label。
4. Verifier 校验 `selected_bundle_ids` 与该 Asset 的有效使用 Evidence 一一对应；没有直接证据的 Asset 从答案列表和 `doc_results_v2` 同时移除。

因此，高分但仅语义相近、内容没有提到 XX 的 Chunk 最多停留在 Retrieval Trace，不能成为展示给用户的案例 Asset。对于附件不可用但 Manifest 明确支持的 Asset，可展示并标注“主信息命中，附件不可用”；没有任何支持 Evidence 的候选不能展示。

## SSE 与兼容

V2 SSE 的最终事件使用独立结构：

```json
{
  "answer": "...",
  "citations_v2": [{"label": "E1", "citation": {"...": "..."}}],
  "doc_results_v2": [{"bundle_id": 101, "title": "...", "used_evidence_ids": [9001]}],
  "grounding_status": "VERIFIED"
}
```

`grounding_status` 为 `VERIFIED/PARTIAL/INSUFFICIENT`。`PARTIAL` 必须标出未被证据支持的子问题，不得静默返回全部候选文档。V1 的 `doc_results` 保持原格式直至独立下线；同一次响应不混用 V1 Chunk reference 与 V2 Asset/Document 卡片。

## 验收

- 前端 `doc_results_v2` 是 `used_evidence_labels` 上卷后的去重集合，不是 Retrieval Top-K。
- 每张列出的 Asset 卡都有至少一个当前、授权、PRIMARY Evidence；移除该 Evidence 后卡片消失。
- 对语义相似但内容不含目标词/事实的 Asset，允许 Discovery 命中，但不得出现在最终答案或 `doc_results_v2`。
- 评测分别记录候选召回、答案 claim 支撑率、`doc_results_v2` Precision/Recall、错误展示率和引用定位准确率。
