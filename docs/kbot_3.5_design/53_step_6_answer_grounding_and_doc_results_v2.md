# 步骤 6 详细设计：Answer Grounding 与 `doc_results_v2`

## 验证规则

回答模型只能提交 `used_citation_labels`，不能提交数据库 ID 或自造引用。`AnswerGroundingVerifier` 将这些标签与本次 Citation Pack 求交集，并确认每个事实 Claim 至少拥有一个仍含 PRIMARY Evidence 的有效 Group。无效标签、context-only Group 和没有引用的事实 Claim 不得进入最终引用集合。

状态为：

- `VERIFIED`：所有事实 Claim 均有有效引用。
- `PARTIAL`：部分 Claim 有引用，部分未被支撑。
- `INSUFFICIENT`：没有可用 PRIMARY Citation，或事实回答完全没有引用。

## 前端投影

`doc_results_v2` 只从有效 `used_citation_labels` 的 PRIMARY Evidence 上卷生成，并按 Collection + Bundle 去重。每张卡包含 Bundle/Revision、采用的 Citation Label、PRIMARY Evidence ID、Document Version 和定位摘要；不包含未采用候选、Chunk Top-K、向量分数、对象存储 URI 或失败附件内部错误。

对于“列出 XX 案例/Asset”问题，模型选择的 Bundle 必须与有效 Citation Group 一一对应；没有直接支持 Evidence 的候选会同时从答案列表和 `doc_results_v2` 移除。

## 已落地内容

- `AnswerDraft`、`AnswerClaim`、`AnswerGroundingVerifier` 和 `GroundingResult` DTO。
- 有效 Citation Label 过滤、Claim 支撑状态、Bundle 去重投影。
- 测试覆盖伪造标签、无引用事实和多候选选择。

该模块是 Skill/Root Agent 发送 SSE 前的纯验证边界，已经由 RootAgentV2 接入；现有 V1
SSE 不变，V2 仅在独立事件中发送 `GroundingResult`。
