# 步骤 6 详细设计：RootAgentV2 Grounded Answer

## 目标

把“检索到的候选内容”和“回答真正使用的引用”彻底分开。回答模型只提交本次请求的
`C1`、`C2` 等标签，不允许提交 Evidence/Document 数据库 ID 作为引用凭据；RootAgentV2
在发送 SSE 前将标签与 KC 返回的 Citation Pack 求交集。

## 模型契约

`LLMAnswerGenerator` 给模型发送问题和限长的 PRIMARY/STRUCTURAL_CONTEXT 内容（不发送
Bundle/Document 数据库 ID），要求
返回 JSON：

```json
{
  "answer_markdown": "答案 [C1]",
  "claims": [{"claim_id": "claim-1", "text": "事实", "citation_labels": ["C1"]}],
  "used_citation_labels": ["C1"],
  "selected_bundle_ids": []
}
```

`selected_bundle_ids` 仅为可选展示提示，不是授权或引用凭据；通常留空，由已验证的
Citation Label 自动上卷 Bundle。解析失败时保留纯文本答案但不信任任何引用；验证器将
状态置为 `INSUFFICIENT`。
`citation_groups_from_payload` 负责 HTTP 字典到领域 DTO 的边界转换。

## SSE 顺序

1. `knowledge_task_result`：检索任务和候选摘要。
2. `citation_candidates_v2`：仅供回答模型使用的候选 Citation Pack。
3. `answer`：模型生成的原始 Markdown。
4. `grounded_answer`：校验后的答案、`citations_v2`、状态和不支持的 Claim。
5. `doc_results_v2`：只投影模型实际引用的 PRIMARY Evidence 所属 Bundle/Document。
6. `grounding_status`、`done`：终态及降级原因。

无效标签会从答案中移除；没有 PRIMARY Evidence 的 Group、未被 Claim 引用的 Bundle、
无引用事实不会进入 `doc_results_v2`。回答模型异常不影响检索结果返回，但终态明确标记
为 `INSUFFICIENT`。

## 模型选择

请求可通过 `KnowledgeTask.answer_model` 注入模型名；生产路由默认按 Agent 配置解析
`llm_model`。这只是回答模型，与 Collection 唯一绑定的 Embedding 模型相互独立。

## 验收

- 伪造 `C99` 不得出现在最终引用或文档卡片中。
- 同一 Bundle 多个 Citation 只生成一张 `doc_results_v2` 卡片。
- V1 `doc_results` 字段和旧 RootAgent 不被调用。
- 回答模型不可用时可观察到 `grounding_status=INSUFFICIENT`。
