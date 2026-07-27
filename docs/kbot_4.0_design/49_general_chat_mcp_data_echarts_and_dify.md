# 通用聊天、问数、ECharts 与 Dify Adapter

## 范围与入口

4.0 的每个 Agent 都是独立入口。普通聊天 Agent 可在自身启用的
`conversation`、`document`、`mcp_data` 中选择一个路由；AIOps Agent 使用独立
入口和独立 Run，不参与普通聊天 Router。当前不实现 Graph、跨领域混合任务、
反馈、Tags 或 Agent Group。未来需要组合多个 Agent 时新增显式 Agent Group，
不让单个 Root Run 隐式跨 Agent 调用。

单能力 Agent 使用确定性路由；多能力 Agent 必须配置
`models.router_llm`，Router 只能从已启用能力中选择
`CONVERSATION / DOCUMENT / MCP_DATA / CLARIFY`。分类结果经过 Pydantic 校验，
模型不能生成 Skill、URL、SQL 或任意执行计划。
低于 `router_confidence_threshold` 的结果转为 `CLARIFY`，并作为一条
`CLARIFICATION_REQUIRED` 助手消息进入会话，而不是以 HTTP 错误结束 Turn。

## 执行计划

```text
Conversation:
  Context Rewrite → Conversation Response → GroundedAnswer

Document:
  Context Rewrite → KC Retrieval → Response Composer → GroundedAnswer

MCP Data:
  Context Rewrite → MCP Data Query → [ECharts] → Response Composer
```

ECharts 只在用户明确要求图表或可视化时加入计划。`QUERY_RESULT` 与
`ECHARTS_CONFIG` 是相互独立的类型化 Artifact；最终回答保留
`query_result_id`、行数据、截断状态和图表配置，不伪装成文档引用。

## SelectAI/AIReport 问数

问数外部协议沿用 3.x：

```json
{
  "profile": "SALES_PROFILE",
  "user": "portal-user",
  "ask": "查询本月销售额"
}
```

Agent 通过 `data_profile_name` 绑定 Profile。服务配置保留
`api_endpoint`、`profiles_endpoint`、超时，并增加行数和响应大小上限。
明文 API Key 只能通过 `KBOT_MCP_DATA_API_KEY` 注入，不能写入 TOML。
`GET /api/v1/data/profiles` 供 Portal 读取可选 Profile。
`QUERY_RESULT` 使用独立 `query_result_id`；`data.query.completed` SSE
事件只携带列名和最多 20 行预览，完整结果进入最终回答 Artifact。

## ECharts 安全边界

图表模型只消费当前 `QUERY_RESULT`，输出纯 JSON 的 `chart_type`、`title` 和
`option`。服务端拒绝原型污染字段、JavaScript URL、函数和箭头函数字符串。
前端直接按 JSON 配置渲染，不执行 `eval` 或动态脚本。完整配置随
`chart.completed` SSE 事件返回，并通过 `query_result_id` 绑定原始问数结果。

## Dify Retrieval Adapter

`POST /api/v1/integrations/dify/retrieval` 实现 Dify External Knowledge Base
协议。`knowledge_id` 是 KBot Agent UUID；Adapter 只检索该 Agent 在当前
Domain 中绑定且启用的 Collection，并将 KC 两阶段检索结果映射为：

```json
{
  "records": [{
    "metadata": {
      "collection_id": "...",
      "bundle_id": "...",
      "document_id": "...",
      "locator": {"page": 3}
    },
    "score": 0.91,
    "title": "case.pdf",
    "content": "可引用正文"
  }]
}
```

Adapter 不创建 Conversation、不写记忆，也不复用聊天 Skill。额外
`metadata_condition` 当前明确返回不支持，避免静默忽略过滤条件。

## 配置与验收

- `ask_data_api` 外部服务配置只属于 Agent Runtime；
- 多能力 ACTIVE Agent 必须配置 Router 模型；
- 启用 `mcp_data` 的 ACTIVE Agent 必须配置 `data_profile_name`；
- AIOps 不能与其他能力同时绑定；
- Router、通用回答、问数解释和 ECharts Prompt 均从 Prompt Registry 读取；
- Dify 结果必须来自 Evidence，不能返回 Discovery 画像代替正文。
