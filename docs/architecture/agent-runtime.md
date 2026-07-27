# Agent Runtime

## Agent 与 Skill

Agent Definition 归属一个 Domain，声明能力、指令、模型角色和状态。模型配置采用
`models_json`，值是对外稳定的 `served_model_name`；例如 `router_llm`、
`context_llm`、`composer_llm`、`memory_llm` 和 `memory_embedding`。不同功能可
使用不同成本和能力的模型。

Skill 是 Runtime 内的版本化执行单元。当前文档链路由上下文改写、知识检索和回答
组合等 Skill 构成；通用聊天、MCP 问数、ECharts 和 Dify Retrieval Adapter 使用
各自的类型化 Artifact。AIOps 是独立服务和独立入口，不作为普通文档 Skill
直接访问其数据库。

## 执行模型

```text
Turn → Run → Plan → Task DAG → Skill/Delegation
                         └─ Artifact → Event → 最终 Artifact
```

Run 固化 Agent、模型、检索范围、Domain、用户和策略快照。Task 通过 Lease、
Heartbeat、重试次数和幂等键执行；Worker 崩溃后只接管未完成 Task。Skill 不共享
可变全局 Context，而是读取声明的输入 Artifact 并产生一个版本化输出 Artifact。

主要 Artifact 包括 `CONTEXT_REWRITE`、`CITATION_PACK`、`QUERY_RESULT`、
`ECHARTS_CONFIG` 和 `GROUNDED_ANSWER`。Response Composer 只能使用已完成 Artifact；
没有 Citation 时必须返回证据不足，不能调用模型补写无来源内容。

## Conversation 与记忆

Conversation 保存 Turn 和用户/助手 Item，可再次打开并按顺序渲染。每个 Turn
先加载最近消息、摘要和可用长期记忆，再由 Context Rewrite 生成独立问题。记忆
分为会话摘要、情景记忆和用户/Agent 范围记忆；抽取使用 `memory_llm`，语义检索
使用 `memory_embedding`。Embedding 身份一旦建立不得在原索引上直接替换。

Prompt 先从数据库的版本化 Registry 读取；缺少数据库记录时回退到
`packages/platform_core/src/platform_core/resources/prompts.toml`。Prompt Key、
版本、变量和输出 Schema 均可追溯。

## SSE 与可追溯性

公开 Run SSE 支持 `Last-Event-ID` 重放。常用事件包括：

- `RUN_CREATED/RUN_STARTED`、`TASK_*`；
- `memory.context_loaded`、`query.rewritten`、`skill.started`；
- `retrieval.completed`、`data.query.completed`、`chart.completed`；
- `thinking.delta`、`answer.delta`、`answer.completed`；
- `RUN_COMPLETED/RUN_FAILED/RUN_CANCELLED`。

`thinking.delta` 只包含可公开的工作过程，例如将调用哪个 Skill、检索到多少候选或
正在组织几组证据，不暴露模型隐藏推理。事件先持久化再由 Main API 转为 SSE，因此
断线重连不会重新执行 Skill。开发环境可用 `tools/dev_console/agent-chat.html` 和
Run 调试台查看事件、Artifact、任务及跨服务日志。
