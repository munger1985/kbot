# Agent Runtime

## Execution Spec 与 Skill

Agent Runtime 不拥有可编辑 Agent Definition。Knowledge Retrieval App 与 AIOps
App 分别拥有自己的 Agent、不可变版本和 Grant；创建会话或运行前，所属 App 将
已授权版本编译为 `AgentExecutionSpec`。Runtime 在 Conversation 和 Run 中冻结该
快照，后续 Agent 修改不会改变既有执行事实。

Execution Spec 声明能力、指令、检索范围、问数绑定和功能模型。不同功能可使用
不同成本和能力的模型；运行时使用冻结的模型身份和调用配置。

Skill 是 Runtime 内的版本化执行单元。知识检索 Agent 链路包括上下文改写、
知识检索、Data Query、Hybrid、回答组合和 ECharts。问数 Skill 可按冻结配置调用
MCP 或 Semantic Data Query。AIOps Agent 使用独立的 AIOps Run、Worker 和状态机，
不进入知识检索 Root Planner。

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
