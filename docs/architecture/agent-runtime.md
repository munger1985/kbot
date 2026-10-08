# Agent Runtime

## Execution Spec 与 Skill

Agent Runtime 不拥有可编辑 Agent Definition。Knowledge Retrieval App 与 AIOps
App 分别拥有自己的 Agent 与不可变版本；创建会话或运行前，所属 App 按统一核心权限
和资源状态编译 `AgentExecutionSpec`。Runtime 在 Conversation 和 Run 中冻结该
快照，后续 Agent 修改不会改变既有执行事实。

Execution Spec 声明能力、指令、检索范围、问数绑定和功能模型。不同功能可使用
不同成本和能力的模型；运行时使用冻结的模型身份和调用配置。

Skill 是 Runtime 内的版本化执行单元。知识检索 Agent 链路包括上下文改写、
知识检索、Data Query、Hybrid、回答组合和 Chart。问数 Skill 可按冻结配置调用
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
`CHART_SPEC` 和 `GROUNDED_ANSWER`。Response Composer 只能使用已完成 Artifact；
没有 Citation 时必须返回证据不足，不能调用模型补写无来源内容。

Chart 是独立、跨 App 的只读 Skill。输入只能来自已经完成的结构化 Artifact，输出
`CHART_SPEC.v1`，描述图表意图、坐标轴、序列、数据点、单位、布局和来源，不包含 ECharts
option、HTML、formatter 或任何可执行代码。常见问数由 `QUERY_RESULT` 确定性生成比较图或
时间序列图；AIOps 独立状态机复用同一平台 Chart Skill，从监控原始点和数据库容量快照生成
趋势图与容量图，不进入知识检索 Root Planner。

`CHART_SPEC.v1` 是呈现合同，不是新证据：Skill 不得从图形重算趋势、根因或预测；所有数据点
必须能回溯到 `source_ids`。时间序列每条最多保留 240 个展示点，采用保留分桶极值的确定性
降采样，原始证据 Artifact 不被替换。前端按 App 设计系统实现渲染，不能让模型生成页面代码。

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
断线重连不会重新执行 Skill。开发环境只保留
`tools/dev_console/operations-logs.html` 查看各服务运行日志和 API 访问日志；
Run、Task、Artifact 与事件应通过正式 API 或 KM 页面提供的业务入口观察。
