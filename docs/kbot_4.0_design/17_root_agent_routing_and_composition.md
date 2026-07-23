# 4.0 Root Agent 路由与结果组合

## 定位

Root Agent/Supervisor 是用户请求的统一编排入口，不拥有 Knowledge、Ops 或业务数据表。它负责理解用户目标、选择领域入口、创建 Run/Task、组合 Artifact 和生成最终回答。

```text
User Request
     ↓
Root Agent / Supervisor
     ├─ Document Agent
     ├─ AIOps Agent
     ├─ MCP Data Adapter
     └─ Conversation Response / Response Composer
```

Root Agent 不直接调用 KC Repository、数据库、DB Executor 或旧 Service；所有领域能力通过 Client、MCP Tool 或持久化 Task 调用。

## Response Composer 的作用

`Response Composer` 是 Runtime 内的最终响应合成组件，不是独立业务 Agent。它接收 Root Run 的原始问题、路由结果和已验证 Artifact，调用 LLM 生成用户可见的答案，并执行最终 Grounding：

- 普通闲聊：基于 Conversation Skill 的草稿完成语言和格式整理；
- 文档问题：只使用 `CITATION_PACK` 和其中实际引用的 Evidence；
- MCP 问数：使用 `QUERY_RESULT`，不把结果伪装成文档引用；
- AIOps：使用 `DIAGNOSIS`、`EXECUTION_REPORT` 和其证据；
- 混合问题：分别标注文档依据、实时数据、运维事实和推断结论。

Response Composer 不负责查询 KC、执行 MCP/DB 操作、创建 Task 或修改 Artifact。它只能消费已完成的 Artifact，输出 `GROUNDED_ANSWER`；引用校验失败时必须降级为证据不足，而不是补写无来源结论。当前 3.5 的 `RootAgentV2` 中 `AnswerGenerator + AnswerGroundingVerifier` 就是这一组件的过渡实现。

## 路由决策 DTO

```text
RouteDecision {
  route_type: CONVERSATION | DOCUMENT | MCP_DATA | AIOPS | HYBRID | CLARIFY
  targets[] {
    target: document_agent | aiops_agent | mcp_data | conversation
    task_key
    intent
    input_refs
    required_scopes
    completion_requirement: REQUIRED | OPTIONAL
  }
  parallel_groups
  confidence
  clarification_question
  policy_warnings
  classifier_version
}
```

路由结果必须是结构化 DTO，不能直接使用 LLM 返回的自然语言作为执行命令。`confidence` 仅用于决定是否需要澄清，不能绕过权限和 Policy Gate。

## 路由规则

| 用户意图 | 目标 | 说明 |
| --- | --- | --- |
| 普通闲聊、解释流程 | Conversation | 不调用领域工具 |
| 查找资产、案例、附件内容 | Document Agent | 返回 CitationPack，再由 Root Grounding 生成回答 |
| 查询 Excel 数值或业务数据 | MCP Data Adapter | 沿用现有受控 MCP 问数链路，返回 QueryResult |
| 数据库健康、锁、连接、表空间、告警 | AIOps Agent | 进入独立 Ops Run，不直接调用 DB Executor |
| “根据 SOP 判断当前数据库是否异常” | Document + AIOps | 文档依据和实时诊断并行，分别保存 Artifact |
| “查询数据并解释相关制度” | MCP Data + Document | 业务数据和制度文档分别检索 |
| 目标、范围或权限不明确 | Clarify | 不启动具有副作用的 Task |

路由器先执行确定性规则（目标资源、关键术语、入口类型、已绑定 Agent），再使用受约束的 LLM 分类器处理模糊自然语言。LLM 只能从允许的 Route Type 中选择，不能指定任意 URL、表名、SQL 或未授权 Collection。

## 单路由和并行路由

简单请求走单路由，减少延迟和成本：

```text
Root Run
  └─ Document Task → CitationPack → Grounded Answer
```

跨领域请求建立并行 Task Group，只有所有必要结果完成后才进入 Compose：

```text
Root Run
  ├─ Document Task  → CitationPack
  ├─ MCP Data Task  → QueryResult
  └─ Compose Task   ← 两个 Artifact
```

AIOps 请求通常不与普通问数混用；如果用户同时要求实时运维状态和业务数据，AIOps Run 与 MCP Data Task 仍然独立执行。并行 Task 必须有独立预算、超时和错误状态，一个领域失败不能伪装成另一个领域成功。

跨服务 AIOps Task 必须持久化为 `KBOT_AGENT_DELEGATION`，使用 Child Run ID、事件游标和有限租约恢复；不能在 Root Worker 中等待子 SSE。完整父子契约见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

## 结果组合规则

Root Composer 只消费已验证 Artifact：

- `CITATION_PACK`：文档、资产、SOP 和 Evidence 引用；
- `QUERY_RESULT`：MCP 问数返回的结构化业务数据；
- `DIAGNOSIS` / `EXECUTION_REPORT`：AIOps 观察、诊断和执行结果；
- `ANSWER_DRAFT`：对话或领域草稿。

AIOps 子 Run 先被映射为受限 `DELEGATED_AIOPS_RESULT` Artifact；Root 不直接引用 Ops 表或远端 Artifact 外键。

组合回答必须区分：

```text
实时事实       ← QueryResult / AIOps Observation
文档依据       ← CitationPack
推断和解释     ← LLM Composer
变更结果       ← ExecutionReport
```

Document Citation 不能替代实时监控事实，MCP QueryResult 不能作为文档引用，AIOps 诊断不能自动证明业务数据结论。Document/Data/AIOps 使用独立 `D*/Q*/O*` 标签命名空间；最终 Grounding 只保留回答中实际使用的类型化来源引用。4.0 不再输出旧 `doc_results/doc_results_v2`。

## 澄清、拒绝和降级

- 缺少 domain、Agent、Collection 或目标实例时进入 `CLARIFY`，不猜测默认资源；
- 权限不足时返回 `SCOPE_DENIED`，不自动改走其他 Agent；
- KC 不可用时 Document Task 返回明确失败，不能回退到旧 KB/TxtChunk；
- MCP Tool 不可用时只报告问数失败，不把问题改写成 Document 查询；
- AIOps 需要审批时 Run 进入 `WAITING_APPROVAL`，不自动执行；
- 一个并行分支失败时，Composer 必须标明缺失来源，不生成看似完整的答案。

## 与各领域接口

Root Agent 使用以下版本化接口：

```text
DocumentAgentClient.query(DocumentQueryTask)
AIOpsDelegationClient.start(AIOpsDelegationRequest)
MCPDataClient.execute(MCPDataRequest)
ResponseComposer.compose(CompositionInput)
```

Document Agent 返回 Artifact；AIOps Agent 返回受限 Result Envelope；MCP Data Adapter 返回受控 QueryResult。Root Agent 不依赖这些实现的数据库、Prompt 或内部状态机。

## 评测指标

路由评测至少覆盖：

- Document / MCP Data / AIOps 分类准确率；
- HYBRID 场景的并行任务召回率；
- 越权路由拦截率；
- 澄清问题有效率；
- 错误分支是否被明确标注；
- 最终回答中 Citation、QueryResult 和 Ops Report 的来源准确率；
- 单路由与并行路由的延迟、成本和重复调用率。

## 第一版范围

第一版实现 Root → Document Agent、Root → AIOps Agent 和 Root → MCP Data Adapter 三条路由；不实现独立 Data Agent，不实现任意 Agent 动态发现，不允许 Agent 之间通过共享数据库或全局内存通信。
