# Agent Conversation、记忆与可追溯流式交互

## 目标和设计结论

4.0 在现有 `Run / Task / Artifact / Event` 执行内核之上增加
Conversation 记忆层。Conversation 负责多轮语义和用户可见历史，Run
负责一次执行，二者不能合并成同一张表或让 SSE 充当状态存储。

本设计同时满足：

- 使用最近对话、会话摘要和长期记忆把当前输入改写为可独立理解的问题；
- 关闭页面后重新打开 Conversation，能够恢复用户问题、最终回答、引用和
  可公开执行过程；
- 流式展示路由、计划、Skill、检索、子 Agent、等待和回答生成进度；
- 不暴露模型隐藏思维链，不把模型生成的“想法”当作审计事实；
- Root Agent 统一拥有 Conversation，Specialist 和独立子 Agent 只接收
  最小必要的类型化上下文。

记忆属于 `agent_runtime` 的独立 bounded module。API、查询改写和上下文组装
运行在 Agent Runtime API/Worker 中；摘要和长期记忆整理由独立
Memory Consolidation Worker 异步执行。4.0 不单独部署 Memory Service，
但通过 Port、DTO 和自己的 Repository 隔离，未来可以直接拆进程或拆库。

问题改写、摘要、记忆提取、冲突判断和回答组合所需 System Prompt 统一登记
在 `packages/platform_core/src/platform_core/resources/prompts.toml`，并按
[46_versioned_prompt_registry.md](46_versioned_prompt_registry.md) 初始化到
Platform Prompt 表。运行时数据库优先、文件兜底，每次模型调用冻结
Prompt Key、Version 和 Hash；不能在各 Skill 中内嵌另一份 Prompt。

## 分层记忆

| 层次 | 内容 | 存储 | 可信语义 |
| --- | --- | --- | --- |
| 会话事实源 | 用户/助手消息、附件引用、Turn/Run 关联 | Conversation Item | 不可变原文 |
| 工作记忆 | 当前主题、目标、未解决项、截至某轮的摘要 | Memory Snapshot | 可重建派生数据 |
| 长期语义记忆 | 用户明确事实、偏好及其时间变化 | Memory Item | 带来源、置信度和有效期 |
| 情景记忆 | 历史问题、结果和 Turn 引用 | Memory Item/Source | 每个成功 Turn 可配置生成 |
| 程序性记忆 | Agent 指令、Skill、SOP 和策略 | Definition/Skill Version | 只能受控发布 |

Knowledge Core 保存组织知识和文件 Evidence，不保存聊天记忆。KC Evidence
可以在当前回答中被引用，但不能复制成用户画像。Assistant 回答和模型推断也
不能自动成为用户事实。

## Conversation、Turn 和 Run

```text
Conversation
 ├─ Turn 1
 │   ├─ USER Conversation Item
 │   ├─ Root Agent Run
 │   │   ├─ Context Rewrite Artifact
 │   │   ├─ Task / Delegation / Evidence Artifact
 │   │   └─ Grounded Answer Artifact
 │   └─ ASSISTANT Conversation Item
 ├─ Turn 2
 │   └─ ...
 └─ Memory Snapshot（覆盖至 committed_turn_sequence）
```

Conversation 创建后固定 `app_id + domain_id + actor_id + agent_id`。
切换 Agent 创建新 Conversation；后续如需分支，显式从某个已提交 Turn
创建新 Conversation，不能修改旧时间线。

一个 Turn 只有一个 Root Run。用户消息持久化后才创建 Run；最终
`GROUNDED_ANSWER` 提交成功后，以其内容和引用创建不可变 Assistant Item，
再把 Turn 标记为 `COMPLETED`。Run 失败或取消时仍保留 User Item，并为 Turn
保存结构化失败状态，前端可以正确恢复。

同一 Conversation 首版只允许一个非终态 Turn。并发提交返回
`409 CONVERSATION_TURN_IN_PROGRESS`，避免两个问题基于不同历史并行改写。
提交 Turn 使用 `Idempotency-Key` 和 Conversation `row_version`；重试不能
产生重复 User Item 或 Run。

## 物理数据模型

所有标识使用 UUIDv7；除 Conversation 外的表通过外键继承 Domain 隔离，
不重复保存 `app_id/domain_id`。

### `KBOT_AGENT_CONVERSATION`

- `CONVERSATION_ID`、`APP_ID`、`DOMAIN_ID`、`ACTOR_ID`、`AGENT_ID`；
- `TITLE`、`STATUS: ACTIVE|ARCHIVED|DELETED`、`ROW_VERSION`；
- `LAST_TURN_SEQUENCE`、`LAST_ITEM_SEQUENCE`；
- `RETENTION_POLICY`、`LAST_ACTIVE_AT`、`CREATED_AT`、`UPDATED_AT`。

索引至少包含 `(APP_ID, DOMAIN_ID, ACTOR_ID, LAST_ACTIVE_AT)` 和
`(AGENT_ID, LAST_ACTIVE_AT)`。APEX 列表视图只投影标题、状态、最后活动时间和
最后一轮安全摘要，不返回记忆画像或完整执行数据。

### `KBOT_AGENT_CONVERSATION_TURN`

- `TURN_ID`、`CONVERSATION_ID`、单调递增 `TURN_SEQUENCE`；
- `USER_ITEM_ID`、`ROOT_RUN_ID`、`ASSISTANT_ITEM_ID`；
- `STATUS: ACCEPTED|RUNNING|WAITING|COMPLETED|FAILED|CANCELLED`；
- `RAW_INPUT_HASH`、`IDEMPOTENCY_KEY`、开始/完成时间。

`(CONVERSATION_ID, TURN_SEQUENCE)` 和
`(CONVERSATION_ID, IDEMPOTENCY_KEY)` 唯一；`ROOT_RUN_ID` 非空时唯一。

### `KBOT_AGENT_CONVERSATION_ITEM`

- `ITEM_ID`、`CONVERSATION_ID`、单调递增 `ITEM_SEQUENCE`、`TURN_ID`；
- `ITEM_TYPE: MESSAGE|ATTACHMENT_REF|NOTICE`；
- `ROLE: USER|ASSISTANT|SYSTEM`；
- `CONTENT_JSON`、`CONTENT_HASH`、`RUN_ID`、`ARTIFACT_ID`；
- `VISIBILITY: USER|INTERNAL`、`CREATED_AT`。

用户/助手正文和最终 Reference Card 保存在这里；大型内容只保存 URI、Hash
和安全摘要。Tool 调用、检索候选和子 Agent 内部过程不复制为 Message，
仍由 Run Event/Artifact 保存。

### `KBOT_AGENT_MEMORY_SNAPSHOT`

- `SNAPSHOT_ID`、`CONVERSATION_ID`、`COVERED_TURN_SEQUENCE`；
- `SUMMARY_JSON`：主题、目标、已确认事实、未解决项和实体；
- `SOURCE_HASH`、`MODEL_NAME`、`PROMPT_VERSION`；
- `STATUS: BUILDING|ACTIVE|SUPERSEDED|FAILED`、创建时间。

同一 Conversation 只有一个 `ACTIVE` Snapshot。新快照成功后再切换；
失败时继续使用旧快照加未覆盖的最近原文。Snapshot 是派生数据，可以从
Conversation Item 重建，不能覆盖原消息。

### `KBOT_AGENT_MEMORY_ITEM`

- `MEMORY_ID`、`MEMORY_TYPE: USER_FACT|USER_PREFERENCE|EPISODIC`；
- `SCOPE_TYPE: USER_AGENT|USER_SHARED`、`ACTOR_ID`、可空 `AGENT_ID`；
- `CANONICAL_KEY`、`VALUE_JSON`、`SEARCH_TEXT`；
- `CONFIDENCE`、`SALIENCE`、`VALID_FROM`、`VALID_TO`；
- `STATUS: ACTIVE|SUPERSEDED|DISPUTED|DELETED`；
- `SENSITIVITY_LEVEL`、`EXPIRES_AT`、`ROW_VERSION`；
- `INDEX_PROFILE_ID`、`EMBEDDING`、审计时间。

默认写 `USER_AGENT`。只有 Agent 的 `config.memory.shared_keys` 明确列出的
Canonical Key 才写为 `USER_SHARED`，并令 `AGENT_ID=NULL`；其他 Agent
只能在同一 `app_id + domain_id + actor_id` 下读取，不能因为属于同一 Domain
就跨用户共享。

### `KBOT_AGENT_MEMORY_INDEX_PROFILE`

- `INDEX_PROFILE_ID`、`APP_ID`、`DOMAIN_ID`、`AGENT_ID`；
- `EMBEDDING_MODEL_NAME`、`EMBEDDING_DIMENSION`、`NORMALIZATION`；
- `CONFIG_SHA256`、创建时间。

同一 Agent 终身只有一个 Profile。Profile 冻结模型技术名称、全局维度和
L2 规范化；Memory 写入使用 `is_query=false`，召回使用 `is_query=true`。
Agent 配置入口禁止更换、移除或停用已设定的模型；若部署配置造成维度漂移，
Worker 也会明确失败。系统不提供重建、换模、双写、代际切换或退役流程。

### `KBOT_AGENT_MEMORY_SOURCE`

保存 `MEMORY_ID + CONVERSATION_ID + TURN_ID + ITEM_ID`、来源摘录 Hash、
提取器和提取时间。更正使用新 Memory Item 并把旧项标记为
`SUPERSEDED`；不得覆盖旧值和来源。

### `KBOT_AGENT_MEMORY_JOB`

Turn 成功后幂等创建一条异步归并任务，保存 `MEMORY_JOB_ID`、
`CONVERSATION_ID`、`TURN_ID`、状态、尝试次数、下次重试时间以及有限租约。
Worker 使用 `FOR UPDATE SKIP LOCKED` 领取；摘要和候选均生成成功后才在一个
Memory UoW 中切换 Active Snapshot、更新 Memory/Source 并完成 Job。
租约过期可恢复，达到最大尝试次数进入 `FAILED`，不反向改变已成功 Turn。
`RESULT_JSON` 保存候选数、ADD/CONFIRM/SUPERSEDE/DISPUTE/IGNORE 决策、
简短理由、结果 Memory ID 以及各 Prompt 的 Key/Version/Hash。

## 记忆驱动的问题改写

每个 Turn 在领域路由和 KC 检索前执行 `context-rewrite` Skill：

```text
原始输入
  + Active Snapshot
  + Snapshot 后的最近原文
  + 当前输入召回的长期记忆
  + Agent 允许的 Collection/能力摘要
        ↓
Context Rewrite
        ↓
ContextRewriteArtifact.v1
  raw_input
  standalone_query
  retrieval_queries[]
  resolved_references[]
  active_topic
  intent_hint
  memory_refs[]
  ambiguity
  clarification_question
```

`standalone_query` 用于 Router、Planner 和领域 Skill；`raw_input` 始终保留给
Response Composer，保证回答符合用户原始表达。Document Skill 使用改写后的
查询召回 KC，但 Citation 必须来自本轮 KC Evidence，不能引用 Memory Item
冒充文档来源。

改写器只能补全历史中有明确依据的实体、时间和范围。例如：

```text
上一轮：介绍案例 A。
当前轮：它使用了什么数据库？
改写：案例 A 使用了什么数据库？
```

存在两个同等可能对象时设置 `ambiguity=true` 并进入 `CLARIFY`，不能猜测。
改写结果必须记录所使用的 Snapshot、Item 和 Memory ID；用户当前输入的明确
表达优先级最高，历史摘要和长期记忆不得覆盖本轮更正。

## 上下文预算和召回

`ConversationContextAssembler` 按以下顺序构造只读 `MemoryContextPack.v1`：

1. 当前 Agent 指令和不可变策略；
2. Active Snapshot；
3. Snapshot 未覆盖且最接近当前轮的原始消息；
4. 按身份、Scope 和有效期强过滤后的长期记忆；
5. 当前用户输入；
6. 当前 Task 的领域 Artifact。

不能简单按固定消息数截断。Assembler 同时控制 token、消息数、时间跨度和
每类数据预算，优先保留当前问题、用户更正、未解决事项和最近原文。KC
Evidence、AIOps Observation 等当前事实拥有独立预算，不能被聊天历史挤出。

Memory 向量索引使用 Agent Runtime 自己的不可变 Index Profile。Agent 创建
时必须显式设置 `models.memory_embedding`；首条归并任务据此创建 Profile，
语义记忆和情景记忆写入规范化向量。新 Turn 在进入会话写事务前生成查询向量，
使用 `向量 0.60 + 词法 0.25 + Salience 0.15` 混合排序；服务异常时只降级为
词法和 Salience，不影响领域检索。

未配置 Profile 时继续使用 `canonical_key + search_text + value` 的字词重合
与 Salience。Profile 只属于聊天记忆，不复用 KC Collection 的模型绑定。
`models.memory_llm` 专门承担摘要、候选提取和冲突判断，可使用低成本蒸馏
LLM；`models.context_llm` 用于上下文改写；`models.composer_llm` 用于最终
回答。Embedding 模型一经设定即成为该 Agent 的永久数据契约，不允许修改。
需要使用另一模型时只能创建新的 Agent，并形成独立 Profile 和记忆空间。

## 长期记忆写入和冲突

原始 User/Assistant Item 同步提交；摘要和长期记忆候选在 Turn 完成后异步
处理：

```text
Turn Completed
  → Candidate Extractor
  → Schema / Sensitive-data Validator
  → Canonical Key Resolver
  → ADD | CONFIRM | SUPERSEDE | DISPUTE | IGNORE
  → Memory UoW
```

LLM 只输出候选和理由，不能直接更新 Repository。确定性规则要求：

- 只有用户明确陈述或明确要求“记住”的内容才能形成用户事实/偏好；
- Assistant 陈述、KC 内容和 LLM 推断不能自动写为用户事实；
- 本轮更正优先于旧记忆，并保留时间有效区间和全部来源；
- 密码、Token、数据库凭据、完整 SQL 结果和高敏感诊断数据禁止写入画像；
- “忘记这件事”走同步受审计命令，废止相关 Memory Item；
- Consolidation 失败不能影响 Turn 成功，也不能留下半完成 Active Snapshot。

候选提取和冲突评估使用独立 Prompt、独立输出 Schema 和独立调用快照。摘要
Prompt 只能压缩已有内容，不能把 Assistant 推断提升为用户事实；问题改写
Prompt 只能补全有明确来源的上下文，不能决定权限、Collection 或 Mutation。

新增和同值确认由确定性规则直接完成，只有同一 `canonical_key` 出现不同值时
才调用 `memory_conflict_assess`。模型只能返回 `SUPERSEDE`、`DISPUTE` 或
`IGNORE`，Application Service 重新锁定当前有效 Memory 并验证版本后才执行。
并发期间有效版本发生变化时整项 Job 重试，不能沿用过期决策。

自然语言忘记请求由 `memory_extract` 输出精确 `forget_keys`。只接受当前输入
明确要求且能对应现有记忆的完整 canonical key；禁止通配符、模糊匹配和模型
推断批量删除。执行后同时删除 Memory Source，避免已忘记信息仍残留在来源表。

启用 `memory.episodic_enabled` 后，每个成功 Turn 额外写入一条
`EPISODIC`：Canonical Key 为 `episode.<turn_id>`，内容保存 Conversation、
Turn 和序号引用，`SEARCH_TEXT` 保存限长的用户问题与最终回答，并绑定原始
User Item 来源。它参与混合召回，但不会冒充用户事实，也不会跨 Agent 共享。

## 历史会话恢复和前端投影

重新打开 Conversation 时，前端调用：

```text
GET /api/v1/conversations/{conversation_id}
GET /api/v1/conversations/{conversation_id}/turns?cursor=...
GET /api/v1/conversations/{conversation_id}/turns/{turn_id}/trace
```

Turn View 返回：

- User/Assistant 消息和创建时间；
- Run 状态、回答 Artifact 和实际使用的 Reference Card；
- `TraceSummary.v1`：路由、问题改写摘要、计划、Skill 执行结果、检索统计、
  子 Agent 状态、等待用户动作和错误；
- `trace_cursor`，用于按需加载完整公开 Trace Event。

Conversation API 不通过重放 SSE 拼装历史。消息来自 Conversation Item，
回答和引用来自最终 Artifact 投影，执行过程来自持久化 Run Event 的公开投影。
因此关闭浏览器、SSE 断线、Worker 重启后仍能恢复完全一致的界面。

历史页面中的“思考过程”实际展示 `TraceSummary` 和 Public Trace，不展示模型
隐藏推理文本。标题建议使用“执行过程”或“分析过程”，避免暗示它是模型未经
处理的内部思维链。

## 流式事件契约

4.0 延续 3.5 的过程可见体验，但所有用户可见事件必须先作为
`KBOT_AGENT_RUN_EVENT` 提交，再由 SSE 输出。禁止 Worker 只向当前连接
`yield` 临时消息。

稳定公开事件按点号命名并携带 `schema_version`：

| 事件 | 用户可见内容 |
| --- | --- |
| `turn.accepted` | Turn/Run 标识和排队状态 |
| `memory.context_loaded` | 使用几轮历史、摘要版本和长期记忆数量 |
| `query.rewritten` | 原问题、独立问题和是否需要澄清 |
| `route.selected` | 选择 Document/AIOps/Conversation 等能力及简短依据 |
| `plan.created` | 用户可理解的步骤、并行关系和预计使用的 Skill |
| `skill.started` | Skill 名称、用途、Task 标识 |
| `skill.progress` | 有限、结构化阶段进度 |
| `retrieval.completed` | 候选 Bundle/Document/Evidence 数量和安全标题摘要 |
| `delegation.started/progress/completed` | 子 Agent 和当前阶段 |
| `input.required` | HITL 类型和授权资源 URL |
| `approval.required` | Proposal/风险摘要和授权资源 URL |
| `thinking.delta` | 受限的公开执行说明，不含隐藏推理 Token |
| `answer.delta` | 最终回答的增量文本 |
| `answer.completed` | Answer Artifact、Reference Card 和 Grounding 状态 |
| `turn.completed/failed/cancelled` | Turn 终态 |

每个 Public Trace Event 至少包含：

```json
{
  "schema_version": "AgentTraceEvent.v1",
  "run_id": "UUIDv7",
  "turn_id": "UUIDv7",
  "task_id": "UUIDv7 or null",
  "sequence_no": 12,
  "stage": "retrieval",
  "title": "已完成知识检索",
  "summary": "在 3 个候选 Bundle 中选出 5 组证据",
  "status": "COMPLETED",
  "resource_refs": [],
  "occurred_at": "UTC timestamp"
}
```

`ResponseComposerSkill.execute_stream()` 直接消费模型服务 SSE；Worker
按约 80 字或句末合并，先通过租约命令提交 `answer.delta`，不再在完整回答生成
后伪切片。每个 Chunk 都有 Task attempt 内幂等键；终态
`answer.completed` 始终引用完整不可变 Artifact。断线重连通过
`Last-Event-ID` 只重放尚未消费的 Chunk。

## “思考过程”的安全边界

系统公开的是可验证的执行说明，不是模型隐藏 Chain of Thought。可以展示：

- 为什么选择某类能力的简短、面向用户的理由；
- 计划使用哪些 Skill、输入范围和执行状态；
- 查询改写结果和使用了哪些历史来源；
- 检索到多少候选、选择了哪些文档以及引用定位；
- 子 Agent、HITL、审批、失败、重试和降级状态；
- 最终回答的 Grounding 与实际 Reference。

禁止展示或持久化：

- 模型原始隐藏推理 Token、内部草稿或逐步自由联想；
- System Prompt、内部策略正文、模型凭据和服务拓扑；
- 未授权文档正文、淘汰候选内容、向量和完整 SQL/命令；
- 可能扩大权限的内部 Header、AuthContext JWT 或 SecretRef；
- AIOps 精确命令正文；SSE 只给授权资源 URL。

Planner/Rewrite Model 如需输出解释，必须单独产生受 Schema 限制的
`decision_summary`，长度受限且经过敏感字段过滤。该摘要是“公开决策说明”，
不是事实来源，也不参与后续记忆提取。

## 多 Agent 的记忆边界

Root Agent 是 Conversation 的唯一所有者。Root 生成
`MemoryContextPack.v1`，再按 Capability 裁剪：

- Document Agent 接收 `standalone_query`、解析后的实体和必要偏好；
- AIOps Agent 只接收诊断目标、当前意图和显式选择的上下文；
- Response Composer 接收原始问题、改写结果和已验证领域 Artifact；
- 子 Agent 的内部消息不写入 Root Conversation，只投影公开进度和最终结果。

子 Agent 不凭 `conversation_id` 自行查询全量历史。跨服务只传类型化 DTO 和
来源引用，避免个人偏好、无关业务内容或高敏感 AIOps 数据横向扩散。

## API 和生命周期

公开 API：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/api/v1/conversations` | 创建固定 Agent 的 Conversation |
| `GET` | `/api/v1/conversations` | 分页查询当前用户会话 |
| `GET/PATCH/DELETE` | `/api/v1/conversations/{id}` | 读取、改名/归档、删除 |
| `POST` | `/api/v1/conversations/{id}/turns` | 提交一轮并返回 Run/SSE 地址 |
| `GET` | `/api/v1/conversations/{id}/turns` | 恢复历史消息、引用和 Trace Summary |
| `GET` | `/api/v1/conversations/{id}/turns/{turn_id}/trace` | 分页读取公开执行过程 |
| `GET/DELETE` | `/api/v1/memories`、`/api/v1/memories/{id}` | 查看和忘记长期记忆 |

`/api/v1/runs/{id}` 和 Run SSE 继续作为执行资源存在，但 Portal 的普通聊天从
Conversation Turn 创建 Run。直接 Run API 只用于无会话集成任务，不能自动
读取或写入用户长期记忆。

删除 Conversation 时，如果没有运行中 Turn，级联删除/归档其 Item、Snapshot
和 Turn，并按策略清理 Run 展示关联；`USER_AGENT` 长期记忆不随单个
Conversation 自动删除，因为可能拥有其他来源。删除长期记忆必须单独执行，
删除某个来源后若无有效来源则废止该 Memory Item。

`RETENTION_POLICY` 仅允许 `DEFAULT`、`KEEP_FOREVER`、`DAYS_30`、
`DAYS_90` 和 `DAYS_365`。归档时计算 `PURGE_AFTER`；重新激活时清空。
Retention Worker 领取到期归档会话后，依次删除 Memory Source、Job、
Snapshot、Item、Turn 和 Conversation；失去全部来源的 Memory Item 同步废止。
存在运行中 Turn 时禁止归档或删除，存在正在处理的 Memory Job 时延后一小时。

## 失败、观测和验收

- Snapshot 过期：使用旧 Snapshot 加未覆盖原文，不阻断新 Turn；
- Rewrite 失败：只在当前问题本身可独立理解时降级为原问题，否则要求澄清；
- Memory 召回失败：不影响领域查询，但写入可观测降级事件；
- Answer 流中断：Run 继续，重连重放已持久化 Chunk，终态重新读取 Artifact；
- Trace 投影失败：原始 Run Event 不丢失，可后台重建 Trace Summary；
- 用户撤权：Conversation、Run、Trace 和 Memory 查询统一返回
  `404 *_NOT_FOUND_OR_DENIED`。

Memory Consolidation 是 Run 终态之后的异步派生流程，禁止再向已完成 Run
追加 `memory.updated` Event。Turn History 通过 `memory_status` 返回
`PENDING/PROCESSING/RETRY_WAIT/COMPLETED/FAILED`。如果未来要求在线推送记忆
完成通知，应增加 Conversation Event Stream，不能破坏 Run 终态游标语义。

测试至少覆盖：

- 代词、省略、话题切换、多候选歧义和用户即时更正；
- Snapshot 边界、长会话截断、并发 Turn、幂等重试和重连；
- 记忆新增、确认、时间变化、冲突、拒答和删除；
- Root/Document/AIOps 间的 Memory Pack 最小披露；
- Public Trace 不包含 Prompt、Secret、未授权正文和隐藏推理；
- 历史页投影与在线 SSE 最终显示一致；
- 中文版多 Session 时间推理、知识更新和正确拒答回归集。

## 实施顺序

1. 建立 Conversation/Turn/Item 表、Repository、UoW 和历史查询 API；
2. 将普通聊天入口改为创建 Turn，再由 Turn 创建现有 Root Run；
3. 增加 Public Trace DTO、事件映射、Trace Summary 和可恢复 SSE；
4. 实现 Snapshot Worker、Context Assembler 和 `context-rewrite` Skill；
5. 让 Router、Document Skill 和 Composer 使用改写查询及 Memory Pack；
6. 实现长期 Memory Item/Source、候选整理 Worker 和查看/忘记 API；
7. 完成 APEX 历史渲染、流式执行过程和断线恢复验收；
8. 通过长期记忆、时序更新、隐私和多 Agent 隔离测试后启用长期记忆写入。

## 当前落地状态

Conversation/Turn/Item、Snapshot/Memory/Index Profile/Source/Job 已由
`database/oracle/agent_runtime/005_conversations.sql` 和
`006_memory.sql` 落库；公开与内部 API、DB-first Prompt Resolver、
`context-rewrite` Skill、Document/AIOps/Composer 改写查询传递、异步归并
Worker、敏感候选拦截、历史 Trace 投影和查看/忘记 API 已接通。
同键异值冲突评估、Job 决策结果、精确自然语言忘记、Conversation 物理删除和
归档保留策略 Worker 也已接通。`USER_SHARED` Allowlist、`EPISODIC`、
不可变 Memory Index Profile、读写模式分离的向量生成、混合召回和真实
Response Composer 流式事件均已接通。

`agent_runtime.context_rewrite`、`conversation_snapshot`、
`memory_extract` 和 `response_compose` 均从统一 Prompt Registry 解析，并在
Artifact/Snapshot/Source 中记录版本信息。开发 Oracle 已通过 52 张 Entity
映射、Prompt DB-first 读取以及 Conversation → Turn → Item → Snapshot →
Memory → Source → Job 的回滚式实库写入验收。
`tests/smoke/smoke_agent_memory.py` 进一步在开发 Oracle 上覆盖 Job 领取、Snapshot
和 Memory 写入、决策结果、归档到期以及隐私清理，并在结束时只清理由随机标识
创建的验收数据。

开发 Oracle Smoke 已覆盖 Profile、向量、语义记忆、情景记忆、决策结果、
归档到期和隐私清理。尚未启用的是基于用户反馈的自动衰减，以及 Portal 侧的
最终交互呈现；这些不改变当前数据契约。
