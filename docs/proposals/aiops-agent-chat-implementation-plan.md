# AIOps Agent 专业 DBA 对话诊断改造方案与实施计划

## 当前实施状态（2026-08-27）

阶段0和阶段1已经完成：Schema 13、自包含重建制品、共享契约、Turn 原子接收、并发序号、
失败回滚、幂等重试和旧 Conversation Run/Action Step 路径均已收口。阶段2已经建立可靠的
`QUEUED → ACCEPTED → PLANNING`调度链、唯一 Primary Run、取消传播和可重放终态事件，
并通过无外部依赖的空流程验收。

当前分支仍是不可部署的开发中版本，也没有操作任何共享数据库。阶段2还需完成既有
Run/Task 运行时向通用状态机的迁移；Intent Router、DBA Skill 和真实流式回答尚未接入。

版本：1.0

状态：实施中

基准日期：2026-08-27

## 1. 目的与基准

本文以以下两份已经批准的设计为唯一产品和技术基准：

- [AIOps Agent 专业 DBA 对话诊断设计](../product/aiops-agent-chat-diagnosis.md)；
- [AIOps Agent 专业 DBA 对话诊断详细设计](../architecture/aiops-agent-chat-diagnosis.md)。

本文回答三个问题：现有实现与目标设计差在哪里、哪些能力可以复用、按什么顺序改造才能
一次切换到新模型。本文不是兼容方案。KBot 4.0 不保留旧 Conversation Run、固定根因报告
或旧聊天接口的双读双写路径。

本次范围是人工发起的专业 DBA 对话诊断，以及它与告警、巡检、补证、变更审批的衔接。
报告模板的产品去留、告警诊断本身的算法和巡检报告格式不在本次重构范围内；它们只需能够
把已有 Run 作为上下文进入同一 Conversation/Turn 模型。

## 2. 总体判断

现有实现不是全部推倒重来。它已经具备可靠的 Run/Task 执行内核、Oracle 诊断 Tool
白名单、Artifact、HITL、变更审批、Outbox 和 SSE 基础设施。主要问题是这些能力被一个
“根因分析报告”模型直接暴露给聊天入口，导致：

1. 用户的一般 DBA 问题也被强制套入假设、根因等级和建议的固定格式；
2. Conversation 没有 Turn，一条用户消息、一次执行和一次回答之间缺少稳定聚合根；
3. LLM 在低层 Tool 范围内选择动作，缺少可审计的专业 DBA Skill 语义层；
4. 回答只在 Run 完成后分片发送 Markdown，不是真正的生成式流式输出；
5. 前端扫描任意 Evidence Fact 并猜测表格或图表，表现契约不稳定；
6. 补证、图片、变更提案和运行状态分别挂在 Conversation 或 Run 上，无法准确说明它们
   属于哪一轮问题；
7. 消息和运行序号采用`MAX + 1`，并发排队时存在重复序号风险。

目标结构是：

```text
Conversation
  -> Turn
       -> User Message
       -> Intent Plan
       -> Skill Plan
       -> Run / Task
       -> Skill Invocation
       -> Artifact -> Turn Evidence
       -> Sufficiency Decision
       -> Answer Block -> Citation
       -> Assistant Message
       -> Change Proposal / Evidence Request
       -> Turn Event
```

Run/Task 继续负责可靠执行，Turn 负责用户语义、证据边界、回答和流式事件。Tool 继续负责
安全执行 SQL 或读取监控源，Skill 负责表达 DBA 能力、测量口径和展示方式。

## 3. 现有代码与目标设计差距

### 3.1 Oracle Schema 与 Manifest

现状：

- `008_ops_conversations_reports.sql`只有 Conversation、Message、Conversation Run、
  Evidence Request、Action Step 和 Image Evidence；
- `002_ops_runtime.sql`中的 Run/Task 是根因分析专用状态和任务类型；
- `003_ops_change.sql`的 Change Proposal 已拥有命令顺序、版本、Hash 和审批状态，
  Action Step 与其重复；
- 当前 Manifest 为 Schema 12、`aiops-oracle-v2`、35张表、10个视图。

修改：

- 按详细设计重建为 Schema 13、`aiops-oracle-v3`、40张表、10个视图；
- 新增 Conversation Turn、Skill Invocation、Turn Evidence、Answer Block、Answer Citation、
  Turn Event；
- 将 Conversation Run 重建为 Turn Run，删除 Action Step；
- Message、Evidence Request、Image Evidence 和 Change Proposal 关联 Turn；
- Conversation 增加`LAST_TURN_NO`与`LAST_MESSAGE_NO`，以行锁分配序号；
- Run/Task 使用通用执行状态和任务类型，根因等级只作为诊断类 Artifact 内容；
- 增加 Agent Version、Target、Conversation、Turn 的所有者复合唯一键和复合外键，
  由 Oracle 阻止跨 Domain 拼接；
- 更新 DDL 渲染器、实体清单、Schema Manifest、重建脚本和 Oracle 验收检查。

切换方式：代码完成后停止相关服务，使用规范 DDL 生成一个可直接在 SQL Developer F5
运行的自包含重建文件。保留共享用户、权限和 KC 对象，不引入运行时迁移、兼容视图、
双写或数据回填。

### 3.2 Entity、Repository 与 Unit of Work

现状：

- `entities/conversation.py`直接映射旧模型；
- `repositories/conversation.py`用`MAX(sequence_no) + 1`分配消息和 Run 序号；
- Conversation 详情一次加载大量 Message、Run 和 Action Step；
- `persistence/uow.py`的事务归属正确，Repository 不提交事务。

修改：

- 新建 Turn、Skill Invocation、Turn Evidence、Answer Block、Answer Citation、Turn Event
  实体，删除 Conversation Run 和 Action Step 实体；
- 将 Conversation Repository 限定为聚合根、行锁和消息序号操作；增加 Turn Repository
  负责 Turn 分页、状态迁移、租约、事件游标和关联对象查询；
- 新增`lock_conversation_and_allocate_numbers()`，同一 UoW 内创建 Turn、User Message、
  初始事件和 Outbox；
- 使用显式分页查询 Turn 和 Message，不再返回完整 Conversation 历史；
- 为状态迁移、唯一 Primary Run、事件游标及取消请求提供条件更新，保持幂等；
- 保留 UoW 单点提交和 Repository 只`flush()`的原则。

### 3.3 Conversation 应用服务

现状：

- `application/conversations.py`同时承担会话、运行、补证、图片解析和展示投影；
- 新消息先写 Message，再由 API 在另一个事务创建并挂接 Run；
- Agent Version 必须直接绑定 Target，不支持从所选监控源解析 Target；
- 通过伪造`AGENT_PROGRESS`消息表达进度；
- 图片 Base64 写入 Artifact JSON，且 API 请求内同步解析；
- Artifact 信任等级命名与 DDL 约束存在不一致。

修改：

- 拆分为 Conversation Service、Turn Command Service、Turn Query Service、
  Evidence Response Service 和 Image Evidence Service；
- 创建 Turn 的事务一次完成用户消息、Turn、事件和 Outbox，不在 API 请求内创建 Run；
- Target 解析规则固定为：显式选择优先；Agent 直连 Target 次之；监控源绑定得到唯一 Target
  时自动选择；多个候选返回明确的选择要求；无候选返回稳定错误码；
- Agent 未配置数据库直连时仍可执行监控型 Skill，只是不具备 DB Tool 能力；
- 进度只写 Turn Event，不再写伪消息；
- 图片先保存为受控对象引用，解析由异步任务完成，结果成为 Artifact 和 Turn Evidence；
- 统一 Artifact 信任等级枚举，禁止代码和 DDL 使用不同值。

### 3.4 Runtime、Worker 与状态机

现状：

- `application/runtime/service.py`包含大量聊天专用逻辑，创建
  `diagnosis.root-cause` Blueprint 并冻结六个 Oracle Tool；
- `orchestration/blueprints.py`把多轮根因分析固化为大型 DAG；
- `domain/states.py`、`domain/operations/run.py`和转换规则使用 SCOPING、OBSERVING、
  DIAGNOSING 等业务阶段；
- Run 完成后才把最终 Markdown 切片为`answer.delta`；
- Worker 的租约、重试、Outbox、Reconciler 已有可复用基础。

修改：

- 保留 Run/Task 的可靠执行、租约、重试、取消、Outbox 和 Reconciler 内核；
- 将 Run 状态改为通用生命周期，将 Task 类型改为 PLAN、INVOKE_SKILL、ASSESS、
  COMPOSE_ANSWER、PROCESS_EVIDENCE、EXECUTE_CHANGE、VERIFY；
- 新增 Turn Planner Worker、Skill Invocation Worker、Sufficiency Worker 和 Answer Worker；
- Chat Turn 的 Skill Plan 编译为通用 Task，不再进入固定根因 Blueprint；
- 告警和巡检仍可使用确定性 Blueprint，但其结果通过 Source Run 关联进 Turn；
- Turn 取消传播到未开始 Task 和正在执行的可取消 Tool，终态保持幂等；
- Turn Event 是前端唯一进度流，Run Event 保留为内部运维审计。

### 3.5 Intent、Skill 与 Tool

现状：

- `contracts/diagnosis/models.py`假定所有问题都要形成假设、根因和报告；
- `workers/diagnosis_handlers.py`让 LLM 在 Tool ID 级别决策，并包含按“表空间”等关键词
  特判的直接回答；
- `diagnostics/`已经有版本、SQL Hash、权限、输入输出 Schema、超时、行数和成本约束，
  是可复用的安全 Tool 层；
- Oracle 目录目前只有实例、活跃会话、阻塞、容量、长事务和复制六类 Tool。

修改：

- 新增代码内版本化 DBA Skill Registry，Skill Manifest 声明意图、领域、Capability、
  Tool DAG、测量语义、Evidence 新鲜度、输出 Schema、展示类型和回退方式；
- Intent Router 输出七个一级意图、领域、对象、时间窗口、排序、数量和期望展示；
- Planner 只能从冻结的 Skill Registry 选择 Skill，不能让 LLM 拼接 SQL 或 Tool ID；
- 复用现有 Tool Registry 和 Executor Allowlist，逐步补齐 Tool，不建立可在线修改 SQL 的
  Skill 数据库表；
- 首批实现实例概览、等待与资源、当前 Top SQL、SQL 详情、执行计划、活跃会话、阻塞、
  长事务、表空间、TEMP/UNDO、Alert Log 和 Data Guard Skill；
- `oracle.sql.top_current`明确为`CUMULATIVE_SINCE_LOAD`；只有存在可靠快照差值、监控时序
  或授权历史源时才声明`oracle.sql.top_window`。不能把累计值伪装成最近15分钟；
- 原有假设、归因和 Grounding 逻辑只供 DIAGNOSE Skill 使用，不再作为所有回答的模板；
- 删除按问题关键词或单个业务字段硬编码回答的路径。

### 3.6 Evidence、充分性与回答生成

现状：

- Evidence 主要按 Run 索引，前端遍历全部 Fact；
- Evidence Request 挂在 Conversation 上并保存裸 SQL；
- 回答正文由固定诊断报告投影生成，Evidence、根因等级和建议混在正文；
- 没有持久化 Answer Block 和关系化 Citation。

修改：

- 每份被采用的 Artifact 都通过 Turn Evidence 显式关联当前 Turn，记录用途、新鲜度和
  支持或反驳关系；历史 Run Evidence 必须重新关联，不能隐式继承全部上下文；
- Sufficiency Gate 输出 ANSWERABLE、PARTIAL、NEEDS_CLARIFICATION、NEEDS_EVIDENCE、
  CAPABILITY_UNAVAILABLE 或 UNSAFE；能力不足要给出可执行说明，不得统一成为“无进展”；
- Evidence Request 由 Planner/Sufficiency Gate 创建，关联 Turn 和 Skill，不允许用户从
  公共 API 任意创建包含 SQL 的请求；用户只提交文字、截图或文件结果；
- 先用结构化、非流式`AIOPS_ANSWER_PLAN.v1`确定可引用 Evidence、Block 类型和叙述提纲，
  经校验后再生成回答；
- 表格和图表由 Skill 输出 Schema 和确定性 Renderer 生成，LLM 只生成叙述块；
- Answer Citation 必须引用当前 Turn Evidence，写库时校验 Block、Evidence 和 Turn 一致；
- Answer Worker 使用 Model Serving 流接口实时生成叙述，按短时间窗口合并 SSE 事件，
  不能每个 Token 写一次 Oracle；最终块、引用、Assistant Message 和终态在事务中收口。

### 3.7 API、共享契约与客户端

现状：

- AIOps Agent 的 Conversation API 使用路由文件内本地 Pydantic DTO；
- Main API 也定义另一套 DTO，并通过泛化`conversation_request()`转发；
- 创建接口把`source_run_id`和`request_report`混在消息载荷中；
- 补证接口允许调用方创建 Evidence Request；
- 前端追踪 Run SSE，而不是 Turn SSE。

修改：

- 在`platform_core.contracts.aiops`新增版本化 Conversation/Turn、Answer Block、Citation、
  Evidence Response 和 Event DTO，Main API 与 AIOps Agent 共用同一契约；
- `POST /conversations`只创建会话；`POST /conversations/{id}/turns`提交新问题并返回
  Turn Receipt；创建会话时可选择原子提交首个 Turn；
- 增加 Turn 列表/详情、Turn Event SSE、取消、补证响应和图片上传接口；
- Source Run 作为受校验的 Turn 上下文关联，不作为普通聊天文本字段；
- 删除公共的“创建 Evidence Request”接口；
- 将 Platform Client 泛化转发替换为类型化方法，统一错误码、幂等键和超时；
- 更新 Main API 的 API Key 路由权限、OpenAPI 快照和契约测试；
- SSE 支持`Last-Event-ID`恢复，事件使用单调`EVENT_CURSOR`，终态后可重放最终块。

### 3.8 前端工作区

现状：

- `ui/aiops/js/aiops-workspaces.js`以 Conversation 和 Run 为中心；
- 通过遍历任意 Fact 和字段名猜测表空间图表；
- 只理解固定诊断报告、根因和 Gap；
- 有未完成补证时，下一条用户输入会被自动解释成补证结果，语义含混。

修改：

- 拆分 Conversation Store、Turn Stream、Block Renderer、Evidence Drawer 和 Proposal Panel；
- 主时间线按 Turn 展示，用户问题立即出现，状态由 Turn Event 更新；
- 叙述使用 Markdown 增量渲染；Table、Chart、Code、Proposal Summary、Evidence Request
  由稳定 Block 类型渲染；
- Evidence 默认折叠为引用入口，展示来源、采集时间、测量口径、新鲜度和必要的原始片段，
  不在正文铺开全部指标；
- 删除`tablespaceChartHtml`和对任意 Fact 的字段猜测；
- 多个 Target 候选时在发送前明确选择；唯一候选自动解析并展示当前上下文；
- 补证进入显式响应模式，标明正在回答哪个请求，并始终允许切回“提出新问题”；
- 变更提案根据 Agent 权限展示审批/执行入口，聊天模型不能绕过审批链；
- 保留现有聊天页面整体布局，分阶段替换状态和 Renderer，不额外建立平行页面。

## 4. 复用、重构与删除清单

### 4.1 直接复用

- Unit of Work 的事务所有权和 Repository 不提交约束；
- Run/Task 租约、重试、取消基础、Outbox Dispatcher 和 Reconciler；
- Artifact 不可变存储、运行事件审计和服务鉴权；
- Tool Catalog、SQL Hash、权限声明、参数/输出 Schema 与 Executor Allowlist；
- Change Proposal、审批、执行和验证主体模型；
- Model Serving 结构化生成能力；
- Main API 已有 SSE 代理与鉴权基础。

### 4.2 重构后复用

- 根因假设、Evidence Fact 和 Grounding：仅用于 DIAGNOSE；
- Blueprint：仅用于告警、巡检等确定性流程，Chat 使用 Skill Plan；
- Run Event：保留内部审计，前端改用 Turn Event；
- 现有聊天 HTML/CSS 外壳：保留视觉框架，替换状态管理和内容 Renderer。

### 4.3 删除

- `KBOT_OPS_CONVERSATION_RUN`与对应实体、Repository、接口投影；
- `KBOT_OPS_ACTION_STEP`及重复的提案展示逻辑；
- `WAITING_EVIDENCE` Conversation 状态和伪`AGENT_PROGRESS`消息；
- Chat 入口硬编码`diagnosis.root-cause`和固定六 Tool 基线；
- 按关键词、表空间字段或任意 Fact 猜答案和图表的逻辑；
- Run 完成后伪流式切分 Markdown 的路径；
- 用户创建 Evidence Request 的公共接口；
- 新旧聊天契约的兼容适配、双读、双写和旧路由。

## 5. 实施阶段

代码可按下列阶段形成独立、可审查提交，但在新 Schema、服务和前端全部就绪前不部署到
共享环境。每阶段必须更新测试，不能在主分支长期保留两个可运行路径。

### 阶段0：契约冻结和 Schema 生成

交付物：共享 DTO、状态枚举、错误码、SSE Event Schema；Schema 13 DDL、实体 Manifest、
自包含重建脚本生成器；序号、复合外键、唯一 Primary Run 和 Event 游标的 Oracle 约束测试。

退出条件：Schema Manifest 精确为40张表、10个视图；重建文件无`@`或`@@`依赖；契约测试
通过；此阶段不操作共享数据库。

### 阶段1：Turn 持久化与原子接收

交付物：新实体、Repository 和 UoW 注册；Conversation 行锁分配 Turn/Message 序号；原子
创建首 Turn、后续排队 Turn、幂等返回、分页查询和取消请求；删除旧 Conversation Run 和
Action Step 代码。

退出条件：并发测试中序号无重复；失败回滚不留下 Message、Event 或半个 Turn；同一会话
最多一个执行中 Turn，允许多个 QUEUED Turn。

### 阶段2：Turn 调度、事件和端到端空流程

交付物：Planner Worker 领取队列并创建唯一 Primary Run；通用 Run/Task 状态机；Turn Event
写入、SSE 重连、终态重放和取消传播；类型化内部 API、Platform Client、Main API 与前端
Turn Store 骨架。

退出条件：一个不调用外部依赖的测试 Turn 能从 QUEUED 到 COMPLETED；浏览器断开重连后
不会重复或丢失事件；公共请求不能直接修改 Run 状态。

### 阶段3：Intent Router 与 DBA Skill 框架

交付物：`DBA_INTENT_PLAN.v1`、Skill Manifest、Registry、Hash、Validator 和 Planner；
Agent/Target/Source Capability Snapshot；Skill Plan 到 Task 的编译器；目录外 Skill、无权限
Tool 和动态 SQL 的拒绝路径。

退出条件：覆盖七个一级意图的测试集通过；同一输入计划可重放；任何 Tool 调用均能追溯到
冻结的 Skill 版本和 Manifest Hash。

### 阶段4：Oracle 首批核心 Skill 与 Evidence

优先顺序：

1. 实例概览、表空间、活跃会话和阻塞链；
2. 当前 Top SQL、SQL 详情、执行计划、等待与资源；
3. TEMP/UNDO、长事务、Alert Log 和 Data Guard；
4. 有可靠历史源后再启用 Top SQL 时间窗口能力。

交付物包括缺失 Tool、最小权限声明、输出 Schema、测量语义、Renderer、Turn Evidence 和
真实 Oracle Smoke。第一批生产验证可以只承诺`top_current`；对“最近15分钟”返回 PARTIAL
并明确实际口径，不阻塞总体切换。

退出条件：每个 Skill 在支持、不支持、权限不足、Target 不可达、Source 不可达场景都产生
稳定状态和可行动说明；Evidence 不跨 Turn 泄漏。

### 阶段5：充分性、真实流式回答与结构化展示

交付物：Sufficiency Gate 与`AIOPS_ANSWER_PLAN.v1`；Model Serving 流式客户端和 Answer
Worker；Narrative/Table/Chart/Code/Notice 等 Block Renderer；Answer Citation 关系校验、
Evidence 抽屉和 Markdown 增量渲染。

退出条件：首个回答增量在完整回答结束前到达；数据库写入频率受合并窗口控制；刷新页面后
可从持久化 Block 重建相同答案；图表不依赖字段名猜测；正文不再强制显示根因等级。

### 阶段6：补证与图片异步处理

交付物：Turn 级 Evidence Request 和显式 Response；文字、截图、文件的受控存储、异步提取、
脱敏和 Artifact 化；WAITING_USER 到恢复规划/评估的幂等状态迁移；前端补证响应模式与普通
新问题的明确分流。

退出条件：大文件不阻塞 API；重复提交不会产生重复 Artifact；提取失败可重试且不会关闭
请求；建议 SQL 只来自受控模板或 Skill，不接受模型生成的任意执行 SQL。

### 阶段7：变更提案、审批和验证关联

交付物：Proposal、审批、执行、验证与 Turn 的外键和事件投影；Proposal Summary Block；
Agent 是否允许变更、审批状态和执行凭据的完整授权校验；执行结果作为新 Evidence 回到同一
Turn，必要时开启后续 Turn。

退出条件：只读 Agent 永远不能进入执行；允许变更也必须经过既有审批链；重放请求不重复
执行命令；审计可从用户问题追踪到命令、审批人和验证结果。

### 阶段8：前端收口与旧路径清除

交付物：完整 Turn 时间线、Target 上下文、Evidence Drawer、Proposal Panel；删除旧 Run
结果循环、固定根因报告、Fact 猜图和旧补证交互；删除后端旧契约、旧状态、旧 Blueprint
Chat 入口和无引用代码；更新用户文档、OpenAPI、静态资源清单和页面契约测试。

退出条件：代码搜索不存在旧表名、旧聊天消息类型和 Chat 硬编码根因 Blueprint；三个入口
都能进入同一会话体验，但保留各自来源上下文。

### 阶段9：开发环境切换与生产验证

执行顺序：

1. 备份审计所需数据并确认本次允许清空 AIOps 业务表；
2. 停止 Main API、AIOps Agent 及相关 Worker；
3. 在目标 Oracle PDB 使用 SQL Developer F5 执行自包含重建文件；
4. 运行 Schema、实体和权限验收；
5. 发布同一版本的 AIOps Agent、Main API 和静态前端；
6. 启动服务并检查进程、端口、Readiness 和最新在线日志；
7. 执行 Oracle 生产验证场景和失败注入；
8. 验收通过后再推广到客户生产环境。

这是一次原子版本切换。出现失败时回退应用版本并恢复数据库备份，不保留旧 Schema 兼容
层。任何共享环境的停止、重建、恢复操作都必须单独确认后执行。

## 6. 测试与验收矩阵

### 6.1 单元测试

- Conversation 行锁与序号分配、Turn 状态迁移和幂等；
- Intent Schema、Skill Registry、Capability 匹配和计划校验；
- Turn Evidence 新鲜度、充分性决策、Block/Citation 校验；
- 通用 Run/Task 转换、租约、取消和重试；
- 各 Oracle Skill 的输入、测量语义、脱敏和 Renderer。

### 6.2 契约测试

- AIOps Agent、Platform Client、Main API 共用 DTO；
- API Key 权限、Domain 隔离、内部 AuthContext Audience；
- Turn Receipt、分页、取消、补证和 SSE 重连；
- OpenAPI 快照与前端 Block/Event 类型一致；
- 静态页面不再引用已删除模块或扫描任意 Evidence Fact。

### 6.3 集成与 Smoke

- Oracle Schema 重建、复合外键、函数唯一索引和 JSON 约束；
- 真实 Oracle 的实例、表空间、会话、阻塞和 Top SQL；
- Prometheus/Loki 正常、空数据、超时、鉴权失败和部分可用；
- Model Serving 结构化规划、流式中断、重连和输出校验失败；
- 图片异步提取、审批执行和验证回写。

### 6.4 产品验收场景

- “当前数据库是否健康”返回概览和可折叠 Evidence；
- “分析现在数据库上的 Top SQL”使用当前累计口径并以表格回答；
- “最近15分钟 Top SQL”在无历史能力时明确 PARTIAL，不伪造窗口数据；
- “为什么数据库变慢”按 DIAGNOSE 形成假设并继续取证，但不强制固定报告格式；
- 用户补交截图或文字后恢复同一 Turn；
- 告警或巡检结果进入会话后，可继续追问且只引用已关联 Evidence；
- 有变更权的 Agent 产生提案并走审批，无变更权的 Agent 只给人工建议；
- 多 Target Agent 要求用户明确选择，唯一 Target 自动选择。

## 7. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| Schema 和应用同时变化导致环境不可用 | 先完成全部代码和自动化验证，再执行停机原子切换 |
| LLM 产生目录外动作或 SQL | Intent/Skill/Answer 三层结构校验，Tool Allowlist 最终兜底 |
| 流式输出产生大量数据库写 | 内存合并短窗口，SSE 增量与持久化 Block 分层 |
| 证据跨轮污染 | 所有回答引用必须经过 Turn Evidence 关系校验 |
| Oracle 指标口径误导 | Skill Manifest 固化测量语义，窗口能力按 Source Capability 开关 |
| 图片包含敏感信息 | 受控对象存储、大小/类型限制、脱敏、短期保留和审计 |
| 大范围重构难以定位回归 | 分阶段提交、每阶段退出门、最终一次部署，不引入兼容分支 |
| 旧固定报告逻辑残留 | 删除清单、代码搜索门和反向契约测试 |

## 8. 建议的代码改动边界

主要修改目录：

- `database/oracle/aiops_agent/`：Schema、视图、Manifest 和重建生成；
- `services/aiops_agent/src/aiops_agent/entities/`：Turn、Evidence、Answer、通用 Runtime；
- `services/aiops_agent/src/aiops_agent/repositories/`与`persistence/`：聚合和事务；
- `services/aiops_agent/src/aiops_agent/application/`：Conversation/Turn 用例；
- `services/aiops_agent/src/aiops_agent/orchestration/`与`workers/`：计划与执行；
- `services/aiops_agent/src/aiops_agent/skills/`：新增 DBA Skill 层；
- `services/aiops_agent/src/aiops_agent/diagnostics/`：扩展低层 Oracle Tool；
- `packages/platform_core/src/platform_core/contracts/aiops/`：共享契约；
- `packages/platform_clients/src/platform_clients/aiops.py`：类型化内部客户端；
- `services/main_api/src/main_api/api/`：公共 API、SSE 和授权；
- `ui/aiops/`：Turn 状态管理和结构化 Block 展示；
- `tests/unit/aiops_agent/`、`tests/contract/`、`tests/integration/`、
  `tests/smoke/`和`tests/acceptance/`：对应验证。

不应修改或建立：

- `integrations/apex/**`；
- 3.x 兼容包、`legacy/`目录或 V1/V2 双路实现；
- 允许在线编辑诊断 SQL 的数据库表；
- 为了模拟时间窗口而未经设计引入的高基数 SQL 快照表。

## 9. 完成定义

只有同时满足以下条件，改造才算完成：

1. Schema 13、实体、Manifest、共享契约和代码一致；
2. 每条用户消息形成可审计 Turn，消息和事件在并发下不重号；
3. Chat 不再硬编码根因 Blueprint，所有外部取证经版本化 Skill 和 Tool Allowlist；
4. 回答真实流式输出，刷新后可由 Answer Block 和 Citation 完整恢复；
5. Evidence 默认折叠且只引用当前 Turn 显式关联内容；
6. 表格和图表由确定性 Renderer 生成，不由前端猜测指标字段；
7. 补证、图片、变更和验证都有清晰 Turn 归属及审计链；
8. Oracle 核心场景、失败场景、权限边界和 Domain 隔离通过测试；
9. 旧表、旧接口、旧状态和旧聊天表现路径已经删除；
10. 开发环境完成自包含 Schema 重建、服务启动和真实 Oracle 生产验证。
