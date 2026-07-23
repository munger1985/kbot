# 4.0 Agent 执行模型

本文定义多 Agent 和 Skill 共用的最小执行内核。它不引入通用工作流引擎，先以数据库持久化状态、任务租约和追加式事件实现可恢复执行。

## 核心对象

| 对象 | 含义 | 持久化重点 |
| --- | --- | --- |
| Run | 一次用户请求及其完整生命周期 | `run_id`、请求者、`domain_id`、`agent_id`、输入、策略/配置快照、状态 |
| Task | Run 中可调度的执行单元 | Skill、Specialist、依赖、输入、租约、尝试次数、输出 Artifact |
| Artifact | 不可变的结构化执行产物 | 类型、schema 版本、内容哈希、来源、存储 URI 或 JSON |
| Delegation | Root Task 与跨服务子 Agent Run 的可恢复关联 | 子 Run、状态、事件游标、租约和结果 Artifact |
| ExecutionContext | 单次 Task 的运行时上下文 | 身份、资源范围、预算、截止时间、模型策略和 Trace；不作为全局单例 |
| Event | Run/Task 的追加式状态变化 | 顺序号、事件类型、操作者、时间、Artifact 引用 |

建议新增五张表：`KBOT_AGENT_RUN`、`KBOT_AGENT_TASK`、`KBOT_AGENT_ARTIFACT`、`KBOT_AGENT_DELEGATION`、`KBOT_AGENT_RUN_EVENT`。Task 的 `depends_on` 第一版保存为结构化 JSON；只有需要复杂 DAG 查询时才拆出依赖表。Task、Artifact、Delegation 和 Event 通过 `run_id` 继承 Run 的 domain 授权，不重复维护跨域权限。

4.0 的 `agent_id`、`run_id`、`task_id` 和 `artifact_id` 均直接使用 UUIDv7 领域主键。Oracle 以 `RAW(16)` 保存 PK/FK，API 使用规范字符串；高频表也不维护数字 PK 与 `*_UID` 双层标识。统一规则见 [31_aiops_step2_persistence_and_identity.md](31_aiops_step2_persistence_and_identity.md)。

## 表字段设计

### `KBOT_AGENT_RUN`

| 字段 | 说明 |
| --- | --- |
| `RUN_ID` | 主键，单次请求的稳定标识 |
| `PARENT_RUN_ID` | 子 Run 或恢复 Run 的父标识，可为空 |
| `DOMAIN_ID` / `AGENT_ID` | 权限和 Agent 配置范围；创建后不可变 |
| `ACTOR_ID` | 用户、服务或系统触发者 |
| `REQUEST_ID` | API 请求链路标识，用于 Trace 关联 |
| `IDEMPOTENCY_KEY` | 调用方幂等键；与 `DOMAIN_ID`、`ACTOR_ID` 组成唯一约束 |
| `ORIGINAL_INPUT` | 用户原始输入，必要时按数据分类加密或脱敏 |
| `STATUS` | Run 状态机当前状态 |
| `ROW_VERSION` | 乐观并发控制版本，每次状态变化递增 |
| `POLICY_SNAPSHOT_JSON` | 创建时的身份、授权和策略快照 |
| `CONFIG_SNAPSHOT_JSON` | Agent、模型和检索配置快照 |
| `BUDGET_JSON` / `DEADLINE_AT` | token、调用次数、并行度预算和截止时间 |
| `FINAL_TASK_ID` | 通过延迟外键冻结本次有效计划的最终 Task |
| `RESULT_ARTIFACT_ID` | 最终回答或错误报告 Artifact |
| `ERROR_CODE` / `ERROR_MESSAGE` | 可解释的终态错误 |
| `CREATED_AT` / `STARTED_AT` / `COMPLETED_AT` | 生命周期时间 |

Run 的 `DOMAIN_ID`、策略快照和幂等键不可修改；重新提交不同请求必须创建新的 Run。
计划安装时写入 `FINAL_TASK_ID`。Validator 要求最终 Task 为 `REQUIRED`，并且
其依赖闭包覆盖所有 `REQUIRED` Task；Runtime 只能使用该 Task 的输出推进
`RUN_COMPLETED`，不能把并行分支中最后提交的任意 Artifact 当作最终结果。

### `KBOT_AGENT_TASK`

| 字段 | 说明 |
| --- | --- |
| `TASK_ID` / `RUN_ID` / `PARENT_TASK_ID` | Task 主键、所属 Run 和父 Task |
| `TASK_KEY` | Run 内稳定的逻辑键，如 `knowledge_retrieval`；与 `RUN_ID` 唯一 |
| `TASK_TYPE` | `ROUTE`、`RETRIEVE`、`DATA_QUERY`、`COMPOSE`、`VERIFY`、`APPROVAL` 等 |
| `SPECIALIST` / `SKILL_ID` / `SKILL_VERSION` | 执行者和版本化 Skill 标识 |
| `STATUS` | Task 状态机当前状态；跨服务子 Run 等待使用 `WAITING_EXTERNAL` |
| `ROW_VERSION` | 乐观并发控制版本，领取、续租和完成时递增 |
| `DEPENDS_ON_JSON` | 前置 Task Key 和完成条件；第一版不拆依赖表 |
| `INPUT_ARTIFACTS_JSON` | 输入 Artifact ID 列表，不复制大段正文 |
| `OUTPUT_ARTIFACT_ID` | 成功时的主输出 Artifact |
| `ATTEMPT` / `MAX_ATTEMPTS` | 当前尝试次数和上限 |
| `LEASE_OWNER` / `LEASE_TOKEN` / `LEASE_UNTIL` | Worker 租约；每次领取生成新 Token 隔离迟到写回 |
| `NEXT_RETRY_AT` | `RETRY_WAIT` 的下一次调度时间 |
| `CANCEL_REQUESTED_AT` | 协作式取消时间 |
| `ERROR_CODE` / `ERROR_MESSAGE` | 最近一次失败信息 |

约束：`(RUN_ID, TASK_KEY)` 唯一；只有 `READY` Task 可以被领取；终态 Task 不允许回写状态或覆盖输出。

### `KBOT_AGENT_DELEGATION`

Delegation 是异步子 Agent 的持久化关联，不用于 Runtime 内普通 Specialist Task。字段包括 `DELEGATION_ID`、Parent Run/Task、Target Service/Capability、Child Run ID、幂等键、状态、Child Event Cursor、轮询时间、结果 Artifact、有限租约、尝试次数、错误、Row Version 和审计时间。

`PARENT_TASK_ID`、`(TARGET_SERVICE, IDEMPOTENCY_KEY)` 和非空 `(TARGET_SERVICE, CHILD_RUN_ID)` 分别唯一。Parent 引用建立内部外键；Child Run 是跨服务 UUIDv7，不建数据库外键。AIOps 首个完整用例见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

### `KBOT_AGENT_ARTIFACT`

| 字段 | 说明 |
| --- | --- |
| `ARTIFACT_ID` / `RUN_ID` / `TASK_ID` | 产物主键、所属 Run 和生产 Task |
| `ARTIFACT_TYPE` | `CITATION_PACK`、`QUERY_RESULT`、`GROUNDED_ANSWER` 等 |
| `SCHEMA_VERSION` | payload 的契约版本，不随代码版本隐式变化 |
| `PRODUCER` / `PRODUCER_VERSION` | Agent、Skill 或服务版本 |
| `PAYLOAD_JSON` | 小型结构化结果；大结果为空 |
| `STORAGE_URI` | 大型结果、文件或脱敏快照的对象存储地址 |
| `CONTENT_HASH` | 内容完整性校验和去重辅助 |
| `PROVENANCE_JSON` | 输入 Artifact、外部调用、模型和 Evidence 来源 |
| `SECURITY_LEVEL` / `EXPIRES_AT` | 数据分级和生命周期 |
| `CREATED_AT` | 创建时间；Artifact 不可变 |

`PAYLOAD_JSON` 与 `STORAGE_URI` 至少有一个非空。删除、脱敏或过期采用生命周期操作，不更新原始内容；引用方通过新 Artifact 表达修订。

### `KBOT_AGENT_RUN_EVENT`

| 字段 | 说明 |
| --- | --- |
| `RUN_ID` / `SEQUENCE_NO` | 复合主键；Run 内单调递增序号，供 SSE 续传 |
| `TASK_ID` | 事件所属的可选 Task |
| `EVENT_TYPE` | 状态、进度、Artifact、审批和错误事件 |
| `EVENT_KEY` | 可选幂等键；跨服务事件投影时必填 |
| `ARTIFACT_ID` | 事件关联的产物，可为空 |
| `EVENT_PAYLOAD_JSON` | 小型事件详情，不保存完整正文或密钥 |
| `ACTOR_TYPE` / `ACTOR_ID` | Worker、Agent、用户或系统操作者 |
| `TRACE_ID` / `CREATED_AT` | 分布式追踪和发生时间 |

事件只追加、不更新、不删除（超过保留期后按审计策略归档）。SSE 使用 `(RUN_ID, SEQUENCE_NO)` 查询，不依赖数据库轮询 Task 状态；非空 `(RUN_ID, EVENT_KEY)` 唯一，避免外部事件重放产生重复父事件。

推荐索引为：Run 的 `(DOMAIN_ID, CREATED_AT)` 和 `(STATUS, CREATED_AT)`，Task 的 `(STATUS, NEXT_RETRY_AT)`、`(LEASE_UNTIL, STATUS)`，Artifact 的 `(RUN_ID, ARTIFACT_TYPE)`，Event 的唯一键 `(RUN_ID, SEQUENCE_NO)`。

## 状态机

Run 状态：

```text
CREATED → RUNNING → COMPLETED
                 ├→ WAITING_INPUT → RUNNING
                 ├→ WAITING_APPROVAL → RUNNING
                 ├→ FAILED
                 ├→ CANCELLED
                 └→ EXPIRED
```

Task 状态：

```text
PENDING → READY → RUNNING → SUCCEEDED
                       ├→ WAITING_EXTERNAL → SUCCEEDED / FAILED
                       ├→ RETRY_WAIT → READY
                       ├→ FAILED
                       ├→ BLOCKED
                       └→ CANCELLED
```

状态迁移必须由 Runtime 的状态服务完成，Repository 不允许任意改写终态。`SUCCEEDED` 的 Task 只能产生新的 Artifact，不能覆盖旧产物。

## 调度和事务边界

API 创建 Run、Root Task 和首个事件时使用同一事务，成功后返回 `run_id`。Scheduler 只领取 `READY` Task，使用有限 `lease_until` 防止多副本重复执行。Worker 完成 Task 时，在一个数据库事务内写入结果 Artifact、Task 状态和事件，再释放后继 Task；事务中不调用 HTTP 或 LLM。

Task 之间只通过版本化 Artifact 传递数据，不共享可变 `dict` 或进程级 Context。Supervisor 负责生成和校验 DAG；Specialist 只能执行分配给自己的 Task。跨服务委派接受后进入 `WAITING_EXTERNAL` 并释放 Worker 租约，由 Delegation Reconciler 按持久化游标恢复。问文和问数可以并行，Answer Composer 必须等待对应 Artifact 完成：

```text
Root
 ├─ Knowledge Retrieval → CITATION_PACK
 ├─ MCP Data Adapter    → QUERY_RESULT（现有问数链路）
 └─ Answer Compose      → GROUNDED_ANSWER
```

`CITATION_PACK` 只引用 KC Evidence，`QUERY_RESULT` 独立保存查询结果；混合回答不得把两者合并成旧式 `doc_results`。

## Artifact 和 Skill 契约

Artifact 至少包含 `artifact_type`、`schema_version`、`producer`、`payload` 或 `storage_uri`、`content_hash`、`provenance` 和 `security_level`。大结果写对象存储，数据库保留稳定 URI、摘要和哈希。

Skill Manifest 声明输入/输出 schema、权限、运行模式、幂等性、超时、重试上限和数据分类。Skill 接收已验证 DTO 与 ExecutionContext，返回 Typed Artifact 或标准进度事件；禁止直接实例化 Agent、访问跨域 Repository 或自行创建数据库 Session。

## 事件、SSE 和恢复

`KBOT_AGENT_RUN_EVENT` 是唯一的执行事件源，事件至少包含 `run_id`、`task_id`、事件序号、类型、时间和 Artifact 引用。事件类型包括 `RUN_STARTED`、`TASK_STARTED`、`TASK_PROGRESS`、`ARTIFACT_CREATED`、`TASK_RETRYING`、`APPROVAL_REQUIRED`、`TASK_COMPLETED`、`RUN_FAILED` 和 `RUN_COMPLETED`。

SSE 只读事件流：`GET /api/v1/runs/{run_id}/events`，通过 `Last-Event-ID` 续传。子 Agent 事件先幂等投影为父 Event，不做 SSE 套 SSE。取消使用协作式取消；租约过期后允许其他 Worker 接管。重试仅适用于幂等 Task，每个外部副作用必须使用 `run_id + task_id` 的幂等键。恢复时复用已成功的 Artifact，不重复执行已完成 Task。

Mutation Skill 先产生 `ACTION_PLAN` Artifact，确定性 Catalog/Policy Builder 再创建不可变 `APPROVAL_PROPOSAL`；经过 Policy/HITL 后才能创建一次性 Execution Task。预算、截止时间、最大并行数和最大重试次数在 Run 创建时冻结，并由 Runtime 强制执行，而不是交给 Prompt 判断。

## 第一版非目标

- 不实现跨数据库事务或通用 BPMN/工作流引擎；
- 不兼容 3.x SkillRuntime、动态反射和全局 ContextMemory；
- 不让 SSE 成为状态存储；
- 不把 Agent 编排逻辑放入 Knowledge Core，KC 只提供领域 API 和 Evidence 契约。
