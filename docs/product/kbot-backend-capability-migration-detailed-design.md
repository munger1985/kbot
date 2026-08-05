# KBot 后端能力迁移详细实施方案

## 1. 总体设计

本方案描述 Ammolite 后端能力进入 KBot 后的目标实现。迁移对象是业务能力和代码组织方式，
不是产品标识、Tenant/App/RBAC/Portal 或 PostgreSQL 基础设施。

目标调用关系：

```text
外部调用方
  → Main API（API Key、Domain、actor_id、组合编排）
      → Agent Runtime（Conversation/Run/Task/Artifact/Memory/Skill）
          → Knowledge Core（知识检索与预览）
          → Data Query（SEMANTIC 问数）
          → MCP Data Client（MCP 问数）
          → Model Serving（LLM/Embedding/VLM/Visual）
      → Notification Projection（Inbox/Work Item/Operation/SSE）
      → Development Logs（跨服务只读日志）
```

所有服务使用 Oracle、`domain_id NUMBER(38)` 和应用生成的 UUIDv7 `RAW(16)` 资源 ID。
内部接口使用 `/internal/v1` 与 audience-bound AuthContext；Main API 对外接口使用
`/api/v1`。

## 2. 代码组织与安装设计

### 2.1 包依赖方向

```text
platform_core
  ← platform_clients
  ← model_serving / knowledge_core / data_query / agent_runtime / aiops_agent
  ← main_api
```

- `platform_core`：配置、安全、Oracle 类型、UoW 原语、稳定 contracts、通知 Outbox 原语；
- `platform_clients`：只包含内部 HTTP Client 和错误映射；
- 服务之间不互相 import 源码；
- Main API 只通过 contracts/client 编排服务；
- AIOps 包参与安装，但不参与本迁移的业务变更。

### 2.2 包名和 KBot 标识

| 路径 | 发行名 | Import 名 |
|---|---|---|
| `packages/platform_core` | `kbot-platform-core` | `platform_core` |
| `packages/platform_clients` | `kbot-platform-clients` | `platform_clients` |
| `services/main_api` | `kbot-main-api` | `main_api` |
| `services/agent_runtime` | `kbot-agent-runtime` | `agent_runtime` |
| `services/knowledge_core` | `kbot-knowledge-core` | `knowledge_core` |
| `services/model_serving` | `kbot-model-serving` | `model_serving` |
| `services/data_query` | `kbot-data-query` | `data_query` |

环境变量统一 `KBOT_*`，进程名统一 `kbot-*`，Oracle 对象统一 `KBOT_*`。只有迁移说明
可以提及 Ammolite；代码、OpenAPI、日志、错误信息和测试数据不得出现其产品标识。

### 2.3 安装脚本

`install_workspace.sh` 维护唯一的 `members` 数组，同时驱动 editable 安装与 wheel 构建。
editable 使用 `--no-deps`，避免 pip 用发布源解析内部固定版本；第三方依赖由根
`requirements.txt` 安装。生产按 members 顺序构建 wheels，再使用明确的 KBot 发行名和
`==4.0.0` 安装。

## 3. Data Query 详细设计

### 3.1 目录映射

从来源结构迁移到以下 KBot 目录，保留分层但重写 PostgreSQL/身份部分：

```text
services/data_query/src/data_query/
  api/{dependencies,management,runtime}.py
  application/{management,model_validation,runs,runtime,schema_metadata,semantic_models,sources}.py
  domain/{errors,query_plan,states}.py
  contracts/{management,query_plan,runtime}.py
  entities/data_query.py
  repositories/{data_query,platform_access}.py
  persistence/uow.py
  connectors/
    base.py
    oracle/{compiler,executor,introspector}.py
    postgresql/{compiler,executor,introspector}.py
    mysql/{compiler,executor,introspector}.py
  adapters/{credential_cipher,query_executor}.py
  workers/{query_runs,result_expiry,schema_snapshots,semantic_model_generation}.py
  bootstrap/{api,common,worker}.py
  entrypoints/{api,worker}.py
  config.py
```

来源中的 `secret_store.py` 改为 KBot Data Query 专用 `credential_cipher.py`；来源中的
`tenant_id`、PostgreSQL JSON/ARRAY、Permission/Role 查询不能复制。

### 3.2 Oracle 表设计

| 表 | 用途 | 关键约束 |
|---|---|---|
| `KBOT_DQ_CREDENTIAL` | 数据源凭据密文 | Domain、状态、密钥版本；密文/nonce 非空且长度合法 |
| `KBOT_DQ_DATA_SOURCE` | 数据源定义 | Domain 内显示名唯一；配置 Hash、状态、行版本 |
| `KBOT_DQ_SCHEMA_SNAPSHOT` | Schema 快照批次 | Source+Hash 唯一；源版本固定 |
| `KBOT_DQ_SNAPSHOT_OBJECT` | 单对象采集状态 | Snapshot+Schema+Object 唯一 |
| `KBOT_DQ_SEMANTIC_MODEL` | 语义模型根 | Domain 内名称唯一；Active Version 指针 |
| `KBOT_DQ_MODEL_VERSION` | 不可变模型版本 | Model+Version No 唯一；条件唯一 ACTIVE |
| `KBOT_DQ_MODEL_GEN_JOB` | 模型生成任务 | 状态、租约、有限重试 |
| `KBOT_DQ_POLICY` | Domain 内执行预算 | 无 User/Role selector；状态与行版本 |
| `KBOT_DQ_AGENT_BINDING` | Agent 与模型绑定 | Domain+Agent+Model 唯一有效绑定 |
| `KBOT_DQ_VERIFIED_QUERY` | 验证问题与期望 | Version+Question Hash 唯一 |
| `KBOT_DQ_RUN` | 一次自然语言查询 | Domain+Actor+Idempotency Key 唯一 |
| `KBOT_DQ_EXECUTION` | 执行尝试 | Run+Attempt No 唯一；租约字段 |
| `KBOT_DQ_RESULT` | 结果与过期时间 | Run 唯一；JSON 结果、行数、字节数 |
| `KBOT_DQ_EVENT` | 查询运行事件 | Run+Sequence 唯一 |
| `KBOT_DQ_AUDIT` | 管理与执行审计 | 只存摘要和 Hash，不存凭据/完整结果 |

JSON 列使用 `CLOB CHECK (... IS JSON)` 或项目已有 Oracle JSON 类型；UUID 列使用
`RAW(16)`；Actor 使用 `VARCHAR2(256)`；时间使用 `TIMESTAMP(6) WITH TIME ZONE`；布尔值
使用项目统一映射。所有外键和索引在 `006_audit_views.sql` 收口，确保全量建库顺序稳定。

### 3.3 数据源凭据

创建/轮换请求允许用户名和密码，返回详情只包含 `configured`、`credential_id`、
`key_version` 和 `updated_at`。用户名和密码分别使用随机 12 字节 nonce 加密；AAD 为：

```text
kbot:data-query:<domain_id>:<data_source_id>:<credential_version>:<field>
```

密钥配置：

- `KBOT_DATA_QUERY_CREDENTIAL_ENCRYPTION_KEY`；
- `KBOT_DATA_QUERY_CREDENTIAL_KEY_VERSION`。

密码不得出现在 Pydantic repr、Loguru extra、审计 JSON、任务载荷或连接失败异常。Connector
错误统一映射为稳定错误码，不回传数据库驱动原始连接串。

### 3.4 状态机

Data Source：

```text
DRAFT → VALIDATING → ACTIVE
  │         └──────→ FAILED
  └────────────────→ DISABLED
ACTIVE/FAILED/DISABLED → VALIDATING
```

Snapshot：

```text
REQUESTED → DISCOVERING → WAITING_SELECTION → CAPTURING
                                              ├→ READY → SUPERSEDED
                                              ├→ PARTIAL_READY → CAPTURING/SUPERSEDED
                                              └→ FAILED
```

Semantic Model Version：

```text
DRAFT → REVIEW → ACTIVE → RETIRED
          ├→ DRAFT
          └→ REJECTED → DRAFT
```

Run：`CREATED → VALIDATING → PREFLIGHT → QUEUED → EXECUTING → COMPLETED/COMPLETED_EMPTY`，
并允许在显式节点进入 `CLARIFICATION_REQUIRED/REJECTED/FAILED/TIMED_OUT/CANCEL_PENDING/CANCELLED`。
状态转换只由 Domain Service 执行。

### 3.5 语义模型和 Query Plan

Semantic Model 使用逻辑键引用 Dataset/Dimension/Measure；物理 Schema、对象、列只来自
已选 Snapshot。AI 生成只能扩充：

- `display_name`；
- `synonyms`；
- `sensitivity`；
- `warnings`。

AI 不得改变物理映射、类型、聚合、对象成员、发布状态和 Query Budget。

Runtime LLM 只能输出 `DataQueryPlanV1`：Dataset、Measures、Dimensions、Filters、Order、
Limit、Timezone。Compiler 从 Semantic Model 解析物理对象并参数化编译，禁止模型输出 SQL。

### 3.6 MCP 与 SEMANTIC 统一契约

Agent Definition：

```python
data_query_mode: Literal["MCP", "SEMANTIC"] | None
data_profile_name: str | None
```

Semantic Model 绑定由 `KBOT_DQ_AGENT_BINDING` 管理，不在 Agent JSON 中重复保存 ID 列表。
激活校验为：

- 无 `data_query` capability：Mode 和 Profile 必须为空，且不能有 ACTIVE Binding；
- `MCP`：`data_profile_name` 必填，不能有 ACTIVE Semantic Binding；
- `SEMANTIC`：Profile 必须为空，至少一个 ACTIVE Binding 指向 ACTIVE Version。

统一 Skill 输入包含 `agent_id`、`domain_id`、`actor_id`、问题、Standalone Query、Run/Task
关联和 Budget。统一输出：

```json
{
  "schema": "QUERY_RESULT.v1",
  "query_result_id": "UUIDv7",
  "provider": "MCP | SEMANTIC",
  "columns": [],
  "rows": [],
  "row_count": 0,
  "truncated": false,
  "warnings": [],
  "provenance": {}
}
```

MCP provenance 保存 Profile 与外部请求 ID；Semantic provenance 保存 DQ Run、Data Source、
Semantic Model Version 和 Query Plan Hash。不得保存数据库密码或未脱敏 SQL 参数。

### 3.7 内部接口

Data Query API 使用：

- `/internal/v1/data-query/management/*`：数据源、Snapshot、Semantic Model、Policy、Binding、
  Verified Query 和 Audit；
- `/internal/v1/data-query/runs`：创建 Run；
- `/internal/v1/data-query/runs/{id}`、`/result`、`:cancel`；
- `/internal/v1/data-query/planning-context/{agent_id}`。

Main API 的公开路径保持 `/api/v1/data-query/*`。所有更新使用 `If-Match` 或显式
`expected_row_version`，创建/命令使用 `Idempotency-Key`。

### 3.8 测试矩阵

- Unit：状态机、Plan Schema、Compiler、敏感字段、Hash、结果规范化；
- Contract：MCP/SEMANTIC 同一 Query Result 和事件；
- Integration：Repository/UoW、乐观锁、租约、结果过期；
- Acceptance：Oracle DDL/ORM、包安装、OpenAPI；
- Smoke：三类数据源连通性，至少 Oracle 完整 Runtime，MCP 回归；
- Security：只读绕过、SQL 注入、Schema 越界、超预算、密文篡改、日志泄露。

## 4. Agent Runtime 详细设计

### 4.1 差异处理

KBot 已有记忆实体、Repository、Worker 和 `database/oracle/agent_runtime/006_memory.sql`，
因此不得从来源覆盖整套 Memory。逐项迁移以下缺口：

- `domain/memory_policy.py` 的受控共享规则；
- `specialists/data_query`，但实现为 MCP/SEMANTIC 门面；
- `specialists/hybrid`；
- `specialists/visualization`；
- 缺失的任务保留、通知 Publisher 和模型/资源引用查询。

来源中的 Tenant Access Repository、Permission、Notification Repository 不直接迁移。

### 4.2 长期记忆策略

记忆作用域保留 KBot 现有定义。只有回复语言、回复格式、时区、单位制和无障碍偏好等固定
低敏键，经确定性 Schema 验证后才允许跨会话共享。模型产生的其他键、业务事实、身份、
联系方式不得自动扩大范围。遗忘操作同时关闭 Memory Item、删除 Source 关联并刷新索引。

### 4.3 Specialist DAG

```text
DATA_QUERY:
context-rewrite → data-query → [visualization] → response-composer

HYBRID:
context-rewrite → knowledge-retrieval + data-query
                → evidence-merge → [visualization] → response-composer
```

Hybrid 只有 Agent 同时配置 Knowledge Collection 与 Data Query 时可用；每个分支独立失败，
合并器明确标注缺失来源，不能把 Query Result 伪装成 KC Evidence。

### 4.4 可靠性

继续使用现有 Task 租约与幂等键。新增 Skill 必须声明 Manifest、版本、Timeout、重试类型、
输入/输出 Schema；外部数据查询超时不自动无限重试。公开事件仅描述执行阶段、候选数量、
结果规模和警告。

## 5. Knowledge Core 详细设计

### 5.1 预览服务

新增：

- `api/preview_router.py`；
- `application/preview.py`；
- Repository 的 scope-safe 查询方法；
- `platform_clients.knowledge` 对应方法；
- Main API 瘦转发接口。

Revision Preview 返回成员顺序、角色、文件名、MIME、字节数和可预览标记。Source File
接口在数据库事务中校验归属，事务结束后用受控存储句柄流式输出；URI、绝对路径和对象
存储凭据永不返回。

### 5.2 模型引用反查

新增内部接口按 `model_id` 返回 Collection、用途（parse/vlm/embedding/profile）和状态。
Model Serving 只使用此接口，不访问 KC 表。引用接口在 KC 不可用时返回 unavailable，模型
删除操作必须失败关闭；列表展示类操作可降级为空并标注 unavailable。

### 5.3 Purge

Purge 使用持久化 Job：标记 DELETING、删除派生产物、删除对象、删除关系、完成 Tombstone。
每步保存进度，可重复执行。对象删除失败保留 Job 和资源引用，不能先删除数据库事实导致
对象无法定位。

## 6. Model Serving 详细设计

### 6.1 单一持久化路径

保留 `common/entities/ai_model.py`，以 `persistence/uow.py + repositories/model.py` 作为唯一
写路径；现有重复 `repository.py` 在调用迁移完后删除。Model Registry Service 所有写命令
在一个 UoW 内完成，Repository 不提交事务。

### 6.2 生命周期与引用

模型状态使用 `DRAFT/ACTIVE/ARCHIVED`。归档禁止新引用但不破坏历史 Run；删除只允许无任何
引用且无运行中模型实例的 ARCHIVED 模型。

引用聚合响应：

```json
{
  "model_id": "UUIDv7",
  "references": [
    {"service": "agent-runtime", "resource_type": "agent", "resource_id": "...", "usage": "router_llm"},
    {"service": "knowledge-core", "resource_type": "collection", "resource_id": "...", "usage": "embedding"},
    {"service": "data-query", "resource_type": "semantic_model", "resource_id": "...", "usage": "generation_llm"}
  ],
  "unavailable_services": []
}
```

删除时任何依赖服务不可用都失败关闭。缓存失效事件包含 model ID、served name、category、
row version，不包含 Provider Secret。

### 6.3 Provider 校验

Provider Options 是代码控制目录，定义类别、必要参数、Secret 字段、是否支持 Tool Calling、
最大上下文和 Embedding 维度。更新时按 Provider Schema 校验，未知参数拒绝。

## 7. Development Logs 详细设计

### 7.1 API

升级现有路径：

- `GET /api/v1/development/logs/services`；
- `GET /api/v1/development/logs/events`；
- `GET /api/v1/development/logs/events/{event_id}`。

列表不返回 Raw/Traceback，详情才返回完整多行文本。查询支持 service、stream、levels、
keyword、request_id、trace_id、error_id、run_id、job_id、HTTP status、时间范围和 cursor。

### 7.2 文件读取

服务目录由 KBot topology/configuration 定义，不接受调用者路径。只读取受控日志根下
`kbot-*` 服务的 `runtime.log*`/`access.log*`。读取 bounded tail，限制文件数、扫描字节、
时间窗口和返回条数。Event ID 由文件身份、偏移和内容 Hash 生成，详情接口据此重新定位。

### 7.3 脱敏

对结构化和 Raw 文本同时处理：Authorization、Cookie、API Key、JWT、password、secret、
credential、连接串、私钥、数据库用户名及大 Result/Prompt 字段。无法安全解析的行只返回
受限 Raw 摘要。

## 8. 通知中心详细设计

### 8.1 所有权

不建立独立服务：

- `platform_core.notifications`：事件目录、Outbox Entry、Repository、投影 DTO；
- Producer Service：业务事务内写 Outbox；
- Main API Repository/Application：投影、Inbox、Work Item、Operation、Watch；
- Main API Worker：领取 Outbox、逐条隔离处理、重试；
- Main API API：读取、已读、关注、SSE。

### 8.2 Oracle 表

| 表 | 唯一性/用途 |
|---|---|
| `KBOT_NOTIFICATION_OUTBOX` | Producer+Event Key 唯一；状态、重试、租约、Payload |
| `KBOT_NOTIFICATION_INBOX` | Outbox+Recipient 唯一；已读时间、保留期 |
| `KBOT_NOTIFICATION_PREF` | Domain+Actor+Event Type 唯一 |
| `KBOT_WORK_ITEM` | Domain+Actor+Resource+Action 的条件唯一 OPEN 项 |
| `KBOT_BACKGROUND_OPERATION` | Producer+Operation ID 唯一；状态 Projection |
| `KBOT_OPERATION_WATCH` | Operation+Domain+Actor 唯一 |

Payload 必须包含 `domain_id`、event type/version、resource type/id/name、initiator actor、
明确 recipient actor IDs、安全 summary、occurred_at、correlation ID。没有明确收件人时
`recipient_actor_ids=[]`，仍可更新 Operation，但不生成 Inbox。

### 8.3 事件目录

事件类型由代码控制，移除 Permission Recipient 和所有 AIOps/平台治理事件。首批目录覆盖
Agent、KC、Data Query 和 Model Serving。事件 Producer 与 KBot service name 对齐，渠道
只允许 `IN_APP`。

### 8.4 投影与 SSE

Dispatcher 使用 `FOR UPDATE SKIP LOCKED` 领取，单条投影使用独立 UoW；一条坏事件不能
回滚整批。超过重试上限进入 QUARANTINED。SSE ID 使用 Inbox/Projection 单调序号，支持
`Last-Event-ID`、心跳、去重和补拉；SSE 不是事实存储。

## 9. Main API 组合编排详细设计

### 9.1 Application Service

新增 `main_api/application/resource_composition.py` 和只读 Projection DTO。Router 不直接串联
多个 Client；每个组合用例由 Application Service 执行：

```text
load current versions
  → validate references and availability
  → execute one authoritative domain command
  → verify resulting references
  → return composition receipt
```

如果命令后验证失败，记录 `COMPENSATION_REQUIRED` Receipt 和安全错误，不谎报成功。后续
重试从 Receipt 继续，不重复创建资源。

### 9.2 组合接口

- Agent Configuration：模型、Collection、Data Query Mode/Binding；
- Collection Configuration：解析、VLM、Embedding、Profile 模型引用；
- Semantic Model Publication：Snapshot、验证模型、Agent Binding；
- Resource Decommission：完整阻塞引用图；
- Run Composition：配置快照、任务、Artifact、KC Evidence、DQ Result、模型和通知摘要。

组合层不创建跨服务外键，也不复制业务实体。Projection 缓存可延迟一致，但每个片段必须
返回 `source_service`、`source_version`、`observed_at` 和 `availability`。

## 10. OpenAPI、错误与测试约定

### 10.1 错误格式

新增稳定错误码按模块前缀：`DATA_QUERY_*`、`AGENT_*`、`KC_*`、`MODEL_*`、`LOG_*`、
`NOTIFICATION_*`、`COMPOSITION_*`。数据库驱动、SQL、路径、密钥和内部 Trace 不进入公开
错误。409 用于自然键/行版本/状态冲突，422 用于契约输入，503 用于依赖不可用。

### 10.2 OpenAPI

每个阶段更新对应 internal/public snapshot，并运行 residual route 扫描。Ammolite 的 Tenant、
App、Permission、User、Admin 路由不得进入 KBot Snapshot。

### 10.3 架构测试

增加自动检查：

- API/Application 不导入 SQLAlchemy Session、Table 或 Repository 实现；
- Repository 不导入 FastAPI/Client，不调用 commit；
- 服务不跨源码 import；
- 所有内部 Client 使用短时 audience token；
- 新表、实体、DDL、Manifest 一致；
- KBot 生产资产无 Ammolite 标识；
- 敏感 DTO 不出现在日志和 repr。

## 11. 配置清单

新增或调整：

- `KBOT_DATA_QUERY_CREDENTIAL_ENCRYPTION_KEY`；
- `KBOT_DATA_QUERY_CREDENTIAL_KEY_VERSION`；
- Data Query API/Worker 端口、audience、轮询、租约、结果保留期；
- Connector 连接/语句超时、行数、字节数、Schema allowlist；
- Main API Notification Dispatcher 批量、租约、重试和 SSE 心跳；
- Development Logs 根目录、扫描字节、时间窗口和最大结果数；
- `KBOT_PYTHON`、安装模式和生产 wheel 目录。

所有配置都进入 `.env.example`/configuration 示例；真实密钥只由初始化脚本生成到 `.env`
或生产 Secret，不进入仓库。

## 12. 分阶段目标文件清单

以下是实施时必须同步检查的最小文件集合；实际新增文件可按现有分层拆分，但不能遗漏
同一能力的 Contract、Client、DDL、测试和 OpenAPI。

### 12.1 S0 工作区

- `pyproject.toml`；
- `requirements.txt`；
- `scripts/deployment/install_workspace.sh`；
- `start_kbot.sh` 及相关启动/停止脚本；
- 所有 `packages/*/pyproject.toml`、`services/*/pyproject.toml`；
- `tests/acceptance/check_workspace_packages.py`。

### 12.2 S1 Data Query

- `services/data_query/src/data_query/**`；
- `services/data_query/pyproject.toml`；
- `packages/platform_core/src/platform_core/contracts/data_query/**`；
- `packages/platform_clients/src/platform_clients/data_query.py` 及导出；
- `services/agent_runtime/src/agent_runtime/{config.py,application/agent_definitions.py}`；
- `services/agent_runtime/src/agent_runtime/specialists/{registry.py,root/planner.py,data_query/**}`；
- `services/main_api/src/main_api/api/data_query.py` 及 Bootstrap；
- `database/oracle/data_query/001_*.sql` 至 `006_*.sql`、Manifest；
- `configuration/`、`.env.example` 和环境初始化脚本；
- `tests/unit/data_query/`、`tests/integration/`、`tests/contract/`、`tests/smoke/`；
- Data Query internal 与 Main API public OpenAPI Snapshot。

### 12.3 S2 Agent Runtime

- `services/agent_runtime/src/agent_runtime/domain/memory_policy.py`；
- `services/agent_runtime/src/agent_runtime/specialists/hybrid/**`；
- `services/agent_runtime/src/agent_runtime/specialists/visualization/**`；
- Conversation/Runtime Application Service、Worker、Repository/UoW 的差异补丁；
- `database/oracle/agent_runtime/006_memory.sql`，仅在实体确有增量时修改；
- `tests/unit/agent_runtime/`、Agent Runtime internal OpenAPI 和 Main API Contract Test。

### 12.4 S3 Knowledge Core

- `services/knowledge_core/src/knowledge_core/api/preview_router.py`；
- `services/knowledge_core/src/knowledge_core/application/preview.py`；
- Collection/Intake/Purge Repository 和 UoW；
- 模型引用查询 Application/Repository/Internal Router；
- `packages/platform_clients/src/platform_clients/knowledge.py`；
- `services/main_api/src/main_api/api/knowledge.py`；
- KC internal/Main API OpenAPI、Unit/Contract/Storage Smoke。

### 12.5 S4 Model Serving

- `services/model_serving/src/model_serving/persistence/uow.py`；
- 唯一 Model Repository、Catalog/Registry Application Service；
- `common/management_router.py`、四类模型 Pool 的失效处理；
- `packages/platform_core/src/platform_core/contracts/model.py`；
- `packages/platform_clients/src/platform_clients/models.py`；
- `services/main_api/src/main_api/api/models.py`；
- `database/oracle/model_serving/001_model_registry.sql` 及新增关系/索引脚本；
- Unit/Integration/Reference Contract/OpenAI Smoke/OpenAPI。

### 12.6 S5 Development Logs

- `services/main_api/src/main_api/api/development_logs.py`；
- `services/main_api/src/main_api/log_reader/{__init__,log_search}.py`；
- `services/main_api/src/main_api/api/development_agent_runs.py`；
- topology/logging 配置；
- 日志搜索、脱敏、分页、轮转、目录安全测试；
- Main API public OpenAPI。

### 12.7 S6 通知中心

- `packages/platform_core/src/platform_core/notifications/{catalog,delivery,outbox_repository}.py`；
- `services/main_api/src/main_api/application/notification_{center,inbox,projection}.py`；
- `services/main_api/src/main_api/repositories/notification_{inbox,projection}.py`；
- `services/main_api/src/main_api/api/notifications.py`；
- Main API Notification Dispatcher 入口和启动配置；
- `database/oracle/platform_core/003_notifications.sql`；
- Agent Runtime、KC、Data Query、Model Serving 各自的 Outbox Publisher 接点；
- Catalog/Projection/Inbox/SSE/Replay/Quarantine E2E 和 OpenAPI。

### 12.8 S7 组合编排

- `services/main_api/src/main_api/application/resource_composition.py`；
- `services/main_api/src/main_api/api/` 下组合管理 Router；
- `packages/platform_core/src/platform_core/contracts/` 下组合 Receipt/Projection DTO；
- `platform_clients` 各服务引用查询；
- Idempotency/Compensation 持久化；
- 故障注入、重放、跨服务引用和 Run Projection Contract Test；
- Main API public OpenAPI。
