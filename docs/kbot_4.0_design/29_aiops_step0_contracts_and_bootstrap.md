# 4.0 AIOps 步骤 0：契约、配置与启动骨架

> 实施状态：已于 2026-07-23 完成。实现位于 `aiops_agent/`、
> `platform_core/contracts/aiops/`、`platform_clients/aiops.py` 和四个
> `apps/aiops_*` 入口；三份冻结 OpenAPI 位于 `docs/openapi/`。步骤 0 没有
> 创建任何 `KBOT_OPS_*` 表或业务路由。

## 目标与约束

步骤 0 只建立可编译、可启动、无业务行为的骨架，为后续 DDL 和运行时提供稳定边界。不得提前创建 `KBOT_OPS_*` Entity、访问旧 Ops 表、调用 LLM/Monitor/目标数据库或实现临时内存状态机。

现有 KC/Model App 的数据库工厂和日志能力可以复用，但 AIOps 不复制以下过渡做法：入口文件集中组装全部业务对象、生产环境共享静态 Internal Token、在 TOML 中保存 Monitor/目标数据库凭据、`allow_origins=["*"]`、模块 import 时创建 Worker/Client 单例。

## 最终包结构

```text
aiops_agent/
  api/
    management/             # Main API 代理的配置、Run、HITL、Report Route
    intake/                 # Root 委派与 Monitoring Intake Route
    executor/               # Executor claim/callback Route
    dependencies.py         # AuthContext、Service Identity、Application Service 注入
    errors.py               # Domain/Application Error → Problem Details
  application/
    targets/                # Command/Query Handler，不含 HTTP
    intake/
    runs/
    diagnosis/
    changes/
    inspections/
    dto/                    # 进程内 Use-case DTO
  domain/
    target/
    operations/
    diagnosis/
    change/
    inspection/
  orchestration/
  diagnostics/
  ports/
  adapters/
  entities/
  repositories/
  persistence/
  workers/
  contracts/                # AIOps 内部 Artifact Schema，不是 HTTP DTO
  bootstrap/
    api.py
    worker.py
    scheduler.py
    executor.py
    common.py
  tests/

apps/
  aiops_api/main.py
  aiops_worker/main.py
  aiops_scheduler/main.py
  aiops_db_executor/main.py

platform_core/contracts/
  auth.py                   # 平台通用 AuthContext/Service Identity Envelope
  aiops/
    types.py
    public.py
    internal.py
    executor.py
    events.py
    errors.py

platform_clients/aiops.py
database/oracle/aiops_agent/
```

`apps/*/main.py` 只加载对应配置、调用 Bootstrap Factory、运行进程并处理退出信号。对象组装、Router 注册、Client/UoW Factory 创建和生命周期清理都属于 `aiops_agent/bootstrap`，不能继续堆入入口文件。

## 依赖方向

```text
apps ──→ bootstrap ──→ api/application/adapters/persistence
                         ↓
application ──→ domain + ports
adapters ──→ ports + external SDK/platform_clients
repositories/entities ──→ domain mapping + platform_core.persistence
api ──→ application + platform_core.contracts

domain ──→ Python stdlib only
```

禁止依赖：

- `domain` import FastAPI、Pydantic、SQLAlchemy、`platform_core` 或任何 Client；
- `application` import FastAPI、具体 Monitor SDK 或 SQLAlchemy Entity；
- `api` 直接 import Repository/Entity 或执行 `commit()`；
- `aiops_agent` import KC/Model/Agent Runtime Repository；
- `platform_core/contracts` import `aiops_agent`；
- `platform_clients.aiops_management/aiops_delegation` import AIOps Entity、Domain 或 App；
- 任意 4.0 AIOps 文件 import `legacy`、旧 `agent`、旧 `services`、旧 `skills` 或 `utils.monitor`。

跨层转换使用显式 Mapper。Domain 状态枚举属于 `aiops_agent/domain`；Wire Enum/Literal 属于 `platform_core/contracts/aiops/types.py`。两者不互相 import，通过映射与枚举一致性测试防止漂移。

## 四个进程边界

| App | 默认端口 | DB/外部权限 | 启动内容 |
| --- | ---: | --- | --- |
| `aiops_api` | `18110` | AIOps Schema、SecretRef 元数据；不连目标 DB | Internal API、SSE、Inbox 接入、健康检查 |
| `aiops_db_executor` | `18111` | Secret Store、目标 DB；不持有 KBot Schema 凭据 | Executor API、模板 Registry、安全校验、连接池 |
| `aiops_worker` | `18112`（仅探针） | AIOps Schema、Monitor/KC/Model/Executor Client | Task 领取、状态机、Outbox Dispatcher、Reconciler |
| `aiops_scheduler` | `18113`（仅探针） | AIOps Schema；不连 Monitor/目标 DB | Plan 领租、创建 Run、推进 `next_run_at` |

端口均可配置；`18100–18109` 预留给 Main/Agent Runtime。Worker/Scheduler 的 HTTP 只允许 `/live`、`/ready`、`/metrics`，不承载领域 API。四个进程分别创建和关闭自己的日志、Trace 和 HTTP Client；需要 AIOps Schema 的进程分别创建独立 `DatabaseRuntime` 和连接池。

DB Executor 刻意不持有 KBot Schema 账号。执行 Mutation 前调用 AIOps API 的原子 Claim 契约，由 API 条件更新 Execution、消费 Approval Token 并返回一次执行许可；重复或过期 Claim 不执行目标命令。执行后通过幂等回调提交结果。网络中断造成结果不确定时标记 `UNKNOWN` 并对账，绝不盲目重复 Mutation。

## Bootstrap Factory

建议入口保持如下规模：

```python
from aiops_agent.bootstrap.api import create_aiops_api
from platform_core.config import get_aiops_api_config

config = get_aiops_api_config()
app = create_aiops_api(config)
```

Factory 负责创建 `AppRuntime`，集中持有需要关闭的资源：

```text
AppRuntime {
  settings, database_runtime?, http_clients,
  service_identity, metrics, background_tasks
}
```

FastAPI lifespan 只调用 `runtime.start()`/`runtime.close()`。不得在模块 import 时读取 Secret、创建 Event Loop Task、连接数据库或实例化 Worker。所有 Worker 都通过 Constructor Injection 获得 Port/Factory，测试时可替换为 Fake。

## 跨服务契约

### 通用表示

- Agent ID 和所有领域资源 ID 使用 UUIDv7；Oracle 内以 `RAW(16)` 保存同一 PK/FK，JSON 契约序列化为规范字符串，不维护数字 Persistence ID；
- 时间使用 RFC 3339 UTC，例如 `2026-07-23T10:30:00.000Z`；
- JSON 字段使用 `snake_case`，Hash 使用 64 位小写十六进制；
- `row_version` 通过响应体返回并映射为 `ETag: "rv-{version}"`；
- 每个顶层 DTO 含固定 `schema_version`，初始值使用 `aiops.public.v1`、`aiops.internal.v1`、`aiops.executor.v1` 或 `aiops.event.v1`；
- Public/Internal DTO 不能包含 SQLAlchemy Entity、Secret、内部 Lease Owner、Prompt 或完整 Policy Snapshot。

### `public.py`

最初冻结以下 DTO 族：

```text
TargetCreate/TargetPatch/TargetView/TargetPage
AgentBindingCreate/AgentBindingPatch/AgentBindingView
MonitorSourceCreate/MonitorSourcePatch/MonitorSourceView
MonitorBindingCreate/MonitorBindingPatch/MonitorBindingView
InspectionPlanCreate/InspectionPlanPatch/InspectionPlanView
InspectionFireSummary/InspectionFireView
OpsRunCreate/OpsRunReceipt/OpsRunSummary
PendingInputView/HitlResponse/HitlResult
ProposalView/ApprovalCommand/RejectionCommand/ManualResultCommand
ReportSummary/ReportView/ReportVersionSummary/UploadSession
ProblemDetails
```

Public DTO 由 Main API 对外发布。AIOps API 不直接信任其中的 actor/domain/app 字段；Mapper 使用已验证的 `AuthContext` 构造 Internal Command。

### `internal.py`

```text
CreateOpsRunCommand/OpsRunReceipt/OpsRunQuery
OpsCommandEnvelope
MonitorWebhookEnvelope/EventReceipt
ArtifactRef/FinalDiagnosisRef
RootDelegationRequest/RootDelegationReceipt
DelegationEventPage/AIOpsDelegationResult
ExecutorClaimRequest/ExecutorClaimGrant
ExecutorEventEnvelope
```

`OpsCommandEnvelope.command_type` 只允许版本化类型，不接受“method + arbitrary payload”。初始类型为 `CANCEL_RUN`、`ANSWER_HITL`、`CANCEL_HITL`、`APPROVE_PROPOSAL`、`REJECT_PROPOSAL` 和 `RECORD_MANUAL_RESULT`。

Root 委派必须带 `delegation_id`、`parent_agent_run_id`、目标范围、用户意图、Deadline 和受签名 AuthContext；不能携带 Approval、Policy 或 Executor 字段。Result 是受限 Envelope，不是 Ops Entity/Artifact 的跨库引用。Monitoring Envelope 保留原始字节的 URI/Hash、允许的签名 Header、接收时间和 Webhook Key Hash，不把 Provider Payload 预先解释为可信 DTO。

### `executor.py`

分别定义只读和变更请求，不能使用一个可选字段很多的万能 DTO：

```text
ReadDiagnosticRequest/ReadDiagnosticResult
MutationExecutionRequest/MutationClaimRequest/MutationExecutionGrant
ExecutionStatusEvent/ExecutionResultRef
```

只读请求只允许 `diagnostic_tool_id + version + typed_parameters`。Mutation 请求只允许 `action_template_id + version + typed_parameters + proposal/policy hashes + approval token`。两者都不接受 SQL 文本、自然语言、连接串或密码。

### `events.py` 与 `errors.py`

SSE 使用带判别字段的 Union：`RunStatusEvent`、`TaskStatusEvent`、`DiagnosticProgressEvent`、`InputRequiredEvent`、`ApprovalRequiredEvent`、`ExecutionStatusEvent`、`ReportReadyEvent` 和终态事件。Unknown Event 必须可跳过，不能使客户端断流。

错误码按领域稳定，异常信息可变：`OPS_NOT_FOUND_OR_DENIED`、`OPS_IDEMPOTENCY_CONFLICT`、`OPS_ROW_VERSION_CHANGED`、`OPS_STATE_CONFLICT`、`OPS_POLICY_DENIED`、`OPS_HITL_EXPIRED`、`OPS_APPROVAL_INVALID`、`OPS_EXECUTION_UNKNOWN`、`OPS_UPSTREAM_UNAVAILABLE`。Client 只按 `code/retryable/status` 决策，不解析自然语言 `detail`。

## AIOps Client 骨架

`AIOpsManagementClient` 与 `AIOpsDelegationClient` 分别封装管理/用户契约和 Root 子 Run 契约，不能让一个宽接口按调用方运行时判断权限。两者都使用长期复用的 Async HTTP Session；构造参数为 `base_url`、Service Identity Provider、超时、重试策略和 Trace Provider，不从全局配置隐式读取 URL 或 Token。

自动重试仅允许：

- GET/HEAD；
- 携带 `Idempotency-Key` 且尚未收到响应头的 Command；
- `429/502/503/504` 且服务端标记 `retryable=true`。

Mutation、Approval 和 HITL Reply 在响应不确定时先按幂等键查询结果，不直接重发。Client 负责 Problem Details → Typed Exception 映射，不记录请求正文、AuthContext Token、SQL 结果或 SecretRef。

## Service Identity 与 AuthContext

现有 `X-KBot-Internal-Token` 只作为旧服务过渡方案，不用于 AIOps。新增平台通用验证组件，内部调用至少携带：

```text
ServiceIdentity {
  issuer, subject, audience, scopes,
  issued_at, expires_at, token_id
}
SignedAuthContext {
  request_id, trace_id, principal_id, domain_id,
  roles, scopes, authorized_agent_ids,
  issued_at, expires_at, delegated_by
}
```

生产使用短期签名 Token；签名密钥/证书来自环境或 Secret Provider，不能写入 TOML。验证固定 audience、调用方白名单、过期时间和最小 Scope，并对 `token_id`/审批 Token 使用不同命名空间。未来换成 mTLS 时保留相同 Claim 和授权语义。

调用权限初始矩阵：

| 调用方 | AIOps 能力 |
| --- | --- |
| Main API | 配置、Run、HITL、审批、报告、Monitoring Intake |
| Agent Runtime | 仅以 Delegation ID 创建子 Run、读取安全事件/Result Envelope、请求取消 |
| Worker | Task/Artifact/Outbox 内部命令、Executor 请求 |
| Scheduler | 仅 Plan Lease、Inspection Fire 和 Schedule Run 内部命令 |
| DB Executor | Mutation Claim、状态/结果回调 |

## 配置模型

组合配置沿用当前已经统一的“平台共享配置 + 服务自有配置模型”结构，定义在
`aiops_agent/config.py`，由 `platform_core.config.load_settings` 分层加载，而
不是继续扩展旧 `ExecutorConfig/PrometheusConfig/ZabbixConfig/OemConfig`：

```text
AIOpsConfig
  api: AIOpsApiConfig
  runtime: AIOpsRuntimeConfig
  worker: AIOpsWorkerConfig
  scheduler: AIOpsSchedulerConfig
  executor: AIOpsExecutorConfig
  clients: AIOpsDependencyEndpoints
  secret_store: SecretStoreConfig
  limits: AIOpsLimitsConfig
```

建议 TOML：

```toml
[aiops.api]
service_name = "kbot-aiops-api"
service_version = "4.0.0"
host = "0.0.0.0"
port = 18110

[aiops.runtime]
system_aiops_agent_id = "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11"

[aiops.worker]
service_name = "kbot-aiops-worker"
probe_port = 18112
concurrency = 4
claim_interval_seconds = 2
lease_seconds = 120
heartbeat_seconds = 30

[aiops.scheduler]
service_name = "kbot-aiops-scheduler"
probe_port = 18113
scan_interval_seconds = 30
lease_seconds = 120

[aiops.executor]
service_name = "kbot-aiops-db-executor"
host = "0.0.0.0"
port = 18111
mutation_enabled = false
readonly_concurrency = 8
mutation_concurrency = 1
statement_timeout_seconds = 60
max_result_rows = 5000

[aiops.clients]
model_serving_url = "http://127.0.0.1:18092"
knowledge_core_url = "http://127.0.0.1:18090"
aiops_api_url = "http://127.0.0.1:18110"
db_executor_url = "http://127.0.0.1:18111"

[aiops.secret_store]
provider = "environment" # 仅本地开发
```

Target、Monitor Source、Provider Endpoint、API Token、数据库连接和密码均是领域数据或 SecretRef，不属于 `[aiops]` 静态配置。AIOps 新代码不读取现有 `[prometheus]`、`[zabbix]`、`[oem]` 中的 Token/密码；这些段随 Legacy 清理删除。

配置启动校验至少包括：`system_aiops_agent_id` 是 UUIDv7、`heartbeat < lease/2`、Mutation 默认关闭、Mutation 并发不大于安全上限、生产不能使用默认 Service Token/Environment Secret Provider、API 与 Executor Service Identity 不同、URL Scheme/Host 在允许范围内、结果/超时/预算为正数。System AIOps Agent 由后台依赖健康检查验证；Agent Runtime 暂时不可用不使配置 API 整体 Not Ready，但自动 Run 必须失败关闭并重试，不能跳过 Agent/Target Binding 校验。

## 健康、日志和 OpenAPI

- `/live` 只表示进程 Event Loop 正常，不访问数据库或 Provider；
- `/ready` 检查本进程的必需依赖：API/Worker/Scheduler 检查 AIOps Schema 版本，Executor 检查模板 Registry、Secret Provider 和身份验证器；
- 单个 Target 或 Monitor Source 不可用不会使整个进程 Not Ready；
- `/metrics` 只在内部网络暴露，不包含 Target 名称、SQL 或用户输入；
- 日志强制携带 `service_name/request_id/trace_id`，后续有 Run 时再加入 `ops_run_id/ops_task_id`。

Public OpenAPI 由 Main API 生成；AIOps Internal 与 Executor 各生成一份独立 OpenAPI。生产默认关闭 `/docs`，CI 保存 OpenAPI Snapshot 并检查破坏性变化。Worker/Scheduler 没有领域 OpenAPI。

AIOps API、Executor、Worker Probe 和 Scheduler Probe 都不安装 CORS Middleware；它们只接受内部网络和 Service Identity。浏览器跨域仅由 Main API 按部署环境的显式 Origin Allowlist 处理，生产不允许 `*`。

## 步骤 0 实现顺序

1. 新增 AuthContext、Service Identity 和 AIOps Wire DTO；
2. 新增 AIOps 组合配置及 `base.toml.example`，不写实际 Secret；
3. 实现 Management/Delegation Client 的认证、超时、错误和空方法骨架；
4. 创建 `aiops_agent` 分层包、Bootstrap Runtime 和四个 App；
5. 实现 `/live`、`/ready`、空 Router 和 OpenAPI Snapshot；
6. 增加依赖方向、DTO 序列化、配置不变量和生产安全检查；
7. 确认四个 App 可分别启动/退出且无业务表、外部 Provider 或模型调用。

步骤 0 不创建兼容路由，不搬迁 `utils.monitor`，不实现 Repository/UoW，也不为了让 `/ready` 通过而自动建表。Schema 尚未部署时，进程可 Live 但 API/Worker/Scheduler 必须 Not Ready。

## 完成定义

- 四个 App 无 import side effect，可独立启动和优雅退出；
- Public/Internal/Executor/Event DTO 和错误码有固定版本；
- Main、Root、Worker、Scheduler、Executor 的调用权限不可互换；
- 所有 JSON ID、时间、ETag、幂等和错误表示一致；
- 生产配置不接受默认 Token、明文 Monitor/目标数据库凭据或宽松 CORS；
- 架构检查能阻止新 AIOps 代码依赖 Legacy 或其他领域 Persistence；
- 下一步可以只通过新增规范建库脚本、Entity 和 Repository 实现步骤 1/2，而无需修改上述边界。
