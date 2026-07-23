# 4.0 Main API / BFF 与 AuthContext

## 服务边界

Main API/BFF 是所有外部请求的唯一入口和协议组合层，不是业务数据 Owner。

```text
KM Portal / APEX / MCP / Slack / Client
                 ↓
             Main API / BFF
       Auth · Rate limit · DTO · SSE
       ↙          ↓          ↓          ↘
 Agent Runtime  AIOps  Knowledge Core  Model Serving
```

Main API 负责认证、请求校验、授权上下文构建、Run API、SSE 代理、错误映射和审计入口；不直接读取 Knowledge Core、Agent 或 Model Repository，不实现检索、解析、规划或模型调用。即使 4.0 共用同一 Schema，调用规则也不改变。

## AuthContext

Main API 验证外部 JWT、API Key 或 APEX 会话后，构建不可由请求体覆盖的上下文：

```text
AuthContext {
  request_id
  trace_id
  principal_type       # USER / SERVICE / SYSTEM
  principal_id
  domain_id
  roles
  scopes
  authorized_agent_ids
  authorized_collection_ids
  max_security_level
  delegated_by
  issued_at
  expires_at
}
```

`domain_id` 是强制隔离边界，来自已验证身份或 APEX 会话；客户端提交的 domain、user、security level 和资源 ID 只能作为更窄的筛选条件。`app_id` 仅在需要 APEX 直连视图过滤时作为平台上下文保留，不参与 Agent 路由、Collection 选择或业务授权决策。

资源范围必须按以下顺序求交集：

```text
身份声明
 ∩ Domain 授权
 ∩ Agent Binding
 ∩ Collection 状态与 ACL
 ∩ 请求体候选资源
```

结果为空时拒绝请求；不能因为客户端指定了 Collection ID 就扩大范围。下游服务再次执行资源校验，BFF 的检查不是唯一安全边界。

## 服务间传播

外部 Bearer Token 不直接转发到 KC、Model Serving 或 Agent Runtime。Main API 使用服务身份调用下游，并附带短期、限定 audience 的签名上下文令牌或等价的内部 DTO：

```text
Authorization: service credential
X-Request-ID: original request id
traceparent: distributed trace
Internal-Auth-Context: signed short-lived context
```

下游必须校验签名、过期时间、audience、调用服务和 domain。客户端伪造的 `X-Domain-ID`、`X-Agent-ID` 或 `X-Collection-ID` 不可信；边界层应删除这些外部转发头。未来部署 mTLS 时只替换服务身份验证方式，不改变 AuthContext DTO。

## Client 边界

Main API 只通过版本化 Client 调用下游：

| Client | 目标 | 允许的能力 |
| --- | --- | --- |
| `AgentRuntimeClient` | Agent Runtime | 创建/查询/取消/审批 Run，订阅事件 |
| `KnowledgeCoreClient` | Knowledge Core | 入库、状态、Discovery、Evidence、文件访问授权 |
| `ModelServingClient` | Model Serving | 受策略限制的推理和模型配置读取 |
| `AIOpsManagementClient` | AIOps Service | Target/Run/HITL/审批/报告和监控事件接入 |
| `MCPDataClient` | 现有 MCP 问数工具 | 受控查询调用和结果回传；当前不引入 Data Agent |

Client 统一处理 URL allowlist、连接/读取/总超时、重试边界、错误 DTO、Trace 和服务身份。Main API 不 import 下游 Repository，也不复用旧 `services` 或旧 Agent 类。

Agent Runtime 另用能力更窄的 `AIOpsDelegationClient`，只允许创建/查询/取消其自身委派的子 Run。Management 与 Delegation Client 使用不同接口和 Service Scope，Root 代码中不存在审批或配置方法。

## v4 API 组合

Main API 对外只发布 v4 契约：

- `/v4/runs`：Run 创建、查询、取消、审批和恢复；
- `/v4/runs/{run_id}/events`：带 `Last-Event-ID` 的 SSE；
- `/v4/knowledge/intake`：普通文件和 Bundle 入库的统一入口，转发到 KC；
- `/v4/knowledge/collections`：Collection 管理和 Agent Binding 管理；
- `/v4/ops/*`：AIOps Target、Run、HITL、审批、巡检和报告；
- `/v4/integrations/monitoring/*`：经限流和来源验证的监控事件接入；
- `/v4/files/{document_version_id}`：权限校验后签发短时下载地址；
- `/v4/health`、`/v4/ready`：只返回服务健康摘要，不泄露数据库凭据或内部拓扑。

耗时操作统一返回 `202` 和资源标识；同步接口只用于轻量查询。请求/响应使用 Pydantic DTO，禁止直接返回 SQLAlchemy Entity。旧 `/api/kb`、旧 Agent SSE 和旧 `doc_results` 不提供兼容路由。

AIOps 的 Public/Internal/Executor 三层契约见 [27_aiops_api_and_contracts.md](27_aiops_api_and_contracts.md)。Main API 不把 AIOps Internal 路由直接暴露给客户端。

## SSE 代理规则

Main API 不重新生成 Agent 事件，只转发经过校验的版本化事件 DTO。连接建立时先检查 Run 访问范围，再从事件游标读取；断线后由客户端带 `Last-Event-ID` 续传。终端 `RUN_COMPLETED`、`RUN_FAILED`、`RUN_CANCELLED` 或 `RUN_EXPIRED` 后发送 `done` 并关闭连接。

SSE 事件中的正文必须遵守 Artifact 安全等级；大正文通过 Artifact 引用或短时 URL 返回。BFF 不把候选 CitationPack 自动当作最终 `doc_results`，最终引用只来自 Grounded Answer Artifact。

Root 委派的 AIOps 子事件先由 Agent Runtime 按 Child Cursor 幂等投影为父 Event；Main API 不把 AIOps SSE 管道拼到 Root SSE。游标过期、心跳、慢客户端和终态规则见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

## APEX 与外部适配器

APEX 可继续通过受控视图直读 Collection、Run 摘要和投影结果，但写操作和文件上传统一经过 Main API/KC API。视图使用 `app_id`、`domain_id` 过滤，业务代码不把 `app_id` 当作路由参数。Cookie 认证的写操作还必须经过 CSRF、Idempotency Key 和必要的 ETag 校验。完整 SQL、命令、Evidence 和 Report 正文不进入视图。KM Portal、MCP、Slack 都实现为调用 v4 Client/HTTP 契约的 Adapter，不直接访问数据库或旧 Controller。

## 错误、审计与观测

BFF 将下游错误映射为稳定错误码，例如 `AUTH_REQUIRED`、`SCOPE_DENIED`、`RESOURCE_NOT_FOUND`、`RUN_CONFLICT`、`UPSTREAM_UNAVAILABLE`、`RATE_LIMITED`；不把内部 SQL、模型供应商错误或堆栈返回给客户端。每个请求至少记录 `request_id`、`trace_id`、主体、domain、动作、下游服务、状态码和 `run_id`，正文、Token 和密钥不得写入日志。

## 后续拆库

Main API 与下游只依赖 Client/DTO，不依赖同库 Session。未来为 KC、Agent Runtime、AIOps Agent 或 Model Serving 配置独立数据库、账号和连接池时，BFF 契约、AuthContext 和外部 API 不需要变化。Data Agent 暂不属于当前部署拓扑，问数继续通过 MCP Adapter 接入。
