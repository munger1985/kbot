# 4.0 Main API / BFF 与 AuthContext

## 服务边界

Main API/BFF 是所有外部请求的唯一入口和协议组合层，不是业务数据 Owner。

```text
KM Portal / APEX / Integration Client
                 ↓
             Main API / BFF
     API Key · Domain · DTO · SSE
       ↙          ↓          ↓          ↘
 Agent Runtime  AIOps  Knowledge Core  Model Serving
```

Main API 负责 API Key 认证、请求校验、内部身份上下文构建、Run API、SSE 代理、错误映射和审计入口；不直接读取 Knowledge Core、Agent 或 Model Repository，不实现检索、解析、规划或模型调用。即使 4.0 共用同一 Schema，调用规则也不改变。

## AuthContext

KBot 不再提供用户密码登录或用户 JWT。门户完成用户认证后，由门户后端携带 `Authorization: Bearer kbot_sk_...`、`X-KBot-Domain-ID` 和 `X-KBot-User-ID` 调用 Main API。Main API 校验 Key、字段格式及 Domain 状态后构建不可由请求体覆盖的上下文：

```text
AuthContext {
  request_id
  trace_id
  api_key_id
  client_id
  calling_service
  domain_id
  asserted_user_id
  issued_at
  expires_at
}
```

API Key 标识受信调用系统，`asserted_user_id` 标识其声明的实际操作人。当前阶段信任门户已完成用户与 Domain 校验，不实现 Role、Scope、资源 ACL 或 KBot 本地用户目录。`domain_id` 仍是强制数据隔离边界，下游按 Domain 限定全部资源查询；`app_id` 从服务器配置读取，只用于 APEX 直连视图过滤，不接受请求参数，也不参与 Agent 路由。

## 服务间传播

门户 API Key 不直接转发到 KC、Model Serving、Agent Runtime 或 AIOps。Main API 使用服务身份调用下游，并签发短期、限定 audience 的内部 AuthContext JWT：

```text
X-KBot-Internal-Token: service credential
X-Request-ID: original request id
traceparent: distributed trace
X-KBot-Auth-Context: signed short-lived JWT
```

下游必须同时校验服务凭证以及 JWT 的签名、过期时间、issuer、audience、调用服务和 Domain。JWT 默认有效期 60 秒，Client 必须为每次请求重新签发，不能缓存到长生命周期 HTTP Session。边界层先删除客户端伪造的内部身份头，再生成可信上下文。内部接口不得出现在公网路由或公开 OpenAPI；未来部署 mTLS 时只替换服务身份验证方式，不改变 AuthContext DTO。

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

Agent Runtime 另用能力更窄的 `AIOpsDelegationClient`，只允许创建、查询和取消其自身委派的子 Run。Management 与 Delegation Client 使用不同接口集合，Root 代码中不存在审批或配置方法。

## API 版本与公开组合

产品版本与 API 契约版本独立。KBot 4.0 的首个公开契约统一使用 `/api/v1`；服务间接口统一使用 `/internal/v1`。只有同一接口发生不兼容变化且需要并行分流时才新增 `v2`，不能按产品大版本机械升级 URL。

当前已实现的 Knowledge Core 公开组合：

- `/api/v1/knowledge/collections`：Collection 管理；
- `/api/v1/knowledge/collections/{collection_key}/ingestions/km-assets`：Asset Bundle 流式入库；
- `/api/v1/knowledge/collections/{collection_key}/ingestions/user-files`：普通文件或显式 Bundle 流式入库；
- `/api/v1/knowledge/agents/{agent_id}/collection-bindings`：Agent 与 Collection 绑定管理；
- `/api/v1/knowledge/bundles/{bundle_id}`：入库、解析和索引状态查询；
- `/healthz`、`/readyz`：不版本化，只返回服务健康摘要，不泄露数据库凭据或内部拓扑。

上述 Collection、Binding、Bundle 和 Revision 资源 ID 均使用 UUIDv7 规范字符串；
Main API 不公开 Oracle `RAW(16)`、数字兼容 ID 或双层 Public UID。

随对应领域实现后再挂载：

- `/api/v1/runs` 和 `/api/v1/runs/{run_id}/events`：Run 命令、查询与 SSE；
- `/api/v1/ops/*`：AIOps Target、Run、HITL、审批、巡检和报告；
- `/api/v1/integrations/monitoring/*`：经限流和来源验证的监控事件接入；
- `/api/v1/files/{document_version_id}`：校验 Domain 后签发短时下载地址。

耗时操作统一返回 `202` 和资源标识；同步接口只用于轻量查询。请求/响应使用 Pydantic DTO，禁止直接返回 SQLAlchemy Entity。旧 `/api/kb`、旧 Agent SSE 和旧 `doc_results` 不提供兼容路由。

AIOps 的 Public/Internal/Executor 三层契约见 [27_aiops_api_and_contracts.md](27_aiops_api_and_contracts.md)。Main API 不把 AIOps Internal 路由直接暴露给客户端。

## SSE 代理规则

Main API 不重新生成 Agent 事件，只转发经过校验的版本化事件 DTO。连接建立时先检查 Run 访问范围，再从事件游标读取；断线后由客户端带 `Last-Event-ID` 续传。终端 `RUN_COMPLETED`、`RUN_FAILED`、`RUN_CANCELLED` 或 `RUN_EXPIRED` 后发送 `done` 并关闭连接。

SSE 事件中的正文必须遵守 Artifact 安全等级；大正文通过 Artifact 引用或短时 URL 返回。BFF 不把候选 CitationPack 自动当作最终 `doc_results`，最终引用只来自 Grounded Answer Artifact。

Root 委派的 AIOps 子事件先由 Agent Runtime 按 Child Cursor 幂等投影为父 Event；Main API 不把 AIOps SSE 管道拼到 Root SSE。游标过期、心跳、慢客户端和终态规则见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

## APEX 与外部适配器

APEX 可继续通过受控视图直读 Collection、Run 摘要和投影结果，但写操作和文件上传统一由门户后端使用 API Key 调用 Main API。视图使用 `app_id`、`domain_id` 过滤，业务代码不把 `app_id` 当作路由参数。完整 SQL、命令、Evidence 和 Report 正文不进入视图。KM Portal 及后续 MCP、IM、Email Adapter 均调用 `/api/v1` 契约，不直接访问业务表或内部接口。

## 错误、审计与观测

BFF 将下游错误映射为稳定错误码，例如 `AUTH_REQUIRED`、`INVALID_DOMAIN`、`RESOURCE_NOT_FOUND`、`RUN_CONFLICT`、`UPSTREAM_UNAVAILABLE`、`RATE_LIMITED`；不把内部 SQL、模型供应商错误或堆栈返回给客户端。每个请求至少记录 `request_id`、`trace_id`、API Key ID、门户用户、Domain、动作、下游服务、状态码和 `run_id`，正文、Token 和密钥不得写入日志。

## 后续拆库

Main API 与下游只依赖 Client/DTO，不依赖同库 Session。未来为 KC、Agent Runtime、AIOps Agent 或 Model Serving 配置独立数据库、账号和连接池时，BFF 契约、AuthContext 和外部 API 不需要变化。角色、Scope 和资源 ACL 留待后续权限阶段加入；AIOps 审批与执行安全闸门不因此削弱。Data Agent 暂不属于当前部署拓扑，问数继续通过 MCP Adapter 接入。

Main API 对共享 Schema 的唯一业务查询是其自有 `KBOT_PLATFORM_DOMAIN`，用于确认 Portal 声明的 Domain 存在且启用。KC 公开请求必须经 `platform_clients.KnowledgeCoreClient` 转发；Main API 不 import KC Entity、Repository 或 Application Service。Domain 由 4.0 Portal/APEX 重新创建，不读取旧 `KBOT_MD_DOMAIN`。
