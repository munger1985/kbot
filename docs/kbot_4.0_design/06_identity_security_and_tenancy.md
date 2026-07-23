# 4.0 身份、安全与租户设计

## 信任边界

4.0 的所有外部请求先进入 Main API / BFF 的认证层；Knowledge Core、Model Runtime、Parser、DB Executor 和 MCP 不信任客户端直接提交的 `user_id`、`app_id`、`domain_id`、`security_level` 或资源 ID。它们只接受由 BFF 签发或转发的、已验证的内部身份上下文。

```text
External token / API key
  → BFF authentication
  → AuthContext (signed / internal token)
  → service authorization
  → domain policy enforcement
```

`AuthContext` 至少包含 `request_id`、主体类型、`subject_id`、租户、角色、权限集、数据安全等级、允许的 Collection/资源范围、令牌过期时间和委托链。服务端根据 Claim 派生访问范围；请求中的筛选字段只能进一步缩小范围，不能扩大权限。

## 租户与资源授权

4.0 保持单一 APEX Schema，但所有访问路径必须有明确的 domain 边界。Knowledge Core 只在 Collection 上持久化 `domain_id`（以及 APEX 视图需要的 `app_id`）；Bundle、Document、Evidence 和 Job 通过 Collection 继承范围，不在每张 KC 表重复保存租户字段。跨 domain 引用一律拒绝。

采用 RBAC + 资源属性策略：

| 场景 | 强制策略 |
| --- | --- |
| Discovery/Evidence | `tenant_id`、Collection ACL、安全等级、文档状态和调用者角色同时过滤 |
| 文件下载 | 检查 Document Version 权限后才签发短时下载 URL；不暴露服务器物理路径 |
| Agent/Skill | Policy Engine 用 AuthContext 限定可见 Skill、数据范围、模型和预算 |
| DB Executor | 目标实例、SQL 类型、环境、变更窗口和操作者权限均须校验 |
| 管理操作 | 采用最小角色、审计和必要时四眼审批 |

Repository 必须接收已解析的 `AccessScope` 或由 Application Service 注入的强制过滤条件；禁止依赖调用者“记得加 where tenant_id”的约定。所有按 ID 查询也必须校验租户与授权范围。

## 服务身份与密钥

外部用户令牌、服务间令牌和数据库凭据必须区分。服务间调用使用短期内部令牌或 mTLS 身份，包含调用服务名与目标 audience；不得复用用户 Bearer Token 或把固定共享 token 写入代码、Skill 或日志。

密钥只从受控环境变量或 Secret Manager 加载，配置文件仅保存 Secret 引用。定义轮换、吊销、过期告警和紧急替换流程。模型 API Key、Oracle 凭据、Slack Signing Secret、MCP 令牌和外部监控凭据均纳入同一密钥清单。

## 防护与审计

- CORS 使用环境化 Origin 白名单；不得在携带凭据时使用通配 Origin。
- 上传文件先执行大小/MIME/扩展名一致性检查、恶意内容扫描和配额检查，再进入持久化任务。
- 所有 HTTP client 设置连接/读取/总超时、重试边界与目标 allowlist；禁止 Skill 根据用户输入拼接任意 URL。
- Agent 输入实施注入、越权、敏感数据和不相关请求拦截；输出实施 PII/凭据泄露扫描。
- 审计事件不可由业务日志替代，至少记录认证主体、租户、资源、动作、结果、来源 IP、`request_id`、`run_id`、审批和变更前后摘要。

## 数据分类与保留

为附件、Evidence、会话、用户画像、运行 Artifact 和日志定义分类（公开、内部、敏感、受限）及允许处理位置。Memory 只保存实现产品目标所必需的结构化摘要和用户显式确认的偏好；不持久化原始内部推理文本。删除请求需级联覆盖数据库记录、对象存储、索引、缓存和异步任务，并保留最小化的合规审计记录。
