# 4.0 身份、安全与租户设计

## 当前认证决策

KBot 4.0 不再维护用户、密码、登录、刷新令牌或退出登录接口。用户身份认证完全由门户负责；门户后端使用预配置的 KBot API Key 调用公开 `/api/v1/*`。浏览器不得直接持有该 Key。

```text
用户登录门户
  → 门户后端校验会话
  → Authorization: Bearer kbot_sk_***
  → Main API 校验 API Key 和请求上下文
  → 签发短期内部 AuthContext JWT
  → /internal/v1/* 下游服务
```

产品版本与接口版本独立：KBot 4.0 的首个公开契约使用 `/api/v1`，内部契约使用 `/internal/v1`；只有同一接口出现不兼容变更并需并行运行时才新增 `v2`。

## API Key 生命周期

API Key 标识调用 KBot 的受信门户或集成系统，不代表最终用户。Key 格式为 `kbot_sk_{key_id}.{secret}`，明文仅在创建时返回一次。当前预配置方案使用部署级 Pepper 计算 HMAC-SHA256 摘要；KBot 配置只保存 Key ID、Client ID、摘要、启用状态和可选到期时间。Key 不得写入浏览器代码、URL、配置样例、异常或日志。

设置 `KBOT_MASTER_KEY` 后，使用
`python scripts/security/generate_portal_api_key.py --key-id <id>` 离线生成 Key 与摘要；
API Pepper 由主密钥按用途派生。摘要写入单一部署文件的
`[[portal_api_keys]]`。轮换时并行配置新旧两个摘要，门户切换后停用旧记录；
不允许调用方使用同一 Key 自行增发 Key。不同环境和外部系统使用不同 Key，
便于独立吊销与审计。

## Domain 与操作人声明

门户在每次请求中通过 `X-KBot-Domain-ID` 传递 `domain_id`，通过 `X-KBot-User-ID` 传递稳定的门户 `user_id`，供聊天、AIOps 审批和审计使用。Main API 校验字段格式及 Domain 是否存在、启用，但当前阶段不重复验证用户与 Domain 的权限关系；该关系由已认证的门户保证。

`domain_id` 仍是强制数据隔离边界。Knowledge Core 只在 Collection 保存 `domain_id` 和 APEX 所需的 `app_id`，其他表通过 Collection 继承边界。Repository 的查询和写入必须显式限定 Domain，禁止仅凭资源 UUID 跨 Domain 访问。`app_id` 来自服务器配置，不接受客户端传入，也不参与业务路由。

## 内部 AuthContext JWT

Main API 校验外部 API Key 后，为单次调用签发短期、限定 audience 的内部 JWT。最小 Claim 包含：

```text
issuer, audience, issued_at, expires_at, jwt_id
request_id, trace_id
api_key_id, client_id
domain_id, asserted_user_id
```

Main API 必须移除外部请求中伪造的内部身份头。当前同 Schema 部署中，服务身份使用 `X-KBot-Internal-Token`，身份上下文使用 `X-KBot-Auth-Context`；两者必须同时通过校验。JWT 默认有效期 60 秒，每次内部请求重新签发并限定下游 audience。Knowledge Core、Agent Runtime、AIOps 以及 Model Serving 的 `/internal/v1` 不接受门户 API Key、用户密码或外部 Bearer Token。Model Serving 仅在明确的 `/api/v1` OpenAI 兼容推理入口接受独立 Model API Key，不复用 Portal Key，也不公开模型配置管理。内部接口不得挂载到公网入口或公开 OpenAPI；将来采用 mTLS 时只替换服务身份校验，不改变 AuthContext 契约。

## 当前不实现的权限能力

本阶段只做调用方认证、Domain 隔离、操作人留痕和内部服务信任，不实现角色、Scope、Collection ACL 或细粒度 RBAC。AIOps 的命令审批、目标策略、只读限制和安全闸门属于执行安全控制，仍必须实现，不能因暂缓权限系统而取消。

未来引入权限时，在 AuthContext 中增加版本化权限声明并由各资源 Owner 执行校验；公开 `/api/v1` 只有在请求或响应发生不兼容变化时才升级。

## 安全与审计

- CORS 使用环境化 Origin 白名单；API Key 调用应来自门户服务端。
- 上传文件执行大小、MIME、扩展名、恶意内容和配额检查。
- HTTP Client 设置目标 allowlist、超时与有限重试，禁止根据用户输入访问任意 URL。
- 审计至少记录 Key ID、门户用户、Domain、资源、动作、结果、来源 IP、`request_id`、`run_id` 和审批前后摘要。
- 日志、SSE 和错误响应不得包含 API Key、JWT、数据库凭据、原始 SQL 结果或内部推理文本。
