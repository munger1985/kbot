# 身份、Domain 与 API

## Domain 与 APEX

Domain 是 KBot 的强隔离边界。Portal 登录成功后，通过受信请求头把 Domain 和
用户上下文传给 Main API。Main API 校验 Portal API Key 后构造 AuthContext；下游
服务只信任该上下文，不信任普通业务参数中的 Domain、Actor 或授权声明。

Main API 拥有平台用户、App Role、Permission、Role-Permission 和 Domain 内成员
角色。知识检索与 AIOps App 在公开 BFF 层校验 App 权限，执行私有 Agent 时还必须
同时满足 Agent Grant。用户、角色和权限不改变 Domain 数据隔离边界。

## 公开认证

Portal 保存预配置的 `sk-...` API Key，并调用 Main API `/api/v1/*`。KBot 不保存
Portal 用户密码，也不重复登录。Main API 校验 Key 摘要，从可信请求头构造用户与
Domain 上下文；业务服务不能信任外部直接提交的 actor、Domain 或内部身份头。

模型公开接口使用独立 Model API Key，不能复用 Portal Key。

Slack Events API 是 Provider 自身验签的公开集成入口，不使用 Portal API Key。
Main API 保真转发完整原始正文和验签 Header，由 KM Asset 校验 Slack 时间戳与
HMAC，再按部署配置把 Workspace 绑定到可信 Domain、Agent 和安全等级。Slack
报文中的 Domain 或 Agent 字段不作为
身份依据。

## 内部认证

Main API 调用下游时同时携带：

1. 服务凭据，证明调用进程身份；
2. audience 绑定、短期有效的 AuthContext JWT，传递用户、Domain、请求和 Trace。

内部 JWT 每次调用签发，不缓存到长生命周期 HTTP Session。Knowledge Retrieval
App、KC、Agent Runtime、Data Query、AIOps 和模型管理的 `/internal/v1/*` 不接受
Portal API Key，且不得通过公网或 APEX 代理暴露。

## 权限执行边界

- Main API 校验 App 成员角色和权限，不把浏览器提交的角色视为可信输入；
- Knowledge Retrieval App 与 AIOps App 拥有私有 Agent 和 Grant；
- Agent Runtime 只执行调用方冻结的不可变 Execution Spec，不查询 App 权限表；
- 内部 AuthContext 只携带本次调用所需身份和授权上下文，不替代资源服务的 Domain
  条件与对象所有权校验。

## 版本规则

产品版本与 API 版本独立。KBot 4.0 的首个公开接口仍为 `/api/v1`，内部接口为
`/internal/v1`。只有同一契约发生不兼容变化且新旧版本必须并存时才增加 `v2`。
健康检查可使用不带版本的 `/healthz` 和 `/readyz`。

耗时操作返回 `202` 和资源 UUID，状态通过查询或 SSE 获取。外部 DTO 不泄漏
SQLAlchemy Entity、Oracle `RAW(16)` 或内部 Lease Token。OpenAPI 冻结快照位于
`docs/openapi/`。
