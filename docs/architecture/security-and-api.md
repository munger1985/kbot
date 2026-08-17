# 身份、Domain 与 API

## Domain 与 APEX

Domain 是 KBot 的强隔离边界。Portal 登录成功后，通过受信请求头把 Domain 和
用户上下文传给 Main API。Main API 校验 Portal API Key 后构造 AuthContext；下游
服务只信任该上下文，不信任普通业务参数中的 Domain、Actor 或授权声明。

Main API 拥有平台用户、App Role、Permission、Role-Permission 和 Domain 内成员
角色。知识检索用户拥有 `knowledge_retrieval:use` 后，可以读取和使用当前 Domain
内全部 ACTIVE Agent；`knowledge_retrieval:agent_manage` 仅控制 Agent 的创建、编辑
和状态维护。AIOps 私有 Agent 仍要求额外 Agent Grant。所有授权都不能跨越 Domain
数据隔离边界。

平台用户还具有 `MAX_SECURITY_LEVEL` 数据可见上限，取值为 `0–3`。普通用户默认
等级为 `1`，保留账号 `ADMIN` 为最高等级 `3`。知识检索 Run、普通会话、多模态会话
和 KM Asset 会话都由 Main API 从用户主数据读取该值，按“用户上限、Agent 上限、
请求等级”三者最小值生成下游检索等级。浏览器参数只能缩小本次检索范围，不能提高
账号权限；Agent Runtime 和 KC 只接收 Main API 计算后的受信等级。

## 公开认证

Main API 支持两种公开用户入口：

- APEX/Portal 后端保存预配置的 `kbot_sk_...` API Key，并通过可信请求头传递用户与
  Domain；API Key 不能写入浏览器 JavaScript。
- KBot 普通页面先调用 `POST /api/v1/auth/domains` 获取账号可访问的 Domain，再调用
  `POST /api/v1/auth/login` 换取绑定用户、Domain 和密码版本的短期 JWT。用户停用、
  成员关系失效或密码被重置后，既有 JWT 会在下一次请求时失效。

`GET /api/v1/auth/me` 返回当前用户、Domain 和成员关系，
`POST /api/v1/auth/switch-domain` 切换 Domain，`POST /api/v1/auth/password` 修改本人
密码。KM 登录入口继续固定选择 `km_portal`，但签发相同的平台用户 Token，不再维护
第二套身份协议。业务服务不能信任外部直接提交的 actor、Domain 或内部身份头。

平台管理接口统一位于 `/api/v1/admin`：

- `GET /users`：平台管理员分页查询用户；
- `POST /users`：平台管理员或当前 Domain 的应用成员管理员创建普通用户；
- `GET/PATCH/DELETE /users/{user_id}`：查看、修改或物理删除普通用户；
- `POST /users/{user_id}/password`：重置密码；
- `PUT /users/{user_id}/memberships/{app_id}/{role_code}`：请求体通过 `domain_ids`
  一次维护一个或多个 Domain 成员关系；整批操作使用同一事务。平台管理员可以跨
  Domain 授权，应用成员管理员只能维护当前 Domain 下本 App 的普通角色；
- `GET /permissions`：查询按 App 分组的权限目录；
- `GET/POST /roles`：查询或创建应用角色；
- `PUT/DELETE /roles/{app_id}/{role_code}`：更新权限集合或逻辑删除角色。

停用用户使用 `PATCH status=DISABLED`，保留用户、登录凭据和成员关系；删除用户会在
同一事务中物理删除其全部成员关系、登录凭据和用户记录。逻辑删除角色会立即使该
角色不再参与权限计算，但保留历史成员关系和权限定义，便于审计和恢复。
应用成员管理员建号时只能使用默认安全等级 `1`；只有拥有
`platform:user_manage` 的平台管理员可以设置或修改 `max_security_level`。

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
- 用户与角色管理接口位于 `/api/v1/admin/*`。用户查询、修改、停用、删除和密码重置
  要求 `platform:user_manage`，角色目录维护要求 `platform:role_manage`。创建普通用户
  还允许当前 Domain 中拥有任一 App `member_manage` 权限的应用管理员；成员授权允许
  对应 App 的 `member_manage`，但应用管理员不得跨 App 或跨 Domain。平台 `ADMIN`
  用户及各 App 的 `system_admin` 角色只能由初始化脚本维护；管理接口只能重置 ADMIN
  密码，不能创建、停用、改名或变更其角色。

## 版本规则

产品版本与 API 版本独立。KBot 4.0 的首个公开接口仍为 `/api/v1`，内部接口为
`/internal/v1`。只有同一契约发生不兼容变化且新旧版本必须并存时才增加 `v2`。
健康检查可使用不带版本的 `/healthz` 和 `/readyz`。

耗时操作返回 `202` 和资源 UUID，状态通过查询或 SSE 获取。外部 DTO 不泄漏
SQLAlchemy Entity、Oracle `RAW(16)` 或内部 Lease Token。OpenAPI 冻结快照位于
`docs/openapi/`。
