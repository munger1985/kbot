# 身份、Domain 与 API

## Domain 与 APEX

Domain 是 KBot 的强隔离边界。Portal 登录成功后，通过受信请求头把 Domain 和
用户上下文传给 Main API。Main API 校验 Portal API Key 后构造 AuthContext；下游
服务只信任该上下文，不信任普通业务参数中的 Domain、Actor 或授权声明。

Main API 把身份和授权拆成平台层与 App 层。平台用户只拥有平台角色；App 用户只归属
一个 App。平台用户需要显式 App Grant 才能进入业务 App，该 Grant 与 App 用户都通过
App Member、Role Binding 和 Domain Scope 参与统一鉴权。知识检索用户拥有
`knowledge_retrieval:use` 后，可以读取和使用当前 Domain
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
- 平台入口调用 `POST /api/v1/auth/platform/login`，只为平台来源账号签发不含 App 和
  Domain 的平台 Token。业务入口先调用 `POST /api/v1/auth/apps` 查询账号可进入的 App，
  再调用 `POST /api/v1/auth/apps/{app_id}/domains` 查询可用 Domain，最后调用
  `POST /api/v1/auth/apps/{app_id}/login` 换取绑定 App、Domain 和密码版本的短期业务
  Token。用户停用、App 停用、成员关系失效或密码被重置后，既有 JWT 会在下一次请求
  时失效。

`GET /api/v1/auth/me` 返回当前入口、用户、App 和 Domain，
`POST /api/v1/auth/switch-domain` 切换 Domain，`POST /api/v1/auth/password` 修改本人
密码。KM 登录入口继续固定选择 `km_asset` App 和 `km_portal` Domain，但签发相同的
业务 Token，不再维护
第二套身份协议。业务服务不能信任外部直接提交的 actor、Domain 或内部身份头。

平台管理接口位于 `/api/v1/platform/*`。平台管理员在这里维护平台来源用户、平台角色、
App 状态与 App-Domain 关系；为每个 App 创建唯一的受保护初始管理员并重置其密码；还可
通过 `/platform/users/{user_id}/app-grants/{app_id}` 显式授予平台用户一个 App 的访问或
管理角色。平台用户不会因创建账号而自动获得任何业务权限。

App 内管理接口位于 `/api/v1/apps/{app_id}/*`。持有该 App 业务 Token 且具有
`{app_id}:member_manage` 的 App 管理员可创建 App 来源用户、分配角色与一个或多个
Domain 范围、停用或删除普通 App 用户；`{app_id}:role_manage` 管理本 App 自定义角色。
App 管理员不能创建平台用户、跨 App 授权或提升到其自身不拥有的权限。

停用用户使用 `PATCH status=DISABLED`，保留用户、登录凭据和成员关系；删除用户会在
同一事务中物理删除其全部成员关系、登录凭据和用户记录。逻辑删除角色会立即使该
角色不再参与权限计算，但保留历史成员关系和权限定义，便于审计和恢复。
App 管理员创建用户时不能设置超过自身的数据安全等级；平台管理员创建平台用户时同样
受自身等级上限约束。

### 全局数据安全等级规则

用户发起的数据查询、全文检索、向量检索和 Agent Run/Turn 必须统一遵守以下规则：

- 用户主数据中的 `security_level`（当前持久化字段为 `max_security_level`）是该用户唯一可信的数据访问等级；
- 数据记录、文档、Chunk、Evidence 和其他可检索对象各自保存 `security_level`；
- 检索层只能返回满足 `data.security_level <= user.security_level` 的数据，高于用户等级的数据必须在数据库或检索查询阶段排除；
- 浏览器、Portal 请求正文、查询参数和上传表单不得提交或覆盖 `security_level`；
- Agent、Collection、模型和普通业务配置不得作为用户安全等级上限，也不得与用户等级取最小值；
- Main API 完成用户认证后，从用户主数据读取等级，并作为可信上下文传给下游 Runtime、Knowledge Core 和 Data Query；下游不得信任调用者自行声明的等级；
- Slack 等非用户入口必须从已验签且受控的主体绑定读取可信安全等级，不能接受消息正文中的等级字段。

该规则是 KBot 全局安全不变量。任何新增 App、Agent、检索通道、缓存、重排或聚合查询都必须在返回结果前保持相同的 `<=` 过滤语义；重排和生成阶段不得重新引入已被安全过滤的数据。

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
- 平台用户管理要求 `platform:user_manage`，平台角色要求 `platform:role_manage`，App
  生命周期和初始管理员要求 `platform:app_manage`，平台用户显式 App Grant 要求
  `platform:app_grant_manage`。
- App 用户管理要求本 App 的 `member_manage`，App 角色目录要求本 App 的
  `role_manage`。业务 Token 的 App 必须与路径 App 一致。
- `ADMIN` 和每个 App 的初始管理员都是受保护账号。`ADMIN` 只能由初始化过程创建；初始
  App 管理员只能由平台接口创建。两者均不可删除、停用、改名或变更角色，但允许通过各自
  的受控接口重置密码。停用 App 会使其所有业务 Token、App 用户和显式 Grant 立即失效。

## 版本规则

产品版本与 API 版本独立。KBot 4.0 的首个公开接口仍为 `/api/v1`，内部接口为
`/internal/v1`。只有同一契约发生不兼容变化且新旧版本必须并存时才增加 `v2`。
健康检查可使用不带版本的 `/healthz` 和 `/readyz`。

耗时操作返回 `202` 和资源 UUID，状态通过查询或 SSE 获取。外部 DTO 不泄漏
SQLAlchemy Entity、Oracle `RAW(16)` 或内部 Lease Token。OpenAPI 冻结快照位于
`docs/openapi/`。
