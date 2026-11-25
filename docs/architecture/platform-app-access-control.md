# 平台与 App 分层授权

## 目标边界

KBot 不设 Tenant。普通账号由具有 `platform:user_manage` 权限的管理员在“平台管理 →
用户管理”统一创建；创建账号本身不授予任何管理或业务权限。进入业务 App 必须由具备
`platform:app_grant_manage` 的管理员显式分配该 App 的角色和业务空间。每个 App 至少
绑定一个业务角色；不同 App 的授权互相独立。

平台管理员负责平台用户、平台角色、App 生命周期、App-Domain 关系和每个 App 唯一的
初始管理员。App 来源仅保留给受保护的初始管理员，初始管理员只能进入所属 App。
App 管理员仍可维护本 App 的角色，不再创建用户或管理全局账号。

## 数据模型

- `KBOT_PLATFORM_APP`：App 目录和启停状态；停用后该 App 的所有业务访问失效。
- `KBOT_PLATFORM_USER`：统一身份主表；`ACCOUNT_ORIGIN` 标识来源，`OWNER_APP_ID` 记录
  App 用户归属，`IS_PROTECTED` 保护系统初始化账号。
- `KBOT_PLATFORM_USER_ROLE`：平台用户的平台角色，不带 Domain。
- `KBOT_APP_DOMAIN`：App 可使用的 Domain 白名单。
- `KBOT_APP_MEMBER`：用户的显式 App 资格；平台 Grant 和 App 用户都落在这里。
- `KBOT_APP_MEMBER_ROLE`：App 成员角色绑定，范围为全部 App Domain 或指定 Domain。
- `KBOT_APP_MEMBER_ROLE_SCOPE`：指定 Domain 范围的明细。
- `KBOT_APP_ROLE`、`KBOT_PERMISSION`、`KBOT_APP_ROLE_PERMISSION`：按 App 隔离的角色与权限。
- `KBOT_ACCESS_AUDIT_EVENT`：不依赖角色外键的持久化权限变更审计。

有效业务权限必须同时满足：用户、App、App 成员、角色绑定、角色、App-Domain 都为
`ACTIVE`，请求 Domain 落在绑定范围内，并且角色包含目标权限。

## 管理职责

平台接口统一位于 `/api/v1/platform`：

- `/users` 为全局用户目录，普通平台账号在这里创建、修改、停用和重置密码；受保护账号的资料、状态和授权只读；
- `/roles` 维护平台角色；
- `/apps/{app_id}/status` 控制 App 生命周期；
- `/apps/{app_id}/domains/{domain_id}` 维护 App 可用 Domain；
- `/apps/{app_id}/initial-admin` 创建唯一的受保护初始管理员；
- `/apps/{app_id}/initial-admin/password` 只允许重置该账号密码；
- `/app-grant-options` 返回实际 App、角色、业务空间和当前管理员可授予的范围；
- `/users/{user_id}/app-grants/{app_id}` 更新指定 App 的全部角色与业务空间范围，或撤销该 App 访问；不影响其他 App。

平台用户抽屉只展示已有 App 授权。新增时在「授予 App 访问」中选择 App、首个业务角色与适用的空间范围。
已有授权可逐角色新增、编辑范围或撤销；KBot 不单独保留无角色的 App 访问，移除最后一个角色须撤销整个 App 访问。

App 接口统一位于 `/api/v1/apps/{app_id}`：

- `/roles` 维护本 App 自定义角色；更新、删除使用 `If-Match`，删除会同步物理删除成员绑定。

拥有 `platform:app_grant_manage` 但没有 `platform:app_manage` 的平台管理员，仅能维护
权限不超出该 App 基础进入权限的角色授权。平台授权不允许自授或修改受保护账号。

所有业务 App 固定使用 `app_user`、`app_manager`、`app_admin` 三个系统角色。专业职责使用
App 内自定义角色；自定义角色不能获得成员或角色管理权限。完整的角色层级、删除、审计和
升级规则见 [App 权限设计规范](app-authorization-standard.md)。

## 认证流程

平台控制台使用 `POST /api/v1/auth/platform/login`，Token 不含 App 或 Domain。业务入口
先用账号密码查询 `/api/v1/auth/apps` 和 `/api/v1/auth/apps/{app_id}/domains`，再调用
`/api/v1/auth/apps/{app_id}/login`。业务 Token 固定一个 App 和一个 Domain，切换 Domain
只能在当前 App 的有效授权范围内进行。

平台来源用户完成平台登录后，可使用 PLATFORM Token 调用 `GET /api/v1/auth/entries`
免密读取其显式授权的 App 与 Domain，再调用 `POST /api/v1/auth/exchange` 换取目标业务
Token。Exchange 必须重新校验 App Grant 和 Domain Scope，不因全局管理员身份绕过业务
授权。有效 Token 可通过 `POST /api/v1/auth/refresh` 续签相同上下文；已过期或已撤销
Token 不能续签。

账号启停、重置密码和删除在平台用户目录行内操作；授权抽屉单独维护安全等级、App 角色和业务空间范围。
平台用户管理不登记个人飞书身份。已登录的用户从工作台的账号菜单进入“我的通知设置”，使用
`PUT /api/v1/notification-delivery/me/bindings/feishu` 绑定、换绑或解绑本人身份，随后本人完成验证码验证。
平台来源与 App 来源的根会话均支持，不需要选择 App 或业务空间。

用户停用、密码重置、成员或角色失效、Domain 移出 App、App 停用都会在下一次请求校验
时使既有 Token 失效。调用者提交的 App、Domain、Actor 或角色声明不能替代服务端授权。

## 受保护账号

全局 `ADMIN` 只能由项目初始化创建，并固定为平台来源的受保护用户。每个 App 最多一个
受保护初始管理员，只能通过平台接口创建。受保护账号不能删除、停用、改名或变更角色，
初始 App 管理员的密码可由具有 `platform:app_manage` 权限的管理员在用户目录重置，
并要求该账号下次登录时修改密码。全局 `ADMIN` 只能由本人验证当前密码后修改密码，
不能由其他管理员重置。初始 App 管理员随父 App 停用而失效，重新启用 App 后恢复。

## Schema 与升级

空环境由规范 DDL 和 Bootstrap 初始化。既有 KBot 4.0 Schema 的角色模型升级使用
`database/oracle/operations/upgrade_app_authorization_roles.sql`；执行前必须先备份和运行
`scripts/db/audit_app_authorization.py`，执行后再次审计。迁移不重置密码，也不自动给成员
分配 `app_manager`。

已有普通 App 自建账号使用 `database/oracle/operations/upgrade_platform_user_management.sql`
纳入平台管理，保留原账号 ID、密码、飞书身份和 App 授权。受保护的 App 初始管理员不变。
