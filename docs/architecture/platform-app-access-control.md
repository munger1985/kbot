# 平台与 App 分层授权

## 目标边界

KBot 的身份分为 `PLATFORM` 与 `APP` 两种来源。平台来源用户由平台管理员创建，默认只能
操作平台资源；App 来源用户由所属 App 的管理员创建，只能进入该 App。平台来源用户如需
参与业务，必须获得显式 App Grant。两类用户进入 App 后使用同一套成员、角色和 Domain
范围判定，不存在隐式全局业务权限。

平台管理员负责平台用户、平台角色、App 生命周期、App-Domain 关系和每个 App 唯一的
初始管理员。App 初始管理员负责本 App 的用户、角色与 Domain 授权。平台管理员不代替
App 管理员维护普通 App 用户。

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

- `/users` 与 `/roles` 只维护平台来源用户和平台角色；
- `/apps/{app_id}/status` 控制 App 生命周期；
- `/apps/{app_id}/domains/{domain_id}` 维护 App 可用 Domain；
- `/apps/{app_id}/initial-admin` 创建唯一的受保护初始管理员；
- `/apps/{app_id}/initial-admin/password` 只允许重置该账号密码；
- `/users/{user_id}/app-grants/{app_id}` 为平台用户显式授予或撤销 App 角色与 Domain 范围。

App 接口统一位于 `/api/v1/apps/{app_id}`：

- `/members` 创建和查询该 App 来源用户；
- `/members/{user_id}/role-bindings` 一次设置角色与全部或选定 Domain 范围；
- `/members/{user_id}` 仅允许修改、停用或删除普通 App 用户；
- `/roles` 维护本 App 自定义角色；更新、删除使用 `If-Match`，删除会同步物理删除成员绑定。

平台用户的显式 App Grant 仍由平台管理员维护，App 管理员不能改变平台账号归属。App
管理员只能分配自己实际拥有的权限，不能通过创建角色实现权限提升。

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

用户停用、密码重置、成员或角色失效、Domain 移出 App、App 停用都会在下一次请求校验
时使既有 Token 失效。调用者提交的 App、Domain、Actor 或角色声明不能替代服务端授权。

## 受保护账号

全局 `ADMIN` 只能由项目初始化创建，并固定为平台来源的受保护用户。每个 App 最多一个
受保护初始管理员，只能通过平台接口创建。受保护账号不能删除、停用、改名或变更角色，
但允许重置密码；初始 App 管理员随父 App 停用而失效，重新启用 App 后恢复。

## Schema 与升级

空环境由规范 DDL 和 Bootstrap 初始化。既有 KBot 4.0 Schema 的角色模型升级使用
`database/oracle/operations/upgrade_app_authorization_roles.sql`；执行前必须先备份和运行
`scripts/db/audit_app_authorization.py`，执行后再次审计。迁移不重置密码，也不自动给成员
分配 `app_manager`。
