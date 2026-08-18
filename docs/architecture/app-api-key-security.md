# App API Key 安全设计

## 目标与边界

KBot 的第三方机器访问由各业务 App 独立管理。配置文件不保存公开 Main API Key，
也不存在一个可以跨 App 使用的平台 Key。App 管理员只能在当前 App、当前 Domain 内
创建 API Client；生成的 Key 只能调用该 App 的 `/api/v1/apps/{app}/*` 公开业务接口，
不能调用平台管理、其他 App、登录换票、API Client 管理或任何 `/internal/v1` 接口。

浏览器用户继续使用 Main API 签发的短期用户 Token。App API Key 只交付给第三方应用
的服务端，不能写入浏览器 JavaScript、URL、日志或客户端安装包。

## 数据模型

- `KBOT_APP_API_CLIENT`：固定绑定 `app_id`、`domain_id`、服务账号、状态和速率上限；
- `KBOT_APP_API_CREDENTIAL`：保存公开定位 ID、HMAC 摘要、有效期、撤销时间和最近使用时间；
- `KBOT_APP_API_CLIENT_SCOPE`：声明允许的机器操作，例如聊天、会话读取和 Run 读取；
- `KBOT_APP_API_CLIENT_AGENT`：声明可访问的 Agent 白名单。

明文 Key 采用 `kbot_ak_<credential-id>.<secret>` 格式，只在创建或轮换成功响应中显示一次。
数据库只保存使用 `KBOT_API_KEY_PEPPER` 派生的摘要。生产环境未配置 Pepper 时 Main API
拒绝启动；开发环境才允许使用明确记录告警的开发值。

## 签发与管理

管理入口为 `/api/v1/apps/{app_id}/api-clients`，只接受绑定当前 App 与 Domain 的用户
BUSINESS Token，并要求 `{app_id}:api_key_manage`。API Key 本身不能创建、查看、轮换或
撤销 Key。创建时必须选择：

1. 当前 App 的有效服务账号；
2. 至少一个 App 支持的机器 Scope；
3. 至少一个 Agent；
4. 5 分钟至 365 天的有效期；
5. 每分钟请求上限。

轮换会在同一事务中撤销现有 Credential 并生成新 Key。停用或调用 `revoke` 会立即撤销
全部有效 Credential；停用是终态，如需恢复应创建新的 API Client。

## 每次请求的授权算法

Main API 对每次 App Key 请求执行以下检查，不缓存长期授权结论：

1. 拒绝调用者提交 `X-KBot-User-ID`、Domain、Tenant、App 或内部 AuthContext Header；
2. 解析 Credential ID，读取数据库记录并恒定时间比较 Key 摘要；
3. 检查 Credential 有效期、撤销状态与 Client 状态；
4. 检查 URL 是否属于绑定 App，且不是 `access`、`auth` 或 `api-clients`；
5. 重新读取绑定服务账号在当前 App、Domain 的实时权限，并与配置 Scope 求交集；
6. 构造固定 App、Domain、用户、Scope 和 Agent 白名单的 `APP_API_CLIENT` AuthContext；
7. 路由层继续检查精确 Scope、Agent 白名单、Domain 条件和对象归属。

因此，停用用户、移除成员角色、收回 Domain Scope、停用 Client 或撤销 Credential 都会在
下一次请求生效。列表接口也必须按 Agent 白名单过滤，不能只在详情或写操作中检查。

## Scope 与权限

机器 Scope 是比 App RBAC 更窄的一层限制。有效权限必须同时满足：Client 配置了该 Scope、
绑定服务账号仍拥有 Scope 映射的 App 权限、目标路由接受该 Scope、目标 Agent 在白名单中。
Scope 不能授予服务账号原本没有的权限。

当前开放范围是 Knowledge Retrieval、KM Asset 和 AIOps 的 Agent 读取、聊天、会话读取及
Run/引用读取等运行时能力。成员、角色、Agent 配置、数据源、资产入库、平台管理和 API
Client 管理不属于机器 Scope，默认拒绝。

## 内部服务与数据安全

第三方 App Key 永远不转发给下游服务。Main API 为每次下游调用签发 audience 绑定、短期
有效的 AuthContext JWT，并同时使用服务身份。所有 `/internal/v1` 路由仍要求内部服务凭据，
不能由 App Key 直接访问。

App Key 继承绑定服务账号的全局 `security_level`。检索只允许返回
`data.security_level <= user.security_level` 的数据；Agent、Scope 或请求参数都不能提高
该等级。

## 审计与运维

管理记录保留创建人、创建时间、状态、版本、Credential 有效期、撤销时间和最近使用时间。
应用日志不得打印原始 Authorization。发现泄露时先撤销 Client，再检查最近使用时间、访问
日志与关联 Run。当前请求频率计数由单个 Main API 进程执行；多实例生产部署应在网关或共享
限流设施中按 `client_id` 实施同等或更严格的全局限流。
