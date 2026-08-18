# KBot 前端 SSO、Token 与权限接入手册

## 1. 适用范围与结论

本文面向 KBot Web、独立 App 页面和 Portal/BFF 开发人员，说明当前 Main API 已实现的
登录、Token、Domain、App 权限和 Agent 可见性契约。

前端必须遵守以下结论：

1. KBot 当前有两类登录上下文：平台会话和业务 App 会话。两类 Token 不能混用。
2. 业务 Token 固定绑定一个 `user_id + app_id + domain_id`。同一用户可以在浏览器会话中
   同时保存多个业务上下文，切换页面时选择正确 Token，不需要重复查询权限。
3. 权限不是组件级接口，也不是 Agent 级接口。进入一个 App 上下文时只调用一次该 App
   的 `/access`，后续路由、菜单和按钮读取前端缓存的权限快照。
4. Agent 列表接口已经执行服务端授权过滤。前端不得为列表中的每个 Agent 再请求一次授权。
5. 平台 Token 是 SSO 根会话。平台登录后通过 `/auth/entries` 免密发现入口，通过
   `/auth/exchange` 换取显式授权范围内的业务 Token，不再为每个 App 重复提交密码。
6. 当前使用“有效 Access Token 主动续签”而不是浏览器持有 refresh token。前端在过期前
   调用 `/auth/refresh` 原子替换 Token；已经过期后必须重新登录。
7. 全局 `ADMIN` 是平台管理员，不是所有业务 App 的隐式管理员。它必须先获得目标 App 的
   显式 App Grant，之后才能签发该 App 的业务 Token。

因此，“切换 App 时反复获取权限”应拆成两个问题：

- 重复调用 `/access`、逐 Agent 查 Grant、页面串行加载属于前端实现问题，应按本文缓存和
  并发规则修复；
- 平台来源账号使用 SSO 根会话完成入口发现和 Token Exchange；App 来源账号仍可使用独立
  业务登录页。两种流程都不得由前端保存密码。

## 2. 身份与会话模型

### 2.1 平台会话

调用：

```http
POST /api/v1/auth/platform/login
Content-Type: application/json

{"user_id":"ADMIN","password":"..."}
```

响应中的 `entry_kind` 为 `PLATFORM`，`app_id` 和 `domain_id` 均为空。该 Token 只用于
`/api/v1/platform/*` 等平台管理接口，不能用于业务 App 接口。

平台管理员给自己或其他平台用户授予 App 资格时，使用：

```text
/api/v1/platform/users/{user_id}/app-grants/{app_id}
```

App Grant 必须包含有效角色和 Domain 范围。平台管理员身份本身不会绕过这一条件。

### 2.2 业务 App 会话

App 来源账号或独立业务登录页使用以下流程：

```text
POST /api/v1/auth/apps
  -> 选择 App
POST /api/v1/auth/apps/{app_id}/domains
  -> 选择 Domain
POST /api/v1/auth/apps/{app_id}/login
  -> 获得绑定 App + Domain 的 BUSINESS Token
```

前两个查询和登录当前都使用账号密码。登录请求为：

```http
POST /api/v1/auth/apps/knowledge-retrieval/login
Content-Type: application/json

{
  "user_id": "ADMIN",
  "password": "...",
  "domain_id": 1
}
```

业务登录响应结构：

```json
{
  "access_token": "...",
  "token_type": "Bearer",
  "expires_at": "2026-08-18T12:00:00Z",
  "user_id": "ADMIN",
  "display_name": "Administrator",
  "entry_kind": "BUSINESS",
  "app_id": "knowledge_retrieval",
  "domain_id": 1,
  "domain_name": "default",
  "must_change_password": false
}
```

所有后续业务请求携带：

```http
Authorization: Bearer <access_token>
X-Request-ID: <每次请求的新 UUID>
```

浏览器不得发送或覆盖受信的 Actor、App、Domain、角色、安全等级或内部认证 Header。
业务 App 和 Domain 以已签名 Token 为准。

平台来源账号登录后使用 SSO 流程，不再重复提交账号密码：

```http
GET /api/v1/auth/entries
Authorization: Bearer <PLATFORM Token>
```

该接口返回当前用户具有显式资格的 App 及其 Domain。用户选择入口后调用：

```http
POST /api/v1/auth/exchange
Authorization: Bearer <PLATFORM Token>
Content-Type: application/json

{"app_id":"knowledge_retrieval","domain_id":1}
```

服务端再次校验用户状态、平台账号来源、App 状态、显式 App Grant 和 Domain Scope，
校验通过后签发 BUSINESS Token。`ADMIN` 不会因为是全局管理员而绕过 App Grant。

### 2.3 App ID 与 URL 的映射

| App | Token/权限中的 `app_id` | 公开 URL 段 |
| --- | --- | --- |
| 知识检索 | `knowledge_retrieval` | `knowledge-retrieval` |
| KM Asset | `km_asset` | `km-asset` |
| AIOps | `aiops` | `aiops` |

不要用字符串替换临时推导。前端应维护一份只读常量映射，避免下划线 ID 和连字符 URL
混用导致 `APP_CONTEXT_MISMATCH`。

### 2.4 KM Asset 固定入口

KM 独立页面可以使用：

```http
POST /api/v1/apps/km-asset/auth/login
```

该入口固定选择 `km_asset` 和 `km_portal` Domain，签发的仍是标准 BUSINESS Token，
不是另一套 Token 协议。

## 3. Token 保存、选择与失效

### 3.1 前端状态结构

推荐将 Token 按上下文保存，而不是只保留一个全局 `token`：

```ts
type EntryKind = "PLATFORM" | "BUSINESS";

type UserSession = {
  access_token: string;
  expires_at: string;
  user_id: string;
  entry_kind: EntryKind;
  app_id: string | null;
  domain_id: number | null;
  must_change_password: boolean;
};

const contextKey = (session: UserSession) =>
  session.entry_kind === "PLATFORM"
    ? `${session.user_id}:PLATFORM`
    : `${session.user_id}:${session.app_id}:${session.domain_id}`;
```

至少维护：

- `sessionsByContext`：上下文到 Token 的映射；
- `activeContextKey`：当前窗口正在使用的上下文；
- `profileByToken`：`/auth/me` 的结果；
- `accessByContext`：App 权限快照；
- `inFlightByKey`：正在进行的 `/me`、`/access` 等 Promise，用于合并重复请求。

优先把 Token 保存在内存中。如果产品要求刷新页面后保持登录，可使用
`sessionStorage`；不建议使用 `localStorage`。无论采用何种方式，都不得把 Token 写入
URL、日志、埋点、错误详情或前端监控事件，也不得保存用户密码。

### 3.2 Token 有效期

当前 JWT 默认有效期为 28,800 秒，即 8 小时；部署配置允许 300 至 86,400 秒。前端以
登录响应的 `expires_at` 为准，不得写死 8 小时。

可以在本地提前 60 秒把 Token 标记为即将过期，停止发起新的长任务并提示重新登录。
服务端签名和会话校验始终是最终依据。

### 3.3 主动续签 Token

当前不向浏览器签发长期 `refresh_token`。在 Access Token 仍有效时调用：

```http
POST /api/v1/auth/refresh
Authorization: Bearer <当前 Token>
```

服务端会重新校验用户、密码版本和业务授权，并签发相同入口上下文的新 Token。推荐在
`expires_at` 前 5 分钟续签，所有并发请求共享同一个 refresh Promise。续签成功后先原子
替换会话，再放行业务请求。因此：

- 不要定时调用登录接口或重复提交密码；
- 不要为了静默刷新而保存或读取密码；
- 不要在 401 后无限重试；
- Token 已过期、被撤销或无效时不能再续签，应清理对应上下文并回到登录入口；
- 修改密码接口会返回一个新 Token，前端必须原子替换当前上下文中的旧 Token。

### 3.4 服务端会实时复核的状态

JWT 未到期也不代表会话一定继续有效。每次认证请求都会检查用户状态、密码版本，以及
业务会话当前 App/Domain 资格。以下变更会让现有会话在下一次请求失效：

- 用户被停用；
- 密码被修改或管理员重置；
- App 被停用；
- App Member、角色绑定或 Domain 范围失效；
- Domain 被停用或移出 App。

权限角色发生变化时，业务接口会立即按数据库中的最新权限判定，而不是信任前端缓存。
前端缓存只控制界面展示和减少查询，不能构成安全边界。

## 4. 权限快照的正确获取方式

### 4.1 每个业务上下文只获取一次

| App | 权限快照接口 |
| --- | --- |
| 知识检索 | `GET /api/v1/apps/knowledge-retrieval/access` |
| KM Asset | `GET /api/v1/apps/km-asset/access` |
| AIOps | `GET /api/v1/apps/aiops/access` |

统一响应：

```json
{
  "app_id": "aiops",
  "domain_id": 1,
  "user_id": "ADMIN",
  "roles": ["admin"],
  "permissions": ["aiops:agent_manage", "aiops:use"]
}
```

前端把 `permissions` 转为 `Set<string>`，路由、菜单、按钮和编辑状态都读取同一个快照：

```ts
const can = (permission: string) =>
  authStore.activeAccess?.permissions.has(permission) === true;
```

禁止以下模式：

- 每个页面组件各自调用一次 `/access`；
- 菜单、按钮和表格行分别查询权限；
- 进入 Agent 详情时重新查询完整权限；
- 先查询角色，再逐角色查询权限；
- 收到任意 403 就重新登录或重新获取所有 App 权限。

### 4.2 single-flight 合并并发请求

App Shell 中使用同一个 Promise 合并同时发生的权限请求：

```ts
const accessCache = new Map<string, AccessSnapshot>();
const accessFlights = new Map<string, Promise<AccessSnapshot>>();

async function ensureAccess(ctx: BusinessContext, force = false) {
  const key = `${ctx.userId}:${ctx.appId}:${ctx.domainId}`;
  if (!force && accessCache.has(key)) return accessCache.get(key)!;
  if (!force && accessFlights.has(key)) return accessFlights.get(key)!;

  const flight = apiFetch<AccessSnapshot>(ctx.accessUrl, {
    token: ctx.accessToken,
  }).then((value) => {
    accessCache.set(key, {
      ...value,
      permissions: new Set(value.permissions),
    });
    return accessCache.get(key)!;
  }).finally(() => accessFlights.delete(key));

  accessFlights.set(key, flight);
  return flight;
}
```

实际类型可把服务端的 `string[]` 与前端缓存的 `Set<string>` 分开定义。

### 4.3 权限缓存失效时机

以下事件清除相应的 `accessByContext`，下次使用时再加载一次：

- 登录、退出或替换 Token；
- 切换 App 或 Domain；
- 当前用户修改密码；
- 管理页面成功修改 App Grant、成员、角色、角色权限或 Domain Scope；
- WebSocket/SSE/通知明确告知授权已变化；
- 当前业务请求返回 `APP_PERMISSION_DENIED`。

收到一次 `APP_PERMISSION_DENIED` 时，可以强制重载当前上下文 `/access` 并重新渲染；
原业务写请求不得自动重放。若重载后仍无权限，直接展示无权限状态。不要形成
“403 → 登录 → 查权限 → 重试 → 403”的循环。

对于没有授权变更通知的普通页面，可为权限快照设置 30 至 60 秒的软 TTL；TTL 到期只在
下一次需要权限时后台更新，不应阻塞每次路由切换。服务端仍会为每个业务请求执行真实鉴权。

## 5. Agent 页面规则

### 5.1 知识检索 Agent

- 页面入口权限：`knowledge_retrieval:use`；
- 管理能力：`knowledge_retrieval:agent_manage`；
- `GET /api/v1/apps/knowledge-retrieval/agents` 已按当前权限返回可见集合；
- 有管理权限时可看到全部 Agent；普通用户只返回 `ACTIVE` Agent。

前端只根据一次 `/access` 控制新建、编辑、启停按钮，然后直接渲染 `/agents` 的结果。
不要为每个 Agent 查询角色或 Grant。

### 5.2 AIOps Agent

- 页面入口权限：`aiops:use`；
- Agent 管理权限：`aiops:agent_manage`；
- 计划管理权限：`aiops:plan_manage`；
- `GET /api/v1/apps/aiops/agents` 会在服务端把用户 Grant 和角色 Grant 合并后过滤；
- 拥有 `aiops:agent_manage` 时服务端直接返回全部 Agent。

`GET /api/v1/apps/aiops/agent-grants` 是管理员维护授权的接口，本身要求
`aiops:agent_manage`。普通用户页面不得调用它来判断每个 Agent 是否可见。

### 5.3 KM Asset

KM 权限快照至少可能包含：

- `km_asset:use`：使用 KM 页面、会话和只读能力；
- `km_asset:source_manage`：管理来源；
- `km_asset:data_manage`：管理问数模型；
- `km_asset:agent_manage`：管理 KM Agent；
- `km_asset:operations_manage`：管理同步与运行任务。

KM `/access` 在缺少 `km_asset:use` 时直接返回 403。页面应以 App Shell 中的一次初始化
结果控制导航和按钮，不能让每个子页面重复初始化身份。

## 6. 页面加载与路由守卫

### 6.1 推荐加载顺序

首次恢复一个已有业务会话：

```text
读取上下文 Token
  -> 检查 expires_at 和上下文是否匹配目标 App/Domain
  -> 并行执行：/auth/me、App /access、当前页面首要列表
  -> 渲染 App Shell
```

`/auth/me` 每个 Token 最多调用一次，用于确认会话身份和成员摘要；普通页面切换不再调用。
`/access` 每个业务上下文最多调用一次；首要列表接口自身会做服务端鉴权，因此不必等待
`/access` 完成后才发出。可使用 `Promise.allSettled` 分别处理身份、权限和业务数据错误。

禁止形成以下瀑布：

```text
/auth/me -> /access -> /agents -> /agent-grants -> 每个 Agent 的详情/权限
```

推荐结构：

```text
                 +-> /auth/me（每 Token 一次）
Token + context -+-> /access（每上下文一次）
                 +-> /agents 或页面首要数据（并行）
```

### 6.2 路由守卫只做本地判断

路由配置声明所需上下文和权限：

```ts
const routes = [{
  path: "/aiops/agents",
  meta: {
    entryKind: "BUSINESS",
    appId: "aiops",
    permission: "aiops:use",
  },
}];
```

守卫逻辑：

1. 选择与路由 `appId`、当前 `domainId` 匹配的 Token；
2. 无 Token 时进入业务登录/上下文选择页；
3. 有 Token 但权限快照尚未加载时调用共享的 `ensureAccess`；
4. 快照存在后只做 `Set.has()`；
5. 无权限进入统一 403 页面，不发起更多权限查询。

组件不得自行跳转登录页。所有 401、403 和上下文切换由统一 Auth Store 与 API Client
协调，避免多个组件同时清理会话或发起登录请求。

### 6.3 App 和 Domain 切换

同一 App 切换 Domain：

```http
POST /api/v1/auth/switch-domain
Authorization: Bearer <当前 BUSINESS Token>
Content-Type: application/json

{"domain_id": 2}
```

响应是一个新的业务 Token。前端必须切换 `activeContextKey`，并为新 Domain 建立新的权限
快照；不得继续沿用旧 Domain 的 `/access`、列表数据、查询缓存或 SSE 连接。

切换到另一个 App 时，优先查找浏览器会话中是否已有目标
`user_id + app_id + domain_id` 的有效 Token。有则直接激活；没有则使用 PLATFORM Token
调用 `/auth/exchange`。平台 Token 不能直接传给目标 App 试探权限。

## 7. 统一 API Client 与错误处理

### 7.1 错误格式

认证中间件错误使用 `application/problem+json`：

```json
{
  "type": "urn:kbot:error:user_token_expired",
  "title": "请求认证失败",
  "status": 401,
  "code": "USER_TOKEN_EXPIRED",
  "detail": "用户登录已过期，请重新登录",
  "request_id": "..."
}
```

部分业务权限错误由 FastAPI 返回，稳定错误码位于 `detail.code`。统一客户端应兼容：

```ts
const code = payload?.code ?? payload?.detail?.code ?? "REQUEST_FAILED";
const message = typeof payload?.detail === "string"
  ? payload.detail
  : payload?.detail?.message ?? payload?.message ?? `HTTP ${response.status}`;
```

所有错误提示应保留 `request_id`，但不得展示 Token、密码、内部 Trace 或数据库异常。

### 7.2 401/403/409 处理矩阵

| HTTP / code | 前端动作 |
| --- | --- |
| 401 `USER_TOKEN_EXPIRED` | 清理当前 Token 和权限快照，进入对应登录入口 |
| 401 `INVALID_USER_TOKEN` | 清理当前 Token，记录 `request_id`，进入登录入口 |
| 401 `USER_SESSION_REVOKED` | 清理该用户所有旧 Token，提示密码已变化 |
| 401 `USER_DISABLED` | 清理该用户所有上下文并禁止自动重试 |
| 401 `DOMAIN_ACCESS_DENIED` | 清理当前业务上下文，返回 App/Domain 选择页 |
| 401 `PASSWORD_CHANGE_REQUIRED` | 保留 Token，只允许进入修改密码页 |
| 403 `APP_PERMISSION_DENIED` | 当前上下文 `/access` 强制更新一次；仍无权限则显示 403 |
| 403 `APP_CONTEXT_MISMATCH` | 前端选错 Token；切换到与 URL App 匹配的上下文 |
| 403 `DOMAIN_CONTEXT_REQUIRED` | 当前不是有效业务 Token；返回上下文选择页 |
| 403 `APP_ACCESS_DENIED` | 账号没有目标 App 资格；提示管理员配置 App Grant |
| 403 `APP_DISABLED` | 清理目标 App 上下文并返回 App 选择页 |
| 409 `BUSINESS_SESSION_REQUIRED` | 平台 Token 调用了业务会话操作；选择 BUSINESS Token |
| 409 `PLATFORM_SESSION_REQUIRED` | 业务 Token 调用了 SSO 根会话操作；选择 PLATFORM Token |

401 和 403 都最多由统一客户端处理一次。写操作、对话发送和长任务创建不能自动重放，
避免重复提交。

### 7.3 SSE 与长连接

建立 SSE 时使用当前业务 Token。切换 App、Domain、退出登录或 Token 失效时立即关闭旧
连接。建立长连接前先执行主动续签；连接因 401 中断后应按错误码决定是否重新登录，
不能在 EventSource 或 fetch 重连循环中重复使用过期 Token。

## 8. 平台管理员的正确跨 App 流程

平台管理员访问业务 App 的前置条件是显式授权，不是反复查询权限：

1. 使用 PLATFORM Token 进入平台管理；
2. 确认目标 App 为 `ACTIVE`，并已关联目标 Domain；
3. 通过 App Grant 接口给平台用户配置目标 App 的角色与 Domain 范围；
4. 调用 `/auth/entries` 获取免密入口目录；
5. 使用 PLATFORM Token 调用 `/auth/exchange` 取得 BUSINESS Token；
6. 前端保存该上下文 Token，并只加载一次目标 App `/access`；
7. 以后在同一浏览器会话切回该 App 时复用仍有效的业务 Token；
8. 目标 App 或 Domain 授权变化后，使对应权限快照失效，不清理其他 App 会话。

如果平台管理员已经有目标业务 Token，页面仍每次进入都“获取权限”，说明前端没有按
上下文复用权限快照。如果页面对每个 Agent 再查一次 Grant，说明前端绕过了服务端列表
过滤契约。这两种行为都应删除。

## 9. Portal/BFF 模式

Main API 也支持 Portal 后端持有 `kbot_sk_...` API Key，并由可信服务器端传递用户和
Domain 上下文。该模式下：

- API Key 只能保存在 Portal/BFF 服务端；
- 浏览器只持有 Portal 自己的同源会话，不能接触 KBot API Key；
- BFF 为 Main API 请求注入受信 Header；
- 浏览器提交的 Actor、Domain、角色或权限不得原样转成受信 Header；
- `/internal/v1/*` 永远不能从浏览器或 BFF 公开代理。

浏览器 JWT 模式和 Portal/BFF 模式只能在边界层选择，不应在单个页面中混用两套凭据。

## 10. 当前契约边界与建议演进

以下能力当前尚未实现，前端不得假设已经存在：

- 浏览器持有的长期 refresh token；当前采用有效 Access Token 主动续签；
- 一次返回用户全部 App/Domain/权限/Agent 的全局业务数据 bootstrap；
- 授权变更推送。

当前 `/auth/entries` 与 `/auth/exchange` 已支持“一次平台登录，在有 Grant 的 App 间免密
切换”。后续可以继续增强：

1. 更长周期的安全刷新机制：使用服务端会话或旋转的 HttpOnly、Secure、SameSite Cookie，
   不把 refresh token 暴露给 JavaScript。
2. 授权版本或变更事件：前端按版本精确失效权限快照，替代固定 TTL 轮询。

长期 refresh cookie 属于新的认证契约，需要单独设计、OpenAPI、威胁建模和测试。当前
前端应使用平台根会话、受控交换、主动续签、多上下文 Token 和权限 single-flight。

## 11. 前端验收清单

- [ ] PLATFORM Token 从未用于 `/api/v1/apps/*` 业务接口。
- [ ] 每个 BUSINESS Token 的 App 和 Domain 与当前路由一致。
- [ ] Token 按上下文保存，切换 App 不会覆盖其他 App 的有效 Token。
- [ ] 平台用户通过 `/auth/entries` 与 `/auth/exchange` 免密进入已授权 App。
- [ ] 续签请求按上下文 single-flight，成功后原子替换 Token。
- [ ] 密码没有写入任何浏览器持久化、日志、Store 或请求重试队列。
- [ ] `/auth/me` 每个 Token 最多请求一次。
- [ ] `/access` 每个业务上下文只初始化一次，并有 single-flight。
- [ ] 路由、菜单、按钮共享同一个权限 `Set`。
- [ ] Agent 列表没有逐 Agent 权限或 Grant 请求。
- [ ] App Shell 的身份、权限和首要数据并行加载，不形成瀑布。
- [ ] 切换 Domain 后关闭旧 SSE，并清除旧 Domain 的业务数据缓存。
- [ ] 401、403、409 按错误码处理，没有无限刷新或重试。
- [ ] 写操作和对话发送不会因认证失败自动重放。
- [ ] 错误页面显示稳定错误码和 `request_id`，不泄漏敏感信息。
- [ ] 平台管理员无 App Grant 时显示配置指引，而不是循环获取权限。

## 12. 当前实现依据

- 登录和会话：`services/main_api/src/main_api/api/auth.py`、
  `services/main_api/src/main_api/application/user_auth.py`；
- App 权限快照：`services/main_api/src/main_api/application/access_control.py`；
- 知识检索 App：`services/main_api/src/main_api/api/knowledge_retrieval_app.py`；
- AIOps App 与 Agent Grant：`services/main_api/src/main_api/api/aiops_app.py`；
- KM Asset App：`services/main_api/src/main_api/api/km_asset_app.py`；
- 认证中间件与错误格式：`packages/platform_core/src/platform_core/security/middleware.py`；
- 平台与 App 授权模型：`docs/architecture/platform-app-access-control.md`。
