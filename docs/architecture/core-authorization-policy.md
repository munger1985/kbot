# KBot 核心授权规则

KBot 的全部业务 App 必须使用同一套授权模型：App 成员资格、App 角色权限和 Domain
范围共同决定人类用户能否进入业务能力。每个 App 必须注册 `{app_id}:use` 和系统
`user` 角色；任何系统角色或自定义角色只要包含其他业务权限，就必须同时包含 `use`。
角色管理服务和公开请求入口会分别在配置时与运行时强制执行该规则。

授权判定只有三条固定链路：

- 人类业务请求：可信身份 → App 成员与角色 → Domain 范围 → `{app_id}:use` → 端点权限
  → 资源 Domain、状态和绑定校验。
- 机器业务请求：有效 API Client → 绑定服务账号权限 → Scope → Agent 白名单 → 资源
  Domain、状态和绑定校验。
- 平台治理请求：平台身份 → `platform:*` 权限。`platform` 是不可分配业务成员的治理平面，
  不属于业务 App，不建立 Domain Agent 或 `{app_id}:use` 入口。

人类用户与机器客户端使用不同的 Agent 规则：

- 人类用户拥有 `{app_id}:use` 后，可读取和使用当前 Domain 的全部 ACTIVE Agent；
  `agent_manage` 只负责创建、编辑和查看非 ACTIVE 生命周期状态。
- App API Client 必须同时满足绑定服务账号权限、机器 Scope 和
  `KBOT_APP_API_CLIENT_AGENT` 显式白名单。空白名单永远表示无权访问，不能解释为全量。
- 不再存在用户或角色级 Agent Grant 表、接口或二次授权调用。

上述 Agent 规则由 `platform_core.authorization` 的执行器直接判定。执行器同时校验
`{app_id}:use`、当前 Domain、ACTIVE、管理权限和机器白名单；即使下游错误返回跨 Domain
Agent，Main API 也必须再次拒绝。业务 App 不能在路由中自行解释这些规则。AIOps 对普通
使用者和 API Client 只返回聊天选择、Target 选择及输入能力所需字段；模型绑定、内部指令、
诊断源和执行策略只对 `aiops:agent_manage` 开放。

管理权限不能成为提权通道：平台 App Grant 管理员不得给自己授权；只有同时拥有
`platform:app_manage` 才能授予超出纯 `{app_id}:use` 的业务角色。密码重置、角色分配和
用户安全等级调整都必须遵守“操作者支配目标”规则，即目标账号的安全等级、有效 Domain
集合和各 Domain 权限集合不能超出操作者。受保护账号和 `ADMIN` 继续使用更严格的专用规则。

资源服务继续强制执行 Domain 隔离、对象所有权、资源状态和绑定关系。历史会话在 Agent
停用后仍可读取；新建或继续会话必须重新确认 Agent 为 ACTIVE、Target 为 ENABLED，且
当前 Agent 版本仍绑定该 Target。AIOps 动作策略和短期执行 Grant 属于执行安全机制，
不属于用户 Agent 授权，必须继续保留。

普通 `aiops:use` 用户读取 Target 目录时，Main API 强制限定为 ENABLED；显式请求
DISABLED Target 返回权限拒绝。Target 详情、连接信息、凭据状态、启停和绑定管理继续要求
`aiops:target_manage`。

所有业务 App 在 `platform_core.authorization` 注册唯一策略。公开 Main API 必须通过
应用级全局依赖强制完成 App 注册、Domain、`{app_id}:use` 和机器 Scope 校验；端点再通过
同一授权入口声明业务权限。权限快照在单次请求内复用，避免全局依赖和端点权限重复访问
数据库。业务入口只接受 PORTAL 用户或 APP_API_CLIENT 主体，内部 SERVICE 和其他 API
Client 不能借用公开 App 路由。机器公开路由与 Scope 映射也由同一注册表提供。

新增 App、权限或公开路由后必须通过
`tests/acceptance/check_core_authorization_policy.py`。该检查直接枚举 Main API 实际构建的
FastAPI 路由表，验证全部 `/api/v1/apps/**` 路由均挂载核心依赖、App Slug 已注册，且端点
继续声明业务权限；未知 App 或未受保护的路由一律拒绝。
