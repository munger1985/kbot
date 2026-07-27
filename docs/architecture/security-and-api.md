# 身份、Domain 与 API

## Domain 与 APEX

Domain 是 KBot 的强隔离边界。Portal 登录成功后把可信 Domain 和用户上下文传给
Main API；后台资源不能跨 Domain 查询。`app_id` 是 APEX 直连页面所需的部署常量，
由配置固定，不作为普通业务参数在服务之间自由传递。

当前 4.0 保留认证和安全等级过滤，暂不实现完整角色权限。未来权限声明加入
AuthContext，由资源所属服务执行，不改变 Domain 边界。

## 公开认证

Portal 保存预配置的 `sk-...` API Key，并调用 Main API `/api/v1/*`。KBot 不保存
Portal 用户密码，也不重复登录。Main API 校验 Key 摘要，从可信请求头构造用户与
Domain 上下文；业务服务不能信任外部直接提交的 actor、Domain 或内部身份头。

模型公开接口使用独立 Model API Key，不能复用 Portal Key。

## 内部认证

Main API 调用下游时同时携带：

1. 服务凭据，证明调用进程身份；
2. audience 绑定、短期有效的 AuthContext JWT，传递用户、Domain、请求和 Trace。

内部 JWT 每次调用签发，不缓存到长生命周期 HTTP Session。KC、Agent Runtime、
AIOps 和模型管理的 `/internal/v1/*` 不接受 Portal API Key，且不得通过公网或
APEX 代理暴露。

## 版本规则

产品版本与 API 版本独立。KBot 4.0 的首个公开接口仍为 `/api/v1`，内部接口为
`/internal/v1`。只有同一契约发生不兼容变化且新旧版本必须并存时才增加 `v2`。
健康检查可使用不带版本的 `/healthz` 和 `/readyz`。

耗时操作返回 `202` 和资源 UUID，状态通过查询或 SSE 获取。外部 DTO 不泄漏
SQLAlchemy Entity、Oracle `RAW(16)` 或内部 Lease Token。OpenAPI 冻结快照位于
`docs/openapi/`。
