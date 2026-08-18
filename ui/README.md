# KM Asset 页面接入说明

`ui/km/` 提供 KM Asset 的正式无框架 JavaScript 页面：工作台、MetaDB、数据来源、Asset、同步任务、Agent 和智能问答。KM 文档固定使用初始化脚本创建的 `km_portal/assets` Collection。

数据来源的后台自动同步开关按来源持久化，默认关闭。项目启动只会启动 Worker 进程，
不会自动轮询 MetaDB；管理员在数据来源页面开启后才会按配置间隔创建自动任务。
“立即同步”属于人工任务，始终可用。

## 后端开发服务器访问

development 环境执行 `start_kbot.sh` 后，Python UI 服务会同时发布既有测试台和 KM 页面：

- 测试台：`http://<python-host>:8080/`
- KM 登录页：`http://<python-host>:8080/ui/km/login.html`

KM 页面的 Main API 地址只从 `configuration/kbot.toml` 的 `[ui].main_api_base_url` 读取，由 Python UI 服务注入 `/ui/runtime-config.js`。登录页不显示也不允许修改该地址；未配置时 UI 服务拒绝启动。使用初始化脚本设置的用户名和固定密码登录后，Main API 签发仅可用于 KM 页面的短期 Token，其余页面请求统一携带该 Token。`configuration/kbot.toml` 的 `api_allowed_origins` 需要允许该 8080 Origin。

## APEX 接入

这些页面只调用公开 BFF 路由 `/api/v1/apps/km-asset/*`，不会调用 `/internal/v1`，也不会在浏览器保存 App API Key 或伪造可信用户 Header。KM 登录 Token 保存在 `sessionStorage`，关闭标签页会话后需要重新登录。Token 中的 Domain 和用户由 Main API 签名保护。第三方 Key 只能由管理员在“API 客户端”页生成，明文仅显示一次。

首次启用 KM Asset 时直接运行 `scripts/db/bootstrap_km_initial_admin.sql`，在 SQL Developer 中使用 Run Script（F5）执行。脚本创建固定 Domain `km_portal`、固定 Collection `assets` 和用户 `kmadmin`，固定密码为 `KmAdmin@2026!`，并仅在该 Domain 中授予 `km_asset/manager`。脚本从已有的启用模型中选择 LLM 与文本 Embedding 作为 Collection 初始模型绑定，因此执行前至少需要各配置一个启用模型。登录后可直接使用；重新执行初始化脚本会把密码恢复为该固定值。脚本不使用输入弹窗或 `@@` 文件引用。

页面会优先复用 APEX 已加载的 `window.KBotApi.request`。如果宿主使用其他请求封装，可在页面脚本执行前配置：

```javascript
KBotKmApi.configure({
  request: (path, options) => portalRequest(path, options),
  blob: (path, options) => portalBlobRequest(path, options)
});
```
