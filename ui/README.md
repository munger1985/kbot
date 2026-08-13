# KM Asset 页面接入说明

`ui/km/` 提供 KM Asset 的正式无框架 JavaScript 页面：工作台、MetaDB、数据来源、Asset、同步任务、Agent 和智能问答。KC Collection 创建与文件上传继续使用现有 APEX 页面。

## 后端开发服务器访问

development 环境执行 `start_kbot.sh` 后，Python UI 服务会同时发布既有测试台和 KM 页面：

- 测试台：`http://<python-host>:8080/`
- KM 登录页：`http://<python-host>:8080/ui/km/login.html`

KM 页面的 Main API 地址只从 `configuration/kbot.toml` 的 `[ui].main_api_base_url` 读取，由 Python UI 服务注入 `/ui/runtime-config.js`。登录页不显示也不允许修改该地址；未配置时 UI 服务拒绝启动。先使用初始化脚本创建的用户名和密码登录，Main API 签发仅可用于 KM 页面的短期 Token，其余页面请求统一携带该 Token。首次登录必须修改初始密码。`configuration/kbot.toml` 的 `api_allowed_origins` 需要允许该 8080 Origin。

## APEX 接入

这些页面只调用公开 BFF 路由 `/api/v1/apps/km-asset/*`，不会调用 `/internal/v1`，也不会在浏览器保存 Portal API Key 或伪造可信用户 Header。KM 登录 Token 保存在 `sessionStorage`，关闭标签页会话后需要重新登录。Token 中的 Domain 和用户由 Main API 签名保护。

首次启用 KM Asset 时直接运行 `scripts/db/bootstrap_km_initial_admin.sql`，在 SQL Developer 中使用 Run Script（F5）执行。脚本创建用户 `kmadmin`，初始密码为 `KmAdmin@2026!`，并在全部启用 Domain 中授予 `km_asset/manager`。初始密码只供首次登录，页面会强制修改。脚本不使用输入弹窗或 `@@` 文件引用。

页面会优先复用 APEX 已加载的 `window.KBotApi.request`。如果宿主使用其他请求封装，可在页面脚本执行前配置：

```javascript
KBotKmApi.configure({
  request: (path, options) => portalRequest(path, options),
  blob: (path, options) => portalBlobRequest(path, options)
});
```

若要显示现有 APEX Collection 页面入口，由宿主注入真实 URL：

```javascript
window.KBOT_UI_CONTEXT = {
  collectionPageUrl: "真实的 APEX Collection 页面 URL"
};
```

不要把 API Key 或可信身份 Header 放入 `KBOT_UI_CONTEXT`。
