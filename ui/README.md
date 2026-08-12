# KM Asset 页面接入说明

`ui/km/` 提供 KM Asset 的正式无框架 JavaScript 页面：工作台、MetaDB、数据来源、Asset、同步任务、Agent 和智能问答。KC Collection 创建与文件上传继续使用现有 APEX 页面。

## 后端开发服务器访问

development 环境执行 `start_kbot.sh` 后，Python UI 服务会同时发布既有测试台和 KM 页面：

- 测试台：`http://<python-host>:8080/`
- KM 页面：`http://<python-host>:8080/ui/km/dashboard.html`

KM 页面在 8080 端口访问时会显示“连接设置”，默认 Main API 为同一主机的 18099 端口。填写已启用的 Domain ID 和具备 `km_asset` 角色的 User ID 后，页面使用开发认证绕过访问 Main API。`configuration/kbot.toml` 必须处于 development 环境、启用 `development_auth_bypass`，并在 `api_allowed_origins` 中允许该 8080 Origin。生产环境不会启用此适配器。

## APEX 接入

这些页面只调用同源公开 BFF 路由 `/api/v1/apps/km-asset/*`，不会调用 `/internal/v1`，也不会在浏览器保存 Portal API Key、Domain ID 或用户 ID。生产环境必须由 APEX/网关完成身份验证，并在服务端向 Main API 注入可信的 API Key、Domain 和用户上下文。

APEX 登录成功后使用 `:APP_USER`（JavaScript 中为 `apex.env.APP_USER`）作为 KBot `USER_ID`。该值区分大小写，必须在 `KBOT_PLATFORM_USER` 中为 `ACTIVE`，并在当前 Domain 的 `KBOT_APP_MEMBER_ROLE` 中拥有启用的 `km_asset` 角色。表中不保存密码，因此登录口令仍由 APEX Authentication Scheme 校验，KBot 用户表只负责登录后的准入和授权。

现有 Schema 首次启用 KM Asset 时，先在 SQL Developer 中运行 `scripts/db/bootstrap_km_asset_permissions.sql` 补充应用权限，再运行 `scripts/db/bootstrap_km_default_user.sql`。修改默认用户脚本顶部的 `KM_DEFAULT_USER_ID`，使其与实际 `APP_USER` 完全一致，然后使用 Run Script（F5）执行。两个脚本都不包含 `@@` 文件引用，可重复执行。

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
