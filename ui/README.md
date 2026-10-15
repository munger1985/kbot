# KBot UI 页面说明

`ui/knowledge_retrieval/`、`ui/aiops/` 与 `ui/media_studio/` 分别提供知识检索、
AIOps 和多媒体创作的正式页面。浏览器只调用 Main API 的 `/api/v1` 公开 BFF，
不得直接访问内部服务、保存 App API Key 或伪造可信身份 Header。

页面使用仓库内锁定的 Marked GFM 与 DOMPurify 渲染流式 Markdown，第三方浏览器
构建产物位于 `ui/vendor/`，运行时不从 CDN 下载代码。

## 开发服务器

development 环境执行 `start_kbot.sh` 后，开发服务器同时发布：

- 运行日志：`http://<python-host>:8080/operations-logs.html`
- 知识检索：`http://<python-host>:8080/ui/knowledge_retrieval/login.html`
- AIOps：`http://<python-host>:8080/ui/aiops/login.html`
- 多媒体创作：`http://<python-host>:8080/ui/media_studio/login.html`

浏览器使用的 Main API 地址只从 `configuration/kbot.toml` 的
`[ui].main_api_base_url` 读取，由开发服务器注入 `/ui/runtime-config.js`。
`api_allowed_origins` 必须允许实际的 UI Origin。
