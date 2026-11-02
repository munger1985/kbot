# KBot UI 页面说明

`ui/knowledge_retrieval/`、`ui/aiops/` 与 `ui/media_studio/` 分别提供知识检索、
AIOps 和多媒体创作的正式页面。浏览器只调用 Main API 的 `/api/v1` 公开 BFF，
不得直接访问内部服务、保存 App API Key 或伪造可信身份 Header。

页面使用仓库内锁定的 Marked GFM 与 DOMPurify 渲染流式 Markdown，第三方浏览器
构建产物位于 `ui/vendor/`，运行时不从 CDN 下载代码。

## 通用时分控件

只选择小时、分钟的字段统一使用 `ui/shared/time-picker.js` 与 `time-picker.css`，
不得在业务页面使用原生 `input[type=time]` 或复制时分弹层。容器使用 `.time-picker`，
内含 `.time-picker-input` 文本输入和 `.icon-button.time-picker-trigger` 展开按钮，
由 ES module 的 `mountTimePicker(container)` 挂载；页面销毁时调用 `destroy()`。
程序赋值后调用 `sync()`；选择确认会发出标准 input/change 事件。

统一使用 24 小时制、分钟精度、`HH:mm` 或空字符串，支持键盘输入、双列选择、
确认/取消、Esc、外部关闭、抽屉内浮层定位以及 required/disabled/readonly/min/max。
日期和完整日期时间不受影响，时区/Cron 留在业务层。
以上两个工件与 Ammolite Portal `src/shared/ui/` 同名文件保持一致；
Portal 业务页面统一使用共享 `TimePicker.vue`，后续备份计划等同样复用。

## 开发服务器

development 环境执行 `start_kbot.sh` 后，开发服务器同时发布：

- 运行日志：`http://<python-host>:8080/operations-logs.html`
- 知识检索：`http://<python-host>:8080/ui/knowledge_retrieval/login.html`
- AIOps：`http://<python-host>:8080/ui/aiops/login.html`
- 多媒体创作：`http://<python-host>:8080/ui/media_studio/login.html`

浏览器使用的 Main API 地址只从 `configuration/kbot.toml` 的
`[ui].main_api_base_url` 读取，由开发服务器注入 `/ui/runtime-config.js`。
`api_allowed_origins` 必须允许实际的 UI Origin。
