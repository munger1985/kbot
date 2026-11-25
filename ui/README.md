# KBot UI 页面说明

`ui/knowledge_retrieval/`、`ui/aiops/` 与 `ui/media_studio/` 分别提供知识检索、
AIOps 和多媒体创作的正式页面。浏览器只调用 Main API 的 `/api/v1` 公开 BFF，
不得直接访问内部服务、保存 App API Key 或伪造可信身份 Header。

## 工作台与系统管理

通过 `/ui/index.html` 统一登录 KBot。平台来源、App 来源账号使用同一表单，
无需选择账号类型或指定 App。工作台只展示服务器授权且已安装的 App；
有多个工作域时先选择工作域，再免密换取该 App 的业务会话。原 App 登录地址
会转到工作台；知识检索管理员仍可在有资格时创建首个工作域。

平台管理从工作台进入。具有 `platform:user_manage` 的账号可进入“用户管理”，在用户目录内打开
详情抽屉查看资料、配置安全等级与授权；配置 App 角色和工作域还需 `platform:app_grant_manage`。
账号启停、重置密码和删除在目录行内操作。个人飞书身份由本人从工作台或 AIOps 的账号菜单进入
“我的通知设置”绑定并验证。
具有 `platform:model_manage` 的账号可进入“AI 模型管理”。
此处维护平台全局模型，不需要多媒体创作的 App 或 Domain 授权；多媒体创作继续维护
本 App 的模型绑定。已有 Oracle 环境需要执行
`database/oracle/operations/upgrade_platform_model_management.sql` 更新权限种子后再验收。

页面使用仓库内锁定的 Marked GFM 与 DOMPurify 渲染流式 Markdown，第三方浏览器
构建产物位于 `ui/vendor/`，运行时不从 CDN 下载代码。

## 知识检索视觉规范

知识检索 App 直接复用 AIOps 的 `ui/aiops/css/ammolite-shared.css` 与蓝色主题 Token，
共用顶栏、浅色分组导航、账户菜单和移动端导航样式。`knowledge.css` 只负责业务组件
适配及知识问答、X 搜索、资源配置的布局，不维护独立的配色基础；认证、权限与 API
仍属于知识检索 App，不调用 AIOps 接口。

顶部“当前工作域”可直接切换 Domain。候选项通过 `GET /api/v1/auth/domains`
按当前登录用户、当前 App 的有效授权返回，不要求 Domain 管理权限；切换复用
`POST /api/v1/auth/switch-domain`，保存新会话后重载当前页面，重新获取知识库与权限。
加载或切换失败时保留原工作域，不在浏览器自行改写隔离上下文。

静态检查：`python3 tests/contract/test_knowledge_retrieval_ui_style.py`。
浏览器回归：`node tests/contract/knowledge_retrieval_ui_style.mjs`，全部 API 使用隔离数据，
覆盖 8 个页面的桌面、平板、手机布局并输出临时截图。默认复用同级 Ammolite Portal 的
Playwright 依赖，也可用 `KBOT_BROWSER_REQUIRE_FROM` 指向已安装依赖的 `package.json`。

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
- KBot 工作台：`http://<python-host>:8080/ui/`

浏览器使用的 Main API 地址只从 `configuration/kbot.toml` 的
`[ui].main_api_base_url` 读取，由开发服务器注入 `/ui/runtime-config.js`。
`api_allowed_origins` 必须允许实际的 UI Origin。
