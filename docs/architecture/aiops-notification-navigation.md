# AIOps 通知定位与会话边界

通知必须携带确切业务空间和资源标识，不能用当前页面空间、默认空间或字符串拼接猜测目标。

- Ammolite 运行、案例复核、变更方案及告警事件由生产端在 `route_params` 写入 `domain_id`；运行使用 `run_id`，告警使用 `situation_id`。通知先经统一工作台重新确认空间授权、获取 BUSINESS 会话，再进入该空间的运行或告警详情。
- KBot4 从已隔离的收件箱记录和可信 Outbox 信封恢复同一组白名单定位，只公开确切资源 ID；当前授权空间必须同时匹配收件记录与定位参数，再进入 `runs.html?runId=...` 或 `situations.html?situationId=...`。
- 缺少定位、标识损坏、空间已撤权时不得跳往其他空间、管理控制台或任意 URL。报告、输出任务和案例版本继续使用各自确切产物定位。
- Ammolite 的未知地址使用独立 404 页面，不挂载管理外壳。控制台初始化的 401 仅触发入口恢复；有效 SSO 不因 TENANT 入口拒绝而退出。TENANT 换票必须使用 SSO Bootstrap 返回的当前租户，只有主会话确实失效才要求重新登录。

本修复不改变数据库结构、不回填历史通知。缺少业务空间的历史通知安全拒绝跳转；不推断其应属空间。

验证入口：两仓库的 `tests/unit/aiops_agent/test_aiops_output_notifications.py`；KBot4 的 `tests/unit/main_api/test_aiops_notification_routes.py`、`tests/contract/test_aiops_notification_links.py`；Ammolite Portal 的 `tests/aiops-notification-links.spec.mjs` 与 `tests/notification-session-recovery.spec.mjs`。浏览器回归使用真实前端页面与隔离 API fixture，不代替线上会话验收。
