# Slack 集成

## 能力范围

Slack 通过 Main API 的 `POST /api/v1/integrations/slack/events` 接入。当前支持
Events API 的普通消息、`app_mention` 和 URL Verification；机器人消息、编辑消息
及其他事件会被安全忽略。Slack 请求使用原始正文、请求时间戳和 Signing Secret
执行 HMAC 校验，不使用 Portal API Key。

Main API 仅保真转发原始正文和 Slack 验签 Header；KM Asset 完成验签并将入站事件
写入自有 Inbox，再由 KM Asset 的独立 Slack Worker 映射为 Agent Runtime
Conversation 与 Turn。会话按 Workspace、频道、根线程和 Slack 用户隔离。Agent
执行仍由持久化 Run/Task/Artifact 完成；Slack Worker 读取最终
`GROUNDED_ANSWER`，通过 Outbox 调用 `chat.postMessage` 在线程内回复。进程重启后
可继续领取未完成 Inbox 和 Delivery。

Slack App 至少需要接收消息与 `app_mention` 的 Event Subscription，以及发送消息、
读取用户基本资料和邮箱所需的 Bot Token Scope；缺少邮箱权限时 Callback 的
`user_email` 为空字符串，但问答流程仍可继续。

## Workspace 与身份

每个 Workspace 在部署配置中固定绑定一个 Domain、Agent UUID 和安全等级。内部
调用使用 `SERVICE` 类型 AuthContext，Actor ID 为
`slack:<workspace_id>:<user_id>`，并把绑定的 Agent 放入授权 Agent 集合。Slack
提交的 Domain、Agent 或用户身份字段不会被信任。

## 外部 Callback

可选 Callback 保持 3.3 字段：`user_id`、`username`、`user_email`、
`user_question` 和 `request_time`。请求仅设置 `Content-Type: application/json`，
不附加鉴权信息。它沿用 3.3 的旁路通知语义；用户资料读取、Callback 调用或其
调试日志写入失败只记录运行日志，不中断 Slack 问答与最终回复。

## 临时调试输出

`callback_payload_log_enabled` 将实际发送给 Callback 的完整五字段报文写入
`<log_dir>/km_asset_app/slack_callback_debug.log`；`slack_reply_dump_enabled` 将
`chat.postMessage` 的完整 JSON Body 写入配置目录，默认是 `/tmp/slackmess`。
两项均默认关闭，不写入 Bot Token、Signing Secret、内部 JWT 或 Authorization
Header。调试文件包含个人资料和业务问题，只能临时开启并限制文件访问权限。

## 配置边界

`.env` 或生产 Secret 只保存 `KBOT_SLACK_SIGNING_SECRET` 与
`KBOT_SLACK_BOT_TOKEN`。`configuration/kbot.toml` 保存 Slack 功能开关、
Workspace 绑定、Callback URL、调试参数和回复展示策略。Workspace 配置中的
`signing_secret_env`、`bot_token_env` 只引用环境变量名称，禁止直接保存密钥。

`[integrations.slack.reply]` 支持配置 `assistant_name`、`max_references`、
`show_warnings`、`show_query_result_summary` 和 `show_visualization_notice`。非
`READY` 状态为防止误用始终展示，不提供关闭开关；文档安全链接机制落地前也不
提供链接开关。

## Asset问答助手回复

Slack Worker 只接受 `GROUNDED_ANSWER` / `GroundedAnswer.v1` 最终报文。回复正文
来自 `payload.answer`；非 `READY` 状态显示中文状态提示；引用严格按
`used_citation_labels` 过滤和排序，并受 `max_references` 限制。文档引用只显示
引用标签、标题和页码，查询引用只显示 Provider 与行数。警告与可视化仅输出面向
用户的摘要，不传送定位框、内部 UUID、查询明细、可视化原始数据和未经授权的
资源 URL。收到不匹配的 Artifact 类型或版本时，返回固定格式错误提示，避免泄漏
内部报文。
