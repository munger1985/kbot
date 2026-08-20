# Slack 集成

## 能力范围

Slack 通过 Main API 的 `POST /api/v1/integrations/slack/events` 接入。当前支持
Events API 的普通消息、`app_mention` 和 URL Verification；机器人消息、编辑消息
及其他事件会被安全忽略。Slack 请求使用原始正文、请求时间戳和 Signing Secret
执行 HMAC 校验，不使用用户 Token 或 App API Key。

URL Verification 报文不保证携带 `team_id`。此时 KM Asset 使用已配置 Workspace
对应的 Slack App Signing Secret 验签，成功后由 Main API 以 `text/plain` 和
HTTP 200 原样返回 `challenge`；普通事件仍必须携带并匹配 `team_id`。

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
组装完成、发送前的原始 Slack JSON Body 写入配置目录，默认是 `/tmp/slackmess`。
原始调试报文保留 `[C1]`、`[C2]` 等内部引用标签；Worker 仅在实际调用
`chat.postMessage` 前，从发送副本的可见文本中删除这些标签。
两项均默认关闭，不写入 Bot Token、Signing Secret、内部 JWT 或 Authorization
Header。调试文件包含个人资料和业务问题，只能临时开启并限制文件访问权限。

## 配置边界

`.env` 或生产 Secret 只保存 `KBOT_SLACK_SIGNING_SECRET` 与
`KBOT_SLACK_BOT_TOKEN`。`configuration/kbot.toml` 保存 Slack 功能开关、
Workspace 绑定、Callback URL、调试参数和回复展示策略。Workspace 配置中的
`signing_secret_env`、`bot_token_env` 只引用环境变量名称，禁止直接保存密钥。

`[integrations.slack.reply]` 支持配置 `assistant_name`、`max_references`、
`show_warnings`、`show_query_result_summary`、`show_visualization_notice` 和
`km_portal_base_url`。非 `READY` 状态为防止误用始终展示，不提供关闭开关。
`km_portal_base_url` 只保存非敏感 Portal 地址；Asset 回复使用该地址与经过 URL
编码的 `asset_id` 拼接 KM Link，目标 Portal 的访问控制仍由 Portal 自身负责。
`max_references` 不截断 DOCUMENT Slack Asset Template，也不会触发旧“参考资料”
回退；无 DOCUMENT 的问数结果最多展示该数量的唯一 Asset。当问数 Asset 行或本次
实际使用的 DOCUMENT 引用数超过该值时，Slack 将本次结果标记为截断并显示
“结果超过上限，当前仅展示部分内容”。`show_query_result_summary` 仅为旧参考资料
展示配置保留。

## 查询 Workspace、Domain 与 Agent

`workspace_id` 是 Slack Events 报文的 `team_id`，不是 KBot 平台生成的 ID。系统
收到过 Slack 事件后，可查询已经出现的 Workspace：

```sql
SELECT WORKSPACE_ID,
       MIN(CREATED_AT) AS FIRST_EVENT_AT,
       MAX(UPDATED_AT) AS LAST_EVENT_AT,
       COUNT(*) AS EVENT_COUNT
  FROM KBOT_KM_SLACK_INBOX
 GROUP BY WORKSPACE_ID
 ORDER BY WORKSPACE_ID;
```

可供 Slack 绑定的有效 Domain 与 KM Asset Agent 使用以下查询。Oracle 中 Agent
UUID 使用 `RAW(16)` 保存，查询时必须转换成带连字符的 UUID 文本再填写 TOML：

```sql
SELECT D.DOMAIN_ID,
       D.NAME AS DOMAIN_NAME,
       LOWER(
           SUBSTR(RAWTOHEX(A.AGENT_ID), 1, 8) || '-' ||
           SUBSTR(RAWTOHEX(A.AGENT_ID), 9, 4) || '-' ||
           SUBSTR(RAWTOHEX(A.AGENT_ID), 13, 4) || '-' ||
           SUBSTR(RAWTOHEX(A.AGENT_ID), 17, 4) || '-' ||
           SUBSTR(RAWTOHEX(A.AGENT_ID), 21, 12)
       ) AS AGENT_ID,
       A.DISPLAY_NAME AS AGENT_NAME,
       A.STATUS AS AGENT_STATUS
  FROM KBOT_PLATFORM_DOMAIN D
  JOIN KBOT_KM_AGENT A ON A.DOMAIN_ID = D.DOMAIN_ID
 WHERE D.STATUS = 'ACTIVE'
   AND A.STATUS = 'ACTIVE'
 ORDER BY D.DOMAIN_ID, A.DISPLAY_NAME;
```

系统已经建立 Slack Thread 映射后，可以直接核对实际使用过的三元组：

```sql
SELECT T.WORKSPACE_ID,
       T.DOMAIN_ID,
       D.NAME AS DOMAIN_NAME,
       LOWER(
           SUBSTR(RAWTOHEX(T.AGENT_ID), 1, 8) || '-' ||
           SUBSTR(RAWTOHEX(T.AGENT_ID), 9, 4) || '-' ||
           SUBSTR(RAWTOHEX(T.AGENT_ID), 13, 4) || '-' ||
           SUBSTR(RAWTOHEX(T.AGENT_ID), 17, 4) || '-' ||
           SUBSTR(RAWTOHEX(T.AGENT_ID), 21, 12)
       ) AS AGENT_ID,
       A.DISPLAY_NAME AS AGENT_NAME,
       MAX(T.LAST_ACTIVE_AT) AS LAST_ACTIVE_AT
  FROM KBOT_KM_SLACK_THREAD T
  JOIN KBOT_PLATFORM_DOMAIN D ON D.DOMAIN_ID = T.DOMAIN_ID
  LEFT JOIN KBOT_KM_AGENT A
    ON A.AGENT_ID = T.AGENT_ID
   AND A.DOMAIN_ID = T.DOMAIN_ID
 GROUP BY T.WORKSPACE_ID, T.DOMAIN_ID, D.NAME,
          T.AGENT_ID, A.DISPLAY_NAME
 ORDER BY T.WORKSPACE_ID, T.DOMAIN_ID;
```

## Asset问答助手回复

Slack Worker 只接受 `GROUNDED_ANSWER` / `GroundedAnswer.v1` 最终报文。回复正文
来自 `payload.answer`，并先转换为安全的 Slack `mrkdwn`；非 `READY` 状态显示中文
状态提示。Asset 字段先从 4.0 回答中按标签确定性提取，缺少的 `asset_id`、
`asset_title`、`solution_briefing`、`author_mail`、`create_time` 从本次实际使用的
DOCUMENT 引用所对应的 `manifest.md` 白名单补齐；Manifest 暂时不可读时，按同一
Domain 和 `bundle_revision_id` 从 KM Asset 持久化元数据恢复相同白名单字段。组装
Template 时，先按原回答中的
独立加粗标题、加粗项目符号或编号条目提取 Asset 顺序，再使用该条目范围内的引用
标签和规范化 `asset_title` 匹配对应 Manifest；标题存在轻微标点或后缀差异时允许
唯一相似匹配。无加粗标题的顶层项目也会作为独立条目分析，但仅在该条目
自身可定位到唯一的已使用 DOCUMENT Asset 引用时，才从对应 Manifest 补齐标题
和字段；其他标题无法可靠匹配的项目，也仅允许回退到该条目唯一引用的 Asset。只展示成功
映射到正文条目的 Asset，歧义、未匹配以及正文未展示的候选均不追加，并记录诊断
日志。完成映射和 Asset 去重后，必须保证正文每个唯一 Asset 都有且仅有一个
Template，且顺序与正文一致；不再应用 `max_references` 截断。完整性只校验
`asset_id`、`asset_title`、`solution_briefing` 三个字段，任一缺失时本次组装失败并
进入 Worker 重试，不发送部分 Template。`author_mail`、`create_time` 为可选字段，
缺失时模板尾行保持可见内容为空，但仍通过 `asset_id` 展示 KM Link。整个过程不修改
`payload.answer`。回答原文保留，每个 Template 由分隔线、Asset Title、Solution
Briefing，以及可选“Contributor 邮箱 | 发布日期”与 KM Link 按钮组成。
当实际使用的 DOCUMENT 引用数超过 `max_references`，或结构化 QueryResult 的
`truncated=true` 时，Slack 均强制显示统一截断提示；该提示不受普通警告展示开关
影响，也不会修改原始 GroundedAnswer Artifact。
无 DOCUMENT 的 Asset 问数若模型正文只返回数量摘要或未完整列出结果标题，Slack
直接按 QueryResult 行顺序补充编号 Asset 字段；展示数量受 `max_references` 限制，
超出的行不展示并触发同一截断提示。该路径不生成 Asset Template。

Knowledge Core 的检索实现、模型或候选顺序调整不作为 Slack Template 的
展示顺序。Slack 按 `payload.answer` 中引用首次出现的顺序恢复文档，
再按正文 Asset 条目组装 Template。

KBot Artifact、Outbox 和 `/tmp/slackmess` 原始调试报文保留引用标签，以支持证据
审计和问题排查；实际发送给 Slack 的报文副本会从所有可见 `text` 字段中删除
`[C1]`、`[D1]`、`[Q1]` 等标签，且不修改 URL、按钮或原始持久化报文。

正文未识别到 Asset 时不生成 Template，也不显示“参考资料”；引用文档只能用于
补齐正文 Asset 的字段，不得独立转成 Template。若完整回复超过 Slack 单条消息的
50 个 Block 上限，Worker 按 Template 边界拆分为 `FINAL`、`FINAL_0001` 等连续
Outbox 消息，保持 Template 完整和原有顺序。警告与可视化仅输出面向用户的摘要，
不传送定位框、内部 UUID、查询明细、可视化原始数据和未经授权的资源 URL。收到不
匹配的 Artifact 类型或版本时，返回固定格式错误提示，避免泄漏内部报文。
