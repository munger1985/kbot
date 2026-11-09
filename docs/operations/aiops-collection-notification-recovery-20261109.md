# AIOps 采集、通知与运行投影修复（2026-11-09）

## 范围与实现

- Ammolite 的 Oracle 工作负载诊断正文含空字符时，PostgreSQL JSONB 拒绝写入。仅对不透明 Artifact 正文采用可逆存储编码，读取还原原始证据；普通 JSON 保持原形，编码标记碰撞也可往返。原始证据内容、Hash 与字节数不变，物理列仍为 JSONB，无需 DDL。
- 两项目的 SQL 检索/展示摘要显式转义空字符，避免后续 TEXT 保存失败；不修改原始诊断结果。
- Ammolite 创建任务关注人及解析 INITIATOR/USER 收件人时，检查用户真实存在、启用及当前租户资格；失效历史身份仅保留为发起审计，不创建占位用户。
- 两项目仅把权威接口明确返回的不可重试 404 OPS_RESOURCE_NOT_FOUND 判为来源不可用。旧运行投影标为 EXPIRED，待办标为 CANCELLED，注明原执行结果未知；权限、超时、503 或普通 404 不关闭投影。写回保留作用域、资源、版本和活动状态 CAS 护栏。
- Webhook 故障来自 10.0.0.195 的独立签名桥接器，目的地为 kbotdev，但使用了当前源不匹配的旧密钥。同步当前有效源的签名配置，保留配置及 Secret 回退副本，仅重启该签名桥接器，未轮换主源或放宽鉴权。2026-11-09 05:42:05 UTC 线上真实投递已从 404 恢复为 202。

## 本地验证

- Ammolite 专项回归：194 passed。
- KBot 专项回归：189 passed。
- 两项目架构边界检查通过；Ammolite PostgreSQL 初始化基线检查、KBot Oracle 建库脚本检查通过。
- Ammolite 旧 AIOps 全表清单验收尚未纳入既有 App 私有知识表；本地实际 PostgreSQL 与当前全量 ORM 亦有既有漂移，故两项全量验收未通过。本次不执行全库初始化或改写其他知识架构 Schema。Artifact 物理类型仍为 JSONB，线上将额外核对真实驱动往返。
- 本轮未修改 Prompt，保留上一轮已同步的 19 项基线和管理员版本。

## 部署验收

- 功能提交：Ammolite `5076940b`（dev）；KBot `0bffdf8a`（kbot4.1.1），均已推送并在 kbotdev fast-forward 拉取。
- 2026-11-09 05:43:05 UTC 定向重启 Ammolite AIOps/Main API；05:43:08 UTC 按官方脚本全量重启 KBot。两套 Main API 与各四项 AIOps readiness 共 10 项全部 200；新进程 cwd 均指向对应仓库。
- Ammolite 两个受影响 Oracle 目标在新进程启动后连续产生 READY 工作负载快照；本轮取样 5 条 Oracle 语句证据持久化成功，逐条 content_hash、byte_size 核验一致。真实 PostgreSQL 驱动完成含 NUL、中文、嵌套键的只读往返校验。
- Ammolite 新通知事件已 published，error 为空；最后一条真实采集错误为 05:42:45 UTC，最后一条真实通知 Outbox 错误为 05:42:59 UTC，均早于本次重启。历史失败事件保留为审计，不批量重放、删除或伪造成功。
- KBot 7 条来源缺失运行投影于 05:45:20–05:45:22 UTC 被正常后台对账标为 EXPIRED / NOTIFICATION_SOURCE_UNAVAILABLE；专项复查候选数为 0，最早 3 条原故障记录均核对一致。原 Run 未被创建、修改或删除。
- KBot 重启后观测到 15 条 Webhook Inbox 为 PROCESSED；旧路由的真实请求已返回 202。10.0.0.195 的唯一配置和两个 Secret 均有 05:41:47 UTC 的受限权限回退副本，未改仓库源码。
- 当前线上日志 API 复查未见新的上述四类失败。对无时间戳的旧 traceback 片段，以原始日志头核对时间，不把文件 mtime 当作新故障发生时间。
- 本轮不需要 Schema 变更，也未修改 Prompt、清理历史或覆盖线上未跟踪配置备份。
