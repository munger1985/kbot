# AIOps 通知第一阶段：基础修复交付与部署检查

日期：2026-11-05。范围：KBot4 与 Ammolite Cube 同步维护。

本阶段只修复已审计的站内通知基础能力，不接入飞书/企业微信，不新增外部 IM 投递队列，不重做通知中心。本记录区分代码验证与真实环境验收；首次代码交付时未执行数据库升级。用户随后授权的服务器增量升级已经完成，详见文末；本次仍未发布应用代码或重启服务。

## 六项实施范围

| 计划项 | 本阶段实现 |
| --- | --- |
| 管理与资源权限 | KBot 隔离事件查询/重试要求人类主体和明确的 `platform:notification_manage` 权限；只返回元数据；重试与真实操作人审计同事务。关注任务不能授予其他用户的任务可见性。 |
| 真实详情与处理入口 | 通知保留确切空间、Run、Proposal、HITL、Execution 标识；在原运行详情内展示并处理原任务，不新建运行。复用审批、查询授权、人工证据、人工执行结果及效果验证能力。资源错配、撤权和无效标识安全拒绝，不猜测目标空间。 |
| 申请人与审批人分离 | 待审批事件不把申请人当收件人；消费时查当前空间的有效成员/角色，要求同时拥有 `aiops:use` 和 `aiops:proposal:approve`，排除申请人。策略不能扩大审批人集合。无合资格审批人时以 `NOTIFICATION_APPROVER_UNAVAILABLE` 隔离，不能伪装送达。审批/拒绝入口保持独立审批校验。 |
| 原始告警等级及升级 | 保留并展示 INFO/WARNING/HIGH/CRITICAL 原始等级。通知紧急度仍用平台 INFO/WARNING/CRITICAL，HIGH 映射 CRITICAL 紧急度但不篡改原级别；告警升级产生独立事实事件。 |
| 事件身份与去重 | 人工交互绑定确切 HITL，不能仅按 Run 去重；待办稳定身份不使用打开/关闭事件各自的去重键。执行通知按执行身份、版本和事件类型区分。 |
| 生命周期与待办闭环 | 补齐输入处理/取消/过期、方案失效/过期、人工执行登记和执行结果事件。同资源全部审批待办关闭不依赖结果通知收件人；终态记录阻止迟到打开事件重开待办或发送过时的可操作提醒。丢事件时由业务权威服务对账。 |

工作流业务事件与订阅投递解耦：没有 IN_APP 订阅人也保存事实事件（允许空显式受众），不等于向全员推送。结果事件包含申请人和当前真实参与者，系统身份不冒充用户。审批队列不指派给申请人；非审批人工交互仍按原任务处理。

## 生命周期语义

新增/补齐事件包括：

- `aiops.situation.escalated`。
- `aiops.diagnostic.input_resolved`、`input_expired`、`input_cancelled`。
- `aiops.proposal.expired`、`invalidated`、`manual_result_recorded`。
- `aiops.execution.started`、`succeeded`、`failed`、`cancelled`、`unknown`、`timed_out`。

执行成功只代表执行回执成功，不代表效果验证成功；UNKNOWN 不伪装成功，TIMED_OUT 明确为开始前超时。运行取消时，尚未审批的方案变为 SUPERSEDED 并关闭审批待办；运行过期时对应方案变为 EXPIRED。报告就绪或 Run 进入终态不能替代具体人工交互状态。

## 权威对账及隔离边界

`POST /internal/v1/aiops/notifications:projection` 只返回指定资源的状态、版本和最小定位，不返回业务正文。

- 沿用内部服务凭据与短期 AuthContext；必须是 SERVICE 主体并拥有 `aiops.notification.reconcile`。不伪造人类用户。
- 读取确切 Domain 下的 HITL/Proposal/Run；Ammolite 进一步核对父 Run 的 Tenant，KBot 拒绝带 Tenant 的请求。调用端核对响应原样回显的作用域，拒绝替换资源或空间。
- 应用层先读取候选，退出 UoW 后调用权威服务，再以新 UoW 写回。写回使用观察到的行版本和资源/空间边界进行 CAS，不覆盖并发新状态。
- 超时、不可达、未知状态或作用域不符均不推定任务已完成；候选延后轮转。已有输出任务对账仍保持独立。
- 对旧 Run 级输入待办，只能在该 Run 实际不存在未决的非审批 HITL 时关闭；不从报告存在推断闭环。不推断历史通知缺失的空间或具体 HITL 身份。

## 部署前提（操作说明）

KBot4：

1. 先审阅并在明确目标环境备份后执行 `database/oracle/operations/upgrade_notification_foundation_6.sql`。它补齐通知管理权限/平台管理员绑定，允许审批队列 ASSIGNEE_USER_ID 为空，并移除旧后台任务状态 CHECK，让取消/过期保持真实状态；不重置用户或业务数据。
2. 规范 DDL、manifest 和 AIOps 生成工件已同步。生成的 `rebuild_aiops_schema.sql` / `rebuild_aiops_preserve_sources_targets.sql` 是完整重建工件，**不是本阶段增量升级入口，不应为通知修复执行重建**。
3. 同步部署 Main API、通知 Worker、AIOps 服务及受影响的静态脚本，确保浏览器获取新脚本。内部投影端点不可对外公开。

Ammolite Cube：

1. 本阶段不新增 PostgreSQL 表/列，不执行本地迁移或回填。规范事件目录已更新；现有环境部署时需要受控同步目录，不能以全库初始化代替。
2. 开发环境可审阅 `scripts/development/sync_notification_event_catalog.py` 后显式指定配置和环境文件执行；脚本仅允许开发环境。非开发环境应走经授权的发布目录同步流程，不绕过环境保护。
3. 同步部署 Main API/通知消费与对账任务、AIOps 服务和 Portal 构建产物。

两平台部署后需用真实测试账号/目标验收：事务 Outbox → Worker → 收件箱 → 精确详情 → 原任务处理 → 全部相关待办关闭；同时覆盖跨空间/租户、申请人自批拒绝、撤权、告警升级、重放/乱序、服务故障后对账，以及取消/过期。此次隔离测试不代替该真实环境验收。

## 验证记录

以下为首次代码交付时的定向测试及开发机验证；后续服务器实库升级结果见下一节，不以开发机的漂移推断线上状态。

- 共同工作流合同/权威状态映射、权限边界、独立 HITL 身份、订阅与输出通知、审批和生命周期定向回归已执行。
- KBot 通知管理拒绝、隔离返回白名单、重试审计失败不提交、关注越权均有定向回归；Ammolite 稳定待办键、关闭墓碑、迟到打开抑制和 CAS 写回均有回归。
- 浏览器隔离回归 31 项通过：运行抽屉处理、原任务审批/证据提交、查询授权、错配资源/跨空间/撤权提示、Portal 会话恢复。使用真实前端脚本/组件与 API fixture，不访问线上业务。
- 最终定向回归：KBot 185 项通过，Ammolite 106 项通过；两者各含 12 个通过的子测试。覆盖新增的取消/过期闭环、申请人自批/自拒绝以及审批幂等重放身份与作用域。
- 静态校验已通过：两仓库架构与跨项目对齐、KBot 规范 Oracle DDL/manifest 与生成工件、Ammolite PostgreSQL 基线及 AIOps ORM metadata、两仓库 OpenAPI。Portal 类型检查与生产构建通过（保留既有大 chunk 告警）。
- 实库校验未通过：本地 Oracle Listener 127.0.0.1:1521 不可达，无法确认实库列定义；现有 PostgreSQL 全服务检查发现旧 Schema 漂移，包括缺少 aiops_output_job、aiops_output_job_attempt、platform_case_resource 表及其他既有列差异。本次不自动修库，这些环境不能据此判为部署就绪。
- 广泛 AIOps 回归：KBot 5171 通过、6 跳过、2 失败；Ammolite 5147 通过、2 失败。这轮广泛测试启动后又补入了取消/过期闭环测试，最终代码另以定向回归验证。
- 两个失败均为 Top SQL：`test_aiops_calculation_graph.py::test_top_sql_worker_freezes_calculation_graph_and_report_projection`（期望均值 100，实际 None）与 `test_aiops_turn_answer.py::DbaTurnAnswerTest::test_window_result_precedes_cumulative_even_when_empty`（窗口标题文案不一致）。KBot 未修改 HEAD 的隔离副本已复现两项；本阶段不修改这些无关逻辑，不能声称全量测试全绿。

后续阶段才考虑飞书优先、企业微信扩展，以及外部渠道的凭据、投递、重试、回执和卡片入口。本阶段没有外部 IM 的实现或送达承诺。

### 提交推送前复验（2026-11-05）

- 本次重新运行的定向集合：KBot 183 项、Ammolite 105 项通过，两者另各有 12 个通过的子测试；浏览器隔离回归 31 项通过。该集合不等同于此前广泛 AIOps 回归，也不覆盖全部既有测试。
- 两仓库架构边界、跨项目对齐、OpenAPI 契约检查通过；KBot Oracle 规范 DDL/manifest 和核心授权策略检查通过；Ammolite PostgreSQL 基线及 AIOps ORM metadata 检查通过。
- Portal 类型检查和生产构建通过，仍有既有大 chunk 警告。本次未复跑广泛回归中的两项 Top SQL 已知失败，也不宣称全量测试通过。
- 本次交付只提交推送第一阶段代码、测试、DDL 工件及文档；不自动拉取服务器代码、不发布或重启服务。真实通知中心由用户手工验收，通过后再继续第二阶段飞书接入。

## 服务器数据库升级记录（2026-11-05，用户授权后执行）

本节更新前文“代码交付时未部署”的状态，只记录数据库操作；应用代码仍未在本次操作中发布或重启。

目标已从运行进程环境与应用配置双重核实：

- SSH 主机：kbotdev，140.238.44.208（私网 10.0.0.190）。
- KBot4：/home/ubuntu/kbot4.0；实际应用 Oracle 位于 146.56.158.44:1521，只操作该应用配置对应的 Schema，未操作其他用户 Schema。
- Ammolite：/home/ubuntu/Ammolite；实际 PostgreSQL 为本机 127.0.0.1:5432。运行中 Main API 使用本机 19090，不能把 18090 的 KBot Knowledge Core 当成 Ammolite。
- SSH 强制 curve25519-sha256；链路另有大报文停滞，使用仅影响本次 SSH 连接的 TCP_MAXSEG=1000 转发。未修改服务器 SSH、网络或应用配置。

已执行的增量变更：

| 数据库 | 操作 | 实测结果 |
| --- | --- | --- |
| KBot4 Oracle | 按已审阅的 upgrade_notification_foundation_6.sql 补通知管理权限及平台管理员绑定 | 权限 39→40，角色权限绑定 123→124；目标权限及绑定各一条 |
| KBot4 Oracle | HITL 指派列允许 NULL；移除 CK_NOTIFY_OPER_STATUS | ASSIGNEE_USER_ID nullable=Y，旧状态约束数为 0 |
| Ammolite PostgreSQL | 按 canonical ORM 创建 platform_case_resource 及到期查询索引 | 全服务实库验收通过：220 张表、3311 列 |
| Ammolite PostgreSQL | 仅插入缺失的 AIOps 受控事件定义 | 新增 17 条，目录总数 49→66；现有事件、策略、收件箱及任务记录未清理 |

全部操作均未使用 DROP TABLE、DROP COLUMN、TRUNCATE、全库初始化或数据重建；Oracle 只移除了阻碍真实取消/过期状态的指定 CHECK 约束。PostgreSQL 结构与目录写入在同一事务中执行，设置短锁等待上限；Oracle DDL 的隐式提交以升级前快照和逐项后置校验保护。

备份与执行证据保存在服务器目录：

`/home/ubuntu/db-upgrades/notification-foundation-20261105-kzFxj0/`

- postgresql-full.dump：完整逻辑备份，23,052,035 字节，SHA256 `7510ad8f678e47ba8e32ed99f94a46d0870cbdff8dea8c2e422d8311173242ff`；已用 pg_restore --list 验证可解析，未执行恢复演练。
- oracle-affected-tables.json.gz：四张受影响表的数据与建表定义；在同一只读事务中导出，已完整解压解析。SHA256 `f6dce0a0a386d4763290fc0d1c3f53e63cc7665e3937af4dea3a83c1cb16a404`。这不是 Oracle 全库备份。
- 两份 *-backup-manifest.json 保存备份校验信息；两份 *-upgrade-result.json 保存升级前后检查结果；目录同时保存本次运维程序和 canonical 升级工件。
- 目录权限 0700，文件权限 0600。备份不包含连接密码，未把备份业务正文输出到日志。

升级事务前后核对：Oracle 用户 9、运行 552、通知 2027、后台任务 1532 均未减少；PostgreSQL 用户 3、租户 1、运行 3889、通知 24、待办 7 在本次变更前后相同。正常服务持续运行，其他时点的自然新增不视作升级改写。

本阶段通知升级后置校验全部通过；两套 Main API 的 /readyz 均返回 ready，原进程仍运行，未重启服务。升级后查询 KBot 在线运维日志 API，Main API/AIOps 的所查时间窗口内未发现 ERROR。此检查不代替真实用户工作流端到端验收。

### Oracle 仍保留的历史差异

不能把本次通知升级通过等同于全库 ORM 完全对齐：

- KBOT_AGENT_CONVERSATION.EXECUTION_SPEC_JSON 仍可空，400 条历史会话中有 35 条为 NULL。
- KBOT_DQ_RUN.AGENT_ID / AGENT_VERSION_ID 仍可空，477 条历史查询中分别有 5 条为 NULL。不能在不补齐真实来源的情况下强加 NOT NULL，更不能删除这些记录或猜测 Agent。
- 5 个既有额外列保留：KBOT_KR_AGENT_VERSION.DO_RERANK，以及 KBOT_AI_MODEL 的 SUPPORTS_X_SEARCH、SUPPORTS_IMAGE_GENERATION、SUPPORTS_RESPONSES_STREAMING、CAPABILITY_VERIFIED_AT。
- NONCE 的规范 Oracle DDL 与实库均为 RAW(12)，但 ORM/通用验收按 BLOB 期待；5 个标志列的规范 DDL 与实库均为 CHAR，而通用验收不识别 CHAR。这属于模型/验收口径问题，不能据此盲目改动凭据数据或标志列。

以上历史数据和模型口径问题未自动修改，需要后续单独核对真实业务来源后设计无损修复。未执行业务数据回填、外部通知发送或应用发布。

用户随后已授权清理上述 35 条会话与 5 条查询；第一阶段代码提交时尚未执行删除。定向清理与通知基础修复分开处理，不能将该授权扩展为清理其他业务或审计记录。
