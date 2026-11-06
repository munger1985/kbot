# AIOps 飞书通知：部署准备与验证

初版：2026-11-05；Schema 升级核验更新：2026-11-06。两仓库同步实现；Schema 已升级不表示服务代码已部署或已通过真实飞书验收。

## 1. 架构与代码入口

采用平台通知模块与独立外部 Worker，复用 Main API 代码／镜像。不新增飞书 HTTP 微服务，不在 API lifespan 启动外发循环，不引入 Redis 或消息中间件。

- 入口：`python -m main_api.entrypoints.external_notification_worker`。
- 配置：部署文件顶层 `[external_delivery]`，经 typed settings 读取。
- API：`/api/v1/notification-delivery`；OpenAPI 为公开精确合同。
- 管理：KBot 分为 `ui/aiops/notification-channels.html`（通知渠道）和 `notification-delivery.html`（通知投递）；Ammolite 分为 `/tenant/governance/notification-channels` 和 `/tenant/governance/notification-delivery`。
- 个人验证：KBot 账户菜单的“我的通知设置”指向 `notification-settings.html`；Ammolite 租户账户菜单指向 `/tenant/notification-settings`。成员绑定仍在创建／编辑成员处维护。
- 2026-11-06 交互修订不改 Schema、责任组或群路由。渠道单卡无分页；投递记录默认每页 20 条，按后端 `has_more` 翻页。群登记入口暂不展示，新消息规则与试发只选择个人绑定，已有数据不清理或重写。
- 卡片详情：门户公开地址下 `/notifications/open/{entry_id}`。入口记录独立于个人收件箱。
- KBot 本地启停脚本和容器拓扑已登记 Worker；Ammolite 进程目标为 `notification-delivery`。
- Worker 独立常驻轮询任务；没有任务时不访问飞书。渠道启停与门户地址从数据库读取，页面保存后生效；只有变更进程／网关等 TOML 技术参数才需按部署流程重启对应进程。

Outbox 消费事务同时保存站内投影及外部任务。外部任务不是 Inbox 的附属队列；清除个人通知不会删除外部入口。业务状态由源服务负责，飞书 ACCEPTED 仅表示接口确定接受，不表示已读或业务完成。

## 2. 数据库工件与无损边界

KBot：

- canonical：`database/oracle/main_api/003_notification_delivery.sql`，Main API manifest v10（含通知公共设置表）。
- 已完成第一阶段的现有尚无外发表的环境使用 `database/oracle/operations/upgrade_notification_delivery_9.sql`，新增八张通知投递表、索引、限流种子和四类权限。
- 增量脚本不 DELETE、TRUNCATE、DROP 或重建已有业务表；不要用全量 `001_access_control.sql` 升级现有环境。
- 新环境用规范初始化，不能再重复执行增量脚本。Oracle DDL 隐式提交，应先核对环境、基线及部署维护窗口；kbotdev 的执行记录见 6.2。
- 定向部署入口：`scripts/db/upgrade_notification_delivery.py`，默认仅验收，显式 `--apply` 才执行增量 SQL。执行前拒绝任何已存在的目标对象；逐步记录完成语句。失败后先核对日志与元数据，不能盲目重跑。

Ammolite：

- 扩展 canonical `platform_notification_delivery_job`，沿用原物理表，不创建平行队列。
- `notification_id` 可空，关联站内记录被删除时 SET NULL；事实来源、冻结目标、独立入口、租约及尝试不依赖站内通知。
- 连接／绑定／策略等表使用 `platform_notification_external_` 前缀，与既有站内策略表区分。
- `core_schema.py`、Entity 和初始化已同步；限流种子由部署初始化写入。
- 不新增本地兼容迁移、历史回填或双读写；不得直接运行全库初始化替代升级。
- 经本次授权新增定向部署入口 `scripts/deployment/upgrade_notification_delivery.py`。从 canonical 导出计划（`--render-plan`），可以将脚本与计划送到旧部署，在推送代码前升级（`--repo-root`、`--plan-file`、`--apply`、`--report`）。
- PostgreSQL 升级在单事务内对原队列取得 ACCESS EXCLUSIVE 锁，再次确认零行及已审阅列／约束基线。若有记录或已有部分新表，立即停止；不清理、不回填。仅空表移除旧 user_id 列及其约束，新增规范列与八表，通知外键改为 SET NULL，权限只增不删。锁超时 5 秒、单语句超时 60 秒；任何执行／验收错误整体回滚。
- 不带 `--apply` 时，只读验收九张目标表的列类型、可空性、主外键、索引、种子及权限，不做全库漂移修复。

所有自身 Schema 校验只在离线验收及部署环节进行，不加入服务启动、readiness、UoW 或业务请求。

### 2.1 已有旧 Domain 范围结构的本次切换

2026-11-06 的历史 v8 升级记录见 6.2；它不代表当前范围修订已部署。发布本版前先停止通知生产者及外发 Worker，核对确切数据库，不要直接启动新代码访问旧结构。

- KBot：`python scripts/db/upgrade_notification_scope.py` 只做预检；显式 `--apply` 才移除空配置表的 Domain 列，允许验证／试发任务无 Domain，并将托管凭据 Domain 改为可空。全局通知密文使用独立 global AAD，旧业务 Domain 凭据行为不变。
- Ammolite：`python scripts/deployment/upgrade_notification_scope.py` 默认预检；`--apply` 在事务内切换空旧结构，保留 Tenant 列与全部业务对象。
- 任一旧外发表有记录即停止；KBot 存在活动旧通知密文也停止。不会删除数据、自动合并连接或改写历史业务任务。有记录时须另行审阅安全转换方案。
- 本次代码修改没有连接线上数据库、执行上述切换、拉取服务器代码或重启服务。未经本版规范验收，不应宣称现有数据库可以直接运行新代码。

### 2.2 通知公共设置表新增（本次发布必做）

先执行第 2.1 节范围升级并验收八张既有外发表，再运行：

- Ammolite：`python scripts/deployment/upgrade_notification_settings.py` 只读预检；显式 `--apply` 才新增 `platform_notification_external_settings`。
- KBot：`python scripts/db/upgrade_notification_settings.py` 只读预检；显式 `--apply` 才新增 `KBOT_NOTIFY_SETTINGS`。
- 脚本先验收既有通知结构、种子和权限，再且仅再创建一张缺失的设置表；不要求现有渠道为空，不删除、不回填、不改写任何已有记录。有旧 Domain 结构或其他漂移时拒绝执行，不能重新跑初始 v9 增量或全库初始化。
- PostgreSQL 创建及验收在一个事务内完成，失败整体回滚；Oracle DDL 隐式提交，预检在前，失败须先只读核对。已存在的设置表必须通过当前规范验收，不盲目重建。
- 完成九表规范验收后再部署代码、更新 Main API 与独立 Worker。后续数据库配置热生效不代表此次代码发布无需更新进程。不得直接打开新版代码访问缺失的新表。

## 3. 安全配置

技术运行配置示例（不再包含渠道启停或门户地址）：

~~~toml
[external_delivery]
# 默认官方端点；无外网 OP 环境可由部署管理员替换为企业 HTTPS 网关。
token_url = "https://accounts.feishu.cn/oauth/v3/token"
message_origin = "https://open.feishu.cn"
poll_seconds = 1.0
request_timeout_seconds = 10
lease_seconds = 60
max_attempts = 5
coalesce_seconds = 0
# 可选静默示例：UTC 14:00 至次日 00:00，即北京时间 22:00 至次日 08:00。
# quiet_start_utc_hour = 14
# quiet_end_utc_hour = 0
~~~

门户地址由通知渠道页面的“通知公共设置”保存到数据库，必须是无路径、无 query 的 HTTPS Origin。默认飞书服务端地址：

- 令牌：`https://accounts.feishu.cn/oauth/v3/token`。
- 消息：`https://open.feishu.cn`。

无外网 OP 场景可在 TOML 配置 token_url（完整 OAuth 路径）和 message_origin（消息网关 Origin，需转发 /open-apis/im/v1/messages）；两者可以是不同网关。允许内部 HTTPS 域名，证书必须受部署环境信任，不关闭 TLS 校验。禁止用户信息、query、fragment 和消息 Origin 路径；请求不跟随重定向。网关需保持飞书请求、状态码、业务码、回执及限流语义，不得记录凭据或 Token。此配置是部署信任边界，不允许租户 API／UI 覆盖为任意地址。

不得在 UI 填自定义服务 URL 或 Webhook。新版 OAuth 使用 client_credentials、client_secret_post、最小 scope `im:message:send_as_bot`。不增加旧接口自动回退。

App Secret 通过现有托管凭据设施加密。API 不回显已有 Secret、Token、密文或内部凭据引用：

- KBot：`KBOT_MANAGED_CREDENTIAL_KEY` 和既有 key-version 配置，API、Worker 使用一致的受管密钥。
- Ammolite：`AMMOLITE_MANAGED_CREDENTIAL_SECRET_KEY`（或部署侧明确配置的对应变量），API、Worker 使用相同密钥。
- 即使是开发环境的真实飞书测试，也必须设置独立密钥，不能依赖开发默认密钥。
- 密钥由受控环境／Secret 注入，不写入仓库、命令历史、截图或运维日志。

本轮没有启用 HTTP 模拟服务。自动化使用 httpx MockTransport 和假凭据；fixture 门禁拒绝真实 Secret，production 拒绝 fixture 配置。

## 4. 真实飞书首次启用顺序

1. 创建隔离测试企业与两个自建应用，分别用于 KBot、Ammolite。开启机器人、申请最小发送权限、发布版本并设置测试用户可用范围。
2. 准备 A 申请人、B 审批人、C 无业务权限三个账号；通过官方调试台取得每个应用对应的 open_id。不同应用下不可复制 open_id。
3. 需要群测试时，把对应机器人加入专用测试群并确认可发言。群只允许低敏告警，不发送审批、执行和报告信息。
4. 确认门户 HTTPS 地址在手机／电脑可达，检查出站 DNS、TLS 和配置的两个飞书 API 端点；无外网环境检查企业网关及其到飞书官方服务的转发链路。不公开内部服务端口或增加回调入口。
5. 经授权完成第 2.1／2.2 节 Schema 预检、升级与代码发布，设置独立托管凭据密钥，启动独立 Worker。页面保存渠道及启停即可生效；没有规则／显式试发任务时不会自动向飞书发消息。检查连接、验证码与纯文本试发不依赖门户地址；启用业务规则前在“通知公共设置”填写可达 HTTPS 门户地址。
6. 管理页选择“飞书”，为当前 Tenant（Ammolite）／全实例（KBot）配置一次连接；无需选择 Domain，App ID／Secret 只由渠道管理员填写。企业微信仅占位。保存不发送；“检查”只验证令牌。新建／编辑用户时可选填应用对应 open_id，随后本人显式发码并确认；此步骤需独立 Worker 消费验证任务。
7. 验证码限时限次，只有对应本地用户可确认。错误输入不改变绑定语义版本；重新发码、停用／修改身份使旧验证任务失效。身份修改通过停用旧绑定并登记新候选完成。
8. 明确选择已验证目标进行真实试发，核对投递 message_id 及飞书客户端。之后仅启用少量测试策略；不补发历史事件。
9. 按设计文档 A／B／C 完整链路验收。手机未登录点击卡片应登录后进入原空间、确切 Proposal／HITL／Execution／Report，而非首页兜底。

连接、绑定、查看及重发权限独立。管理页面按服务端 capabilities 显示；后端每次重新鉴权。绑定管理员通过只读 targets 获取父连接，不因此获得连接写权限。KBot 页面复用既有 AIOps 登录入口；Ammolite 管理入口使用当前 Tenant 会话，本人验证核对当前用户与 Tenant 成员有效性，不以任意 Domain 作为身份绑定前提。

### 4.1 获取并填写个人 open_id

首次手工联调优先使用 [飞书 API 调试台](https://open.feishu.cn/api-explorer)，无需自行调用通讯录接口：

1. 登录后选择**通知渠道配置使用的同一个企业自建应用**，找到“发送消息”接口。
2. 将接收者 ID 类型（receive_id_type）设为 open_id，点击“快速复制 open_id”。
3. 搜索或选择要绑定的飞书成员，点击“复制成员 ID”。这里只复制身份，不需要点击“发送请求”或试发消息。
4. 将以 ou_ 开头的值填入本系统创建／编辑成员的飞书身份字段。保存只登记候选身份，不发送消息；新增或更换后由本人进入“我的通知设置”显式发码并确认。字段可留空；编辑时清空并保存表示解绑。

同一用户在不同应用中的 open_id 不同，不能从其他应用复制，也不能用 user_id、union_id 或群 chat_id 代替。KBot 与 Ammolite 使用不同应用时必须分别获取。操作依据：[如何获取指定用户的 Open ID](https://open.feishu.cn/document/uAjLw4CM/ugTN1YjL4UTN24CO1UjN/trouble-shooting/how-to-obtain-openid)。

如后续确需通过手机／邮箱批量查询，使用官方“通过手机号或邮箱获取用户 ID”接口并指定 user_id_type=open_id；其返回字段 user_id 才表示对应 open_id。这需要额外申请 contact:user.id:readonly，本阶段不自动增加通讯录权限，也不将它作为手工绑定前提。

成员身份区域使用公共表单、状态标签及按钮：状态显示“未绑定／待本人验证／已验证／已解绑”，不直接显示内部枚举。Ammolite 将字段、帮助和操作分区，输入宽度受控并适配明暗主题与窄屏；KBot 复用现有 ops-form／ops-field，不另建未定义的表单布局。

## 5. 故障处理

### 操作条件与配置位置

- capabilities 仅报告当前身份权限，不再返回 TOML 外发开关。渠道和公共设置读取／写入使用原有渠道治理授权；Ammolite 租户管理员在租户控制台维护，KBot 全局管理员维护。
- App ID、加密 Secret、渠道 enabled 在数据库连接表；public_origin 在独立公共设置表，各厂商共用。GET／PUT /api/v1/notification-delivery/settings 返回公共地址和 row_version，首次未配置为 0；更新必须携带 expected_row_version，并写审计。
- 检查连接仅检查令牌，无需门户地址，也不自动发测试消息。发码要求渠道启用及本人身份；试发要求渠道启用、个人目标已验证和显式确认；纯文本验证码／试发均不含任务链接，不依赖门户地址。
- 启用业务规则缺少门户地址返回 422／NOTIFICATION_PUBLIC_ORIGIN_REQUIRED；仍有启用规则时清空地址返回 409／NOTIFICATION_PUBLIC_ORIGIN_IN_USE。未配置门户地址不生成业务任务，页面明确提示；不会补发历史消息。已有任务内容与入口保留冻结，不因改地址批量重写或重发。
- 数据库配置不跨请求缓存，Outbox 按当前租户／实例设置生成新链接，Worker 每次按当前渠道、绑定、策略与业务权限复核。保存即生效，无需改 TOML 或重启；保存本身不做网络请求。
- TOML 只放网关、Worker、超时、限流等部署参数。升级前移除旧的 external_delivery.enabled 和 external_delivery.public_origin，不保留回退或启动导入；既有 App ID、Secret 与任务不迁移、不删除。技术参数变更仍需正常重启相关进程。
- 站内铃铛与外部渠道不是同一请求；失败应按 URL、状态码及服务端请求 ID 分别定位，不能以按钮可点击替代真实数据库和飞书验收。

### 抽屉共用规范

通知配置、试发、规则与投递详情统一复用项目 `admin-drawer` 的遮罩、面板、标题栏、正文、底栏、按钮与响应式样式。Portal 使用共享 `AdminDrawer`；KBot 使用共享 `KBotDrawer` 控制器。禁止增加页面私有的原生 dialog 抽屉样式或关闭策略。

遮罩点击与 Escape 均不关闭抽屉，只有显式关闭／取消或业务成功后关闭；写请求期间显式关闭按钮锁定。关闭编辑清空本次 Secret，重新打开不回显。共享确认框位于抽屉上层，取消确认不会关闭底层抽屉。

- PENDING／RETRY_WAIT：短事务领取后才发送；全局→连接→目标固定锁序限流。
- ACCEPTED：只有 HTTP200、业务 code=0 和合法 message_id 同时成立才记录。
- UNKNOWN：超时、响应丢失、已标记网络起点后的 Worker 崩溃等；不能当作“未发送”。
- 同 uuid、冻结内容的自动重试只在首次可能发送起 3300 秒安全窗口内进行。超过窗口保留 UNKNOWN，不盲发。
- 人工重试仅允许 UNKNOWN／BLOCKED／DEAD，重新检查版本、当前连接／绑定／权限和源资源。窗口外必须明确确认重复风险，创建新代次，旧任务与尝试保留。
- BLOCKED：修复凭据、权限、绑定或配置后受控重试；策略暂停、域不可用、凭据无法解密均不会触发网络发送。
- 未绑定用户不生成个人外发任务；站内通知不受影响。已登记未验证、配置停用等情况仍明确阻断；不改写冻结目标或回放历史事件。
- CANCELLED／EXPIRED：解绑、任务关闭或超过期限，不重发过时可操作提醒。
- 外部调用不持有数据库事务。旧 Worker 的租约、版本或时限不匹配时，不能写回 ACCEPTED。
- 管理页与 health API 展示各状态／安全错误码的数量、最早任务时间、近24小时接受率、耗时及令牌／限流错误。指标不以用户或资源 ID 作为维度；不能递归向故障渠道发送自身故障告警。
- 投递记录支持按状态筛选及分页，每页最多100条；历史保存在数据库，未新增自动删除投递历史的清理任务。
- 静默／合并默认关闭，只作用于 INFO／WARNING 的新告警与恢复；升级、审批、人工处理、执行／报告结果、HIGH／CRITICAL 和等级缺失消息不受影响。静默时段以 UTC 小时指定，支持跨日；不会顺延消息期限。合并窗口为固定 0～300 秒，按空间、连接／凭据、绑定／版本、目标、事件、确切资源及等级隔离，不做跨资源摘要。
- 被合并的旧任务保留 CANCELLED／NOTIFICATION_COALESCED 及关联新任务的审计，入口与事实均保留；已尝试发送或不确定任务不得合并。静默等待不耗尽网络重试预算，超出原 TTL 明确 EXPIRED／NOTIFICATION_QUIET_EXCEEDS_TTL，不删除任务。

回滚启用状态时通过页面停用连接／策略；必要时经授权停止对应 Worker。保留新增 Schema、密文及投递证据。不要删除业务数据以“恢复”通知。

## 6. 本地验证与未完成的真实验收

本轮验证范围：

- 飞书新版 OAuth、singleflight、长 Token、HTTP／业务失败、缺回执、限流、超时与安全错误分类。
- 独立任务投影、未绑定跳过、受众交集、冻结消息、租约 CAS、SQLite 文件上的真实并发事务。
- 应用层无事务网络、策略暂停、撤权／解绑、验证码本人确认与限次、UNKNOWN 证据及窗口内外受控重发。
- 源服务的确切父子关系、Tenant／Domain、Report／Job／版本及当前处理人核验。
- 两前端的真实组件／页面 + 假 HTTP 浏览器测试，包括手机未登录恢复、撤权、错空间和保存不试发。
- 双仓库架构、Schema、授权、进程拓扑、共同合同与 OpenAPI 离线校验。

自动化 HTTP 和 SQLite 测试不是 Oracle／PostgreSQL 真库并发验收，也不验证飞书实际可用范围、卡片渲染或手机真实网络。真实凭据、两个官方应用接受回执及客户端展示仍由测试企业联调完成。2026-11-05 未做线上数据库升级；2026-11-06 的定向升级见 6.2，仍未部署应用代码、重启服务或启用真实飞书推送。

### 6.1 本轮实际验证记录（2026-11-05）

- KBot：通知、工作流、输出、订阅及发布工具定向回归 340 passed。
- Ammolite：通知、工作流、输出、订阅定向回归 229 passed。
- Portal／KBot 页面通知浏览器回归 40 passed；使用真实页面与假 HTTP，不是飞书客户端。
- Portal `npm run build` 通过；保留既有大 chunk 警告。
- 两项目架构、canonical Schema、进程拓扑、共同合同及 OpenAPI 检查通过；KBot 另含统一授权检查。OpenAPI 分别为 16 快照／857 路径、18 快照／998 路径。
- 新增静默边界、紧急消息绕过、TTL 不延长、重放不取消新任务、并发合并及保留全部入口／记录的回归。
- 只读 `tests/acceptance/check_postgresql_entity_schema.py` 真库检查**未通过**：当前配置的 PostgreSQL 尚未部署新投递结构，另有输出任务等既存差异。本轮仅读取元数据，没有修复、初始化或改动该数据库。canonical 离线通过不等于现有环境已升级，发布前需另行审阅准确目标及无损差异。

回归命令（对应仓库根目录，使用各自 Python 环境；本轮临时测试依赖位于 `/tmp/kbot-feishu-test-deps`）：

~~~bash
# 两项目共同覆盖
PYTHONPATH=/tmp/kbot-feishu-test-deps python -m pytest -q \
  tests/unit/main_api/test_feishu_notification_delivery.py \
  tests/unit/main_api/test_notification_delivery_transactions.py \
  tests/unit/main_api/test_notification_delivery_application.py \
  tests/unit/main_api/test_notification_delivery_auth.py \
  tests/unit/aiops_agent/test_notification_delivery_resource.py \
  tests/unit/main_api/test_aiops_workflow_reconciliation.py \
  tests/unit/main_api/test_aiops_output_reconciliation.py \
  tests/unit/aiops_agent/test_aiops_workflow_projection.py \
  tests/unit/aiops_agent/test_aiops_notification_subscription.py \
  tests/unit/aiops_agent/test_aiops_output_notifications.py
# 上述共同命令的同一次运行中，KBot 另包含：
# tests/unit/main_api/test_notification_center_s6.py
# tests/unit/main_api/test_notification_foundation_security.py
# tests/unit/main_api/test_aiops_notification_routes.py
# installation/tests/test_release_tooling.py
# Ammolite 另包含：tests/unit/main_api/test_notification_reconciliation.py
# 浏览器：在 Ammolite apps/portal-web 中
npx playwright test tests/notification-delivery.spec.mjs \
  tests/notification-session-recovery.spec.mjs tests/aiops-notification-links.spec.mjs \
  --workers=1 --reporter=line
~~~

### 6.2 kbotdev 定向 Schema 升级与发布前验证（2026-11-06）

本次按“Schema 升级及核验完成后，再推送代码”的顺序执行。目标为 kbotdev 的现有 KBot Oracle 和 Ammolite PostgreSQL；没有运行全库初始化，没有删除、清空、重建已有业务表，没有导出业务记录或改写 Prompt。只保存了元数据、行数与执行日志。

- Oracle：基线无 KBOT_NOTIFY_* 对象。执行 Main API v8 增量，新增八表、七个普通索引、external 频控种子、四权限与平台管理员四项授权。全部 25 条语句执行完成后，再用独立只读命令核验列类型／可空性、主外键、索引、种子及权限通过。
- Oracle 验收按实际数据字典等价表示识别 SMALLINT，以及带时区列普通索引的 SYS_EXTRACT_UTC 表达式；没有为匹配验收而改动表或索引。
- PostgreSQL：锁内确认原 platform_notification_delivery_job 为 0 行；原位扩展，不建第二条队列，新增七表。63 条定向语句、三权限与既有内置管理员九项授权在单事务中完成；提交后独立只读验收通过。
- 行数核对：Oracle 183 张、PostgreSQL 220 张既有表均未减少。预期权限增量分别为 4／4、3／9；运行期间 Oracle 任务／运行／事件／Outbox 和 PostgreSQL 登录刷新令牌有自然增加，不能归因于 Schema 升级。既有通知及业务会话、报告等记录未清理；新渠道、绑定、策略及投递表均为空，只有频控种子各 1 行。
- 服务器操作证据目录：/tmp/notification-schema-20261106-yNtRHP；包含 oracle/postgres-before.json、oracle/postgres-after.json、row-count-verification.json、postgres-upgrade-report.json、postgres-verify-report.json 和 Oracle 逐步日志。该目录是临时运维证据，不是业务备份。

本轮发布前验证在隔离工作树执行，仅包含通知变更，不混入并行知识模块重构：

- KBot 后端定向回归：341 passed；Ammolite：232 passed，包含新增无损升级计划测试。
- Portal／KBot 浏览器回归：40 passed。适配既有确认弹窗的测试桩；没有更改审批业务行为。
- Portal 生产构建通过，仅既有大 chunk 警告。
- 两项目架构边界、canonical Schema、进程拓扑、跨项目合同及 OpenAPI 验收通过。OpenAPI 分别为 16 快照／869 路径、18 快照／1010 路径。
- PostgreSQL 本次只验收通知目标结构，不代表全库所有历史漂移均已消除；Oracle／PostgreSQL 真库投递并发、真实飞书接收及手机访问仍需后续联调。

本次没有更新服务器 Git 工作树、重启服务、修改 external_delivery 开关、配置真实飞书凭据或向外部用户发消息。Schema 保留，后续按第 4 节部署代码与启用测试。

### 6.3 渠道范围与用户可选绑定修订（2026-11-06）

本次修改同步进入两个主工作区，保留并行知识模块改动。以下记录仅覆盖代码与本地验证；代码提交／推送不代表线上部署。本次没有连接或修改线上数据库、部署服务、重启进程或发出真实飞书消息。

- 渠道／绑定／策略的管理范围为 KBot 全局、Ammolite 当前 Tenant，不再要求选择 Domain。企业微信仅展示占位，前后端都不允许保存或发送。
- 新建用户可不填飞书身份；编辑用户支持补绑、更换和清空解绑。用户与候选绑定在同一事务中保存，验证由本人完成。未绑定不生成个人外发任务，站内通知不受影响。
- 业务消息及独立入口保留真实资源 Domain，业务访问和处理仍检查当前授权；身份绑定不赋予任何业务权限。
- 两主工作区后端定向回归：KBot 230 passed、3 subtests passed；Ammolite 142 passed。覆盖可选绑定、版本冲突、身份占用、解绑、无绑定跳过、个人收件箱聚合及空旧结构切换保护。
- Portal／KBot 通知浏览器回归 43 passed，Portal 生产构建通过（保留大 chunk 警告）；两项目架构边界、离线 canonical Schema 与跨项目语义对齐检查通过。
- 本次未进行 Oracle／PostgreSQL 实库结构切换或真实飞书验收。现有旧范围结构必须先按 2.1 预检；存在任何旧外发记录或活动旧通知凭据时停止，不清数据、不自动转换。

### 6.4 kbotdev 渠道新增 500 的范围升级补漏（2026-11-06）

- 真实异常是 PostgreSQL 的 `platform_notification_external_connection.domain_id` 仍为 NOT NULL；已部署代码按 Tenant 保存，不再提供 Domain。历史 v8 外发表创建成功，并不代表后续范围切换已完成。此次不是飞书接口失败，也不能通过默认 Domain 或兼容写入规避。
- 已在 `/home/ubuntu/Ammolite` 执行既有 `scripts/deployment/upgrade_notification_scope.py` 的定向切换逻辑。事前核对七张相关外发表均为零行、通知托管凭据为零行；外发总开关原本关闭，Worker 空闲，Outbox 不入外发队列。PostgreSQL 在事务内持有目标表排他锁，移除四张空配置表的旧 Domain 列，放宽 Job／Attempt 的 Domain 可空性，替换租户范围普通索引。
- 升级后八张通知外发表的列、类型、可空性、主外键、普通索引、限流种子及通知权限规范验收通过。未清理业务记录、站内通知或投递历史；限流种子保留。未修改服务器代码、停启服务或开启外发。
- 新增 Ammolite 实库验收入口 `tests/smoke/smoke_notification_connection_postgresql.py --run`。使用部署配置、假飞书凭据和既有租户外键；只接受外发关闭且目标外发结构为空的环境。应用 UoW 和真实加密凭据存储共享一个外层事务，内部提交仅释放保存点，结束时整体回滚；不调用飞书。
- 此次已在 kbotdev PostgreSQL 执行上述验收：生产 HTTP 路由新增返回 201；读取、改名、密钥轮换、重复配置冲突及凭据清理、过期版本冲突及凭据清理、跨租户连接／凭据隔离、三次写操作审计均通过。回滚后各目标表与通知凭据数量恢复基线，测试数据零残留。
- 验收 HTTP 使用进程内 ASGI 与注入测试身份，未替代用户真实登录会话、在线网关或飞书接收验收。KBot Oracle 未在本次事件中执行范围切换；其发布仍须分别完成第 2.1 节预检、升级和 Oracle 规范验收，不能套用本次 PostgreSQL 结果。

后续发布顺序固定为：核对目标部署与数据 → 范围升级预检／应用 → 当前 canonical 验收 → 真实数据库保存回归 → 页面手工验收。单元测试、按钮可点击、服务 readiness 或历史 v8 升级报告均不能替代当前结构与持久化链路验收。自身 Schema 验收仍只在部署／测试入口执行，不加到运行时请求或启动门禁。

### 6.5 公共设置热生效与企业 API 网关修订（2026-11-06）

- App ID、加密 Secret、渠道启停继续存数据库；门户 HTTPS Origin 改为独立公共设置，按 Tenant／实例保存，具备行版本与审计。移除 TOML 的 enabled／public_origin，不做启动导入或双读。检查连接、本人验证码和纯文本试发不依赖门户地址；仅业务规则启用与带详情入口的任务生成要求该地址。
- token_url／message_origin 保留在 TOML，默认官方 API，支持部署管理员指定企业 HTTPS 网关。拒绝不安全地址及消息 Origin 路径，OAuth 与消息请求均不跟随重定向，不向租户开放地址覆盖。网关配置变更仍需正常更新相关进程。
- 新增公共设置表必须按第 2.2 节先预检／定向升级再发布代码；既有渠道、密文、任务和业务数据不回填、不改写。独立 Worker 常驻读取数据库当前状态，后续页面配置修改无需重启；旧任务链接保持冻结。
- 两项目后端通知定向回归：Ammolite 194 passed，KBot 248 passed。覆盖配置模板、无门户检查／试发、版本冲突、租户隔离、配置热读取、任务内容冻结、网关路由、重定向阻断及仅新增设置表的部署边界。两项目 OpenAPI 校验通过（18 快照／993 路径、16 快照／843 路径），架构边界、离线 canonical Schema 和跨项目语义对齐检查通过。
- Portal／KBot 真实页面加假 HTTP 的通知浏览器回归 40 passed；Portal 类型检查及生产构建通过，保留既有大 chunk 警告。这些结果不是在线网关或飞书客户端验收。
- 本地真实 PostgreSQL 使用 smoke_notification_connection_postgresql.py --run --isolated，在事务内创建隔离 Schema，实际调用新增表升级、生产 HTTP／应用／仓储与加密凭据链路。新增返回 201、密钥轮换、公共设置热读取、无门户检查和试发入队、租户隔离及审计通过；另一租户渠道哨兵保留，最终整体回滚，测试数据零残留，未调用飞书。只跳过隔离测试中真实 IAM 目录验收；部署脚本仍强制验收真实通知权限，测试身份注入不替代在线登录。
- 本地全库 check_postgresql_entity_schema.py 检查未通过：存在既有知识等模块结构漂移及未部署外发表；本轮没有修复或改写该业务 Schema。隔离通知结构通过不能表述为全库实库通过。Oracle 新公共设置表仅完成离线和模拟升级测试，未做在线验收。

本轮未连接或升级线上数据库、拉取服务器代码、重启服务或发送真实飞书消息。两仓库提交推送不代表完成部署；用户实际飞书应用、企业网关、客户端送达与手机任务入口仍需经授权联调。

### 6.6 kbotdev PostgreSQL／Oracle 通知 Schema 定向升级（2026-11-06）

本次在用户明确授权后执行，只升级通知相关结构，不修复其他模块的全库漂移，不拉取服务器代码、不停止或重启服务，不发出真实飞书消息。

- Ammolite 已部署工作树为 a2df9ba8，GET /api/v1/notification-delivery/settings 的真实异常为 UndefinedTableError：platform_notification_external_settings 不存在。先核对既有八张通知表、索引、种子与权限，再执行 scripts/deployment/upgrade_notification_settings.py --apply；PostgreSQL 单事务仅新增设置表，提交后独立完整九表验收再次通过。
- PostgreSQL 既有飞书渠道 1 条、审计 1 条、频控种子 1 条均保留；绑定、规则及外发任务仍为空，未迁移、回填或覆盖 App ID／加密 Secret。使用现有渠道的真实 Tenant，通过部署代码的应用服务、UoW 与仓储只读调用 get_settings，返回空 public_origin／row_version 0，通过。该验证不借用用户登录、不调用飞书，不替代浏览器真实登录验收。
- KBot 工作树仍为 3870ec44，没有外发生产者代码或外部 Worker，未为升级拉代码。Oracle 原八表仍为旧 Domain 范围：七张外发表均零行、活动 notifications 凭据零条，完整旧规范、七普通索引、频控种子及四权限预检通过后，才执行已审阅的定向范围脚本。仅移除四张空配置表的 Domain 列，放宽 Job／Attempt 和托管凭据的 Domain 可空性，替换全局连接普通索引；不改写任何现有凭据或业务行。
- Oracle 随后从当前 canonical DDL 仅创建 KBOT_NOTIFY_SETTINGS，完整九表／七普通索引／种子／四权限验收通过；独立连接复验确认旧配置 Domain 列已移除、Job／Attempt 可无 Domain、业务 Entry 仍必需 Domain。Oracle DDL 隐式提交，各 DDL 的执行前后状态均记录，没有盲目重跑。
- 前后逐表只读核对：PostgreSQL 246 → 247 张表，Oracle 290 → 291 张普通表，各只新增一张设置表，没有表被删除，两库全部既有表的行数均未减少，既有通知表行数不变。Oracle 运行期间 KBOT_OPS_RUN／RUN_EVENT／TASK 自然增加 4／4／12 行；升级没有 DML，不能将这类并发业务增长归因于升级。Oracle 核对范围排除了回收站、嵌套存储、辅助和 IOT 溢出对象。
- kbotdev 证据目录：/tmp/notification-settings-schema-7BafEV，含 postgresql-before.json／postgresql-after.json、oracle-before.json／oracle-after.json、row-count-verification.json 与 oracle-upgrade.jsonl；仅保存结构、计数和部署步骤，不包含业务记录、凭据明文或备份。

升级结束时 Ammolite／KBot 原 Main API、站内 Worker 与外部 Worker 的 PID 均未改变；本次未修改配置或渠道开关。可刷新 Ammolite 通知渠道页面继续手工验证。KBot 应用代码发布仍需另行授权，本次 Oracle Schema 就绪不代表其飞书功能已经上线。
