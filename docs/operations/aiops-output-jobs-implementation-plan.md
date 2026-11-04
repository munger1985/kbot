# AIOps 菜单、报告与案例异步产出修改计划

日期：2026-11-03。状态：本次代码实施与本地专项回归已交付，真实测试由用户手动执行。
持久化输出任务、当前授权、会话／报告来源的异步报告和案例、可恢复处理、事务终态和通知、
两端生成记录及菜单、案例知识投影和固定资源维护已实现。
通知精确定位、版本化权威对账及报告服务端分类分页／完成刷新已接线；
结构化适用性链、可解释有界重排与检索评测工具已补离线实现；真实语义质量和真实依赖端到端验收仍未完成。
最新责任划分、交付范围及已知限制见第 6.23 节；第 6.7 节保留分批落点。
此前记录保留作证据，后来的完成项不反向改变当时的验证范围，也不能把离线通过当成真实环境验收。
依据：[已确认业务设计](../product/aiops-work-report-knowledge.md)。
案例定制详见[KC 入库与检索专项设计](../proposals/aiops-diagnosis-case-knowledge-design.md)。
两仓按同一工作包同步维护，不能将菜单改名或接口返回 202 单独视为整体交付完成。

## 1. 当前代码核对基线

| 仓库 | 本轮核对的提交 | 前端 | 持久化平台 |
| --- | --- | --- | --- |
| Ammolite Cube | 696f3d7 | apps/portal-web | PostgreSQL、Tenant／ResourceDomain |
| KBot4 | 5f866600 | ui/aiops | Oracle、Domain |

最初核对开始时两个工作区均无未提交修改；当前两仓已有大量本项工作中的未提交代码与文档。
以下起始结论不代表当前线上部署状态。提交号是最初检查定位，不要求未来实施回退到该版本；
后续工作须保留现有改动并按当前工作区复核，不能按起始基线重复实施。

### 1.1 开工时已有能力与真实缺口

以下是起始代码核对结果；已完成改动及验证另见第 6 节，不能将本表直接当作当前进度。

| 模块 | 当前代码证据 | 复用内容 | 本次缺口 |
| --- | --- | --- | --- |
| 导航 | Ammolite 的 businessNavigation.ts、BusinessAppShell.vue；KBot 的 aiops-shell.js | Agent 固定入口、分组、权限、详情归属和现有样式 | 仍是“态势与事件／巡检与演练／报告与知识”；无法直接按既定顺序插入单独报告入口 |
| 会话报告 | AIOpsDiagnosisView.vue 的 generateConversationReport；KBot aiops-workspaces.js；内部 runtime/routes.py 的 /reports:generate | 模板、来源聚合、报告冻结和版本幂等 | 内外接口仍返回 201，前端等待 report_id 后跳转；generate_conversation_report 在请求内装配报告 |
| 会话案例 | operations_knowledge.py 的 close_diagnosis → case_closure.py 的 close_conversation_case | 有效轮次、source_fingerprint、人工处置、来源重放、脱敏和案例版本 | 接口虽为 202，返回前仍等待语义脱敏、KC ingest_user_file 及受理审批；尚无完整持久化生成回执 |
| 案例后台处理 | operations_knowledge.py 的 reconcile_version／reconcile_next | KC 对账、PROCESSING 到 DRAFT、来源围栏和人工发布 | 需与输出任务阶段、终态及通知可靠关联，不能把 KC 受理当作草稿就绪 |
| 案例知识适配 | contracts/knowledge.py 的 DiagnosisCaseProfile、_diagnosis_case_markdown、diagnosis-cases Collection | 问题／环境／动作／效果、脱敏 Markdown、精确 Revision、发布和检索复核 | 完整排查路径、支持／反证、措施与效果逐项关系及引用上下文尚需定制；不能只上传报告 PDF |
| 知识召回范围 | operations_knowledge.py::search、repositories/knowledge.py::list_published_candidates | Registry 资格筛选和 KC Discovery／Evidence | 当前先以 request.max_results 截断发布候选，再交 KC 检索；须分离候选预算与最终结果数，避免漏召回 |
| 案例资源与来源门禁 | operations_knowledge.py::_collection、case_closure.py::validate_closure_sources；KBot 初始化工件 | 固定案例资源定位、来源原件重放、Agent 来源类型开关 | 两仓资源／模型须核验；当前全会话活动轮次门禁不能原样用于冻结任务，否则继续会话会干扰已提交输出 |
| 领域任务设施 | entities/messaging.py、Outbox、workers/outbox_dispatcher.py；entities/runtime.py 的 OpsTaskEntity | 至少一次投递、租约、恢复和既有 Worker 进程 | OpsTask 属于诊断 Run，不能直接当跨会话产物任务；现有 Sink 没有统一报告／案例输出任务消费者 |
| 平台通知 | platform_notification.py、共享 notifications/catalog.py、Main API 通知投影 | 已有 aiops.report.ready、事务通知 Outbox 和后台任务投影 | report.ready 的 operation_id 仍为 Run ID；缺少独立生成任务生命周期和案例草稿就绪／失败事件 |
| 深链与对账 | Ammolite notificationRoutes.ts、notifications/outbox_repository.py、Main API notification_projection.py；KBot NotificationEnvelope 与投影 | 受控通知路由、业务空间和身份边界 | Ammolite 报告通知只返回通用 /apps/aiops/reports，未定位 Domain 和报告；无案例／失败输出任务路由；Ammolite 任务对账尚无 AIOps 输出任务分支 |
| KBot 通知前端 | ui/aiops/js/aiops-shell.js、Main API api/notifications.py | 后端通知摘要、列表、SSE、已读及业务任务投影接口 | 当前 AIOps 页壳未挂载通知中心；需增加共享前端组件接线，不是已有完整通知前端可直接改深链 |
| 成果列表 | AIOpsRecordsView.vue、AIOpsDiagnosisCasesView.vue；KBot aiops-records.js、aiops-cases.js | 正式报告、巡检批次和案例资产查询 | 无排队／失败输出任务视图；报告分类需显式合同；巡检批次仍称“巡检报告” |
| 上传资料 | AIOpsOperationsKnowledgeView.vue；KBot aiops-operations-knowledge.js | MANUAL 内容筛选、上传、审批、版本和解析流程 | 内容页应更名“运维资料库”，与配置区“运维知识库”区分；不重做解析体系 |

共同后端根目录为 services/aiops_agent/src/aiops_agent；共享合同位于
packages/platform_core/src/platform_core/contracts/aiops，客户端位于 packages/platform_clients/src/platform_clients/aiops.py。
平台不同不改变上述产品结论；不得直接复制另一仓的数据库或认证实现。

### 1.2 本轮发现的文档和测试注意点

- KBot 的旧报告文档仍叙述同步生成和两类案例，并一概排除不利结果；本次文档修正为独立输出任务和四类案例。
- Ammolite 的 tests/contract/test_frontend_aiops_information_architecture.py 仍有“智能诊断”和旧“运维知识”字面断言，
  与当前 Agent 工作台菜单及本次改名不一致，实施时需按新业务合同更新；本轮未运行该测试，不能把阅读发现记为测试失败。
- 现有报告已能发通知，案例也已有来源失效通知；不能声称“完全没有通知”或重新建设整个通知平台。
- 专业巡检正文、调查判断和 VERIFIED 门禁不属于本次放宽或重写范围。

### 1.3 本次追加需求的当前代码落点：案例复用 KC

以下记录追加需求时的未提交工作区复核（第 6.9 节），后续落地见第 6.10–6.22 节；
不重复实现已经存在的适配，也不将模块存在视为端到端完成：

| 业务环节 | 可复用的当前模块 | 还需完成或证明的内容 |
| --- | --- | --- |
| 结构化案例 | contracts/knowledge.py、contracts/case_knowledge.py、case_closure.py、case_report.py | 检查、反证、逐项效果、来源和禁忌贯通各来源入口；报告来源仍需异步接入 |
| 可索引正文 | case_knowledge_projection.py::compile_case_knowledge、case_semantic_redaction.py | 复用现有稳定正文、哈希和关联单元；真实 KC 后仍保留可引用章节与措施上下文 |
| KC 入库与恢复 | conversation_case_output.py、case_intake_recovery.py、operations_knowledge.py::reconcile_version | 验证实际受理、索引等待、确切版本恢复、失败重试和撤回；不是再次上传报告 PDF |
| 发布与引用 | knowledge_search.py、knowledge_preview.py、repositories/knowledge.py | 保持草稿管理预览与已发布 Agent 检索隔离；检索期间撤权、修订和来源更正不能泄漏旧范围 |
| 检索适用性 | knowledge_search.py、case_knowledge_projection.py 的关联上下文 | 仍需案例质量集、环境冲突／未知判断及排序验证；不得声称自然语言禁忌已全部硬过滤 |
| 固定资源与模型 | case_collection.py、两层 case-collection/models API、各仓规范初始化 | Ammolite 补齐固定资源初始化证据；KBot 核验现有资源绑定；两仓显式模型及 TEXT 策略真实可用 |

资源配置保持原菜单和表单。若现有表单无法维护固定案例资源，先用规范初始化和现有管理 API／KC
管理能力提供受控配置路径，记录所需参数和权限；不将新增表单当作已获授权的必需实施项。
初始化不能重置管理员或无关业务资源；KBot 现有全量初始化会涉及账号凭证，不能为补一个案例资源直接运行。

## 2. 目标合同与数据边界

### 2.1 独立输出任务

新增 AIOps 自有 OpsOutputJob 业务聚合，只支持本次两类任务 REPORT_GENERATION／CASE_EXTRACTION。
它不是新的诊断引擎、通知表或 Run，不触发数据库取证，不复用来源 Run ID 作为自身 ID。

最低持久化字段：

- job_id、job_type、发起人、可信租户／业务空间、来源 conversation_id／锚点 run_id、trace_id；
- 冻结输入、来源原件清单与指纹、模板确切版本／哈希、人工归纳及其来源性质、权限和安全级摘要；
- request_idempotency_key、input_hash、status、stage、attempt_count、可重试标记、下一可运行时间；
- lease_owner／lease_token／lease_until、heartbeat、row_version、受理／开始／完成时间；
- result_report_id 或 result_asset_id＋result_version_id、脱敏错误码和简要失败原因；
- 重试关系和外部 KC 确切受理版本的恢复定位。优先引用现有冻结原件及 Registry，不复制第二份事实真相。

业务状态限定 QUEUED → RUNNING → SUCCEEDED／FAILED；必要等待外部索引时保持 RUNNING 并记录阶段及下一检查时间。
报告 READY／PARTIAL、案例 DRAFT／REVIEW_REQUIRED／PUBLISHED 和真实处置效果分别查询、分别显示。
任务成功不自动改变案例发布状态，不把报告 PARTIAL 变成生成故障。

### 2.2 HTTP 与客户端

保留现有来源命令路径，直接切换调用方，不另建旧版／新版并行接口：

| 路径后缀 | 目标行为 |
| --- | --- |
| POST /reports:generate | 从同步 201 报告回执改为 202 输出任务回执；现有会话、告警和显式周期报告调用方同批适配 |
| POST /operations-knowledge/conversations/{conversation_id}:close-diagnosis | 保留现有来源选择和人工归纳合同，返回 202 任务回执，不在请求内等待脱敏及 KC |
| POST /operations-knowledge/reports/{report_id}:extract-case | 报告详情的同类入口复用案例后台任务，不能保留另一条同步重处理链 |
| GET /output-jobs | 分页、按 job_type／来源会话／状态筛选，遵循既有结果读取及发起人权限 |
| GET /output-jobs/{job_id} | 返回权威阶段、失败原因、重试能力和有权限的结果定位 |
| POST /output-jobs/{job_id}:retry | 仅重试允许恢复的失败任务，保留尝试历史；输入变化须重新提交新任务 |

路径后缀挂在各仓已有公开 AIOps 前缀和 /internal/v1/aiops 下，不新增公开内部路由。
回执含 job_id、job_type、status、stage、source、submitted_at、结果定位（初始可空）及查询位置。
提交请求使用 Idempotency-Key，作用域为 Domain＋发起人＋任务类型＋来源；同一作用域内
相同键不同输入拒绝，同源同模板同输入复用仍有效的任务／结果；
幂等命中也重验当前权限和来源。不同任务类型的幂等域独立。

冻结发生在受理事务内，Worker 必须消费冻结输入，而不是调用“读取当前全部轮次”的方法偷偷扩大范围。
沿用 case-candidate 的 source_fingerprint 预览校验；报告精确冻结所选模板与提交时已结束轮次。
新轮次可以继续，但不修改排队任务输入；已选来源被反驳、更正或撤权时按既有规则拒绝、降级或要求重新确认，
不得擅自改用新来源。后台重建可信当前授权，不保存或重用过期 JWT／浏览器令牌。

## 3. 分阶段修改清单

### P1：合同、实体与规范持久化

新增共享输出任务请求／回执／查询合同、任务实体、Repository 和应用服务；接入各仓 UoW。
Repository 仅负责持久化，不提交事务，不承担模型、KC、通知投影或 HTTP 编排。

- 两仓：packages/platform_core 的 AIOps 合同；services/aiops_agent 下新增 output_jobs 合同／应用／实体／仓储。
- Ammolite：canonical ORM、persistence/uow.py、scripts/deployment/init_postgresql.py 的实体登记及 PostgreSQL acceptance。
- KBot：canonical Oracle DDL、实体、仓储、UoW、schema_manifest.json、受控升级／重建渲染工件和 Oracle acceptance。
- 不手写 Ammolite 本地迁移、回填或兼容表；实际数据库更新另按授权发布。KBot 不增加 CHECK／UNIQUE 业务约束。
- 同步规范初始化中的案例资源登记、Domain 绑定和模型配置前置条件；KBot 核对
  database/oracle/bootstrap/aiops/initial_admin.sql 与 scripts/db/initialize_aiops.py，
  Ammolite 补齐本轮未找到的固定案例资源初始化证据及对应 acceptance。不在运行时做 Schema 自检或自动选模型。
- 测试：并发幂等、原子受理与 Outbox、作用域隔离、租约与版本围栏、输入指纹及不保存秘密。

完成门禁：两仓能持久化并分页读取任务，提交与待执行事件同生共死，未调用模型或 KC。

### P2：报告与案例处理下沉后台

修改 runtime/service.py、operations_knowledge.py、case_closure.py，将“受理冻结”和“消费冻结输入执行”拆开。
复用已有报告装配、原件重放、模板、案例脱敏和 Registry 生命周期，不另造报告或案例生成器。

- 向现有领域 Outbox 投递 OPS_OUTPUT_JOB_REQUESTED，以 job_id 定位，接入 workers/outbox_dispatcher.py 和 Worker 依赖装配。
- 领域 Outbox 唤醒业务任务；租约、心跳、重试、终态以业务任务为权威，不用“消息已投递”表示业务完成。
- 无消费者配置须明确失败，不允许落入 LoggingOutboxSink 后视为成功。
- 报告完成事务关联确切 Report ID／版本；已有产物幂等重放不重复创建版本。
- 案例在模型前、索引前后、审批前后保留来源围栏；沿既有 reconcile_version／reconcile_next，
  到可完善草稿时再结任务。KC 等待不长期占有数据库事务或一条 Outbox 租约。
- 拆分 validate_closure_sources／_conversation_case_context 的全会话活动门禁与冻结来源校验：
  提交时无进行中轮次，受理后无关新轮次不阻断处理；实际来源失效和后续反证仍须处理，
  发布／检索也不能因会话出现无关活动轮次就误判既有案例无效。
- 外部已受理但业务写回前后失败时，复用确切 Revision 继续，不能重试即重复上传。
- 有上限地恢复过期租约和临时故障；权限／来源实质变化为不可盲目重试错误。
- 同一个来源报告的原有提炼入口与直接会话入口共用此机制；自动巡检发布不为本批改成用户点击。
- 测试：关闭 API 进程后完成、Worker 崩溃恢复、重复消息、过期 Worker 回传、KC 慢／失败／已受理、
  新轮次与原件反证、部分报告、案例未发布。

完成门禁：两个提交接口均快速返回持久化回执，重进页面可读任务；没有前端后台 Promise 冒充异步。

### P2-K：案例专属知识投影、KC 入库与后续检索

与 P2 共同形成案例输出闭环，按专项设计扩展现有能力，不重建知识底座。

- 扩展 contracts/knowledge.py 的 DiagnosisCaseProfile，明确检查／发现／反证、结论强度、
  实际措施与效果对应、适用与禁忌及来源定位；保留四类用途和现有 VERIFIED 资格校验。
- 新增 application/case_knowledge_projection.py，将冻结且脱敏的结构化案例编译为稳定正文、
  Profile 及语义单元关联，替换散落的简化 Markdown 拼装；case_report.py 与 case_closure.py 共用。
- 复用 KC CONTENT／SUPPLEMENT、结构 IR、通用 Locator 和精确 Revision；结构化案例走适配的
  TEXT 策略，不重复 OCR，也不让通用摘要重写事实、原因强度或效果。
  核对 KC application/parse_policy.py 的实际策略优先级和 ParsePlan，不能假设临时 overrides
  能覆盖 Collection 规范策略；必要时扩展通用内容类型处理档，手册及管理员显式策略不被静默改写。
  通用语义单元元数据不足时才扩展 KC 合同／Evidence 链，不向 KC 塞 AIOps 业务判定。
- 一个业务版本绑定确定的正文哈希、策略／模型快照及精确 KC Revision；沿现有版本隔离方法，
  确保新版本处理期间旧发布版本仍可检索。外部受理、解析审批、索引对账和业务审核分开。
- 索引与引用复核后形成可完善草稿，关联 job_id 完成任务；人工发布再次核验来源、权限和效果资格。
  管理预览允许有权限的草稿，Agent 只接受当前 PUBLISHED 精确版本范围。
- 修改 operations_knowledge.py::search 和 repositories/knowledge.py::list_published_candidates，
  将候选预算、稳定分页／分片与最终 Top-K 分离；召回前不按最终结果数或时间取前 N 条。
  权限、安全、明确禁忌为硬边界；其他环境兼容、相关性、证据质量及用途用于解释和排序。
- 命中措施时按关联取回前提、实际效果、限制和失败警告，引用落到当前确切发布版本正文。
  不将历史案例当当前现场证据，不把失败动作送作自动执行建议；检索后再次核验来源与发布资格。
- 新增评测：旧相关／新无关、同症状异因、版本冲突与未知、四类用途、措施上下文、
  精确引用、草稿隔离、同源去重、Agent 范围、撤权与撤回；回归手册检索。
- 涉及 Profile／索引模型／共享 KC 合同的变化，同批同步两仓 Schema 工件、客户端、OpenAPI 和测试；
  不自动回填历史缺失字段，不改资料上传流程或配置菜单。
- 验证既有 Agent 的知识开关、source_kinds 与知识库绑定：案例不归入任意手册库，
  只有启用 DIAGNOSIS_CASE 且符合现有知识能力／绑定门禁时才加入授权范围；未启用不回流。

完成门禁：案例能处理为可引用草稿；发布后新会话可按业务语义检索，未发布或失效版本不能回流。

### P3：结果通知、深链与权威对账

扩展两仓共享 notifications/catalog.py、AIOps repositories/platform_notification.py、
Main API notification_projection.py 及相关通知合同。

- 新增任务启动／失败、案例草稿就绪事件；报告完成复用 aiops.report.ready，但关联输出 job_id。
- 可采用 aiops.output_job.started、aiops.output_job.failed、aiops.case.draft_ready 稳定事件名；
  进度只更新任务投影，不反复提醒。命令事件和通知事件不得混用。
- 业务结果、任务终态和平台通知 Outbox 同事务登记；任务回收／重放不重复通知。
- 报告有缺口仍为生成成功，以警示摘要说明；案例处理成功只通知“草稿已生成”，不通知“已发布”。
- 同一会话的报告、案例和后续重新生成必须分别显示为独立后台任务，不能覆盖同一 Run 任务投影。
- 生成完成／失败定向发起人；现有订阅规则保持，人工送审后才产生审核待办。
- Ammolite：扩展通知受控路由模型，补充 Domain、job_id、case asset/version 定位；
  修改 notificationRoutes.ts 和业务入口解析；Main API 增加 AIOps 输出任务权威对账分支。
- KBot：扩展 NotificationEnvelope 所需受控资源定位；在 ui/shared 增加可复用的通知中心前端组件
  （例如 kbot-notification-center.js），由 aiops-shell.js 挂载，使用现有 Main API 通知摘要、列表、
  已读和 SSE／断线恢复接口及当前 API Client，不新增通知后端或侧栏任务入口。
  从通知携带的可信 Domain／job／结果标识进入对应静态页并重新鉴权。
- 失败通知打开确切失败任务，不能跳到不存在的报告；通知延迟不影响页面读取已完成结果。
- 测试：Outbox 重放、乱序、通知积压、补偿对账、跨空间深链、撤权后点击、部分报告和草稿未发布。

完成门禁：两种成功输出和两种失败均能在通知中心找到，并定位到有权限的确切任务或结果。

### P4：菜单与命名单一来源

两仓落实“Agent 工作台／运维工作／报告中心／知识资产／配置与治理”的既定顺序。

- Ammolite：apps/portal-web/src/app/businessNavigation.ts、BusinessAppShell.vue、必要的 router.ts 和共享导航样式。
  导航数据需表达按顺序混排的单独链接与折叠组，报告中心不放进第二个同名组，也不挤到顶部主入口旁。
- KBot：ui/aiops/js/aiops-shell.js、对应导航样式及页面标题／详情父项映射。
- 保留已有分区卡片风格、移动端、当前项高亮、权限隐藏、默认 Agent 入口和已有折叠交互。
- 精确保护配置区七项的名称、顺序、路由、权限及表单；不移动“运行记录”或删除“运维知识库”。
- 资料内容页、案例页、报告页统一新标题；巡检页签“巡检报告”改“巡检记录”。
- 测试：导航快照、配置区逐项不变、标题权威、详情归属、报告单独直达、无重复入口、其他 App 不回归。

完成门禁：不仅改标签，两端菜单结构、页面标题及入口职责均符合设计。

### P5：会话动作与对应页面输出

- Ammolite：AIOpsDiagnosisView.vue、AIOpsRecordsView.vue、AIOpsDiagnosisCasesView.vue、
  AIOpsOperationsKnowledgeView.vue、AIOpsReportDetailView.vue、相关类型与通知路由。
- KBot：ui/aiops/js/aiops-workspaces.js、aiops-records.js、aiops-cases.js、aiops-operations-knowledge.js、
  aiops-inspections.js、aiops-report-detail.js，以及对应 HTML、资源版本和缓存引用。
- 将“生成会话报告／诊断收口”改为“生成报告／沉淀案例”，保留模板、轮次预览和人工归纳等必要确认。
- 提交后提示可离页，不等待结果或强制跳转；会话下显示两类任务、阶段、去向和确切结果链接。
- 报告中心增加生成记录和报告分类；案例库增加提炼记录、可完善草稿及审核入口。
  任务与产物分区，避免一条任务和一份结果被统计成两份报告／案例。
- 刷新、重进、轮询或通知更新均从服务端恢复；运行中只做有界刷新，不依赖组件内存。
- 任务读取／结果跳转重验权限；若结果页权限不允许，明确提示，不通过通知绕过。
- 失败可看简要原因与合法重试；进度／报告完整度／案例发布／实际效果分别表达。
- 报告分类以共享显式枚举映射和服务端筛选实现；不要仅在当前 100 条前端数据上假装全量分类或分页。
- 上传资料仍只管理 MANUAL，配置入口只提供关联跳转，不在内容页复刻解析模型配置。
- 测试：两个按钮独立、只做一种或同时提交、离页返回、错误重试、同会话多任务、
  空会话、较早活动轮次、WAITING_USER、失败／取消无有效结果、权限不足和可见禁用原因，
  PARTIAL 报告与草稿审核、资料／案例互不混列、巡检与报告中心共用 Report ID。

完成门禁：会话、结果页和通知三处看到同一任务及同一产物，不用额外任务中心侧栏补洞。

### P6：合同、回归与两仓联调

- 同批更新 packages/platform_clients、两层 API、共享合同和 docs/openapi/aiops_public_v1.json／aiops_internal_v1.json。
- Ammolite 公共入口主要在 services/main_api/src/main_api/api/ops.py；
  KBot 报告入口在 api/ops.py，案例入口在 api/aiops_app.py，不遗漏其中一路。
- 更新既有报告、案例、会话收口、Outbox、通知与 Main API 边界单测，并新增输出任务专项单测／集成测试。
- Ammolite 重点复用 test_aiops_formal_reporting.py、test_aiops_conversation_case_closure.py、
  test_aiops_report_case_extraction.py、test_aiops_notification_subscription.py、test_notification_reconciliation.py，
  以及 Portal 的 aiops-conversation-closure.spec.mjs、导航及信息架构合同。
- KBot 同名后端测试、tests/contract/aiops_shell_navigation.mjs、aiops_page_alignment.mjs、
  aiops_conversation_list.mjs 及会话交互用例同批适配。
- 旧同步回执、自动跳转和旧菜单断言直接替换，不保留双接口、双读写或临时兼容行为。

完成门禁：代码、规范 Schema、客户端、OpenAPI、测试和文档均使用同一异步合同。

### P7：真实环境验收与发布

按用户最新要求“真实测试由我手动做，你只负责改代码”，本阶段交由用户执行，
不作为代理继续部署、测试或请求外部写入授权的待办。下列内容是手动交接清单，不表示已通过。
结构／Prompt 的既有授权执行结果见第 6.20 节；本轮不新增数据库、部署、重启或资源变更。
真实验收至少覆盖 PostgreSQL／Oracle 持久化、真实模型及 KC、通知投影、两个浏览器前端。

必须演练：提交后关闭页面、API／Worker 重启、通知消费中断再恢复、同输入并发、
同会话先后两类输出、来源撤权与更正、KC 长时处理／故障、失败重试和深链重新鉴权。
另行执行“会话沉淀 → 真实 KC 解析／索引 → 草稿通知 → 人工发布 → 新会话检索命中 →
引用预览 → 来源更正／撤回后阻断”，核对措施、效果与禁忌的返回关系。
验证正式产物数、案例版本数、任务终态和通知数，不能只看 HTTP 202 或页面按钮。
不将替身测试、构建成功或进程 readiness 写成真实闭环已验收。

## 4. 依赖与推荐实施顺序

P1 → P2 与 P2-K 联合完成 → P3；P4 可在 P1 后独立开发，P5 依赖 P1–P4 及 P2-K 的合同与查询能力；
P6 贯穿各阶段，P7 在所有开发门禁之后。
菜单可以先本地验证，但最终同批发布，不把“前端可点、后台尚未接通”作为完成交付。

不需要重建报告、案例库或资料解析引擎，不新增平行 Agent 或通知中心产品入口。
案例提炼／脱敏如需调整 Prompt，沿现有配置和开发 1.0.0 基线维护，不覆盖网页手工版本。
主要新增成本是 AIOps 业务输出任务、恢复机制、案例知识投影和通知资源定位；
主要改造成本是既有同步调用链、两端列表、KBot 通知前端挂载与 KC 候选召回／上下文链。

## 5. 验证执行清单

| 层次 | Ammolite | KBot4 |
| --- | --- | --- |
| 架构 | conda run -n cube python tests/acceptance/check_architecture_boundaries.py | python3 tests/acceptance/check_4_0_boundaries.py |
| 跨仓语义与 OpenAPI | tests/acceptance/check_aiops_cross_project_alignment.py、check_openapi_contracts.py | 同名检查，使用本仓环境 |
| Schema 离线／环境检查 | check_aiops_entity_schema.py、check_postgresql_entity_schema.py；真实 PostgreSQL acceptance 另行记录 | check_aiops_entity_schema.py、check_oracle_schema.py、check_oracle_entity_schema.py；实库检查按脚本前置条件与授权执行 |
| 后端 | 按本仓环境运行输出任务、报告、案例、通知和 BFF 专项，区分替身与真实依赖 | 同一语义专项及 Oracle 持久化测试 |
| 知识质量 | 案例语义召回、候选不预截断、上下文完整、精确版本引用、发布／撤回和手册回归评测 | 同一评测集与质量门禁，不以一仓结果代替 |
| 前端 | apps/portal-web 下 npm run build，Portal 交互与导航合同 | Node 语法、静态页面／导航／交互合同和浏览器验收 |
| 真实闭环 | 发起、持久化、后台恢复、结果页面、通知及深链逐项留证 | 同样留证，不以另一仓通过代替 |

执行前查看各测试脚本帮助和环境前置条件；未运行或因依赖缺失跳过的项明确记录。
以下执行记录仅覆盖明确列出的项目；其余仍待运行，不代表真实业务闭环已经通过。

## 6. 实施记录（2026-11-03）

本节按批次保留当时状态，不累计通过数。当前状态以最后一节为准；
第 6.10 节已完成显式报告生成的异步合同与离线验证，替代此前报告仍同步的中间态记录。

P1 已完成部分：共享输出任务合同、独立任务／尝试实体、仓储、UoW 接线、原子受理与领域
Outbox、同源幂等、发起人作用域、租约／重试领域规则和分页查询应用基础；两仓规范持久化工件同步。
KBot 已生成非破坏性的 Oracle 55→56 升级工件，未实际执行。案例资源初始化验证仍未完成。

已运行：Ammolite 输出任务与 PostgreSQL 持久化专项 31 项通过，离线 AIOps metadata 85 表检查通过；
KBot 使用专用 kbot4 环境运行输出任务、升级工件及 Oracle 持久化专项 35 项通过；
Oracle 规范 DDL、重建／验证／升级生成工件一致性、两仓架构与跨项目语义检查通过。
这些专项含事务替身与 SQL 编译检查，不是实库并发或恢复验收。

实库只读验收限制：PostgreSQL 尚未应用新任务表，且发现已有巡检／报告字段漂移；
本地 Oracle 127.0.0.1:1521 监听不可达。未应用 DDL、未部署、未重启服务。

### 6.1 P4 菜单与命名

两仓已落实 Agent 工作台固定入口、运维工作、独立报告中心、知识资产及配置与治理的顺序。
配置区七项名称、顺序、路由和权限不变；资料内容页更名运维资料库，巡检批次更名巡检记录。
Portal 已支持有序混排的分组与独立链接，保留其他 App 的导航回退；两端详情页归属保持一致。

验证：Portal 生产构建通过；Portal 信息架构与 Main API Domain 合同 48 项通过；
KBot 导航 Node 测试 16 项通过。隔离浏览器对照通过案例、资料、报告和巡检 4 页的
空／有数据及至少 3 种宽度，包含报告 PDF、8 类报告表单以及草稿／待审核／发布案例的正文预览。
本轮浏览器工件 /tmp/aiops-page-alignment-YuUnGl；这是替身 API 的两端交互验证，不是线上验收。

### 6.2 P2-K 案例知识适配

已同步实现：

- 明确检查／Claim 来源、逐措施效果与语义单元合同；确定性编译脱敏正文、稳定章节、
  正文哈希和关联单元，接入会话案例、报告案例及效果案例生产者。
- KC 的 AUTO 策略按检测到的结构化文本类型选择原生 TEXT 档，不要求 OCR／图像模型；
  Collection 策略优先级和管理员显式策略保持。复用现有结构 IR，不新增解析器。
- 发布候选使用稳定 keyset 分页，与最终 Top-K 分离；KC Discovery／Evidence 采用精确 Revision
  范围及分片预算，来源和环境适配分开处理，候选覆盖不全时显式警告。
- 检索前后复核 Agent 当前状态、目标授权、知识开关、来源类型、有效知识库绑定、必选标签及安全上限；
  批量排除已删除／归档知识库；知识目录输出也使用同一当前授权和确切发布版本围栏；
  案例正文核对 CONTENT 哈希、文档版本、已批准／就绪和撤回状态，命中时带出必要上下文。
- 来源后续轮次使用 keyset 分页复核，不因无关新轮次或超过固定 100 轮就判来源失效；
  真实反证、更正和撤权仍阻断。冻结标题参与投影与脱敏封印，资产改名不篡改旧发布正文。
- 独立管理正文预览贯通 Main API、窄客户端、AIOps 和既有 KC Document Preview，
  两端案例详情提供纯文本预览。正式 search-preview 明确要求选择 Agent，不把管理用户冒充为 Agent。
  KC 状态查询增加既有 withdrawn_at 的返回，不改变其持久化模型。

此前批量离线回归：Ammolite 422、KBot 407 项通过，覆盖知识管理／候选／生命周期、来源更正、
效果案例、投影、会话收口、报告提炼、治理、HTTP 传参、任务基础、Turn 应用、
报告生命周期，以及 KC 解析策略、Revision 状态、受理、解析 Worker 和预览服务。
两仓架构边界、跨仓语义、OpenAPI 快照校验通过；Ammolite 离线 metadata 85 表检查通过，
KBot 规范 Oracle DDL 检查通过（9 个服务、40 个脚本）。
KBot 实库 Entity 检查因本地 Oracle Listener 不可达退出，未通过；没有用离线 DDL 检查替代实库结果。
这些是离线单测／事务替身／SQL 编译与隔离浏览器检查，不等价于检索相关性质量评测或真实 KC 入库。

### 6.3 P2 执行与授权基础、会话案例冻结

本节记录当时的能力与测试；之后的 Worker 装配、终态和 API 进度以第 6.6 节为准：

- workers/output_jobs.py 实现持久化领取、短事务／值快照、心跳、过期执行围栏、
  阶段恢复定位、索引等待释放租约、取消／崩溃恢复、有界临时故障重试及总等待／执行时限。
  最近尝试按上限读取，避免索引长期等待时反复加载完整执行历史。
  当前授权、冻结输入处理器和事务终态接收方均为必需依赖，没有默认日志成功实现。
- Outbox 明确识别 OPS_OUTPUT_JOB_REQUESTED，核对持久化任务的 Tenant／Domain／Actor 后唤醒；
  没有消费者时明确失败，不能被 LoggingOutboxSink 标记投递成功。轮询可恢复丢失的唤醒。
  此时 Bootstrap 尚未提供实际消费者，因此不能声称后台已能处理用户生成命令。
- Main API 新增仅限指定 Worker 的 POST /internal/v1/platform/aiops/output-authority。
  双 JWT、调用方、scope、SERVICE 主体和 audience 均校验；接口独立生成内部契约，不进入公开快照。
  Ammolite 复用当前账号／Tenant 成员、App 安装／模块／许可和 Domain 角色规则；
  KBot 复用当前用户、有效 Domain、App 成员／角色和用户安全级。
  会话沉淀沿现有 aiops:use，报告提炼保留 aiops:knowledge_manage，不额外授予业务权限。
  现有机器路由目录没有开放这些生成命令，本次也不以伪造人类来源替机器扩权。
- 窄客户端每次请求新签服务 JWT，不保存用户 JWT 或会话；执行授权适配器拒绝缺少服务端
  冻结来源、过期／异主体快照、权限撤销及安全级收窄。业务来源围栏仍由实际处理器负责。
  两仓拓扑增加客户端 main_api 绑定；非默认 Worker 服务名由 Main API 的
  api.output_authority_caller 显式配置，与真实 Worker service_name 对齐。
- 会话案例拆为 prepare_conversation_case、freeze_conversation_case_input 和
  execute_prepared_conversation_case；冻结阶段只读已有来源、人工归纳和精确 Agent 清单，
  不调用模型／KC。执行只读取冻结所选原件，不再重新预览全部会话。
  新受理仍拒绝活动会话；已受理后新增无关活动轮次不干扰处理，真实反证／更正和撤权继续拒绝。
  受理时间与预分配产物 ID 不进入业务输入哈希；恢复时必须复用已登记的确切产物 ID。
  现有同步调用方已共用该拆分，但公开生成命令本身尚未切换为异步任务。

专项验证覆盖双 JWT 拒绝、当前权限变化、独立报告／案例权限、冻结输入完整性、幂等时钟、
无关新轮次、原件更正、租约抢占、取消恢复、索引等待、通知事务回滚、临时故障上限及执行超时。
这些使用事务／HTTP 替身，不代表生产 Worker、真实 KC 或通知已接通。

扩展批量回归：Ammolite 644 项、KBot 591 项通过，各有 7 项子测试通过；覆盖原知识适配／KC
回归，并加入输出执行器、当前授权 HTTP 边界、冻结输入、既有 Task Worker、配置、Main API
及 Domain 合同；Ammolite 另包含安全与 App Domain 合同专项。
首次扩展运行发现 Ammolite 的新依赖误用了服务配置名作为端点别名，已改为拓扑中的 main_api，
随后完整重跑通过。两仓架构、跨仓语义和 OpenAPI 再验通过；Ammolite metadata 85 表、
KBot Oracle 规范工件 9 个服务／40 个脚本检查通过。没有重新执行浏览器或实库验收。
Main API 内部授权接口已生成独立内部快照，公开契约不暴露该路径。

### 6.4 会话案例的可恢复入库处理器

新增 application/conversation_case_output.py 的事务受理回调与
ConversationCaseOutputProcessor，复用现有 case_closure、Registry 和 reconcile_version，
不再只提供抽象执行循环。尚未装配 Bootstrap、公共命令和事务终态接收方，不能据此声称已上线异步链路。

本节记录的是处理器批次的验证边界；之后工作区新增了终态接收方和通知初稿，未包含在本节测试结果内，
其后验证与当前状态见第 6.6、6.7 节。

- 受理回调只读取并冻结用户所选来源，核对真实会话发起人；模型和 KC 均在处理器阶段执行。
- 首次执行先持久化闭环产物 ID；KC 受理前保存 CaseIntakeRecovery，
  固定脱敏标题／Profile、正文哈希、来源指纹及 Collection。恢复时验证这些封印，
  不重新调用模型；外部回执丢失时用相同正文及幂等键重放，回执已保存则不重复上传。
  Registry 写回后恢复优先按确切收口 Artifact 定位，不能因同会话后来已有其他案例版本而重复写入原产物。
- Worker 增加独立外部回执登记：当前租约／作用域仍须有效，登记本身不依赖刚被撤销的执行权限；
  不能借此批准索引或成功。已受理恢复时先登记 Registry，再处理原件失效及精确撤回。
- 后台使用 SERVICE 调 KC，业务署名显式使用当前已核验的任务发起人，不伪造 USER／PORTAL，
  不把服务主体记为案例创建人。执行前后核对实际来源 Agent 清单及 ACTIVE 状态，
  不能把 Main API 回显 Agent 标识当作来源有效性证明；既有发布与检索来源复核也补齐 ACTIVE 校验。
- 平台实质撤权时不再批准 KC，并保留待审核版本与撤回命令；
  授权查询临时故障只等待重试，不把暂时不可用误判为永久撤权而撤回。
- KC 受理或 PROCESSING 只返回 INDEXING 等待材料；索引就绪后再核验正文投影、
  精确 Revision、APPROVED／READY、未撤回及 CONTENT 哈希／文档版本，
  复核来源后才返回终态材料。不会自动发布案例，仍由后续人工审核决定 Agent 可检索性。

专项包含真实案例应用和持久化 Worker 的组合执行，KC、授权、通知及事务仍为替身；
覆盖断点、相同正文重放、外部回执登记后撤权、旧租约拒绝写回、确切索引等待、
无关活动轮次、原件失效撤回及错误版本／哈希／审批／撤回状态拒绝。
本轮最终批量回归：Ammolite 513 项、KBot 511 项通过；包括案例、知识、输出任务、
报告生命周期／来源、KC 解析策略／Revision 和当前授权 HTTP 专项，属于本次独立批次，
不与此前 644／591 项累计。两仓架构边界、跨仓合同、OpenAPI 和 git diff --check 通过；
Ammolite metadata 85 表、KBot Oracle 9 个服务／40 个规范脚本检查通过。
未运行真实数据库／模型／KC 入库或浏览器端到端验收，未部署、未重启。

### 6.5 仍需完成

P2 报告来源与模板冻结、实际报告处理器、报告提炼案例的同类异步改造；
P3 通知中心路由、权威对账及 KBot 共享通知前端接线；
P5 报告按钮异步回执与全量报告分类筛选、两类生成记录与产物列表的完整联动；
案例 Collection、模型和 Domain 规范初始化／配置路径核验；
检索质量评测以及 P7 实库、真实模型／KC 和通知端到端验收仍待完成。
明确禁忌目前随上下文保留，结构化适用性决策与排序质量尚未验收，不能声称已完成语义禁忌硬过滤。
会话案例已使用异步任务；报告与通知链仍未形成完整异步合同，不得将案例或菜单通过视为两个动作均已交付。

### 6.6 本轮接线、终态与页面验证

以下记录覆盖第 6.4 节之后的代码，替代此前“终态初稿未测、Worker 未装配、无任务查询”的现状判断；
第 6.1–6.4 节的批次通过数保留原验证边界，不累计到本批。

已同步接通：

- OutputJobFinalizer 与实际 Worker 的事务终态集成：任务类型与完整结果定位校验、确切索引／正文哈希、
  来源复核、KC 已受理未登记 Registry 的失败恢复、版本再锁定及并发发布保护。
  通知写入或事务提交失败时回滚终态；不通过可选通知依赖掩盖测试接线问题。
- 实际 bootstrap/output_jobs.py 构造当前授权客户端、SERVICE 执行上下文、会话案例处理器、
  事务终态接收方及输出 Worker；bootstrap/worker.py 注入真实 Outbox Sink，接入启停及健康检查。
  Ammolite 同时接入既有 Worker 指标。身份来自各自平台合同，不伪造人类主体，不借用模型。
- 两层 GET /output-jobs、GET /output-jobs/{job_id} 和 POST /output-jobs/{job_id}:retry，
  窄客户端、发起人／Domain／Tenant 隔离、版本受控重试及冻结来源复核；
  报告提炼案例重试仍按权威来源要求 knowledge_manage，不相信浏览器提供的任务种类。
  普通回执读取不返回冻结正文／KC 定位；实际结果页独立重新授权。
- 会话 :close-diagnosis 已切到 submit_conversation_case，要求 Idempotency-Key，
  事务冻结后返回 202 OutputJobReceipt；HTTP 专项确认受理不调用模型、KC 或现场取证。
  双击复用任务，但幂等命中仍重验原件。损坏的恢复 ID 映射为受控拒绝，不泄漏内部异常。
- 两端会话、报告中心、案例库新增同语义“我的生成记录”，复用当前任务查询／重试。
  会话采用可折叠摘要，不占用侧栏入口；提交案例后留在会话，可离页后重新读取。
  列表按服务端游标翻页，处理中有界刷新，隐藏页暂停，连续读取失败停止自动轮询。
  展示阶段、失败原因和合法重试；任务数不计入报告／案例资产数。
- 两端按钮命名已改为“生成报告／沉淀案例”，保留原模板与人工确认字段。
  案例按钮已使用异步回执；报告按钮当前仍走既有同步生成，不能据此声称两动作均已异步完成。
- 案例详情新增可选 asset_version_id，逐层保真传递并校验版本属于所请求案例；
  不匹配明确拒绝，不静默回退最新版本。生成记录使用确切案例版本链接，
  Portal 同页查询参数变化也会重新读取。该能力尚不代表通知深链接通。
- 任务通知事件适配器已验证 task operation_id、安全摘要、确切时间／受控版本，
  草稿事件拒绝错误类型或不完整定位；通知中心路由、权威对账及 KBot 前端仍待后续接线。

本批后端专项：Ammolite 188 项通过；KBot 193 项及 3 个子测试通过，覆盖真实应用／Worker、
HTTP／BFF／客户端、终态、通知适配器、来源重试和知识预览，外部依赖及事务仍为替身。
KBot 原 BFF 单测仍导入已退役固定知识聊天合同，已改为验证统一 Agent 入口，
未恢复旧生产接口。新增会话异步受理／确切版本测试已纳入跨仓同语义清单。

隔离浏览器专项 9 项通过，直接使用两仓真实会话确认处理函数、Portal 任务组件和 KBot 任务模块，
覆盖选择轮次／Target、请求幂等键、202 后留页、索引等待、草稿结果链接、错误文本安全、受控重试及浏览器缓存页恢复。
Portal 生产构建通过；KBot 新增及修改模块 Node 语法检查通过。
这些是替身交互，不是在线页面或真实 KC 验收。

两仓架构、跨仓语义与 git diff --check 通过；Ammolite 离线 metadata 85 表通过，
KBot Oracle 规范 DDL 9 个服务／40 个脚本通过。
KBot check_aiops_entity_schema.py 再次因 127.0.0.1:1521 Listener 不可达退出，实库检查未通过；
未创建或修改数据库，未部署、重启、提交或推送代码。
最终复核：Ammolite 信息架构、OpenAPI 和案例收口合同 37 项通过；
KBot OpenAPI 合同 5 项通过、静态页面合同 60 项及导航 Node 专项 16 项通过。
Portal 最后一次同页查询监听和可折叠任务摘要修改后的生产构建通过，仅有既有大包提示。
这些结果不包括尚未重跑的页面视觉对齐程序，不用构建替代视觉或真实环境验收。

### 6.7 下一步具体修改顺序

| 顺序 | 当前缺口 | 修改位置与内容 | 完成门禁 |
| --- | --- | --- | --- |
| 1 | 显式报告生成已切换异步；当前离线验证见第 6.10 节 | report_preparation.py、report_output.py、组合 Worker、两层 API／客户端、通知适配器及两端提交处理器已接线；移除旧同步运行时生成方法 | 冻结确切模板与已选原件、202 和 Worker 原子终态专项通过；真实数据库与通知消费闭环仍须 P7 验收 |
| 2 | 报告详情沉淀案例已切换异步；离线处理验证见第 6.11 节、整页视觉修正见第 6.16 节 | report_case_output.py、case_intake.py、共用 CaseOutputProcessor、finalizer、两层 API／客户端及报告详情 UI 已接线；移除旧同步 extract_case | 冻结报告与原件、共用入库恢复／草稿就绪条件、跨入口复用及报告整页对照通过；真实依赖仍待 P7 验收 |
| 3 | 固定案例资源定向初始化与模型管理路径已补齐，离线验证见第 6.12 节 | 规范初始化、provision_aiops_case_collection.py、现有案例模型 GET／PUT 和 KC 管理合同；保持原配置表单 | 域隔离、只补缺、TEXT 基线、配置保留与事务专项通过；真实部署／模型调用及 KC 索引闭环仍须 P7 验收 |
| 4 | 通知精确定位、两端消费和版本化权威对账已补，见第 6.13–6.14 节 | Portal 通知定位／空间守卫、KBot scoped Outbox 路由和共享顶栏；来源版本投影、窄 SERVICE 状态查询、当前授权和 CAS 写回 | 离线重放／乱序／迟到／重试、核权和慢响应围栏专项通过；真实 Outbox 积压、数据库并发与 P7 全链仍待验收 |
| 5 | 报告分类、分页和任务完成刷新已补，见第 6.15 节 | 穷尽共享类别合同、两层类型化 API／客户端、仓储交集查询、签名游标、两端 Records、任务确切链接和浏览器合同 | 服务端先筛选后分页、独立计数和抗竞态专项通过；真实环境中两个按钮分别／同时操作、通知和成果联动仍须 P7 验收 |
| 6 | 结构化适用性、业务重排与检索评测工具已补，见第 6.16–6.17 节；真实质量与依赖证据不足 | case_applicability.py、knowledge_search.py、knowledge_ranking.py、Prompt、tests/evaluation 及独立测试环境清单；P7 须另行授权 | 离线门禁不替代真实模型语义准确性、KC 召回／排序与发布／撤回全链留证 |

明确禁忌已接入版本化决策及当前环境核验，处置用途有保守门禁；实际语义准确性和排序质量
尚未用真实模型／KC 评测证明，不能宣称自然语言禁忌已全面准确过滤。
已有解析／投影和离线检索测试也不等于真实模型召回质量验收。
配置区七项导航及既有资料入库流程保持；资源配置问题在原边界内解决，不增加技术 Collection 选择页。

### 6.8 案例固定集合模型配置后端

两仓已新增 GET／PUT /operations-knowledge/case-collection/models，经现有公开 App 前缀、
窄客户端和内部 API 到达固定案例集合；公开需 aiops:knowledge_manage，内部需 aiops.manage。
不会接收手册 knowledge_base_id，也不由浏览器传入 Collection ID 或业务空间。

案例绑定要求 parser_llm、retrieval_llm、embedding 三个角色显式指定；
复用各平台当前可用模型目录及 KC 变更规则，不隐式借用、不自动选模型、不绕过 Embedding 不可变限制。
资料库仍要求原有四个模型角色，案例不增加 OCR／视觉依赖。
读取仅报告 CONFIGURED／MODELS_REQUIRED／TEXT_POLICY_REQUIRED，不代表真实模型健康、索引就绪或已发布。
管理员显式 VISUAL／未知策略会被诊断并阻断案例模型保存，不静默改写策略。

本批新增模型配置／资料回归／案例受理专项：Ammolite 63 项、KBot 56 项通过；
其中新增共享案例集合配置测试各 20 项。OpenAPI 重新生成并校验：
Ammolite 18 个快照／971 条路径，KBot 16 个快照／827 条路径。
未增加菜单或改变现有配置表单；固定资源规范初始化、配置页面完整接线及真实依赖验收仍未完成。
当前 API 在资源缺失／重复／身份不符时明确拒绝，不自动创建或挪用任何手册集合。

本批最后回归：Ammolite 129 项、KBot 162 项通过，覆盖 OpenAPI、页面合同、
案例模型配置、真实会话案例处理器、知识投影、候选范围及异步受理；
两仓架构、跨仓语义和 git diff --check 通过。测试使用替身，不代表实库／模型／KC 已验收。

### 6.9 追加 KC 定制需求后的工作区复核与计划修订

本轮仅修改设计与实施计划，保留两仓已有未提交代码，不继续实施或修复生产代码。
产品设计补清案例知识字段到正文、召回和上下文的映射，以及“入库属于沉淀任务、发布属于独立治理”的边界；
配置区原七项及表单保持不变。完整端到端仍未交付。

当前报告改造处于中间态，不能沿用第 6.8 节通过数证明当前工作区通过：

- 已有 report_preparation.py、report_output.py 和 output_processing.py；Worker 工厂已要求
  runtime_service 并装配案例／报告组合处理器，不再准确地表述为“Worker 只装配案例处理器”。
- 两仓内部 POST /reports:generate 仍为 201 并调用同步运行时生成方法；前端异步回执尚未切换。
- 新报告终态代码向 emit_report_ready 传 output_job，但两仓通知仓储的该方法尚无此参数。
  此分支的成功事务尚未接通；该结论来自代码核对，不是已完成真实报告执行实验。
- test_aiops_output_bootstrap.py 仍按旧工厂参数和旧处理器结构调用，需与组合处理器合同一起更新，
  并补真实报告处理器／终态／通知失败回滚测试，不能只修改测试以绕开缺少的生产依赖。

本次实际执行两仓相同四组单测：test_aiops_case_collection_models.py、
test_aiops_case_knowledge_projection.py、test_aiops_knowledge_candidate_scope.py、
test_aiops_output_bootstrap.py；Ammolite 与 KBot 各为 74 passed、2 failed。
失败均为 Worker 工厂新增必需 runtime_service，而两项装配测试尚未传参；
测试在该处中断，不能据此判断后续组合处理器执行已经正确。
这批测试使用替身，未进行真实数据库、模型、KC、通知或浏览器端到端验收。
两仓 check_aiops_cross_project_alignment.py 与 git diff --check 通过，
本批三份共享设计／计划文档逐字一致；这些检查不替代上述失败测试或未完成报告链的验证。

后续按第 6.7 节完成报告合同与装配、报告提炼案例、案例资源配置、通知深链和列表联动；
案例检索以专项设计第 3.2、7.1 节补充质量门禁：资料／案例身份不混淆、建议与实际执行分开、
逐项效果与禁忌随措施返回、同源不重复计证、环境冲突不推荐、未知不冒充适用。
最后真实演练“沉淀 → KC 就绪 → 草稿通知 → 人工发布 → 新会话检索 → 精确引用 → 撤回阻断”。
本轮未应用 DDL、部署、重启、提交或推送。

### 6.10 显式报告生成异步闭环与回归

以下替代第 6.5–6.9 节中“报告按钮仍同步、报告通知参数不匹配、Worker 装配测试失败”的中间态，
不代表报告提炼案例、通知中心或真实环境已经全部完成。

已同步实现：

- 两层 POST /reports:generate 返回 202 OutputJobReceipt，要求 Idempotency-Key；
  受理事务冻结来源、已结束会话轮次、确切模板版本与内容哈希、周期范围和对照报告依赖。
  受理不调用模型、KC 或目标取证；相同键重放仍核验来源。
- 实际 ReportOutputProcessor、OutputJobFinalizer 和组合 Worker 共用既有报告装配及发布代码；
  来源身份、Agent／Target／来源绑定、原件哈希、安全级、模板有效性、Turn 归属和后续反证
  在执行及终态复核。无关后续轮次不扩大冻结范围；来源实质变化阻断。
- 精确模板版本不随当前指针切换；模板定义损坏或停用阻断。
  周期对照依赖作为确切报告原件登记与重验，不静默改用新报告。
  幂等复用报告还核验报告及正文安全级，不把低密级旧产物用于高密级来源。
- 报告产物、任务终态和平台通知 Outbox 同事务；通知写入失败时报告回滚，
  重试与重复消息不多生成产物。READY／PARTIAL 均可成功完成输出任务，
  PARTIAL 的通知只提示缺口，不携带完整诊断正文。
- 显式提交无论来自会话、告警还是计划运行，都给真实发起人发送任务级完成通知，
  保留自动报告原有订阅逻辑；通知携带 job_id 与确切报告版本。
  已移除 runtime/service.py 的旧同步 generate_user_report／generate_conversation_report 及回执类。
- 两端报告提交只等待受理，关闭确认窗后留在会话，刷新生成记录并提示可离页；
  不读取同步 report_id 或自动跳转。生成记录保留确切报告、版本和任务链接。
  报告内容回归直接测试准备／发布阶段，完整受理和 Worker 另有专项，未保留生产兼容入口。
- OpenAPI、跨仓同语义清单、装配和旧报告回归同批更新。

本批验证：

- 报告、输出任务、案例、来源、通知订阅及 Main API 当前授权的大范围替身回归：
  Ammolite 571 passed；KBot 582 passed、3 subtests passed，非实库／模型／KC 验收。
  新增报告专项在两仓各 29 项通过，已包含在上述大范围计数内，不叠加。
- 隔离浏览器 15 项通过：执行两仓真实提交处理器及任务组件，覆盖案例选轮次、
  报告确切模板与幂等键、提交中防重复、受理失败恢复、202 后留页、索引／装配进度、
  确切结果链接、错误文本安全及重试。HTTP 为替身，不代替通知消费或真实 Worker 验收。
- Portal 生产构建通过；KBot 相关模块 Node 语法检查、导航与会话 Node 专项 17 项通过。
  Ammolite 信息架构／OpenAPI／诊断页面合同选择性回归 61 passed、2 deselected；
  完整诊断页面合同另有 2 项失败：旧测试仍要求巡检列表变量 run 及旧模板“尚未可用”字面表达，
  当前页面已改为 selectedRun／独立编辑器。这两项未修复，不能将该完整套件记为通过。
  KBot OpenAPI／静态页面合同另行回归 65 项通过。
- 两仓架构边界、跨仓语义和 OpenAPI 快照校验通过：
  Ammolite 18 个快照／971 条路径；KBot 16 个快照／827 条路径。
  Ammolite 离线 metadata 85 表及 KBot Oracle 9 个服务／40 个规范脚本检查通过。
  KBot 实库 Entity 检查仍因 127.0.0.1:1521 Listener 不可达退出，实库门禁未通过。

剩余按第 6.7 节第 2–6 项推进：报告详情提炼案例异步化、固定案例资源规范初始化与模型维护路径、
通知中心精确深链／当前授权对账及 KBot 共享通知组件、服务端报告分类和列表联动、
结构化适用性决策与检索质量评测、真实依赖验收。完整目标仍在实施中。
本批未应用数据库工件，未部署、重启、提交或推送。

### 6.11 报告详情沉淀案例异步化与共用 KC 管线

本节替代第 6.7／6.10 节中“报告详情沉淀案例仍待异步化”的状态，
不代表固定资源配置、通知中心消费、检索质量或真实环境已经全部完成。

已在两仓同步实现：

- 两层 POST /operations-knowledge/reports/{report_id}:extract-case 返回 202 OutputJobReceipt，
  要求并透传 Idempotency-Key；缺失返回 428。受理事务冻结确切报告 ID／版本／正文哈希、
  证据清单及安全级、目标版本、运行身份／计划和 Profile，不调用模型、KC 或目标取证。
  重放受理仍重验来源；报告、原件、安全级、绑定或实际授权变化时拒绝。
- 提取 case_intake.py 作为会话与报告共用的生产脱敏、知识正文投影、KC 受理、回执登记、
  审批和断点恢复管线；来源适配仍各自负责冻结／校验／Registry，不复制解析器或检索引擎。
  已移除 OperationsKnowledgeService.extract_case 的旧请求内重处理方法。
- 共用 CaseOutputProcessor 等待 KC 确切 Revision 已批准／READY、CONTENT 文档与正文哈希一致，
  经业务 Registry 对账才允许形成任务成功。人工发布仍独立；草稿不进入 Agent 发布白名单。
  报告版本的 source_hash 保留原报告正文哈希；恢复封套的 intake_id 统一表达来源入库身份，
  会话闭环 Artifact ID 仍是其自身生产者的恢复定位，不保留旧字段兼容路径。
- 终态及重试重新核验冻结来源。受理已发生但 Registry 未登记时，失败终态先恢复确切登记；
  临时故障保留恢复能力，实质撤权不再批准索引，并持久化精确撤回请求。
  并发多余 Revision 作为退役重复项登记并撤回，不遮蔽有效版本，恢复重放不再次创建重复项。
- 纯标题／版式报告修订可复用原案例；报告引用了已有会话闭环案例时同样复用确切版本。
  终态证明它属于本次冻结来源允许的复用范围，不改写原案例请求 Hash／来源指纹，
  不以取消一般结果校验来允许任意版本。资产安全级不能低于任务冻结原件。
- 两端报告详情统一“沉淀案例”：提交中防重复，202 后留在报告阅读页，提示草稿就绪后通知，
  提供指向案例库指定 job_id 的生成记录链接，不从受理回执读取尚不存在的 asset。
  API／客户端、OpenAPI 和跨仓共享语义清单同批更新；配置菜单与既有配置表单未改版。

本批实际验证：

- 大范围替身回归：Ammolite 687 passed；KBot 698 passed、3 subtests passed。
  包含两仓各 27 项新增报告案例异步专项，覆盖真实 HTTP 受理、处理器、Worker、
  通知失败的终态回滚、来源／安全级变化、外部受理恢复、多余 Revision 撤回、
  展示修订复用，以及真实会话案例被报告复用；这些数字不重复相加。
- 隔离浏览器 19 项通过，执行两端生产提交处理器／任务组件。
  其中新增报告详情案例 4 项覆盖仅等待受理、留页、确切任务链接、提交失败和同键重试；
  HTTP 为替身，不等于真实后台或通知中心验收。
- Portal 生产构建通过，仍有已有大 Bundle 警告；KBot 改动脚本和页面对照程序 Node 语法检查通过。
  先前第 6.10 节的两项旧页面合同按现有 selectedRun、独立巡检模板编辑器及实际质量标签更新，
  完整诊断页面合同通过，已计入 Ammolite 大范围回归。
- OpenAPI 重新生成并校验通过：Ammolite 18 个快照／971 条路径；KBot 16 个快照／827 条路径。
  两仓架构边界、共享语义与 git diff --check 通过；未新增本批持久化表或数据库工件。
- 额外执行 report-detail 整页真实组件浏览器对照，交互完成，但整体视觉断言未通过：
  440 项差异，主要为日／月巡检报告的背景、字体和布局。截图及差异记录在
  /tmp/aiops-page-alignment-UUq6Ik。此项尚未修复，不能把提交处理器测试说成整页视觉已验收。

下一步按第 6.7 节第 3–6 项继续：固定案例资源规范初始化及现有管理能力下的模型维护、
通知中心精确深链与当前授权对账、报告服务端分类和完整列表联动、适用性／检索质量，
并修复整页视觉对照差异。真实数据库、模型、KC 及通知消费闭环仍须另行留证。
本批未应用 DDL、部署、重启、提交或推送，完整目标仍处实施中。

### 6.12 固定案例资源规范初始化与定向维护

本节更新第 6.7 项 3 的离线状态，不代表已执行真实数据库初始化或模型／KC 验收。
两仓新增 scripts/deployment/provision_aiops_case_collection.py，限定明确 Tenant／Domain 和审计操作者，
写模式锁定父空间，仅补缺失固定案例资源；check-only 只读，无 DML、行锁或提交。
新资源为 TEXT、空模型映射，不借手册或 Agent 模型，不上传内容，不改变发布条件。

- Ammolite 校验活动 App 安装、Tenant／Domain 归属和固定 external_key／所有权身份；
  同事务补所属空间 manage 绑定。跨空间绑定、consume 权限、重复、停用或身份冲突明确失败，
  不自动重新启用、合并或提权。规范 init_postgresql.py 在建表和平台边界对齐后为当前已启用空间补缺。
  全新空库没有业务空间时不造账号／Domain；后续启用新空间须按部署清单执行定向入口。
- KBot 校验活动 AIOps App、独立活动 Domain，拒绝 KM／其他 App 共用空间。
  首次 bootstrap 对新案例资源采用 TEXT；已有案例仅核验身份，不再重写元数据／状态，
  模型和管理员策略原样保留。广域初始化仍会重置管理员密码，不作为日常案例维护入口。
- 现有案例模型 GET／PUT 继续校验 parser_llm、retrieval_llm、embedding；
  VISUAL 等策略仅报告 TEXT_POLICY_REQUIRED，管理员经现有 KC 管理合同显式调整。
  未新增配置菜单或表单；新增双仓一致的 aiops-case-resource-maintenance.md，记录具体命令、
  两平台 API 路径／请求体、AuthContext 边界、KC 并发／模型规则和部署审计责任。
- 部署工具事务由入口提交；失败回滚、关闭连接池；不打印原始驱动错误或连接详情。
  工具不做应用 Schema 检查，不进入 readiness、会话或 Worker 运行流程。

实际离线验证：Ammolite 相关资源／模板／空间合同回归 134 passed；
KBot 相关资源／模型和初始化合同 47 passed、5 skipped（仅 PostgreSQL 空间绑定专项）。
新共享资源专项分别为 28 项和 23 项通过，已包含在上述数字内。
两仓架构边界、git diff --check 通过，KBot 规范 Oracle 建库脚本 9 服务／40 脚本检查通过。
测试使用连接替身，不证明真实锁、Oracle JSON 驱动、模型调用、索引或通知消费成功。
本批未执行初始化命令、应用 DDL、部署、重启、提交或推送。

下一步继续第 6.7 项 4–6：通知精确深链与当前授权对账、报告分类分页和列表联动、
结构化适用性与检索质量；整页报告视觉差异及真实依赖闭环仍待验收。

### 6.13 通知结果定位、作用域守卫与 KBot 共享通知组件

本节推进第 6.7 项 4 的结果消费部分；尚未完成通知后台状态对账与真实投递全链验收，
不能把深链通过说成迟到／重试／乱序全部已经闭环。

- Portal notificationRoutes.ts 将输出任务和报告定位统一转到 SSO 工作台，保留确切 Domain、
  job_id、报告 ID／版本及案例 asset/version；缺失空间、非法 UUID、错误任务类型或不完整案例定位拒绝跳转。
  PlatformHomeView.vue 按通知指定空间换取会话；明确指定空间已不可访问时不回退其他空间。
  可信选中 domainId 不会被旧查询参数覆盖；手动进入其他 App 时不转发原通知的空间和结果参数。
- Ammolite 报告通知同时携带 Domain 和报告版本，自动／计划报告也不再产生缺少空间的报告链接。
  KBot 案例输出事件 safe_data 增加确切 job/type、来源版本和成功案例 asset/version。
- KBot 不在 Inbox 新增或复制 safe_data 列；Repository 按当前 Actor、Domain、页内 Inbox ID、
  过期时间及 aiops 生产者批量读取原 Outbox，应用层复核事件／资源身份，只序列化白名单 route_key/params。
  列表、流、单条和读状态响应都使用该受控定位；原信封或定位损坏时不产生链接。
- 新增 ui/shared/kbot-notifications.js/css，经真实 aiops-shell.js 动态挂到顶栏，不增菜单或改变配置表单。
  使用现有认证 request 读取已持久化通知，显示未读数和最近通知；可手动刷新、页面可见时定时更新，
  离页停止并在返回时重读。点击前重读通知、按确切 row_version 标记已读，再进入对应结果页面。
  文本采用 textContent，拒绝跨 Domain／不完整定位；失败任务链接返回对应页面的确切生成记录。
  通知或组件不可用有明确降级提示，不误报成功、不阻断 App；所有引用 Shell 的页面缓存版本同批刷新。
- 结果链接不充当授权证明，目标任务／案例／报告读取仍按当前身份与来源校验。
  本批没有通过伪造用户会话来补后台对账，也未将旧通知中的授权当成当前授权。

实际验证：

- 大范围替身回归 Ammolite 725 passed；KBot 752 passed、5 skipped、3 subtests passed。
  包含两仓资源专项、输出／报告／案例、通知适配器、Main API 与页面合同，不重复相加。
  KBot 新增通知路由与作用域序列化／Repository SQL 专项 15 项，已包含在总数内。
- 隔离浏览器 28 项通过，执行生产组件／处理器；其中新增通知 9 项，
  覆盖 Portal 路由及真实工作台跳转函数、指定空间撤权、KBot 案例／报告精确链接、恶意文本、
  点击前重读和断网降级，并验证真实 Shell 动态加载共享组件。HTTP 和身份为替身，不是真实部署验收。
- Portal 最后修改后的生产构建通过（保留已有大 Bundle 提示）；KBot Node 语法检查、
  导航专项 16 项、两仓架构边界及 OpenAPI 检查通过；API 路径数保持 971／827。
  本批未修改持久化模型／DDL，未执行资源初始化、部署、重启、提交或推送。

下一步仍需处理通知 operation 的权威状态查询及 source_version／重试／迟到／终态乱序对账，
随后完成报告服务端分类分页、列表联动、结构化适用性与检索质量；
第 6.11 节的整页报告视觉差异尚未修复，真实数据库／模型／KC／通知闭环尚未验收。

### 6.14 输出通知的版本顺序、终态刷新与当前授权对账

本节补齐第 6.7 项 4 的代码与离线协议门禁，不代表真实投递、数据库锁或恢复已验收。

- 两仓新增共享 output_projection.py 最小合同、source_order.py 顺序规则和
  AIOps OutputProjectionService。内部 POST /internal/v1/aiops/output-jobs:projection
  只允许 Main API 的真实 SERVICE 身份及 aiops.output.reconcile 窄 scope；
  普通用户上下文、其他服务及缺少 scope 的请求不能读取。公开 App API 不增加该接口。
  请求绑定 job/type、发起人、Domain 和 Ammolite Tenant；只返回精确状态／阶段、
  来源版本、时间、安全级与授权核验所需标识，不返回冻结正文、凭据、租约或产物内容。
- Main API 的 AIOpsOutputReconciliation 调用上述窄客户端，再复用真实平台账号、
  App／Domain、操作权限及 KBot 最高密级核验；校验请求回显、授权新鲜度和确切身份。
  不把浏览器令牌、旧 JWT 或硬编码权限保存成后台身份，也不复用既有通用对账中的合成 USER 上下文。
  撤权、账号停用或服务不可用只延期对账，不伪造任务失败或成功。
- 以任务已经落入事务的 row_version 作为事件 source_version。
  Worker 先完成业务终态并显式 flush，再调用独立 notify_succeeded；
  案例／报告组装阶段不再提前发成功事件。失败清理后也先刷新再写失败通知。
  产物、终态、尝试记录与通知仍共用原事务，未提前 commit；通知入队失败仍回滚并按原任务恢复。
  报告通知同时保留 job_type 和精确报告版本，PARTIAL 的通知仍明确提示缺口。
- 两仓输出任务投影按业务版本处理：严格较新才更新，低版本或同版本冲突不回写；
  更高版本可表达重试和后续终态，即使事件时间较早也不倒退。
  对账可能先于终态事件到达，因此同版本同终态仍允许创建尚未消费的通知；
  原 Outbox／Inbox 幂等机制继续防重复。过时尝试不新增误导性提醒。
  同任务的发起人、空间和任务类型必须一致，不能借新事件更换归属。
- 后台对账在候选读取事务结束后调用权威服务，再开启独立 UoW 写回；
  Repository 仅持久化，不持有 HTTP 回调。写回同时比较观察到的投影 row_version、
  来源版本和任务／空间身份；较慢旧响应或并发事件导致比较失败时不覆盖新状态。
  失败延期也有观察版本条件，避免撤权项长期占据批次首部。
  Ammolite 在现有通知对账循环接线；KBot 在现有 notification_worker 接线，不新增进程或侧栏。
- Ammolite 在规范 platform_background_operation 增加可空 source_version，
  现有 ORM 初始化器自动覆盖新库；未创建兼容表、迁移别名或历史回填。
  KBot 在 BackgroundOperationEntity／003_notifications.sql 增加 SOURCE_VERSION 和
  LAST_RECONCILED_AT，platform_core manifest 更新为 schema 5；
  定向升级工件 upgrade_notification_output_projection_5.sql 仅新增两列，不涉及账号或知识数据。
  本批未在任何真实环境执行该工件，不新增业务 CHECK／UNIQUE 或唯一索引。

实际离线验证：

- 大范围专项回归 Ammolite 823 passed；KBot 852 passed、5 skipped、3 subtests passed。
  包含新增共享通知专项各 46 项，不重复相加；覆盖顺序、同版本投递、跨作用域拒绝、
  撤权／停用、慢查询与新事件竞态、生产双 JWT 中间件和窄 scope。
  两项隔离 SQLite 专项执行真实 ORM 刷新，验证成功／失败通知使用最终落库版本；
  SQLite 仅替换 DDL 类型拼写，不证明生产数据库驱动或锁语义。
- Ammolite 平台通知合同及 Schema 验收工具本身的专项 15 项通过；
  两仓架构边界、跨项目一致性及 git diff --check 通过。
  PostgreSQL 规范 ORM 初始化基线检查通过，Oracle 规范 DDL 9 服务／40 脚本通过。
- OpenAPI 快照按当前代码更新并通过：Ammolite 18 快照／972 路径，
  KBot 16／828；各仅增加上述内部状态对账路径，公开路由不增加该能力。

真实 PostgreSQL／Oracle、
模型、KC、通知积压与浏览器完整闭环仍属 P7 验收，不能由 SQLite／HTTP 替身代替。
第 6.11 节整页报告的 440 处视觉差异尚未修复，本批未重跑浏览器或视觉对齐，
也未执行初始化、数据库变更、部署、重启、提交或推送。

剩余按第 6.7 项 5–6 继续：报告服务端分类分页与任务／成果列表联动，
结构化适用性与检索质量；另须修复整页视觉差异并完成另行授权的真实依赖验收。

### 6.15 报告中心服务端分类、筛选分页与完成刷新

本批在两仓同步接线，未变更配置区七项菜单、表单或 KC 资源。报告、任务和知识资产继续保持不同业务对象。

合同与查询：

- 新增共享 report_filters.py，显式定义 INSPECTION／DIAGNOSIS／SPECIAL_ANALYSIS，
  覆盖全部 20 个 ReportType，且穷尽、不重叠；“全部”不传 category，不通过标题或模板名推断。
  两种诊断类型进入诊断类，六种 INSPECTION_* 进入巡检，其余明确列入专项分析。
- Main API、AIOps Internal API 和平台客户端贯通 category／template_ref，report_type 改为枚举查询合同。
  应用层对直接调用同样验证；补齐 KBot 共享 ReportType 缺少的九种现有 MySQL／PostgreSQL 专项类型，
  移除两仓各自不完整的手工 allowed 集合，避免已有资产在列表响应或过滤时被错误拒绝。
- 仓储先施加 Domain、当前版本、Target、分类类型集合、精确 report_type 和 template_id 的交集，
  再按 created_at／report_id 做降序 keyset 和 limit+1；精确类型与分类冲突返回空，不扩大范围。
  分类和模板绑定进既有签名游标，修改任一条件、Domain 或调用身份均不能复用旧游标。
  不增加持久化分类列，不做运行时 Schema 发现，也不生成本批数据库迁移。

两端页面：

- AIOpsRecordsView.vue 与 aiops-records.js／reports.html 显示四种分类、模板和真实上一页／下一页。
  每页最多 50 份，明示“本页”数量；不再下载前 100 条后做浏览器模板过滤或冒充全部。
- 更换分类或模板、手工刷新、生成记录完成回调回到当前条件首页；
  已成功任务保留确切报告链接，即使报告不在当前分类／模板／页中也可打开，不把任务数量算作报告数量。
- 使用请求序号围栏；切换条件或 Portal 业务空间后，旧响应／旧错误不得写回。
  加载失败清空旧成果、显示可重试错误并关闭下一页，刷新恢复；组件卸载后禁止旧响应回写。
- 保留原数据库报告、pgBadger 导入与下载表单；本批没有将这些既有确定性生成入口改造成诊断会话按钮。

实际验证：

- 新增共享报告查询专项各 44 项；包含两层适配与客户端透传、所有枚举类型、分类穷尽、
  无效条件、签名游标的身份／过滤器隔离，以及执行真实仓储 SQL 的隔离 SQLite 用例。
  SQLite 用 130 份更晚但不匹配的报告验证先筛选后分页，覆盖同时间戳主键翻页、Domain、当前版本、
  模板与类型冲突；不代表生产 PostgreSQL／Oracle Schema、驱动或并发验收。
- 修正原静态 UI 合同对前 100 条、旧 v-else-if 和仅本地 computed 筛选的过时假设；
  保留生成时间、即时生效、PDF、表单和来源内容断言。
- 最终较大范围专项：Ammolite 876 passed；KBot 896 passed、5 skipped、3 subtests passed。
  上述 44 项已包含其中，不重复相加。
- 浏览器隔离执行两端生产脚本／Vue 组件：本次新增 7 项通过；与任务、通知深链和两类提交专项合跑
  31 项通过。它们替换 HTTP，不等于实库、真实 KC 或真实通知消费验收。
- Portal typecheck 和 production build 通过；仍有既有大 chunk 提示。
  报告列表整页对照 AIOPS_PAGES=reports 通过：空／有数据、至少三种宽度、PDF 下载及八种数据库报告表单；
  产物在 /tmp/aiops-page-alignment-A49XvK。本次为对照脚本补齐明确的输出任务空页和通知摘要替身，
  保持“未知数据请求即失败”的检查；这不覆盖报告详情的日／月报视觉问题。
  OpenAPI 按官方 writer 重新生成并通过（Ammolite 18 快照／972 路径，KBot 16／828），
  两仓架构边界、双向跨项目一致性、PostgreSQL canonical ORM 基线、Oracle 9 服务／40 脚本和
  git diff --check 通过。

剩余工作仍按第 6.7 项 6 推进：结构化适用性与检索质量集、报告整页视觉修正，以及另行授权后的
真实数据库／模型／KC／通知全链验收。第 6.11 节报告详情 440 处视觉差异在本批时尚未修复，后续结果见第 6.16 节。
本批没有执行数据库变更、案例资源初始化、部署、重启、提交或推送。

### 6.16 案例当前适用性、逐项措施门禁、质量评测工具与报告详情修正

本批继续同步两仓；没有改配置区七项菜单、表单或模型角色，也没有另建 KC／向量库。

案例复用：

- 新增共享 case_applicability.py 合同及应用模块，冻结当前 Agent 版本、显式 planner_llm、
  当前已授权 ENABLED Target 的安全技术事实与行版本。请求中的环境提示不是现场事实。
- 数据库类型、数值主版本及拓扑确定性比较；缺少值、字面 UNKNOWN 或无法规范解析的自由文本版本
  均为未知，不能因两段自由文本相同就认定主版本匹配。
  前提、禁忌、总体及逐项效果限制使用稳定 ID；由当前 Agent 规划模型单批判断自然语言条件。
  输出必须绑定精确上下文和案例集合、完整条件集合、原条件引文及真实当前事实 ID。
- 每次最多 20 份案例，单份 96 条条件，总量 512 条／100,000 字符，限时 30 秒；
  技术 LLM 输出上限仍由现有模型服务配置提供，不写死 token 上限。
  缺少 Target／模型／Prompt、依赖错误、非法输出或超预算时明确 UNKNOWN／范围不完整，不自动放行。
- OPS_CASE_REUSE_DECISION.v1 同时写入结果、reuse_context 和每份 Citation.locator.case_applicability。
  PLAN／CHANGE／VERIFY 不输出未满足推荐条件的正向案例；失败教训保留为 WARNING_ONLY。
  DIAGNOSE／EXPLAIN 可解释冲突和未知，但不能授予执行权限或冒充当前证据。
- 整体“已验证”不能复制给每个动作。只有对应原受控验证引用、有 RESOLVED／IMPROVED
  单项效果且无单项限制的动作，才可作为仍需当前验证的处置参考；
  无测量、引用不对应、效果不利的动作均不具备推荐资格。
- 模型和 KC 调用不占用数据库事务；全部外部调用结束后复核确切 Registry 版本及当前
  Agent／模型／Target 指纹，期间变化则丢弃旧输出。资料正文检索和 KC 相关性主排序保持原边界。

Prompt：

新增 aiops_agent.case_applicability，并更新知识问答／答案合成 Prompt，要求保留引用中的复用限制。
文件开发基线均保持 1.0.0；Ammolite 使用 model_name、KBot 使用 served_model_name 的目录适配。
未同步任何实际数据库 Prompt。部署前须按来源／Hash 定向同步基线、保留人工网页版本与历史运行；
真实环境缺少新 Prompt 时保守降为未知，不能称为语义链已经在该环境启用。

质量评测：

新增 tests/evaluation/aiops_case_retrieval_cases.json 与 evaluate_aiops_case_retrieval.py，
20 个场景、15 份固定资产定义，覆盖四类案例、DIAGNOSE／EXPLAIN／PLAN／CHANGE／VERIFY、
三种数据库、主版本／拓扑冲突、禁忌与未知、较旧相关、同源去重、精确修订和单项效果。
评分器检查 Recall／nDCG／倒数排名与独立安全门禁，未知资产仍占原排名，不删除后虚增质量分。
完整资产清单和响应缺一不可；合成响应必须标识 OFFLINE_CONTRACT。
导出文件中的 LIVE_DEPENDENCIES 只是来源声明；只有 --endpoint 真正完成请求才记录
live_requests_performed=true，且始终不宣称 production_acceptance_proven。
认证头不写文件、不打印，拒绝携带凭据的 URL 和重定向。具体准备与命令见
[检索评测说明](aiops-case-retrieval-evaluation.md)。

本批只运行数据集结构检查和评分器离线合同；未准备真实固定资产、未访问真实模型／KC 评测端点。
真实语义判断、召回和排序质量尚未验收；仍以 KC 分数为主，不能声称已验证综合排序目标。

报告详情：

- 修正 KBot 的静态样式生成器，保留 Vue 父页面 scoped 与日报子组件内部的边界；
  移除已被现行区域式日报替代的旧裸选择器样式，定向重新生成报告详情样式。
- 月报恢复同一四项元信息、结论标题、六列表格与行内趋势、无数据提示和页脚；
  不另画一个不同于 Ammolite 的趋势卡片区，不改变冻结报告的事实内容。
- 完善月报隔离数据的真实展示状态字段，新增标题／副标题／状态徽章对照。
  只将品牌文字作为明确的平台差异统一测量；布局、换行、样式和下载／提交断言继续执行。
- 共享样式定向同步本次报告分页所需样式。报告列表和详情、共享样式的基线检查通过。
  不带范围的全量样式生成检查仍在未修改的 reference-dashboard.css 报出既有工件漂移；
  本批未覆写所有配置页和其他页面工件，不能将报告专项通过写成全站样式基线已通过。

实际最终验证：

- 较大范围回归：Ammolite 974 passed；KBot 997 passed、5 skipped、3 subtests passed。
  两仓各含 45 项适用性与 45 项评分器离线合同；这些数量已包含在较大范围结果中。
- 数据集各 20 个场景结构有效，明确 retrieval_quality_evaluated=false。
  两端任务、通知深链、两类提交及分类分页隔离浏览器合跑 31 passed；KBot 导航 16 passed。
- 报告列表及详情整页对照通过：标准／日报／月报／对比、空／有数据、详情 1440／1024／900／420／320，
  并覆盖 PDF／原始附件下载、报告提炼受理和八类数据库报告表单。最终产物：
  /tmp/aiops-page-alignment-3vGfYt。此前第 6.11／6.15 节的报告详情 440 处差异已由本批修正，
  不是通过删除对照选择器或提高误差阈值消除。
- Portal typecheck、KBot 修改脚本语法、报告范围样式基线、双向跨仓合同与两仓架构通过；
  OpenAPI 通过（Ammolite 18 快照／972 路径；KBot 16／828），无新增 API 版本。
- PostgreSQL 规范 ORM 85 表及初始化基线、Oracle 9 服务／40 规范脚本检查和 git diff --check 通过。
  这些不是实库 Schema 校验或初始化执行。

后续仍须独立测试环境、固定资产和真实模型／KC 执行质量集，验证当前适用性、综合排序目标，
以及完整 P7“会话双按钮 → 后台任务 → 草稿／报告 → 通知 → 发布 → 新会话命中 → 引用预览 →
来源更正／撤回”的真实依赖闭环和数据库并发恢复。全量样式基线的其他工件漂移另行定向复核。
本批未进行数据库变更、资源初始化、Prompt 数据库同步、部署、重启、提交或推送；整体目标尚未完成。

### 6.17 有界联合重排与全量样式基线复核

继续复核追加的案例 KC 定制要求：资料与案例仍共用 KC，但使用不同 Profile／提取及复用规则。
本批不重做上传、解析器、配置表单、模型角色或知识库产品入口。

知识排序：

- 新增共享 application/knowledge_ranking.py 及 50 项专项合同测试，接入精确发布与当前权限、
  正文引用、当前适用性和处置用途门禁之后、同源去重和最终 Top-K 之前。
- 采用 OPS_KNOWLEDGE_RANKING.v1：KC 正文相关性乘以有界业务加分，最多 20%；
  当前适用性 6%、证据结构 4%、验证关联 4%、用途适配 4%、版本年龄 2%。
  权重是首版离线工程策略，未声称经过真实质量调优；完整信号及降级见专项设计第 7.3 节。
- 只认可真实当前案例判断，不将请求筛选命中当作现场相符；资料没有该判断时不加适用性分。
  案例逐项受控验证引用及同口径测量参与质量，失败经验不因结果不利受罚。
  资料的来源步骤及登记验证项只表达文档结构，不暗示已执行验证。
- 版本 created_at 在短 UoW 中冻结，排序在外部调用及最终重验后执行；
  不把发布时间当成事实新鲜度，不让缺失、无时区、无效或未来时间加分。
  输出 ranking 评分卡，coverage 保留策略版本；CATALOG 不冒充正文相关性搜索。
- 不按知识类型设置绝对优先，不以引用／步骤数量加分；零相关性不能制造命中，
  相差超过 20% 的正文相关性不被附加元数据反转。旧相关／新无关场景仍须真实 KC 质量评测。

样式基线：

- 第 6.16 节残留的 reference-dashboard.css 漂移已定向定位：仅两条独立媒体规则的生成顺序，
  没有选择器／声明变化。只重新生成 dashboard 工件，未覆写配置页或所有样式。
- 全量 sync_aiops_reference_styles.mjs 检查通过：26 份工件、27 个登记页面。
  dashboard 空／有数据与至少三种宽度的隔离浏览器整页对照通过，
  产物 /tmp/aiops-page-alignment-0QJIDa。使用替身 API，不是线上浏览器闭环验收。

本批最终验证及剩余验收见下；历史第 6.16 节的样式失败记录仍保留，不再视为当前未修复项。

实际验证：

- 新增重排专项每仓 50 项；与适用性、候选范围和评分器合跑各 176 passed。
  首次完整回归发现两仓“会话外不得读取 ORM”替身未提供本批新读取的 created_at，
  为测试补齐真实实体已有字段，并新增排序年龄断言；不在生产代码吞掉异常或绕过事务边界。
- 修正后完整重跑：Ammolite 1024 passed；KBot 1047 passed、5 skipped、3 subtests passed。
  五项跳过为仅适用于 PostgreSQL 空间资源绑定的测试，不作为 Oracle 实库证据。
  通过数已包含重排、适用性和评分器专项，不与历史批次累计。
- 两仓架构、双向跨项目一致性及 OpenAPI 通过：
  Ammolite 18 快照／972 路径，KBot 16／828。未增加持久化列或新 API 版本。
- KBot 导航 16 项、全量样式基线和上述 dashboard 整页对照通过；
  本批未重跑第 6.16 节的 31 项任务／通知隔离浏览器套件或 Portal 构建，不将历史结果写成本批新执行。
- 质量数据集各 20 个场景结构有效，retrieval_quality_evaluated=false。
  两仓 git diff --check 通过；无真实模型／KC 质量评测、实库并发或恢复验收。

当前未完成事项与下一步：

| 顺序 | 待办与所需输入 | 验收输出 |
| --- | --- | --- |
| 1 | 用户指定两仓独立测试环境、Tenant／Domain、身份和可用模型，并授权数据库工件应用、资源补缺／配置及必要部署；不得借此重置账号或全库初始化 | 精确环境与变更范围记录；沿第 6.12 节定向维护案例资源，显式配置模型并核对 Prompt 来源／Hash，保留网页人工版本 |
| 2 | 获授权后分别验证 PostgreSQL／Oracle 任务持久化、相同输入并发、API／Worker 中断与租约恢复、KC 慢／故障和通知停消费再恢复 | 精确任务、产物版本、终态与通知数及错误恢复证据，不能只用 202／readiness 证明完成 |
| 3 | 准备独立核验的 15 份质量资产及 20 场景金标，完成真实解析／索引、人工发布、新会话检索、精确引用和撤回阻断 | 实际模型／KC 召回、排序与安全评分；保留排名评分卡，验证较旧相关知识与失败警告，不反抄响应作为金标 |
| 4 | 在两端真实浏览器执行双按钮分别／同时提交、离页返回、通知精确深链和成果页显示，复核配置区未变 | P7 完整链路报告；明确通过、失败及未执行项，达标后才可标记整体完成 |

本批没有进行实际数据库变更、资源初始化、Prompt 数据库同步、部署、重启、提交或推送。
代码及离线门禁的进展不构成真实环境交付；整体目标继续未完成，以上外部状态变更须取得明确授权。

### 6.18 双按钮来源门禁、可用性解释与前端回归

本批对照已确认产品设计继续核验；保持既有配置七项、表单和业务实现不变。

发现并修正：

- 两端原先将 WAITING_USER 同时视为交互暂停和产出终态；Portal 还只检查当前轮次。
  这会在仍需补充信息或较早轮次仍运行时启用按钮，与服务端 report_preparation.py、
  case_closure.py 的来源门禁不符。现要求全部轮次属于 COMPLETED／PARTIAL／FAILED／CANCELLED，
  且至少一项 COMPLETED／PARTIAL；失败／取消不能独立作为报告或案例的有效来源。
- Ammolite 的 AIOpsDiagnosisView.vue 和 KBot 的 aiops-workspaces.js 在已选会话中保留两个按钮；
  空会话、活动轮次、等待补充、无有效结果或无 aiops:use 权限时，显示禁用原因。
  保留 title、可见状态说明和 aria-describedby，不以隐藏按钮代替解释。
- 两类繁忙态分别处理：Portal 案例来源核验／提交不禁用报告，报告提交不禁用案例；
  KBot 提交处理器继续只禁用各自触发按钮。服务端仍为来源资格、授权和异步受理的最终权威。
- KBot chat.html 增加说明区域；五个实际引用 aiops-workspaces.js 的页面统一更新缓存版本。
  未修改配置页面脚本、表单或处理逻辑。
- 新增 aiops-output-availability.spec.mjs：执行实际 Vue 可用性计算／按钮模板和
  KBot 生产可用性、渲染、绑定及清理片段，覆盖上述条件、原生禁用阻止点击、
  轮次结束后恢复和切换新会话清理。它不连接后台，不代替整页或真实依赖验收。

实际验证：

- 新增可用性浏览器专项 15 passed；与既有两类异步提交、报告提炼、轮次选择、任务恢复、
  通知深链、分类分页和案例撤回合跑，最终 56 passed。
  首次合跑为 53 passed／3 failed，定位为案例撤回测试未加载新任务面板脚本；
  补齐与正式页面相同的真实脚本和明确的空任务列表替身后完整重跑通过。
  未删除撤回断言、未将任务组件替换为默认成功空实现。
- Ammolite 会话／信息架构合同 58 passed；KBot 静态页面合同 60 passed、导航 16 passed。
  Portal typecheck 及 production build 通过，仍有既有大 chunk 提示；KBot 修改脚本语法通过。
- 整页隔离对照 AIOPS_PAGES=chat,cases,operations-knowledge,reports,report-detail 通过，
  产物 /tmp/aiops-page-alignment-4fkIpH：案例、资料、报告列表／详情空与有数据及多宽度，
  标准／日报／月报／对比、报告下载和八类数据库报告表单；
  Agent 工作台另含六宽度、流式、范围确认、人工补证、变更审批、效果验证和下载失败恢复。
  此前单独 chat 对照亦通过，产物 /tmp/aiops-page-alignment-o4sJdB。
- 全量样式基线通过：26 份工件、27 个登记页面；两仓架构边界通过。
  两仓文档同步后，双向跨项目合同／关键语义及 git diff --check 通过。
  本批未重跑第 6.17 节的 1024／1047 项后端大范围套件，不累加历史通过数。

未通过／未执行边界：

- 本批默认全站整页套件在 report-templates 的旧 .aiops-template-editor 选择器处中断；
  较大子集又在 inspection-plans 的“创建并启用”步骤中断。
  对照夹具缺少现行巡检模板策略／生命周期元数据，编辑器也已改为三步全页式，
  并观察到两端原有空名称保存按钮禁用表现不同；不能把两次中断写成全站通过。
  用于定位的临时配置夹具／辅助函数已移除，未修改配置业务或放宽断言来消除差异；
  后续须按现行配置合同独立维护该回归基线。
- 上述 56 项及整页对照均使用隔离请求，不证明真实双任务同时提交、数据库并发、
  通知消费、KC 索引或模型语义质量。第 6.17 节四步真实验收清单继续有效，P7 仍未完成。
- 本批未执行数据库变更、案例资源初始化、Prompt 数据库同步、部署、重启、提交或推送。
  下一步需要明确两仓独立测试环境、身份和模型，并取得第 6.17 节所列外部变更授权后，
  才能验证“案例草稿索引就绪 → 人工发布 → 新会话检索 → 精确引用 → 撤回阻断”的完整闭环。

### 6.19 巡检入口遗漏修正与当前交付审计

本批继续按目标设计审计，而非把既有通过数当成整体完成。配置区七项、现有表单和处理器保持不变。

本批修正：

- 第 6.18 节 inspection-plans 中断的初步归因进一步收敛：测试没有显式选择必填的巡检模板。
  该夹具不是默认模板，当前表单不会自动选首项，因此禁用提交是正确行为。
  对照脚本现先证明未选模板不能提交，再选择确切模板；无效时间检查在其余条件有效时执行，
  有效输入再证明按钮启用。未给夹具伪造默认身份、未放宽提交或关闭断言。
- 随后真正执行 inspection-plan-detail 时发现 KBot 深链入口把含 import 的脚本当 classic 加载，
  浏览器抛 Cannot use import statement outside a module，尚未进入重定向。
  修正该 HTML 为 type=module，并与计划列表使用同一脚本版本；不改计划配置逻辑或表单。
- KBot 计划页返回页签仍为“巡检报告”，违反本次“巡检记录／巡检计划”命名要求。
  现与记录页及 Portal 统一；报告中心的“巡检报告”分类保持原义，不做全局文字替换。
- 增加两个静态合同：两个入口均以 ES module 加载同版本脚本，两个页面页签文字完全一致。
  整页对照同步校验两端的真实页签文字，而非只比较等宽区域，避免版式通过掩盖命名遗漏。

实际验证：

- Ammolite 较大范围完整重跑：1024 passed；均为本次选择的离线测试，不代表整个仓库全部测试。
- KBot 较大范围在修正前重跑 1047 passed、5 skipped、3 subtests passed；
  两项新增静态合同后完整重跑 1049 passed、5 skipped、3 subtests passed。
  单独静态页面合同 62 passed、导航 16 passed。通过数不与历史或重叠专项相加。
- 原 10 页子集（监控、工作项、告警、巡检、批次、演练、运行、运行详情、监控源及详情）
  整页对照通过，产物 /tmp/aiops-page-alignment-gIjsds。
- 修正后 inspections／inspection-plans／inspection-plan-detail 三页整页对照通过，
  产物 /tmp/aiops-page-alignment-I1wQuc，覆盖空／有数据、多宽度、计划创建／编辑、
  缺失模板和无效时间禁用、确切请求、失败后恢复及深链抽屉。
  此前修正模块加载后的两页对照亦通过，产物 /tmp/aiops-page-alignment-eRAmqc。
- business-systems／business-system-workspace／knowledge-bases／responsibility-groups／knowledge-tags／agents
  六页隔离对照通过，产物 /tmp/aiops-page-alignment-vfhZ6t；保留现有新增、编辑、权限和失败断言。
  Git 差异复核未发现两仓配置表单或 KBot 对应配置处理器改动，不为修复旧模板测试而改变配置功能。
- 全量样式基线通过：26 份工件／27 登记页面；Node 对照脚本语法通过。
  两仓架构边界、文档同步后的双向跨项目一致性及 git diff --check 通过。
- PostgreSQL AIOps canonical metadata 85 表和初始化基线检查通过；Oracle 9 服务／40 脚本离线检查通过。
  OpenAPI 通过：Ammolite 18 快照／972 路径，KBot 16／828；本批不写回快照、不增加 API 版本。
  PostgreSQL／Oracle 的实库 Entity acceptance 未执行，不能以这些离线工件结果代替。

按计划的完成审计：

| 工作包／不变量 | 当前依据与结论 | 尚缺的权威证据 |
| --- | --- | --- |
| P1 受理、冻结、分页、幂等及事务 Outbox | 共享合同、output_jobs 应用／仓储／实体、规范工件和本批离线回归通过 | PostgreSQL／Oracle 当前表结构及真实并发、提交／回滚 |
| P2 后台装配、租约恢复、当前授权、独立结果和终态 | Bootstrap 注入实际 OutputJobWorker／Finalizer，冻结来源、迟到 Worker、重放与故障注入通过 | 真实进程中断／恢复、外部已受理写回故障、独立双任务产物计数 |
| P2-K 案例提炼、脱敏、KC 精确版本、草稿／发布隔离 | 共用案例投影与入库恢复、来源核验、精确 Revision／Hash、预览和发布门禁有离线证据 | 指定环境资源／模型和 Prompt、真实 KC 解析／索引、人工发布和新会话命中 |
| P2-K 检索候选、适用性、效果限制、引用与撤回 | 分页候选、有界重排、当前 Target 与逐项措施资格、20 场景评分器的离线合同通过 | 独立金标及真实模型／KC 召回、排序、自然语言判断和撤回质量 |
| P3 成功／失败通知、确切深链、来源版本排序和对账 | 两仓通知合同／消费接线、乱序／重试／当前权限及隔离浏览器证据见第 6.14／6.18 节 | 真实 Outbox 积压恢复、数据库 CAS 并发、跨空间与撤权后的实际浏览器点击 |
| P4 导航、标题、页面归属、配置保持 | 当前导航 16 项及两仓信息架构；本批补齐计划页遗漏文字与深链入口 | 真实身份／部署后的菜单权限、窄屏及配置区保留复核 |
| P5 双按钮、任务与产物分区、分类分页和成果刷新 | 第 6.18 节 56 项隔离浏览器及核心五页对照；本批巡检／计划对照通过 | 真实双任务分别／同时受理、离页返回、通知及对应页面的同一产物闭环 |
| P6 全部开发门禁与文档同步 | 本批回归、规范工件、OpenAPI、页面子集和样式检查留证；不累计历史通过数 | 默认全站套件的模板中心旧编辑器／夹具基线仍不适配，不能写成全站通过 |
| P7 实库、真实依赖、恢复演练与发布 | 未执行，不满足整体完成条件 | 用户指定两仓验收环境、身份、模型，并明确数据库工件应用、定向资源／模型／Prompt 维护与必要部署／重启授权 |

第 6.18 节所列模板中心旧选择器及原有保存按钮表现差异仍保留为未解决的回归基线问题；
本批未删改默认套件的页面清单、提高视觉阈值或修改配置表单以使其通过，也不把独立子集拼成默认全站通过。

当前审计结论：代码与离线证据继续前进，整体交付仍未证明完成。
在缺少明确环境和外部变更授权时，不运行广域初始化，不自动创建测试资产、选模型或发布案例，
不代用户执行数据库变更、部署、服务重启、提交或推送。

### 6.20 kbotdev 实库无损结构升级与开发 Prompt 同步（2026-11-03）

本批外部授权仅为：连接用户指定的测试服务器，升级数据库表结构、保留数据并同步 Prompt。
不包含应用代码部署、服务停止／重启、模型或 Collection 配置、案例发布、测试资产创建及提交／推送。
服务器 ubuntu@140.238.44.208（私网 10.0.0.190），使用指定本地密钥；SSH 显式设置
KexAlgorithms=curve25519-sha256。较大文件返回曾断连，使用小分块传输完成，未修改服务器 SSH 或网络配置。

实际目标与留证：

- Ammolite 部署根 /home/ubuntu/Ammolite，使用其 configuration/ammolite.toml、
  .env 和现有连接解析器；目标 PostgreSQL database/schema 均为 ammolite，development。
- KBot 部署根 /home/ubuntu/kbot4.0，使用其 typed settings、configuration/kbot.toml 与 .env，
  目标 Oracle PDB DB2026_PDB1，development；没有在输出或文档中记录密码、完整 DSN 或账号。
- 远端私有目录 /home/ubuntu/db-schema-upgrades/20261103-aiops-XFrgH8 保存本批审计工件，
  目录权限 0700、备份与报告文件 0600。本批采用本地当前 canonical 导出物，
  三份结构／契约 JSON 传输前后 SHA256 相同，不覆盖部署代码和配置。
- pg-before.json、oracle-before.json 含只读结构清单和完整 Prompt 定义／版本备份；
  两份 structural-backup.json 含受影响表的结构证据、后台任务行及 16 张关键业务表的主键集合。
  这些是定向结构／Prompt／业务身份快照，不等同于 PostgreSQL 全库 dump 或 Oracle Data Pump 全量数据备份。

已实际提交的结构变化：

- PostgreSQL 在单事务、5 秒锁超时及有限语句超时下创建 aiops_output_job、
  aiops_output_job_attempt，增加 platform_background_operation.source_version BIGINT（可空）；
  last_reconciled_at 已存在，未重复添加。两张新表主键、尝试到任务外键和 6 个普通索引齐备。
- Oracle 创建 KBOT_OPS_OUTPUT_JOB、KBOT_OPS_OUTPUT_JOB_ATTEMPT，
  增加 KBOT_BACKGROUND_OPERATION.SOURCE_VERSION、LAST_RECONCILED_AT，
  创建任务表的 6 个普通索引，并补齐 6 个原有普通索引；未新增业务 CHECK／UNIQUE 或业务唯一索引。
- Oracle 原版本视图为 54，但只读物理契约检查证明巡检模板 CURRENT_VERSION_ID 已可空且
  MANAGEMENT_JSON 已就绪，现有 AIOps 列与 schema 55 等价。本批按审核后的 canonical 差异执行，
  不是绕过前置条件运行仅支持 55 的升级脚本，也未通过重建表追版本。
  所有新增 AIOps 结构验证后才更新视图为 56 / aiops-oracle-v56-output-jobs。
- Oracle 另外 4 个 canonical 非空差异经查询确认 NULL 行数为 0 后，仅用结构 DDL 收紧：
  KBOT_KC_COLLECTION.PARSE_POLICY_JSON、KBOT_DQ_POLICY.SUBJECT_SELECTOR_JSON、
  KBOT_DQ_AGENT_BINDING.CONSUMER_APP_ID、AGENT_VERSION_ID。
  所有 Oracle DDL 自动提交，使用 5 秒 DDL 锁超时，执行顺序／语句 Hash 留在 oracle-execution.jsonl。

已实际提交的 Prompt 同步：

- 两端各新增 aiops_agent.case_applicability，原地更新 3 个 AIOps 基线：
  answer_compose、diagnosis_answer_compose，以及 Ammolite 的 knowledge_answer／KBot 的 knowledge_answer_compose。
  全部维持 1.0.0；Ammolite 来源 BUILTIN，KBot 来源 FILE_SEED，激活确切基线，不增加管理员版本。
- 写前按 Key／Source／Version／Hash／激活指针核对并加锁；没有运行初始化、force-reset，
  没有删除版本或改写冻结运行。Ammolite 原 41 个 Version ID、KBot 原 71 个 Version ID 全部保留。
- KBot 下列 7 个 Key 有人工版本或管理历史，定义、版本与激活指针完整保留，不覆盖：
  agent_runtime.context_rewrite、conversation_snapshot、data_query_plan、intent_route、
  memory_conflict_assess、memory_extract、response_compose。
  当前 Catalog 外的 Prompt 亦未删除或退役。
- 数据库核验 Ammolite 41 个当前激活基线、KBot 29 个非人工管理基线的 Hash、来源、版本与状态；
  两端还通过已部署 PromptResolver 新建实例及 StrictPromptRenderer 从实库读取／渲染验证。
  这不证明既有服务进程缓存已切换，也未调用真实模型或执行会话／案例业务。

实库验收与明确留项：

- PostgreSQL 当前 canonical 219 表／3298 列，精确比对表列集合、类型、可空性和主键通过；
  新任务表索引／外键通过。16 张关键业务表的原 2120 个主键全部仍存在。
- Oracle AIOps 当前列契约、新表 6 索引及外键通过，版本视图为 56，当前用户 INVALID 对象为 0；
  16 张关键业务表的原 11656 个主键全部仍存在。此项不是对全库每行内容未变的证明。
- Oracle 全库 canonical 仍有 3 个非空差异，原因是已有真实 NULL，按“仅结构、不删数据”保留：
  KBOT_AGENT_CONVERSATION.EXECUTION_SPEC_JSON 有 35 行，
  KBOT_DQ_RUN.AGENT_ID 和 AGENT_VERSION_ID 各有 5 行。
  未补写缺失 Execution Spec／Agent 身份，未删除记录，未宣称全库零漂移。
- 全服务 Oracle ORM 比对还发现既有 6 个 ORM／DDL 类型定义差异：
  KBOT_MANAGED_CREDENTIAL.NONCE 的 ORM BLOB 对比 canonical／实库 RAW(12)，以及
  IS_INITIAL_ADMIN、IS_SYSTEM、MEMBER_ASSIGNABLE、MUST_CHANGE_PASSWORD、IS_PROTECTED
  五个标志的 ORM VARCHAR2(1) 对比 canonical／实库 CHAR(1)。未转换凭据或改动这些数据。
  canonical 中缺失的 UX_AGENT_MEMORY_ACTIVE_KEY 为业务唯一索引，按现行仓库约束政策未创建。
  5 张既有非 canonical 表保留，不做清理。因此不能把 Oracle 全库 Entity／索引一致性写成通过。
- 本批重新运行的离线检查：两仓架构边界通过；Ammolite PostgreSQL 基线、85 表 AIOps metadata 通过；
  KBot Oracle canonical 9 服务／40 脚本检查通过。离线检查不替代上列实库报告。
- 执行日志与实库／解析器报告为 pg-execution.jsonl、oracle-execution.jsonl、
  pg-verification.json、pg-exact-contract-verification.json、oracle-verification.json、
  pg-runtime-verification.json、oracle-runtime-verification.json。

本批没有执行 DROP／TRUNCATE／业务数据 DELETE／历史回填、全量初始化、服务停止或重启，
没有部署应用或修改模型／Collection，也没有提交或推送。
P7 仅推进“指定环境的结构工件与开发 Prompt”部分；真实双任务、通知／KC 索引闭环、
模型质量、恢复演练、模板中心旧整页回归基线及部署后权限验收仍未完成，整体目标不标记完成。

### 6.21 模板中心回归基线更新与 kbotdev 案例资源只读核验（2026-11-03）

本批延续已授权实施范围，只补隔离测试、文档与远端只读核验；
未部署应用、重启服务、创建 Collection／案例或改模型配置。

默认全站浏览器基线更新：

- KBot 的 tests/contract/aiops_page_alignment.mjs 保留全部 25 页及既有严格几何／样式断言。
  新增 aiops_template_alignment.mjs，按当前三步整页巡检模板编辑器替换失效的旧抽屉选择器；
  会话模板原抽屉测试仍保留，未删页、跳过配置页或放宽阈值。
- 隔离 Catalog 使用真实十二章节定义和 SHA256，明确三种数据库的基线与补充检查策略；
  其 Catalog／DAILY_REPORT_LAYOUT 通过当前 Pydantic 合同，不连接真实数据库或模型。
- 桌面／窄屏分别验证创建和编辑、规划检查不可选、空名称不写入、区域顺序与内容、
  预览失败恢复及人工确认、保存中禁用和重复提交阻止、失败保留输入、
  草稿持久化后发布失败重试、相同请求正文与行版本锁。
- 保留两端既有空名称按钮表现：Portal 禁用保存草稿，KBot 在处理器阻止写入；
  测试验证真实边界，不修改配置表单以统一外观。关键区域对照前统一窗口滚动位置，不改 CSS。
- 实际运行默认 node tests/contract/aiops_page_alignment.mjs：25 页及各状态全部执行到结束，
  交互／请求断言未中断，最终严格视觉断言失败，共 302 项，全部属于 report-templates。
  产物为 /tmp/aiops-page-alignment-TJTcsn/differences.json 及同目录截图。
  其中 300 项集中在巡检模板目录与新整页编辑器，2 项是会话模板菜单的纵向位置；
  其他页面未记录几何／样式差异。这是隔离请求测试，不是线上业务验收。
- 现有差异包括巡检策略提示在列表容器内外的不同结构，以及 KBot 编辑器继承页壳 padding
  与 Portal 编辑器的尺寸／位置不同。配置部分按已确认范围保持，未为消除差异修改生产表单或样式。
  旧选择器和无效夹具问题已修正，但全站视觉一致性仍未通过，不能改写为全部通过。
- 本批复核：两份 mjs 的 node --check 通过；
  test_inspection_template_presentation.py 与 test_aiops_ui_static_pages.py 合计 66 passed；
  aiops_shell_navigation.mjs 为 16 passed。只记录本批执行结果，不累计历史后端通过数。

指定测试环境的案例资源核验：

- 使用第 6.20 节同一服务器、部署目录和连接解析器，以显式 READ ONLY 事务检查现有资源与绑定；
  不输出模型服务地址、密钥或连接凭据，不选择默认模型或调用模型。
- Ammolite 已启用 AIOps 自有域 aiops-default：符合固定案例身份与管理绑定的 Collection 数量为 0。
- KBot 已启用 AIOps 域 aiops_portal：固定 diagnosis-cases Collection 数量为 0。
- 因资源不存在，本批不能验证 parser_llm／retrieval_llm／embedding 的实际绑定与可用性，
  也不能证明解析、索引、人工发布及新会话检索闭环。没有自动初始化或创建占位资源。
- 远端证据位于 /home/ubuntu/db-schema-upgrades/20261103-aiops-XFrgH8/
  的 pg-case-resource-audit.json 和 oracle-case-resource-audit.json，记录 mutations=false、
  runtime_models_validated=false；私有目录与文件权限沿用第 6.20 节。
- 另读取复核第 6.20 节保存的结构、Prompt 及新解析器报告，结果与该节记录一致；
  该复核不是重跑升级或证明运行中进程缓存已刷新。

P7 下一步须先确认验收域／身份、三个显式模型及其使用许可，
并单独获得定向创建案例资源、真实测试资产写入和必要应用部署／重启的授权。
仅有表结构／Prompt 同步授权不包含这些变更；不运行会重置管理员的全量初始化。
Oracle 历史 NULL、ORM／DDL 类型差异和全站模板视觉差异仍按上述证据保留，整体目标不标记完成。

### 6.22 真实配置装配漏项与扩大后端回归修正（2026-11-03）

本批继续本地实施／验收，不把持续目标视为新增外部变更授权。
服务器结构与 Prompt 的执行范围仍以第 6.20 节为准；没有部署、重启、创建资源或调用真实模型。

本批实际修正：

- 扩大到整个 tests/unit/aiops_agent 后，KBot 首次收集在 test_aiops_step0.py 失败：
  新的 clients.main_api 依赖已声明，但 resources/topology.toml 的 endpoints 漏了 main_api，
  真实 load_settings 报“依赖绑定 main_api 引用了未知端点：main_api”。
  这会阻止 AIOps 各进程配置装配，不只是测试夹具错误。
  现补 main_api 到既有 Main API 进程的映射；未硬编码客户端 URL、绕过依赖解析或新增配置表单。
- Ammolite 已有正确映射，生产配置无需改动。两端 test_service_config_loading.py
  均增加默认端点和部署覆盖两种真实加载测试，验证 URL、服务 audience 和 10 秒超时，
  不再只用手工构造的 SimpleNamespace 证明输出 Worker 可装配。
- 扩大回归的首轮 Ammolite 为 42 failed／5060 passed，KBot 在修复收集问题后为
  45 failed／5018 passed／5 skipped；这些是本批发现的问题，不沿用此前较小范围的通过数。
- 两端各 42 项失败来自六组来源重放测试的旧 Agent 夹具缺少 status，
  与本次案例发布／检索必须核验活动 Agent 的合同不符。
  已在 future_observation_evidence、oem_events、oem_native_metrics、
  physical_identity_graph、prior_sql_source_replay、sql_capture_evidence 中显式声明 ACTIVE。
  仍运行真实 validate_case_sources；不 mock 掉生产校验，不删除原件篡改、撤权、缺失和反证断言。
- test_aiops_case_governance.py 两端各新增 Agent 停用／无当前版本必须拒绝的负向场景。
  生产状态门禁未放宽，测试不以缺字段的替身代表合法 Agent。
- KBot 另外 3 项来自 recovery_assurance、resource_governance、target_facts 的旧 Schema 版本断言。
  精确更新为当前 56／aiops-oracle-v56-output-jobs，保留各业务表／工件 Hash 和运行时不建表断言；
  Ammolite 共享资源治理测试的 Oracle 分支同样对齐，不接受多版本范围或读取期望值自证。

本批完成的专项验证：

- 六组来源重放及案例治理分两组执行：两端分别为 204 passed、244 passed。
  两组互不相同，但仍不把它们与下列完整重跑或其他历史结果累加。
- KBot 三份旧 Schema 断言修正后的专项：76 passed、3 subtests passed。
  Ammolite 共享资源治理单独复核：59 passed。
- 配置／拓扑／输出装配联合专项：Ammolite 19 passed、2 subtests passed；
  KBot 加上四进程入口边界后 32 passed、2 subtests passed。
- 两端架构检查通过；OpenAPI 当前快照 Ammolite 18 份／972 路径、
  KBot 16 份／828 路径通过；双向跨项目合同检查通过。
- PostgreSQL canonical 基线与 AIOps 85 表 metadata 通过；
  Oracle canonical 9 服务／40 脚本通过。这些离线工件检查不代表新增实库验收。
- Portal npm run build 通过（保留 chunk 体积提示）；两端真实组件的八份隔离浏览器专项合计
  56 passed，覆盖独立双按钮、异步回执／恢复、生成记录、分类分页、通知深链和撤回。
  不将隔离请求当作真实模型、KC 或线上通知消费。
- 两端修正后的扩大范围完整重跑均以退出码 0 结束：
  Ammolite 5104 passed、332 subtests passed（234.86 秒）；
  KBot 5065 passed、5 skipped、268 subtests passed（227.75 秒）。
  范围为全部 AIOps 单测、KC parse_settings／revision_status、Main API output_authority／
  output_reconciliation；KBot 另含 Main API operations_knowledge 适配，不代表整个仓库所有测试。
  KBot 的 5 个跳过项只针对 PostgreSQL 的独立空间资源绑定，不是忽略失败或缺失依赖。
  保留第三方依赖的弃用警告，不把首轮失败、重叠专项和最终通过数相加。
- 本批修改路径的 git diff --check 通过；两仓实施文档逐字一致，
  新增文档按完整文件另检无空白错误。未提交、推送或执行外部业务变更。

第 6.21 节的模板配置页视觉差异未修饰或豁免。本批不改变配置菜单和表单，
也不修复第 6.20 节需要业务数据决策的 Oracle 历史 NULL／类型差异。
真实双任务、通知恢复、案例 Collection／三个模型绑定、KC 精确索引与人工发布后的检索闭环仍待授权验收；
整体目标不标记完成。

### 6.23 代码交付收尾、前端基线及手动测试分工（2026-11-03）

用户最新明确：“真实测试由我手动做，你只负责改代码”。
代理交付范围确定为两仓代码、规范工件、文档及本地自动化回归；真实准备、部署、
恢复演练及业务验收交给用户，不再作为代理继续操作服务器或申请外部写入的待办。
历史章节中“整体未完成／待授权验收”保留当时语境，以本节最新分工为准。
这不将未执行的真实测试改写为通过，也不表示整个 AIOps 产品的所有历史规划均已完成。

代码交付范围：

- P1／P2：持久化任务、冻结输入、幂等、当前授权、租约恢复、重试、实际处理器和原子终态已接线。
  会话报告、会话案例及报告详情案例入口共用后台机制，返回独立持久化回执。
- P2-K：案例专属 Profile／正文与关联投影、KC 精确版本入库恢复、草稿／发布隔离、
  资格复核、适用性及逐项效果限制、候选分页与可解释有界重排已实现。
  固定资源定向维护、三角色模型管理 API 及离线质量评测工具随代码交付。
- P3：事务通知、结果深链、两前端消费、来源版本排序及当前授权对账已实现。
- P4／P5：Agent 主入口、运维工作、报告中心直达、知识资产菜单及页面职责已调整；
  独立双按钮、生成记录、失败重试、确切结果、报告分类分页和完成刷新已实现。
  配置区七项及原有表单保持，不为视觉回归改变它们。
- P6：Schema 工件、客户端、OpenAPI、跨仓检查、专项测试及文档已同步；
  已知配置页视觉失败单列，不声称全部测试全绿。后端完整专项结果见第 6.22 节；
  本批没有再改后端，不重复执行或累加该批通过数。

本批修正仅涉及测试／文档：

- Portal 旧空能力断言改为必选 MONITORING_QUERY 的默认、编辑补齐及禁止移除不变量。
- KBot Agent 初始化 VM 补齐 document.head 和通知动态加载失败夹具，
  仍执行真实 Shell／Agent 脚本，并验证样式加载及降级提示不阻塞页面。
- 业务系统缓存键及上传资料文案断言对齐当前实际版本，未回退生产脚本版本；
  刷新恢复、摘要、人工审批等原断言保留。
- 侧栏对照从真实 Shell 提取导航 computed，覆盖 sections、独立直达项和路由归属；
  保留全部入口、1px 阈值、样式与图标精确比较，并增加直达项测量。
- Agent 浏览器夹具按实际 HTML 顺序加载完整 CSS，不再遗漏共享、参考及适配层；
  未修改生产 CSS 或放宽视觉断言。相关文档清除“仅文档／入口装配尚未实现”的过期当前状态。

本批实际执行结果（不连接真实模型、KC 或业务数据库）：

| 检查 | 结果 |
| --- | --- |
| Ammolite：全部 AIOps Python 合同＋OpenAPI＋process_topology＋service_config_loading | 178 passed、2 subtests passed，退出码 0 |
| KBot：同一范围合同 | 176 passed、2 subtests passed，退出码 0 |
| KBot Agent 实际脚本初始化 | 12 passed，含认证、等待、五类依赖失败、恢复与通知降级 |
| 真实侧栏模板对照 | 24 个入口，1440／900／420 宽度及当前组交互通过；产物 /tmp/aiops-shell-alignment-XVXniM |
| Agent 配置保护浏览器专项 | 26 passed、1 failed；功能通过，严格外观测试失败 |
| 两仓架构边界、双向跨项目对齐 | 通过 |

已知本地失败：

1. 第 6.21 节默认全站视觉对照的模板中心 302 项差异仍保留；本批未重跑该套件。
2. Agent 配置专项完整重跑为 26 passed／1 failed。必选监控、边界、迟到响应、
   隐藏动作授权保护、自动触发、知识关闭、字段归属、跳步校验和失败恢复等功能通过。
   完整 CSS 加载后，严格视觉比较先通过卡片区域，在抽屉宽度处失败：
   1440 视口下 KBot 1440px、Portal 1120px。其后外观比较未执行，不声称只差一个属性。
   错误上下文位于 Portal test-results 下该专项的 error-context.md。
   未为消除差异修改配置页、抽屉或生产样式，也未跳过或放宽该测试。

交给用户手动执行：

1. 准备固定案例资源及显式 parser_llm／retrieval_llm／embedding。
   第 6.21 节历史只读核验中两端目标域均无固定资源，本批未重新核验或创建。
   按[资源维护说明](aiops-case-resource-maintenance.md)定向补缺，不运行全量初始化。
2. 分别／同时提交两类输出，离页返回，核对独立任务、产物与通知；
   READY／PARTIAL、案例草稿、发布状态及处置效果分别判断。
3. 核对确切 KC Revision 索引及草稿就绪；人工发布前不可检索，发布后新会话命中确切版本，
   带出前提、效果与禁忌，撤回／撤权后阻断。
4. 演练 Worker／通知消费中断、KC 慢或失败、重试、重复请求和来源更正，
   核对不重复报告、案例版本或终态通知。
5. 按[质量说明](aiops-case-retrieval-evaluation.md)准备独立金标及真实响应，
   复核相关性、异因、未知／冲突、手册混合检索和精确引用；评分器通过不等于真实质量已验收。

本批没有 SSH、部署、重启、数据库写入、模型调用或测试资源创建；未提交或推送。
按最新分工，本次代码实施交付完成；真实验收仍为“由用户执行，尚无本批结果”。

### 6.24 案例 KC 自动资源生命周期（2026-11-04）

本节替代 6.12 中依赖定向部署脚本的资源补缺流程，历史记录不代表当前操作要求。

- 平台业务事务登记一 Domain 一条持久化资源事件，KC 内部用例负责实际建库与所属绑定。
- Domain/App 生命周期入口与规范初始化统一登记；后台键集分页对账自动纳管已有空间。
- 资源记录持有已知 Collection ID、租约、重试时间、失败码和独立模型配置状态。
- KC 父空间行锁串行初始化，复用有效资源、不覆盖配置、不重建已知丢失资源；
  通用物理删除入口保护系统案例资源。
- 案例页面显示资源/模型配置状态；配置七项不变，模型必须显式配置。
- 删除定向 provision_aiops_case_collection.py；运行时无部署脚本依赖，无 Schema 探测。
- 本地离线验证结果与实际环境验收分开记录；本次不执行服务器部署、DDL 或真实检索。

离线验证记录（不代表真实环境验收）：

- 最新资源生命周期、Domain 事务事件与案例模型专项：Ammolite 71 passed；
  KBot4 67 passed / 4 个仅 PostgreSQL 绑定场景 skipped。
- Main API / KC / 资源联合回归：Ammolite 550 passed / 1 deselected，
  KBot4 324 passed / 4 skipped。Ammolite 排除项是既有会话 starter 夹具缺失
  business_systems；相关实现函数 AST 与测试文件均核对为本次未修改。
- 完整 AIOps 单元目录批次：KBot4 4985 passed / 4 skipped；
  Ammolite 5029 passed / 1 failed，失败为动态采样 POSTGRESQL 场景返回 PARTIAL。
  该采样文件独立复跑 16 passed；未修改采样代码，不将原完整批次标记为全绿。
- 两仓架构边界、规范 Schema、OpenAPI 快照和跨项目对齐门禁通过；
  Ammolite Entity 所有权检查通过。
- Portal typecheck 与 build 通过（保留既有大包体提示），KBot 页面脚本语法检查通过。
- 案例页隔离对照通过空/有数据及至少三种屏宽；补测资源初始化中、状态接口失败时列表独立可用。
- 无 SSH、服务器部署/重启、数据库写入或真实模型/检索调用；本批未提交、未推送。
