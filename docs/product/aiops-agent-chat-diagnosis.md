# AIOps Agent 专业 DBA 对话诊断设计

## 1. 定位与原则

智能诊断不是把所有问题都送入“根因分析”流水线的聊天机器人，而是一名在明确
Target、权限和证据边界内工作的专业 DBA。它先理解用户正在完成什么工作，再选择
经过登记的 DBA Skill，从监控、日志、数据库直连和知识库取得当前问题所需的最小
证据，最后用适合该问题的自然语言、表格或图形回答。

本文约束人工发起或从告警、巡检继续的 Conversation。告警自动诊断和计划巡检可以
使用预定义 Blueprint，但用户进入续聊后，每条消息都必须进入本文定义的 Turn 编排。

核心原则：

1. **先理解工作，再调用工具**：问题决定工作方式，不能默认启动完整根因诊断。
2. **Skill 是能力边界**：LLM 只能选择已登记 Skill 和参数，不能生成任意 Tool ID、
   SQL、URL 或执行计划。
3. **按本轮问题取证**：历史证据是可选上下文，不自动成为本轮事实或展示内容。
4. **事实、推断和建议分离**：直接观测、用户补证、知识解释和模型推断保持来源。
5. **只取最小必要证据**：简单查询不运行全套诊断；复杂故障才进入假设与反证。
6. **回答服从问题**：普通聊天不套报告模板；表格和图形由 Skill 输出语义决定。
7. **变更必须显式进入审批链**：聊天中的“同意”不能替代 Proposal 审批。
8. **保留数据库特性**：公共语义之外，Evidence 保留各数据库专有字段和许可约束。

## 2. 专业 DBA 的工作模型

每条用户消息形成独立 `ConversationTurn`：

```text
用户消息
  → Turn Scope：冻结 Agent、Target、来源、时间和历史上下文
  → Intent Plan：识别工作意图、专业领域、对象和参数
  → Skill Plan：从当前可用能力中选择最小 Skill 集合
  → Evidence Collection：并行或按依赖采集证据
  → Sufficiency Gate：判断是否足以回答当前问题
  → Answer / Clarification / Evidence Request / Proposal
  → 可选 Verify：用相同口径验证处理结果
```

Conversation 保存连续交流和用户目标；Turn 保存本轮问题与权威结果；Run 保存本轮
执行；Skill Invocation 保存具体能力调用。不能把整个 Conversation 的事实集合直接
复制给每个 Turn，也不能依据浏览器状态判断本轮使用了哪些证据。

## 3. 意图模型

### 3.1 七个一级意图

| 一级意图 | 业务含义 | 典型问题 | 默认终点 |
| --- | --- | --- | --- |
| `OBSERVE` | 查询当前或历史事实 | 当前 Top SQL、活动会话 | 事实回答 |
| `DIAGNOSE` | 解释异常原因和影响 | 数据库为什么慢 | 根因或明确缺口 |
| `EXPLAIN` | 解释错误、机制或配置 | ORA-01555 是什么 | 知识解释 |
| `PLAN` | 制定方案但不执行 | 扩容、升级、迁移方案 | 方案与检查项 |
| `CHANGE` | 请求产生或执行变更 | 杀会话、改参数 | 建议或 Proposal |
| `VERIFY` | 验证故障或变更结果 | 参数是否生效 | 前后对比结论 |
| `INSPECT` | 按范围进行综合检查 | 健康检查、上线前检查 | Finding 或报告 |

一级意图决定编排和安全流程，不承载具体 DBA 主题。一条消息可以编译为有依赖关系
的多个阶段，例如“确认测试账号造成阻塞后杀掉会话”必须拆成：

```text
OBSERVE → DIAGNOSE → CHANGE → Approval → VERIFY
```

### 3.2 DBA 专业领域

| 专业领域 | 主要对象 |
| --- | --- |
| `SQL_PERFORMANCE` | Top SQL、SQL统计、执行计划、解析和游标 |
| `SESSION_AND_LOCK` | 会话、阻塞链、死锁、长事务 |
| `INSTANCE_PERFORMANCE` | CPU、等待、负载、内存、后台进程 |
| `STORAGE_AND_CAPACITY` | 表空间、数据文件、TEMP、UNDO、ASM、增长趋势 |
| `BACKUP_AND_RECOVERY` | RMAN、备份、归档、恢复能力 |
| `HIGH_AVAILABILITY` | RAC、Data Guard、故障切换、服务状态 |
| `REPLICATION` | 数据同步、延迟和复制错误 |
| `CONFIGURATION` | 参数、资源限制和配置差异 |
| `SECURITY_AND_PRIVILEGE` | 用户、角色、权限、审计和账号状态 |
| `CONNECTION_AND_NETWORK` | Listener、连接、连接池、网络与客户端 |
| `MAINTENANCE` | 统计信息、对象状态、空间维护和例行任务 |
| `PATCH_AND_UPGRADE` | 补丁、版本兼容、升级和回退准备 |
| `DATA_INTEGRITY` | 数据一致性、坏块、约束和对象有效性 |
| `ALERT_AND_LOG` | Alert Log、错误事件和故障时间线 |
| `HOST_AND_OS` | 主机CPU、内存、磁盘、进程和网络 |

专业领域决定候选 Skill，数据库对象和参数决定精确查询。领域允许多选，但必须选出
主要领域；“数据库慢”通常以 `INSTANCE_PERFORMANCE` 为主，并关联
`SQL_PERFORMANCE`、`SESSION_AND_LOCK` 和 `HOST_AND_OS`。

### 3.3 Intent Plan

Intent Router 输出受契约约束的 `DBA_INTENT_PLAN.v1`：

```json
{
  "primary_intent": "OBSERVE",
  "secondary_intents": [],
  "primary_domain": "SQL_PERFORMANCE",
  "related_domains": [],
  "subject": "TOP_SQL",
  "target_scope": "BOUND_AGENT_TARGET",
  "time_scope": {"mode": "RECENT", "duration_seconds": 900},
  "parameters": {"ranking": "ELAPSED_TIME", "limit": 10},
  "requires_fresh_evidence": true,
  "requires_change": false,
  "confidence": "HIGH",
  "clarification": null
}
```

Router 只能使用枚举和允许的参数。确定性校验器负责Target来源、时间范围、返回预算、
变更意图和安全冲突。低置信度只有在影响Skill选择或安全流程时才要求澄清。

不影响安全的歧义使用专业默认值并在回答中说明。例如“看一下Top SQL”默认解释为
最近15分钟活跃SQL、按总耗时排序、返回前10条，并允许用户继续要求按CPU或逻辑读
重排。Target、跨库范围或变更对象不明确时才先澄清。

## 4. DBA Skill模型

Skill表示一个专业DBA任务，不等同于一条SQL或一个底层Tool。例如
`oracle.sql.top_realtime`负责解释“实时Top SQL”任务，它可以编排实例身份确认、
SQL统计查询、会话关联和指标交叉验证；Tool只负责执行其中一个受控动作。

每个Skill使用版本化的`DBA_SKILL_MANIFEST.v1`声明：

```json
{
  "skill_id": "oracle.sql.top_realtime",
  "version": "1.0.0",
  "supported_intents": ["OBSERVE", "DIAGNOSE"],
  "domains": ["SQL_PERFORMANCE"],
  "subjects": ["TOP_SQL"],
  "database_types": ["ORACLE"],
  "required_source_capabilities": [],
  "optional_source_capabilities": ["PROMETHEUS_QUERY"],
  "required_target_capabilities": ["DB_READONLY_SQL_STATS"],
  "required_privileges": ["ORACLE_SELECT_V_SQLSTATS"],
  "required_entitlements": [],
  "input_schema": "oracle.sql.top_realtime.input.v1",
  "defaults": {"window_seconds": 900, "ranking": "ELAPSED_TIME", "limit": 10},
  "limits": {"max_rows": 50, "timeout_seconds": 20},
  "tool_dag": ["db.instance.identity", "db.sql.top_realtime"],
  "output_schema": "oracle.sql.top_realtime.output.v1",
  "presentation_kind": "TABLE_AND_SUMMARY",
  "fallback_skills": ["oracle.session.active"],
  "manual_evidence_template": "oracle.sql.top_realtime.manual.v1"
}
```

Skill Planner必须同时检查意图、领域、数据库类型、Target连接状态、监控源能力、授权、
许可证声明、数据新鲜度和执行预算。不能因为某个Tool可调用就认为该任务已被支持。
数据库差异由不同Skill适配，不在提示词中拼接Oracle、PostgreSQL和MySQL分支SQL。

## 5. 证据获取与最小充分计划

不同证据源承担不同职责：

| 来源 | 适合回答 | 不应承担 |
| --- | --- | --- |
| Prometheus | 趋势、阈值、时间窗口、跨实例对比 | SQL文本、精确会话上下文 |
| Loki/Alert Log | 错误时间线、ORA错误、后台进程事件 | 当前对象状态的唯一依据 |
| 数据库只读连接 | 动态性能视图、对象、会话、SQL和配置事实 | 未审批的变更 |
| Node Exporter | 主机CPU、内存、磁盘、网络 | 数据库内部根因的单独结论 |
| 用户补充 | 截图、命令结果、业务影响和现场上下文 | 未验证内容直接作为已证实事实 |

Evidence Planner先生成“最小充分证据计划”，只取回答当前问题所需的数据，再按不足点
扩展。它必须说明每项证据要区分哪个假设，禁止把所有可用指标无差别塞给模型。

一个典型Top SQL计划为：确认实例和时间窗口；读取窗口内SQL统计；必要时关联活跃会话；
只有用户继续追问某条SQL为什么慢时，才获取执行计划、等待分布、对象统计等深层证据。

所有性能数据必须声明测量语义，至少区分：

- `CURRENT_ACTIVITY`：当前会话或当前活动状态；
- `CUMULATIVE_SINCE_LOAD`：游标或实例加载后的累计值；
- `SNAPSHOT_DELTA`：两个可靠快照之间的增量；
- `HISTORICAL_SAMPLES`：历史采样或诊断仓库中的时间窗数据。

“最近15分钟Top SQL”只能使用可信的快照增量、监控序列或已经授权的历史采样实现。
如果当前只有`V$SQLSTATS`累计值，系统必须明确回答“当前累计Top SQL”的实际口径，
不能把游标加载后的累计数据伪装成15分钟窗口数据。时间范围、排序字段、单位、采样间隔、
实例启动或游标重载造成的断点都属于Evidence元数据。

## 6. 以Turn隔离证据

Conversation保存上下文，Turn定义本轮问题边界。每个用户消息创建独立
`conversation_turn_id`，意图、Skill计划、调用结果、证据、充分性判断和回答都必须关联
到该Turn。历史证据可以复用，但必须显式建立`TurnEvidenceLink`并记录复用原因和新鲜度。

前端和回答生成器只能读取本轮批准的证据集合，不能扫描整个Run的Facts决定展示什么。
因此，上一轮查看表空间后再询问Top SQL，页面不会因为历史事实中存在表空间数据而自动
显示表空间图表。

## 7. 证据充分性与退化状态

每轮取证后输出`SufficiencyAssessment`：

| 状态 | 含义 | 后续行为 |
| --- | --- | --- |
| `ANSWERABLE` | 已足够直接回答 | 生成结论并引用证据 |
| `PARTIAL` | 可回答一部分 | 明确边界，给出下一步 |
| `NEEDS_CLARIFICATION` | 问题范围影响结果或安全 | 在对话中询问用户 |
| `NEEDS_EVIDENCE` | 用户可补充现场证据 | 给出采集方法并等待下一轮 |
| `CAPABILITY_UNAVAILABLE` | 当前环境没有所需Skill、权限或数据源 | 说明缺失能力和启用方法 |
| `UNSAFE` | 请求越权或风险不可接受 | 拒绝执行并给出安全替代方案 |

“没有产生新的有效证据”不是用户可理解的最终答案。系统必须进一步说明：缺的是哪类
证据、为何缺失、已能确认什么、用户下一步最小动作是什么。Tool失败也不能被其他无关
证据掩盖。

## 8. 回答策略与呈现契约

回答形式随意图和数据变化，不套固定的“根因等级、建议、已验证事实”模板：

- `OBSERVE`优先直接回答事实，适合时使用表格或趋势图。
- `DIAGNOSE`说明现象、主要假设、支持/反对证据和下一步区分动作。
- `EXPLAIN`用数据库概念解释当前事实及影响。
- `PLAN`给出有顺序、有前置条件和回退点的方案。
- `CHANGE`先形成变更提案，不在普通聊天文本中暗示已经执行。
- `VERIFY`比较变更前后或故障前后的同口径指标。
- `INSPECT`明确检查范围、异常项和未覆盖项。

回答主体采用自然的DBA对话语气。证据通过引用标记关联到可折叠证据区，不在正文罗列
大量原始指标。结构化展示只能由服务端输出白名单块：

- `answer.markdown`
- `answer.table`
- `answer.chart`
- `evidence.references`
- `clarification.question`
- `evidence.request`
- `proposal.summary`
- `verification.comparison`

`answer.chart`必须包含图表语义、单位、时间范围、序列和来源证据ID。前端只负责渲染，
不得通过字段名猜测应该画表空间、会话或其他图表。没有适合图表的数据时使用表格，
不要为了视觉效果生成无意义图形。

## 9. 对话式补证

证据不足时，AI在对话中像DBA一样提出问题，并给出最小补证方法。例如说明需要某条
命令或只读SQL的结果、用途和敏感信息遮盖要求。用户可粘贴文字或上传截图，系统把它
保存为用户提供的证据，完成解析和确认后进入下一轮。

补证不是独立“卡片流程”，也不要求用户理解内部Capability或Tool。系统有权限时应自行
取证；只有能力不可用、权限不足、信息存在于业务现场或需要用户确认范围时才请求用户。

## 10. 变更、审批与验证闭环

诊断对话可以逐步进入变更，但必须满足Agent允许动手、Target已绑定且连接健康、变更Skill
存在、执行策略允许和用户显式确认。完整闭环为：

```text
对话结论 -> 结构化变更提案 -> 风险和影响分析 -> 人工审批
         -> 受控执行 -> 即时校验 -> 观察窗口 -> 最终验证
```

变更提案包含目标、前置检查、具体动作、预期影响、风险、回退和验证查询。批准的是某个
不可变提案版本；用户继续聊天导致动作变化时必须生成新版本并重新审批。执行结果自动
回到原Conversation，`VERIFY`使用与变更前一致的指标口径给出比较。

## 11. 专业DBA对话工作区

页面沿用现有AIOps视觉系统，信息组织围绕用户的诊断任务，而不是暴露内部Run、Fact、
Capability等对象。

1. 顶部显示当前Agent、数据库Target、连接健康、已选监控源和动手权限；切换Target会开启
   新上下文或要求确认，避免跨库误判。
2. 中间是主对话流。文字、表格、图表、澄清、补证方法、提案和验证结果都按Turn排列。
3. 流式阶段使用用户可理解的状态，如“正在确认实例”“正在查询最近15分钟SQL统计”或
   “正在比较变更前后等待事件”，不展示无意义的内部循环日志。
4. 证据抽屉按本轮默认折叠，展示来源、时间、新鲜度、查询摘要和引用关系；敏感字段脱敏。
5. 变更提案和审批采用清晰的独立区域，但仍保留在原对话时间线中。

页面首次进入提供面向DBA工作的示例问题，而不是功能菜单，例如“最近15分钟哪条SQL消耗
最多”“为什么应用连接变慢”“检查当前阻塞链”“给出表空间未来7天增长风险”。

## 12. 核心领域对象和事件

新增或明确以下对象，均使用版本化契约：

- `ConversationTurn`：本轮用户输入、目标范围和生命周期。
- `IntentPlanSnapshot`：经过校验的意图、领域、对象和参数。
- `SkillPlanSnapshot`：本轮选中的Skill及选择理由。
- `SkillInvocation`：Skill和Tool调用状态、耗时与错误域。
- `TurnEvidenceLink`：证据与Turn的使用关系、新鲜度和用途。
- `SufficiencyAssessment`：证据是否足够以及退化原因。
- `ConversationAnswer`：流式回答和结构化展示块。
- `ChangeProposal`、`ApprovalDecision`、`VerificationResult`：变更闭环。

Turn生命周期为：

```text
ACCEPTED → PLANNING → COLLECTING → ASSESSING → ANSWERING → COMPLETED
                         └→ WAITING_USER
                         └→ PROPOSAL_PENDING
任意执行中状态 → FAILED | CANCELLED
```

同一Conversation默认只允许一个执行中的Turn，后续输入按顺序排队；用户可取消当前Turn，
但已固化的证据和审计记录不删除。`WAITING_USER`收到补证后创建新的响应阶段并保留原问题，
不把补证内容误判成一个无关任务。SSE事件具有单调序号，刷新或断线后从服务端重放，
`answer.completed`之前的正文和展示块都可恢复。

建议的事件顺序为：

```text
turn.created
intent.planned
skill.plan.created
skill.started / skill.progress / skill.completed | skill.failed
evidence.linked
sufficiency.assessed
answer.delta / answer.block / answer.completed
```

事件必须携带`conversation_id`、`turn_id`和关联ID。`answer.delta`只承载正文增量，表格、
图表和引用使用`answer.block`发送，避免客户端从Markdown或事实集合二次推断。

模型职责也必须分离：Intent Router只输出结构化意图；Skill Planner只在Manifest目录内
规划；回答模型只接收本轮已批准的证据摘要、引用ID和对话必要上下文。所有模型使用Agent
绑定的有效LLM配置和服务端预算，任何阶段都不接收数据库凭据、完整连接串或未脱敏原始
日志。确定性代码校验模型输出，不能用Prompt代替权限、安全和契约约束。

## 13. Oracle首批专业Skill

Oracle生产验证阶段至少补齐：

| 领域 | Skill |
| --- | --- |
| 实例 | 实例概览、负载与主要等待、资源限制 |
| SQL性能 | 实时Top SQL、SQL详情、执行计划、解析与游标概览 |
| 会话与锁 | 活跃会话、阻塞链、长事务、会话详情 |
| 存储 | 表空间容量与趋势、数据文件、TEMP、UNDO |
| 告警 | Alert Log错误时间线、实例异常关联 |
| 高可用 | Data Guard状态和延迟 |
| 主机 | 数据库进程、CPU、内存、磁盘和网络关联 |

历史AWR/ASH类Skill单独声明授权和许可前提，未明确启用时不作为实时Top SQL的隐式依赖。
系统根据所选Target执行能力发现，展示“支持、缺少权限、缺少数据源、未授权”状态；所需
最小授权脚本由Skill Manifest生成并校验，避免文档、运行时SQL和授权语句漂移。

## 14. 安全与失败边界

- LLM不直接选择任意SQL、URL或Shell命令，只能选择注册Skill并提供契约内参数。
- 只读连接只执行预定义、带上限和超时的查询；自定义补证SQL默认由用户自行执行。
- 变更Tool与诊断Tool分离，审批后才能取得短时执行资格。
- 每个错误归属到监控源、Target、Skill、Tool、内部存储或模型服务，不能用统一
  `SOURCE_UNREACHABLE`掩盖真实原因。
- 后台审计表故障不得伪装成被监控数据库权限错误，也不得覆盖已成功的用户Turn。
- 超时、重试和预算按Skill声明；重复得到同一证据时应改变计划或明确退化，不能空转。

## 15. 验收场景

1. 用户问“分析现在数据库上的Top SQL”，系统调用Top SQL Skill，返回目标实例、实际
   时间口径、排序规则、排名表和简短观察；不会把累计值伪装为15分钟增量，也不会显示
   历史表空间图表。
2. 用户追问“第一条为什么慢”，系统复用对应SQL ID，补取计划、等待和会话证据，不要求
   用户重新描述对象。
3. Target缺少查询SQL统计的权限时，回答明确缺少的Capability和授权方法，不以无进展结束。
4. 用户问“数据库慢不慢”，系统先看趋势和实例负载，再根据异常选择SQL、等待或主机Skill，
   不固定执行所有工具。
5. 用户要求执行高风险变更时，普通对话只生成提案；未审批不执行，审批后自动验证并回填。
6. 用户上传命令截图后，系统标记为用户证据，要求确认关键字段或解析结果后再形成结论。
7. 同一Conversation连续讨论表空间和Top SQL时，两轮证据和展示严格隔离。
8. 流式连接中断后按事件序号恢复同一Turn；取消Turn后不再调用新Skill，也不会丢失已完成
   的证据审计。

## 16. 实施顺序

1. 建立Turn、Intent Plan、Skill Manifest、证据关联和展示块契约。
2. 实现Intent Router的结构化输出与确定性校验，移除“每轮必走根因链”的假设。
3. 补齐Oracle首批Skill，优先实时Top SQL、SQL详情、主要等待和资源限制。
4. 重构取证编排和充分性判断，建立明确退化路径。
5. 重构流式协议和对话工作区，删除前端基于全部Facts猜图表的逻辑。
6. 接入变更提案、审批、执行和验证闭环。
7. 通过上述验收场景和真实Oracle环境验证后，再扩展PostgreSQL和MySQL Skill。

这是KBot 4.0的新契约，不保留旧的固定诊断回答、全局Facts展示或并行兼容路径；旧实现
在新流程覆盖并验证后直接删除。
