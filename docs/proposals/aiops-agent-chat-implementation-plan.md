# AIOps Agent 专业 DBA 调查改造方案与实施计划

版本：2.2
状态：实施中
基准日期：2026-08-31

## 1. 目标与基准

本计划以以下文档为唯一基准：

- [AIOps Agent 专业 DBA 调查设计](../product/aiops-agent-chat-diagnosis.md)；
- [AIOps Agent 专业 DBA 调查详细设计](../architecture/aiops-agent-chat-diagnosis.md)。

目标是把“单一Intent精确匹配固定Skill”的聊天链替换为“输入理解—Task Frame—调查计划—
Tool取证—证据评估—重规划—回答”的AI DBA调查循环，不在旧链旁新增兼容分支。

已登记受控动作从单会话终止扩展到 DBA 日常运维领域的目标设计、单人审批边界、破坏性动作
人工执行规则和分阶段实施计划，统一由
[AIOps 已登记受控动作设计与实施计划](aiops-controlled-actions-implementation-plan.md)约束。
该文档同时记录已完成和待实施范围；具体可执行能力必须以状态说明和当前 Action Catalog 为准。

## 2. 当前基线与替换范围

复用Conversation、Turn、Run、Task、Artifact、Evidence、Answer、Event、Outbox、租约、SSE、
Target、Source、Agent版本、Policy、凭据、监控Adapter、Oracle诊断执行器和变更审批。

替换：

- 单文本输入；
- `DbaIntentRouter`第一道硬路由；
- 旧Skill Planner精确匹配；
- 普通问题只选一个Skill；
- `AIOPS_SKILL_UNAVAILABLE`规划终止；
- 一次规划且不能Replan；
- Skill Invocation作为Tool和Evidence必经父对象；
- 只有`DATABASE_OVERVIEW`才查询监控的特殊分支。

## 3. 交付原则

1. 目标Schema直接升级到16、`aiops-oracle-v6`，不写增量升级或兼容代码；
2. DDL、重建制品、Manifest、Entity、Repository、契约、API、UI、测试和文档同步；
3. 每阶段形成可运行纵向切片，不以编译通过冒充功能完成；
4. 模型输出使用版本化结构契约并由服务端校验；
5. Tool范围由确定性策略控制，不依赖模型自律；
6. 不自动重建共享数据库或重启dev服务。

### 3.1 当前落地状态

截至2026-08-28，Schema 15纵向切片已进一步完成以下闭环：用户提供的Artifact可按Artifact
Key或ID进入评估器；数据库访问决定由Agent Policy、Target状态、连接状态、凭据和Endpoint在
规划时冻结，执行阶段不得覆盖；模型Action与数据库Tool按一对一关系冻结和审计，Playbook不再
扩大实际执行范围；Task Frame支持单一Target内的多个任务目标；告警或巡检来源Run的最终
Artifact会复制为当前Turn的继承Evidence。一个Agent版本可绑定多个逻辑Target；聊天创建会话时
必须先选择其中一个Target，会话冻结该Target，后续Turn不再接受Target覆盖。

首轮Assessment若存在系统仍可采集的关键缺口，系统会冻结回答Task，通过Outbox可靠生成第二版
Investigation Revision和Task DAG；模型选择带限制回答不能跳过这次确定性补证，无参数变化的
重复Tool调用会被拒绝。告警Turn还会主动尝试已授权、已绑定的指标和日志来源。第二轮结束后统一回答，
重规划失败则回退到首轮真实证据，避免Turn卡在`REPLANNING`。当前调查预算固定为最多两轮。

聊天中的变更意图现已进入独立的确定性链路：`DBA_SUFFICIENCY.v1 → ACTION_PLAN.v1 →
PROPOSAL_OUTCOME.v1 → PROPOSAL_SUMMARY`。只有当前Turn内`SOURCE_VERIFIED`数据库事实能够完整
绑定已发布Action Template参数、Agent允许执行、Target可连接且配置执行凭据时，才生成
`PENDING_APPROVAL`；自然语言、用户粘贴证据和模型推断均不能直接授权动作。审批后仍复用原有
Proposal Hash、策略复核、执行凭据和效果验证链路。

受控动态查询已形成运行时纵向链路：项目固定使用`sqlglot==30.17.0`按Oracle方言解析AST，动态
SQL策略只接受单条SELECT，允许受控系统诊断视图的星号投影，限制诊断对象、敏感源列、函数、bind、Schema、Database
Link、锁和返回行数，并生成Query Hash与Policy Hash。`db.oracle.readonly_query`仅在Oracle
Target具有`DB_READONLY`能力时进入模型Tool Discovery；规划端规范化SQL并冻结策略，Worker签发
短期动态Grant，隔离Executor在领取一次性诊断凭据前重新验证SQL、参数、策略和投影列，并在
只读事务中执行。成功结果进入`DBA_TOOL_RESULT.v1`成为`SOURCE_VERIFIED` Evidence，失败只形成
Evidence Gap；固定目录Grant与动态Grant不能串用。固定目录SQL也继续执行方言AST复核。

受控监控查询也已接入调查运行时。PromQL使用`promql-parser==0.10.0`解析官方语法AST，每个
Vector Selector都必须以`instance="${external_target}"`或
`target_key="${host_target}"`精确约束当前Target，并限制Range、时间窗、采样点、Series数量、
响应字节和`@`/`offset`时间逃逸；规划端和Worker都会复核Query/Policy Hash。LogQL不允许模型
提供任意Selector，只能使用`${binding_selector}`引用冻结Binding标签，并附加有限个`|=`或`!=`
字面量过滤；查询结果沿现有`OBSERVATION_SET.v1`和`LOG_EVIDENCE_SET.v1`进入Evidence链。

文件和图片输入链路已经完成：受控原始流上传、Domain/用户归属、大小和Hash校验、长期
Artifact保存、UTF-8文本提取、按Agent能力选择VLM/OCR、独立提取Artifact以及
`USER_PROVIDED` Evidence关联均已接入；单个附件解析失败不会抹掉原始证据或阻止其他材料
继续诊断。Oracle固定工具目录已进一步覆盖参数、TEMP/UNDO、Redo、近期Alert Log、失败调度
任务、无效对象、RMAN作业和Data Guard延迟。Oracle诊断账号统一使用`CREATE SESSION`和
`SELECT ANY DICTIONARY`，覆盖动态系统视图以及AWR/ASH调查，不在AIOps运行时执行许可证
能力门控。以下项目仍按后续阶段
实施，不冒充完成：动态查询在dev真实Oracle、Prometheus、Loki和Alertmanager联调。阶段完成
情况以本节和验收记录为准，不能仅凭Schema字段或类名判断功能已经交付。

2026-08-31恢复实施时已完成第一批基线收口：工作区根依赖和AIOps服务包统一固定
`sqlglot==30.17.0`；诊断目录验收按各数据库必备Tool集合检查，不再错误要求Oracle、MySQL和
PostgreSQL工具数量相等。该调整只修复依赖与验收契约漂移，不代表真实环境联调已经完成。

同日完成旧Skill物理迁移：调查应用迁入`application/investigation/`并拆出上下文、错误映射、
Tool/Playbook发现和动态查询冻结模块；Playbook目录、Tool编译与执行快照、Worker Handler和共享
合同均采用Playbook/Tool命名，执行快照改为`investigation_execution`，结果合同改为
`DBA_TOOL_RESULT.v1`，旧`AIOPS_SKILL_UNAVAILABLE`不再参与运行时终态语义。

专业评测Runner现会在云端Main API逐场景创建真实Conversation，消费Turn SSE，读取最终Tool、
Evidence、引用、回答和安全事件并评分；完整模式再调用独立OpenAI兼容裁判复核正向与反向行为。
Runner会按`tool.completed`或`tool.gap`中的具体数据库`tool_id`验证Top SQL、阻塞链、容量、归档、
FRA和长尾只读场景，兼容SSE结束帧、裁判Markdown JSON代码块和文字包裹，并拒绝未知场景、非法
超时与分数门槛。本地只保留数据集和生成物静态校验，真实评测结果必须来自云端运行及在线
Operations Logs。

本轮审计整改进一步完成：Turn原始文字和上传定位会在输入理解/VLM/OCR与调查模型之前以独立
事务保存；固定Oracle Tool直接从Diagnostic Registry发现并编译为原子Task，不再要求存在
Playbook父调用；`DBA_SUFFICIENCY.v1`内嵌`aiops.investigation-assessment.v1`，由诊断模型更新
假设、未知项、Evidence Gap、进展和`ANSWER/REPLAN/ASK_USER/STOP_UNSAFE`决策；已启用但上次
健康检查失败的Target或Source允许在本Turn预算内尝试一次。聊天公开契约不允许后续Turn覆盖
`target_id`；新建Conversation时从Agent版本绑定的多个逻辑Target中选择一个并冻结。专业评测集覆盖停库、Top SQL、锁、容量、归档、
权限、监控健康陈旧、动态只读、来源Run、单Target多任务目标和变更安全。

### 3.2 2026-08-31静态验收与云端待执行

本次恢复实施未在本机运行任何Python单测、契约测试、数据库测试或真实评测。本机Oracle不可达，
所有代码、Schema和外部依赖测试必须在云端`kbot4` Conda环境执行；失败取证只使用
`http://140.238.44.208:8080/operations-logs.html`及其在线Logs API，不使用本地或历史日志。

本次已完成且仅能视为静态证据的检查：

- `git diff --check`通过；
- AIOps OpenAPI、Schema Manifest、Playbook Manifest和评测数据集均可由`jq`解析；
- `002_ops_runtime.sql` SHA-256为
  `66eed35dd7f89910c45398a9313e9bbf77c5c32ea08bc9b5b1e6e3c81a04a2cc`，与Manifest一致；
- `rebuild_aiops_schema.sql`内嵌的`002_ops_runtime.sql`与规范源文件逐字一致；
- AIOps活跃Python、JSON和SQL中不存在旧Skill模块导入、旧类型或旧运行字段；历史升级脚本和评测
  负向断言保留旧名称用于迁移与回归验证。

云端代码基线验收命令：

```bash
conda run -n kbot4 python -m unittest discover \
  -s tests/unit/aiops_agent -t . -p 'test_*.py'
conda run -n kbot4 python -m unittest \
  tests.contract.test_aiops_dba_evaluation_dataset \
  tests.contract.test_aiops_rebuild_schema_script \
  tests.contract.test_aiops_schema_upgrade \
  tests.contract.test_aiops_ui_static_pages
conda run -n kbot4 python tests/acceptance/check_aiops_diagnostic_catalog.py
conda run -n kbot4 python tests/acceptance/check_oracle_schema.py
conda run -n kbot4 python tools/db/render_aiops_rebuild_schema.py --check
```

云端12场景真实评测命令使用环境变量提供Main API、Agent、Target和裁判参数，不在命令行或报告中
写入密钥。完整模式需要`KBOT_AIOPS_EVAL_BASE_URL`、`KBOT_AIOPS_EVAL_API_KEY`、
`KBOT_AIOPS_EVAL_AGENT_ID`、`KBOT_AIOPS_EVAL_TARGET_ID`、`KBOT_AIOPS_EVAL_JUDGE_URL`、
`KBOT_AIOPS_EVAL_JUDGE_KEY`和`KBOT_AIOPS_EVAL_JUDGE_MODEL`：

```bash
conda run -n kbot4 python tests/evaluation/evaluate_aiops_dba_chat.py \
  --report /tmp/aiops-dba-chat-evaluation.json
```

来源Run续查场景还必须提供`KBOT_AIOPS_EVAL_SOURCE_RUN_ID`。真实联调完成前，文档状态保持
“实施中”。

## 4. 阶段1：Schema 15和共享契约

- 修改`008_ops_conversations_reports.sql`；
- 更新`006_ops_fks_views.sql`的版本、FK、索引、注释和视图；
- 更新重建文件、Manifest和README；
- 新增Agent版本–Target多对多关系、Conversation冻结Target以及Target连接能力开关；
- 调整Turn列和状态；
- 新增Input Item、Investigation Revision、Tool Invocation；
- Skill Invocation重命名为Playbook Invocation；
- 扩展Turn Evidence和Turn Event；
- 更新Entity、Repository和UoW；
- 新增输入、Task Frame、Plan、Assessment、Tool和Playbook契约；
- 删除旧Intent/Skill Plan作为运行控制源的契约。

门槛：16、`aiops-oracle-v6`、44张表、10个视图；FK索引、Entity、Manifest、自包含重建和
UoW回滚测试全部通过。

## 5. 阶段2：输入理解和用户证据

- Main API和内部API采用内容项契约；
- Turn接收原子写入Message和Input Item摘要；
- Primary Run建立后持久化Raw Artifact；
- 实现`TurnInputUnderstandingService`；
- 识别问题、Alert Log、ORA栈、SQL结果、Shell输出、配置和普通文本；
- 创建Parsed Artifact和`USER_PROVIDED` Evidence；
- 实现数据/指令隔离和输入脱敏；
- SSE和UI展示输入理解进度。

门槛：ORA-27157日志形成用户证据；原文和解析结果可审计；无外部Source仍可回答；日志中的
Prompt Injection不能改变权限。

## 6. 阶段3：Task Frame和调查计划

- 实现单一Target内支持多任务目标的`DbaTaskFramingService`；
- 实现`DbaInvestigationPlanner`和Plan Validator；
- 建立Tool Discovery和Playbook候选检索；
- 持久化Task Frame和Plan Revision；
- 删除旧Router、Planner和`turn_planning.py`入口；
- 删除`AIOPS_SKILL_UNAVAILABLE`聊天终止语义；
- 无Playbook时进入通用调查。

门槛：单一Target内的多任务目标保留；目录外问题可规划；模型提出未知Tool、试图覆盖Agent
Target或循环依赖时被确定性拒绝并安全重试。

## 7. 阶段4：Tool执行和Assess–Replan

- Tool Registry、Manifest、Validator和Invocation Repository；
- Coordinator把Plan Action编译为Ops Task DAG；
- Tool结果统一写Artifact和Turn Evidence；
- Evidence Assessor与Replan；
- 调查预算、重复调用和无进展检测；
- Source/Target健康只影响策略，不阻止Turn；
- Answer Handler读取Task Frame、Hypothesis、Assessment和Evidence；
- SSE增加Tool、Evidence、Assessment和Replan事件。

门槛：单Tool失败可选替代来源；Target离线仍工作；全部Tool不可用仍有边界明确的回答；调查
在预算内收敛；回答引用当前Turn Evidence。

## 8. 阶段5：Playbook和Oracle Tool

把现有实例性能、等待、归档、会话、阻塞、Top SQL和表空间Skill转为Playbook，保留版本和
Hash审计，移除Subject精确准入。

Oracle Tool优先级：

1. Alert Log、ORA错误和时间线；
2. 实例/PDB可用性、启动停机和服务；
3. Prometheus、Loki、Alertmanager和Node Exporter关联；
4. 会话、锁、事务、等待、Top SQL和SQL详情；
5. SGA、PGA、process/session；
6. 表空间、数据文件、TEMP和UNDO；
7. Redo、Archive和FRA；
8. 参数、Listener和Service；
9. Data Guard、ASM、RAC、Scheduler、备份、统计信息、对象和权限。

门槛：Tool声明版本、口径、超时和脱敏；AWR/ASH可按调查需要调用；停库案例能
关联监控时间线并给出最小主机补证。

## 9. 阶段6：受控动态查询

- Oracle SQL AST和只读策略；
- `db.oracle.readonly_query`；
- 对象、函数、Schema、行数、耗时和字节围栏；
- 诊断凭据授权和审计；
- PromQL和LogQL语法、标签、Target范围和预算校验；
- 失败只形成Evidence Gap，不回退到不受控执行。

门槛：长尾只读问题无需新增固定Skill；危险SQL全部拒绝；Secret和敏感列不进入模型或SSE；
动态诊断不能绕过变更审批。

## 10. 阶段7：UI、API和专业评测

- 聊天已支持文本、粘贴材料、文件、图片和来源Run；
- 会话按Agent隔离；
- 展示理解、规划、取证、评估、重规划和回答；
- Evidence统一折叠，补证自然呈现，SSE断线恢复；
- 变更意图在证据评估后单独编译Action Plan；页面展示Proposal摘要并逐条批准或拒绝；
- 建立停库、日志、离线、口径冲突、权限、网络、SQL、锁、容量、归档、内存、单Target多任务
  目标、错误假设、无Playbook、全来源不可用和变更审批评测集。

评价输入理解、假设、Tool选择、无效调用、Evidence引用、结论边界、补证质量和安全性。

## 11. 代码改造矩阵

| 现有位置 | 处理方式 |
| --- | --- |
| `application/turns.py` | 接收内容项并创建Input Item摘要 |
| `application/turn_planner.py` | 保留Primary Run骨架，改投递UNDERSTANDING |
| `application/investigation/service.py` | 调查上下文冻结、计划编译和持久化 |
| 旧Skill Router | 已删除，由Input Understanding和Task Framing替代 |
| `tools/compiler.py` | 将已验证的Investigation Plan编译为Task DAG |
| `tools/execution_snapshot.py` | 冻结原子Tool与可选Playbook执行快照 |
| `playbooks/registry.py` | 加载可选Playbook Manifest并生成目录Hash |
| `workers/tool_handlers.py` | 分别执行Playbook和原子Tool |
| `workers/evidence_handlers.py` | 统一Tool Evidence归一 |
| `workers/turn_answer_handlers.py` | 使用Task Frame、Assessment和Evidence |
| `application/runtime/service.py` | 继续提供Run/Task内核，接收新Plan DAG |
| `ui/aiops/js/aiops-workspaces.js` | 多内容输入和新SSE事件 |

## 12. 切换和数据库部署

完成Schema 15全链后：运行离线测试；停止AIOps API、Worker和Scheduler；备份；执行规范重建
脚本；验证`AIOPS / 16 / aiops-oracle-v6`；重配Agent、Target、Source和绑定；执行聊天、告警、
巡检Smoke；确认审批未被放宽。

不提供Schema 13在线迁移、兼容读取或回滚表。回退使用Git和数据库备份恢复完整版本。

## 13. 完成定义

- 文档、Schema、契约、代码、UI和测试一致；
- 旧Intent→Skill主链已删除；
- 用户材料先成为Evidence；
- 无Playbook、Target离线和Source异常不阻止Agent；
- 调查可基于Evidence Replan并在预算内收敛；
- Oracle生产核心Tool和安全动态只读查询可用；
- 回答引用Evidence并区分事实与推断；
- 诊断不能绕过变更审批；
- dev真实Oracle、Prometheus、Loki和Alertmanager场景通过。
