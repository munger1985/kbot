# AIOps Agent 专业 DBA 调查改造方案与实施计划

版本：2.0
状态：实施中
基准日期：2026-08-28

## 1. 目标与基准

本计划以以下文档为唯一基准：

- [AIOps Agent 专业 DBA 调查设计](../product/aiops-agent-chat-diagnosis.md)；
- [AIOps Agent 专业 DBA 调查详细设计](../architecture/aiops-agent-chat-diagnosis.md)。

目标是把“单一Intent精确匹配固定Skill”的聊天链替换为“输入理解—Task Frame—调查计划—
Tool取证—证据评估—重规划—回答”的AI DBA调查循环，不在旧链旁新增兼容分支。

## 2. 当前基线与替换范围

复用Conversation、Turn、Run、Task、Artifact、Evidence、Answer、Event、Outbox、租约、SSE、
Target、Source、Agent版本、Policy、凭据、监控Adapter、Oracle诊断执行器和变更审批。

替换：

- 单文本输入；
- `DbaIntentRouter`第一道硬路由；
- `DbaSkillPlanner`精确匹配；
- 普通问题只选一个Skill；
- `AIOPS_SKILL_UNAVAILABLE`规划终止；
- 一次规划且不能Replan；
- Skill Invocation作为Tool和Evidence必经父对象；
- 只有`DATABASE_OVERVIEW`才查询监控的特殊分支。

## 3. 交付原则

1. 目标Schema直接升级到14、`aiops-oracle-v4`，不写增量升级或兼容代码；
2. DDL、重建制品、Manifest、Entity、Repository、契约、API、UI、测试和文档同步；
3. 每阶段形成可运行纵向切片，不以编译通过冒充功能完成；
4. 模型输出使用版本化结构契约并由服务端校验；
5. Tool范围由确定性策略控制，不依赖模型自律；
6. 不自动重建共享数据库或重启dev服务。

### 3.1 当前落地状态

截至2026-08-28，Schema 14纵向切片已进一步完成以下闭环：用户提供的Artifact可按Artifact
Key或ID进入评估器；数据库访问决定由Agent Policy、Target状态、连接状态、凭据和Endpoint在
规划时冻结，执行阶段不得覆盖；模型Action与数据库Tool按一对一关系冻结和审计，Playbook不再
扩大实际执行范围；Task Frame支持多个目标；告警或巡检来源Run的最终Artifact会复制为当前
Turn的继承Evidence；多Target Agent在聊天页面显式选择Target。

首轮Assessment若存在可重试的关键缺口，系统会冻结回答Task，通过Outbox可靠生成第二版
Investigation Revision和Task DAG；无参数变化的重复Tool调用会被拒绝。第二轮结束后统一回答，
重规划失败则回退到首轮真实证据，避免Turn卡在`REPLANNING`。当前调查预算固定为最多两轮。

聊天中的变更意图现已进入独立的确定性链路：`DBA_SUFFICIENCY.v1 → ACTION_PLAN.v1 →
PROPOSAL_OUTCOME.v1 → PROPOSAL_SUMMARY`。只有当前Turn内`SOURCE_VERIFIED`数据库事实能够完整
绑定已发布Action Template参数、Agent允许执行、Target可连接且配置执行凭据时，才生成
`PENDING_APPROVAL`；自然语言、用户粘贴证据和模型推断均不能直接授权动作。审批后仍复用原有
Proposal Hash、策略复核、执行凭据和效果验证链路。

受控动态查询已形成运行时纵向链路：项目固定使用`sqlglot==30.0.2`按Oracle方言解析AST，动态
SQL策略只接受单条显式投影SELECT，限制诊断对象、敏感源列、函数、bind、Schema、Database
Link、锁和返回行数，并生成Query Hash与Policy Hash。`db.oracle.readonly_query`仅在Oracle
Target具有`DB_READONLY`能力时进入模型Tool Discovery；规划端规范化SQL并冻结策略，Worker签发
短期动态Grant，隔离Executor在领取一次性诊断凭据前重新验证SQL、参数、策略和投影列，并在
只读事务中执行。成功结果进入`DBA_SKILL_RESULT.v1`成为`SOURCE_VERIFIED` Evidence，失败只形成
Evidence Gap；固定目录Grant与动态Grant不能串用。固定目录SQL也继续执行方言AST复核。

受控监控查询也已接入调查运行时。PromQL使用`promql-parser==0.10.0`解析官方语法AST，每个
Vector Selector都必须以`instance="${external_target}"`或
`target_key="${host_target}"`精确约束当前Target，并限制Range、时间窗、采样点、Series数量、
响应字节和`@`/`offset`时间逃逸；规划端和Worker都会复核Query/Policy Hash。LogQL不允许模型
提供任意Selector，只能使用`${binding_selector}`引用冻结Binding标签，并附加有限个`|=`或`!=`
字面量过滤；查询结果沿现有`OBSERVATION_SET.v1`和`LOG_EVIDENCE_SET.v1`进入Evidence链。

以下项目仍按后续阶段实施，不冒充完成：文件和图片上传解析、旧Skill内部命名与目录的物理
迁移，以及动态查询在dev真实
Oracle、Prometheus、Loki和Alertmanager联调。阶段完成情况以本节和验收记录为准，不能仅凭
Schema字段或类名判断功能已经交付。

## 4. 阶段1：Schema 14和共享契约

- 修改`008_ops_conversations_reports.sql`；
- 更新`006_ops_fks_views.sql`的版本、FK、索引、注释和视图；
- 更新重建文件、Manifest和README；
- 调整Turn列和状态；
- 新增Input Item、Investigation Revision、Tool Invocation；
- Skill Invocation重命名为Playbook Invocation；
- 扩展Turn Evidence和Turn Event；
- 更新Entity、Repository和UoW；
- 新增输入、Task Frame、Plan、Assessment、Tool和Playbook契约；
- 删除旧Intent/Skill Plan作为运行控制源的契约。

门槛：14、`aiops-oracle-v4`、43张表、10个视图；FK索引、Entity、Manifest、自包含重建和
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

- 实现多目标`DbaTaskFramingService`；
- 实现`DbaInvestigationPlanner`和Plan Validator；
- 建立Tool Discovery和Playbook候选检索；
- 持久化Task Frame和Plan Revision；
- 删除旧Router、Planner和`turn_planning.py`入口；
- 删除`AIOPS_SKILL_UNAVAILABLE`聊天终止语义；
- 无Playbook时进入通用调查。

门槛：多目标保留；目录外问题可规划；模型提出未知Tool、越权Target或循环依赖时被确定性
拒绝并安全重试。

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

门槛：Tool声明最小权限、版本、口径、超时和脱敏；无AWR许可不调用受限能力；停库案例能
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

- 聊天支持文本、粘贴材料、文件、图片和来源Run；
- 会话按Agent隔离；
- 展示理解、规划、取证、评估、重规划和回答；
- Evidence统一折叠，补证自然呈现，SSE断线恢复；
- 变更意图在证据评估后单独编译Action Plan；页面展示Proposal摘要并逐条批准或拒绝；
- 建立停库、日志、离线、口径冲突、权限、网络、SQL、锁、容量、归档、内存、多目标、错误
  假设、无Playbook、全来源不可用和变更审批评测集。

评价输入理解、假设、Tool选择、无效调用、Evidence引用、结论边界、补证质量和安全性。

## 11. 代码改造矩阵

| 现有位置 | 处理方式 |
| --- | --- |
| `application/turns.py` | 接收内容项并创建Input Item摘要 |
| `application/turn_planner.py` | 保留Primary Run骨架，改投递UNDERSTANDING |
| `application/turn_planning.py` | 拆分后删除 |
| `skills/router.py` | 删除，由Input Understanding和Task Framing替代 |
| `skills/planner.py` | 删除，由Investigation Planner替代 |
| `skills/execution.py` | Tool执行和Playbook编排分别接管 |
| `skills/registry.py` | 迁移为Playbook Registry |
| `workers/skill_handlers.py` | 迁移Playbook Handler |
| `workers/evidence_handlers.py` | 统一Tool Evidence归一 |
| `workers/turn_answer_handlers.py` | 使用Task Frame、Assessment和Evidence |
| `application/runtime/service.py` | 继续提供Run/Task内核，接收新Plan DAG |
| `ui/aiops/js/aiops-workspaces.js` | 多内容输入和新SSE事件 |

## 12. 切换和数据库部署

完成Schema 14全链后：运行离线测试；停止AIOps API、Worker和Scheduler；备份；执行规范重建
脚本；验证`AIOPS / 14 / aiops-oracle-v4`；重配Agent、Target、Source和绑定；执行聊天、告警、
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
