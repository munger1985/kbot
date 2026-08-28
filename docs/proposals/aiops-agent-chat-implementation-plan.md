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

截至2026-08-28，本轮已完成首个可运行纵向切片：Schema 14及审计实体、内容项契约、
用户材料理解、Task Frame、首轮调查计划、Tool/Playbook目录发现、用户证据入库、Oracle/
Prometheus/Loki现有受控能力编排、证据归一、自然回答上下文和新SSE进度事件。纯粘贴Oracle
Alert Log且不调用外部Tool的路径已经由端到端单元测试覆盖。

以下项目仍按后续阶段实施，不在首个切片中冒充完成：基于Assessment创建第二轮Task DAG、
文件和图片上传解析、受控动态只读SQL/PromQL/LogQL、旧Skill内部命名与目录的物理迁移，以及
dev真实Oracle、Prometheus、Loki和Alertmanager联调。阶段完成情况以本节和验收记录为准，
不能仅凭Schema字段或类名判断功能已经交付。

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
