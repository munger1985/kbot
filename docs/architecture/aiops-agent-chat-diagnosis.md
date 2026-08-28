# AIOps Agent 专业 DBA 调查详细设计

版本：2.0
状态：已批准，实施中
目标数据库契约：Schema 14 / `aiops-oracle-v4`

## 1. 范围和替换原则

本文把
[AIOps Agent 专业 DBA 调查设计](../product/aiops-agent-chat-diagnosis.md)
下钻为服务、契约、Oracle表结构、状态机、Tool、Playbook、Worker、API、SSE和验收设计。
实施顺序见
[AIOps Agent 专业 DBA 调查改造方案与实施计划](../proposals/aiops-agent-chat-implementation-plan.md)。

本次直接替换现有 `Intent Router → Skill Planner → Skill Compiler` 主链。KBot 4.0 不保留
Schema 13兼容列、旧规划事件、双读双写或旧Planner回退路径。Run/Task、Artifact、Outbox、
SSE、审批执行器等可靠性基础继续复用。

## 2. 当前实现问题

当前实现把Message压缩为单一`text`，模型先输出单一Intent、Domain和Subject，确定性Planner
再精确匹配最多一个Skill。由此导致：

1. 用户日志、SQL结果和命令输出没有成为Evidence；
2. 多目标问题被过早压缩；
3. Skill目录定义了Agent能回答的问题全集；
4. 无Skill、Target离线或能力不足时在规划阶段终止；
5. Intent成功但Skill失败时Intent也没有持久化；
6. 只规划一次，不能根据新证据调整调查；
7. Skill Invocation同时承担Playbook、Tool和Evidence来源职责；
8. 固定SQL目录无法覆盖真实DBA的长尾只读问题。

## 3. 目标聚合

```text
Conversation
  └─ Turn
      ├─ Message
      ├─ Input Item → Raw / Parsed Artifact
      ├─ Task Frame Artifact
      ├─ Investigation Revision 1..N
      │   ├─ Plan Artifact
      │   ├─ Playbook Invocation 0..N
      │   ├─ Tool Invocation 0..N → Task → Artifact
      │   └─ Assessment Artifact
      ├─ Turn Evidence 0..N
      ├─ Evidence Request 0..N
      ├─ Answer Block → Citation
      └─ Turn Event
```

| 对象 | 职责 |
| --- | --- |
| Conversation | Agent维度的持续会话和历史 |
| Turn | 一轮用户输入、调查、回答和事件边界 |
| Input Item | 用户本轮提供的独立内容项 |
| Task Frame | 多目标问题定义、范围和约束 |
| Investigation Revision | 一版不可变调查计划及评估结果 |
| Playbook Invocation | 可选专业DBA调查套路 |
| Tool Invocation | 一次受控原子能力调用 |
| Artifact | 原始材料、模型输出、工具结果和最终产物 |
| Turn Evidence | Artifact在本轮中的证据角色和口径 |
| Answer Block / Citation | 服务端批准的展示和证据引用 |

## 4. Turn状态机

```text
QUEUED → ACCEPTED → UNDERSTANDING → PLANNING
                                      ↓
                                  COLLECTING
                                      ↓
             REPLANNING ←──────── ASSESSING
                 │                    ├→ WAITING_USER
                 └────→ COLLECTING    ├→ ANSWERING
                                      └→ PROPOSAL_PENDING

ANSWERING → COMPLETED / PARTIAL
任意非终态 → CANCELLED
系统级不可恢复错误 → FAILED
```

同一Conversation最多一个处于`UNDERSTANDING`、`PLANNING`、`COLLECTING`、`ASSESSING`、
`REPLANNING`或`ANSWERING`的Turn。Source/Target不健康、无专用Playbook、单Tool失败和权限
不足不是`FAILED`条件。

## 5. 输入与Task Frame契约

公开API使用`AIOPS_TURN_INPUT.v2`：

```json
{
  "content": [
    {"type": "TEXT", "text": "数据库现在停了，分析原因"},
    {
      "type": "PASTED_CONTENT",
      "media_type": "text/plain",
      "content": "2026-08-28...ORA-27157..."
    }
  ],
  "target_id": null,
  "source_run_id": null
}
```

内容类型为`TEXT`、`PASTED_CONTENT`、`FILE`、`IMAGE`、`SOURCE_RUN_REFERENCE`。文件和图片
使用受控上传Artifact，不在JSON中传输无界Base64。

输入理解输出`TURN_INPUT_ENVELOPE.v1`：`request_fragments`、`material_items`、`systems`、
`target_hints`、`time_anchors`、`error_codes`、`stated_facts`、`user_hypotheses`、
`desired_outcomes`和`clarification_need`。原始输入先持久化，解析错误形成Gap，不删除原文。

`DBA_TASK_FRAME.v1`替代单一Intent Plan：

- `objectives[]`；
- `intent_tags[]`和`domain_tags[]`；
- `systems[]`和`target_scope`；
- `supplied_evidence_ids[]`；
- `constraints[]`和`desired_outputs[]`；
- `risk_context`和`clarification_question`。

Intent和Domain是多值标签，只用于Playbook/Tool检索、UI和审计，不决定是否继续。

## 6. Investigation Plan

`DBA_INVESTIGATION_PLAN.v1`包含`revision_no`、`objectives`、`hypotheses`、`evidence_needs`、
`planned_actions`、`playbook_candidates`、`available_capabilities`、`known_gaps`、`budget`和
`stop_conditions`。

Hypothesis包含ID、陈述、初始置信度和支持/反证需要。Plan Action包含ID、类型、Tool或
Playbook、输入、依赖、Evidence Need、优先级和原因。

模型输出后由Plan Validator确定性检查：

- Tool和Playbook存在且版本有效；
- 参数符合Schema；
- Target、Source和凭据在Agent授权范围；
- 依赖图无环；
- 调用数、成本、超时和总预算；
- 动态查询进入额外安全校验；
- 变更动作没有进入诊断计划。

模型可以选择零个Playbook，但至少要使用已有Evidence、调用Tool、提出补证或回答之一。

## 7. Collect–Assess–Replan

Coordinator按依赖图创建Tool Task。无依赖动作可并行，外部调用不持有数据库事务。每个结果
写Artifact、Tool Invocation和Turn Evidence。

`DBA_INVESTIGATION_ASSESSMENT.v1`保存假设支持/反证和置信度、Evidence Need状态、证据
矛盾、本轮新增信息量、是否可回答、是否Replan、是否需要用户以及建议下一步。

仍有授权Tool可区分主要假设且未超预算时Replan；所需证据不在任何授权Tool范围时才等待用户；
证据足够或预算已到但仍可给出有边界的结论时回答。无进展按新Evidence、假设变化和Need状态
判断，不按Tool是否返回HTTP 200判断。

## 8. Tool与Playbook

Tool Manifest包含ID、版本、描述、输入输出Schema、Target/Source能力、数据库版本、
Safety Profile、执行限制和Evidence Kind。

首批Namespace：

- `evidence.*`：日志、ORA、SQL结果、命令输出、配置和时间线；
- `monitor.prometheus.*`、`monitor.loki.*`、`monitor.alertmanager.*`；
- `db.oracle.*`：实例、会话、锁、SQL、等待、内存、存储、归档和参数；
- `host.*`：CPU、内存、磁盘、网络、重启、OOM和服务；
- `knowledge.*`：KC运行手册、历史案例和产品文档；
- `analysis.*`：时间关联、单位归一、差异比较和趋势。

Tool Invocation是语义审计对象，Ops Task继续负责租约和执行状态。Tool不要求隶属于Playbook。

现有Skill Manifest转为`DBA_PLAYBOOK_MANIFEST.v1`，保存适用问题描述、标签、推荐Evidence、
推荐Tool、常见假设、反证规则、停止条件和风险提示。Playbook通过语义召回产生候选，不使用
Subject精确相等作为准入；没有候选时使用通用调查提示和Tool Registry。

## 9. 受控动态查询

`db.oracle.readonly_query`执行前必须解析Oracle SQL AST，仅接受单条`SELECT`或只读`WITH`，
禁止DDL、DML、PL/SQL、DB Link、外部过程和副作用函数；校验Schema、对象、列和诊断用户
权限；强制行数、耗时、返回字节和并发限制；绑定参数；保存SQL Hash、策略Hash、参数和结果；
当前策略直接拒绝凭据、绑定值、SQL正文等高风险源列，后续只有在具备确定性列级分类时才允许
服务端脱敏后返回。

动态SQL只能使用诊断凭据，不能复用变更执行器。PromQL和LogQL使用相同的解析、范围注入、
标签限制和预算策略。

PromQL使用语法AST检查所有Vector Selector；数据库指标必须精确包含
`instance="${external_target}"`，主机指标必须精确包含
`target_key="${host_target}"`。禁止`@`与`offset`绕开调查窗口，并限制Range Selector、查询
窗口、Step、采样点和Series。LogQL不接受模型自定义标签Selector，只允许
`${binding_selector}`引用已冻结的Binding精确标签，后面附加有限个字面量包含或排除过滤。

Oracle动态SQL的解析实现固定使用SQLGlot Oracle方言，并在规划端和执行端消费同一份
`ORACLE_DYNAMIC_QUERY_POLICY.v1`。解析成功不视为可执行：还必须验证根节点、对象族、Schema、
投影列、函数、bind、Database Link和锁语义，注入服务端行数上限并绑定Query/Policy Hash。
执行端仍必须开启只读事务、使用诊断凭据并强制超时、行数、列数、字节和单元格长度限制。

## 10. Evidence模型

Evidence来源为`USER_PROVIDED`、`TOOL_OBSERVED`、`SOURCE_OBSERVED`、`KNOWLEDGE_CONTEXT`和
`MODEL_INFERENCE`。只有前三类可直接支持实时事实；Knowledge用于解释，Model Inference
必须引用依据。

每个Evidence保存来源Artifact、Tool Invocation、Evidence Kind、观测时间或时间窗、测量
口径、新鲜度、使用原因、信任等级及支持/反证/上下文角色。Answer Citation只能指向当前
Turn显式关联的Evidence；历史证据需重新建立Link。

## 11. Oracle Schema 14

### 11.1 Turn

删除控制型列`PRIMARY_INTENT`、`PRIMARY_DOMAIN`、`SUBJECT`、`INTENT_PLAN_*`和
`SKILL_PLAN_*`。新增：

- `INPUT_ANALYSIS_ARTIFACT_ID`；
- `TASK_FRAME_ARTIFACT_ID`；
- `CURRENT_PLAN_ARTIFACT_ID`；
- `ASSESSMENT_ARTIFACT_ID`；
- `CURRENT_PLAN_REVISION`；
- `INVESTIGATION_ROUND`；
- `TOOL_CALL_COUNT`；
- `NO_PROGRESS_COUNT`。

### 11.2 新表

`KBOT_OPS_TURN_INPUT_ITEM`：Turn、Message、顺序、内容类型、MIME、Raw Artifact、Analysis
Artifact、Evidence Kind、状态、安全级别和时间戳。

`KBOT_OPS_INVESTIGATION_REVISION`：Turn、Revision No、触发原因、Task Frame Artifact、Plan
Artifact、Assessment Artifact、状态和时间戳。

`KBOT_OPS_TOOL_INVOCATION`：Turn、Revision、Run、Task、Action、Tool、输入输出Artifact、
Policy Hash、状态、重试、错误和时间戳；`(REVISION_ID, ACTION_ID)`和
`(REVISION_ID, ORDINAL)`唯一，允许同一Turn在后续Revision重新编号。

### 11.3 现有表调整

- `KBOT_OPS_SKILL_INVOCATION`重命名为`KBOT_OPS_PLAYBOOK_INVOCATION`，并通过
  `REVISION_ID`归属一次调查计划修订；
- Turn Evidence增加`TOOL_INVOCATION_ID`、`SOURCE_KIND`、`EVIDENCE_KIND`、`CONFIDENCE`和
  `EXTRACTION_ARTIFACT_ID`；
- Turn Event增加可选Tool Invocation和Playbook Invocation关联；
- Message Payload升级为内容项摘要，大对象进入Artifact。

目标为43张表、10个视图，Manifest版本14、契约`aiops-oracle-v4`。规范DDL、自包含重建文件、
Entity、Repository、UoW、Manifest、视图和验收脚本必须同步更新。

## 12. 事务与Outbox

```text
aiops.turn.created
→ aiops.turn.understanding_requested
→ aiops.turn.framing_requested
→ aiops.turn.planning_requested
→ aiops.tool.execution_requested
→ aiops.turn.assessment_requested
→ aiops.turn.replanning_requested | answer_requested | evidence_requested
```

每阶段使用短事务冻结上下文，事务外调用模型或工具，新事务验证并写Artifact、投影、Event和
下一条Outbox。Repository不得提交事务。模型重试使用稳定幂等键，Plan Revision和Tool
Invocation具有业务唯一键。

## 13. API与SSE

公开Main API保持`/api/v1/apps/aiops`前缀，但Conversation和Turn采用新内容契约，不保留旧
`message`兼容字段。文件先上传得到Artifact引用，再提交Turn。

新增事件：`input.analysis.started/completed`、`task.frame.completed`、
`investigation.planned/replanned`、`tool.started/completed/gap`、`evidence.added`、
`assessment.completed`、`thinking.delta`、`answer.delta/completed`、`turn.status`和`done`。
事件只展示简洁进度，不暴露模型Chain of Thought、Secret、完整DSN或未脱敏结果。

## 14. 服务代码目标结构

```text
application/investigation/
  intake.py
  input_understanding.py
  task_framing.py
  planning.py
  coordination.py
  assessment.py
  replanning.py
  answer_composition.py

tools/
  registry.py
  validation.py
  evidence/
  monitoring/
  oracle/
  host/
  knowledge/
  analysis/

playbooks/
  registry.py
  catalog/oracle/
```

删除现役`skills/router.py`、`skills/planner.py`和`skills/execution.py`；所需Manifest加载和Hash
逻辑迁入Playbook。现有`turn_planning.py`拆分后删除，不保留旧Planner入口。

## 15. 安全边界

- 用户材料、日志和工具输出使用数据通道，与系统提示隔离；
- Secret只由执行器解析，不能进入模型、Artifact正文、日志或SSE；
- Tool参数通过Schema和授权范围校验；
- 动态查询经过AST和策略校验；
- 模型不能直接写数据库、调用Shell或执行变更；
- 变更继续使用独立Action、Policy、审批令牌和执行凭据；
- Answer事实引用Evidence，推断标明不确定性。

聊天中的变更建议不得从回答Markdown反向解析。Task Frame明确`requires_change=true`后，系统在
Sufficiency Assessment之后运行独立的Action Plan编译器，只接受当前Turn中
`SOURCE_VERIFIED`且字段完整的数据库事实，把参数绑定到已发布Action Template。用户粘贴结果、
模型推断、监控推断、目录外动作或缺少参数时均输出`NO_ACTION`。只有Agent策略允许执行、Target
启用且可连接、执行凭据存在、全局Mutation开关开启时才生成`PENDING_APPROVAL`；批准时再次按
Proposal Hash、Target版本、策略和模板Hash复核，模型始终不能直接调用变更执行器。

## 16. 验收矩阵

- Schema 14、43张表、Manifest、Entity和自包含重建文件一致；
- 输入、计划Revision、Tool调用和Evidence全部可审计；
- 并发Turn和重试不产生重复业务记录；
- 粘贴Alert Log形成用户证据并完成诊断；
- 无专用Playbook仍进入通用调查；
- 多目标问题不被单一Intent截断；
- Target离线时继续使用Prometheus、Loki和用户证据；
- Tool失败触发Gap、替代来源或Replan；
- 动态SQL安全围栏和跨Domain授权测试通过；
- 前端支持多内容输入、新SSE进度、统一Evidence和断线恢复。
