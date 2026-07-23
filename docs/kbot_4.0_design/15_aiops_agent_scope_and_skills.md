# 4.0 AIOps Agent 边界与 Skill

## AIOps Agent 是什么

AIOps Agent 是独立部署的运维领域 Agent，负责把监控事件、数据库指标、日志、CMDB 资产和运维 SOP 组合成可验证的诊断或变更流程。它拥有自己的 Ops 表、领域状态机和流程编排，不是通用聊天 Agent；它不拥有目标数据库凭据，也不直接执行任意 SQL。

第一版支持 Oracle/MySQL、Prometheus/Zabbix/OEM、Chat/告警/定时巡检触发，并生成故障、性能、日巡检、周巡检和处理前后对比报告。IM/Email 仅保留端口，不在本版实现。详见 [20_aiops_monitoring_inspection_and_reporting.md](20_aiops_monitoring_inspection_and_reporting.md)。

它处理的典型请求包括：

- “当前数据库 CPU 为什么持续升高？”
- “这个告警影响哪些实例和业务？”
- “是否可以清理某类临时对象？”
- “执行变更后是否恢复正常？”

AIOps Agent 可以读取受授权的运维数据，也可以通过 Knowledge Core 检索 SOP、历史案例和操作依据；但最终答案必须区分监控事实、知识依据、推断结论和执行结果。

## 与其他 Specialist 的区别

| Specialist | 数据和目标 | 典型输出 |
| --- | --- | --- |
| Knowledge | 文档、资产附件、SOP、案例知识 | `CITATION_PACK` |
| Data | 业务数据集、Excel 工件、受控 NL2SQL | `QUERY_RESULT` |
| AIOps Agent | 指标、日志、告警、CMDB、目标实例和变更窗口 | `DIAGNOSIS`、`APPROVAL_PROPOSAL`、`EXECUTION_REPORT` |
| Conversation | 无需领域工具的对话和澄清 | `ANSWER_DRAFT` |

“查询数据库销售数据”当前通过受控 MCP 问数工具处理，未来才考虑 Data Agent；“查询数据库连接数、锁等待或表空间”属于 AIOps Agent。两者即使访问同一 Oracle 实例，也必须使用不同的权限、数据源绑定和审计策略。

## AIOps Agent 内部模块

AIOps Agent 是独立服务，但内部仍拆分为事件接入、资产/目标管理、诊断编排、策略/HITL、执行适配和验证报告模块：

```text
Monitoring Adapter / Alert Source
          ↓
     AIOps Agent
  Event Intake / Target Binding
  Diagnosis Orchestration
  Change Proposal / HITL
          ↓
   Policy / HITL Gate
          ↓
 DB Executor / External Adapter
          ↓
   Verification + Execution Report
```

AIOps Agent 负责标准化 `OpsEvent`、告警聚合、资产绑定、领域 Run/Task、诊断编排和 HITL 状态；它可以复用 Agent Runtime 的租约、事件和 Artifact 基础协议，但拥有独立的 `KBOT_OPS_*` 表和状态机。DB Executor 负责目标数据库的连接、SQL 安全校验和实际执行。

## 独立领域数据模型

AIOps Agent 不使用 Knowledge Core 的 Bundle/Evidence 表，也不把运维流程塞进通用 Agent Run 表。建议由以下 Ops 聚合根组成：

| 聚合 | 典型表 | 作用 |
| --- | --- | --- |
| Target | `KBOT_OPS_TARGET` | 数据库实例、环境、可选诊断/执行凭据引用和 Agent 绑定 |
| Monitor | `KBOT_OPS_MONITOR_SOURCE`、`KBOT_OPS_TARGET_MONITOR` | 多监控源、外部目标映射、优先级和指标覆盖 |
| Event/Alert | `KBOT_OPS_EVENT`、`KBOT_OPS_ALERT` | 原始监控事件、去重、关联、抑制和告警生命周期 |
| Diagnostic Run | `KBOT_OPS_RUN`、`KBOT_OPS_TASK` | 诊断、补充数据、审批、执行和验证流程 |
| Artifact | `KBOT_OPS_ARTIFACT` | 不可变的观测、诊断、审批上下文、执行结果和验证报告 |
| Change | `KBOT_OPS_CHANGE_PROPOSAL`、`KBOT_OPS_EXECUTION` | 变更提案、执行令牌、前后状态和回滚结果 |
| Governance | `KBOT_OPS_POLICY`、`KBOT_OPS_HITL` | 操作策略、审批记录、超时和拒绝原因 |
| Inspection/Report | `KBOT_OPS_INSPECTION_PLAN`、`KBOT_OPS_REPORT` | 日报、周报、故障/性能报告和处理前后对比 |

每个 Ops 根对象都必须从 AuthContext 获得 domain 范围；子表通过外键继承范围。APEX 页面需要直连时，使用包含 `app_id`/`domain_id` 的受控视图，不让 AIOps Agent 接收客户端伪造的范围字段。

通用 `KBOT_AGENT_RUN` 记录用户请求和最终 Artifact；`KBOT_AGENT_DELEGATION` 保存 Root Task、`ops_run_id`、子事件游标和受限结果引用。AIOps Run 反向记录 `PARENT_AGENT_RUN_ID/PARENT_DELEGATION_ID`，但内部诊断状态、告警关联和执行报告仍由 `KBOT_OPS_*` 表负责。

## 与 Root Agent 的关系

Root Agent 不把 Ops Task 展开到自己的通用 Planner 中，而是通过版本化
`AIOpsDelegationClient` 委派：

```text
Root Run / Supervisor
        ↓ AIOpsDelegationClient
      AIOps Agent
        ↓
  Ops Run + Ops Task DAG
        ↓
  Ops Artifact / Execution Report
        ↓
Root Artifact（引用 ops_run_id）
```

AIOps Agent 可以复用通用 Agent Runtime 的身份、租约、事件和 Artifact 协议，但其领域流程由自身的 `Ops Orchestrator` 控制。Root Agent 只能看到诊断摘要、引用、交互/审批资源状态和执行报告，不能直接写 `KBOT_OPS_*` 表，也不能跳过 AIOps Agent 调用 DB Executor。父子事件、Result Envelope 和 Composer 边界见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

Ops 表字段、状态机、HITL、审批令牌、执行/验证/回滚和 DB Executor 请求契约详见 [19_aiops_domain_model_workflow_and_executor.md](19_aiops_domain_model_workflow_and_executor.md)。

从 Observation、候选假设、反证到根因级别的完整诊断内核见 [24_aiops_diagnosis_orchestration_and_evidence.md](24_aiops_diagnosis_orchestration_and_evidence.md)。
步骤 7 的可恢复诊断轮次、Evidence Index、模型契约和确定性等级上限见 [36_aiops_step7_diagnosis_orchestration_and_llm.md](36_aiops_step7_diagnosis_orchestration_and_llm.md)。
步骤 9 的 Action Catalog、Advisory、逐命令审批、Mutation Grant 与执行验证见 [38_aiops_step9_advisory_approval_and_execution.md](38_aiops_step9_advisory_approval_and_execution.md)。

## 最小 Skill 集合

### `ops-metric-query`（只读）

读取经过数据源绑定的 Prometheus、OEM、Zabbix 或其他指标，输出带时间窗口、实例、单位和采集来源的 `METRIC_OBSERVATION`。

### `ops-log-query`（只读）

按授权实例和时间窗口读取日志或事件，输出脱敏的 `LOG_OBSERVATION`。禁止根据用户文本拼接任意日志查询 URL。

### `ops-db-diagnose`（只读）

通过 DB Executor 的只读接口采集锁、会话、表空间、等待事件和执行计划等诊断数据，输出 `DIAGNOSIS_EVIDENCE`。LLM 只提交 `tool_id + parameters`，Oracle/MySQL `DatabaseDialect` 选择版本化预置 SQL。Skill 不接收数据库密码、不提交 SQL 文本，不直接创建数据库 Session。详见 [22_aiops_database_diagnostic_catalog.md](22_aiops_database_diagnostic_catalog.md)。

### `ops-sop-retrieval`（只读）

调用 Knowledge Core 检索相关 SOP 和历史案例，输出 `CITATION_PACK`，作为诊断依据，不把文档内容伪装成实时监控事实。

### `ops-diagnose`（只读）

整合 Metric、Log、DB 和 SOP Artifact，输出：问题范围、观察事实、候选原因、置信度、缺失数据、建议动作和引用来源。诊断不代表已经获得执行授权。

### `ops-action-plan`（无副作用）

根据诊断生成候选 `ACTION_PLAN`，包含目标资源、Action Template Ref、参数来源、影响范围、风险、前置条件、回滚、验证步骤和 Evidence 引用。确定性 Proposal Builder 再根据 Catalog、Policy 和 Hash 创建 `APPROVAL_PROPOSAL`；该 Skill 不能直接调用变更执行接口。

### `ops-change-execute`（高风险）

只有在 Policy Gate 通过、HITL 批准、Approval Authorization 未过期且目标绑定未改变时才能运行。Dispatcher 只提交 Execution ID，DB Executor 通过 Claim 获取受实例约束的 Mutation Grant；Skill 不接受自由 SQL 或自然语言命令。

### `ops-verify`（只读）

在变更后按 Proposal 中的验证步骤采集结果，输出 `VERIFICATION_REPORT`，明确成功、部分成功、失败和需要人工处理的状态。

### `ops-report`（只读）

基于不可变 Observation、Diagnosis、Execution 和 Verification Artifact 生成故障、性能、巡检或处理前后对比报告。报告不能改写原始指标和执行结果。

## 典型状态流程

```text
OpsEvent / User Request
        ↓
Scope + Target Authorization
        ↓
Metric/Log/DB Observation（可并行）
        ↓
Ops Diagnose
        ├─ Chat + 信息不足 + DB 不可连
        │       → 生成只读诊断 SQL → WAITING_INPUT
        │       → 用户回贴结果 → Ops Diagnose（可多轮）
        ├─ Alert/Schedule + 信息不足
        │       → PARTIAL/INCONCLUSIVE Report → COMPLETED
        ├─ 只读回答 → Grounded Answer
        └─ 需要变更 → ActionPlan → ChangeProposal
                              ↓
                         Policy / HITL
                              ↓
                      DB Executor / Adapter
                              ↓
                         Ops Verify
                              ↓
              Verification + Comparison Report
                              ↓
                     Execution/Inspection Report
```

诊断阶段可以自动重试和并行采集；变更阶段必须串行、可取消、带租约和幂等键。执行失败不能自动扩大权限或修改原始 Proposal，必要时生成新的 Proposal。

## 安全边界

- AIOps Agent 只能访问 Agent 绑定的运维实例、指标源和环境；
- DB Executor 通过 Claim/Grant 再次校验目标、Template/Command Hash、环境和变更窗口；
- 只读诊断和变更执行使用不同的 Service Identity；
- 密码和 Token 不进入 Artifact、Prompt 或日志；变更 SQL 仅保存在受控 Proposal，人工诊断 SQL 仅保存在受控 Request Artifact，二者都不写普通日志；
- 所有观察、决策、审批、执行和验证都关联 `run_id`、`task_id`、`target_id` 和 `trace_id`；
- 用户要求“直接执行”不能绕过 Proposal、Policy 和 HITL。

## 第一版非目标

第一版不实现自主无限循环、自主扩权、自主发现未绑定资产、IM/Email Adapter，或基于自然语言生成并执行 SQL。可执行动作仅限已登记的 Oracle/MySQL SQL 模板，每条命令单独审批。Shell、OEM Job 和 Zabbix Remote Command 只作为人工建议。
