# 4.0 AIOps 实施步骤

## 实施目标

本计划将 [19](19_aiops_domain_model_workflow_and_executor.md) 至 [27](27_aiops_api_and_contracts.md) 的设计落成一个全新的 `aiops_agent` 领域服务。实现不修改或复用 3.x Ops Controller、Agent、Skill、明文凭据和请求内编排；旧代码只作为监控 Provider、Oracle/MySQL SQL 和评测样本来源。

实施按风险递增：先完成只观测闭环，再加入只读诊断、人工补充、Advisory，最后开放逐命令审批执行。数据库仍使用同一 Oracle/APEX Schema，但代码、表前缀、Session Factory、UoW 和进程入口均由 AIOps 独立拥有。

## 依赖顺序

```text
契约与骨架
    ↓
DDL → Entity/Repository/UoW → 配置 API
    ↓
Run/Task/Artifact/Event 内核
    ↓
Monitor Intake/Observe ──→ 只读 DB Diagnostic
    ↓                           ↓
     Evidence/Hypothesis/RCA 编排
                    ↓
        Chat HITL → Advisory → Approval/Execution
                    ↓
          Scheduler/Inspection/Report
                    ↓
       Main API/Root/APEX 集成与统一验收
```

后续步骤不得绕过前置层直接把逻辑写入 Prompt、Controller 或定时任务。

## 步骤 0：冻结契约与目录骨架

**状态：已完成（2026-07-23）。**

详细设计见 [29_aiops_step0_contracts_and_bootstrap.md](29_aiops_step0_contracts_and_bootstrap.md)。

新增以下空包和 App 入口，但此阶段不实现业务：

```text
aiops_agent/{api,application,domain,orchestration,diagnostics,
             ports,adapters,entities,repositories,persistence,
             workers,contracts,tests}/
apps/aiops_api/
apps/aiops_worker/
apps/aiops_scheduler/
apps/aiops_db_executor/
database/oracle/aiops_agent/
platform_core/contracts/aiops/
platform_clients/aiops.py
```

先固化 Public、Internal、Executor、SSE 和 Error DTO；枚举和状态迁移由 `aiops_agent/domain` 定义，HTTP Schema 与 Entity 只能映射，不能复制业务规则。增加 import 架构检查，禁止 `aiops_agent` 引用 `legacy`、旧 `agent/services/skills`、KC Repository 或模型 Entity。

**完成物：** 四个可独立启动的最小 App、三份独立 OpenAPI、配置样例、依赖规则和契约版本说明。

实施结果：Public/Internal/Executor/Event DTO 已冻结为严格 Pydantic 契约；
AIOps 使用独立的短期 Service Identity JWT 与 AuthContext JWT，不复用静态
Service Token；Management/Delegation Client 权限面已分离；API、Worker、
Scheduler 和 DB Executor 均可独立启动并只暴露系统探针。DB Executor 不创建
KBot Schema 连接，另外三个进程在步骤 1 Schema 尚未部署时保持 Live 但返回
Not Ready。

## 步骤 1：Oracle DDL 与 APEX 投影

**状态：已完成（2026-07-23）。**

详细 Oracle 契约见 [30_aiops_step1_oracle_schema.md](30_aiops_step1_oracle_schema.md)。

按固定顺序实现：

```text
001_ops_roots.sql
002_ops_runtime.sql
003_ops_change.sql
004_ops_inspection.sql
005_ops_messaging.sql
006_ops_fks_views.sql
```

DDL 必须包含 UUIDv7 `RAW(16)` 主键、JSON Check、外键、函数唯一索引、领取索引和延后 Artifact 外键。同步创建 `KBOT_V_OPS_*` 只读视图，但不授予 APEX 对状态机表的 DML 权限。脚本只用于创建空的 4.0 Schema，App 启动时不得建表或自动修复 Schema。

4.0 不迁移旧 Run、Chat、审批、Target 或 Monitor 配置。Target、Monitor 和 `SECRET_REF` 全部通过 4.0 配置 API 重新创建，旧密码和旧标识不得复制。

**完成物：** 可在空 Schema 顺序执行的六段规范建库脚本、Schema manifest 和受控视图。

实施结果：21 张 `KBOT_OPS_*` 表和 10 个 `KBOT_V_OPS_*` 视图已加入统一空库
初始化器；Manifest 固定脚本 Hash、对象清单、5 个延后 Artifact 外键和 5 个
函数唯一索引。Oracle 26ai 的 `MODE` 保留字和带时区时间唯一键限制已分别通过
`EXECUTION_KIND` 与 `SCHEDULED_FOR_UTC` 虚拟列显式适配，完整空库重放及
事务 Smoke 已通过。下一步进入步骤 2 Persistence 与事务内核。

## 步骤 2：Persistence 与事务内核

资源 ID、Entity、Repository、UoW 和并发事务的详细设计见 [31_aiops_step2_persistence_and_identity.md](31_aiops_step2_persistence_and_identity.md)。

按聚合实现 Entity、Repository 和 `AIOpsUnitOfWork`：

1. Target、Monitor Source、Policy、Inspection Plan；
2. Event、Alert、Run、Task、Artifact、Run Event；
3. Proposal、HITL、Approval Token、Execution；
4. Inbox、Outbox 和 Report。

Repository 接受 Session，不创建 Session、不提交事务、不调用外部服务。Application Service 通过 UoW 完成条件更新和乐观锁；租约领取使用数据库原子更新。实现 Outbox Dispatcher、Inbox 去重器和 Run Event 序号分配器，禁止 Python 全局计数。

**完成物：** 完整字段映射、聚合查询、UoW Factory、事务命令和并发/幂等测试夹具。

实施结果：21 张表的 Entity、九个聚合 Repository、显式单次提交 UoW 和
API/Worker/Scheduler 注入均已完成。Oracle Catalog 逐列校验通过；服务端游标
解决了 `FETCH FIRST ... FOR UPDATE` 的 `ORA-02014` 与驱动预取扩大锁范围问题。
真实双 Worker Smoke 已验证 Task/Outbox `SKIP LOCKED`、租约栅栏、自动回滚和
Run Event 连续序列。下一步进入步骤 3 配置与权限 API。

## 步骤 3：配置与权限 API

详细设计见 [32_aiops_step3_configuration_and_authorization_api.md](32_aiops_step3_configuration_and_authorization_api.md)。

先实现 Target、Agent Binding、Monitor Source/Binding、Policy 和 Inspection Plan 管理，不创建诊断 Run。Main API 通过 `AIOpsManagementClient` 映射 `/api/v1/ops/*` 到内部契约。

本步骤完成 AuthContext 求交、ETag、Cursor、SecretRef 校验、Webhook Key 轮换、健康检查 Command 和不可硬删除规则。Target 与 Agent 属于同 Domain 的验证通过 Agent Runtime Client 完成，不直接查 Agent 表。

**完成物：** 配置 CRUD/停用 API、APEX 列表投影、权限审计和 Secret Provider Adapter。

实施结果：六类配置资源已完成 Main API → AIOpsManagementClient → AIOps
Internal API 闭环；Domain/Actor 只取可信 AuthContext，ETag、父 Plan 并发边界、
签名 Cursor、Inbox 幂等、Outbox 配置事件、SecretRef 元数据校验、Webhook
Key 安全轮换和异步健康检查版本栅栏均已实现。真实 Oracle 全配置 Smoke 与
OpenAPI 快照通过。下一步进入步骤 4 确定性 Run 内核。

## 步骤 4：确定性 Run 内核

详细设计见 [33_aiops_step4_deterministic_run_kernel.md](33_aiops_step4_deterministic_run_kernel.md)。

在不调用 LLM、监控源和数据库的前提下实现：

- Chat/API/Alert/Schedule 的类型化 `CreateOpsRunCommand`；
- Run/Task 状态机、依赖图、租约、重试、截止时间和协作式取消；
- Artifact 不可变写入、Run Event/SSE、`Last-Event-ID` 恢复；
- Task Worker、Outbox Dispatcher、超时/租约 Reconciler；
- 崩溃后只恢复未完成 Task，不重复生成 Artifact。

先使用确定性测试 Handler 让 Run 从 `CREATED` 到 `COMPLETED/FAILED`。没有通过状态机和恢复验证前，不接入 Planner 或 LLM。

**完成物：** AIOps API/Worker 可多副本运行的持久化执行内核。

实施结果：固定 `kernel.observe-report@1` Blueprint、版本化 Handler Registry、
Run/Task 单点状态机、`Run → Task` 锁序、Lease Token fencing、确定性 Retry、
Deadline/取消 Reconciler、不可变 Artifact、严格序号 Event、Outbox Dispatcher、
Internal API、Public Run/SSE 和 Worker 多循环均已完成。Oracle Session 与所有
DDL 默认时间已统一为 UTC；真实 Oracle Smoke 已覆盖完整闭环、并发 Claim、取消
收敛、过期接管、旧 Token 拒绝和并发幂等创建。下一步进入步骤 5
监控接入与只观测闭环。

## 步骤 5：监控接入与只观测闭环

详细设计见 [34_aiops_step5_monitoring_observe_loop.md](34_aiops_step5_monitoring_observe_loop.md)。

实现 `MonitorPort`、Provider Registry 和 Prometheus/Zabbix/OEM Adapter，将 Provider 数据标准化为 Metric、Alert 和 Availability Artifact。Webhook 流程固定为：请求限流与 Source 解析 → 对原始字节验签 → 经验证请求入 Inbox → 精确映射 Target → Event 去重 → Alert 聚合 → Policy 判断是否创建 Run。

本阶段只执行 `SCOPE → OBSERVE → REPORT`，不查询目标数据库、不生成解决命令。Monitor Source 部分不可用时仍保存来源状态、缺失证据和 `PARTIAL` 报告。

**完成物：** Chat 查询监控信息、Critical Alert 自动生成只观测 Run，以及来源可回溯的初步事件报告。

## 步骤 6：只读数据库诊断目录

详细设计见 [35_aiops_step6_readonly_database_diagnostics.md](35_aiops_step6_readonly_database_diagnostics.md)。

实现 Oracle/MySQL `DatabaseDialect`、版本检测、Diagnostic Tool Registry、参数 Schema、SQL 模板、AST/Token 安全校验和结果规范化。Executor 只接收已解析的 `tool_id + version + parameters`，不能接收任意 SQL。

本步骤先使用确定性 Blueprint，不调用 LLM；步骤 7 的 LLM 只能建议 `tool_id + parameters`，Catalog Snapshot 决定版本。DB Executor 首先仅启用 `READ_ONLY_DIAGNOSTIC` 能力：独立 Service Identity、短期签名 Grant、只读 Secret、超时、行数/字节/并发限制、脱敏和结果 Hash。只读结果直接写成 Task Artifact，不写 `KBOT_OPS_EXECUTION`。连接失败转换为 Evidence Gap，不触发变更凭据回退。

**完成物：** Oracle/MySQL 首批确定性诊断工具、离线模板验证器和受控只读执行链路。

## 步骤 7：诊断编排与 LLM 接入

详细设计见 [36_aiops_step7_diagnosis_orchestration_and_llm.md](36_aiops_step7_diagnosis_orchestration_and_llm.md)。

按固定状态机实现 Scope、Observation、Hypothesis、Evidence Request、Evidence Sufficiency、Root Cause 和 Solution Draft。LLM 只输出版本化结构 DTO；Plan Validator 校验 Task 类型、预算、Target 能力和允许的 Tool。监控结果、数据库结果、KC/SOP Citation 与模型推断分别保存 Trust Level 和 Provenance。

根因只能输出 `CONFIRMED/PROBABLE/POSSIBLE/INCONCLUSIVE`。确定性 `RootCauseGradePolicy` 计算等级上限，LLM 只能维持或降低，不能靠语言置信度提升。只有前两类可生成定向 Solution Draft 和候选 Action Template Ref；本步骤尚不创建 Proposal。

**完成物：** Monitor + Read-only DB + SOP 的可恢复诊断链路，以及 Incident/Performance 结构化报告草稿。

## 步骤 8：Chat 人工诊断循环

详细设计见 [37_aiops_step8_chat_manual_diagnosis_hitl.md](37_aiops_step8_chat_manual_diagnosis_hitl.md)。

仅为 `TRIGGER_TYPE=CHAT` 实现 `DATA_REQUIRED` 和 `MANUAL_DIAGNOSTIC_SQL` HITL。系统优先使用预置模板；模板不足时，LLM 可生成只展示给用户的只读 SQL，经静态校验后进入不可变请求 Artifact，绝不发送给 Executor。

实现 `WAITING_INPUT`、受限上传、结果 Schema 校验、不可信内容隔离、同一 HITL 幂等回复、超时/跳过，以及在同一 Run 中创建后继 Task 恢复诊断。Alert/Schedule/API 自动 Run 即使证据不足也只能生成 `PARTIAL/INCONCLUSIVE`，不得进入该循环。

**完成物：** 数据库不可连接时可持续多轮补证且可在进程重启后恢复的 Chat Run。

## 步骤 9：Advisory 与受控变更

详细设计见 [38_aiops_step9_advisory_approval_and_execution.md](38_aiops_step9_advisory_approval_and_execution.md)。

先落地 `ADVISORY`：版本化 Action Template、影响/风险/前置条件/回滚/验证计划、人工处理结果和处理前后 Comparison。确认无自动执行路径后，再启用 `AGENT_EXECUTE`：

1. Policy Gate 生成不可变 Decision；
2. 每条命令创建独立 Proposal 和 `CHANGE_APPROVAL`；
3. 一次显式审批签发短期一次性 Token；
4. Executor 再次验证 Target/Policy/Template/Parameter Hash；
5. 变更串行执行，结果回调经 Inbox 对账；
6. Verify/Compare 决定实际效果，回滚始终创建新 Proposal 并重新审批。

Mutation 默认通过部署级 Kill Switch 禁用；它只能降低能力，不能绕过 Target 的 `ADVISORY` 配置。首期不实现 Shell、OEM Job 或 Zabbix Remote Command 自动执行。

**完成物：** Advisory 完整闭环，以及可独立开关、逐命令审批、不可重放的 Oracle/MySQL 变更执行链路。

## 步骤 10：巡检、报告与对比

详细设计见 [39_aiops_step10_inspection_reporting_and_comparison.md](39_aiops_step10_inspection_reporting_and_comparison.md)。

实现 Scheduler 租约、Daily/Weekly/Cron Plan、幂等 Inspection Fire、Target 展开、Misfire/重叠策略和终态对账。巡检复用同一 Observe/Diagnose/Report Task，不复制 Agent 流程。报告内容先形成不可变 Artifact，再通过 `REPORT_KEY + REPORT_VERSION + IS_CURRENT` 发布 `KBOT_OPS_REPORT` 投影供 APEX 查询。

Incident/Performance 处理前冻结 Comparison Plan 和基线，处理后使用相同定义采集主指标与护栏指标；结论限定为 `IMPROVED/UNCHANGED/DEGRADED/INCONCLUSIVE`。Email/IM 只实现 `ReportDeliveryPort` 空接口，不实现发送 Adapter。

**完成物：** 日报、周报、故障/性能报告和处理前后对比报告的持久化及 APEX 展示数据。

## 步骤 11：Root Agent、Main API 与前端集成

详细设计见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

- Main API 发布完整 `/api/v1/ops/*` 和 Monitoring Integration API；
- Agent Runtime 使用 `KBOT_AGENT_DELEGATION`、Child Event Cursor 和有限租约管理跨服务子 Run；
- Root Agent 通过窄 `AIOpsDelegationClient` 创建带 `PARENT_AGENT_RUN_ID/PARENT_DELEGATION_ID` 的委派 Run；
- Root 只投影必要进度、交互资源和受限 Result Envelope，不接管 Ops Task 或转发子 SSE；
- Response Composer（原 Conversation Composer）使用 AIOps 最终 Artifact 生成用户表达，但不能改变根因等级、命令、风险或审批状态；
- APEX 通过只读视图展示 Target、Run、待审和 Report，所有写操作仍调用 API。

上线前直接将调用方切换到 v4，不做 3.x/4.0 双写、切流或兼容 Adapter。

**完成物：** Direct AIOps Chat、Root 委派、Alert 和 Schedule 四类入口的统一用户体验。

## 步骤 12：统一验收与发布

完整方案见 [41_kbot4_step12_acceptance_release_and_cutover.md](41_kbot4_step12_acceptance_release_and_cutover.md)。

完整测试按 4.0 总计划统一执行，但每个前置步骤必须保留最小安全回归，不能将状态机、Policy 或 Executor 的错误留到最终阶段才发现。最终至少覆盖：

- Oracle 全量建库脚本、约束、UoW 回滚、租约接管和 Outbox/Inbox 重放；
- Domain/Target 权限、Webhook 伪造、SSE 恢复和跨 Domain 隐藏；
- Oracle/MySQL 诊断目录版本矩阵、只读验证、超时和脱敏；
- LLM 输出污染、Prompt Injection、证据不足和错误根因降级；
- HITL 重复回复、审批竞争、Token 过期/重放和 Executor 乱序回调；
- Scheduler 多副本、告警风暴、Provider 故障、报告/Comparison 正确性；
- API/Worker/Scheduler/Executor 独立重启、扩缩、降级和 Kill Switch。

DDL 先部署，随后暗部署 API/Worker、Monitor Adapter、只读 Executor、Scheduler 和上层 Runtime；正式入口一次切换，Mutation 最后通过独立 Gate 启用。失败时优先关闭入口或能力开关并保留已写事实；不回滚已执行 DDL、不删除 Run/Artifact、不用旧系统接管同一 AIOps Run。

## 建议提交边界

每个步骤至少一个独立提交，规范建库脚本与对应 Entity 放在同一提交或连续不可分割提交中。建议 Conventional Commit Scope：

```text
feat(aiops-contracts): ...
feat(aiops-db): ...
feat(aiops-runtime): ...
feat(aiops-monitoring): ...
feat(aiops-diagnostics): ...
feat(aiops-hitl): ...
feat(aiops-executor): ...
feat(aiops-reporting): ...
```

任何提交都不能同时搬迁无关 Legacy 代码或修改 KC/Model Serving 的领域表。每步完成后更新 Schema manifest、OpenAPI 快照、配置示例和对应设计文档，避免实现与设计分叉。

## 完成定义

AIOps 完成不是“Agent 能回答运维问题”，而是四类入口都创建可恢复 Ops Run；所有事实都有 Provenance；所有只读 SQL 来自受控目录或由用户自行执行；所有变更逐命令审批、可审计、不可重放；报告可由 APEX 查询；四个进程可独立部署；未来拆库时不需要修改 Domain、Application 或 API 契约。
