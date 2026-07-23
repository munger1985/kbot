# 4.0 AIOps 服务包结构与运行时

## 物理边界

`aiops_agent` 是一个独立领域服务包，内部通过多个进程角色分离 API、编排、定时调度和数据库执行。这些进程属于同一 AIOps 业务边界，不是四个互相共享内存的 Agent。

```text
APEX / User / Monitoring Webhook ──→ Main API / BFF ──┐
Root Agent ────────────────────────────────────────────┤
                                                      ▼
                                                 AIOps API
                                                      ↓ durable Run/Task/Event
                                                 AIOps Worker
                                  ├─ Monitor Adapters
                                  ├─ Knowledge Core / Model Serving Clients
                                  └─ DB Executor Client ──→ AIOps DB Executor

AIOps Scheduler ──→ due Inspection Plan ──→ Ops Run
```

API、Worker 和 Scheduler 共享 `KBOT_OPS_*` 领域表；DB Executor 不编排 Run，只解析已登记模板、获取短期凭据并返回执行结果。

## 推荐代码布局

```text
aiops_agent/
  api/
    management/              # Main API 代理的 Target、Run、HITL、Approval、Report
    intake/                  # Root Agent 委派、监控事件接入
    executor/                # Executor 回调与对账
    schemas/                 # HTTP 请求/响应，不是 Entity
  application/
    targets/                 # Target、Monitor Binding 用例
    intake/                  # Chat、Alert、Schedule 触发
    diagnosis/               # Scope、Evidence、Hypothesis、RCA
    changes/                 # Proposal、Approval、Execution、Verify
    inspections/             # Plan、Report、Comparison
    dto/
  domain/
    target/                  # Target/Monitor 聚合和不变式
    operations/              # Run/Task/Artifact 状态机
    diagnosis/               # Evidence、Hypothesis、RootCause
    change/                  # Proposal、Policy、Approval
    inspection/              # Schedule/Report 规则
  orchestration/
    planner.py               # 只产生类型化计划
    diagnosis_machine.py
    change_machine.py
    verification_machine.py
  diagnostics/
    contracts.py
    registry.py
    validator.py
    dialects/oracle/
    dialects/mysql/
  ports/
    monitor.py
    model.py
    knowledge.py
    secret_store.py
    db_executor.py
    artifact_store.py
    trigger.py
    report_delivery.py
  adapters/
    monitoring/prometheus.py
    monitoring/zabbix.py
    monitoring/oem.py
    model_serving.py
    knowledge_core.py
    secret_store.py
    db_executor_http.py
    object_store.py
  entities/                  # 仅 KBOT_OPS_* SQLAlchemy 模型
  repositories/              # 仅 KBOT_OPS_* 查询/持久化
  persistence/
    uow.py
  workers/
    task_worker.py
    outbox_dispatcher.py
    scheduler.py
    reconciliation.py
  contracts/                 # 领域内部版本化 Artifact schema
  tests/

apps/
  aiops_api/main.py
  aiops_worker/main.py
  aiops_scheduler/main.py
  aiops_db_executor/main.py
```

`api/schemas`、`application/dto`、`domain` 和 `entities` 不得互相混用。API 不返回 SQLAlchemy Entity；Domain 不 import FastAPI、SQLAlchemy、HTTP Client 或 `platform_core.database`。

## 进程职责

### AIOps API

- 验证 AuthContext、Service Identity、Webhook 签名和 DTO；
- 创建/查询 Target、Monitor Source/Binding、Ops Run、HITL、Proposal、Inspection Plan 和 Report；
- 将 Chat/Root 委派、Alert Webhook、审批、取消和用户回贴转换为持久化 Command；
- 从 Event 投影输出 SSE，不在请求进程中执行 LLM、监控查询或 SQL。

### AIOps Worker

- 领取 `READY/RETRYABLE` Task 并维护租约；
- 执行 Scope、Observe、Diagnose、Propose、Verify、Compare 和 Report 状态机；
- 调用 Monitor、KC、Model Serving 和 DB Executor Client；
- 在短事务中写入 Artifact、Task/Event 状态和后继 Task；
- 不保存可恢复状态到 Python 全局对象。

### AIOps Scheduler

- 使用数据库租约领取 due Inspection Plan；
- 原子创建 Ops Run/Initial Task/Outbox 并推进 `next_run_at`；
- 扫描 HITL/Run/Execution 超时和对账任务；
- 不自己采集指标或生成报告。

### AIOps DB Executor

- 只暴露内部版本化契约，不接收用户请求；
- 根据 `tool/action template id + version + parameters` 加载 SQL；
- 从 Secret Store 解析只读或变更凭据，不接受请求传入密码；
- 强制超时、行数、并发、幂等、审批 Token 和结果脱敏；
- 不持有 KBot Schema 凭据；Mutation 前通过 AIOps API 原子 Claim 并消费审批 Token；
- 不直接推进 Ops Run，结果以签名回调/查询契约返回 AIOps API/Worker。

DB Executor 是 AIOps 服务的高权限子进程，应使用独立 Service Identity、网络策略和日志资源。未来可以单独部署，但不需要为它创建第二套运维领域模型。

只读诊断使用由 AIOps Worker 签发、受 Task Lease 限制的短期 `DiagnosticExecutionGrant`；Grant 只携带受签名保护的连接 Profile 和 `SecretRef`，不携带明文凭据。Executor 必须以本地 Catalog 的模板版本与 Hash 为准，不接受 SQL 文本。完整设计见 [35_aiops_step6_readonly_database_diagnostics.md](35_aiops_step6_readonly_database_diagnostics.md)。

Mutation Dispatcher 只发送 Execution ID；Executor 通过绑定进程实例的 Claim 原子消费审批授权并获取 `MutationExecutionGrant`。每个 Execution 最多进行一次数据库命令投递，结果未知时只对账/验证，不自动重试。完整设计见 [38_aiops_step9_advisory_approval_and_execution.md](38_aiops_step9_advisory_approval_and_execution.md)。

## Repository 与 UoW

完整表字段、约束、索引、保留策略和 migration 分段见 [26_aiops_physical_data_model.md](26_aiops_physical_data_model.md)。

`AIOpsUnitOfWork` 只组合 AIOps Repository：

```text
targets, monitor_sources, target_monitors
events, alerts, runs, tasks, artifacts
proposals, hitl, executions, policies
inspection_plans, inspection_fires, reports
inbox, outbox
```

Route/Worker 从 UoW Factory 获取短生命周期 Session。Repository 不创建 Session、不 `commit/rollback`；Application Service 定义事务内命令，UoW 在边界统一提交。任何 HTTP、LLM、Monitor、KC、Secret Store 或目标数据库调用都在事务外。

诊断阶段通过窄 `AIOpsModelPort` 调用 Model Serving，并以严格 Schema 接收 Hypothesis、Evidence Request 和 Assessment；不直接复用旧客户端的自由文本 JSON 提取/修复作为状态迁移边界。模型调用、动态诊断轮次和证据等级的详细设计见 [36_aiops_step7_diagnosis_orchestration_and_llm.md](36_aiops_step7_diagnosis_orchestration_and_llm.md)。

## 关键事务

| 用例 | 同一事务中必须写入 |
| --- | --- |
| Chat/Root 创建 Run | Run（Root 含 Parent Delegation ID）、Initial Task、Event、Outbox |
| Webhook 接入 | Inbox、Event、Alert 去重/聚合、可选 Run/Task、Outbox |
| Task 完成 | Artifact、Task 状态、Run 版本、后继 Task、Event/Outbox |
| Chat 回贴 | HITL Response、User Result Artifact、结束等待 Task、创建后继 Task、Event |
| 批准命令 | HITL/Proposal 状态、Token Hash、Execution、Outbox |
| Executor 回调 | Inbox 去重、Execution、Result Artifact、Task/Event |
| Scheduler Tick | Plan Lease/`next_run_at`、Fire、每 Target Run/Initial Task、Event/Outbox |
| Fire 对账 | 子 Run 终态计数、Fire 状态、可选排队 Fire 展开 |
| Report 发布 | Content Artifact、Report Current 版本切换、Run/Event/Outbox |

事务提交后 Dispatcher 才发起跨服务调用。外部结果应用 `request_id + status_version` 进入 Inbox，不在远程回调中直接修改多个无关表。

## 对外契约与 Client

Main API 通过 `AIOpsManagementClient` 调用管理与用户资源；Agent Runtime 通过更窄的 `AIOpsDelegationClient` 创建/读取/取消自身子 Run。稳定跨服务 DTO 放入 `platform_core/contracts/aiops/`；AIOps 内部 Domain/Artifact schema 仍归 `aiops_agent/contracts/` 所有。未来拆分仓库时，前者发布为独立契约包。

详细 DTO、鉴权、幂等、SSE 和错误规范见 [27_aiops_api_and_contracts.md](27_aiops_api_and_contracts.md)。外部 API 由 Main API/BFF 发布，AIOps 进程只接受内部 Client 契约：

```text
/v4/ops/...                              # 浏览器/APEX 看到的 Main API
/v4/integrations/monitoring/...          # 监控系统看到的 Main API
/internal/v1/aiops/...                   # Main API/Agent Runtime 按 Service Scope 调用
/internal/v1/db-executor/executions      # AIOps Worker 调用 Executor
```

Root Agent 只获取 `ops_run_id`、安全事件页和终态 Result Envelope，不获取 AIOps Repository/UoW、内部 Artifact 或完整命令。`platform_clients/ops.py` 遗留兼容 Client 不迁移，4.0 分别建立 Management/Delegation Client。父子事件投影与 Composer 契约见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

## 配置和凭据

每个 App 使用独立配置段、日志 `service_name`、连接池和 Service Identity，但 4.0 可以指向同一 Oracle/APEX Schema。配置至少包含：

- API/Internal API 地址、鉴权、超时和 CORS；
- `system_aiops_agent_id`，供 Alert/Schedule 等无人触发流程选择固定受托 Agent；
- Worker 并发、租约、重试、诊断预算和 Artifact 限额；
- Scheduler 扫描间隔、时区、领租和对账频率；
- KC/Model Serving/DB Executor/Secret Store URL 和 Service Identity；
- 对象存储、监控 Provider 和报告渲染限额。

Monitor/Target 凭据是领域数据中的 `SECRET_REF`，不写入 `base.toml`。本地开发可使用环境变量 Secret Provider，服务器使用实际 Vault/Secret Manager Adapter。

## 部署、健康和扩缩

- API 按 HTTP/SSE 并发扩展；Worker 按 READY Task 数量扩展；
- Scheduler 可多副本部署，但同一 Plan 仅有一个有效租约；
- DB Executor 按只读/变更并发分配独立资源池，变更池不根据队列自动无上限扩展；
- `/live` 只判断进程，`/ready` 检查自有数据库/目录/必需 Client，`/health/dependencies` 提供详细状态；
- Monitor/KC/Model 部分失败可以使 Worker `ready`，但必须在 Run 中生成数据源不可用 Artifact；
- Executor 无法加载模板、Secret Store 不可用或审批验证失败时不得 `ready`。

开发环境可由一个启动脚本同时启动四个进程，但不在一个 Python 进程中使用全局对象模拟生产通信。

## 同 Schema 与未来拆分

4.0 中通过代码依赖检查、Repository 包边界、表前缀和 API 契约强制所有权；APEX 仅通过受控视图读取目标、待审、Run 和 Report 投影。

未来拆分时，将 `aiops_agent + apps/aiops_* + platform_core + platform_clients 必要契约 + configuration` 构建为独立镜像，替换 DB/Secret/Client 配置即可。不需要重写 Domain、Application、API 或 Repository 边界。步骤 0 的具体包、配置和启动约束见 [29_aiops_step0_contracts_and_bootstrap.md](29_aiops_step0_contracts_and_bootstrap.md)。

Scheduler、Inspection Fire、版本化 Report 和 Comparison 的具体事务及代码布局见 [39_aiops_step10_inspection_reporting_and_comparison.md](39_aiops_step10_inspection_reporting_and_comparison.md)。

## 迁移与验收

具体施工顺序、阶段完成物和启用次序见 [28_aiops_implementation_plan.md](28_aiops_implementation_plan.md)。

- `utils/monitor` 仅作为协议与响应 Fixture 来源；按新 MonitorPort 重写到 `aiops_agent/adapters/monitoring`，禁止原类直接迁移或继续读取全局配置；
- 3.x Ops Agent、Orchestrator、Planner、Entity/Repository、Scheduler 和 Skill 不被 4.0 import；
- API、Worker、Scheduler 和 Executor 可独立启动、关闭、重启和扩展；
- 任意进程崩溃后不丢 Run/HITL/Approval/Execution 状态；
- 架构测试阻止 `aiops_agent` 导入 KC/Agent Runtime Repository、旧 `services`、旧 `agent` 或旧 `skills`；
- DB Executor 没有外部用户路由，AIOps Worker 没有目标数据库明文凭据。
