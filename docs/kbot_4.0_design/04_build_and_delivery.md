# 4.0 建设、质量与交付计划

## Phase 0：边界冻结与代码清理

建立架构测试和依赖规则：`knowledge_core` 不可 import `agent`、`skills`、旧 `services/kb`；其他领域不可直接 import Knowledge Core Repository。先确认 4.0 消费者，再直接删除旧接口、兼容导出、V1 Entity/Repository 和未使用的 3.x Agent/Skill；不建立 `legacy/` 代码保留区，也不保留兼容 import。所有新增或修改的注释、Docstring 和日志正文使用中文，机器契约保持英文。暂不进行最终质量测试，测试统一安排在所有 4.0 能力完成后。

**退出条件：** 服务边界、Owner、表权限、API 命名、UoW/Outbox 模板和全量建库 DDL 流程经评审通过。

## Phase 1：平台基础与独立 App 边界

完善 `platform_core` 的配置、日志、认证、数据库连接和 Session Factory；每个 App 使用自己的配置实例和连接池，但 4.0 阶段仍指向同一 Schema。提供通用 `AsyncUnitOfWork`、领域 UoW factory、事务测试夹具和 Outbox dispatcher。所有 4.0 领域从第一行代码起使用它们。

**验收：** 一个跨多个 Repository 的写用例可验证原子提交和异常回滚；无新 Repository 调用 `commit()`。

## Phase 2：Knowledge Core 与解析检索闭环

以 3.5 已实现的 `knowledge_core`、独立进程入口、Parser/Projection Worker 和两阶段检索为代码基线；不新建平行实现。`KBOT_KC_*` 由 Knowledge Core 自有全量建库脚本定义。补齐 Bundle Ingestion、版本状态机、Job/Outbox、Discovery/Evidence、检索质量、权限、观测和失败恢复闭环。此阶段不把 Agent 职责放入 KC。

**验收：** 一个含 Manifest 和多附件的 Bundle 可创建、更新、删除、恢复；重复提交不产生重复 Version/Job。

## Phase 3：Main API 与领域客户端

按 [14_main_api_bff_and_auth_context.md](14_main_api_bff_and_auth_context.md) 建立新的 Main API/BFF、AuthContext、Portal/APEX/MCP/Slack 客户端契约和服务身份上下文。所有跨服务调用使用版本化 DTO、HTTP 或 durable job；3.x Parser 轮询协议和旧上传接口不进入 4.0。

**验收：** 多副本 Worker 不重复领取任务；崩溃后租约可恢复；同一 PDF 可保留多视图并稳定定位页码。

## Phase 4：Agent Runtime 与新 Skill Runtime

先按 [11_agent_execution_model.md](11_agent_execution_model.md)、[12_agent_runtime_api_and_state_transitions.md](12_agent_runtime_api_and_state_transitions.md) 和 [13_agent_planning_and_skill_contract.md](13_agent_planning_and_skill_contract.md) 落地 Run/Task/Artifact/ExecutionContext、状态机、租约、命令接口、事件持久化、Planner 和 Skill Manifest，再按 [16_document_agent_boundary_and_retrieval_contract.md](16_document_agent_boundary_and_retrieval_contract.md) 重构 Document Agent 检索契约，按 [17_root_agent_routing_and_composition.md](17_root_agent_routing_and_composition.md) 和 [18_agent_runtime_packaging_and_deployment.md](18_agent_runtime_packaging_and_deployment.md) 完成 Root 路由、Runtime API/Worker 和多来源组合，最后实现 Supervisor、Plan Validator、Policy Gate、预算、取消和恢复。旧 SkillRuntime、动态反射和 Prompt 授权规则不迁入 4.0。

**验收：** 建立人工标注集并达成 Bundle Recall@K、Document Recall@K、Evidence Recall@K、页码定位准确率和延迟目标；每个回答证据均可回链。

## Phase 5：身份、安全、运维与外部集成

按 [19_aiops_domain_model_workflow_and_executor.md](19_aiops_domain_model_workflow_and_executor.md) 至 [27_aiops_api_and_contracts.md](27_aiops_api_and_contracts.md) 的设计，并依照 [28_aiops_implementation_plan.md](28_aiops_implementation_plan.md) 的风险递增顺序，补齐 AuthContext、租户/资源授权、服务身份、审计、AIOps Agent 的 Ops Event、HITL、DB Executor Policy、多监控源、巡检/对比报告、SLO、Trace、告警和 Runbook。模型托管、Parser、KC、Main API、MCP 问数 Adapter 和 AIOps Agent 均通过明确的 Client/DTO 通信；Data Agent 暂不实现。

**验收：** 每次执行可追踪到 Agent、Skill 版本、输入范围、Artifact、策略决定和证据；不合规的计划或变更无法通过运行时执行。

## Phase 6：统一验收、空库发布与启用

在所有 4.0 能力完成后统一进行 Oracle、Portal、APEX、Parser、检索、Agent、Skill、Ops 和安全测试。目标环境从空 Schema 执行按服务拆分的规范建库脚本，不复制 3.x Domain、模型、知识、Agent 或 Ops 数据。Portal/APEX 在 4.0 中创建新 Domain 和配置，业务来源在入口启用后通过 4.0 API 重新入库，采用一次启用而非线上双写、双读或旧 API Adapter。

**验收：** 所有已批准调用方均使用 `/api/v1` 契约；Schema 中不存在 3.x KBot 表或数据，生产运行中不存在旧 Worker、旧 Skill 动态适配或新旧数据同步任务。

统一测试分层、质量阈值、Release Evidence、空库建库、生产启用和前向修复门禁见 [41_kbot4_step12_acceptance_release_and_cutover.md](41_kbot4_step12_acceptance_release_and_cutover.md)。入口启用并产生 4.0 写入后，通过能力 Kill Switch、维护模式和幂等重放前向恢复。

## 测试与观测

- 单元测试：领域状态机、检索融合、幂等、权限、UoW 回滚、Skill schema 与策略规则。
- 集成测试：Oracle Text/Vector、真实全量建库 DDL、对象存储、HTTP client 契约。
- 契约测试：Portal→Ingestion、Main/Agent→Discovery/Evidence、Parser Result。
- 端到端测试：Bundle 入库到引用证据的完整链路、Agent→Skill 委派、审批拒绝，以及重试、取消、Worker 崩溃恢复。
- 指标：Job 队列延迟、领取冲突、解析/索引耗时、失败率、Embedding 调用量、Discovery/Evidence 延迟、召回质量、PARTIAL 比例、Skill 成功率、路由准确率和变更拦截率。
- 日志和 Trace 必须含 `request_id`、`run_id`、`task_id`、`bundle_id`、`document_version_id`、`job_id`、`skill_id`；敏感正文与凭据不得写入日志。

## 发布原则

DDL 先于 4.0 代码部署；当前开发阶段只维护 `database/oracle/<service>/` 下的规范全量脚本，并通过重建测试 Schema 验证。脚本与发布物一起冻结校验和，不支持 3.x/4.0 旧新读写并行，也禁止 App 启动时自动改表。
