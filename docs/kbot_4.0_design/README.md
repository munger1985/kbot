# KBot 4.0 架构设计

## 目标

KBot 4.0 是一次**Clean-slate 重构**：保留单仓库、同一 Oracle/APEX Schema 和统一发布节奏，同时按稳定业务边界拆分可独立运行的进程与 API。4.0 先完成逻辑上的独立微服务边界，数据库仍共享同一个 Schema；未来真正微服务化时，主要变化是为每个服务提供独立数据库连接和存储，不重新设计服务边界。

3.5 已实现的 `knowledge_core`、KC Migration、Parser/Projection Worker 与两阶段检索骨架直接构成 4.0 的 KC 基线；4.0 在这套实现上补齐质量、边界和生产能力，不再平行重建第二套 KC。本次重构同时建立适用于全系统的数据访问、事务、任务和服务边界规范。4.0 不保留旧 `kbot_md_*`、`KBOT_BIZ_*` 表模型、旧 `/api/kb` 契约、动态 Skill 适配器或旧 Agent 编排兼容层。旧系统仅作为原始数据和评测样本来源，代码历史由 Git 保存，不参与 4.0 工作树或运行时。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [01_architecture.md](01_architecture.md) | 目标运行架构、服务边界、调用规则和代码布局 |
| [02_data_and_transactions.md](02_data_and_transactions.md) | 数据所有权、Repository、Unit of Work、Outbox 与任务规范 |
| [03_knowledge_core.md](03_knowledge_core.md) | Bundle 入库、Parser Worker、Discovery 与 Evidence 的完整设计 |
| [04_migration_and_delivery.md](04_migration_and_delivery.md) | 分阶段重建、测试、观测、全量切换与验收门槛 |
| [05_multi_agent_and_skills.md](05_multi_agent_and_skills.md) | 多 Agent 协作、Skill 契约、策略与执行运行时 |
| [11_agent_execution_model.md](11_agent_execution_model.md) | Run/Task/Artifact、状态机、租约、事件流与恢复 |
| [12_agent_runtime_api_and_state_transitions.md](12_agent_runtime_api_and_state_transitions.md) | v4 Run API、内部命令、并发控制与状态迁移 |
| [13_agent_planning_and_skill_contract.md](13_agent_planning_and_skill_contract.md) | Supervisor、Planner、Specialist、Skill Manifest 与执行契约 |
| [14_main_api_bff_and_auth_context.md](14_main_api_bff_and_auth_context.md) | Main API/BFF、AuthContext、服务间传播与 v4 外部契约 |
| [15_aiops_agent_scope_and_skills.md](15_aiops_agent_scope_and_skills.md) | 独立 AIOps Agent、Ops 表、诊断与受控变更边界 |
| [16_document_agent_boundary_and_retrieval_contract.md](16_document_agent_boundary_and_retrieval_contract.md) | Document Agent、KC 检索编排与 CitationPack 契约 |
| [17_root_agent_routing_and_composition.md](17_root_agent_routing_and_composition.md) | Root Agent 路由、并行调用与多来源结果组合 |
| [18_agent_runtime_packaging_and_deployment.md](18_agent_runtime_packaging_and_deployment.md) | Agent Runtime、API/Worker、AIOps 服务和代码布局 |
| [19_aiops_domain_model_workflow_and_executor.md](19_aiops_domain_model_workflow_and_executor.md) | AIOps 表、状态机、HITL、执行/验证与 DB Executor 契约 |
| [20_aiops_monitoring_inspection_and_reporting.md](20_aiops_monitoring_inspection_and_reporting.md) | 多监控源、告警触发、日/周巡检与处理前后对比报告 |
| [21_aiops_interactive_diagnosis.md](21_aiops_interactive_diagnosis.md) | Chat 中数据库不可直连时的多轮只读 SQL、用户回贴与诊断恢复 |
| [22_aiops_database_diagnostic_catalog.md](22_aiops_database_diagnostic_catalog.md) | LLM 诊断动作、Oracle/MySQL Dialect、版本化 SQL 目录与只读执行护栏 |
| [23_aiops_policy_hitl_and_command_lifecycle.md](23_aiops_policy_hitl_and_command_lifecycle.md) | 单命令审批、Policy Decision、Advisory、一次性令牌与执行恢复 |
| [24_aiops_diagnosis_orchestration_and_evidence.md](24_aiops_diagnosis_orchestration_and_evidence.md) | 诊断编排、时间窗口、假设/反证、根因级别与解决方案 |
| [25_aiops_service_packaging_and_runtime.md](25_aiops_service_packaging_and_runtime.md) | AIOps API/Worker/Scheduler/DB Executor、包结构、UoW、事务与部署 |
| [26_aiops_physical_data_model.md](26_aiops_physical_data_model.md) | `KBOT_OPS_*` 字段、外键、唯一约束、索引、APEX 视图与保留策略 |
| [27_aiops_api_and_contracts.md](27_aiops_api_and_contracts.md) | AIOps 外部/内部/Executor API、DTO、SSE、鉴权、幂等与错误契约 |
| [28_aiops_implementation_plan.md](28_aiops_implementation_plan.md) | AIOps DDL、运行内核、监控、诊断、HITL、执行、报告与集成的实施顺序 |
| [29_aiops_step0_contracts_and_bootstrap.md](29_aiops_step0_contracts_and_bootstrap.md) | AIOps 包结构、四进程入口、DTO、Service Identity、配置与启动骨架 |
| [30_aiops_step1_oracle_schema.md](30_aiops_step1_oracle_schema.md) | 21 张 AIOps 表的 Oracle 类型、约束、索引、外键、APEX 视图与 Migration 顺序 |
| [31_aiops_step2_persistence_and_identity.md](31_aiops_step2_persistence_and_identity.md) | 单一 UUIDv7 主键策略、Oracle/PG 映射，以及 Entity、Repository、UoW、租约和 Inbox/Outbox |
| [32_aiops_step3_configuration_and_authorization_api.md](32_aiops_step3_configuration_and_authorization_api.md) | Target、Binding、Monitor、Policy、Inspection 配置 API、权限求交、ETag 与 Secret 生命周期 |
| [33_aiops_step4_deterministic_run_kernel.md](33_aiops_step4_deterministic_run_kernel.md) | 确定性 Run/Task/Artifact/Event 内核、租约 fencing、取消、重试与 SSE 恢复 |
| [34_aiops_step5_monitoring_observe_loop.md](34_aiops_step5_monitoring_observe_loop.md) | Prometheus/Zabbix/OEM Adapter、Metric Catalog、Webhook、Alert 与只观测报告闭环 |
| [35_aiops_step6_readonly_database_diagnostics.md](35_aiops_step6_readonly_database_diagnostics.md) | Oracle/MySQL 诊断目录、签名 Grant、只读 DB Executor、结果限界与 Artifact 集成 |
| [36_aiops_step7_diagnosis_orchestration_and_llm.md](36_aiops_step7_diagnosis_orchestration_and_llm.md) | Evidence Index、诊断轮次、工具计划校验、根因等级上限与结构化 LLM 接入 |
| [37_aiops_step8_chat_manual_diagnosis_hitl.md](37_aiops_step8_chat_manual_diagnosis_hitl.md) | Chat 人工 SQL、HITL 挂起/恢复、受限上传、用户证据和并发幂等 |
| [38_aiops_step9_advisory_approval_and_execution.md](38_aiops_step9_advisory_approval_and_execution.md) | Action Catalog、Advisory、单命令审批、Executor Claim、at-most-once 与执行验证 |
| [39_aiops_step10_inspection_reporting_and_comparison.md](39_aiops_step10_inspection_reporting_and_comparison.md) | Inspection Fire、多副本调度、版本化报告和确定性处理前后对比 |
| [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md) | Root Delegation、父子事件投影、Composer、Main API/SSE 和 APEX 集成 |
| [41_kbot4_step12_acceptance_release_and_cutover.md](41_kbot4_step12_acceptance_release_and_cutover.md) | 全系统测试矩阵、质量门禁、数据重建、生产切换、Mutation 启用与旧表退出 |
| [06_identity_security_and_tenancy.md](06_identity_security_and_tenancy.md) | 身份、租户、权限、密钥与数据安全治理 |
| [07_platform_operations_and_integrations.md](07_platform_operations_and_integrations.md) | 平台运行、可观测性、AIOps、MCP、Slack 与外部适配器 |
| [08_api_data_migration_and_lifecycle.md](08_api_data_migration_and_lifecycle.md) | API 契约、APEX Schema 变更、数据迁移与文件生命周期 |
| [09_clean_slate_implementation.md](09_clean_slate_implementation.md) | 4.0 最终目标蓝图、重写边界与实施原则 |

## 架构决策

1. 新增 `knowledge-core` 进程，拥有知识库新表、任务状态机和对外检索契约。
2. Parser 保留为独立 Worker，但变为 Knowledge Core 的任务执行者，不再拥有文件生命周期。
3. LLM、Embedding、VLM、Visual 保留为模型运行时服务；模型配置由 Platform 领域拥有，运行时只读。
4. 所有新领域采用“Repository 注入 Session、UoW 控制事务”；Repository 内禁止 `commit()`。
5. 跨进程可靠协作使用数据库任务/Outbox，不在数据库事务中调用 HTTP，也不以轮询旧业务表作为协议。
6. Agent 负责协作与决策，Skill 负责受限能力执行；高风险动作由独立策略/HITL 决定，不能仅依赖 Planner 提示词。

## 非目标

- 本阶段保留单仓库、单 Oracle 实例和单一 APEX Schema，不引入分布式事务。
- 每个可部署 App 都可以携带自己的配置、日志和 `platform_core` 依赖，并通过它创建独立的 DB Session Factory；当前连接同一 Schema，未来只替换数据库配置即可拆库。
- 不为 3.x API、表、Skill 包、工作流或 Agent 状态提供线上兼容、双写、双读或适配运行时。
- 不把 Agent、Skill、最终回答或图谱推理混入 Knowledge Core。
- 不重新实现已经进入 4.0 基线的 KC；只按新契约补齐、加固和重构其内部实现。
- 不保留旧接口、兼容导出、兼容 Adapter 或 `legacy/` 运行时代码；确认无 4.0 消费者后直接删除，必要时从 Git 历史查阅。

## 实施编码规则

- 新增或修改的代码注释、Docstring 和面向运维人员的日志正文统一使用中文。
- API 字段、错误码、枚举值、Trace/Metric 名称、Skill ID 等机器契约保持稳定英文，不做本地化。
- 旧表可以在数据归档和破坏性 Migration 获批前物理保留，但 4.0 不读、不写、不轮询；这属于数据安全措施，不是向下兼容。
