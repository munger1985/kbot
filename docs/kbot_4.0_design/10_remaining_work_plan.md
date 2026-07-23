# 4.0 剩余工作计划

## 目标修正

4.0 仍然保持单仓库、单 Oracle 实例和同一 APEX Schema。`knowledge_core`、
`model_serving`、Parser、Main API、DB Executor 和未来的 Agent Runtime 都是
独立进程，但数据库拆分不属于 4.0 验收条件。每个 App 通过自己的配置加载
`platform_core`，创建自己的日志上下文、连接池和 Session Factory；未来真正
拆库时只替换数据库配置、账号和连接池。

4.0 不保留旧接口、兼容导入、V1 Adapter、旧 SkillRuntime 或无用的 3.x 代码。
旧代码确认没有 4.0 消费者后直接从工作树删除，只由 Git 历史留档。旧表可为归档和
回退取证暂时物理保留，但 4.0 不读、不写、不轮询。

## 当前基线

已经具备：

- `platform_core`：配置、日志、认证、数据库、ORM 基础类型和跨服务契约；
- `model_serving`：模型 Entity、Repository、配置 CRUD、Provider 和模型进程；
- `knowledge_core`：Collection、Bundle、Document、Version、Parse View、Evidence、Discovery、Job 和 UoW；
- `knowledge_core/repositories`：KC Repository 已脱离通用 `dao/repositories`；
- Parser Worker：Docling 后处理、结构 IR、质量评估和 Evidence 规划；
- Discovery → Evidence → Citation Pack 的两阶段检索骨架；
- DocumentAgentV2、KnowledgeRetrievalSkillV2 和 grounded response 基础链路。

其中 Knowledge Core 是 3.5 已完成并正式晋级的 4.0 基线，不再另建或重写一套；
其剩余工作是质量加固、契约收敛和生产闭环。DocumentAgentV2、旧 Main API、V1
Agent/Skill、旧业务 Service、旧模型兼容层和旧 SkillRuntime 不因 KC 晋级而自动
成为 4.0 组成部分。当前仍保留的 `agent/common`、`utils/` 及 `legacy/` 必须逐项
确认消费者：迁入明确 Owner 后删除原实现，或直接删除。

## 实施阶段

### 阶段 0：架构护栏和旧代码删除

- 建立 import 依赖检查：新领域不得依赖旧 `dao`、`services/kb`、`TxtBaseSearch`、`DocService` 或旧 Agent 编排。
- 删除不再使用的兼容导出；仍被 4.0 使用的算法先迁移到有明确 Owner 的新包并补充接口和测试，再删除原实现。
- 删除 `legacy/`、旧 API（含 schemas）、V1 Controller、旧 Skill/SkillRuntime、旧 Parser、旧 Entity/Repository、DB Executor 和入口脚本；不把它们保留到最终验收。
- 将 `apps/` 收敛为按服务命名空间组织的入口，明确 `knowledge_core`、`model_serving` 和 Main API 的归属。
- 把新增或修改代码中的注释、Docstring 和日志正文统一为中文；API 字段、错误码、枚举及可观测性键保持英文。

### 阶段 1：平台基础和服务边界

- 完成 `platform_core` 的服务级配置、日志、认证、DB Session Factory 和观测上下文。
- 跨服务客户端已迁移到 `platform_clients`；稳定 DTO 放入 `platform_core/contracts`，后续只补齐版本和契约测试。
- 统一 UoW、Outbox、任务租约、重试、取消和幂等语义。
- 确认所有 App 在同一 Schema 下也只能访问自己拥有的表和 API。

### 阶段 2：Knowledge Core 基线加固

- 以现有 `knowledge_core` 和 `migrations/kc` 为唯一实现，审核表 Owner、DDL、Migration 和索引，不重复建模。
- 完成真实 Bundle 入库、Parser、PROFILE、INDEX、Discovery、Evidence、Relation 和 Excel 结构化工件。
- 完成 KC→模型服务的配置和推理 Client，移除 KC 对模型 Entity/Repository 的直接依赖。
- 完成 KM Portal、普通文件上传、APEX 读取视图和对象存储协议。

### 阶段 3：Main API 与领域集成

- 按 `14_main_api_bff_and_auth_context.md` 重建 v4 Main API/BFF，只提供新契约。
- 建立 AuthContext、Service Identity、Domain/Collection 授权和请求上下文传播。
- 迁移 Portal、APEX、MCP、Slack 等 Adapter；禁止继续调用旧 `/api/kb` 和 V1 Agent 接口。

### 阶段 4：多 Agent 与 Skill Runtime

- 按 `11_agent_execution_model.md`、`12_agent_runtime_api_and_state_transitions.md` 和 `13_agent_planning_and_skill_contract.md` 实现 Run、Task、Artifact、ExecutionContext、状态机、租约、命令接口、事件持久化、Planner 和 Skill Manifest。
- 按 `17_root_agent_routing_and_composition.md` 和 `18_agent_runtime_packaging_and_deployment.md` 实现 Supervisor、Document/Conversation 路由及 Runtime API/Worker；AIOps 通过独立 Agent 服务接入，问数继续使用 MCP Adapter。
- 实现 Skill Manifest、Typed DTO、Skill Registry、Plan Validator、预算、超时、取消和恢复。
- 实现 Policy Gate、HITL 和 Mutation Skill 的 ChangeProposal 两阶段执行。
- 将问文能力重构为 Document Agent 契约；问数继续通过受控 MCP Adapter；运维能力由独立 AIOps Agent 管理其 Ops Skill 和流程，禁止任何 Skill 直接实例化 Agent 或访问跨域数据库。

### 阶段 5：平台治理和业务扩展

- 完成统一审计、SLO、Trace、告警、成本统计和 Runbook。
- 保留现有 MCP 问数链路；Data Agent、Excel Dataset 和 NL2SQL 独立服务暂不进入本轮实现，仅保留未来扩展契约。
- 按 `19_aiops_domain_model_workflow_and_executor.md` 至 `27_aiops_api_and_contracts.md` 的详细设计，以及 [28_aiops_implementation_plan.md](28_aiops_implementation_plan.md) 的施工顺序，完成独立 AIOps Agent、`KBOT_OPS_*` 表/DDL/APEX 视图、Public/Internal/Executor 契约、API/Worker/Scheduler/DB Executor 进程、监控/诊断/执行编排、假设/反证与根因分析、交互式诊断、Oracle/MySQL 诊断目录、DB Executor Policy、单命令审批、巡检报告、外部系统 Adapter 和失败补偿。

### 阶段 6：统一测试、重建和切换

具体 Gate、数据集、RC 证据、切换和 Legacy 退出策略见 [41_kbot4_step12_acceptance_release_and_cutover.md](41_kbot4_step12_acceptance_release_and_cutover.md)。

前置开发阶段不单独冻结 3.5 验收。待阶段 0～5 全部完成后，统一执行：

- Oracle DDL、事务回滚、任务租约和多副本故障演练；
- Portal/APEX/MCP/API 端到端入库和查询；
- Docling 解析质量、Discovery/Evidence Recall 和引用定位评测；现有 MCP Excel 问数仅做独立回归，不纳入 Data Agent 验收；
- Agent 路由、Skill 契约、权限、预算、HITL、取消和恢复测试；
- 压测、安全测试、观测验证和发布回滚演练。

测试成功并完成生产 Soak 后，确认工作树和部署物中不存在旧代码、旧接口、旧配置或兼容 Adapter；旧表先归档、撤销写权限，再通过单独批准的破坏性 Migration 删除。不保留线上双读、双写或兼容 Adapter。

## 完成定义

4.0 完成的判断标准不是“目录拆完”，而是：每个 App 可以独立启动和扩缩，所有
跨服务调用使用明确 Client/DTO，所有领域使用自己的 UoW/Repository，Agent/Skill
执行可恢复和可审计，KC 检索结果可引用，且在同一 Schema 下没有跨域直接读写。
