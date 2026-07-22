# 4.0 剩余工作计划

## 目标修正

4.0 仍然保持单仓库、单 Oracle 实例和同一 APEX Schema。`knowledge_core`、
`model_serving`、Parser、Main API、DB Executor 和未来的 Agent Runtime 都是
独立进程，但数据库拆分不属于 4.0 验收条件。每个 App 通过自己的配置加载
`platform_core`，创建自己的日志上下文、连接池和 Session Factory；未来真正
拆库时只替换数据库配置、账号和连接池。

4.0 不保留旧接口、兼容导入、V1 Adapter、旧 SkillRuntime 或无用的 3.x 代码。
迁移期间的旧代码统一放入 `legacy/`，最终验收后直接删除。

## 当前基线

已经具备：

- `platform_core`：配置、日志、认证、数据库、ORM 基础类型和跨服务契约；
- `model_serving`：模型 Entity、Repository、配置 CRUD、Provider 和模型进程；
- `knowledge_core`：Collection、Bundle、Document、Version、Parse View、Evidence、Discovery、Job 和 UoW；
- `knowledge_core/repositories`：KC Repository 已脱离通用 `dao/repositories`；
- Parser Worker：Docling 后处理、结构 IR、质量评估和 Evidence 规划；
- Discovery → Evidence → Citation Pack 的两阶段检索骨架；
- DocumentAgentV2、KnowledgeRetrievalSkillV2 和 grounded response 基础链路。

这些属于可复用基础和 3.5 主链路，不等于 4.0 已完成。旧 Main API、V1
Agent/Skill、旧 Parser、旧业务 Service、旧模型兼容层和旧 SkillRuntime 已移入
`legacy/`；当前仍保留的 `agent/common` 仅作为 V2 Context 类型的过渡共享包。
`utils/` 仍保留文本/监控等迁移期工具，后续阶段会按实际消费者迁移或删除。

## 实施阶段

### 阶段 0：架构护栏和 Legacy 清理

- 建立 import 依赖检查：新领域不得依赖旧 `dao`、`services/kb`、`TxtBaseSearch`、`DocService` 或旧 Agent 编排。
- 删除不再使用的兼容导出；仍被 4.0 使用的旧代码先迁移到新包并补充明确接口。
- 将旧 API（含 schemas）、V1 Controller、旧 Skill/SkillRuntime、旧 Parser、旧
  Entity/Repository、DB Executor 和入口脚本移入 `legacy/`（已完成首轮归档）。
- 将 `apps/` 收敛为按服务命名空间组织的入口，明确 `knowledge_core`、`model_serving` 和 Main API 的归属。

### 阶段 1：平台基础和服务边界

- 完成 `platform_core` 的服务级配置、日志、认证、DB Session Factory 和观测上下文。
- 跨服务客户端已迁移到 `platform_clients`；稳定 DTO 放入 `platform_core/contracts`，后续只补齐版本和契约测试。
- 统一 UoW、Outbox、任务租约、重试、取消和幂等语义。
- 确认所有 App 在同一 Schema 下也只能访问自己拥有的表和 API。

### 阶段 2：Knowledge Core 闭环完成

- 完成 4.0 专属表前缀、DDL、migration、表 Owner 和索引。
- 完成真实 Bundle 入库、Parser、PROFILE、INDEX、Discovery、Evidence、Relation 和 Excel 结构化工件。
- 完成 KC→模型服务的配置和推理 Client，移除 KC 对模型 Entity/Repository 的直接依赖。
- 完成 KM Portal、普通文件上传、APEX 读取视图和对象存储协议。

### 阶段 3：Main API 与领域集成

- 重建 v4 Main API/BFF，只提供新契约。
- 建立 AuthContext、Service Identity、Domain/Collection 授权和请求上下文传播。
- 迁移 Portal、APEX、MCP、Slack 等 Adapter；禁止继续调用旧 `/api/kb` 和 V1 Agent 接口。

### 阶段 4：多 Agent 与 Skill Runtime

- 实现 Run、Task、Artifact、ExecutionContext 和事件持久化。
- 实现 Supervisor、Knowledge/Data/Ops/Conversation Specialist。
- 实现 Skill Manifest、Typed DTO、Skill Registry、Plan Validator、预算、超时、取消和恢复。
- 实现 Policy Gate、HITL 和 Mutation Skill 的 ChangeProposal 两阶段执行。
- 将问文、问数、运维能力重写为独立 Skill，禁止 Skill 直接实例化 Agent 或访问跨域数据库。

### 阶段 5：平台治理和业务扩展

- 完成统一审计、SLO、Trace、告警、成本统计和 Runbook。
- 完成 Data Query、Excel Dataset、NL2SQL 权限和结果 Artifact。
- 完成 Ops Core、DB Executor Policy、外部系统 Adapter 和失败补偿。

### 阶段 6：统一测试、重建和切换

前置开发阶段不单独冻结 3.5 验收。待阶段 0～5 全部完成后，统一执行：

- Oracle DDL、事务回滚、任务租约和多副本故障演练；
- Portal/APEX/MCP/API 端到端入库和查询；
- Docling 解析质量、Discovery/Evidence Recall、引用定位和 Excel 问数评测；
- Agent 路由、Skill 契约、权限、预算、HITL、取消和恢复测试；
- 压测、安全测试、观测验证和发布回滚演练。

测试成功后，删除 `legacy/` 中确认无用的代码、旧表、旧接口和旧配置，不保留
线上双读、双写或兼容 Adapter。

## 完成定义

4.0 完成的判断标准不是“目录拆完”，而是：每个 App 可以独立启动和扩缩，所有
跨服务调用使用明确 Client/DTO，所有领域使用自己的 UoW/Repository，Agent/Skill
执行可恢复和可审计，KC 检索结果可引用，且在同一 Schema 下没有跨域直接读写。
