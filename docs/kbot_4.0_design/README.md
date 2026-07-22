# KBot 4.0 架构设计

## 目标

KBot 4.0 是一次**Clean-slate 重构**：保留单仓库、同一 Oracle/APEX Schema 和统一发布节奏，同时按稳定业务边界拆分可独立运行的进程与 API。4.0 先完成逻辑上的独立微服务边界，数据库仍共享同一个 Schema；未来真正微服务化时，主要变化是为每个服务提供独立数据库连接和存储，不重新设计服务边界。

本次重构以知识库为第一个完整落地领域，并同时建立适用于全系统的数据访问、事务、任务和服务边界规范。4.0 不保留旧 `kbot_md_*`、`KBOT_BIZ_*` 表模型、旧 `/api/kb` 契约、动态 Skill 适配器或旧 Agent 编排兼容层。旧系统仅作为能力、数据和评测样本来源，不参与 4.0 运行时。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [01_architecture.md](01_architecture.md) | 目标运行架构、服务边界、调用规则和代码布局 |
| [02_data_and_transactions.md](02_data_and_transactions.md) | 数据所有权、Repository、Unit of Work、Outbox 与任务规范 |
| [03_knowledge_core.md](03_knowledge_core.md) | Bundle 入库、Parser Worker、Discovery 与 Evidence 的完整设计 |
| [04_migration_and_delivery.md](04_migration_and_delivery.md) | 分阶段重建、测试、观测、全量切换与验收门槛 |
| [05_multi_agent_and_skills.md](05_multi_agent_and_skills.md) | 多 Agent 协作、Skill 契约、策略与执行运行时 |
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
- 不要求一次迁移所有旧模块；新标准先用于 4.0 新领域，再按风险逐步替换旧代码。
- 4.0 运行时不保留旧接口、兼容导出或兼容 Adapter；未迁移代码统一移入 `legacy/`，4.0 验收后删除。
