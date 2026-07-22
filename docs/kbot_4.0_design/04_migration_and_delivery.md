# 4.0 迁移、质量与交付计划

## Phase 0：边界冻结与代码清理

建立架构测试和依赖规则：`knowledge_core` 不可 import `agent`、`skills`、旧 `services/kb`；其他领域不可直接 import Knowledge Core Repository。将旧接口、兼容导出、V1 Entity/Repository 和未使用的 3.x Agent/Skill 移入 `legacy/`，4.0 包中不保留兼容 import。暂不进行最终质量测试，测试统一安排在所有 4.0 能力完成后。

**退出条件：** 服务边界、Owner、表权限、API 命名、UoW/Outbox 模板和迁移 DDL 流程经评审通过。

## Phase 1：平台基础与独立 App 边界

完善 `platform_core` 的配置、日志、认证、数据库连接和 Session Factory；每个 App 使用自己的配置实例和连接池，但 4.0 阶段仍指向同一 Schema。提供通用 `AsyncUnitOfWork`、领域 UoW factory、事务测试夹具和 Outbox dispatcher。所有 4.0 领域从第一行代码起使用它们。

**验收：** 一个跨多个 Repository 的写用例可验证原子提交和异常回滚；无新 Repository 调用 `commit()`。

## Phase 2：Knowledge Core 与解析检索闭环

在同一 APEX Schema 创建 4.0 专属表前缀和 migration；完成 `knowledge_core` 包、独立进程入口、Bundle Ingestion API、版本状态机、Job/Outbox、Parser、Discovery/Evidence 和检索质量链路。此阶段不接入 Agent。

**验收：** 一个含 Manifest 和多附件的 Bundle 可创建、更新、删除、恢复；重复提交不产生重复 Version/Job。

## Phase 3：Main API 与领域客户端

建立新的 Main API/BFF、Portal/APEX/MCP/Slack 客户端契约和服务身份上下文。所有跨服务调用使用版本化 DTO、HTTP 或 durable job；3.x Parser 轮询协议和旧上传接口不进入 4.0。

**验收：** 多副本 Worker 不重复领取任务；崩溃后租约可恢复；同一 PDF 可保留多视图并稳定定位页码。

## Phase 4：Agent Runtime 与 Skill Runtime

建立 Run/Task/Artifact/ExecutionContext、Supervisor、Specialist、Plan Validator、Policy Gate、预算、取消、恢复和事件流。用 Manifest、Typed DTO 和契约测试重写 Knowledge/Data/Ops Skill，旧 SkillRuntime、动态反射和 Prompt 授权规则不迁入 4.0。

**验收：** 建立人工标注集并达成 Bundle Recall@K、Document Recall@K、Evidence Recall@K、页码定位准确率和延迟目标；每个回答证据均可回链。

## Phase 5：身份、安全、运维与外部集成

补齐 AuthContext、租户/资源授权、服务身份、审计、Ops Event、HITL、DB Executor Policy、SLO、Trace、告警和 Runbook。模型托管、Parser、KC、Main API、Data Query 和 Ops Core 均通过明确的服务 Client/DTO 通信。

**验收：** 每次执行可追踪到 Agent、Skill 版本、输入范围、Artifact、策略决定和证据；不合规的计划或变更无法通过运行时执行。

## Phase 6：统一验收、重建与切换

在所有 4.0 能力完成后统一进行 Oracle、Portal、APEX、Parser、检索、Agent、Skill、Ops 和安全测试。根据原始来源重建 4.0 数据，采用发布切换而非线上双写、双读或旧 API Adapter；测试成功后删除 `legacy/` 中确认无用的代码、旧接口和旧表。

**验收：** 所有已批准调用方均使用 v4 契约；生产运行中不存在旧表写入、旧 Worker 轮询、旧 Skill 动态适配或新旧数据同步任务。

## 测试与观测

- 单元测试：领域状态机、检索融合、幂等、权限、UoW 回滚、Skill schema 与策略规则。
- 集成测试：Oracle Text/Vector、真实迁移 DDL、对象存储、HTTP client 契约。
- 契约测试：Portal→Ingestion、Main/Agent→Discovery/Evidence、Parser Result。
- 端到端测试：Bundle 入库到引用证据的完整链路、Agent→Skill 委派、审批拒绝，以及重试、取消、Worker 崩溃恢复。
- 指标：Job 队列延迟、领取冲突、解析/索引耗时、失败率、Embedding 调用量、Discovery/Evidence 延迟、召回质量、PARTIAL 比例、Skill 成功率、路由准确率和变更拦截率。
- 日志和 Trace 必须含 `request_id`、`run_id`、`task_id`、`bundle_id`、`document_version_id`、`job_id`、`skill_id`；敏感正文与凭据不得写入日志。

## 发布原则

DDL 先于 4.0 代码部署，并在 4.0 自身的后续小版本中保持前向兼容；不支持 3.x/4.0 的旧新读写并行。每次变更附带回滚方案、重建统计和验收记录。禁止使用无版本的“直接改生产表”方式交付 4.0。
