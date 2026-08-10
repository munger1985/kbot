# Ammolite AIOps 与知识检索后台迁移实施说明

## 范围

本迁移将 Ammolite Cube 的 AIOps 与知识检索 App 后台能力迁入 KBot 4.0。KBot 保留数字型 `domain_id` 作为业务和授权边界，不引入 Tenant、共享业务空间、跨 Domain 资源共享或兼容路由。

## 公开路由目标

- 知识检索 App：`/api/v1/apps/knowledge-retrieval`
- AIOps App：`/api/v1/apps/aiops`
- 服务内部路由：`/internal/v1`
- 现有 `/api/v1/agents`、`/runs`、`/conversations`、`/knowledge`、`/data-query` 与 `/ops` 在 App BFF 完成后删除或转为内部实现，不保留双路由。

知识检索 App 包含 Agent、Collection、文件处理、证据、数据源、语义模型、查询策略、会话和运行。AIOps App 包含 Agent、Target、监控源、策略、会话、证据、Proposal、审批、执行和报告。

## 权限目录

知识检索 App：

- `knowledge_retrieval:use`
- `knowledge_retrieval:upload`
- `knowledge_retrieval:review`
- `knowledge_retrieval:member_manage`
- `knowledge_retrieval:knowledge_manage`
- `knowledge_retrieval:data_manage`
- `knowledge_retrieval:agent_manage`
- `knowledge_retrieval:operations_manage`

AIOps App：

- `aiops:use`
- `aiops:domain_manage`
- `aiops:member_manage`
- `aiops:operations_manage`
- `aiops:target_manage`
- `aiops:monitor_source_manage`
- `aiops:policy_manage`
- `aiops:plan_manage`
- `aiops:agent_manage`
- `aiops:proposal:approve`

默认角色为知识检索的 `user`、`contributor`、`reviewer`、`manager`，以及 AIOps 的 `operator`、`approver`、`manager`。应用成员权限与 Agent Grant 同时满足后才能执行 Agent。

## 数据所有权

- Main API：用户目录、应用角色、成员角色和权限判定。
- Knowledge Retrieval App：知识检索 Agent、版本和授权。
- AIOps App：AIOps Agent、版本、授权、会话、证据和报告模板。
- Agent Runtime：不可变 Execution Spec、Conversation、Run、Task、Artifact、Event 和 Memory。
- Knowledge Core：Collection、文件摄取、Evidence、Visual Asset 和检索。
- Data Query：数据源、快照、语义模型、策略、绑定和查询运行。
- Platform Core：数据库 Prompt Registry、统一托管凭据和内部身份合同。

## 数据库迁移原则

- Oracle full-build DDL 是唯一 Schema 基线。
- 使用 AES-256-GCM 托管凭据，AAD 绑定 Domain、命名空间、种类和凭据 ID。
- Repository 不提交事务，Application Service 通过 UoW 提交。
- 不保留本地凭据文件、环境变量业务凭据、旧 Agent Definition、Target Binding 或专用凭据表的双读写路径。

## 已迁移能力

- 应用级 RBAC、成员角色和用户目录，权限与私有 Agent Grant 双重授权。
- 知识检索私有 Agent、不可变版本、Execution Spec 快照、会话、运行和记忆。
- Collection、文件摄取与重处理、证据、视觉资产、关系检索和检索策略。
- Data Query 数据源、统一托管凭据、快照、语义模型、策略、绑定、审计和查询运行。
- AIOps 私有 Agent、Target、Oracle/MySQL/PostgreSQL 诊断、监控源、策略、巡检、会话、图片 OCR/VLM 证据、Proposal、审批、执行、报告模板和通知。
- Main API 只公开 App BFF 路由；内部服务均使用服务凭据与 audience-bound AuthContext JWT。

明确不迁移 Tenant、共享业务空间、跨 Domain 共享与 3.x 兼容路径。

## 部署变化

- 新增 `knowledge_retrieval_app` 服务（18150）和 `model_ocr` 进程（18096）。
- 空库初始化新增 `knowledge_retrieval_app`，并以 `platform_core` 的 `KBOT_MANAGED_CREDENTIAL` 统一保存加密凭据。
- 生产部署只需主密钥；需要单独轮换凭据密钥时使用 `KBOT_MANAGED_CREDENTIAL_KEY` 和版本变量。

## 自动化验收

- Python `compileall` 与导入检查。
- Unit、Contract、OpenAPI、Oracle Schema Acceptance 和可用环境内的 Integration/Smoke。
- 权限矩阵、Agent Grant、跨 Domain 越权、Execution Spec 快照和凭据泄漏检查。
- 残留旧路由、Tenant、共享资源和旧表名扫描。

人工浏览器和人工外部依赖验证按本次要求暂不执行。
