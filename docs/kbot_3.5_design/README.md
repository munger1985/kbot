# KBot 3.5 Knowledge Core 改造档案

## 目标与范围

3.5 建立独立的 Knowledge Core（KC）服务，并重构问文 Skill 的检索链路，完成来源对象及附件的版本化入库、解析、检索和可引用证据输出。首个来源是 KM Portal 的 Asset；其只是 Bundle Ingestion Adapter，不定义 KC 的领域边界。AIOps 与问数服务不在本期实现范围。

本方案仍是单仓库、同一 Oracle Schema 的分布式单体：KC、Parser、模型服务均独立进程、通过 HTTP 契约协作；KC 是 V2 知识数据的唯一写入者。3.5 采用双轨隔离：V1 保持既有 `KB → File → TxtChunk`、旧 Skill 和旧 API；V2 使用 Collection/Bundle/Evidence、V2 Skill 和 V2 API。旧表暂不删除，但 V2 不读写旧表，也不在请求内回退到 V1；迁移通过路由切流和重新入库逐步完成。

## 文档导航

| 文档 | 用途 |
| --- | --- |
| [01 架构与边界](01_architecture_and_boundaries.md) | 服务职责、API 边界、与问数的关系 |
| [02 领域模型与检索](02_domain_model_and_retrieval.md) | `KBOT_KC_*` 表、版本、解析、检索与关系设计 |
| [03 解析与 Excel](03_parsing_and_excel.md) | Parser Worker、多视图和 Excel 双表示策略 |
| [04 KM Asset 接入](04_km_asset_ingestion.md) | Portal 的首个 Bundle 接入契约与最小改造 |
| [05 实施计划与验收](05_delivery_plan_and_acceptance.md) | 实施阶段、迁移、评测和验收条件 |
| [06 实施路线图](06_implementation_roadmap.md) | 按依赖顺序推进的实施步骤与逐步完成门槛 |
| [07 步骤 0：范围与 Domain](07_step_0_scope_and_tenancy.md) | Domain 隔离、`app_id` 规则与 V1/V2 并行基线 |
| [08 步骤 0：Collection 与 Binding API](08_step_0_collection_and_binding_api.md) | Collection 管理、Agent 多对多绑定、删除与并发契约 |
| [09 步骤 1：表基础与根聚合](09_step_1_table_foundations.md) | Scope、审计、Collection 与 Binding 的字段、索引与不变量 |
| [10 步骤 1：Bundle 与 Revision](10_step_1_bundle_and_revision.md) | 来源身份、不可变修订、幂等、切换与附件快照 |
| [11 步骤 1：失败与恢复](11_step_1_ingestion_failure_and_recovery.md) | 上传、解析、部分可用、补传与重试状态机 |
| [12 步骤 1：Revision 文件成员](12_step_1_revision_document_member.md) | 附件清单、角色、缺失、补传与解析状态 |
| [13 步骤 1：Document 与 Version](13_step_1_document_and_version.md) | 逻辑文件、不可变内容、存储与重解析语义 |
| [14 步骤 1：Parse View 生命周期](14_step_1_parse_view_reparse_lifecycle.md) | 解析产物、质量门、成功后替换与物理清理 |
| [15 步骤 1：Evidence](15_step_1_evidence.md) | 可检索证据、向量、定位、可见性与引用契约 |
| [16 步骤 1：Parser 到 Evidence 契约](16_step_1_parser_to_evidence_contract.md) | Docling 输出映射、缺口、V2 DTO 与验收样例 |
| [17 步骤 1：Discovery Object](17_step_1_discovery_object.md) | Bundle/附件两阶段召回画像、版本切换与 Evidence 边界 |
| [18 步骤 1：Relation](18_step_1_relation.md) | 有依据的 Bundle 内语义关系、可见性与检索扩展边界 |
| [19 步骤 1：Ingestion Job](19_step_1_ingestion_job.md) | 异步编排、幂等、租约、回调和有限重试 |
| [20 步骤 1：入库流程演练](20_step_1_ingestion_walkthrough.md) | 从接收、解析到检索切换的逐表写入与状态模拟 |
| [21 步骤 2：服务与 API 契约](21_step_2_service_and_api_contract.md) | KC 进程、服务认证、KM Asset 入库与 Parser 内部协议 |
| [22 步骤 2：原子接收与对象发布](22_step_2_atomic_intake_and_object_publish.md) | multipart 暂存、校验、不可变发布、幂等与补偿 |
| [23 步骤 2：应用分层与事务](23_step_2_application_layer_and_transactions.md) | Application Service、UoW、Repository 与外部资源边界 |
| [24 步骤 2：Parser Worker 协议](24_step_2_parser_worker_protocol.md) | claim、租约、Evidence 批次、完成/失败和协议幂等 |
| [25 步骤 2：运行、安全与可观测性](25_step_2_runtime_security_and_observability.md) | 配置、健康检查、最小权限、审计、指标与限流 |

`archive/` 保存本轮整理前的原始方案，供追溯，不作为实施依据。

## 关键决策

- KC 输出两种查询能力：Discovery 用于找 Bundle/Document，Evidence 用于在候选范围内找可引用内容。
- Parser 是可替换的内容处理 Worker，不轮询旧文件表，也不直接写 KC 表。
- V2 问文 Skill 直接使用 KC 的两阶段检索与引用模型，不适配 `TxtBaseSearchResult`；V1 Skill 保持独立运行。
- 文档内容不可变；新来源修订创建新的 Document Version，索引就绪后才切换当前版本。
- 问数是并行的 Data Query 领域服务；KC 可检索数据字典，但不生成或执行 SQL。
- Excel 同时保留检索证据与规范化表格工件；按问题意图路由问文或问数。
