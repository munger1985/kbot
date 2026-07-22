# KBot 3.5 Knowledge Core 改造档案

## 目标与范围

3.5 建立独立的 Knowledge Core（KC）服务，并重构问文 Skill 的检索链路，完成来源对象及附件的版本化入库、解析、检索和可引用证据输出。首个来源是 KM Portal 的 Asset；其只是 Bundle Ingestion Adapter，不定义 KC 的领域边界。AIOps 与问数服务不在本期实现范围。

本方案仍是单仓库、同一 Oracle Schema 的分布式单体：KC、Parser、模型服务均独立进程、通过 HTTP 契约协作；KC 是知识数据的唯一写入者。项目仍处于开发阶段，3.5 直接以 Collection/Bundle/Evidence 和 KC Parser Worker 作为唯一实现基线，不保留 Parser/DTO/Schema 兼容分支，也不回退旧 `File/Chunk` 链路。

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
| [26 步骤 3：KM Portal Bundle Adapter](26_step_3_km_portal_bundle_adapter.md) | Portal 现状差异、字段映射、单次投递、回写与验收 |
| [27 步骤 3：Portal 代码改造清单](27_step_3_km_portal_code_change_plan.md) | 当前文件映射、职责拆分、配置、错误处理与测试 |
| [28 步骤 4：Parser Worker 改造](28_step_4_parser_worker_refactor.md) | V1 轮询改为 KC 租约 Worker 的模块、产物和测试边界 |
| [29 步骤 4：V2 Evidence DTO](29_step_4_v2_evidence_dto.md) | Parser 结果、稳定键、定位、来源与当前 Docling 字段映射 |
| [30 步骤 5：Discovery 与 Evidence API](30_step_5_discovery_and_evidence_api.md) | 两阶段检索、Scope、候选验证、Citation Pack 与错误契约 |
| [31 Docling 与检索待深化专题](31_open_decisions_docling_and_retrieval.md) | 解析、切块、召回、重排与评测的方案决策清单 |
| [32 步骤 5：检索基线与投影](32_step_5_retrieval_baseline_and_projections.md) | 可评测的 Profile/Evidence 索引、RRF 基线与策略版本化 |
| [33 步骤 6：KnowledgeRetrievalSkillV2](33_step_6_knowledge_retrieval_skill_v2.md) | V2 问文编排、Citation Pack、覆盖校验与 Agent/SSE 边界 |
| [34 步骤 6：回答溯源与 doc_results](34_step_6_answer_grounding_and_doc_results.md) | 真实引用验证、Asset/Document 卡片投影与 V2 SSE 契约 |
| [35 步骤 7：评测、直接上线与 V1 退役](35_step_7_evaluation_direct_release_and_retirement.md) | 样本标注、上线门槛、直接发布与旧模型清理 |
| [36 步骤 7：V1 退役清单](36_step_7_v1_retirement_inventory.md) | V1 API/表/服务盘点、保留对象、删除门禁与执行顺序 |
| [37 步骤 1：Schema 迁移计划](37_step_1_schema_migration_plan.md) | KC DDL 分组、索引、APEX 视图、回滚与验收 |
| [38 步骤 4：Docling 后处理重构](38_step_4_docling_postprocessing_redesign.md) | Docling 边界、新解析流水线、不可变工件与 KC 完成契约 |
| [39 步骤 4：Atom 与 Structure IR](39_step_4_atom_and_structure_ir.md) | 原子提取、阅读顺序、章节树、来源追踪与结构不变量 |
| [40 步骤 4：质量与 Evidence 规划](40_step_4_quality_and_evidence_planner.md) | Evidence 边界、表格/Excel、质量门、benchmark 与实施顺序 |
| [41 步骤 4：解析改造完成基线](41_step_4_parser_completion_baseline.md) | 已落地模块、运行闭环、支持格式、失败语义与后续评测输入 |
| [42 步骤 5：LLM 候选选择与证据判断](42_step_5_llm_selection_and_evidence_judging.md) | Bundle/Document 级 Listwise 选择、Evidence Group 支持判断与显式降级 |
| [43 步骤 5：Retrieval QueryPlan](43_step_5_retrieval_query_plan.md) | 检索多维意图、Facet 硬/软约束、Policy 选择与默认降级 |
| [44 步骤 6：DocumentAgentV2 与多 Agent 边界](44_step_6_document_agent_v2_and_multi_agent_boundary.md) | Agent→Skill→KC 调用方向、任务 DTO 与 4.0 演进接缝 |
| [45 步骤 5：Discovery 候选聚合](45_step_5_discovery_candidate_aggregation.md) | 单文档快速路径、Document 上卷 Bundle、Bundle 级 RRF 与 Collection 平权 |
| [46 步骤 2：普通用户文件上传](46_step_2_user_file_upload_api.md) | 每文件独立/单 Bundle 两种分组、批次幂等与原子接收语义 |
| [47 步骤 5：Evidence Group 与引用单位](47_step_5_evidence_group_and_citation_unit.md) | ANCHOR/上下文组装、Judge 后 PRIMARY、Group 级 citation 与预算 |
| [48 步骤 5：Embedding 一致性](48_step_5_embedding_space_invariant.md) | 全局维度、Collection 模型绑定、多模型分组召回与变更策略 |
| [49 步骤 5：INDEX 向量流水线](49_step_5_index_embedding_pipeline.md) | 解析与向量解耦、单一模型入口、向量身份校验与状态转换 |
| [50 步骤 5：Discovery Profile 基础](50_step_5_discovery_profile_foundation.md) | Bundle/Document 画像、覆盖摘要、幂等 Profile 文本与当前版本边界 |
| [51 步骤 5：Discovery 查询与候选聚合](51_step_5_discovery_query_and_candidates.md) | Oracle Text/Vector 初筛、Bundle 上卷、Collection 平权与 V2 查询契约 |
| [52 步骤 5：Evidence 查询与 Citation Pack](52_step_5_evidence_query_and_citation_pack.md) | 候选范围校验、Evidence Group、结构上下文和引用标签 |
| [53 步骤 6：Answer Grounding 与 doc_results_v2](53_step_6_answer_grounding_and_doc_results_v2.md) | 回答后引用校验、Claim 支撑状态与 Bundle 卡片投影 |
| [54 步骤 6：DocumentAgentV2 与 Skill 接入](54_step_6_document_agent_v2_skill_integration.md) | 无状态问文 Skill、KC Client、任务 DTO 与 Agent 边界 |
| [55 步骤 6：RootAgentV2 显式路由](55_step_6_root_agent_v2_route.md) | V2 Agent API、V2-only SSE 和 V1 隔离 |
| [56 步骤 6：RootAgentV2 Grounded Answer](56_step_6_root_agent_v2_grounded_answer.md) | 回答模型、引用校验和最终 doc_results_v2 |
| [57 服务打包与目录重组](57_service_packaging_and_directory_layout.md) | `knowledge_core` 文件职责、Parser 边界、独立服务打包和 4.0 演进 |

`archive/` 保存本轮整理前的原始方案，供追溯，不作为实施依据。

## 当前实施状态（开发分支）

已落地 KC Schema 001–007、Collection/Agent Binding 管理、KM Asset 与普通 `user-files` 入库、Parser 租约 Worker、Docling 后处理与可选 Excel 结构化工件、单一 Collection Embedding INDEX/PROFILE/PURGE Worker、两阶段 V2 检索与 Citation Pack 基础，以及 QueryPlan/对象级选择接口。尚未完成真实 Oracle/APEX 迁移演练、LLM Selector/Judge 的模型托管接入、离线标注评测和 Portal 最终切换；这些仍是实施计划中的阻塞项。

## 关键决策

- KC 输出两种查询能力：Discovery 用于找 Bundle/Document，Evidence 用于在候选范围内找可引用内容。
- Parser 是可替换的内容处理 Worker，不轮询旧文件表，也不直接写 KC 表。
- V2 问文 Skill 直接使用 KC 的两阶段检索与引用模型，不适配 `TxtBaseSearchResult`；V1 Skill 保持独立运行。
- 文档内容不可变；新来源修订创建新的 Document Version，索引就绪后才切换当前版本。
- 问数是并行的 Data Query 领域服务；KC 可检索数据字典，但不生成或执行 SQL。
- Excel 同时保留检索证据与规范化表格工件；按问题意图路由问文或问数。
- Docling 仅作为底层转换引擎；V2 重写其后的 Atom 规范化、结构解析、质量评估和 Evidence 规划，不沿用 V1 Chunk 结构。
- RRF 负责高召回和稳定先验；KC 使用 `LLMDiscoverySelector` 比较知识对象、使用 `LLMEvidenceSupportJudge` 判断证据支持，不引入逐 Chunk 数值 reranker。
- 3.5 同步重构 `DocumentAgentV2` 与问文 Skill：Agent 调用无状态 Skill，Skill 调用 KC；未来多 Agent 通过版本化任务 DTO 委派，不共享检索内部状态。
- `base.toml` 只锁定向量维度；每个 Collection 绑定 Embedding 模型，索引和查询严格使用该模型。多 Collection 若模型不同则分组生成查询向量、分别召回后再融合。
