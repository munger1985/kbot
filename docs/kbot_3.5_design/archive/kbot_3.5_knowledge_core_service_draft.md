# KBot 3.5 独立 Knowledge Core 服务方案

## 1. 决策

在 3.5 新增独立 `knowledge-core` 服务，形态与现有 LLM、Embedding、VLM、Visual 服务一致：独立 FastAPI 进程、独立端口、独立 API、独立领域代码，但仍位于同一仓库、同一 Oracle Schema 和同一部署编排中。KM Portal 的 Asset + 附件是第一个接入场景，不是 Core 的专用领域边界。

3.5 直接建立通用 Collection、Bundle、Document Version、Parse View、Evidence、Discovery、Relation 和 Job 模型；但只接入 KM Asset 这个来源，不改 Agent/Skill 工作流、主会话编排、身份体系或 AIOps。后续项目、工单、产品资料等来源只需实现 Bundle Ingestion Adapter，无需重新设计知识数据层。

## 2. 为什么 Parser 不是 Knowledge Core

当前 Parser 已完成 Docling/OCR/VLM 解析、分块、Embedding、文档元数据和关系抽取，但它轮询 `FileEntity` 并直接写 File/Chunk 表。它是高资源的**内容处理 Worker**，不是知识领域核心：它不拥有来源修订、Bundle 附件清单、版本切换、生命周期状态、Discovery 排序、Evidence 查询或对外稳定契约。

3.5 的职责切分如下：

```text
KM Portal ──HTTP──► Knowledge Core (18090, 建议)
                         │
                         │ claim / result HTTP
                         ▼
                    Parser Worker (18095)
                         │
                         ▼
              Embedding / LLM / VLM services

Main API / existing DocService ──HTTP──► Knowledge Core
```

Knowledge Core 是控制面和查询面；Parser 是无状态或低状态执行面。Parser 不再轮询旧 `kbot_md_kb_files`，也不直接写 Knowledge Core 的业务表。

## 3. 3.5 通用数据模型

Knowledge Core 不使用旧 `FileEntity`、`TxtChunkEntity` 作为主存储。新表使用 `KBOT_KC_` 前缀，旧表保持不变以继续支撑尚未迁入 Core 的知识库。

| 表 | 3.5 职责 | 关键约束 |
| --- | --- | --- |
| `KBOT_KC_COLLECTION` | 知识边界、默认安全等级和元数据 | `collection_key` 唯一 |
| `KBOT_KC_BUNDLE` | 来源业务对象、修订、Facet、状态和主信息 | `(collection_id, source_system, source_type, source_id)` 唯一 |
| `KBOT_KC_DOCUMENT` | Bundle 内逻辑文件与角色 | `(bundle_id, external_document_id)` 唯一 |
| `KBOT_KC_DOCUMENT_VERSION` | 不可变物理内容、hash、URI、当前版本和解析状态 | 同一 Document 仅一个 `is_current=1` |
| `KBOT_KC_PARSE_VIEW` | TEXT / VISUAL / HYBRID 解析表示、质量和配置快照 | 同一范围仅一个 ACTIVE 主视图 |
| `KBOT_KC_EVIDENCE` | 可检索片段、向量、页码、层级、坐标和质量 | 归属 Version 与 Parse View |
| `KBOT_KC_DISCOVERY_OBJECT` | Bundle/Document 检索画像、全文、向量和 Facet | 当前版本索引才可见 |
| `KBOT_KC_RELATION` | Bundle 内确定性/可回溯关系 | 必须记录来源与 Evidence 依据 |
| `KBOT_KC_INGESTION_JOB` | PARSE / PROFILE / INDEX / RELATE 任务、租约和重试 | 幂等键 + 有限租约 |

KM Asset 映射为 `source_system=metadb`、`source_type=asset`、`source_id=<asset_id>` 的 Bundle；Portal 生成的结构化主信息是 `MANIFEST` Document，SharePoint 文件是 `ATTACHMENT` Document。新来源只需提供相同的 Bundle Manifest 与附件描述。

Evidence 的 `retrieval_text` 必须由 Bundle 标题/Facet、Document 角色/名称、标题路径和正文拼接而成；不得仅复制正文，也不得将多个附件拼接成单个大文本。为 Evidence 建立 Oracle Text、Vector、`bundle_id/document_id/is_active/security_level` 索引；为 Discovery Object 建立全文/向量和高频 Facet 索引。

## 4. 服务 API

### 对 Portal 和管理端

- `POST /v1/bundles`：接收 Bundle Manifest、附件描述和 multipart 文件；返回 `bundle_id`、`accepted_revision`、`status`。
- `GET /v1/bundles/{bundle_id}`：返回 Bundle、Document、Version、任务和 `READY/PARTIAL/FAILED` 状态。
- `DELETE /v1/bundles/{bundle_id}`：撤销检索可见性并清理 Bundle 文件、任务和 Evidence。

### 对 Parser 的内部协议

- `POST /internal/v1/parse-tasks/claim`：Parser 以 `worker_id`、能力（OCR/VLM/格式）和容量领取租约任务。
- `POST /internal/v1/parse-tasks/{task_id}/result`：提交 Document 解析结果、文档元数据、Evidence 批次、parser/model 版本和失败信息。
- `POST /internal/v1/parse-tasks/{task_id}/heartbeat`：长文件处理时续租。

Parser 通过 API 领取和回传；Knowledge Core 在单个事务内校验 lease、写 Evidence、更新 Document、推进 Bundle 状态和 Discovery Object。大批量 Evidence 使用受限批次提交或临时结果 URI，不能将数百页结果塞进单个请求。

### 对主 API、页面与未来 Agent

- `POST /v1/discovery`：按自然语言、Collection、Facet、安全等级和 `top_k` 返回 Bundle/Document 卡片、附件命中与状态。
- `POST /v1/evidence`：仅在指定 `bundle_ids`/`document_ids` 内返回可引用 Evidence 和相邻上下文。

现有 Agent/Skill 不直接调用这些 API。只在 `TxtBaseSearch` / `DocService` 这个单一检索接缝增加 `KnowledgeCoreClient` 适配：当 KB 配置指定 `retrieval_backend=knowledge_core` 时，将 Evidence DTO 映射为当前 `TxtBaseSearchResult`。这样 ask-doc Skill、Doc Orchestrator、Root Agent 和现有 SSE 契约无需修改；新的 Asset 页面可直接使用 Discovery/Evidence API。

## 5. 入库、解析与检索流程

```text
Portal POST Bundle
  → Core 写 Bundle + Documents + Parse Tasks (PENDING)
  → Parser claim task
  → parse / OCR / VLM / embed
  → Parser POST result batches
  → Core 写 Evidence + document metadata
  → Core 重建 Discovery Object
  → Bundle READY / PARTIAL / FAILED

Discovery: Discovery Object 混合召回 → Bundle 排序 → Document/附件命中
Evidence: 限定 bundle/document → Evidence 混合召回 → 主 Parse View 选择、章节/页码去重与邻接扩展
```

Bundle 只有在 MANIFEST 与至少一个可用附件/主信息 Evidence 就绪时进入 `READY`；附件局部失败为 `PARTIAL`。来源修订重复提交返回现有状态；新修订创建新的 Document Version，并在新 Evidence/Search 就绪后切换 current version。3.5 可以保留历史 Version 供审计，但不必实现历史版本检索 UI。

## 6. 与问数（NL2SQL）的边界

问数不应成为 Knowledge Core 的一个能力模块，而应是与其并行的 `data-query`（或 Semantic Query）领域服务。两者都可被主 API/未来 Agent 调用，但解决的是不同问题：Knowledge Core 从非结构化内容中返回可引用 Evidence；Data Query 从受治理的结构化数据中生成、校验并执行查询，返回表格、指标和 SQL 审计信息。

现有 `kbot_db_executor.py` 应继续作为 Data Query 的**底层受控执行器**，不作为对用户或 Agent 的问数入口。它目前接收 SQL 和连接配置并执行，缺少数据源注册、元数据同步、指标口径、语义规划、查询权限和结果生命周期，不能由 Knowledge Core 承担或绕过。

```text
Main API / Agent Router
  ├─ 非结构化问题 ──► Knowledge Core ──► Discovery / Evidence / citations
  ├─ 指标、明细、聚合问题 ──► Data Query ──► semantic plan / SQL / table result
  └─ 混合问题 ──► 并行调用并合成；每段结论标明 Evidence 或 query_result_id

Data Query ──受控 SQL──► DB Executor ──► registered datasource
```

Data Query 后续应拥有 `DataSource`、Schema Snapshot、Dataset/Table、Metric、Dimension、Semantic Model、Query Request/Plan/Result/Audit` 等模型，并负责：只读策略、行/列权限、连接凭据引用、SQL AST/白名单校验、成本/超时/行数限制、脱敏与审计。连接配置不得继续由普通调用方在请求中提交。

Knowledge Core 可以索引数据字典、指标口径、报表说明、数据契约等文档，并以 `external_ref` 与 Data Query 的 Dataset/Metric 建立 Relation；但不保存业务事实表副本、不生成或执行 SQL，也不以 Evidence 替代指标计算。查询结果默认只保存在 Data Query 的短期结果存储；只有经显式审批的解释性摘要才可作为知识重新入库，避免陈旧数值和权限泄漏。

3.5 不实现完整 Data Query 服务，只预留 Agent Router 的 `retrieval` / `data_query` 能力路由和跨服务引用字段；现有 DB Executor 维持兼容。这样 KC 的表模型不被结构化数据执行语义污染，后续可以独立演进问数服务。

### Excel：同一 Document 的双表示与按意图路由

Excel 不在入库时被二选一地定义为“问文”或“问数”。它首先是 Knowledge Core 的 `Document Version`；Parser 产生一个 `SPREADSHEET` Parse View，同时输出两类派生结果：

- 表/子表、标题层级、说明文字、截图与关键行作为 Evidence，供定位、解释、总结和引用；VLM 用于增强版式和语义识别。
- 规范化的 Sheet/Table/Column/Row 结构化工件（建议 Parquet、Arrow 或 JSON URI），供可选的数值查询；数值计算必须基于此工件的单元格值和公式计算值，不能基于 VLM 文本描述。

```text
Excel Document Version
  └─ SPREADSHEET Parse View
       ├─ Evidence：表名、标题、说明、关键行、页/区域定位
       └─ structured_artifact_uri：规范化表格数据
                                      │
                                      └─ Data Query File Dataset（可选注册）
```

路由由问题意图决定，而不是由文件类型决定：定位/解释/总结问题走 Knowledge Core；筛选、聚合、分组、排序、同比环比等数值问题走 Data Query；需要解释计算结论的混合问题并行调用，两段答案分别附 Evidence 或 `query_result_id`。

不要默认将每个 Excel 物化导入 Oracle 业务库。Data Query 应将结构化工件注册为只读的 `dataset_type=file_derived`，以隔离分析引擎查询（例如 DuckDB 读取 Parquet）；只有明确存在跨文件、长期且受治理的分析需求时才进入数仓。该 Dataset Version 必须绑定 `document_version_id + parse_view_id`，继承 Collection/Document 权限，记录 Sheet/表来源、逻辑计划或 SQL、行数与审计信息；新 Document Version 生效时创建新 Dataset Version，旧版本不再作为默认查询对象。

3.5 先实现 `SPREADSHEET` Parse View、`structured_artifact_uri`、表清单/字段推断/解析置信度和 Evidence 检索；完整 File Dataset 注册与自然语言问数属于后续 Data Query 服务。这样既保留当前 VLM 切表能力，也不因 Excel 提前建设数仓或混淆 KC 的所有权。

## 7. 代码与部署

```text
knowledge_core/
  api/                # bundles, discovery, evidence, parser-internal routes
  application/        # bundle ingestion, task/result application, retrieval
  entities/ repositories/
  parser_contract/    # request/result Pydantic DTO
  retrieval/
kbot_app_knowledge.py
utils/clients/knowledge_core.py
```

`start_kbot.sh` 增加 Knowledge Core 进程和端口检查。Knowledge Core 复用 `core` 的配置、日志、Oracle 异步连接和模型 HTTP Client，但不 import `agent`、`skills`、旧 `services/kb` 或 `services/search`。Parser 可复用 Docling、OCR/VLM、ChunkResult 生成和 embedding 调用，但改为使用 `KnowledgeCoreClient`；旧 `FileProcessor` 留给非 Asset 旧流程。

## 8. 分阶段实施

1. 建立 `KBOT_KC_*` DDL、Entity/Repository、Knowledge Core 健康检查和 Bundle 上传 API；Portal 改为一次 Bundle 上传。
2. 实现 Parse Task claim/result；将现有 FileProcessor 的解析/embedding逻辑提取为 Parser Worker handler，输出 Core DTO，不再访问旧 File/Chunk Repository。
3. 实现 Discovery、Evidence、Profile 和确定性 Relation，建立 Bundle 标注集和解析/检索评测。
4. 为已迁入 Collection 在 `TxtBaseSearch` 增加仅一处 Core Client 适配；不修改 Agent/Skill。
5. 稳定后再接入普通 KB 和其他来源；多 Agent、通用策略/权限、复杂关系推理属于 4.0 后续范围。

## 9. 边界规则与验收

- Portal 是来源同步 Adapter：下载 SharePoint、规范化 Metadb 字段、调用 Bundle API；不写 KBot 数据库。
- Knowledge Core 是 Collection/Bundle/Document/Evidence、任务状态和检索结果的唯一写入者。
- Parser 只处理已领取的 Document Version，不能自行发现 Bundle、创建任务或改变 Bundle 状态。
- Model 服务只推理，不持久化知识数据；Main API/Agent/Skill 不直连 `KBOT_KC_*` 表。

验收包括：一个 Asset Bundle 的多附件原子接收、Parser 崩溃后租约重领、部分附件失败的 PARTIAL、来源修订幂等、旧 KB 与 Core Collection 检索共存、Discovery 的 Bundle Recall@K 和 Evidence 的附件/页码定位准确率。3.5 完成时，已迁入 Collection 的所有数据访问必须经过 Knowledge Core API 或其内部 Repository，不能回落到旧 File/Chunk 表。
