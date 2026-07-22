# KBot 知识库核心重构方案

## 1. 目标与范围

将 KBot 的知识库入库与检索重构为独立的知识库核心：输入 Bundle 和文件，输出相关 Bundle/文档或可引用证据。核心不负责 Agent、Skill、工作流或最终问答；这些上层能力后续调用 Discovery API 和 Evidence API。

KM Portal 保持为前置转换应用：从 Metadb 获取增量 Asset，整理 Manifest、下载附件，并以 Bundle 形式上传 KBot。

```text
Bundle Ingestion → 解析与索引 → Discovery API → Evidence API → 上层回答/Agent
```

不考虑旧表、旧 API 或旧业务模型的向下兼容；开发阶段直接采用新模型。

## 2. 核心领域模型

```text
Collection（知识库）
└─ Bundle（逻辑业务对象）
   ├─ Document（主信息文档或附件）
   │  └─ Document Version（物理版本）
   │     ├─ Parse View（Text / Visual / Hybrid）
   │     ├─ Document Profile
   │     └─ Evidence Unit（章节、表格、页、图片）
   ├─ Bundle Profile
   └─ Bundle Relation
```

Bundle 是泛用父对象，KM Asset 只是一个来源适配：`source_system=metadb`、`source_type=asset`、`source_id=<asset_id>`。项目、工单、产品资料等可沿用同一模型。

同 Bundle 的文件可关联检索、关联理解和关联组装，但不能简单拼接为一篇大文档；必须保留文件、版本、页码和引用边界。

## 3. 数据库模型

所有表使用 Portal/KBot 专属 Oracle 26ai schema。常用过滤字段显式列化；低频扩展属性使用 JSON。

### 3.1 `KB_COLLECTION`

知识库边界：`collection_id`、`collection_key`、名称、描述、状态、默认安全等级、`metadata` 和审计字段。该表不保存 Agent、Skill、App 耦合字段。

### 3.2 `KB_BUNDLE`

逻辑业务对象：`bundle_id`、`collection_id`、`source_system`、`source_type`、`source_id`、`source_revision`、标题、`canonical_url`、安全等级、状态、`metadata`、`content_hash` 和审计字段。

唯一约束：`(collection_id, source_system, source_type, source_id)`。KM 高频字段如作者、作者邮箱、产品、方案、分类、行业和创建时间，应映射为可查询 Facet 或显式列。

### 3.3 `KB_DOCUMENT`

Bundle 内逻辑文件：`document_id`、`bundle_id`、`external_document_id`、`role`（MANIFEST / ATTACHMENT / SUPPLEMENT / DERIVED）、标题、原始文件名、`source_url`、MIME 类型、排序号、安全等级、状态和 `metadata`。

唯一约束：`(bundle_id, external_document_id)`。Manifest 是普通但特殊角色的 Document，可被索引和引用。

### 3.4 `KB_DOCUMENT_VERSION`

物理文件与解析状态：`document_version_id`、`document_id`、`version_no`、`content_hash`、`storage_uri`、文件大小、`parse_policy`、`parse_status`、`parse_error`、`is_current`、`parser_version` 和审计字段。

更新时新增版本，而非覆盖；检索只读当前版本。

### 3.5 `KB_PARSE_VIEW`

保存同一版本的解析表示：`parse_view_id`、`document_version_id`、`view_type`（TEXT / VISUAL / HYBRID）、质量分、覆盖范围、状态（ACTIVE / CANDIDATE / REJECTED）、配置快照和 `output_uri`。

同一页段允许保存多个候选视图，但只启用一个主检索视图，防止重复召回。

### 3.6 `KB_EVIDENCE_UNIT`

证据最小单元，取代旧的单一 Chunk 表：`unit_id`、`bundle_id`、`document_id`、`document_version_id`、`parse_view_id`、`section_id`、`parent_unit_id`、单元类型、标题、层级路径、页码范围、坐标、`content`、`retrieval_text`、`embedding VECTOR`、质量分、`is_active` 与 `content_hash`。

`retrieval_text` 由 Bundle 关键事实、文档标题/角色、标题路径和当前正文构成。不得通过只截取正文前 500 字的方式无声丢失语义。

### 3.7 `KB_DISCOVERY_OBJECT`

专门服务“找文档”：`discovery_id`、`collection_id`、`object_type`（BUNDLE / DOCUMENT）、`bundle_id`、`document_id`、标题、`search_text`、`embedding VECTOR`、安全等级、状态、`metadata` 和 `index_version`。

Bundle 的检索画像包含 Manifest 事实、附件目录、附件 Profile、标题树、术语和编号；Document 画像包含文件 Profile、目录树、关键词和关键实体。LLM Summary 仅是可审计的辅助字段，不是唯一召回来源。

### 3.8 `KB_BUNDLE_RELATION`

Bundle 内关联：`relation_id`、`bundle_id`、源/目标对象类型与 ID、`relation_type`（REFERENCES / SUPPORTS / DUPLICATES / CONTRADICTS / SAME_ENTITY）、置信度、依据 Evidence Unit、来源（DETERMINISTIC / LLM）和状态。

首期优先写入确定性关系：Manifest 附件清单、父子关系、显式文件名/链接引用、相同编号或版本、共享产品或项目标识。LLM 关系必须保存可回溯的原文依据。

### 3.9 `KB_INGESTION_JOB`

可靠任务状态：`job_id`、`bundle_id`、`document_version_id`、`job_type`（PARSE / PROFILE / INDEX / RELATE）、状态、租约、尝试次数、下次重试时间、错误信息和参数快照。任务通过租约领取，支持多 Worker、重启恢复与失败重试。

## 4. 索引策略

首期保留应用侧 embedding、Oracle Text 和 Oracle VECTOR 的可控组合：

- `KB_DISCOVERY_OBJECT.search_text`：Oracle Text；`embedding`：HNSW 向量索引。
- `KB_EVIDENCE_UNIT.retrieval_text`：Oracle Text；`embedding`：HNSW 向量索引。
- Bundle/Document 权限、状态、来源、角色及高频 Facet：B-tree/JSON 索引。

Oracle 26ai Hybrid Vector Index 可作为后续替代方案；在 embedding 模型可部署到数据库侧、质量与运维策略完成验证前，不将其作为首期依赖。

## 5. 代码与架构改造

当前上传、存储、解析、embedding、元数据和搜索职责耦合在 `FileService`、`FileProcessor`、`TxtBaseSearch` 与 `DocService` 中。重构后的核心建议如下：

```text
knowledge/
├─ api/
│  ├─ bundle_ingestion_api
│  ├─ discovery_api
│  └─ evidence_api
├─ domain/
│  ├─ bundle
│  ├─ document
│  ├─ evidence_unit
│  └─ retrieval_models
├─ ingestion/
│  ├─ bundle_ingestor
│  ├─ document_version_manager
│  ├─ job_dispatcher
│  └─ bundle_coordinator
├─ parsing/
│  ├─ parse_policy_selector
│  ├─ docling_parser
│  ├─ visual_pdf_parser
│  ├─ view_quality_evaluator
│  └─ evidence_builder
├─ indexing/
│  ├─ document_profile_builder
│  ├─ bundle_profile_builder
│  ├─ discovery_indexer
│  └─ evidence_indexer
├─ retrieval/
│  ├─ discovery_service
│  ├─ evidence_service
│  ├─ rank_fusion
│  ├─ bundle_aggregator
│  └─ context_selector
└─ repositories/
```

可复用当前 KBot 的 Docling、OCR/VLM、层级树、跨页缝合、表格/PPT 专项解析、Oracle 异步连接池及 Oracle Text/向量参数绑定。应移除或替换当前以 Agent 为入口的多 KB 调度、直接返回 Chunk 的 `TxtBaseSearch` 职责、解析器对图谱/Skill 的隐式依赖，以及以 LLM YES/NO 为默认核心排序的机制。

## 6. Bundle 入库流程

```text
POST Bundle
→ 校验来源身份与版本
→ 写 Bundle / Document / Version / Job
→ 文件落临时目录并原子发布
→ 生成 Parse View 与 Evidence Unit
→ 构建 Document Profile
→ Bundle Coordinator 等待当前附件处理完成
→ 构建 Bundle Profile 与确定性关系
→ 写 Discovery Index
→ Bundle 进入 READY 或 PARTIAL
```

Manifest 直接以 Markdown/结构化字段构建 Profile，不执行复杂附件解析。附件失败不会阻止其他附件入库，Bundle 可进入 `PARTIAL`，Profile 必须列明处理状态。上游变更通过 `source_revision + content_hash` 幂等处理；新版本索引完成后才切换为当前版本。

## 7. PDF 多视图解析

PDF 不再全文件固定选择一个 `engine_mode`。解析策略选择器根据页段质量决定主视图：

| 条件 | 主检索视图 |
|---|---|
| 原生文本、标题层级可靠、阅读顺序正常 | TEXT |
| 扫描件、多栏、复杂图文混排、结构质量门失败 | VISUAL |
| 局部页面质量差、跨页结构异常 | HYBRID |
| 关键表格/图表 | TEXT 与 VISUAL 校验后选择 |

Text View 保存 Docling/OCR 的可验证文本、标题树和页码；Visual View 保存视觉模式的结构化 Markdown、页图和模型版本。Visual View 提升 Chunk 连贯性，但不能取代原始文本校验：编号、数值、日期和表格字段需保留 OCR/原文本依据。

解析时向模型提供受控 Bundle 上下文：标题、产品/方案、已知缩写、当前附件名和附件目录。提示词必须禁止补充原文件未出现的事实，并要求保留原始术语、编号、页码和不确定性。

## 8. Discovery API

Discovery 只负责返回相关 Bundle/Document，不回答问题，不输出大量 Chunk。

输入：`query`、`collection_id`、结构化过滤、权限上下文、结果范围（BUNDLE / DOCUMENT / BOTH）和 `top_k`。

处理步骤：

1. 权限与 Collection 过滤。
2. 查询 Bundle 和 Document Discovery Index 的关键词与向量结果。
3. 融合召回结果。
4. 将 Document 命中聚合回 Bundle。
5. 使用 Bundle Profile、最佳 Document、精确字段命中和文件多样性重新排序。
6. 返回 Bundle 卡片、相关附件、命中理由、稳定 ID 和入库状态。

Bundle 分数不使用单个 Chunk 最大值，建议由 Bundle Profile 分、最佳 Document 分、精确元数据奖励、多文档覆盖奖励和冗余惩罚组成。

## 9. Evidence API

Evidence API 必须在候选范围内工作。输入：问题、`bundle_ids` 或 `document_ids`、权限上下文、`top_k` 和上下文预算。

处理步骤：

1. 仅在允许范围内查询 `KB_EVIDENCE_UNIT`。
2. 混合召回文本、表格、图片和幻灯片单元。
3. 按章节、邻近页和 Parse View 去重，选取主视图。
4. 扩展同章节相邻单元。
5. 仅通过有明确依据的 Bundle Relation 扩展跨附件证据。
6. 按证据覆盖度选择最终结果。

输出必须包含 `bundle_id → document_id → unit_id → section/page/bbox` 的稳定引用定位。后续 LLM 回答层只能消费这些证据，不再自行选择知识库、附件或章节。

## 10. 实施阶段

### Phase 1：Bundle 领域模型与入库

建立新表、Repository、Bundle 上传 API、Manifest/附件契约、版本与任务状态机。

**验收：** 一个 KM Asset 的主信息与多附件能以一个 Bundle 入库、更新、删除和恢复。

### Phase 2：解析与 Evidence Unit

迁入 Docling、OCR、VLM、层级 Chunk；实现 Text / Visual / Hybrid Parse View 与选择策略。

**验收：** 同一 PDF 可保留多解析视图，Evidence Unit 可稳定定位文件、章节和页码。

### Phase 3：Document / Bundle Profile 与 Discovery

建立 Document Profile、Bundle Profile、附件目录、术语和确定性关系，构建 Discovery Index 和 Discovery API。

**验收：** 可按自然语言、作者、分类、标题和附件主题返回正确 Bundle/Document。

### Phase 4：Evidence Retrieval

实现候选范围内的混合检索、视图去重、邻接上下文、跨附件关系扩展与引用定位。

**验收：** 在指定 Bundle 内返回支持问题的章节、表格和页码，不依赖最终回答模型。

### Phase 5：评测与排序优化

建立人工标注集和持续评估：Bundle Recall@K、Document Recall@K、Evidence Recall@K、页码定位准确率、跨附件覆盖率、延迟、索引耗时和失败率。质量稳定后，再把 Agent、问答、Skill 或图谱接回核心。
