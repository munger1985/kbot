# 4.0 Knowledge Core 设计

## 定位与 API

Knowledge Core 是 KBot 内唯一拥有知识资产生命周期和检索索引的领域服务。它接收 Bundle 与附件，输出可发现的 Bundle/Document 和可引用的 Evidence；它不生成最终答案。

| API | 用途 | 调用方 |
| --- | --- | --- |
| Bundle Ingestion API | 创建、更新、删除、恢复 Bundle/Document Version；返回 job 状态 | KM Portal、管理端 |
| Discovery API | 根据自然语言与 Facet 查找 Bundle/Document | Agent、Portal、Skill |
| Evidence API | 在指定候选 Bundle/Document 内返回可引用证据 | Agent、回答层 |
| Job/Admin API | 查询、取消、重试、审计任务 | 管理端、运维 |

API 必须携带权限上下文、`collection_id`、稳定 ID 和版本字段。Discovery 不能返回大段 Chunk；Evidence 必须返回 `bundle_id → document_id → document_version_id → unit_id → section/page/bbox`。

4.0 的 `collection_id/bundle_id/document_id/document_version_id/evidence_id` 均是同一个 UUIDv7 领域主键：Oracle 表以 `RAW(16)` 保存 PK/FK，API 序列化为规范 UUID 字符串；不再并存数字 ID 和 `*_UID`。具体规则见 [31_aiops_step2_persistence_and_identity.md](31_aiops_step2_persistence_and_identity.md)。

## 数据模型与所有权

Knowledge Core 独占 `KBOT_KC_COLLECTION`、`KBOT_KC_COLLECTION_BINDING`、`KBOT_KC_INGESTION_RECEIPT`、`KBOT_KC_BUNDLE`、`KBOT_KC_BUNDLE_REVISION`、`KBOT_KC_DOCUMENT`、`KBOT_KC_DOCUMENT_VERSION`、`KBOT_KC_BUNDLE_REVISION_DOCUMENT`、`KBOT_KC_PARSE_VIEW`、`KBOT_KC_EVIDENCE`、`KBOT_KC_DISCOVERY_OBJECT`、`KBOT_KC_RELATION`、`KBOT_KC_INGESTION_JOB` 和本领域消息表。精确字段、约束和索引由 `database/oracle/knowledge_core/` 的全量建库脚本维护，历史方案只作为决策来源。

Bundle 是业务对象，Document 是其成员，Version 是不可变物理版本。新版本索引完成前不得替换 current version。附件解析失败可将 Bundle 标记为 `PARTIAL`，但必须保留失败原因和已可用证据。

## 入库与任务编排

```text
POST Bundle
  → validate source revision/content hash
  → UoW: Bundle + Document + Version + PARSE job + Outbox
  → Parser claims PARSE
  → apply parse views/evidence; enqueue PROFILE and INDEX
  → build document/bundle profiles and deterministic relations
  → build Discovery Object
  → READY / PARTIAL
```

上传使用临时对象路径并原子发布；数据库仅保存稳定 `storage_uri`。入库接口必须幂等：相同来源和 `content_hash` 返回已有处理结果，不重复创建版本或任务。

## Parser Worker 的新契约

Parser 已是独立进程，4.0 保留这一部署边界。它只消费任务并调用 Docling、OCR、VLM、表格/PPT 专项解析；可按 CPU/GPU 和队列积压独立扩缩。

Parser 不轮询 Document 业务状态、不直接将 Bundle 改为 READY，也不决定当前版本。它提交带 `parser_version`、质量指标、页码/坐标和输入 hash 的 Parse Result；Knowledge Core 的 result applier 校验任务租约和版本后写入 Parse View/Evidence Unit 并推进状态机。

Parser 使用统一流水线，但支持 `TEXT / AUTO / VISUAL / HYBRID` 四种解析策略。默认 OCR 由 Docling 提供；配置独立 DeepSeek OCR 后关闭 Docling 内置 OCR，由 Parser Worker 直接调用其 OpenAI 兼容端点并把 grounding 结果转换为 OCR Atom，DeepSeek OCR 不进入 Model Serving。`AUTO` 再按页面文本覆盖率、提取置信度和乱码比例选择整页 VLM；`VISUAL` 强制逐页视觉解析；`HYBRID` 对全部有页面图像的页面执行视觉结构校正。输出统一进入 Atom IR、Structure IR、质量门和 Evidence Planner，不恢复多套 Chunk 模型。

健康页面保留 Docling 的精确文本、表格和坐标，VLM 只校正匹配标题并补充图表语义；低质量页使用整页 VLM Markdown 替换该页 Docling Atom。视觉结果必须记录模型、选择原因、页面级定位和替换方式，并保存为 `visual_analysis` 解析产物。某一页面的 VLM 调用失败时保留 Docling 结果，不得让局部增强失败破坏整份文档。

每个 Document Version 和解析配置只能发布一个 active 主视图：无视觉能力时为 `TEXT`，强制纯视觉时为 `VISUAL`，自适应或融合解析时为 `HYBRID`。这样避免多视图重复召回，同时保留旧版纯视觉 PDF 对扫描件、复杂排版、跨栏、表格和图表的优势。详细策略与验收要求见 [44_adaptive_hybrid_document_parsing.md](44_adaptive_hybrid_document_parsing.md)。

## 检索链路

Discovery 在 `KBOT_KC_DISCOVERY_OBJECT` 上执行 Oracle Text + Vector 混合召回，
先做权限、Collection、状态和 Facet 过滤，再进行 RRF 融合、
Document→Bundle 聚合与文件多样性处理。Agent 开启 `do_rerank` 时，KC 再按每个
Collection 的 Retrieval LLM 做对象级类别判断；关闭或模型降级时保留 RRF 顺序。

Evidence 只能在调用方给出的候选 Bundle/Document 范围内检索 `KBOT_KC_EVIDENCE`。它执行混合召回、视图去重、相邻章节扩展和有依据的关联扩展，再按上下文预算返回证据。Agent 只能消费 Evidence API 的结果，不能再次对全库自行选 Chunk。
开启重排时，KC 还会验证 Evidence Group 的支持关系并选定真实引用的 PRIMARY；
整个过程不对裸 Chunk 生成数值 rerank 分数。

## 与 3.x 的关系

旧 `/api/kb`、`FileService`、`FileProcessor`、`TxtBaseSearch` 和 `DocService` 不进入 4.0 运行时，也不提供适配路由、双写或双读。只选择性复用已验证的 Docling/OCR/VLM、层级解析和 Oracle 检索参数绑定能力，并在新契约、新表和新测试下重新封装。Portal、Agent 与 Skill 直接迁移到 Knowledge Core API。
