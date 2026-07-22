# 步骤 1 详细设计：Evidence

## 定位与归属

`KBOT_KC_EVIDENCE` 是可检索、可引用的最小知识证据单元，取代 V1 的通用 TxtChunk。它归属 `Document Version + Parse View`，而不归属单个 Bundle Revision：同一内容 Version 被后续 Revision 复用时，Evidence 也复用；查询通过当前 Bundle Revision 的 Member 清单限制可见范围。

```text
Current Bundle Revision
  → Revision Document Member
  → Document Version + ACTIVE Parse View
  → ACTIVE Evidence
```

Bundle 标题、Facet、Manifest 主信息等可变来源上下文不写入 Evidence 的持久 `retrieval_text`，避免同一 Version 被复用时产生过期或重复 Evidence。它们由 Discovery Object 召回 Bundle，并在最终 Citation Pack 组装时从当前 Bundle Revision 动态附加。

## 表字段

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `evidence_id` | `NUMBER(38)` PK identity | 证据标识 |
| `collection_id`, `bundle_id`, `document_id`, `document_version_id`, `parse_view_id` | 非空 `NUMBER(38)` | 归属链与过滤加速 |
| `evidence_key` | `VARCHAR2(256)` 非空 | Parser 在 View 内生成的稳定单元键 |
| `source_item_ref` | `VARCHAR2(512)` 可空 | Docling `self_ref` 或其他解析器的源项引用；用于幂等、追溯和定位 |
| `fragment_index` | `NUMBER(19)` 非空，默认 `0` | 同一源项拆分出的片段序号；与源项引用共同构成稳定键输入 |
| `parent_evidence_key` | `VARCHAR2(256)` 可空 | Parser 输出的父单元键；KC 批量写入后解析为下列 ID |
| `parent_evidence_id` | 可空 `NUMBER(38)` | 标题/表格/图文层级父节点，不强制外键 |
| `evidence_type` | `VARCHAR2(24)` 非空 | `TEXT/TABLE/IMAGE/SLIDE/CAPTION/CELL_RANGE` |
| `ordinal` | `NUMBER(19)` 非空 | View 内阅读顺序 |
| `heading_path_json` | JSON CLOB 可空 | 标题路径与层级标识 |
| `section_key` | `VARCHAR2(256)` 可空 | 解析器生成的稳定章节键，不采用 V1 的随机 UUID |
| `hierarchy_depth`, `heading_level` | `NUMBER(8)` 可空 | 当前层级深度和最近标题级别，支持邻接扩展与调试 |
| `content` | `CLOB` 非空 | 可验证的原始解析内容 |
| `retrieval_text` | `CLOB` 非空 | 内容、解析标题路径和稳定结构标签组成的检索文本 |
| `content_hash` | `VARCHAR2(64)` 非空 | Evidence 内容/定位的 SHA-256 |
| `page_start`, `page_end` | `NUMBER(8)` 可空 | 常用页范围过滤；完整多页定位仍在 `locator_json` |
| `locator_schema_version` | `VARCHAR2(32)` 非空 | 定位 JSON 的解释版本，避免解析器升级后歧义 |
| `locator_json` | JSON CLOB 非空 | 页码、bbox、Sheet/表/单元格范围、幻灯片等定位 |
| `payload_uri` | `VARCHAR2(2048)` 可空 | 图片、表格结构或大对象的受控 URI |
| `provenance_json` | JSON CLOB 可空 | OCR/VLM/规则提取来源、模型与提示词/配置摘要；不混淆原文与生成补充 |
| `language_code` | `VARCHAR2(16)` 可空 | 内容语言；由 KC/Worker 识别后写入，支持多语言检索观测 |
| `embedding` | Oracle `VECTOR` 可空 | 与当前嵌入模型维度一致 |
| `embedding_model_key` | `VARCHAR2(128)` 可空 | 向量生成模型/配置标识 |
| `quality_score` | `NUMBER(8,6)` 可空 | 单元质量或可信度 |
| `security_level` | `NUMBER(3)` 非空 | 从 Document Version 派生的检索安全过滤字段 |
| `evidence_status` | `VARCHAR2(16)` 非空 | `STAGED/ACTIVE/DELETING/FAILED` |
| `token_count` | `NUMBER(19)` 可空 | 上下文预算与观测 |
| 审计列 | 基础约定 | 生成服务与时间 |

`security_level` 是为检索过滤冗余的派生字段；它必须在 Evidence 写入和 Version 隔离时由 KC 同步，不由客户端提交。`app_id/domain_id` 不在 Evidence 表中，始终通过 `collection_id` 关联。

## 约束、索引与可见性

- `UK(parse_view_id, evidence_key)`：同一 View 内稳定去重。`evidence_key` 必须由 `source_item_ref + fragment_index + evidence_type`（或无源项引用时的确定性结构定位）生成，不能使用 V1 的随机 `chunk_id`、随机 Spreadsheet `section_id` 或单纯可变的阅读序号。
- 索引 `(collection_id, bundle_id, document_id, document_version_id, evidence_status)`：候选范围与回溯。
- 索引 `(document_version_id, parse_view_id, evidence_status, ordinal)`：邻接扩展和重解析清理。
- B-tree 索引 `(collection_id, security_level, evidence_status)`：检索预过滤。
- Oracle Text 索引：`retrieval_text`；Vector 索引：`embedding`。所有检索查询必须显式谓词 `evidence_status=ACTIVE`；STAGED/DELETING 行即使尚未完成物理清理，也不能进入结果。

`STAGED` Evidence 是候选 Parse View 构建产物，不参与任何查询。Parse View 成功切换时，候选 Evidence 批量变为 ACTIVE，旧 View 的 Evidence 立即撤销可见性并由清理任务物理删除。`FAILED` 仅用于生成中止后的短暂诊断，不能长期作为历史保留。

## 生成规则

Parser 根据候选 Parse View 输出章节、段落、表格、图片、幻灯片和 Excel 单元格范围等 Evidence；KC 仅在 View 切换成功后将其激活。每个单元必须具备可验证 `content` 和 `locator_json`；表格/图片可用 `payload_uri` 保存结构化数据或资源，但 `content` 仍需包含足以解释命中的文本摘要和列/标题信息。`locator_json` 至少包含 `pages[]`（页码、bbox、坐标系/页面尺寸）或 Spreadsheet 的 `sheet_name`、`cell_range`、行列边界；不能再把 Excel sheet 序号伪装成普通页码。

`retrieval_text` 只由 Document Version 实际 MIME、Evidence 类型、标题路径和 Evidence 内容确定性生成。当前 Parser 的 `search_helper` 混入 LLM 文档摘要，不能原样迁入；它只能作为重建策略的参考。Document Member 的角色/声明文件名、Bundle 标题、Facet 与 Manifest 主信息都可能随 Revision 变化，不能持久写入可复用 Evidence；它们在查询排序和 Citation Pack 组装时从当前 Member/Revision 动态附加。禁止在该字段写入 LLM 补充事实；OCR/VLM 的补充描述须在 `provenance_json` 中标明来源。

V2 不支持直接编辑单个 Evidence 文本。用户发现解析错误时，触发 Document Version 重解析；新 Evidence 成功后替换旧 Evidence，确保内容、定位、向量和解析配置一致。

## 查询与引用

Evidence API 先由 Discovery 选择 Bundle/Document，再根据当前 Bundle Revision Member、ACTIVE Parse View、Evidence Status 和安全等级过滤。混合召回后按标题路径、邻近页/单元格范围和 `parent_evidence_id` 去重、扩展上下文。

每条引用必须返回：`collection_id`、`bundle_id`、`bundle_revision_id`、`document_id`、`document_version_id`、`parse_view_id`、`evidence_id` 和 `locator_json`。其中 `bundle_revision_id` 来自查询时的当前 Member 关联，而非 Evidence 表持久字段。
