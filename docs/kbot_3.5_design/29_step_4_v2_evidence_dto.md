# 步骤 4 详细设计：Parser V2 Evidence DTO

## 目标

V2 DTO 是 Parser 与 KC 的 Evidence 写入契约，但不是唯一解析产物。Parser 还必须发布 Atom IR、Structure IR、质量报告和 Evidence Manifest；DTO 中的每条证据通过稳定来源跨度引用这些工件。KC 将 DTO 规范化为 `KBOT_KC_EVIDENCE`，不接收 V1 `ChunkResult/TxtChunkEntity`。

## 顶层任务结果

```json
{
  "dto_version": "kc-parser-evidence/v1",
  "parse_view_id": 501,
  "document_version_id": 301,
  "input_fingerprint": "...",
  "parser_version": "docling-worker/3.5.0",
  "policy_fingerprint": "...",
  "artifact_manifest": {
    "raw_docling": {"uri": "...", "sha256": "...", "schema": "docling/v1", "generator": "docling/2.x"},
    "atom_ir": {"uri": "...", "sha256": "...", "schema": "kc-atom/v1", "generator": "atom-normalizer/1"},
    "structure_ir": {"uri": "...", "sha256": "...", "schema": "kc-structure/v1", "generator": "outline-resolver/1"},
    "evidence_manifest": {"uri": "...", "sha256": "...", "schema": "kc-evidence-manifest/v1", "generator": "evidence-planner/1"}
  },
  "evidences": []
}
```

`parse_view_id/document_version_id/input_fingerprint` 必须与 claim 响应一致；Worker 不接受调用方自由指定。大结果拆为多个 batch，但每个 batch 使用同一顶层身份。

## 单个 Evidence DTO

| 字段 | 必填 | 规则 |
| --- | --- | --- |
| `evidence_key` | 是 | `ev1:{parse_view_id}:{source_key}:{fragment_index}:{type}`；不得随机生成 |
| `parent_evidence_key` | 否 | 使用相同算法；父节点未输出时为空 |
| `source_item_ref` | 否 | 兼容单 Atom 的 Docling `self_ref`；多 Atom Evidence 以 `source_spans` 为准 |
| `source_spans` | 是 | 有序 Atom ID、可选字符/单元格跨度和 locator；必须能回到 Atom IR |
| `source_key` | 是 | 对规范化 `source_spans` 的 hash，不能只用 ordinal 或合并后的文本 |
| `fragment_index` | 是 | 同源项分裂的从 0 开始序号 |
| `evidence_type` | 是 | `DOCUMENT/SECTION/PARAGRAPH/TABLE/TABLE_ROW/IMAGE/SHEET/CELL_RANGE` |
| `ordinal` | 是 | View 内阅读序；用于展示和邻接，不是身份 |
| `content` | 是 | 可由 `source_spans` 验证的文本/表格；不含摘要、cross-reference 或无来源 LLM 事实 |
| `heading_path` | 否 | 标题文本数组；KC 保存为 JSON |
| `section_key`、`hierarchy_depth`、`heading_level` | 否 | 确定性章节归属与层级信息 |
| `locator_schema_version`、`locator` | 是 | 下节定义的可引用定位 |
| `payload_descriptor` | 否 | 图像、表格结构或大对象的受控临时 URI、hash、MIME |
| `provenance` | 是 | 提取器、结构规则、OCR/VLM 的来源、版本、配置 hash 和决策记录 |
| `language_code` | 否 | BCP-47/ISO 风格语言标识 |

`source_key` 构造为 `sha256(canonical_source_spans)`：段落合并必须使用保持阅读顺序的 Atom/span 列表；Excel 使用 `sheet_ref + table_ref + cell_range`。KC 用相同算法重新计算并拒绝不匹配的 `evidence_key`。VLM 不得用 `page + local_block_index` 伪造原文身份；视觉生成描述使用其输入图像 Atom 和独立派生类型。同一 Parse View 重试须得到相同 key；配置变更创建新 View，因此允许不同 View 有不同 key。

## Locator JSON

文档类使用 `locator_schema_version="document/v1"`：

```json
{
  "pages": [
    {
      "page_no": 3,
      "bbox": [0.10, 0.22, 0.88, 0.41],
      "coordinate_space": "page_normalized",
      "page_size": {"width": 595, "height": 842}
    }
  ]
}
```

Spreadsheet 使用 `locator_schema_version="spreadsheet/v1"`：

```json
{
  "sheet_name": "Revenue FY26",
  "sheet_ref": "sheet:2",
  "table_ref": "table:2:1",
  "cell_range": "A12:F80",
  "row_start": 12,
  "row_end": 80,
  "column_start": 1,
  "column_end": 6,
  "is_sub_table": true
}
```

Slide 是 document locator 加 `slide_no`；图片额外包含 `image_ref` 和受控 `payload_descriptor`。只有页码而无坐标的解析器可省略 bbox，但必须显式写 `coordinate_space="unavailable"`，不能伪造坐标。

## 从当前实现迁移

| 当前 `ChunkResult`/Docling 字段 | V2 DTO | 必须补齐 |
| --- | --- | --- |
| `content`、`chunk_type`、`chunk_num` | `content/evidence_type/ordinal` | 类型转为 KC 枚举 |
| `hierarchy_path/depth/heading_level` | 同名层级字段 | `section_id` 改确定性 `section_key` |
| `metadata.page_num/bbox/image_name/is_sub_table` | `locator/payload_descriptor` | 全 `page_range`、坐标系/页面尺寸、真实 Sheet/cell range |
| `SemanticNode.self_ref` | `source_item_ref/source_spans/source_key` | 由 Atom IR 显式承接，不依赖旧 Node 透传 |
| `parent_chunk_id` | `parent_evidence_key` | 按 V2 key 算法生成，不引用 V1 UUID |
| `search_helper/doc_summary` | 不传 | KC 确定性生成 `retrieval_text`；摘要可用于 Discovery 输入但非 Evidence |
| V1 `_get_embeddings()` | 不传 | KC `INDEX` Job 对最终 `retrieval_text` 生成 embedding |

`provenance` 至少包含 `extractors` 数组（`DOCLING/OCR/VLM/RULE`）、各自版本或模型 key、源项引用和可选配置 hash。若图片文本混合 OCR 与 VLM，必须拆成可识别部分或在 provenance 中标明文本片段来源，避免最终引用无法说明事实出处。

## 构建与批次校验

新增纯函数 `EvidencePlanner`，输入 Parse View identity、Structure IR 和策略，输出 DTO；它不直接消费旧 Docling 语义树或 ChunkCandidate，也不调用数据库、Embedding 或 HTTP。`KcParseClient` 只按 batch size 发送 DTO，不重新解释内容。

完成时 Worker 生成有序 `output_manifest`：解析工件 URI/hash、Evidence key、source spans hash、content hash、locator hash 和总数。KC 以该清单校验缺批、重批、来源缺失或同 key 内容漂移，再执行质量门并决定是否允许 Parse View 激活。清单是审计和回放产物，不替代 Evidence 表。
