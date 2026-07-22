# 步骤 4 详细设计：Parser V2 Evidence DTO

## 目标

V2 DTO 是 Parser 与 KC 的唯一内容契约。它既要承接当前 Docling 的正文、表格、图片、层级和 bbox，又要为后续更充分的版面/视觉/Spreadsheet 解析预留稳定表达；KC 将它规范化为 `KBOT_KC_EVIDENCE`，不接收 V1 `TxtChunkEntity`。

## 顶层任务结果

```json
{
  "dto_version": "kc-parser-evidence/v1",
  "parse_view_id": 501,
  "document_version_id": 301,
  "input_fingerprint": "...",
  "parser_version": "docling-worker/3.5.0",
  "policy_fingerprint": "...",
  "evidences": []
}
```

`parse_view_id/document_version_id/input_fingerprint` 必须与 claim 响应一致；Worker 不接受调用方自由指定。大结果拆为多个 batch，但每个 batch 使用同一顶层身份。

## 单个 Evidence DTO

| 字段 | 必填 | 规则 |
| --- | --- | --- |
| `evidence_key` | 是 | `ev1:{parse_view_id}:{source_key}:{fragment_index}:{type}`；不得随机生成 |
| `parent_evidence_key` | 否 | 使用相同算法；父节点未输出时为空 |
| `source_item_ref` | 否 | Docling `self_ref`；无此值时必须有 `source_key` 的确定性结构定位 |
| `source_key` | 是 | 对 `self_ref` 或页/Sheet/表/段落定位的规范化 hash，不能只用 ordinal |
| `fragment_index` | 是 | 同源项分裂的从 0 开始序号 |
| `evidence_type` | 是 | `TEXT/TABLE/IMAGE/SLIDE/CAPTION/CELL_RANGE` |
| `ordinal` | 是 | View 内阅读序；用于展示和邻接，不是身份 |
| `content` | 是 | 可验证文本/Markdown 表格；不含无来源 LLM 事实 |
| `heading_path` | 否 | 标题文本数组；KC 保存为 JSON |
| `section_key`、`hierarchy_depth`、`heading_level` | 否 | 确定性章节归属与层级信息 |
| `locator_schema_version`、`locator` | 是 | 下节定义的可引用定位 |
| `payload_descriptor` | 否 | 图像、表格结构或大对象的受控临时 URI、hash、MIME |
| `provenance` | 是 | 解析器/规则/OCR/VLM 的来源与版本列表 |
| `language_code` | 否 | BCP-47/ISO 风格语言标识 |

`source_key` 的建议构造为 `sha256(canonical_source_ref)`：Docling 项使用 `self_ref`；段落合并使用排序后的源项引用列表；无引用的 VLM Markdown 使用 `page + heading_path + local_block_index`；Excel 使用 `sheet_ref + table_ref + cell_range`。同一 Parse View 重试须得到相同 key；配置变更创建新 View，因此允许不同 View 有不同 key。

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
| `SemanticNode.self_ref` | `source_item_ref/source_key` | 当前必须从 Node 透传到生成结果 |
| `parent_chunk_id` | `parent_evidence_key` | 按 V2 key 算法生成，不引用 V1 UUID |
| `search_helper/doc_summary` | 不传 | KC 确定性生成 `retrieval_text`；摘要可用于 Discovery 输入但非 Evidence |
| V1 `_get_embeddings()` | 不传 | KC `INDEX` Job 对最终 `retrieval_text` 生成 embedding |

`provenance` 至少包含 `extractors` 数组（`DOCLING/OCR/VLM/RULE`）、各自版本或模型 key、源项引用和可选配置 hash。若图片文本混合 OCR 与 VLM，必须拆成可识别部分或在 provenance 中标明文本片段来源，避免最终引用无法说明事实出处。

## 构建与批次校验

新增纯函数 `KcEvidenceBuilder`，输入 Parse View identity、Docling 语义树/ChunkCandidate 和策略，输出 DTO；它不调用数据库、Embedding 或 HTTP。`KcParseClient` 只按 batch size 发送 DTO，不重新解释内容。

完成时 Worker 生成有序 `output_manifest`：Evidence key、content hash、locator hash 和总数的规范化清单。KC 以该清单校验是否存在缺批、重批或同 key 内容漂移，再允许 Parse View 激活。这个清单是重试诊断产物，不替代 Evidence 表。
