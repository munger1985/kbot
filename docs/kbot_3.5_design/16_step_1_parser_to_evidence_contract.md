# 步骤 1 详细设计：Parser 到 Evidence 契约

## 审核结论

当前 Docling 链路只用于确认 V1 已具备的字段下限；它不是 KC 表结构的上限，也不应成为 V2 的长期契约。后续 Parser 将以更充分的结构、版面、视觉和表格解析提升召回质量。KC Worker 应接收版本化的解析产物 DTO，写入、计算哈希、生成向量和 View 激活仍只能由 KC 完成。

## 当前字段映射

| KC Evidence | 当前 Docling/Parser 输出 | 处理结论 |
| --- | --- | --- |
| `content`、`evidence_type`、`ordinal` | `content`、`chunk_type`、`chunk_num` | 可直接映射；类型须收敛为 KC 枚举。 |
| 标题路径与层级 | `hierarchy_path`、`hierarchy_depth`、`heading_level` | 可直接映射，保留为独立字段。 |
| 页码、bbox、图片名、子表 | `ChunkMetadata` | 可生成基础定位；须升级为标准 `locator_json`。 |
| `payload_uri` | `engine.py` 导出图片，Chunk 携带 `image_name` | 可生成，但 Parser 必须返回受控对象 URI/资源描述，不能只给本地文件名。 |
| `embedding`、模型标识 | `_get_embeddings()` 使用 `txt_embed_model` | 可生成，但 V2 必须对 KC 重建的 `retrieval_text` 嵌入。 |
| `security_level` | `FileParams.security_level` | 由 KC 从 Document Version 派生，非 Parser/客户端权威输入。 |
| `content_hash`、`token_count` | 当前未输出 | KC Worker 对规范化内容计算。 |
| `quality_score` | 当前无统一单元质量分 | 先可空；后续由规则/评测器填充。 |

## 面向充分解析的 DTO 字段

- `source_item_ref`：透传 Docling `item.self_ref`；一个源项被切分时同时输出 `fragment_index`。当前 `SemanticNode` 已保留 `self_ref`，但 `ChunkResult` 丢失它。
- `parent_evidence_key`、`section_key`：当前已有 `parent_chunk_id`、`section_id` 的雏形；V2 改为确定性键。Excel 当前随机生成 `section_id`，不可用于幂等。
- `page_start/page_end` 与 `locator_json`：当前通用 Chunk 只保留 `page_range[0]`。须输出全部页及每页 bbox、坐标系和页面尺寸。
- Spreadsheet 定位：输出 `sheet_name`、稳定的 `table_ref`、`row_start/row_end`、`column_start/column_end`、`cell_range` 和 `is_sub_table`。当前仅将 `page_no` 当作 Sheet 序号，无法可靠引用单元格。
- `provenance_json`：图片区块当前可能混合 OCR 和 VLM 描述；须标记每一内容部分的提取方法、模型/配置摘要和源项引用，避免把生成描述误作原始文本。
- `language_code`：可由 Worker 语言识别补齐，不阻塞首期解析。

这些字段并非要求当前 Docling 一次性全部生成：它们是 KC 的稳定承载面。解析器可随 Parse View 的 `parser_version` 与 `policy_snapshot` 演进，逐步补充新的定位、结构和多模态信息，而无需再次改动 Evidence 的主模型。

## V2 生成顺序

```text
Docling / OCR / VLM → Parser Evidence DTO（含确定性 evidence_key、定位与来源）
  → KC 校验与规范化 → 计算 content_hash、retrieval_text、token_count
  → 嵌入 → STAGED Evidence → Parse View 质量门 → ACTIVE
```

`retrieval_text` 不复用当前 `search_helper`：其包含 `doc_summary` 和内容截断，且摘要可能由 LLM 生成。V2 以可追溯的结构化解析结果重建检索文本：确定性拼接 MIME、类型、标题路径、稳定结构标签和完整 Evidence 内容，并在有明确来源标记时纳入 OCR、表格标题/列名、图片描述等补充信息。Bundle/Revision 的标题、Facet、角色在检索时动态加入排序和引用上下文。

## 验收样例

- Word/PDF 跨页段落：引用返回完整页范围，非仅起始页；bbox 标明坐标系。
- 图片：原始 OCR 与 VLM 描述可区分，`payload_uri` 可访问且引用能回到原页。
- Excel 拆分子表：每个 Evidence 返回真实 Sheet 名与 `A12:F80` 一类范围；重跑同一 View 的键不漂移。
- 同一 Parse View 重试：按 `evidence_key` 幂等 upsert，不产生重复 Evidence。
