# 步骤 1 详细设计：Parser 到 Evidence 契约

## 审核结论

当前 Docling 链路只用于识别 V1 的数据来源和缺陷，不作为 V2 DTO 的直接映射基础。Docling 保留为底层转换引擎；其后由 Atom IR、Reading Order、Structure IR、质量评估和 EvidencePlanner 重新生成版本化解析产物。写入、计算哈希、生成向量和 View 激活仍只能由 KC 完成。

## 当前字段映射

| KC Evidence | 当前 Docling/Parser 输出 | 处理结论 |
| --- | --- | --- |
| `content`、`evidence_type`、`ordinal` | `content`、`chunk_type`、`chunk_num` | 仅用于回归对照；V2 由 Structure IR 重新规划。 |
| 标题路径与层级 | `hierarchy_path`、`hierarchy_depth`、`heading_level` | 当前树可能漏内容或只改 level 不重建，不能直接迁移。 |
| 页码、bbox、图片名、子表 | `ChunkMetadata` | 仅有基础信息且坐标空间可能混用；在 Atom Adapter 层统一。 |
| `payload_uri` | `engine.py` 导出图片，Chunk 携带 `image_name` | 可生成，但 Parser 必须返回受控对象 URI/资源描述，不能只给本地文件名。 |
| `embedding`、模型标识 | V1 `_get_embeddings()` 使用 `txt_embed_model` | V2 Parser 禁止生成或提交；后续 INDEX 阶段使用 Collection 唯一绑定模型。 |
| `security_level` | `FileParams.security_level` | 由 KC 从 Document Version 派生，非 Parser/客户端权威输入。 |
| `content_hash`、`token_count` | 当前未输出 | KC 对规范化内容计算。 |
| `quality_score` | 当前无统一单元质量分 | V2 必须输出分层质量报告，Evidence 分数可作为其中投影。 |

## 面向充分解析的 DTO 字段

- `source_spans`：透传一个或多个 Atom ID、可选字符/单元格跨度与 locator；单源项时可同时输出 `source_item_ref`。它是可追溯和稳定键的权威输入。
- `parent_evidence_key`、`section_key`：当前已有 `parent_chunk_id`、`section_id` 的雏形；V2 改为确定性键。Excel 当前随机生成 `section_id`，不可用于幂等。
- `page_start/page_end` 与 `locator_json`：当前通用 Chunk 只保留 `page_range[0]`。须输出全部页及每页 bbox、坐标系和页面尺寸。
- Spreadsheet 定位：输出 `sheet_name`、稳定的 `table_ref`、`row_start/row_end`、`column_start/column_end`、`cell_range` 和 `is_sub_table`。当前仅将 `page_no` 当作 Sheet 序号，无法可靠引用单元格。
- `provenance_json`：图片区块当前可能混合 OCR 和 VLM 描述；须标记每一内容部分的提取方法、模型/配置摘要和源项引用，避免把生成描述误作原始文本。
- `language_code`：可由 Worker 语言识别补齐，不阻塞首期解析。

这些字段是 V2 激活的最低契约，而非对当前 V1 Chunk 的兼容要求。解析器可随 Parse View 的 `parser_version` 与 `policy_snapshot` 演进，但内容覆盖、来源跨度、标准定位和工件 hash 不允许为空或以后补录。

## V2 生成顺序

```text
Docling → Atom IR → Reading Order → Structure IR → Quality → EvidencePlanner
  → Parser 工件清单与 Evidence DTO（含确定性 key、source spans、定位）
  → KC 校验与规范化 → 计算 content_hash、retrieval_text、token_count
  → STAGED Evidence → Parse View 质量门 → ACTIVE → 独立 INDEX Job 生成唯一文本向量
```

`retrieval_text` 不复用当前 `search_helper`：其包含 `doc_summary` 和内容截断，且摘要可能由 LLM 生成。V2 以可追溯的结构化解析结果重建检索文本：确定性拼接 MIME、类型、标题路径、稳定结构标签和完整 Evidence 内容，并在有明确来源标记时纳入 OCR、表格标题/列名、图片描述等补充信息。Bundle/Revision 的标题、Facet、角色在检索时动态加入排序和引用上下文。

## 验收样例

- Word/PDF 跨页段落：引用返回完整页范围，非仅起始页；bbox 标明坐标系。
- 图片：原始 OCR 与 VLM 描述可区分，`payload_uri` 可访问且引用能回到原页。
- Excel 拆分子表：每个 Evidence 返回真实 Sheet 名与 `A12:F80` 一类范围；重跑同一 View 的键不漂移。
- 同一 Parse View 重试：按 `evidence_key` 幂等 upsert，不产生重复 Evidence。
- 标题误识别/短章节样本：Structure IR 保留完整正文、修复理由与置信度，Evidence 不出现无上下文短块。

详细 IR schema、质量门和规划算法见 38–40 文档。
