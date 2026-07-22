# 步骤 4 详细设计：Parser 改造为 KC 租约 Worker

## 改造边界

`kbot_app_parser.py` 现在只启动 KC Worker，通过 `/internal/v2/knowledge/parse-tasks/*` claim 租约、读取不可变 Document Version、回传解析工件和 Evidence DTO；KC 是唯一 KC 表写入者。旧 `FileParseEngine → ChunkResult → TxtChunkEntity` 不再是运行路径。

Worker 不读取 `kbot_md_kb_files`、`KBOT_BIZ_TXT_CHUNK`、V1 KB/Domain 参数，也不调用 V1 `FileRepository`、`TxtChunkRepository`。

## 模块改造映射

| 当前模块 | V2 改造 |
| --- | --- |
| `kbot_app_parser.py` | 只启动 KC Worker Loop 和优雅停机，不启动文件表轮询。 |
| `knowledge_core/parsing/pipeline.py` | 纯 `parse(document, policy) -> ParserOutput`，不调用数据库、Embedding 或 KC HTTP。 |
| `parser_schema.py:ChunkResult` | 不进入新链路；契约由 Atom IR、Structure IR、Evidence DTO 和完成清单定义。 |
| `engine.py` / Docling | 仅承担基础转换并保留原始结果；视觉工件写 Worker 受控临时 URI，不写 V1 `KBOT_MD_EXTRACTED_IMAGES`。 |
| `chunk_generator.py` / `hierarchy_builder.py` / `chunk_reflector.py` | 不进入运行链路。由 `AtomNormalizer → ReadingOrderResolver → OutlineResolver → SemanticBlockBuilder → StructureQualityEvaluator → EvidencePlanner` 替代。 |
| Spreadsheet Adapter | 输出真实 Sheet/表/行列/单元格范围、原始值/显示值和规范化数据工件。 |
| 新 `kc_parse_client.py` | 封装 claim、heartbeat、Evidence batch、complete/fail；不包含业务解析规则。 |

## Worker Loop

```text
启动：声明 worker_id、MIME/View 能力、并发槽位、软件版本
  ↓
claim PARSE Job（KC 返回短期源读取 URI + Parse View policy snapshot）
  ↓
下载不可变内容到 Worker 临时目录 → Docling 转换 → Atom/Structure IR → 周期 heartbeat
  ↓
发布不可变解析工件 → 按大小/数量分批回传 Evidence DTO（STAGED）
  ↓
complete（工件清单、输出指纹、质量报告、组件版本）或 fail（受控分类）
  ↓
清理本地/临时视觉工件；继续 claim
```

每个任务单独持有取消信号和临时目录。优雅停止时 Worker 停止 claim，尝试完成正在发送的短批次；不能伪造 complete 或主动释放其他 Worker 的租约。心跳失败、`JOB_STALE` 或 `JOB_LEASE_INVALID` 时立即终止解析并清理本地资源。

## V2 Parser 产物规则

- `evidence_key` 由 `parse_view_id + hash(canonical_source_spans) + fragment_index + evidence_type` 构造；父子关系使用同一规则的 `parent_evidence_key`。
- `content` 保留可验证解析文本；每个 Evidence 携带 Atom/span 来源。OCR、VLM、规则抽取区分写入 provenance，不把 LLM 文档摘要或 cross-reference 混入内容。
- 文档定位输出 `pages[]`、页范围、bbox、坐标系/页面尺寸；当前仅保留起始页的行为必须移除。
- Excel `SPREADSHEET` View 输出真实 `sheet_name/table_ref/cell_range/row_start/row_end/column_start/column_end`，并生成受控 `structured_artifact_uri`；不得以 Sheet 序号冒充页码。
- Parser 不生成最终 KC `retrieval_text`、不决定 Evidence 状态、也不切换 Parse View；向量生成由 KC 的 `INDEX` 流程按其确定性检索文本执行。
- Parser 必须提交 `raw_docling/atom_ir/structure_ir/evidence_manifest` 的 URI、hash 和 schema/generator 版本；具体结构见 38–40 文档。

## 失败、重试与测试

Worker 仅分类上报：`TRANSIENT`（模型限流、短暂网络/存储）、`PERMANENT`（损坏、加密或不支持文件）、`POLICY`（安全/配额拒绝）。KC 决定 Job 重试、Member 状态和 Revision 汇总。任何 V2 回调都携带 `job_id/lease_owner/input_fingerprint`，迟到回调不得重新激活旧 View。

测试分四层：Atom/Structure IR 的 golden tests；Evidence 稳定键、来源跨度、定位和 Spreadsheet 测试；Client 协议 mock 测试；KC 集成测试验证租约过期、重复 batch、质量拒绝、Parser 中断和重解析替换。验收要求 V2 Parser 的依赖、运行日志和 SQL 查询中均不出现 V1 File/Chunk 链路。
