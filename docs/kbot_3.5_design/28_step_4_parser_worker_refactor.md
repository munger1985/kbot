# 步骤 4 详细设计：Parser 改造为 KC 租约 Worker

## 改造边界

当前 Parser 由 `kbot_app_parser.py` 启动，`microservices/file_processor/services/file_processor.py` 轮询/更新 V1 文件状态，并将 `ChunkResult` 映射到 `TxtChunkEntity` 和旧表。V2 不修改或复用该入库路径：新增 KC Worker 运行模式，通过 `/internal/v2/knowledge/parse-tasks/*` claim 租约、读取不可变 Document Version、回传 Evidence DTO；KC 是唯一 KC 表写入者。

V1 Parser 进程与 V2 Parser Worker 可在过渡期分别部署。V2 Worker 不读取 `kbot_md_kb_files`、`KBOT_BIZ_TXT_CHUNK`、V1 KB/Domain 参数，也不调用 V1 `FileRepository`、`TxtChunkRepository`。

## 模块改造映射

| 当前模块 | V2 改造 |
| --- | --- |
| `kbot_app_parser.py` | 增加 `parser_mode=v1|kc_v2` 启动选择；V2 启动 Worker Loop、能力注册和优雅停机，不启动 V1 文件轮询。 |
| `microservices/file_processor/services/file_processor.py` | 保留 Docling/OCR/VLM 解析核心；拆出纯 `parse_document(input, policy) -> ParserEvidenceDTO[]`，移除 V2 路径中的 `_save_chunks()`、`_get_embeddings()` 与 V1 状态更新。 |
| `parser_schema.py:ChunkResult` | 演进为版本化 V2 DTO：稳定 `evidence_key`、父键、源项引用、完整页范围、标准定位和 provenance。旧 DTO 可为 V1 保留，不互相映射。 |
| `chunk_generator.py` / `hierarchy_builder.py` | 透传 `self_ref`、全部 `page_range`、每页 bbox、章节稳定键；Spreadsheet 输出真实 Sheet/表/行列/单元格范围。 |
| `engine.py` | 视觉工件先写 Worker 受控临时 URI，并以 `payload_descriptor` 回传；不写 V1 `KBOT_MD_EXTRACTED_IMAGES`。 |
| 新 `kc_parse_client.py` | 封装 claim、heartbeat、Evidence batch、complete/fail；不包含业务解析规则。 |

## Worker Loop

```text
启动：声明 worker_id、MIME/View 能力、并发槽位、软件版本
  ↓
claim PARSE Job（KC 返回短期源读取 URI + Parse View policy snapshot）
  ↓
下载不可变内容到 Worker 临时目录 → 解析 → 周期 heartbeat
  ↓
按大小/数量分批回传 Evidence DTO（STAGED）
  ↓
complete（输出指纹、质量报告、版本）或 fail（受控分类）
  ↓
清理本地/临时视觉工件；继续 claim
```

每个任务单独持有取消信号和临时目录。优雅停止时 Worker 停止 claim，尝试完成正在发送的短批次；不能伪造 complete 或主动释放其他 Worker 的租约。心跳失败、`JOB_STALE` 或 `JOB_LEASE_INVALID` 时立即终止解析并清理本地资源。

## V2 Parser 产物规则

- `evidence_key` 由 `parse_view_id + source_item_ref（或确定性结构定位）+ fragment_index + evidence_type` 构造；父子关系使用同一规则的 `parent_evidence_key`。
- `content` 保留可验证解析文本；OCR、VLM、规则抽取的文本区分写入 `provenance_json`。不把 LLM 文档摘要混入 Evidence 内容或检索文本。
- 文档定位输出 `pages[]`、页范围、bbox、坐标系/页面尺寸；当前仅保留起始页的行为必须移除。
- Excel `SPREADSHEET` View 输出真实 `sheet_name/table_ref/cell_range/row_start/row_end/column_start/column_end`，并生成受控 `structured_artifact_uri`；不得以 Sheet 序号冒充页码。
- Parser 不生成最终 KC `retrieval_text`、不决定 Evidence 状态、也不切换 Parse View；向量生成由 KC 的 `INDEX` 流程按其确定性检索文本执行。

## 失败、重试与测试

Worker 仅分类上报：`TRANSIENT`（模型限流、短暂网络/存储）、`PERMANENT`（损坏、加密或不支持文件）、`POLICY`（安全/配额拒绝）。KC 决定 Job 重试、Member 状态和 Revision 汇总。任何 V2 回调都携带 `job_id/lease_owner/input_fingerprint`，迟到回调不得重新激活旧 View。

测试分三层：纯解析 DTO 的稳定键/定位/Spreadsheet 测试；Client 的 claim/批次/complete/fail 协议 mock 测试；KC 集成测试验证租约过期、重复 batch、Parser 中断和重解析替换。验收要求 V2 Parser 的运行日志和 SQL 查询中不出现 V1 File/Chunk 表。
