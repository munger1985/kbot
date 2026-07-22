# 3.5 解析、结构化提取与 Excel

## Parser Worker 协议

KC 写入 Bundle、Document Version 与 PENDING Job 后，Parser 以能力和容量领取任务：

- `POST /internal/v2/knowledge/parse-tasks/claim`：按 `worker_id`、格式能力、OCR/VLM 能力领取有限租约。
- `POST /internal/v2/knowledge/parse-tasks/{task_id}/heartbeat`：长任务续租。
- `POST /internal/v2/knowledge/parse-tasks/{task_id}/result`：回传解析结果、Evidence 批次、质量、parser/model 版本或错误。

KC 校验租约后，在受控事务中应用结果、更新 Document 状态、重建 Discovery 并汇总 Bundle 状态。Parser 不能轮询 `kbot_md_kb_files`、创建 Bundle 或直接写 `KBOT_KC_*` 表。大批量结果使用受限批次或临时结果 URI。

## 多视图解析

同一 Document Version 可拥有 TEXT、VISUAL、HYBRID 等不同用途的 Parse View。原生文本和可靠标题树优先 TEXT；扫描、多栏或复杂图文可采用 VISUAL；局部质量差或关键表格可使用 HYBRID。View 必须保存范围、质量、配置、解析器和模型版本；同一 `view_type + coverage_key` 的重解析采用候选构建、成功后原子替换、删除旧产物的流程，召回只使用 Active View，避免重复证据和旧结果残留。详见[Parse View 生命周期](14_step_1_parse_view_reparse_lifecycle.md)。

## Excel 的双表示

Excel 首先是 KC 的 Document Version，不在入库时二选一为“问文”或“问数”。`SPREADSHEET` Parse View 产出：

1. Evidence：Sheet/表/子表标题、说明、关键行、区域定位与上下文，支持定位、解释和总结；VLM 负责版式和语义增强。
2. `structured_artifact_uri`：规范化的 Sheet/Table/Column/Row 工件，建议存为 Parquet、Arrow 或 JSON。数值计算只以单元格值、公式计算值和字段类型为准，不能使用 VLM 描述作为数值事实。

```text
SPREADSHEET Parse View
  ├─ Evidence ──► Knowledge Core 问文
  └─ structured_artifact_uri ──► Data Query File Dataset（按需注册）
```

按意图路由：定位、解释、总结走 KC；筛选、聚合、分组、排序、同比环比走 Data Query；混合问题并行执行。不要默认把 Excel 物化进 Oracle 业务库。后续 Data Query 将工件注册为只读 `file_derived` Dataset，在隔离引擎查询（如 DuckDB 读取 Parquet），并绑定 `document_version_id + parse_view_id`、继承权限、审计 SQL/计划和结果。跨文件长期分析才进入受治理数仓。

3.5 交付 `SPREADSHEET` View、工件 URI、表清单、字段推断、解析置信度和 Evidence；不交付完整 File Dataset/NL2SQL。
