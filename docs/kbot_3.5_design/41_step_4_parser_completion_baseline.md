# 步骤 4 实施记录：解析改造完成基线

## 完成范围

Parser 已形成独立、可运行闭环：

```text
KC claim/source URL → Docling conversion → Atom IR → Reading Order
→ Outline/Semantic Blocks → Structure Quality → EvidencePlanner
→ 四类不可变工件 → Evidence batches → KC fingerprint verify → ACTIVE
```

`kbot_app_parser.py` 只启动 `KcParserWorker`，不轮询或写入旧 File/Chunk 表。Worker 持续 heartbeat；租约过期、输入 hash 不符、质量拒绝和临时依赖失败分别使用明确失败语义。

## 实现模块

| 模块 | 职责 |
| --- | --- |
| `converter.py` | PDF/Office/Markdown/CSV/HTML/图片转换；旧 Office 用隔离 LibreOffice profile |
| `docling_adapter.py` | Docling → Atom IR、坐标/逻辑定位、Sheet/cell、派生视觉 Atom |
| `reading_order.py` | 多栏顺序、重复区域隔离、caption 邻接、跨页延续 |
| `structure_builder.py` | 全文标题层级重建、非法跳级修复、Semantic Block |
| `quality.py` | Document/Page/Section/Evidence 指标和硬质量门 |
| `evidence_planner.py` | 短段合并、长文本精确 span、TABLE_ROW、SHEET/CELL_RANGE |
| `pipeline.py` | 生成 raw/Atom/Structure/Evidence Manifest 与统一输出指纹 |
| `kc_client.py` / `kc_worker.py` | claim、source、heartbeat、工件、批次、complete/fail 协议 |

## 格式与定位

- PDF、图片：`document/v1` 页码与归一化 bbox，同时保留原始坐标原点。
- DOC/DOCX、Markdown/TXT、HTML：`document-logical/v1` 的 Docling source refs。
- PPT/PPTX：隔离转换为 PDF 后保留 slide/page 定位。
- XLS/XLSX、CSV：Sheet 与真实 cell range；大表按行组输出子范围并继承表头。
- 图片描述：仅在 Parse View policy 指定 VLM 时生成，独立保存 provenance，不混入来源原文。

## 激活门禁

正文覆盖、树合法性、来源跨度、定位、Evidence key、token 上限、四类工件和输出指纹任一不一致时，不激活 Parse View。新 View 失败不会替换已有 ACTIVE View。KC 根据实际工件 hash 和 STAGED Evidence 顺序复算输出指纹，不信任 Worker 声明值。

重解析成功时，KC 在激活事务中删除同 Version/View Kind 的旧 View 与 Evidence，提交后清理旧工件；最终失败的候选结果同样删除。TRANSIENT 重试保留同一候选 View 和确定性 Evidence key，避免重复结果。

自动化测试覆盖 IR 不变量、多栏/跨页、长短文本、表格行组、Excel、视觉描述、工件幂等和 Worker 完整协议。真实 smoke test 已验证 DOCX 的逻辑定位和 PDF 的页/bbox 定位。生产阈值仍需用既定 golden corpus 调优，但不再改变 Parser/KC 契约或模块边界。

离线 corpus 使用 `tests/fixtures/kc_parser/golden_manifest.example.json` 的格式，并通过以下命令生成逐文件质量报告：

```bash
python scripts/evaluate_kc_parser.py /path/to/golden_manifest.json \
  --artifacts-path /path/to/docling_models --parser-version candidate-1
```
