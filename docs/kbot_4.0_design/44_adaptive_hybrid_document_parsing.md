# KC 自适应混合文档解析

本文中的整页视觉、图片描述和 DeepSeek OCR Prompt 在实现 Prompt Registry
后统一迁入 `configuration/prompts.toml`。KC 配置只引用稳定 Prompt Key，
运行时数据库版本优先、文件版本兜底，并把 Prompt Version/Hash 写入 Parse
View 和解析 Artifact。详见
[46_versioned_prompt_registry.md](46_versioned_prompt_registry.md)。

## 目标

4.0 不再维护“Docling Chunk”和“纯视觉 Chunk”两套结果。Parser Worker 统一输出 Atom IR、Structure IR、Evidence 和可审计产物；视觉模型只作为页面级解析器、结构校正器和失败兜底。

```text
Docling 布局转换
  → Docling 内置 OCR 或独立 DeepSeek OCR
  → 页面质量评估
  → 可选整页 VLM Markdown
  → 页面替换或结构融合
  → Atom IR
  → Reading Order / Structure IR
  → Quality Gate
  → Evidence
```

## OCR 提供方

默认使用 Docling 内置 Tesseract/EasyOCR。配置 `parse_policy.ocr_model` 后，Parser 关闭 Docling 内置 OCR，保留其布局转换和页面渲染，再直接调用独立 DeepSeek OCR 的 OpenAI 兼容端点。DeepSeek OCR 不登记到 `AI_MODEL`，不经过 Model Serving，也不由模型池管理。

独立 `[dsocr]` 配置与 3.5 保持一致：

```toml
[dsocr]
enabled = true
api_endpoint = "http://deepseek-ocr:18097/v1/chat/completions"
timeout = 600
crop_mode = true
max_tokens = 8192
temperature = 0.0

[parse_policy]
ocr_model = "deepseek-ocr-2"
```

Parser 对缺少可提取文本的页面发送 grounding prompt，将返回的 `text/title/table_caption/table/image/code/formula` 及 0～999 坐标直接转换为 OCR Atom。嵌入图片可单独执行 `Parse the figure.`，并作为 OCR 派生描述进入 Atom IR。成功的整页 OCR 替换该页 Docling 低质量结果；单页 OCR 失败则保留 Docling 结果。

## 策略

| 策略 | 行为 | 适用场景 |
| --- | --- | --- |
| `TEXT` | 仅 Docling/OCR | 结构稳定、成本敏感 |
| `AUTO` | 只对低质量页调用 VLM | 默认策略 |
| `VISUAL` | 所有有页面图像的页面由 VLM 重建 | 扫描件或已知视觉解析更优的 PDF |
| `HYBRID` | 所有页面调用 VLM；健康页校正结构，低质量页替换 | 高质量知识资产 |

`AUTO` 当前使用三个确定性信号：页面有效文本字符数、Docling/OCR 平均置信度和乱码比例。阈值由 Knowledge Core 的 `[parse_policy]` 配置，并进入 `parse_config_fingerprint`，修改策略会生成新的不可变 Parse View。

## 融合规则

低质量页把 VLM Markdown 转换为标题、段落、列表、表格和视觉描述 Atom，并使用页面级 locator 替换该页的低质量 Docling Atom。与视觉块精确匹配的 Docling/DeepSeek OCR Atom 优先保留其原文和 bbox；未匹配的 DeepSeek OCR Atom 始终作为精确文字证据保留，其他未匹配但高置信且包含数字，或属于表格、公式、代码的 Atom 也作为校验证据保留。健康页保留 Docling 文本和 bbox；VLM 标题与原文匹配时只升级标题类型和层级，图表描述作为独立 `VISUAL_DESCRIPTION` Atom，其余重复转写不进入 Evidence。

VLM 结果不得伪造精确 bbox。整页视觉 Atom 使用归一化全页坐标，并在 provenance 中记录 `locator_precision=PAGE`、模型、选择原因和融合用途。DeepSeek OCR grounding 坐标标记为 `coordinate_space=grounding_0_999`；Docling 原始输出始终保存在 `raw_docling`，OCR 与视觉过程分别保存为可选 `deepseek_ocr_analysis`、`visual_analysis` 产物。

## 失败语义

- 单页 VLM 超时、空结果或异常：记录失败页并继续使用 Docling；
- 单页 DeepSeek OCR 超时、空结果或异常：记录失败页并继续使用 Docling；
- 配置 `ocr_model` 但未启用 `[dsocr]`：配置加载失败；
- `AUTO` 未配置 VLM：退化为 `TEXT`，不是任务失败；
- `VISUAL/HYBRID` 未配置 VLM：解析配置无效，任务失败；
- 融合后的 IR 仍必须通过 Atom 覆盖、Evidence 来源、定位和长度质量门。

## 验收重点

正式数据集至少覆盖原生 PDF、扫描 PDF、双栏排版、标题误判、复杂表格、架构图和图文混排。分别统计页面选择准确率、标题层级准确率、表格完整率、Evidence Recall、引用定位完整率、VLM 失败回退率和单页成本；不能只以“生成了文本”作为成功标准。
