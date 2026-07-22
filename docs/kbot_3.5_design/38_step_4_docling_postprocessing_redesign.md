# 步骤 4 详细设计：Docling 后处理重构

## 目标与边界

Docling 继续作为底层文档转换引擎，负责版面元素、文本、表格、图片与坐标的基础提取。V2 不复用当前 `HierarchyBuilder → SmartChunker` 的结构与切块结果，而是在 Docling 输出之后建立独立、可测试的结构解析与 Evidence 规划流水线。唯一优化目标是提高检索召回、答案支撑和引用定位质量，而不是增加切片数量。

```text
不可变文件
  → Docling Adapter
  → AtomNormalizer
  → ReadingOrderResolver
  → OutlineResolver
  → SemanticBlockBuilder
  → StructureQualityEvaluator
  → EvidencePlanner
  → Parser V2 Manifest
```

KC 仍负责任务、状态、持久化、索引和 Parse View 激活；Parser Worker 负责生成解析工件与候选 Evidence，不直接写 `KBOT_KC_*` 表。

## 当前实现的主要缺口

- `chunk_generator.py` 主要从根标题的直接子节点生成内容，嵌套章节、无标题正文和跨层内容存在漏取风险。
- 标题修复主要修改 `level`，没有基于修复结果重新构建整棵树；新增标题也不能可靠接管其后的正文。
- 质量判断偏页面局部和节点计数，缺少全文编号体系、样式簇、目录一致性、内容覆盖率与树合法性校验。
- Docling、OCR/VLM 与跨页逻辑存在坐标空间混用风险；阅读顺序和跨页连接因此不稳定。
- 短块仅按局部规则向前合并，长块拆分后可能产生短尾；字符数代替 token、缺少精确源跨度，表格和图片上下文容易丢失。
- `search_helper`、全文摘要或 LLM 反思文本混入每个 chunk，会污染可引用原文并放大相似度噪声。
- 当前结构、切块模块缺少覆盖复杂版面和失败文档的 golden tests。

因此 V2 不对旧类做增量打补丁；旧链路在 V1 退役前保持稳定，新流水线以 Docling 原始结果为输入重新实现。

## 四类不可变解析工件

每次 Parse View 生成以下版本化工件，并以 hash、schema 版本和生成器版本登记到 `artifact_manifest_json`：

| 工件 | 作用 |
| --- | --- |
| `raw_docling.json` | 保留底层引擎原始输出，用于回放和问题定位 |
| `atom_ir.json` | 统一元素类型、坐标、阅读序和来源引用 |
| `structure_ir.json` | 保存重建后的章节树、语义块和每次结构决策依据 |
| `evidence_manifest.json` | 保存 Evidence 稳定键、源跨度、内容/定位 hash 与质量结果 |

工件写入不可变对象 URI；数据库只保存清单、版本、hash 与必要的检索投影。`KBOT_KC_PARSE_VIEW.artifact_manifest_json` 已纳入开发期基线迁移，complete 未提交完整清单时不得激活 View。

## 核心设计约束

1. `content` 只包含可从文件定位的提取文本；摘要、关键词、关系和 VLM 推断属于派生字段，不能伪装成原文。
2. 所有标题升级/降级、段落合并、跨页连接和表格上下文继承都必须记录输入 Atom、规则版本、置信度和理由。
3. 坐标在 Adapter 层统一为左上原点、页面归一化空间，同时保留原始 bbox、原点和页面尺寸。
4. 先完成全文结构重建和质量评估，再规划 Evidence；不得由目标 chunk 大小反向篡改章节树。
5. VLM 只处理规则无法确定的局部区域，输入和输出必须引用既有 Atom；需要新增视觉描述时生成独立派生 Evidence，并明确 provenance。
6. 重解析先生成新 View 和全部工件，通过质量门后激活；随后才清理旧非 ACTIVE View，与既定“成功后替换”语义一致。

## Parser 与 KC 的新完成契约

Worker 分批提交 Evidence 后，`complete` 同时提交工件清单、结构质量报告、Evidence 清单指纹及解析器组件版本。KC 校验批次完整性和关键质量门，只接受与 claim 中 `document_version_id/parse_view_id/input_fingerprint` 完全一致的结果。质量失败时 View 保持 `FAILED` 或 `QUALITY_REJECTED`，已有 ACTIVE View 不受影响。

本设计不在步骤 4 引入检索模型或 LLM rerank。结构正确性通过解析 benchmark 验证；最终价值再由步骤 5 的 Discovery/Evidence Recall、答案支撑率和引用准确率验证。
