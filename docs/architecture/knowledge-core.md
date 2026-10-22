# Knowledge Core

## 边界与数据模型

Knowledge Core（KC）是封闭的知识生成与检索服务。Domain 是强隔离边界，一个
Domain 可有多个 Collection；App 私有 Agent 通过 Execution Spec 关联多个平权
Collection。Collection 之间可建立关系，但不会重复保存 Domain 身份。

KC 与 Agent Runtime 共同构成 [Knowledge Answering 基础能力](knowledge-answering-capability.md)。KC 不拥有
业务会话、发布态或回答，只接受服务端签发的 Agent 身份和精确范围，并返回可验证的 Citation Pack。

核心聚合如下：

```text
Collection
  └─ Bundle（来源业务对象）
       └─ Bundle Revision（一次不可变修订）
            ├─ Revision Document ── Document ── Document Version
            │                                      └─ Parse View
            │                                           ├─ Evidence
            │                                           └─ Visual Asset
            └─ Discovery Object
```

精确表、约束和索引以 `database/oracle/knowledge_core/` 为准。所有业务 ID 使用
UUIDv7，并以 Oracle `RAW(16)` 保存。

## 入库与审批

普通用户可选择“每个文件独立 Bundle”或“多文件组成一个 Bundle”。`USER_UPLOAD`
上传后停在待审批状态，批准后才创建解析任务。文件先写入 staging，数据库事务成功后发布到正式对象目录；失败
会清理临时文件、空目录和未形成有效 Revision 的空 Bundle。

Bundle、Revision、Document Version 均保留不可变事实。重解析创建新的 Parse
View，成功激活后删除旧解析产物；失败时继续保留上一成功视图。

## 自适应解析

Docling 是底层转换引擎，其后由 KC 自有流水线完成：

1. Docling 文本、结构和图片对象提取；
2. `TEXT/AUTO/VISUAL` 路径选择；
3. 按 Collection 的 `parser_ocr` 绑定调用 Model Serving OCR；
4. AUTO 图片对象理解，或 VISUAL 异步整页增强；
5. Atom IR、Structure IR 与标题层级修复；
6. 表格、子表、短块合并和上下文补全；
7. 质量评分与 Evidence 规划；
8. 可选视觉 Embedding 生成。

Collection 的解析角色统一为 `parser_ocr`、`parser_vlm`、`parser_llm`。`AUTO` 必须配置
`parser_ocr`，`VISUAL` 必须配置 `parser_vlm`；`parser_llm` 只供解析期结构化提炼或 App
预分析使用，不承担最终问答。VLM 和视觉 Embedding 均为可选能力。缺少 VLM 时跳过视觉转文字；缺少视觉模型
时跳过以图搜图，不影响纯文本 Evidence。Collection 的文本 Embedding 模型唯一，
解析索引与查询必须使用同一模型身份，不能只因维度相同而混用。

模型绑定在创建解析任务时冻结到 Parse View，后续改绑只影响新任务；旧资料必须显式重解析。
Worker 记录实际模型 revision、失败页和 OCR Artifact，不保存 API Key、Token 或其他凭据。

## Projection 与索引

`PARSE → INDEX → PROFILE → INDEX(DISCOVERY)` 由持久化 Ingestion Job 驱动。
VISUAL 在基础版本发布后追加 `VISUAL_ENRICH → INDEX`；增强失败不会回滚基础版本。
Evidence 保存正文、检索文本、层级、页码、坐标、来源跨度、质量、向量及模型快照。
Discovery Object 保存 Bundle/Document 检索画像。Oracle Text 为
`RETRIEVAL_TEXT` 和 `PROFILE_TEXT` 建立 `CONTEXT` 索引并在提交时同步。

## 二阶段检索

Discovery 请求必须使用显式 `bundle_revision_ids`，并同时校验 Domain、Agent—Collection Binding、
Collection 与 Revision 归属。超过单次 1000 个 Revision 时，调用方按稳定 UUID 排序分片；KC 对各分片
使用相同查询计划和评分语义，合并使用稳定 tie-break。业务 Registry 不得在分片前按创建或更新时间截断。

第一阶段面向文件对象：

- 对问题分词并生成 Oracle Text `ACCUM` 查询；
- 检索 Discovery Profile；Profile 未覆盖正文词时，从 Evidence 全文索引桥接回
  所属 Document/Bundle；
- 使用 Collection 唯一 Embedding 模型检索 Discovery 向量；
- 按 Bundle 聚合、RRF 融合并保证 Collection 平权；
- 使用 RRF 对 Bundle/Document 候选进行确定性融合排序。

第二阶段只在候选 Bundle 内检索 Evidence：

- 全文与向量并行召回；
- 去重、锚点选择和相邻上下文扩展；
- 将同一 Bundle 的 Evidence Group 聚合为一个引用候选；
- 生成 Citation Pack，保留页码、定位器、Document 和 Bundle 身份。

全文或向量任一通道异常时保留另一通道结果；只有全部通道失败才终止请求。最终
回答只返回模型实际引用的文档级 Reference，不把所有召回 Chunk 直接暴露给用户。
