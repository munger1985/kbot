# Knowledge Core

## 边界与数据模型

Knowledge Core（KC）是封闭的知识生成与检索服务。Domain 是强隔离边界，一个
Domain 可有多个 Collection；App 私有 Agent 通过 Execution Spec 关联多个平权
Collection。Collection 之间可建立关系，但不会重复保存 Domain 身份。

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
上传后停在待审批状态，批准后才创建解析任务；`KM_ASSET` 已代表 MetaDB 发布资产，
可自动进入解析。文件先写入 staging，数据库事务成功后发布到正式对象目录；失败
会清理临时文件、空目录和未形成有效 Revision 的空 Bundle。

Bundle、Revision、Document Version 均保留不可变事实。重解析创建新的 Parse
View，成功激活后删除旧解析产物；失败时继续保留上一成功视图。

### KM Asset 元数据的问文与问数边界

KM Asset App 从 Asset MetaDB 读取完整业务元数据和全部附件，并通过 KBot 内部服务身份
调用 KC 的 `km-assets` 入库接口。外部 Portal 不得直接调用 KC 内部接口。元数据保存在不可变
`Bundle Revision.manifest_json.metadata` 中，并进入 Manifest 与 Discovery Profile，
用于标题、主题、行业、解决方案等语义检索。

`KBOT_V_KM_ASSET_CURRENT` 保留当前同步状态用于运维；所有用户搜索统一从
`KBOT_V_KM_ASSET_SEARCHABLE` 读取。后者只投影 Asset 和当前 Revision 均为 `READY`、
Bundle 与 Bundle Revision 完整且一致的记录。FAILED、处理中和映射不完整的 Asset
不能参与问文、问数或混合搜索。

Root 只生成一次 `AssetSearchPlan.v1`，完整表达多个硬条件、布尔关系、软偏好、排序和
数量。精确元数据条件由确定性编译器转换为 `DataQueryPlan.v1`，Data Query 继续执行
模型目录校验、Domain 强制范围和参数化 SQL，不接受自由 SQL，也不调用第二个 LLM
重新规划。语义条件先从可搜索视图形成候选范围，再由 Knowledge Core 验证正文相关性。
语义召回 Top-K 不作为总数；主题总数请求返回明确的能力说明和最多 5 个较新相关 Asset。

所有成功回答都投影 Asset：精确清单逐项使用 `Qn`；文档回答使用同 Bundle 的 `Cn`；
混合结果逐项同时使用资格集合 `Qn` 和正文 `Cn`。精确 COUNT/GROUP 在同一个可审计
查询快照中返回聚合结果 `Q1` 和同条件下 3 至 5 个较新 Asset 样例 `Q2`。

## 自适应解析

Docling 是底层转换引擎，其后由 KC 自有流水线完成：

1. 页面探测和文本覆盖率评估；
2. `TEXT/AUTO/VISUAL/HYBRID` 路径选择；
3. 可选 Docling OCR 或独立 DeepSeek OCR；
4. 可选 VLM 页面理解和结构校正；
5. Atom IR、Structure IR 与标题层级修复；
6. 表格、子表、短块合并和上下文补全；
7. 质量评分与 Evidence 规划；
8. 可选视觉 Embedding 生成。

VLM 和视觉 Embedding 均为可选能力。缺少 VLM 时跳过视觉转文字；缺少视觉模型
时跳过以图搜图，不影响纯文本 Evidence。Collection 的文本 Embedding 模型唯一，
解析索引与查询必须使用同一模型身份，不能只因维度相同而混用。

## Projection 与索引

`PARSE → INDEX → PROFILE → INDEX(DISCOVERY)` 由持久化 Ingestion Job 驱动。
Evidence 保存正文、检索文本、层级、页码、坐标、来源跨度、质量、向量及模型快照。
Discovery Object 保存 Bundle/Document 检索画像。Oracle Text 为
`RETRIEVAL_TEXT` 和 `PROFILE_TEXT` 建立 `CONTEXT` 索引并在提交时同步。

## 二阶段检索

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
