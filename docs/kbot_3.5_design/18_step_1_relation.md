# 步骤 1 详细设计：Relation

## 定位与边界

`KBOT_KC_RELATION` 保存同一 Bundle 内、可回溯且可用于检索扩展的语义关系。它不是通用外键镜像：Revision Member 已表示“Revision 包含附件”，Evidence 的 `parent_evidence_id` 已表示章节/图文层级，二者不得重复写为 Relation。

3.5 不建立跨 Collection 或跨 Domain Relation；也不把未经依据约束的 LLM 猜测写为 ACTIVE 关系。

## 表字段

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `relation_id` | `NUMBER(38)` PK identity | 关系标识 |
| `collection_id`, `bundle_id`, `bundle_revision_id` | 非空 `NUMBER(38)` | Scope、所属业务对象和关系有效的来源快照 |
| `subject_type`, `subject_id` | 非空 `VARCHAR2(32)` + `NUMBER(38)` | 主语对象；首期支持 `DOCUMENT_MEMBER/DOCUMENT_VERSION/EVIDENCE` |
| `predicate` | `VARCHAR2(48)` 非空 | 受控关系类型，见下方 |
| `object_type`, `object_id` | 非空 `VARCHAR2(32)` + `NUMBER(38)` | 宾语对象；类型白名单与主语一致 |
| `directionality` | `VARCHAR2(16)` 非空 | `DIRECTED/SYMMETRIC`；对称关系按 ID 规范排序 |
| `support_json` | JSON CLOB 非空 | 支持关系的 Evidence、Member、页码/定位和抽取片段引用 |
| `derivation_type` | `VARCHAR2(16)` 非空 | `MANIFEST/RULE/MODEL/MANUAL` |
| `derivation_key` | `VARCHAR2(128)` 非空 | 规则版本、模型版本或受控人工操作标识 |
| `confidence` | `NUMBER(8,6)` 可空 | `RULE/MANIFEST` 可为 1；模型推理必须写入 |
| `attributes_json` | JSON CLOB 可空 | 编号、链接文本、关系限定词等低频属性 |
| `relation_status` | `VARCHAR2(16)` 非空 | `STAGED/ACTIVE/INVALID/DELETING` |
| 审计列 | 基础约定 | 生成者、时间与受控操作审计 |

`support_json` 至少保存一个可解析的锚点：Evidence 时包括 `evidence_id + locator_json` 快照或键；Manifest/Member 时包括 `bundle_revision_document_id` 或规范化 Manifest 路径。它保存依据引用而非复制大段正文。

## 首期谓词与约束

首期只开放：`REFERENCES`（显式引用）、`SAME_IDENTIFIER`（同编号/同条款）、`DERIVED_FROM`（有明确来源的派生）和 `RELATED_TO`（规则或模型验证后的相关）。`RELATED_TO` 不参与默认扩展，除非调用方显式允许且其置信度达到策略阈值。

- 所有主语、宾语必须属于同一 `collection_id + bundle_id`；服务层校验，不使用复合外键。
- 每个 Relation 必须属于一个 `bundle_revision_id`，使标题、附件目录和引用有效性可随新快照重建；不复用旧 Revision 的 Relation 行。
- `UK(bundle_revision_id, subject_type, subject_id, predicate, object_type, object_id, derivation_key)` 防止同一生成器重复写入。
- 索引 `(collection_id, bundle_revision_id, relation_status, predicate)` 供当前 Revision 扩展；索引 `(subject_type, subject_id, relation_status)`、`(object_type, object_id, relation_status)` 供双向邻接查询。

## 构建、可见性与查询

`RELATE` Job 在相关 Evidence/Member 已就绪后生成候选关系并先写 `STAGED`。验证 `support_json` 完整、对象仍属于本 Revision 且策略通过后激活；任一依据被隔离、Parse View 替换或新 Revision 切换时，将受影响关系置为 `INVALID`，随后异步清理或重建。

Evidence 检索只可从当前 Bundle Revision 的 ACTIVE Evidence 出发，按策略取得一跳 ACTIVE Relation，再重新执行安全、当前 Member 和定位校验。Relation 只能扩展候选范围，不能自身充当回答引用；Citation Pack 必须仍返回目标 Evidence。

首期以 Manifest 显式链接、稳定编号匹配和规则提取为主。模型关系是后续能力，必须保留模型/提示词或规则版本、置信度和 Evidence 依据，并可整体按 `derivation_key` 失效重建。
