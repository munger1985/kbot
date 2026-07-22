# 步骤 1 详细设计：Bundle Revision Document Member

## 定位

`KBOT_KC_BUNDLE_REVISION_DOCUMENT` 是某个 Bundle Revision 的不可变文件声明清单，也是上传、补传、解析和状态归约的最小成员单元。

它不替代 `Document` 或 `Document Version`：

```text
Bundle Revision
  └─ Revision Document Member（该次快照声明一个文件）
       ├─ Document（跨 Revision 的逻辑文件身份）
       └─ Document Version（本次收到的具体内容；可空）
```

即使 Portal 无法下载附件，也要创建 Member 与 Document；仅 `document_version_id` 为空并标记 `SOURCE_UNAVAILABLE`。这样附件目录、失败原因和后续补传都有稳定锚点。

## 表字段

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `bundle_revision_document_id` | `NUMBER(38)` PK identity | 成员标识 |
| `collection_id` | 非空 `NUMBER(38)` | 从 Bundle Revision 关联的 Scope |
| `bundle_revision_id` | 非空 `NUMBER(38)` | 所属不可变快照 |
| `document_id` | 非空 `NUMBER(38)` | 对应稳定逻辑文件 |
| `document_version_id` | 可空 `NUMBER(38)` | 已收到内容时的具体版本 |
| `document_role` | `VARCHAR2(24)` 非空 | `CONTENT/MANIFEST/ATTACHMENT/SUPPLEMENT/DERIVED` |
| `ordinal` | `NUMBER(19)` 非空 | Bundle 内稳定展示和处理顺序，从 0 开始 |
| `required_flag` | `NUMBER(1)` 非空 | `1/0`；MANIFEST 必须为 `1` |
| `external_document_id` | `VARCHAR2(256)` 非空 | 来源文件稳定标识的快照，便于审计与补传定位 |
| `declared_name` | `VARCHAR2(512)` 可空 | 本次来源声明的文件名 |
| `declared_mime_type` | `VARCHAR2(255)` 可空 | 来源声明 MIME；实际 MIME 在 Version 记录 |
| `source_url` | `VARCHAR2(2048)` 可空 | 本次来源链接或对象地址 |
| `member_status` | `VARCHAR2(24)` 非空 | 成员接收与处理状态 |
| `failure_stage` | `VARCHAR2(32)` 可空 | `SOURCE_DOWNLOAD/STORAGE/PARSE/INDEX/VALIDATION` |
| `failure_code`, `failure_message` | 可空 | 结构化错误码与受限错误摘要 |
| `received_at`, `completed_at` | 带时区时间戳 | 内容接收与最终处理时间 |
| `row_version` + 审计列 | 基础约定 | 状态并发控制与审计 |

`source_url`、声明名称、MIME 与角色是 Revision 快照字段，不能在补传时改写；补传只补齐内容 Version 与接收/处理状态。

## 约束与索引

- `UK(bundle_revision_id, document_id)`：同一 Revision 内一个逻辑文件只出现一次。
- `UK(bundle_revision_id, external_document_id)`：防止来源重复声明相同文件。
- `CHECK document_role`、`CHECK required_flag`、`CHECK member_status`。
- 服务校验：`MANIFEST` 为 Adapter 可选角色、同一 Revision 至多一个且存在时 `required_flag=1`；通用 Revision 不要求 MANIFEST Member。
- 索引 `(bundle_revision_id, ordinal)`：成员目录与状态汇总。
- 索引 `(collection_id, member_status)`：管理端失败列表与巡检。
- 索引 `(document_id)`、`(document_version_id)`：审计和 Evidence 回溯。

不使用外键强制父子 Scope；KC 服务在创建 Member 时从 Bundle Revision 派生 `collection_id`，并由一致性巡检检查父对象存在性与 Scope 对齐。

## 状态机

```text
DECLARED
  ├─ 文件已接收 ─────► RECEIVED → PARSING → READY
  ├─ Portal 下载失败 ─► SOURCE_UNAVAILABLE ──补传──► RECEIVED
  ├─ 接收/校验失败 ───► FAILED ──显式重试/补传──► RECEIVED
  └─ Revision/Collection 删除 ─► CANCELLED

PARSING ──终态错误──► FAILED
```

- `DECLARED` 只存在于 KC 已登记、尚未确认文件或失败情况的短暂阶段。
- `SOURCE_UNAVAILABLE` 仅表示上游声明的下载失败，不能由 Parser 写入。
- `FAILED` 表示已接收或已投递的内容无法通过校验、解析或索引；必须保存 `failure_stage/code`。
- `READY` 要求存在可用 `document_version_id`，且该 Version 至少有一个 Active Parse View/Evidence（MANIFEST 的生成路径除外，由 Core 同样产出 Version）。

状态更新必须采用 `row_version` 或 Job lease 条件更新，防止迟到的 Parser 回调覆盖补传后的新状态。

## CONTENT 与可选 MANIFEST

普通单文件和普通文件组使用平权 `CONTENT` Member，不要求 PRIMARY。每个 Bundle Revision 都有不可变 `manifest_json`，但它是来源快照，不等于可引用 Document。

只有来源主信息本身需要成为答案依据时，Adapter 才创建固定 `external_document_id=__manifest__` 的 MANIFEST Member、Document 和 Version；同一 Revision 至多一个。KM Asset Adapter 使用该能力把 Asset 主信息确定性生成 Evidence，普通用户文件上传不自动创建。Bundle Discovery Profile 可以聚合 Manifest 与 Document 画像，但不能替代 Evidence。

## 与失败恢复的接口

补传接口以 `bundle_revision_id + external_document_id` 定位且仅允许 `SOURCE_UNAVAILABLE`/可恢复 `FAILED` Member。它必须校验声明名称、来源 URL、MIME 与角色不变；实际内容 hash 写入新 Document Version。成功后将 Member 置为 `RECEIVED` 并创建 PARSE Job；Revision 状态由 KC 统一重算。
