# 步骤 1 复盘：一次完整文件入库流程

本文以“已存在的 Collection 中，Portal 投递一个 KM Asset，带 Manifest 和一个 PDF 附件”为例，说明 KC 表在接收、解析和可检索切换时分别写入什么。示例中的 ID 仅用于说明。

```text
Collection assets (已存在)
  → Bundle: metadb / KM_ASSET / A-10086
  → Revision #1
      ├─ Member __manifest__ → Document manifest → Version M1
      └─ Member driveitem:7f… → Document manual → Version D1
          → Parse View TEXT → Evidence
  → Discovery Object（Bundle + Document） → Bundle current_revision_id
```

## 0. 前置条件

`KBOT_KC_COLLECTION` 已有一行：`collection_id=10`、`domain_id=20`、`app_id` 由 `base.toml` 固化、`collection_key=assets`、`status=ACTIVE`。请求只携带经认证的 Domain、Collection Key 与来源业务数据；Portal 不传 `app_id`、`collection_id`、`source_system` 或 `source_type`。

## 1. 接收来源快照与附件声明

KC Adapter 验证 Domain/Collection，并规范化 Asset 主信息、附件描述和 `source_revision`。在一个事务中写入：

| 表 | 新增/更新内容 | 初始状态 |
| --- | --- | --- |
| `KBOT_KC_BUNDLE` | 首次创建 `(10, metadb, KM_ASSET, A-10086)`；尚无当前 Revision | `availability_status=EMPTY` |
| `KBOT_KC_BUNDLE_REVISION` | Revision #1、标题、Facet、规范化 `manifest_json`、附件声明 fingerprint | `ACCEPTED` |
| `KBOT_KC_DOCUMENT` | `__manifest__` 与 `driveitem:7f…` 两个稳定逻辑 Document | `ACTIVE` |
| `KBOT_KC_BUNDLE_REVISION_DOCUMENT` | 两个 Member，记录角色、名称、MIME、顺序、URL、是否必需 | Manifest `DECLARED`；附件 `DECLARED` |

此时没有 `Document Version`、Parse View、Evidence 或 Discovery Object；外部查询也找不到该 Bundle，因为 `current_revision_id` 仍为空。

若 Portal 在下载附件时已失败，附件 Member 仍写入，只置为 `SOURCE_UNAVAILABLE` 并填写 `failure_stage=SOURCE_DOWNLOAD`。它不能被静默省略，后续可按原 Member 补传。

## 2. 接收和发布不可变内容

KC 由 `manifest_json` 自行生成 Manifest 内容；附件则先下载/上传到隔离暂存区。暂存 URI、临时凭据和下载令牌不写 KC 事实表。

校验 hash、大小、真实 MIME 和安全策略通过后，将对象发布到不可变 URI，例如 `kc/10/102/sha256...`，并在事务中写入：

| 表 | 新增/更新内容 | 状态 |
| --- | --- | --- |
| `KBOT_KC_DOCUMENT_VERSION` | Manifest 的 M1、附件的 D1；各自 `version_no=1`、`content_hash`、`storage_uri`、MIME、安全级别 | `storage_state=AVAILABLE` |
| `KBOT_KC_BUNDLE_REVISION_DOCUMENT` | 对应 Member 填入 `document_version_id`、`received_at` | `RECEIVED` |
| `KBOT_KC_BUNDLE_REVISION` | 开始处理 | `PROCESSING` |
| `KBOT_KC_PARSE_VIEW` | 为 M1/D1 各建立候选 View，保存 parser/policy snapshot | `BUILDING` |
| `KBOT_KC_INGESTION_JOB` | 每个候选 View 一条 `PARSE` Job，含 input fingerprint 与租约参数 | `PENDING` |

同一 Document 的相同内容 hash 重传时复用原 Version；不同内容 hash 才新建 Version。此规则不改变 Member 的 Revision 快照语义。

## 3. 领取解析任务与写入候选产物

Parser Worker 以能力向 KC claim `PARSE` Job；KC 以条件更新签发租约，写入 `lease_owner/lease_until`，状态改为 `RUNNING`，并返回 Version URI、Parse View 策略和 Job 身份。Parser 不查询、也不写 KC 表。

Parser 回调的是版本化 Evidence DTO。KC 校验 Job 租约、`input_fingerprint`、Version 存储状态与 View 状态后，规范化 DTO 并写入：

| 表 | 新增/更新内容 | 状态/可见性 |
| --- | --- | --- |
| `KBOT_KC_EVIDENCE` | 每个段落/表格/图片/Sheet 范围一行；写 content、定位、层级、来源、检索文本、hash、token、向量 | `STAGED`，不可查询 |
| `KBOT_KC_PARSE_VIEW` | 输出 URI、质量报告、完成信息 | 仍为 `BUILDING` |
| `KBOT_KC_INGESTION_JOB` | 批次统计、心跳或最终结果 | 仍为 `RUNNING` |

此阶段允许分批回调，但任何 `STAGED` Evidence 都不参与检索；迟到回调、过期租约、被删除 Version 或已替换 View 的结果必须被 KC 拒绝。

## 4. 单个文件解析完成：View 和 Member 就绪

全部 Evidence 写完并通过 View 质量门后，KC 在一个事务中：

1. 将该 View 及其 Evidence 切为 `ACTIVE`；若是重解析，先撤销旧 View/Evidence 的可见性，再异步清理。
2. 将对应 Revision Member 置为 `READY`，记录 `completed_at`。
3. 将 `PARSE` Job 置为 `SUCCEEDED`。

Manifest 与附件各自独立完成。附件永久解析失败时，附件 Member 为 `FAILED`，相关 View 为 `FAILED`、Job 为 `FAILED`；Manifest 或其他已成功附件不被回滚。此时 Evidence 虽已 ACTIVE，但 Bundle 还未切换为当前 Revision，外部检索通过 Revision 过滤仍不可见新快照。

## 5. 建立检索画像、关系并切换当前 Revision

KC 根据全部 Member 汇总 Revision：Manifest 和所需内容均成功则 `READY`；符合部分可用规则则 `PARTIAL`；没有可用内容则 `FAILED`。对于 READY/PARTIAL Revision：

| 表 | 新增/更新内容 | 状态 |
| --- | --- | --- |
| `KBOT_KC_INGESTION_JOB` | 一条 `PROFILE` Job；可选 `RELATE` Job | `PENDING` → `RUNNING` |
| `KBOT_KC_DISCOVERY_OBJECT` | 一条 Bundle 画像，及每个 READY Member 的 Document 画像 | 先 `STAGED` |
| `KBOT_KC_RELATION` | 有明确 Manifest/规则/Evidence 依据的关系（可选，不阻塞首期可检索） | 先 `STAGED` |

PROFILE 完成后，KC 在切换事务中将 Bundle 的 `current_revision_id` 指向 Revision #1、将 `availability_status` 设为 `READY/PARTIAL`、激活合格 Discovery Object，并完成 Revision 状态更新。旧 Revision 若存在，仍保留审计记录但不再满足“current revision”查询条件。

此刻 V2 Discovery 才能召回 Bundle/附件；V2 Evidence 再通过当前 Revision Member → Document Version → Active Parse View → Active Evidence 返回可引用内容。Relation 可稍后独立激活，只用于扩展候选，绝不替代 Evidence 引用。

## 6. 可恢复分支与幂等边界

- **重复投递相同来源修订和 fingerprint**：定位已有 Bundle Revision，直接返回其处理状态；不新增 Document、Version 或 Job。
- **附件下载失败**：保留 `SOURCE_UNAVAILABLE` Member；补传成功后创建/复用 Version，重新走 Parse → Profile → 切换。
- **解析临时失败**：Job 按租约和退避重试；Member 在终态失败前不丢失。
- **解析永久失败**：保留失败 Member 和受限错误摘要；若满足部分可用规则，可切换 Revision，但 Discovery `coverage_json` 必须记录缺失附件。
- **重解析同一内容**：新建候选 Parse View 与 PARSE Job，不创建 Document Version；新 View 成功后替换旧 View/Evidence。
- **来源内容更新**：新建 Bundle Revision 和需要变更的 Document Version；旧 Revision 继续被检索，直到新 Revision 的画像完成并原子切换。
