# 步骤 1 详细设计：Document 与 Document Version

## 与 Collection、Bundle、Revision 的关系

Collection 划定知识边界；Bundle 是稳定来源业务对象；Bundle Revision 是该对象一次不可变来源快照；Document 是 Bundle 内稳定的逻辑文件身份。Document 不从属于某一个 Revision，而由 Revision Document Member 指明“该次快照使用了哪一个 Document Version”。

```text
Domain
└─ Collection
   └─ Bundle
      ├─ Document                         ← 稳定逻辑文件身份
      ├─ Bundle Revision #1               ← 不可变来源快照
      │  └─ Revision Document Member
      │     └─ Document Version 1
      └─ Bundle Revision #2
         └─ Revision Document Member
            └─ Document Version 1（内容未变，复用）
               或 Document Version 2（内容变化，新建）
```

- Bundle 只保存稳定来源键和 `current_revision_id`，不保存会变化的标题、Facet 或附件清单。
- Revision 保存当时的标题、Facet、Manifest 和成员清单；新 Revision 就绪后才成为 Bundle 当前 Revision。
- Document 以稳定 `external_document_id` 识别，不能以文件名、临时下载 URL 或上传 UUID 作为身份。
- Member 即使下载、存储或解析失败也必须保留；它既记录该附件属于该快照，也承载后续补传和失败追溯。

## 职责与版本边界

`Document` 是 Bundle 内跨 Revision 稳定的逻辑文件身份；`Document Version` 是一次实际收到、不可覆盖的二进制内容。当前可检索版本不由 Document 自己的 `is_current` 决定，而由**当前 Bundle Revision 的 Member** 指向的 `document_version_id` 决定。

```text
Bundle
  └─ Document（external_document_id 稳定）
       ├─ Document Version 1（hash A） ← Revision 1 Member
       ├─ Document Version 2（hash B） ← Revision 2 Member
       └─ Document Version 3（hash C） ← Revision 3 Member
```

相同内容 hash 的文件可复用已有 Document Version；新内容 hash 创建新 Version。旧 Version 永不覆盖，直到其所属 Collection 按 purge 规则删除。

## `KBOT_KC_DOCUMENT`

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `document_id` | `NUMBER(38)` PK identity | 逻辑文件身份 |
| `collection_id` | 非空 `NUMBER(38)` | 通过 Collection 关联 Scope |
| `bundle_id` | 非空 `NUMBER(38)` | Document 只属于一个 Bundle |
| `external_document_id` | `VARCHAR2(256)` 非空 | 来源内稳定文件标识 |
| `document_status` | `VARCHAR2(16)` 非空 | `ACTIVE/RETIRED`；通常由 Revision 成员关系推导 |
| 审计列 | 基础约定 | 首次发现和最后维护审计 |

约束与索引：`UK(bundle_id, external_document_id)`；索引 `(bundle_id, document_status)`。Document 不保存文件名、URL、MIME、内容 hash、存储 URI、角色或当前 Version：这些均会在不同 Revision 中改变，分别属于 Member 或 Document Version。

### 外部文件 ID 规则

Adapter 必须优先使用来源系统的稳定文件 ID。KM Asset 附件优先使用 SharePoint DriveItem/UniqueId；若上游只能提供 URL，Portal 必须规范化 URL（去除临时签名与查询参数）后计算稳定 ID，并保留原 URL 在 Member 的 `source_url`。禁止使用上传文件名或下载时生成的随机 UUID 作为 `external_document_id`。

MANIFEST 固定使用 `__manifest__`，在同一 Bundle 内唯一。

## `KBOT_KC_DOCUMENT_VERSION`

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `document_version_id` | `NUMBER(38)` PK identity | 不可变内容版本 |
| `collection_id`, `bundle_id`, `document_id` | 非空 `NUMBER(38)` | 父链与 Scope 加速关联 |
| `version_no` | 非空 `NUMBER(19)` | Document 内单调递增 |
| `content_hash` | `VARCHAR2(64)` 非空 | KC 计算的 SHA-256 |
| `storage_uri` | `VARCHAR2(2048)` 非空 | 已发布的不可变对象 URI |
| `storage_state` | `VARCHAR2(16)` 非空 | `AVAILABLE/QUARANTINED/DELETED` |
| `byte_size` | `NUMBER(19)` 非空 | 实际对象大小 |
| `detected_mime_type` | `VARCHAR2(255)` 非空 | KC 检测/校验后的 MIME |
| `security_level` | `NUMBER(3)` 非空 | 该内容的有效安全等级；可比 Collection 默认值更严格 |
| `content_metadata_json` | JSON CLOB 可空 | 文件校验、来源协议等低频事实，不存解析结果 |
| `received_at` | 带时区时间戳 | 内容通过发布校验的时间 |
| 审计列 | 基础约定 | 写入者和操作审计 |

约束与索引：`UK(document_id, version_no)`、`UK(document_id, content_hash)`；索引 `(collection_id, document_id)`、`(storage_state)`。Version 不保存 `is_current`、Parser 状态、Chunk 或 Evidence；解析策略和结果属于 Parse View，处理状态属于 Member/Job。

## 内容接收与不可变存储

1. 文件先落入隔离暂存 URI，计算 hash、大小、真实 MIME 并执行安全校验。
2. 校验成功后，以不可变路径发布，例如 `kc/{collection_id}/{document_id}/{content_hash}`；禁止覆盖同一路径。
3. 在 KC 事务中创建或复用 `(document_id, content_hash)` Version，并将对应 Member 指向该 Version。
4. 对象发布成功前不得创建可被 Parser 领取的 Job；发布失败不产生 AVAILABLE Version。

同一 Document 的相同 hash 重传复用既有 Version，仍可为新的 Bundle Revision 创建独立 Member。不同 Document 即使内容 hash 相同，首期不跨 Document 复用 Version 行，以保持来源、权限和删除边界简单；底层对象去重以后如需引入，必须通过独立对象引用计数设计。

## 重解析、补传与删除

- 重解析同一内容只创建新的 Parse View，不创建 Document Version。
- 补传此前 `SOURCE_UNAVAILABLE` Member 时创建或复用该 Document 的 Version；不得替换已接收 Version。
- 已接收内容若发现安全风险，Version 置为 `QUARANTINED`，关联 Member 不能为 READY，Evidence 立即撤销可见性。
- Collection purge 先撤销检索可见性，再按 Member 引用关系删除 Version 数据与对象；不能由单个页面直接删除 Version。
