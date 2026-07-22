# 步骤 3 详细设计：KM Portal Bundle Adapter

## 改造目标与现状差异

KM Portal 继续负责轮询 Metadb、取得业务元数据和下载 SharePoint 附件，但从“逐文件上传器”改为 KC 的来源 Adapter：一个 Asset 只发起一次 Bundle multipart 接收，不生成手工 Markdown、不直连 KBot 数据库、不等待或驱动解析。

当前实现的关键问题是：`FileProcessor.process_asset()` 以 `^^^` 拆分 `first_sp_url` 后逐个调用 V1 `/api/kb/upload`，再上传拼接的 Markdown；部分或全部附件失败仍可能写 `processed=Y`。此外，`KMFileMetaService` 已读取 `last_update_time`，但 `AssetMeta` 未保留它，无法构造稳定 `source_revision`。

## Asset 主信息必须检索：由 KC 生成 Manifest，而非取消它

Asset 本身是一个完整知识对象，不是附件的容器。因此 `asset_id`、标题、Solution Briefing、方案/产品/行业、作者、来源链接及附件**声明目录**必须可检索；无附件或附件 URL 错误时尤其不能丢失这些内容。

V2 的变化不是删除这份文本，而是改变其所有权和版本语义：Portal 传递结构化 `bundle + facet + metadata`，KC 从已规范化的 `manifest_json` 确定性物化 `external_document_id=__manifest__` 的 `MANIFEST Document/Version`。该内容可以采用固定 Markdown/纯文本模板并正常解析为 Evidence，例如：

```text
Asset: <title>
Asset ID: <source_id>
Solution Briefing: <solution_briefing>
Product / Solution / Industry: ...
Author / Created at / Source URL: ...
Declared attachments: <name, stable external ID, source URL> ...
```

因此 Asset 主信息既能参与 Bundle/Document Discovery，也能通过 Manifest Evidence 被最终引用。Portal 不再上传“伪附件 Markdown”的原因是它会与 Bundle Revision 的标题、Facet、附件清单产生两份可漂移事实；KC 生成的 Manifest 才是该 Revision 的唯一、可重建、可版本化主信息文档。附件的下载/解析失败是运行状态，不写入不可变 Manifest 正文，而由 Member 状态和 Discovery `coverage_json` 表达。

## Portal 内部职责

```text
Metadb 轮询
  → Asset Metadata Normalizer
  → 下载全部附件至受控临时目录，并计算大小/SHA-256
  → Bundle Request Builder
  → KnowledgeCoreClient：一次 multipart POST
  → 收到 KC 202 后回写 Metadb processed=Y
```

建议新增 `KnowledgeCoreClient` 与 `AssetBundleBuilder`，并将当前 `FileProcessor` 缩为协调器。下载必须写入每个 Asset 独立的临时目录，不能继续将多个附件完整保存在 `BytesIO`；请求结束后清理临时文件。现有 SharePoint Client 需优先取得 DriveItem/UniqueId；若上游暂时只能提供 URL，则以去除临时签名/查询参数后的规范化 URL hash 作为 `external_document_id`，文件名不得作为身份。

## 来源字段映射

| KC Bundle 数据 | Metadb/Portal 来源 | 规则 |
| --- | --- | --- |
| `source_id` | `asset_id` | 原样传递 |
| `source_revision` | `last_update_time` | 必须加入 `AssetMeta`；有单调修订号时优先使用 |
| `title` | `asset_title` | 为空时使用安全展示兜底，不改变来源 ID |
| `canonical_url` | APEX Asset 页面 URL | 由 `asset_id` 构造 |
| `security_level` | Portal Collection 配置默认值 | 首期不由 Metadb 客户端字段决定 |
| Facet | `asset_product/sub_type/industry_id/asset_solution/asset_language/asset_type/content_category/pillar/...` | 高频筛选字段规范化为受控 Facet |
| metadata | 作者、客户、简介、时间、联系信息及其余低频原始字段 | 保留结构化原值；不拼接为 Markdown |

`first_sp_url` 的 `^^^` 分隔仍仅是当前上游适配细节；每个拆出的 URL 都必须成为一个附件声明，或成为一个明确的 `document_failures` 条目。没有附件、或所有附件均下载失败的 Asset 仍是有效 Bundle：KC 仍生成并解析 Manifest Document，Asset 主信息可以 `READY` 可检索；附件缺失则在 Discovery 覆盖信息中展示，不能让 Asset 整体消失。

## 单次 multipart 请求

Portal 调用：

```text
POST /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}/ingestions/km-assets
```

- `bundle`：上述来源主信息、Facet 和 metadata；不得传 `app_id`、`kb_id`、`collection_id`、`source_system`、`source_type`。
- `documents`：所有下载成功附件的 `part_name`、稳定外部 ID、原始 URL、声明文件名/MIME、顺序、必需标记、实际 `byte_size/content_sha256`。
- 文件 Part：临时文件流；不拼接或额外上传 Asset Markdown。
- `document_failures`：下载失败的 URL、稳定外部 ID、顺序、必需标记和受限失败码。下载失败不能静默跳过。

Portal 在全部下载完成并得到附件摘要后，使用 `collection_key + asset_id + source_revision + request_fingerprint` 确定性生成 `Idempotency-Key`。网络超时可用同一键重试；不得为每次重试随机生成新 UUID。

## Metadb 回写与重试

| KC 响应/结果 | Portal 行为 | Metadb `processed` |
| --- | --- | --- |
| `202 Accepted`（含已有同来源 Revision 的幂等结果） | 记录 KC Bundle/Revision、清理临时文件 | 写 `Y` |
| 网络超时、`429`、`5xx` | 保留为可重试；下次轮询用同一来源修订重新投递 | 保持 `N` |
| `409 SOURCE_REVISION_CONFLICT` | 告警并停止自动覆盖；要求上游提供新修订 | 保持 `N` 或按现有运维流程标 `F` |
| `4xx` 声明/配置错误 | 记录安全错误与告警；不盲目重试 | 标 `F`，待人工修复后重置 |

`processed=Y` 仅表示 KC 已受理来源快照，**不表示解析完成**。Portal 不轮询 Parser、不根据解析结果重新上传；如需运维展示，可调用 Bundle/Revision 状态 API 做只读对账。对于 `PARTIAL/FAILED`，补传、重解析和状态恢复由 KC API/运维流程管理。

## 配置、兼容与切换

Portal 的 KBot 配置改为：`knowledge_core_base_url`（或明确 V2 intake URL）、`domain_id`、`collection_key`、默认安全等级、超时/最大附件数和最大总字节。`app_id` 与 `kb_id` 从 V2 Portal 配置及请求中移除；认证材料只从环境 Secret 读取。

Portal 改造完成并在测试环境验证后，生产部署直接使用 V2 Bundle Adapter；不进行长期双部署、影子双写或按流量比例切换。V1 接口和旧表在 V2 调试稳定前仅保留为未路由资产，V2 发生异常时也绝不自动调用旧 V1 URL。V2 稳定验收后，再统一删除 V1 接口、Skill 和对应旧表。

## 实施与验收

1. 扩充 `AssetMeta` 保留 `last_update_time` 与所需 Facet 原字段；实现稳定附件 ID/URL 规范化。
2. 实现临时文件下载、hash/MIME/大小采集与 Bundle Request Builder；删除 Portal 侧手工 Markdown 生成，验收 KC 侧确定性 Manifest 生成与解析。
3. 实现 `KnowledgeCoreClient` 的 multipart、Idempotency-Key、超时和安全错误映射；替换逐文件 V1 上传。
4. 调整 `processed` 回写和日志，增加 KC Bundle/Revision/request ID；保留 V1/V2 显式开关。
5. 对无附件、多附件、部分下载失败、全部下载失败、网络重试、相同修订重复投递和新修订更新做集成验证。

验收时一次 Asset 只能产生一个 KC Bundle Revision；所有声明附件均有 Member 或失败记录；无附件/附件失败的 Asset 仍可通过 Manifest 被发现和引用；Portal 不再生成 Markdown、传递 Scope 字段或调用 V1 上传端点；`processed=Y` 只发生在 KC 返回 `202` 后。
