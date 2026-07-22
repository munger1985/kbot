# KBot 3.5 Knowledge Core：KM Asset 接入 Profile

> 归档说明：本文保留整理前的草案内容；实施请以[3.5 文档索引](../README.md)及其中的正式分册为准。

本文是独立 Knowledge Core 服务方案的首个来源接入规范。Knowledge Core 采用通用 Collection、Bundle、Document、Document Version、Parse View、Evidence、Discovery Object、Relation 和 Ingestion Job 模型；KM Asset 只是 `source_system=metadb`、`source_type=asset` 的 Bundle。

## 当前 Portal 与改造目标

当前 `~/km_portal` 轮询 Metadb 的 `processed=N` 记录，将 `first_sp_url` 以 `^^^` 拆成附件，并逐个调用旧 `/api/kb/upload`，最后上传 Portal 拼接的 Markdown。独立上传无法表达完整附件清单、文件角色、来源修订和一次 Asset 的原子接收边界。

3.5 改为：Portal 对一个 Metadb Asset 发起一次 `POST /v1/bundles` multipart 上传；Knowledge Core 写入一个 Bundle、一个由 Core 生成的 MANIFEST Document 及多个 ATTACHMENT Document/Version，随后投递解析任务。Portal 不再上传手工生成的 Markdown，也不直接访问 KBot 数据库。

## 上传契约

请求包含以下 multipart parts：

| Part | 内容 |
| --- | --- |
| `bundle` | 结构化 Bundle Manifest JSON |
| `documents` | 附件描述数组；每项声明 `part_name`、`role=ATTACHMENT`、`external_document_id`、`source_url`、文件名、顺序和 `required` |
| 文件 parts | `part_name` 对应的附件字节 |
| `document_failures` | 可选；下载失败的来源 URL、是否 required、错误码 |

Bundle Manifest 的必填字段：

```json
{
  "collection_id": 1,
  "source_system": "metadb",
  "source_type": "asset",
  "source_id": "<asset_id>",
  "source_revision": "<last_update_time>",
  "title": "<asset_title>",
  "canonical_url": "<APEX asset URL>",
  "security_level": 1,
  "facets": {
    "author_email": "<author_mail>",
    "product": "<asset_product>",
    "solution": "<asset_solution>",
    "industry": "<industry_id>",
    "sub_type": "<sub_type>",
    "asset_type": "<asset_type>",
    "status": "<asset_status>",
    "language": "<asset_language>",
    "content_category": "<content_category>",
    "pillar": "<pillar>",
    "country": "<country>",
    "customer": "<customer>"
  },
  "metadata": { "remaining_metadb_fields": "..." }
}
```

`asset_details`、`solution_briefing`、`biz_background`、`business_challenges`、`results`、`contact_info`、`engagement_id`、`team` 和 `audience` 保留在 `metadata`，由 Core 确定性写入 MANIFEST。空字段不写入。`source_revision` 优先使用 Metadb 的 `last_update_time`；若上游可提供单调版本号或内容 hash，应优先使用。

## 处理与状态

Portal 下载 SharePoint 附件，传入真实文件名、来源 URL 和稳定 `external_document_id`；无法下载的附件必须写入 `document_failures`，不能静默跳过。Core 接受成功即返回 `202`、`bundle_id`、`accepted_revision` 和 `PENDING`，随后由 Parser Worker 通过 Core 内部任务协议完成解析、Parse View、Evidence 和 Discovery 索引。

Bundle 在主信息及至少一个可用 Document/Evidence 就绪时为 `READY`；局部附件失败为 `PARTIAL`，全部失败为 `FAILED`。相同来源修订幂等返回已有状态；新修订创建新的 Document Version，待新索引可用后切换 current version。

Metadb `processed=Y` 的语义固定为“该修订已被 Knowledge Core 接收”，不是“全部解析完成”。Portal 可通过 `GET /v1/bundles/{bundle_id}` 同步最终状态；网络/校验失败则保留 `N` 或写 `F`，按 Portal 的重试策略处理。

## Portal 最小改动

- `AssetMeta` 补充 `last_update_time`，并保留完整原始字段，避免同步层丢失 Facet。
- `FileProcessor.process_asset()` 下载完全部附件后只调用一次 Bundle API；删除逐附件 `_upload_file()` 与 Portal 内 Markdown 生成。
- `KBotConfig` 改为 `knowledge_core_url` 或 `bundle_upload_api_url`/`bundle_status_api_url`；不再复用旧 `upload_api_url`。
- 现有 Agent/Skill 不改；已迁入 Collection 的检索由 Main API 内 `KnowledgeCoreClient` 适配到当前 `TxtBaseSearchResult`。
