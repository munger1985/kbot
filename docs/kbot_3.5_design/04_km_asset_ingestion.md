# 3.5 KM Asset Bundle 接入

KM Asset 是 KC 的首个来源 Profile：`source_system=metadb`、`source_type=KM_ASSET`、`source_id=<asset_id>`。它不改变通用 Collection/Bundle 模型，也不要求或创建名为 `km_assets` 的 Collection；Collection 由 APEX UI/管理端预先在 Domain 下创建，Portal 使用配置好的目标 Collection。

## 当前问题与目标

当前 Portal 轮询 Metadb 的 `processed=N` 记录，按 `^^^` 拆分 `first_sp_url` 后逐个调用旧 `/api/kb/upload`，最后额外上传拼接 Markdown。这丢失了来源对象、附件清单、修订和原子接收边界。

改造后，接入 V2 的 Portal 下载附件后只调用一次 `POST /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}/ingestions/km-assets` multipart。该入口由 KM Asset Adapter 在服务端固定注入 `source_system=metadb` 与 `source_type=KM_ASSET`；KC 从结构化 Asset 主信息确定性生成 `MANIFEST` Document（可解析、可检索、可引用），并创建多个 `ATTACHMENT` Document/Version 和解析任务。Portal 不生成或上传手工 Markdown，不直连 KBot 数据库；尚未切换的来源可继续走 V1。

## 上传契约

| multipart Part | 内容 |
| --- | --- |
| `bundle` | `source_id`、修订、标题、安全级别、Facet 与 metadata JSON；不接受 `app_id`、`domain_id`、`collection_id`、`source_system` 或 `source_type` |
| `documents` | 附件描述：part、稳定外部 ID、角色、URL、名称、顺序、是否必需 |
| 文件 parts | 对应附件字节 |
| `document_failures` | 可选下载失败 URL、必需标记和错误码 |

Manifest 必填 `source_id`、`source_revision`、标题与安全级别；Domain/Collection 来自经过认证的 URL Scope，`app_id` 来自服务配置，`source_system/source_type` 由 KM Asset Adapter 注入。`last_update_time` 优先作为 `source_revision`；若有单调版本号或内容 hash 则优先使用。作者、产品、方案、行业、语言、分类等高频字段映射 Facet，低频 Metadb 字段保存在 metadata 并由 KC 写入 Manifest。

## 状态与 Portal 改动

接收成功返回 `202`、`bundle_id`、`accepted_revision`、`PENDING`。至少主信息和一个可用 Document/Evidence 就绪为 `READY`；局部附件失败为 `PARTIAL`；全部失败为 `FAILED`。同修订幂等返回现有状态，新修订创建 Version 并在索引完成后切换。`processed=Y` 仅表示 KC 已接收该修订，最终状态通过 `GET /api/v2/knowledge/domains/{domain_id}/bundles/{bundle_id}` 查询。

Portal 需补全 `last_update_time` 和原始字段；删除逐文件上传及手工 Markdown；将 `upload_api_url` 改为 KC Bundle 上传/状态地址；下载失败必须显式上报，不能静默跳过。
