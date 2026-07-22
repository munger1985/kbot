# 步骤 3 详细设计：KM Portal 代码改造清单

本文件将步骤 3 映射到当前 `~/km_portal` 实现。它是 Portal 分支 `feature/kbot-3.5-bundle-adapter` 的实施边界；不修改 KBot V1 上传代码。

## 当前代码路径与替换目标

| 当前文件/行为 | V2 改造 |
| --- | --- |
| `file_loader/km_meta.py` 读取上游字段 | 保留全量原始 Asset 字段；将 `last_update_time`、语言、类型、分类、Pillar 等加入 `AssetMeta`，不在模型转换时丢失。同步 HTTP 改为非阻塞 client 是独立可靠性改进。 |
| `file_loader/file_params.py:AssetMeta` | 扩展为来源 DTO；新增 `AttachmentDeclaration`、`DownloadedAttachment`、`BundleRequest` 等 V2 DTO。 |
| `file_loader/file_processor.py` | 移除逐文件 `_upload_file()` 与 `asset_doc` 手工 Markdown。改为下载汇总、构建 Bundle、调用 KC Client、按受理结果回写 Metadb。 |
| `services/sharepoint.py` | 下载接口返回受控临时文件、原始文件名、真实 MIME/大小、稳定 DriveItem ID（可取得时）和原始 URL；失败产生结构化失败码。 |
| `core/config/settings.py` 与 `configuration/*.toml` | `KBotConfig` 改名/替换为 `KnowledgeCoreConfig`，保留 `domain_id/collection_key/default_security_level`，移除 `app_id/kb_id` 与 V1 `upload_api_url`。认证 Secret 只读环境变量。 |
| `file_loader/km_engine.py` | 保留轮询、队列和并发模型；成功定义改为“KC 返回 202”，不是每个附件上传完成。 |

## 建议的最小类职责

```text
KmEngine
  → FileProcessor.process_asset(asset)
      → AssetBundleBuilder.build(asset)
          → SharePointDownloader.download_all(...)
          → BundleRequest + local temporary files + document_failures
      → KnowledgeCoreClient.accept_km_asset(...)
      → KMFileMetaService.update_asset_metadata(...)
```

`AssetBundleBuilder` 只负责来源字段规范化、附件声明、SHA-256/大小和临时文件生命周期；它不发 HTTP。`KnowledgeCoreClient` 只负责 V2 multipart、认证、响应和错误映射；它不读取 Metadb 或 SharePoint。`FileProcessor` 承担编排与 Metadb 回写，确保临时资源在 `finally` 清理。

## 数据与错误处理细节

1. 下载前，为每个 `^^^` URL 创建有稳定 ordinal 的附件声明；不可下载也保留声明。
2. 下载成功：流式写入 `<temp_root>/<asset-id>/<request-id>/<ordinal>`，边写边计算 SHA-256 和大小；不要加载完整文件到 `BytesIO`。
3. 稳定外部 ID 优先采用 Graph DriveItem/UniqueId；回退为规范化 URL 的 SHA-256。文件名仅作展示字段。
4. 下载失败：加入 `document_failures`，包含同一外部 ID、URL、ordinal、`required_flag` 与受控失败码；不再只记录 warning 后丢弃。
5. 所有成功下载项在一次 multipart 中以独立 `part_name` 上传。Portal 不上传 `__manifest__`，也不产生本地 Asset Markdown。
6. `202` 或来源幂等 `202` 才调用 Metadb `processed=Y`。网络/限流/服务端错误保留 `N`；永久输入错误转 `F` 并记录 KC `request_id/code`。任何状态下均清理临时文件。

附件全部下载失败不是 Portal 接收失败：只要 Asset 元数据有效，仍向 KC 发送 `bundle + document_failures`，让 KC 生成 Manifest 并将 Revision 作为 `PARTIAL` 可检索来源受理。

## 配置与 HTTP 规则

```toml
[knowledge_core]
base_url = "http://localhost:18090"
domain_id = 1
collection_key = "assets"
default_security_level = 1
ingestion_mode = "v2"
connect_timeout_seconds = 10
request_timeout_seconds = 300
```

`KNOWLEDGE_CORE_API_KEY`（或 mTLS 配置）仅来自环境。V2 URL 在 Client 内由 `base_url + domain_id + collection_key` 构造，避免多个配置项漂移。`Idempotency-Key` 根据 Collection、Asset ID、`source_revision` 和已下载/失败附件声明的规范化指纹确定性生成；HTTP 重试使用同一键。

本 Adapter 分支只支持 `v2`。完成测试环境验证后，Portal 生产部署直接改用该 V2 Adapter；`v2` 发生异常时绝不调用旧 V1 URL。V1 Portal 上传路径仅在 V2 调试稳定前保留在代码/接口中且不参与路由，稳定验收后与旧表一并删除。

## 测试与完成条件

新增无需真实 SharePoint/KC 的 mock 测试：

- Asset 字段映射和 `last_update_time → source_revision`；
- 无附件：只发 Bundle，KC 可生成 Manifest；
- 多附件：一次请求含所有文件、稳定 ordinal 与 hash；
- 部分/全部下载失败：生成 `document_failures`，仍只调用一次 V2 接收；
- `202` 才写 `processed=Y`；`409/4xx`、网络超时、`5xx` 的 Metadb 状态符合步骤 3 策略；
- 同一 Asset/Revision 重试产生相同 Idempotency-Key；
- 断言 V2 模式没有 `/api/kb/upload` 调用，也没有 `app_id/kb_id` 或手工 Markdown part。

在 KC 可用后，再以非生产 Asset 做集成验证：检查 KC 中恰有一个 Bundle Revision、一个 Manifest Member、每个 URL 一个附件 Member 或 `SOURCE_UNAVAILABLE` Member，并验证无附件/坏 URL Asset 仍能检索到 Asset 主信息。
