# 步骤 2 详细设计：普通用户文件上传 API

## 边界与入口

普通用户一次选择多个文件时，KC 不能根据文件数量或 multipart batch 猜测 Bundle 边界。新增独立于 KM Asset Adapter 的入口：

```http
POST /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}/ingestions/user-files
Content-Type: multipart/form-data
Idempotency-Key: <stable request UUID>
```

认证主体必须拥有 Domain/Collection 上传权限；`app_id` 仍由配置注入，`source_system=KBot`、`source_type=USER_UPLOAD` 由服务器固定。请求显式选择 `grouping_mode=EACH_FILE|SINGLE_BUNDLE`，UI 分别展示“每个文件独立”和“这些文件属于同一资料包”。首期不根据目录、文件名或模型推断分组。

## `EACH_FILE`

每个文件创建一个独立 Bundle Revision 和一个角色为 `CONTENT` 的 Document Member。Bundle 标题默认取安全规范化后的文件名；每个 `client_file_id` 派生独立来源身份和 Receipt 子幂等键。

```text
10 files → 10 BundleIntakeCommand → 10 independent Bundles
```

各文件独立发布、解析、索引、删除和重试。一个文件失败不得回滚其他九个；响应以一个 `ingestion_batch_id` 聚合每项 `ACCEPTED/REJECTED` 状态，并返回各自 `bundle_id/revision_id/status_url`。批次是操作追踪对象，不是知识聚合、Revision 或检索边界。

## `SINGLE_BUNDLE`

请求必须包含稳定 `client_bundle_id`、标题、可选描述/Facet、安全等级和文件声明。全部文件作为同一 Revision 的平权 `CONTENT` Member；用户可显式标记 `SUPPLEMENT`，不能创建 `MANIFEST/DERIVED`。

```text
5 files → 1 BundleIntakeCommand → 1 Bundle Revision → 5 CONTENT Members
```

multipart 声明与文件 Part 必须完整匹配，任一内容校验或对象发布失败都不接受半份 Revision。受理成功后的解析按 Member 独立运行，允许按 required/optional 状态归约为 READY/PARTIAL。普通文件组始终有 Revision `manifest_json`，但不自动生成 MANIFEST Document；Bundle Discovery Profile 由该 Manifest、成员目录和各 Document 画像构建。

## 请求与幂等

JSON Part 至少包含：

```text
grouping_mode,
bundle? { client_bundle_id, title, description?, security_level?, facets? },
files[] { part_name, client_file_id, display_name, declared_mime_type,
          byte_size, content_sha256, role?, required? }
```

客户端选定文件后生成并在网络重试中保留请求级 `Idempotency-Key`、每文件 `client_file_id`，以及单 Bundle 模式的 `client_bundle_id`。KC 不支持 V1 `overwrite=true`：重试复用原结果，主动再次上传默认创建新来源对象；修改既有 Bundle 必须使用后续“创建新 Revision”契约并提交完整成员快照。

`EACH_FILE` 将请求展开为多个子 Receipt/`BundleIntakeCommand`，单项失败形成 `PARTIAL_ACCEPTED` 批次；`SINGLE_BUNDLE` 只有一个 Receipt，接收阶段全有或全无。二者最终复用相同的对象暂存、不可变发布、Revision/Member/Version 和 Job Application Service，不复制入库流水线。

## 响应与状态

成功受理返回 `202`，响应包含 `ingestion_batch_id`、`grouping_mode`、批次状态和 `items[]`。`EACH_FILE` 的 item 对应独立 Bundle；`SINGLE_BUNDLE` 只有一个 item。同步契约错误返回 `422`，幂等键复用冲突返回 `409`，存储不可恢复受理失败返回 `503`。解析完成与否只能通过各 Bundle/Revision 状态查询，不能由 HTTP 上传完成推断。

## 验收

- 同一组十个文件分别以两种模式上传时，稳定地产生十个 Bundle 或一个含十个 Member 的 Bundle。
- 重排 multipart Part、网络重试或重复提交不产生重复 Bundle；`EACH_FILE` 的单项失败不影响其他项。
- 普通多文件 Bundle 没有强制 PRIMARY/MANIFEST Member，Document Profile 可上卷同一 BundleCandidate，最终引用仍落到具体 CONTENT Evidence。
- 普通用户不能注入 `app_id/source_system/source_type/created_by`，不能声明 MANIFEST/DERIVED，也不能越过 Collection Scope。
