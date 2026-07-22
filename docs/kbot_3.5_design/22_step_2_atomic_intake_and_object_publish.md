# 步骤 2 详细设计：文件接收、对象发布与补偿

## 目标与限制

一次 Bundle multipart 接收必须做到：不把半份附件当作已入库、不让对象存储失败产生可检索 Version、网络重试不产生重复 Revision，并且在数据库事务与对象存储无法共享 ACID 事务时可恢复清理。

为此新增运行态辅助表 `KBOT_KC_INGESTION_RECEIPT`。它不参与检索、不承载业务知识事实，只保存一次 HTTP 接收的幂等与补偿信息。

## `KBOT_KC_INGESTION_RECEIPT`

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `ingestion_receipt_id` | `NUMBER(38)` PK identity | 接收收据标识 |
| `collection_id` | 非空 `NUMBER(38)` | 已验证的目标 Scope |
| `actor_id` | `VARCHAR2(256)` 非空 | Portal 服务主体；与幂等键共同限定 |
| `idempotency_key` | `VARCHAR2(128)` 非空 | HTTP 重试稳定键 |
| `request_fingerprint` | `VARCHAR2(64)` 非空 | 规范化 bundle、附件声明、失败声明及声明 hash/大小的 hash |
| `receipt_status` | `VARCHAR2(24)` 非空 | `RECEIVING/STAGED/COMMITTING/ACCEPTED/REJECTED/CLEANUP_PENDING` |
| `bundle_id`, `bundle_revision_id` | 可空 `NUMBER(38)` | 成功受理后的结果锚点 |
| `staging_manifest_json` | JSON CLOB 可空 | 暂存对象键、大小/hash、清理期限；不存下载令牌 |
| `failure_code`, `failure_message` | 可空 | 安全的接收失败摘要 |
| `expires_at` + 审计列 | 带时区时间戳 | 暂存/收据回收与审计 |

约束为 `UK(collection_id, actor_id, idempotency_key)`。相同键、不同 `request_fingerprint` 返回 `409 IDEMPOTENCY_KEY_REUSED`；相同键、相同 fingerprint 返回已有受理结果或正在处理状态，不重复流式发布。收据及暂存对象由 Reaper 按 `expires_at` 清理；`ACCEPTED` 收据按 API 审计策略归档。

## multipart 接收流程

```text
认证 + Scope 校验
  → 校验 bundle/documents/document_failures 声明
  → 预占 Receipt
  → 流式写入隔离暂存、边写边 hash/病毒扫描/MIME 检测
  → 全部附件通过后，创建事实行并发布不可变对象
  → 提交 Revision + Member + Version + Parse Job
  → Receipt=ACCEPTED，返回 202
```

1. API 要求 JSON Part 在文件 Part 前出现，先校验 `source_id`、`source_revision`、附件完整性与每个文件声明的 `byte_size/content_sha256`，计算 `request_fingerprint`。
2. 用 `(collection_id, actor_id, Idempotency-Key)` 创建或读取 Receipt。已有 `ACCEPTED` 直接返回原 `202` 结果；已有进行中 Receipt 返回 `202` 及状态地址；同键不同指纹返回冲突。
3. 文件流只写入隔离前缀，如 `kc-staging/{receipt_id}/{part_name}`。边写边计算 SHA-256/大小，做 MIME、恶意文件和大小策略校验；文件字节与声明不符即拒绝。此阶段没有 Bundle Revision、Document Version、Evidence 或 Job 可见。
4. 全部文件与显式 `document_failures` 校验通过，Receipt 置 `STAGED`；任何中途失败将 Receipt 置 `REJECTED`，并最佳努力删除暂存对象。删除失败则置 `CLEANUP_PENDING`，交给 Reaper；不得返回 `202`。

## 发布与数据库提交

对象存储与 Oracle 不能形成单一事务，因此采用“隔离暂存 + 不可变发布 + 数据库原子提交 + 可恢复补偿”：

1. KC 以来源键锁定/定位 Bundle，并在短数据库事务中创建或定位 Bundle、不可变 Bundle Revision、Document、Member，分配 Document ID；同来源修订/fingerprint 已存在则跳过发布并返回该 Revision。
2. 将暂存对象以条件复制/发布到不可变键 `kc/{collection_id}/{document_id}/{content_hash}`；对象必须禁止覆盖，并写入 hash、大小、MIME 元数据。发布成功前，不创建 `AVAILABLE` Version 或 Parse Job。
3. 在同一最终数据库事务中创建/复用 `KBOT_KC_DOCUMENT_VERSION(storage_state=AVAILABLE)`，令 Member 指向 Version，创建 Manifest Version、候选 Parse View 和 `PARSE` Job；Revision 置 `PROCESSING`，Receipt 填入 Bundle/Revision 并置 `ACCEPTED`。
4. 只有该事务提交后才返回 `202 Accepted`。Parser 只能 claim 已提交 Job，因而永远不会读到未提交对象。

若不可变对象已发布但最终数据库事务失败，Receipt 保留发布清单并置 `CLEANUP_PENDING`；Reaper 删除无任何 AVAILABLE Version 引用的对象。反之，数据库提交后不得删除已发布对象；Version/Member/Job 共同构成可恢复的事实锚点。

## 幂等与失败分界

- **HTTP 重试**：Receipt 负责同一客户端请求键的重放安全。
- **来源重复投递**：Bundle Revision 的 `(bundle_id, source_revision, snapshot_fingerprint)` 规则负责业务幂等；与 HTTP 键无关。
- **文件部分失败**：只接受在 `document_failures` 中明确声明的上游失败附件；它们创建 `SOURCE_UNAVAILABLE` Member，但不阻止其他已成功暂存文件受理。
- **校验/发布失败**：无 `ACCEPTED` Receipt、无可解析 Job；暂存或孤儿对象进入补偿清理。
- **进程崩溃**：Receipt Reaper 根据状态重试安全提交或清理；不得猜测一个未完成上传已成功。

## 安全与可观测性

暂存 URI、最终对象 URI、下载令牌、反病毒原始报告和完整异常栈仅写内部受控日志/Receipt，不进入外部状态 API。API 记录 `request_id`、Receipt、来源键、声明/实测 hash、对象发布耗时与清理结果。对象读取仅通过 KC 签发的短期受控 URI 或 Worker 身份访问策略，Portal 不能凭 `storage_uri` 直接读取其他 Domain 的内容。
