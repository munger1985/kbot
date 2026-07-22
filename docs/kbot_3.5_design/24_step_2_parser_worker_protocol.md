# 步骤 2 详细设计：Parser Worker 协议

## 范围

本协议冻结 KC 与 Parser 的内部 HTTP 契约；步骤 4 才改造现有 Parser 实现。Worker 使用拉取模型：声明能力、claim 有限租约、分批提交解析产物并完成或失败。Worker 不知道 Bundle 状态机、不访问 Oracle、也不决定 Revision 是否切换。

## Claim 与租约

```http
POST /internal/v2/knowledge/parse-tasks/claim
```

请求包含 `worker_id`、`capacity`、支持的 MIME、`view_types`、能力标签（如 `DOCLING`、`OCR`、`VLM`、`SPREADSHEET`）和 Worker 软件版本。KC 只分配 `PENDING/RETRY_WAIT` 且能力匹配的 `PARSE` Job，并以条件更新写入 `lease_owner/lease_until`。

每个任务响应包含：

| 字段 | 说明 |
| --- | --- |
| `job_id`、`lease_owner`、`lease_until` | 后续调用必须原样携带的有限租约身份 |
| `input_fingerprint` | 防止迟到结果覆盖新输入 |
| `document_version_id`、`content_hash`、`detected_mime_type` | 不可变解析输入身份 |
| `source_read_url` | 绑定当前租约的 KC 内部下载 URL；不返回对象存储物理 URI |
| `parse_view_id`、`view_type`、`coverage_key` | 候选 View 目标 |
| `policy_snapshot` | 解析、OCR/VLM、分块、最大批次等冻结策略 |

`heartbeat` 只可由持有同一 `lease_owner` 的已认证 Worker 续租；KC 限制最大续租时长，防止失联 Worker 无限占有任务。租约失效后 Worker 必须立即停止上传结果。

## Evidence 批次 DTO

```http
POST /internal/v2/knowledge/parse-tasks/{job_id}/evidence-batches
```

请求带 `lease_owner`、`input_fingerprint`、单调 `batch_no`、确定性 `batch_key` 和 `evidences[]`。每个 Evidence DTO 至少含：

```text
evidence_key, parent_evidence_key?, source_item_ref?, source_spans, fragment_index,
evidence_type, ordinal, heading_path, section_key?, hierarchy_depth?, heading_level?,
content, locator_schema_version, locator, payload_descriptor?, provenance?, language_code?
```

`evidence_key` 由 Parser 根据 `parse_view_id + hash(canonical_source_spans) + fragment_index + evidence_type` 生成；不可使用随机 UUID 或单纯 ordinal。`parent_evidence_key` 使用同一规则。KC 校验 Atom/span 是否存在、Scope、定位 schema 与内容限制，规范化后计算 `content_hash/retrieval_text/token_count`，再写入 `STAGED` Evidence；PARSE 事务不生成向量。

Parser DTO 明确禁止 `embedding`、`query_vector`、`embedding_model_id/key`。文本向量不属于 PARSE Job；Parse View 激活后由 INDEX Job 读取 Collection 唯一的 `embedding_model_id` 统一生成。OCR/VLM 产生的文字描述属于可追溯解析内容，不是第二套文本向量。

重放相同批次或网络超时后重传是安全的：同一 `(parse_view_id, evidence_key)` 且规范化 hash 相同则幂等确认；相同键但内容/定位不同返回 `409 EVIDENCE_KEY_CONFLICT` 并使 Job 失败，要求重新创建候选 View。`batch_no` 只用于诊断和顺序校验，不作为 Evidence 身份。

`locator` 必须符合该 `locator_schema_version`：文档类输出完整 `pages[]`、bbox、坐标系/页面尺寸；Spreadsheet 输出真实 Sheet 名、表/行列/单元格范围；图片/幻灯片输出可回到源对象的定位。`payload_descriptor` 只声明 Worker 已写入受控临时或对象 URI 的资源与 hash，KC 校验后才保存为 `payload_uri`。

非分页 DOCX/Markdown/HTML 使用 `document-logical/v1`，以有序 Docling `source_ref` 定位，不能伪造页码。Worker 通过 `GET /internal/v2/knowledge/parse-tasks/{job_id}/source` 下载源文件；请求必须携带当前 `worker_id/input_fingerprint`，租约失效后立即拒绝。

四类 JSON 工件分别通过 `POST .../{job_id}/artifacts/{artifact_name}` 上传。KC 规范化计算 hash、写入不可变内容地址并返回 descriptor；complete 时依据 descriptor hash 与已落库 Evidence key 顺序复算 `output_fingerprint`。

## 完成与失败

```text
POST /internal/v2/knowledge/parse-tasks/{job_id}/complete
POST /internal/v2/knowledge/parse-tasks/{job_id}/fail
```

`complete` 提交 `lease_owner/input_fingerprint`、`artifact_manifest`、`output_manifest_uri`、`output_fingerprint`、Evidence 总数、分层质量报告、Parser/模型版本和耗时。工件清单至少包含 `raw_docling/atom_ir/structure_ir/evidence_manifest` 的 URI、hash、schema 与生成器版本。KC 在一个短事务中校验：所有已提交 Evidence 数量/键与输出 manifest 一致、source spans 与定位完整、硬质量门通过、目标 View 仍为 BUILDING、输入 Version 仍可用。通过后才按 [Parse View 生命周期](14_step_1_parse_view_reparse_lifecycle.md) 激活候选 View/Evidence，更新 Member 与 Job。

该回调只表示 `PARSE` 成功；响应使用 `status=PARSED,index_status=PENDING`。后续 `INDEX` Job 完成前，客户端不得把该文档当作已可检索。

`fail` 提交受控失败类别：`TRANSIENT`（网络、限流、短暂依赖）、`PERMANENT`（格式损坏、密码保护、不支持）、`POLICY`（安全或配置拒绝）。KC 决定是否退避重试、最终标记 View/Member 失败或等待人工处理；Worker 不自行修改 Job 尝试次数或 Revision 状态。

任一调用若 Job 已过期、输入已被补传/重解析替代、View 已删除、Version 已隔离，KC 返回 `409 JOB_LEASE_INVALID/JOB_STALE`。Worker 只能清理自身临时资源，不能再次 claim 该 Job 的旧租约。

## 安全、限额与兼容

- Worker 身份通过 mTLS 或短期服务令牌认证，并在 KC 注册允许的 MIME/能力范围内领取任务。
- 结果批次限制数量、正文大小、图片/工件大小与并发；超限使用 `output_manifest_uri` 分片，不允许无限大 HTTP Body。
- `policy_snapshot` 是 Parse View 的不可变快照。Worker 升级不会改变已领取任务的策略；新策略必须新建候选 View。
- 协议版本通过媒体类型或 `dto_version` 明示。KC 在 3.5 首期仅接受冻结版本；不为旧 V1 Chunk DTO 做兼容映射。
