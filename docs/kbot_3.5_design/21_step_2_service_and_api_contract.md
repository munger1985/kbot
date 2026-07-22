# 步骤 2 详细设计：KC 服务、认证与 API 契约

## 服务边界

新增独立 FastAPI 进程 `kbot_app_knowledge.py`（建议端口 `18090`）。它是 `KBOT_KC_*` 的唯一写入者，负责 Domain/Collection Scope 校验、对象发布、版本状态机、任务租约和 V2 查询。它不暴露 SQLAlchemy Session、不代理 V1 上传 API、也不让 Portal/Parser/APEX 直写 KC 表。

路由分为三类：

| 路由 | 调用者 | 认证与用途 |
| --- | --- | --- |
| `/healthz`、`/readyz` | 编排/监控 | 不返回配置、凭据或数据统计 |
| `/api/v2/knowledge/*` | APEX、Portal、Main API/Skill | 用户或受控服务身份；执行 Domain 权限校验 |
| `/internal/v2/knowledge/*` | Parser、KC 运维 Worker、Agent 管理后端 | mTLS 或短期服务令牌；按服务能力授权 |

`app_id` 不来自任何 HTTP 请求，而由 KC 的 `base.toml` 服务配置注入。所有带 `{domain_id}` 的端点先验证认证主体能否访问该 Domain，再解析 `collection_key → collection_id`；下游对象 ID 也必须反查属于同一 Domain。响应中的 ID 可供后续受限查询，不能成为绕过 Scope 的授权凭据。

## 外部入库 API

外部写入按 Adapter 语义分为 KM Asset 与普通用户文件上传，共用 KC Application Service，但不共用含混的“上传一批文件”契约。

```http
POST /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}/ingestions/km-assets
Content-Type: multipart/form-data
Idempotency-Key: <client request UUID>
```

此端点仅接受已授权的 KM Portal 服务身份；`source_system=metadb`、`source_type=KM_ASSET` 固定在服务器端。multipart 内容如下：

| Part | 格式 | 要求 |
| --- | --- | --- |
| `bundle` | JSON | `source_id`、`source_revision`、`title`、`security_level`、Facet、metadata；禁止 Scope 和来源类型字段 |
| `documents` | JSON 数组 | `part_name`、`external_document_id`、角色、声明名称/MIME、顺序、必需标记、来源 URL、`byte_size`、`content_sha256` |
| `<part_name>` | 文件字节 | 必须与一个 documents 条目一一对应 |
| `document_failures` | JSON 数组，可空 | 未取得字节的已声明附件、失败码和受限说明 |

同一附件只能同时出现在文件 Part 或 `document_failures` 中；每个声明附件都必须二选一，杜绝静默遗漏。文件的实际大小和 SHA-256 必须与声明一致。`__manifest__` 由 KC 创建，客户端不得上传或声明它。

成功接收返回 `202 Accepted`：

```json
{
  "bundle_id": 101,
  "bundle_revision_id": 301,
  "source_revision": "2026-07-22T10:15:00Z",
  "acceptance_status": "ACCEPTED",
  "status_url": "/api/v2/knowledge/domains/20/bundles/101"
}
```

`Idempotency-Key` 防止网络重试产生重复 HTTP 受理；来源幂等仍以 `source_revision + snapshot_fingerprint` 为准。相同来源修订但不同快照返回 `409 SOURCE_REVISION_CONFLICT`；非法 Domain/Collection 返回 `403/404`，格式或附件清单不一致返回 `422`，存储未能原子发布则返回可重试的 `503`，不得返回已接收。

普通用户入口为：

```http
POST /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}/ingestions/user-files
```

请求必须显式选择 `EACH_FILE`（N 个文件生成 N 个单文档 Bundle）或 `SINGLE_BUNDLE`（N 个文件生成一个含 N 个平权 CONTENT Member 的 Bundle）。前者允许逐项受理和 `PARTIAL_ACCEPTED`，后者在接收/发布阶段全有或全无；两者都不自动生成 MANIFEST Document。完整字段、幂等和响应见[普通用户文件上传 API](46_step_2_user_file_upload_api.md)。

## 状态与管理 API

```text
GET /api/v2/knowledge/domains/{domain_id}/bundles/{bundle_id}
GET /api/v2/knowledge/domains/{domain_id}/bundles/{bundle_id}/revisions/{bundle_revision_id}
GET /api/v2/knowledge/domains/{domain_id}/bundles/{bundle_id}/revisions/{bundle_revision_id}/members
POST /api/v2/knowledge/domains/{domain_id}/bundles/{bundle_id}/revisions/{bundle_revision_id}/members/{external_document_id}/retry
POST /api/v2/knowledge/domains/{domain_id}/bundles/{bundle_id}/revisions/{bundle_revision_id}/members/{external_document_id}/supplement
```

状态响应只返回当前状态、成员级失败码、可安全展示的摘要、时间戳和下一步允许操作；不返回存储 URI、下载令牌、内部 Job payload 或原始异常。`retry` 只重建可重试任务；`supplement` 只接受同一不可变 Member 声明的缺失内容，不能修改名称、角色、URL 或 `external_document_id`。

Collection 管理和 Agent Binding 端点沿用[步骤 0 API 设计](08_step_0_collection_and_binding_api.md)。Discovery/Evidence 查询端点属于步骤 5；本步骤不以临时“查 Chunk”接口替代它们。

## Worker 内部 API

```text
POST /internal/v2/knowledge/parse-tasks/claim
POST /internal/v2/knowledge/parse-tasks/{job_id}/heartbeat
POST /internal/v2/knowledge/parse-tasks/{job_id}/evidence-batches
POST /internal/v2/knowledge/parse-tasks/{job_id}/complete
POST /internal/v2/knowledge/parse-tasks/{job_id}/fail
```

`claim` 输入 Worker ID、支持 MIME、能力标签（Docling/OCR/VLM/Spreadsheet）、并发槽位；KC 返回一个或多个有限租约任务、不可变内容读取 URI 或短期受控读取令牌、Parse View 策略、`lease_owner` 与 `input_fingerprint`。Worker 绝不收到数据库连接信息。

所有后续调用均携带 `lease_owner + input_fingerprint`。`evidence-batches` 只接受版本化 DTO 并写入 STAGED Evidence；`complete` 提交质量报告和产物 URI 后触发 View 激活；`fail` 提交受控失败分类。过期租约、目标已删除/替换、输入不符或无该服务能力时返回 `409 JOB_LEASE_INVALID` 或 `409 JOB_STALE`，Worker 必须停止该任务。

## 统一响应与审计

所有写操作返回 `request_id`，并记录 `actor_type/actor_id`、Domain、Collection、目标 ID、V1/V2 路由版本、幂等键（如有）和结果码。错误体使用稳定的 `code/message/request_id`，不回传 Oracle 错误、堆栈、对象 URI 或模型凭据。

写操作默认异步：成功受理是 `202`，同步校验失败才是 4xx/5xx。只有纯管理更新（如 Collection 显示名）可返回 `200/204`。这一区分避免 Portal 将“已受理”误解为“已解析完成”。
