# AIOps 运维资料上传合同

上传界面使用“单个文件”和“多文件资料”。多文件资料仍是一份运维资料的一个版本，每个文件分别登记为本 App 的文档、摄取任务和索引，不创建公共 KC Bundle。

## Multipart 字段

公共入口为 `POST /api/v1/apps/aiops/operations-knowledge/uploads`；内部入口为 `POST /internal/v1/aiops/operations-knowledge/uploads:ingest`。必须携带 `Idempotency-Key`，由 Main API 原样转发 multipart。

- `knowledge_base_id`：所属运维知识库。
- `grouping_mode`：`EACH_FILE` 将每个文件登记为独立资料；`SINGLE_ASSET` 将本次文件登记为同一份资料的同一版本。页面单文件模式每次选择一个文件。
- `metadata`：JSON 对象，`display_name` 是唯一资料名称来源，去除首尾空白后须为 1–256 个字符；`security_level` 为 1–5。
- `files`：JSON 文件声明数组，每个声明包含 `part_name`、`client_file_id`、`display_name`、`byte_size`、`content_sha256`、`role`、`required_flag`，对应同名二进制分片。

文件顺序以声明数组顺序为准，服务端按顺序生成 `ordinal`。角色为 `CONTENT`（正文）或 `SUPPLEMENT`（补充材料），多文件资料至少包含一个正文文件。每次最多 20 个文件、总大小 64 MiB；服务端核对每个文件的大小和 SHA-256。

多文件上传的各文件回执引用相同的 `asset_id` 与 `asset_version_id`，不同原文保留独立的 `document_id`、`document_version_id` 和 `ingestion_job_id`。文件顺序、角色、必要性随版本保存。上传后仍需人工审批，不改变标签识别、解析与发布流程。

## 发布边界

旧 `SINGLE_BUNDLE` 模式不再接受，不读取 `bundle` / `client_bundle_id` 字段，不保留兼容分支。前端与 AIOps Agent 必须配套发布；KBot 页面同步更新上传脚本缓存版本。本次无需数据库变更或历史数据处理。

隔离回归覆盖 multipart 校验、分组回执、双端真实页面的单文件替换与多文件追加、角色和必要性。这些检查不代表线上模型解析、索引和发布全链路验收。
