# AIOps 运维资料上传合同

两端 AIOps 与知识库 App 统一支持一次选择一个或多个独立文件。每个文件独立登记、审批、解析、发布和检索，不创建资料组。

## Multipart 字段

公共入口为 `POST /api/v1/apps/aiops/operations-knowledge/uploads`；内部入口为 `POST /internal/v1/aiops/operations-knowledge/uploads:ingest`。必须携带 `Idempotency-Key`，由 Main API 原样转发 multipart。

- `knowledge_base_id`：所属运维知识库。
- `grouping_mode`：可省略，默认为 `EACH_FILE`。显式请求其它值会被拒绝。
- `metadata`：JSON 对象，单文件请求使用 `display_name`（去除首尾空白后 1–256 个字符）；多文件请求使用各文件声明的 `display_name`；`security_level` 为 1–5。
- `files`：JSON 文件声明数组，每个声明包含 `part_name`、`client_file_id`、`display_name`、`byte_size`、`content_sha256`，对应同名二进制分片。

接口每次最多接收 20 个文件、总大小 64 MiB，并核对每个文件的大小和 SHA-256。页面逐文件提交，每个文件生成独立幂等键并显示接收结果。

每个接收回执引用独立的 `asset_id` 与 `asset_version_id`。AIOps 用资料版本主键约束每个版本只映射一个规范文档版本；两 App 的规范文档版本都只保存一个原文件描述和存储引用。诊断案例关联的报告、运行记录及证据来源不属于上传文件。

## 异步标签识别

手册审批和标签识别重试在任务状态提交后即返回当前资料版本，不等待正文解析或模型结果。审批冻结执行规格；重试沿用该规格，并将解析任务重置和版本进入 `TAGGING` 一起提交。已有可复用的解析结果不会被重置。

现有 AIOps Worker 消费 `TAGGING` 版本。页面收到受理确认后关闭抽屉，并刷新列表中的处理中资料。可重新打开资料查看最新版本的进度与失败原因。对 `TAGGING` 调用 `:reconcile` 只返回版本状态，不会在请求中执行识别。

待审批资料可多选后批量审批。全选只覆盖当前列表，确认时仅提供所选版本共同可用的解析模式。提交仍使用每份资料的版本号与原有审批接口；各项独立受理，未确认受理的资料会逐项列出。

## 发布边界

前端与服务端配套发布，Oracle 规范 Schema 62 收紧 AIOps 文档映射。已在 Schema 61 的环境使用 `database/oracle/operations/apply_aiops_schema_62.sql`；脚本发现历史版本包含多个文件时，在执行 DDL 前停止，无自动删除或拆分。

隔离回归覆盖 multipart 校验、独立文件回执、拒绝分组、单文件映射及页面追加文件。这些检查不代表线上模型解析、索引和发布全链路验收。
