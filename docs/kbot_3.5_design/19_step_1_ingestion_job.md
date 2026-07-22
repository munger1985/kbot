# 步骤 1 详细设计：Ingestion Job

## 定位

`KBOT_KC_INGESTION_JOB` 是 KC 的持久化任务账本和租约队列，用于驱动接收后的解析、画像、索引、关系构建及清理。它记录工作意图与执行结果，不是业务事实表，也不是跨服务共享数据库的入口：Parser 只能接收 KC Dispatcher 发送的版本化 HTTP 任务，回调 KC 的受控内部接口。

```text
Member / Version / Parse View 状态变化
  → KC 在同一事务创建或唤醒 Job
  → Parser / KC Worker 按能力向 KC claim 有限租约
  → Worker 回调 KC，KC 校验 Job + 输入版本后写入状态和产物
```

## 表字段

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `ingestion_job_id` | `NUMBER(38)` PK identity | 任务标识 |
| `collection_id` | 非空 `NUMBER(38)` | 所属 Scope |
| `bundle_id`, `bundle_revision_id` | 可空 `NUMBER(38)` | Bundle/Revision 级任务的关联锚点 |
| `bundle_revision_document_id` | 可空 `NUMBER(38)` | Member 级接收、解析或索引任务锚点 |
| `document_version_id`, `parse_view_id` | 可空 `NUMBER(38)` | 内容/视图级任务锚点 |
| `job_type` | `VARCHAR2(32)` 非空 | `PARSE/PROFILE/INDEX/RELATE/COLLECTION_PURGE/VIEW_CLEANUP` |
| `idempotency_key` | `VARCHAR2(256)` 非空 | 同一逻辑工作单元的稳定键 |
| `input_fingerprint` | `VARCHAR2(64)` 非空 | 输入内容、策略和目标版本的规范化 hash |
| `payload_json` | JSON CLOB 可空 | Worker 所需的非敏感参数和对象引用，不存凭据 |
| `job_status` | `VARCHAR2(16)` 非空 | `PENDING/RUNNING/RETRY_WAIT/SUCCEEDED/FAILED/CANCELLED` |
| `priority`, `available_at` | 数字、带时区时间戳 | 调度优先级与下次可领取时间 |
| `attempt_count`, `max_attempts` | `NUMBER(8)` 非空 | 已尝试次数与有限重试上限 |
| `lease_owner`, `lease_until`, `heartbeat_at` | 可空 | Worker 租约与失联回收依据 |
| `started_at`, `completed_at` | 带时区时间戳 | 首次运行和最终完成时间 |
| `result_json` | JSON CLOB 可空 | Evidence 数量、画像 hash、耗时等可观测结果 |
| `failure_class`, `failure_code`, `failure_message` | 可空 | `TRANSIENT/PERMANENT/STALE/CANCELLED` 与受限摘要 |
| `trace_id` | `VARCHAR2(128)` 可空 | API 请求、Dispatcher 与下游调用关联 |
| `row_version` + 审计列 | 基础约定 | 租约/回调条件更新与审计 |

`idempotency_key` 表达“要做什么”，`input_fingerprint` 表达“以什么输入做”。相同键且相同 fingerprint 复用同一 Job 行并按状态返回；相同键但输入变化时，旧 Job 标记 `CANCELLED` 或 `STALE` 语义的失败记录后创建新 Job，不能让迟到回调覆盖新输入。

## 任务类型与依赖

- `PARSE`：目标为 Document Version + 候选 Parse View；成功后写入 STAGED Evidence。
- `PROFILE`：目标为 Bundle Revision；基于当前可用 Member/Version/View 构建 STAGED Discovery Object。
- `INDEX`：为 Evidence 或 Discovery 生成全文/向量索引；首期可由对应生成流程内联完成，但仍保留独立类型以便迁移。
- `RELATE`：目标为 Bundle Revision；只读取已激活或候选验证通过的解析产物。
- `VIEW_CLEANUP` 与 `COLLECTION_PURGE`：只做已撤销可见性对象的异步物理清理，必须可恢复、可重试。

KC Application Service 在状态改变的同一数据库事务中创建/唤醒下游 Job，避免“状态已提交但任务丢失”。依赖未满足时不领取，或转为 `RETRY_WAIT`；不以无限轮询代替明确依赖检查。

## 租约、回调与重试

Worker 通过 KC 的 claim API 以条件更新领取：仅 `PENDING/RETRY_WAIT` 且 `available_at <= now`、能力匹配的 Job 可被置为 `RUNNING`，同时写入随机 `lease_owner`、有限 `lease_until` 与 `row_version`。心跳只可续租同一 owner；租约到期的 RUNNING Job 由 Reaper 依据错误类别重新入队或终止。调度器如需存在，只负责唤醒/限流，不向 Parser 推送数据库任务。

Worker 回调必须携带 `ingestion_job_id`、`lease_owner`、`input_fingerprint` 和产物摘要。KC 只接受仍属该租约、输入未过期且目标对象仍可处理的回调；其他回调记审计并忽略。这样补传、重解析或删除后的迟到结果不能重新激活旧 Evidence。

可重试的网络、模型限流和暂时存储错误使用指数退避进入 `RETRY_WAIT`；格式损坏、策略校验失败等永久错误直接 `FAILED` 并更新 Member/Parse View 的对应失败状态。达到 `max_attempts` 后停止自动重试，保留显式重试入口；显式重试复用或重置同一逻辑 Job，但必须记录新的尝试审计。

## 约束与运维查询

- `UK(collection_id, idempotency_key, input_fingerprint)`；同一输入只保留一个逻辑 Job 行。
- 索引 `(job_status, available_at, priority)` 供领取；索引 `(lease_until, job_status)` 供失联回收；索引 `(bundle_revision_id, job_type, job_status)` 与 `(parse_view_id, job_type, job_status)` 供管理面诊断。
- Job 删除仅随 Collection Purge 发生；普通成功/失败记录按审计保留策略归档，不能立即删除，否则无法解释部分入库或重复投递。
