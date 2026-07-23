# AIOps 步骤 8：Chat 人工补证与可恢复 HITL

## 目标与边界

本步骤只为 `TRIGGER_TYPE=CHAT` 实现 `DATA_REQUIRED` 和 `MANUAL_DIAGNOSTIC_SQL`。当监控与已登记只读工具仍不足、目标数据库不可连接、权限不足或目录缺少必要诊断能力时，AIOps 可以请求当前对话用户补充上下文，或在目标数据库上手工执行受控只读 SQL 并回贴结果。

人工 SQL 永远不会发送给 DB Executor，不产生 Approval Token，也不等同于变更审批。Alert、Schedule 和 API 自动 Run 不创建此类 HITL；监控 Webhook 先形成 Alert Run，证据不足时直接生成 `PARTIAL/INCONCLUSIVE` 报告。

## 进入条件

`RoundDecision` 只有同时满足以下条件才可选择 `WAIT_FOR_CHAT_INPUT`：

1. Run 的原始 `TRIGGER_TYPE=CHAT`，不能由客户端命令修改；
2. `ASSIGNEE_USER_ID` 是创建对话 Run 的授权用户；
3. 已存在具体 `EVIDENCE_GAP`，并关联尚可区分的 Hypothesis；
4. 自动 Monitor/DB/KC 工具已尝试或明确不可用；
5. 请求预期能改变根因等级或排除关键假设；
6. HITL 轮次、数据量、Token 和 Run Deadline 仍有预算；
7. 能生成通过安全校验的响应 Schema 或 SQL。

模型表达“需要更多数据”本身不能创建 HITL。确定性 `InteractiveDiagnosisPolicy` 校验上述条件，并可将请求降级为最终 `INCONCLUSIVE`。

## 两类人工输入

### `DATA_REQUIRED`

用于补充事件开始时间、影响范围、拓扑、近期变更、是否主备切换等上下文。请求必须是版本化表单 Schema，字段类型、长度和枚举固定；不能索取密码、连接串、Token、私钥、完整日志包或未限定的数据库导出。

### `MANUAL_DIAGNOSTIC_SQL`

用于获取一个明确数据库事实。每轮最多建议少量能区分假设的查询，不把用户当作通用 SQL 执行器。优先使用 Diagnostic Catalog 中已经评审的模板；只有目录无法回答必要问题时，才允许专用 LLM Role 生成候选 SQL。

无论 SQL 来源如何，用户都应看到用途、目标显示名、预期实例身份、预计开销、超时提示、敏感字段和“仅使用只读账号执行”的说明。

## 可恢复状态机

```text
DIAGNOSIS_ROUND_ASSESSMENT
          ↓ Evidence Gap
CREATE_INPUT_REQUEST
  ├─ MANUAL_SQL_REQUEST / DATA_REQUEST Artifact
  ├─ KBOT_OPS_HITL(PENDING)
  ├─ Task → WAITING_INPUT，清除 Lease
  └─ Run → WAITING_INPUT + Event
          ↓
 User response / skip / expiry
          ↓
VALIDATE_AND_ACCEPT_RESPONSE
  ├─ Response/Gap Artifact
  ├─ HITL → ANSWERED/CANCELLED/EXPIRED
  ├─ waiting Task → SUCCEEDED/EXPIRED
  ├─ Run → DIAGNOSING
  └─ 创建 Normalize/Evidence Index/Next Round Task
```

等待期间没有 Worker 持有 Lease，也不序列化 Python Context、Prompt 或调用栈。所谓恢复，是从 Run/Task/HITL/Artifact 创建新的后继 Task，继续同一个 Ops Run；不重新执行旧 Handler。

`SuspendTaskForInputCommand` 在一个 UoW 中锁定顺序为 Run → Task → HITL，写入请求 Artifact/HITL、将 Task 从 `RUNNING` 迁移为 `WAITING_INPUT`、清除 Lease、将 Run 置为 `WAITING_INPUT` 并追加用户事件。唯一 `(OPS_RUN_ID, IDEMPOTENCY_KEY)` 和 Pending 函数索引防止重复请求。

## Manual SQL Request

SQL 不直接保存在 `HITL.PROMPT_TEXT` 或 SSE Event 中，而保存为不可变 `MANUAL_SQL_REQUEST.v1` Artifact；HITL 的 `INPUT_ARTIFACTS_JSON` 引用它。

```text
hitl_id, run_id, round_no, target_id
db_type, db_version, expected_instance_identity
evidence_gap_refs[], hypothesis_ids[]
queries[] {
  query_id, origin: CATALOG | MODEL_GENERATED
  purpose, diagnostic_question
  sql_text, sql_sha256
  expected_columns[], expected_types[], expected_shape
  required, max_rows, timeout_hint_seconds, cost_warning
  sensitivity_labels[]
  supports_if, contradicts_if
}
instructions, parser_version, expires_at
```

Artifact 的 Provenance 保存 Catalog/Prompt/Model 版本、验证器版本和输入 Hash。一次 HITL 最多包含少量相关查询；回复必须为每个 required Query 提交 `SUCCEEDED/FAILED/SKIPPED` 之一。更正已接受的结果必须创建新 Task、HITL 和修订 Artifact，并引用旧结果，不能覆盖原记录。

## SQL 生成与校验

Catalog SQL 通过专用 `ManualSqlRenderer` 生成可复制语句，值使用方言安全字面量渲染；标识符只能来自模板 Variant。Model SQL 使用独立 `ManualSqlCandidate.v1` Schema，并经过比普通代码展示更严格的离线规则：

- Oracle/MySQL 首期都只允许单条 `SELECT/WITH`；
- 只允许已登记诊断视图与系统对象，例如 Oracle 动态性能/受控字典视图，以及 MySQL `performance_schema/information_schema/sys` 的明确 Allowlist；
- 禁止业务表、DB Link、用户函数、PL/SQL、Procedure、DML、DDL、锁定读取、会话修改和多语句；
- 禁止注释/Hint、`SELECT INTO`、文件输出、网络/外部访问、Sleep/Benchmark 和可能产生副作用的函数；
- 禁止 `EXPLAIN ANALYZE`；需要执行计划时只能使用单独评审的 Catalog 流程；
- 必须包含受控行数范围；查询历史数据还必须包含 Scope 内时间条件；
- AST、对象、函数、复杂度、预计结果 Schema 和敏感字段策略全部通过校验；
- SQL 长度、Join/子查询数量和 Cost Level 超限时拒绝展示。

静态验证不能证明 SQL 在所有数据库上低成本，因此 UI 必须提示超时并允许用户自行终止。若候选 SQL 无法安全通过，不创建 HITL，Run 以 Gap 结束。用户即使修改后自行执行，回贴结果仍只按 `USER_PROVIDED` 处理。

为降低“执行在错误实例”的风险，Manual Request 必须包含预期实例身份；Catalog/候选查询应尽可能返回数据库唯一名、实例名或 Server UUID。身份不匹配的结果不能归入目标 Evidence：接口返回 `422`，只审计提交 Hash 和拒绝原因，HITL 保持 `PENDING`。此机制仍不是密码学证明，因此不会改变 `USER_PROVIDED` 信任等级。

## 用户回复契约

正常回复：

```json
{
  "expected_row_version": 4,
  "responses": [
    {
      "query_id": "oracle.session_waits.v2:q1",
      "status": "SUCCEEDED",
      "format": "CSV",
      "upload_id": "019c...",
      "inline_data": null,
      "error": null
    }
  ],
  "execution_attestation": {
    "executed_at": "2026-07-23T10:30:00+08:00",
    "target_display_name": "ERP-PROD",
    "used_readonly_account": true
  },
  "note": "在主库执行"
}
```

同一 Query 只能选择 `upload_id` 或受大小限制的 `inline_data`。`FAILED` 提交稳定错误类别和短错误信息，不要求伪造空表；`SKIPPED` 必须给出原因。错误文本和 Note 都是不可信数据，限制长度并在进入模型前脱敏。

普通聊天文本不会自动视为 HITL 回复。前端可以提供自然的聊天式表单，但提交时必须附带 `hitl_id`、`expected_row_version` 和 `Idempotency-Key`，防止一句“好了”误恢复错误 Run。Root Agent 只能转发待输入通知，不能代替用户回答。

## 上传和内容检查

推荐接口：

```text
POST /v4/ops/hitl/{hitl_id}/uploads
POST /v4/ops/hitl/{hitl_id}/uploads/{upload_id}/complete
POST /v4/ops/hitl/{hitl_id}/responses
POST /v4/ops/hitl/{hitl_id}/skip
```

上传会话绑定 `actor/domain/run/hitl/query_id/format/max_size/security_level/expiry`，使用一次性签名 Upload Grant。AIOps 通过 `UploadPort` 管理暂存对象，不在 `KBOT_OPS_*` 增加通用文件表。对象先进入隔离区，完成 MIME/Magic、Hash、编码、病毒/内容检查后才能被回复命令引用。

首期支持受控 `CSV/JSON/TEXT_TABLE`：

- Inline 默认不超过 64 KiB；文件、行、列、字段和 JSON 深度上限由配置锁定；
- 不接受 XLSX、压缩包、可执行文件、HTML、宏、外部链接或多文件容器；
- CSV/JSON 列必须与 Request Schema 匹配，未知列按策略拒绝而不是喂给模型；
- `TEXT_TABLE` 只使用与预期列绑定的确定性 Parser，不让 LLM猜测表格结构；
- NUL、控制字符、超长字段、公式前缀和异常编码在展示前中和；
- 原始隔离对象短期保留且访问受限；长期 Artifact 只保存脱敏、规范化结果和原始 Hash。策略需要时可完全删除原始对象。

未完成、被拒绝或并发输掉的 Upload 由 Lifecycle Job 清理。对象存储和扫描调用发生在事务外；最终 UoW 再次验证 HITL 状态和 Upload Binding。

## 接受回复的事务

`RespondToHITLCommand` 分为事务外准备和事务内接受：

1. Main API 验证用户身份、Scope 和请求大小；
2. AIOps 只读加载 HITL Snapshot；
3. 事务外读取已完成 Upload，执行解析、Schema、实例身份、脱敏和结果限界；
4. 生成规范化结果及 Hash；
5. 新 UoW 按 Run → Task → HITL 加锁，重新校验 Assignee、`PENDING`、Expiry、Row Version、Idempotency 和 Upload Binding；
6. 插入 `USER_DIAGNOSTIC_SUBMISSION.v1`、每个 Query 的 `USER_PROVIDED_DB_RESULT.v1` 或 `USER_PROVIDED_DB_ERROR.v1`，Trust Level 为 `USER_PROVIDED`；
7. 更新 HITL `ANSWERED`、等待 Task `SUCCEEDED`，创建 `HITL_OUTCOME.v1` 和后继 Evidence Index/诊断 Task；
8. Run 回到 `DIAGNOSING`，追加内部审计和用户可见事件后提交；
9. Commit 后发布/保留对象并发送 Outbox。

输入结构不合格时返回 `422 HITL_RESPONSE_INVALID`，HITL 保持 `PENDING`；只记录无正文的拒绝事件和稳定 Field Error。格式有效但查询执行失败是一个可接受的诊断事实，不返回 422。

相同 Idempotency Key 和请求 Hash 返回原 Receipt；相同 Key 不同内容返回 `409 IDEMPOTENCY_CONFLICT`。并发回复只有一个获胜，其他返回 `409 HITL_ALREADY_RESOLVED`。过期返回 `410 HITL_EXPIRED`，跨 Domain/非 Assignee 统一返回 `404`。

## 跳过、过期与取消

`skip` 表示放弃当前补证但继续生成现有证据下的报告：

- HITL → `CANCELLED`，`RESPONSE_JSON` 保存结构化 Skip 原因；
- 等待 Task 以 `HITL_SKIPPED.v1/EVIDENCE_GAP.v1` 正常收敛；
- Run 回到 `DIAGNOSING` 并直接进入最终评估。

取消整个 Run 仍调用 Run Cancel Command，不能复用 `skip`。HITL `EXPIRES_AT` 不晚于 `RUN.DEADLINE_AT`；等待不消耗工具/模型调用预算，但消耗墙钟时间。

Reconciler 对过期请求使用同一锁顺序，原子将 HITL/Task 标记 `EXPIRED`、写 Gap、创建 Finalize Task 并把 Run 恢复到 `DIAGNOSING`。回复与超时竞争时先提交者获胜，不能在两个分支都生成后继任务。

## 多轮与终止

接受结果后，只把新的 User Evidence 加入 Evidence Index，并重新评估受影响 Hypothesis，不重跑全部 Monitor/DB/SOP Task。下一轮只能请求新的、非重复、仍有区分度的最小数据。

循环在以下任一条件终止：

- 达到 Root Cause Sufficiency；
- 用户 Skip/无权执行/持续返回无法解析的结果；
- 请求指纹或 Evidence Index 无进展；
- 达到 HITL 轮次、总行数、字节、Token 或 Deadline；
- 无法生成安全 SQL。

达到轮次上限不会在当前 Run 中通过一句聊天文本扩大预算。系统输出 `PARTIAL/INCONCLUSIVE`，用户若确需继续应显式创建新的 Chat Run，并引用原报告。

## 自动 Run 的行为

`ALERT/SCHEDULE/API` 等自动流程：

- 不创建 `DATA_REQUIRED/MANUAL_DIAGNOSTIC_SQL`；
- 不进入 `WAITING_INPUT`，不产生用户待办；
- 保存 Observation、Hypothesis、反证和 Evidence Gap；
- 报告列出缺少的诊断问题或已登记 Tool ID，不生成未评审的临时 SQL；
- 以 `PARTIAL/INCONCLUSIVE` 正常结束，后续步骤 9 也不能因证据不足绕过根因等级门槛。

## API 与 SSE

```text
GET  /v4/ops/runs/{run_id}/pending-input
GET  /v4/ops/hitl/{hitl_id}
POST /v4/ops/hitl/{hitl_id}/uploads
POST /v4/ops/hitl/{hitl_id}/uploads/{upload_id}/complete
POST /v4/ops/hitl/{hitl_id}/responses
POST /v4/ops/hitl/{hitl_id}/skip
```

SSE `diagnostic.input_required` 只包含 `hitl_id/request_type/expires_at/request_artifact_ref`，不含 SQL、结果 Schema 或敏感正文。授权用户通过 GET 获取完整请求；重连后 `pending-input` 恢复 UI。所有写接口需要 `ops:diagnostic:respond`、Assignee、Domain、Agent/Target Binding、ETag/Row Version 和 Idempotency Key。

APEX 可以通过受控视图展示待输入计数，但 SQL、回贴和上传都必须调用 API，不能直接更新 `KBOT_OPS_HITL`。

## 代码布局

```text
aiops_agent/
  contracts/hitl/{requests,responses,uploads}.py
  domain/hitl/{transitions,interactive_policy,errors}.py
  application/hitl/
    suspend_for_input.py
    respond.py
    skip.py
    expire.py
  orchestration/hitl/
    manual_sql_builder.py
    response_normalizer.py
  ports/{upload,content_inspection}.py
  adapters/{object_storage_upload,content_inspection}.py
  tests/hitl/
```

Manual SQL Validator 复用步骤 6 的 AST/方言基础设施，但采用独立、更严格的对象 Allowlist；不能 import Legacy `sql_validator` 或 `sql_to_run` 流程。

## 实施顺序

1. 固化 Data/Manual SQL Request、Response、Upload 和 Outcome Schema；
2. 实现 InteractiveDiagnosisPolicy、SQL Builder/Validator 和请求 Artifact；
3. 实现 Suspend/Respond/Skip/Expire Command 及锁竞争；
4. 实现 UploadPort、隔离、内容检查、规范化、脱敏和生命周期；
5. 接入 Evidence Index、Diagnosis Round 和 Root Cause Grade Policy；
6. 实现 Public/Internal API、SSE 投影和 APEX 待办只读视图；
7. 使用 Oracle SQL*Plus、SQLcl、MySQL CLI 和常见 CSV/JSON 样本完成兼容测试；
8. 删除 4.0 运行时对旧 Pending Snapshot、`sql_to_run` 和内存 Timeline 的引用。

## 验收门槛

- 非 Chat Run 无法创建或回复 Manual Diagnostic HITL；
- SQL 不进入 Executor，HITL 不产生 Approval Token；
- Model SQL 的多语句、业务表、Hint、DB Link、文件/网络、过程和副作用函数全部被拒绝；
- SSE、日志和 APEX 视图不泄露 SQL、上传正文或敏感数据库结果；
- 跨用户/Domain、过期、重复、乱序和并发回复均由稳定规则处理；
- Worker/API 重启后可从表和 Artifact 恢复，不依赖 Python Context；
- 错误格式不会消费 HITL，合法的数据库执行错误会成为 User Evidence；
- 上传炸弹、CSV 公式、恶意 JSON/Text、Prompt Injection 和超限内容不能进入模型；
- User Evidence 始终保留 `USER_PROVIDED`，不能单独把根因提升为 `CONFIRMED`；
- Skip、Expiry、无进展和预算耗尽都能生成可解释的 `PARTIAL/INCONCLUSIVE` 报告。
