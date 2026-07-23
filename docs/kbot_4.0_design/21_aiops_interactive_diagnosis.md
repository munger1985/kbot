# 4.0 AIOps Chat 交互式人工诊断循环

步骤 8 的可实施级 HITL 状态机、Manual SQL 安全规则、上传、回复事务和恢复设计见 [37_aiops_step8_chat_manual_diagnosis_hitl.md](37_aiops_step8_chat_manual_diagnosis_hitl.md)。

## 适用场景

本功能只适用于 `TRIGGER_TYPE=CHAT` 的 AIOps 对话，不适用于告警、定时巡检、Webhook 或其他自动发起的 Ops Run。

当 AIOps Agent 可读取 Prometheus、Zabbix 或 OEM，但 Target 没有配置只读数据库凭据、连接不可用或权限不足时，Agent 先使用已有监控事实。只有 Evidence Sufficiency Check 明确证明缺少某项数据库事实时，才请求当前对话用户执行诊断 SQL，不把用户当成无目标的查询执行器。

## 诊断循环

```text
Monitor/SOP Observation
          ↓
Evidence Sufficiency Check
    ├─ SUFFICIENT → Root-cause Analysis → Solution
    └─ INSUFFICIENT
             ↓
       Diagnostic Gap Artifact
             ↓
  Read-only SQL Plan + Safety Validator
             ↓
   MANUAL_DIAGNOSTIC_SQL HITL
             ↓
     WAITING_INPUT / 暂停租约
             ↓
 User executes SQL and submits result
             ↓
 Parse + Validate + User-provided Artifact
             ↓
      Resume same Ops Run with new Tasks
             └───────────────┘
```

每轮只请求能最大化缩小根因范围的最小查询集。恢复时复用已有 Observation、SOP 和上轮结果 Artifact，不重跑整个诊断，也不从对话文本猜测状态。

## 请求与回复契约

`MANUAL_SQL_REQUEST` 是不可变 Artifact：

```text
hitl_id, ops_run_id, round_no, target_id
db_type, version_code, diagnostic_gap
queries[]: query_id, purpose, sql, expected_columns,
           expected_shape, timeout_hint, sensitivity_hint
instructions, expires_at, content_hash
```

用户通过 `hitl_id` 回传 `query_id`、执行状态、结果文本/表格/CSV 或文件、错误信息和可选备注。大结果存对象存储，Artifact 仅保存 URI、Hash 和摘要。同一 HITL 只接受一次有效回复；更正数据需创建新的 Task/HITL 和修订 Artifact，并保留原结果。

## SQL 生成与安全

Agent 优先从 Oracle/MySQL `DatabaseDialect` 选择版本化诊断模板；模板无法覆盖时，可生成仅供人工执行的自定义只读 SQL。即使 Target 已配置只读连接，这类自定义 SQL 也不会转交 Executor。所有请求在展示前必须经过语法树和策略校验：

- Oracle/MySQL 首期都只允许单条 `SELECT/WITH`，`SHOW/EXPLAIN ANALYZE` 不进入临时生成路径；
- 只允许受控诊断视图和系统对象，不查询业务表；
- 禁止 DML、DDL、PL/SQL、存储过程、文件/网络访问、会话修改和多语句；
- 限制时间窗口、返回行数和预计耗时，对高开销查询显式警告；
- SQL 不包含凭据，并按 Target 的数据库类型和版本生成；
- 该路径由用户自主执行，不生成 Change Approval Token，永远不转交 DB Executor。

预置目录与直连执行的完整契约见 [22_aiops_database_diagnostic_catalog.md](22_aiops_database_diagnostic_catalog.md)。

## 结果可信性与 Prompt 隔离

回贴结果生成 `USER_PROVIDED_DB_RESULT`，Provenance 必须记录用户、时间、Query Hash 和 `trust_level=USER_PROVIDED`。系统先检查列结构、行数、类型和敏感数据，再转换为结构化 Observation。

用户输入和查询结果一律作为不可信数据，不能把其中的文字当成 Agent/Tool 指令。解析失败或与预期 schema 不匹配时，HITL 保持 `PENDING` 并请求更正，不由 LLM 虚构缺失列。

## 终止与恢复

循环以产出 `CONFIRMED/PROBABLE` 根因及可执行解决方案为正常终点。用户可随时跳过、取消或回复无权执行；Agent 应基于已有证据生成 `INCONCLUSIVE` 报告，而不是永久挂起。

每个 Run 设置可配置轮次、Token、时间和数据量预算。到达轮次上限时不猜测根因，也不通过普通聊天文本扩大当前 Run 的预算；需要继续时显式创建引用原报告的新 Chat Run。Worker 崩溃后仍从 HITL + Artifact 恢复，不重复消费已提交的回复。

## 自动触发流程

Alert、Schedule、Webhook 和其他自动 Run 不创建 `MANUAL_DIAGNOSTIC_SQL` HITL，不进入 `WAITING_INPUT`，也不生成待用户领取的队列。当监控证据不足且数据库不可读时，它们应：

1. 保存已采集的 Observation 和 `DIAGNOSTIC_GAP`；
2. 在报告中明确列出已知事实、可能原因、缺失证据和已登记 Tool/诊断问题，不生成临时 SQL；
3. 将报告标记为 `PARTIAL`、结论标记为 `INCONCLUSIVE`；
4. 正常结束 Run，不影响后续告警和巡检调度。

## API 与前端交互

- Chat SSE 只返回 `diagnostic.input_required`、`hitl_id`、过期时间和请求 ArtifactRef，不携带 SQL 正文；
- `GET /v4/ops/runs/{run_id}/pending-input` 在对话重连后恢复待回复请求；
- `GET /v4/ops/hitl/{hitl_id}` 经授权读取完整请求；
- `POST /v4/ops/hitl/{hitl_id}/responses` 提交表格、文本、CSV/文件或执行错误；
- `POST /v4/ops/hitl/{hitl_id}/skip` 放弃等待并生成当前证据下的结论。

只有当前 Chat Run 的授权对话用户可提交结果。响应接口使用乐观锁和幂等键；过期 Request 不能恢复旧 Task，需要时由 Agent 重新生成并校验新 SQL。
完整请求 DTO、上传和错误契约见 [27_aiops_api_and_contracts.md](27_aiops_api_and_contracts.md)。
