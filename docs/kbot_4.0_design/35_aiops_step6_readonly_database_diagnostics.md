# AIOps 步骤 6：只读数据库诊断目录与执行链路

## 目标与边界

本步骤为 Oracle/MySQL 建立可审计、可版本化、可限流的数据库只读诊断能力，并将结果写成 AIOps Artifact。此时不接入 LLM Planner、不执行变更、不支持 PostgreSQL，也不提供任意 SQL 接口。步骤 7 才允许 LLM 从已登记工具中选择 `tool_id + parameters`，具体版本由冻结的 Catalog Snapshot 解析；步骤 8 生成的临时 SQL 也只展示给 Chat 用户手工执行。

只读诊断不写 `KBOT_OPS_EXECUTION`。它是 `KBOT_OPS_TASK` 的外部读取行为，由 Task 租约和 Artifact 唯一约束保证提交幂等；变更/回滚继续使用独立 Execution、审批 Token 和执行链路。

## 现状审计与取舍

3.x 代码可提供 DBA 意图、Oracle/MySQL 差异和测试样本，但不能直接迁移运行时：

- `DatabaseDiagnosticTools` 将多版本 SQL 嵌在 Python 方法中，缺少模板版本、输出 Schema、许可声明和稳定 Hash；
- DB Executor 请求携带原始 `sql`、`connection_config` 和密码，扩大了信任面；
- `SQLValidator` 依赖正则和首词检查，并修改 SQL 注入 `LIMIT/FETCH`，不能可靠识别注释、函数副作用、嵌套语句和方言差异；
- Mutation 白名单使用字符串包含判断，存在明显越权风险；
- 驱动使用 `fetchall()`/DataFrame，缺少字节、列、单元格和内存上限；MySQL 查询路径没有实际传入 bind 参数；
- 日志和错误响应可能暴露 SQL、DSN、账号及数据库原始错误。

因此旧工具逐条进入迁移清单，结论只能是 `REWRITE/REJECT/DEFER`。4.0 不保留 `/api/v1/execute`、`/api/v1/ops/execute` 或任何兼容入口。

## 组件边界

```text
AIOps Task Handler
  ├─ 读取 Target/Binding/Policy/Task 快照
  ├─ Diagnostic Registry 解析 tool_id
  ├─ 签发短期 DiagnosticExecutionGrant
  └─ DB Executor Client
          ↓ mTLS + Service Identity
AIOps DB Executor
  ├─ Grant Verifier
  ├─ 本地 Diagnostic Registry / Template Validator
  ├─ Secret Provider
  ├─ Oracle/MySQL Driver
  └─ Result Normalizer / Redactor
          ↓ bounded DatabaseObservation
AIOps Task Handler
  └─ 事务性保存 Artifact、Event 并完成 Task
```

DB Executor 不持有 KBot Schema 凭据、不查询 Run/Task/Target 表、不推进状态机，也不保存领域事实。Worker 不获取目标数据库明文密码。双方安装同一版本的只读诊断资产；请求中的版本和 Hash 必须与 Executor 本地资产完全一致。

建议目录：

```text
aiops_agent/
  diagnostics/
    contracts.py
    registry.py
    grants.py
    validation.py
    dialects/
      oracle/{manifest.yaml,sql/,normalizers.py}
      mysql/{manifest.yaml,sql/,normalizers.py}
  orchestration/handlers/database_diagnostic.py
  adapters/db_executor_client.py
apps/aiops_db_executor/
  app.py
  settings.py
  executor/{service.py,limits.py,redaction.py}
  drivers/{base.py,oracle.py,mysql.py}
```

## 版本化 Diagnostic Tool Catalog

工具 ID 描述诊断意图，不包含数据库名称，例如：

```text
db.instance.identity
db.session.active
db.session.blocking_chain
db.transaction.long_running
db.wait.current
db.statement.top_load
db.storage.capacity
db.temp.usage
db.replication.status
```

每个方言可为同一工具提供多个互斥 Variant。Registry 按 `db_type + db_version + capabilities + entitlements` 精确选择一个 Variant，不做模糊降级。

```yaml
tool_id: db.session.blocking_chain
version: 1.0.0
db_type: oracle
variant: oracle_19c_single_instance
supported_versions: ">=19,<27"
required_capabilities: [dynamic_performance_views]
required_privileges: [select_catalog_role_subset]
required_entitlements: []
parameter_schema: blocking_chain.input.v1
output_schema: blocking_chain.output.v1
template_ref: sql/session/blocking_chain_19c.sql
template_sha256: "..."
normalizer: oracle_blocking_chain_v1
timeout_seconds: 8
max_rows: 200
max_bytes: 1048576
cost_level: MEDIUM
sensitive_columns: [username, machine, program, sql_text]
```

Manifest、SQL、参数 Schema、输出 Schema 和 Normalizer 同版本发布且不可原地修改。启动时 Registry 必须校验：

1. ID/版本/Variant 唯一，引用文件存在且 Hash 匹配；
2. 参数 Schema 拒绝未知字段，并限定长度、范围、枚举和格式；
3. SQL 解析为单条 `SELECT/WITH`，不存在 DDL、DML、PL/SQL、存储过程、文件输出、锁定读取或多语句；
4. 所有值均使用 bind，占位符与参数 Schema 一一对应；
5. 动态标识符只能从 Manifest 枚举映射到受评审模板 Variant，不能由字符串替换生成；
6. 输出 Schema、敏感列策略、超时和结果上限完整。

AST/Token 校验是发布和启动时的辅助防线，不是只读安全的唯一依据。数据库最小权限账号、只读会话、固定资产和网络隔离共同构成安全边界。

## 首批工具包与能力探测

首批工具分四组实施：

| 工具包 | 内容 | 默认开销 |
| --- | --- | --- |
| Identity | 产品、版本、角色、实例/集群身份、当前时间 | LOW |
| Concurrency | 活跃会话、长事务、等待、锁和阻塞链 | LOW/MEDIUM |
| Capacity | 表空间/数据文件、临时空间、连接与容量 | LOW/MEDIUM |
| HA | Oracle RAC/Data Guard、MySQL Replica/Group Replication | MEDIUM |

Top SQL、执行计划、历史会话和大对象排名属于扩展包，完成真实数据库开销评测后才启用。Oracle AWR/ASH/Diagnostic Pack/Tuning Pack 相关视图必须声明 `required_entitlements`，默认禁用，不能因账号可查询就推断已获许可。MySQL `performance_schema`、`sys` 和复制视图同样通过能力探测确认。

每个 Run 首先执行固定的 `db.instance.identity`，形成不可变 `DB_CAPABILITY_SNAPSHOT.v1`。能力来自数据库实测与 Target 管理员声明的 entitlement 求交；客户端参数和 LLM 均不能扩大能力。版本不支持、权限不足和功能未启用是不同的结构化 Gap。

## Diagnostic Execution Grant

Worker 在 Task 租约内，根据已提交快照签发短期 JWS Grant：

```text
grant_id, audience, issued_at, expires_at
run_id, task_id, lease_token_hash
target_id, target_version, db_type
connection_profile, diagnostic_secret_ref
tool_id, tool_version, variant, template_sha256
parameters_sha256, limits, trace_id
```

`connection_profile` 只含受签名保护的主机、端口、服务名/数据库名和 TLS Profile，不含账号密码。Grant 的过期时间不得晚于 Task Lease，单次查询超时必须短于 Grant 剩余寿命。Executor 校验签名、issuer、audience、时间、Service Identity、参数 Hash、本地模板 Hash 和上限后，才从 Secret Store 解析 `diagnostic_secret_ref`。

只读请求使用：

```text
POST /internal/v1/db-executor/diagnostics
{
  "schema_version": "diagnostic-execution-request.v1",
  "executor_request_id": "uuidv7",
  "grant": "<signed-jws>",
  "parameters": {...},
  "idempotency_key": "..."
}
```

请求模型设置 `extra=forbid`，因此 `sql`、`connection_config`、密码、用户 Token、自然语言、Policy 覆盖值或自定义 limit 会直接返回 `400`. 只读查询允许网络重试造成重复读取，但同一请求不得改变模板或参数；最终 Artifact 写入仍由 Task lease token fencing，过期 Worker 的结果会被丢弃。

## 只读执行护栏

Executor 按以下顺序执行，任一步失败均不建立更高权限连接：

1. 验证调用身份、Grant、目录版本、Target 类型和参数；
2. 解析只读 SecretRef，建立带 TLS 的目标连接；
3. 设置数据库级 statement timeout、只读事务/会话和标识信息；
4. 使用方言原生 bind 执行已加载模板；
5. 以 `fetchmany()` 流式读取至 `max_rows + 1`，同时限制列数、总字节和单元格长度；
6. 规范化类型、校验输出 Schema、脱敏并计算结果 Hash；
7. rollback/reset；超时、取消或协议异常时销毁连接，不放回池。

禁止通过改写用户 SQL追加 LIMIT。模板自身可包含经过评审的范围约束，而最终结果上限由客户端流式读取强制执行。Oracle 和 MySQL Driver 均使用命名参数的方言适配，业务代码不得自行替换占位符。

数据库账号是首要防线：

- 诊断 Secret 与未来 Mutation Secret 必须分离，任何失败都不能回退到变更凭据；
- 账号只授予明确视图/系统表查询权限，不授予 DDL、DML、过程执行、文件、调度或管理权限；
- Oracle 禁止 PL/SQL、`FOR UPDATE` 和可产生副作用的包/函数，使用只读事务及 Resource Manager/服务级限制；
- MySQL 使用只读事务、最小 GRANT、执行时间限制，禁止用户变量、临时表、`INTO OUTFILE/DUMPFILE` 和多语句；
- 目标网络、Secret namespace 和 Executor Service Identity 分环境隔离。

连接池以 `target_id + target_version + secret_fingerprint + tls_profile` 分区，设置很小的空闲和生命周期上限。Secret 轮换或 Target 版本变化立即废弃旧池。日志只记录 Target ID、Tool ID、模板 Hash、耗时、行数和稳定错误码，不记录 SQL、bind 值、地址、账号、DSN 或原始数据库错误。

## 结果与错误契约

Executor 返回有界 `DatabaseObservation.v1`：

```text
executor_request_id, target_id
tool_id, tool_version, variant, template_sha256
db_type, db_version, capability_snapshot_hash
captured_at, duration_ms
columns[{name, logical_type, sensitivity}]
rows, row_count, truncated, result_sha256
parameters_sha256, warnings, provenance
```

时间统一为 UTC，Decimal 使用无损字符串和逻辑类型，二进制/LOB 默认拒绝，未知列或不兼容类型视为输出契约错误。空结果是有效事实；截断结果必须标记 `truncated=true`，不能伪装为完整数据。敏感列先按确定性规则屏蔽/Hash，再离开 Executor。

Worker 验证响应身份、请求 ID、Task 租约和 Hash 后，保存：

- 成功：`DATABASE_OBSERVATION.v1`，Trust Level 为 `SOURCE_VERIFIED`；
- 能力/权限/连接/超时问题：`EVIDENCE_GAP.v1`；
- 必要时同时保存 `DB_CAPABILITY_SNAPSHOT.v1`。

稳定错误码至少区分 `AUTH_FAILED`、`TARGET_UNREACHABLE`、`TIMEOUT`、`PRIVILEGE_MISSING`、`VERSION_UNSUPPORTED`、`CAPABILITY_UNAVAILABLE`、`ENTITLEMENT_REQUIRED`、`OUTPUT_SCHEMA_INVALID`、`RESULT_LIMIT_EXCEEDED` 和 `EXECUTOR_INTERNAL_ERROR`。原始错误只进入受限安全日志。

连接不可用或证据缺失不自动令 Run 失败：诊断 Task 生成 Gap 后完成，后续步骤决定 `PARTIAL/INCONCLUSIVE`。只有 Grant/目录契约损坏、Artifact 无法持久化等系统错误才按 Task 重试策略失败。Fallback 由 Handler 按冻结在 Task Input 的确定性候选链显式调度；Executor 不在内部偷偷换 SQL。

## Run/Task 集成

步骤 6 增加确定性 `DB_DIAGNOSTIC_BASELINE.v1` Blueprint：

```text
SCOPE
  → DB_CAPABILITY_DETECT
  → DB_DIAGNOSTIC[identity/concurrency/capacity/ha]
  → DATABASE_OBSERVATION_AGGREGATE
  → REPORT
```

只有同时满足以下条件才创建 DB Task：

- Target 为 ACTIVE，Task 所属 Agent 与 Target Binding 允许诊断；
- Target 配有 `DIAGNOSTIC_SECRET_REF`，Policy 和 Trigger 允许只读直连；
- Tool Variant 与数据库版本、能力、entitlement 匹配；
- Run 预算、Target 并发和 Cost Level 未超限。

同一 Target 的 HIGH 成本工具串行，LOW/MEDIUM 使用小并发限额；Run Deadline 始终优先。步骤 6 的 Blueprint 不调用 LLM。Alert/Schedule 缺少直连条件时生成 Gap；Chat 的人工补证循环留到步骤 8。

## 迁移与实施顺序

1. 建立契约、Grant 验证器、Registry 和离线目录校验 CLI；
2. 实现无网络的 Fake Dialect/Driver 契约测试；
3. 实现 Oracle Identity/Concurrency，再实现 MySQL 对等工具；
4. 加入 Capacity/HA、能力与 entitlement Gate；
5. 实现 Executor 限界、脱敏、错误映射和 Secret/TLS Adapter；
6. 接入 Task Handler、Artifact/Event 和确定性 Baseline Blueprint；
7. 对 3.x 的 17 个工具形成逐条迁移矩阵，通过评审的 SQL 才进入目录；
8. 在受控 Oracle/MySQL 环境执行权限、版本和开销验收。

旧 DB Executor 和内嵌 SQL 不被新包 import；仅审查过的行为与测试样本可迁入新目录，原实现随后直接删除并由 Git 历史留档。

## 验收门槛

- API 契约证明任意 SQL、连接配置、密码和未知字段无法进入 Executor；
- Manifest/Hash、重复 ID、参数 bind、单语句和危险 AST 均有失败测试；
- Oracle/MySQL 覆盖支持版本、缺权限、功能关闭、空结果、超时、取消、截断和 Secret 轮换；
- SQL 注入、标识符注入、第二语句、文件输出、存储过程和锁定读取全部被拒绝；
- 结果在离开 Executor 前完成 Schema 校验和脱敏，日志/错误/SSE 不含 SQL 与 Secret；
- Executor 无 KBot Schema 连接，Worker 无目标数据库明文凭据；
- stale Grant/Task Lease 的结果不能提交，重试不重复写 Artifact；
- Oracle 许可特性默认禁用，只有显式 entitlement 才能选择对应工具；
- Target 不可达时自动 Run 仍产出可解释的 `PARTIAL/INCONCLUSIVE` 报告。
