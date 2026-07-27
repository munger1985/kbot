# 4.0 AIOps 步骤 5：监控接入与只观测闭环

## 目标与边界

本步骤接入 Prometheus、Zabbix 和 OEM，形成两条确定性链路：

```text
Chat/API Run → SCOPE → OBSERVE → REPORT

Verified Webhook → Inbox → Event/Alert
                 → Policy → SCOPE → OBSERVE → REPORT
```

本阶段只读取监控数据并生成可追溯 Artifact，不连接目标数据库、不调用 LLM、不推断根因、不生成 SQL 或变更建议。报告只能描述观测事实、覆盖范围、异常和数据缺口，`root_cause_level` 固定为 `INCONCLUSIVE`。

Prometheus Alertmanager 和 Zabbix Webhook 支持自动触发；OEM 首期支持指标、可用性和 Incident 查询作为 Observe 输入。OEM 自动 Incident 轮询留到 Scheduler 阶段，除非部署侧能把 OEM Incident 转换为经过认证的 Webhook。

## 现有代码评估

`utils/monitor` 中 Prometheus、Zabbix、OEM 和 Metric Registry 可以提供协议参考及响应 Fixture，但不能直接迁入生产路径：

- Provider 从全局配置读取 Endpoint/明文凭据，而 4.0 必须使用 Monitor Source + SecretRef；
- 接口接受已渲染的任意 Query 字符串，无法证明 Metric、Template、Target 和参数均已授权；
- 每次查询创建新的 `aiohttp.ClientSession`，缺少服务级连接池、总预算和统一重试；
- `MetricResult` 缺少 Unit、Query Window、质量、截断、映射版本和完整 Provenance；
- 非法数值被转换为 `0.0`，会把坏数据伪造成真实零值；
- Zabbix 找不到 Host 时按 Item Key 全局搜索，可能读取其他 Target；
- OEM/Prometheus 日志会输出 URL、Query 或 Provider 错误，存在标签和拓扑泄露；
- Registry 同时承担兼容字段、字符串格式化和 LLM Prompt 生成，职责混杂。

因此只迁移经过验证的协议细节、Metric 定义和脱敏响应样本。新代码完成后直接删除 `utils/monitor` 及其生产引用，历史实现由 Git 保存。

## MonitorPort

Domain/Application 不接触 PromQL、Zabbix JSON-RPC 或 OEM URL：

```python
class MonitorPort(Protocol):
    async def health_check(
        self, request: MonitorHealthRequest
    ) -> MonitorHealthResult: ...

    async def query_metrics(
        self, request: MetricQueryRequest
    ) -> MetricQueryResult: ...

    async def query_alerts(
        self, request: AlertQueryRequest
    ) -> AlertQueryResult: ...

    async def verify_and_parse_webhook(
        self, request: RawWebhookRequest
    ) -> NormalizedWebhookBatch: ...
```

请求只包含 `source_id/source_version/target_binding_id/metric_codes/window/limits/trace_id` 等类型化字段。Adapter Factory 根据已校验的 `source_type` 构造 Provider Context，在调用前通过 Secret Store 解析短期凭据；凭据、原始 Query 和 HTTP Client 不进入 Task/Artifact。

Provider 返回结构化结果或稳定错误：

```text
MONITOR_AUTH_FAILED
MONITOR_UNREACHABLE
MONITOR_RATE_LIMITED
MONITOR_QUERY_UNSUPPORTED
MONITOR_TARGET_NOT_FOUND
MONITOR_NO_DATA
MONITOR_RESPONSE_INVALID
MONITOR_RESULT_TRUNCATED
```

`NO_DATA` 是有来源的有效观测，不等于 `0`；连接失败和响应解析失败也不能返回空成功。

## Metric Catalog

建立只读、版本化 `MonitorMetricCatalog`：

```text
metric_code
semantic_version
name, description
unit, value_kind, expected_dimensions
supported_db_types
provider_definitions:
  prometheus: template + required_labels
  zabbix: exact_item_key + value_type
  oem: target_type + metric + collection + value_columns
allowed_aggregations
default_window, min_step, max_points, max_series
quality_rules
```

Catalog 在启动时通过 JSON/YAML Schema 校验并计算 Manifest Hash。Run Snapshot 和 Artifact 保存 Catalog Version、Metric Definition Hash、Provider Template ID/Version，不保存由 LLM 生成的 Query。

模板参数来自 `TARGET_MONITOR.EXTERNAL_TARGET_KEY` 和受控 Mapping Overrides。Prometheus Label Value 使用专用 Escaper；Zabbix 必须按绑定的精确 Host/Item 查询，禁止全局 Item 搜索；OEM Target/Metric 路径分别 URL Encode。禁止使用通用 `str.format` 拼接不可信参数。

首个 Baseline Pack 按数据库类型定义稳定 Metric Code：

```text
db.availability
db.cpu.utilization
db.connection.active
db.connection.utilization
db.transaction.throughput
db.response.latency
db.storage.utilization
db.error.rate
```

Provider 不支持某项时生成 `UNSUPPORTED` Gap；不能用意义相近但单位不同的指标静默替代。

现有 `configuration/metrics_mapping.yaml` 通过一次性审核脚本转换，例如 `db_availability → db.availability`、`db_active_sessions → db.connection.active`。Active Count 不能冒充 Connection Utilization；后者只有在 Catalog 定义了可靠分母时才生成。无法补齐 Unit、Value Kind、Provider Version 或精确 Target 参数的条目拒绝导入。运行时不加载旧文件，也不保留旧字段兼容分支。

## 标准观测模型

`MetricObservation.v1` 至少包含：

```text
metric_code, semantic_version, unit, value_kind
window: [start_at, end_at)
requested_step, effective_step
source_id, source_type, source_version
target_id, target_binding_id, external_target_fingerprint
series[]:
  dimensions
  points[{observed_at, value, quality}]
summary: count/min/max/avg/p95/last
coverage: expected_points/actual_points/ratio
truncated, warnings[]
provenance:
  catalog_hash, template_id/version, request_hash
  provider_response_hash, collected_at, adapter_version
```

时间统一为 UTC，窗口使用左闭右开。范围过大时根据 `max_points` 提升 Step，并在结果中记录实际 Step；绝不静默截断。NaN、Infinity、无法解析值和未知单位记录为 Invalid Sample/Gap，不转换为零。

维度 Label 必须经过 Catalog Allowlist、长度限制和稳定排序；Instance、Host、SQL、URL、Token 和高基数随机 Label 不直接进入 Artifact。原始 Provider 响应在安全策略允许时写受控对象存储，数据库 Artifact 只保存 URI、Hash 和脱敏标准结果。

统计摘要由确定性代码计算，不由 LLM 解释。不同 Provider 的同一 Metric 分别保存 Source Provenance；只有 Unit、Semantic Version、Dimension 语义和 Window 兼容时才允许生成对比摘要，不能把多个来源的点拼成一条伪时间序列。

Counter 必须按 Catalog 规则计算 Rate，并检测 Counter Reset；样本不足时输出 Gap，不能直接用首尾差值。P95 等统计量达不到最小样本数时返回 `null + warning`，禁止用 Last Value 伪装。

## 多监控源选择

Target 的 Active Monitor Binding 按：

```text
ROLE: PRIMARY 优先于 SUPPLEMENTARY
→ PRIORITY 升序
→ TARGET_MONITOR_ID 稳定排序
```

每个 Binding 生成独立 Observe Task，同一 Run 可并行。Primary 负责首选覆盖，Supplementary 用于补充缺失 Metric 和交叉验证；Secondary 值不同不代表 Primary 错误，报告必须并列展示来源。

单个 Provider/Metric 失败时 Observe Task 生成 Observation + Gap 集合并以成功结束，从而允许最终 `PARTIAL` 报告；只有 Handler 自身契约损坏、Artifact 无法持久化或全部配置不可用才让 Task 失败。Runtime 不判断 Provider 失败是否可降级。

## Run Blueprint

创建 Run 时根据当前 Active Binding 和 Catalog 生成确定性 Blueprint：

```text
scope
 ├─ observe:{binding_id}:baseline
 ├─ observe:{binding_id}:availability
 └─ ...
          ↓ ALL_SUCCEEDED
report
```

Task 数量超过 Run 上限时，按 Binding 合并 Metric Pack，而不是丢弃来源。Run Snapshot 冻结 Binding ID/Row Version、Source ID/Row Version、Catalog Hash、Metric Pack 和 Query Window；配置后续修改不改变已创建 Task。

每个 Run 同时冻结 Provider 调用数、Source 数、Metric 数、Series 数、Points 数、响应字节和总采集时间预算。达到上限时停止继续分页并生成带精确原因的 Truncation Gap；Adapter 不能自行扩大预算。

Observation Window 由类型化请求或 Trigger Policy 给出：

- Chat/API 未明确时间时使用部署默认窗口；
- Alert 以 `occurred_at` 为锚点使用策略化前后窗口；
- Window End 和数据库当前时间在 Run 创建时冻结；
- 步骤 5 不从自然语言自由解析“最近一阵子”等时间表达。

最终 `OBSERVE_REPORT.v1` 只包含可用性、Metric 摘要、Active Alert、Source 覆盖、Gap、时间窗口和 Provenance。禁止出现“根因是”“建议执行”“SQL”等诊断性表述。

## HTTP Client 与 Secret 生命周期

- App Bootstrap 创建长期复用的异步 HTTP Client/Connector；Adapter 不在每次 Query 新建 Session；
- 连接池按 Provider Origin/TLS Profile 隔离，设置连接、首字节、读取和总超时；
- Endpoint 必须来自已验证 Monitor Source，经过 Scheme/Host/Port Allowlist 和部署网络策略；
- Secret 每次用 Source Version + SecretRef 获取，最多使用短 TTL 内存缓存；缓存只保存 Provider Credential Object，不写日志/Artifact；
- GET/只读 JSON-RPC 仅对连接中断、429 和选定 5xx 做有限重试，遵守 `Retry-After` 和 Run Deadline；
- 本地 Circuit Breaker 只是减少无效调用，数据库 Health 状态才是共享事实；
- Provider 原始错误映射为稳定 Code，日志只记录 Source UUID、请求 Hash、状态码和 Trace。

认证失败不自动回退到匿名访问或其他 Secret。OEM Token Refresh、Zabbix Login 和 Prometheus Bearer/mTLS 都在 Adapter 内部完成，但不能修改领域状态；Health Result 由 Application Service 另行条件写回。

## Webhook 信任链

入口固定为：

```text
Main API: size/content-type/rate/trace checks
  → AIOps: hash route key，解析唯一 Monitor Source
  → Secret Store: 解析 WEBHOOK_SECRET_REF
  → Provider Adapter: 对原始 bytes 验签/认证
  → 保存经验证的 Raw Payload/Hash
  → UoW: Inbox + Event + Alert + optional Run/Task/Outbox
```

Webhook Key 只是不可枚举路由，不是唯一认证因素。每个 Source 必须配置 HMAC、Bearer/mTLS Gateway Identity 等认证方案；签名校验使用原始 bytes、时间戳和常量时间比较，拒绝超出重放窗口的请求。无效请求返回 `401/403`，不创建 Inbox/Event；Secret Store 暂不可用返回 `503` 让 Provider 重试。

Main API 删除客户端伪造的 Provider/Domain/Target Header，只向内部契约转发允许的签名 Header、Raw Body、Route Key、Received At 和 Trace。Route Key、签名、Bearer 和原始正文不得进入访问日志。

## Normalization 与 Target 映射

Adapter 由已解析 Source Type 决定，禁止根据 Payload 内容猜测 Provider。一个 Alertmanager Batch 可规范化为多个 Event：

```text
NormalizedMonitorEvent {
  source_event_key
  external_target_key
  event_type
  event_status: FIRING | RESOLVED | INFORMATIONAL
  severity: INFO | WARNING | HIGH | CRITICAL
  occurred_at
  fingerprint
  summary
  provider_attributes
  normalizer_version
}
```

- Prometheus 优先使用 Alert Fingerprint；缺失时对稳定 Label 子集、StartsAt 和状态计算 Key；
- Zabbix 使用 `eventid`，External Target 使用严格 Host 技术名；
- OEM 使用 Incident ID 和 Target Name；
- Severity 使用版本化映射表，原始值保存在受限 Provider Attributes；
- Fingerprint 使用 `source_id + external_target_key + normalized_problem_key` 计算，不信任请求体直接提交的 Domain/Target ID。

External Target 的提取规则来自受控 Source/Adapter 配置：Prometheus 使用登记的 Label Name（默认 `instance`），Zabbix 使用 Host 技术名，OEM 使用 Target Name。Payload 不能通过 `target_id/domain_id` 或动态 Label Name 改写规则；需要非标准标签时，由管理员在 Monitor Binding 的 Mapping Overrides 中显式配置并通过 Provider Schema 校验。

Target 只能通过同一 Source 下 Active `TARGET_MONITOR` 的精确 `external_target_key` 匹配，数据库以 `(source_id, external_target_key)` 唯一约束消除正常配置歧义。找不到映射时 Inbox 标记 `IGNORED`；若数据损坏导致多条则标记 `FAILED` 并告警。两者都不做模糊匹配、Target Key 回退或跨 Source 搜索。

## Inbox、Event 与 Alert 事务

经验证的 Delivery 先以 `source_id + raw_request_hash + provider_delivery_id` 写 Inbox，重复 Delivery 返回原 Receipt。每个 Normalized Event 再通过 `(monitor_source_id, source_event_key)` 唯一约束去重。

Event 增加可空 `SOURCE_INBOX_ID` 和必填 `NORMALIZER_VERSION`，关联原始验证请求。该字段不建外键，因为 Inbox 的保留期短于 Event；Event Payload/Hash 自足，Inbox 清理后仍可验证规范化事实。原始内容只在保留期内存在于 Inbox URI/Hash。

处理单个 Event 时锁定 Target/Fingerprint 对应 Alert：

1. 插入或读取幂等 Event；
2. 创建或更新当前 Active Alert；
3. `FIRING` 推进 Last Seen、Severity 和 Count；
4. `RESOLVED` 仅在事件时间不早于当前状态依据时关闭，乱序旧事件只关联不回退状态；
5. 根据 Active Policy、Severity、Cooldown 和现有活跃 Run 决定是否创建 Run；
6. 同事务写 Run/Tasks/Event/Outbox。

不存在 Active Alert 时，并发事务可能同时尝试 Insert；以 Active Alert 函数唯一索引裁决，失败方捕获唯一冲突后重新读取并锁定已创建行，不能把冲突当成接入失败。

默认仅 `CRITICAL` 自动创建 Observe Run。Alert 行锁串行化同一 Fingerprint 的并发事件；同一 Alert 已有活跃 Run 时只关联新 Event，不重复创建。Alert Storm 通过 Source/Target/Fingerprint 速率限制与 Policy Suppression 控制，但经过认证的事实 Event 不因限流而伪装成未发生。

自动 Run 使用服务器配置的 `system_aiops_agent_id` 并重新验证 Active Agent–Target Binding；Payload 不能选择 Agent。没有有效 Binding 时保留 Event/Alert，记录 `AUTO_RUN_NOT_AUTHORIZED`，不扩大权限。

## Source Health

Target、Monitor Source 和 Monitor Binding 分离 `ROW_VERSION` 与 `HEALTH_VERSION`：前者只保护配置/ETag，后者只保护运行健康归并。单次 Query 结果先形成 Artifact，随后 Application Service 使用 Config Row Version + Health Version 条件更新健康摘要并只递增 Health Version：

- 成功不立即把长期 `UNREACHABLE` 提升为 `HEALTHY`，按恢复阈值收敛；
- Auth Failure 直接标记 `DEGRADED` 并触发配置告警，不在 Run 中暴露原因；
- 连续连接失败达到阈值后为 `UNREACHABLE`；
- No Data 影响 Binding/Metric Coverage，不等同 Source 不可达；
- 多 Worker 只提交 Observation，Health Reducer 确定性计算共享状态。

旧 Source Version 的迟到 Health Result 不得覆盖新配置。Health 状态只影响新 Observe 选择和报告，不修改历史 Artifact。

三层 Health 语义不能混用：

- Monitor Source Health：Provider API/认证是否可用；
- Monitor Binding Health：External Target Mapping 是否存在、是否能返回预期数据；
- Target Health：仅由来源已验证的 Availability Observation 归并。

Provider 不可达时 Source 可为 `UNREACHABLE`，但 Target 必须是 `UNKNOWN` 而不是“数据库宕机”；External Object 消失使 Binding `UNREACHABLE`。多个有效来源对 Target 可用性结论冲突时 Target 为 `DEGRADED` 并保留冲突 Artifact。

## Artifact 与报告

步骤 5 新增：

```text
METRIC_OBSERVATION.v1
ALERT_OBSERVATION.v1
AVAILABILITY_OBSERVATION.v1
OBSERVATION_GAP.v1
OBSERVATION_SET.v1
OBSERVE_REPORT.v1
```

Artifact 的 `TRUST_LEVEL=SOURCE_VERIFIED`，但 Trust 表示来源已验证，不表示指标一定正确。所有 Artifact 保存 Source/Binding/Catalog/Adapter Version 和采集窗口；Report 引用 Artifact ID，不复制完整点序列。

Report Result 只允许 `READY/PARTIAL/FAILED`：

- `READY`：所有必需 Metric 有足够覆盖；
- `PARTIAL`：存在 Unsupported/No Data/Provider Failure/Truncation；即使没有有效采样，只要 Gap 完整且可追溯，仍生成 `PARTIAL/INCONCLUSIVE` 报告；
- `FAILED`：报告契约、持久化或 Runtime 本身失败，无法形成可信 Artifact。

`READY/PARTIAL` Report 都使 Observe Run 进入 `COMPLETED`；Provider 不可用不能把一份完整记录了 Gap 的报告伪装成系统执行失败。只有 `FAILED` 才按步骤 4 的 Task/Run 错误语义收敛。

步骤 5 的 Report 是最终 Artifact，不提前写 `KBOT_OPS_REPORT` 长期投影；步骤 10 再建立日报、周报、Incident 和 Comparison 投影。

## 代码布局

```text
services/aiops_agent/src/aiops_agent/ports/monitor.py
services/aiops_agent/src/aiops_agent/domain/monitoring/
  metrics.py
  observations.py
  events.py
  health.py
services/aiops_agent/src/aiops_agent/adapters/monitoring/
  base.py
  prometheus.py
  zabbix.py
  oem.py
  registry.py
  normalizers/
services/aiops_agent/src/aiops_agent/application/monitoring/
  observe.py
  webhook_intake.py
  alert_correlation.py
  health_reducer.py
services/aiops_agent/src/aiops_agent/contracts/artifacts/monitoring.py
services/aiops_agent/src/aiops_agent/tests/monitoring/fixtures/
services/aiops_agent/src/aiops_agent/resources/metrics/
```

Provider Adapter 不 import Repository/UoW；Application 不 import `utils.monitor`。Catalog 与响应 Fixture 可由现有配置迁移后逐项审核，禁止保留旧兼容字段解析分支。

## 测试矩阵

- 三种 Provider 的即时/范围/空值/非法值/截断/限流/认证失败 Fixture；
- Prometheus Label、Zabbix Item、OEM Path 参数注入与转义；
- Zabbix Host 不存在时严格失败，绝不全局搜索；
- Client Session 复用、Deadline、Retry-After、连接池和 Secret 轮换；
- 多 Source 排序、并行、部分失败、单位不兼容和 Coverage；
- Webhook 假签名、过期时间戳、旧/新 Route Key、Body 重放和批量 Event；
- Source Event Key 去重、Target 精确映射、Resolved/Firing 乱序；
- 同 Fingerprint 并发只创建一个 Active Alert/Observe Run；
- 无 Agent Binding、停用 Target、Suppression/Cooldown 时不自动创建 Run；
- Provider Error 生成 Gap/Partial Artifact，不绕过 Runtime 重试和事务；
- Report 不包含根因、SQL、命令、Secret 或未脱敏 Label；
- API/Worker 重启后 Inbox、Event、Alert 和 Observe Run 可继续恢复。

## 完成定义

- Prometheus、Zabbix、OEM 均通过同一 MonitorPort 产生版本化 Observation；
- Alertmanager/Zabbix 经认证 Webhook 可幂等形成 Event/Alert 和 Critical Observe Run；
- 多 Provider 的来源、时间窗、单位、质量、截断和 Gap 全部可追溯；
- Provider/Secret/网络失败只影响相应 Artifact 与 Health，不破坏 Run 内核；
- 所有自动 Run 都经过 System Agent Binding 和 Policy，Payload 不能扩大权限；
- 只观测报告不声称根因，不生成 SQL 或操作建议。

## 实施结果

步骤 5 已于 2026-07-23 完成，代码按以下边界落地：

```text
services/main_api/src/main_api/api/integrations.py
  → services/aiops_agent/src/aiops_agent/api/intake
  → application/monitoring/webhook_intake.py
  → Inbox / Event / Alert / Outbox
  → AIOpsDomainOutboxSink
  → monitor.observe-report@1
  → monitor.scope / monitor.observe / monitor.report
```

监控正文在 Main API 按流式大小上限读取，Route Key 在访问日志中脱敏；AIOps
收到的只有 Route Key Hash、允许的签名 Header 和原始字节。验签成功后正文写入
内容寻址的私有不可变存储，Inbox 只保存 URI、Hash 和安全摘要。开发与单机部署
使用 `LocalMonitorPayloadStore`，生产配置必须指向持久化加密卷；OCI/S3
实现只需替换同一 Port。

Run 创建没有复制到 Intake 事务。Event/Alert 与
`OPS_ALERT_AUTO_RUN_REQUESTED` 在同一事务提交，Outbox Dispatcher 再幂等调用
唯一的 Runtime Service。这样在保持至少一次恢复的同时，Run/Task 状态机仍只有
一个写入者。Alert ID 同时作为 Outbox 与 Run 幂等边界，同一 Active Alert 不会
生成多个 Run。

当前实现冻结每个 Active Binding 的 Source 版本、Catalog Hash、指标定义、窗口
和调用预算。每个 Binding 一个 Observe Task，报告等待全部来源；无来源时仍生成
可追溯的 `PARTIAL/INCONCLUSIVE` 报告。Source、Binding 和 Target Health 使用
配置版本与 Health Version 栅栏归并，多来源 Availability 冲突时 Target 为
`DEGRADED`。`tests/smoke/smoke_aiops_monitoring.py` 已在真实 Oracle Schema 验证
Webhook 重放、Alert、Outbox、动态 Run、三个 Task 和最终
`OBSERVE_REPORT.v1`，并在结束后清理全部测试数据。
