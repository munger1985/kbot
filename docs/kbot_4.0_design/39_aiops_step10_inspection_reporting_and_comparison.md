# 4.0 AIOps 步骤 10：巡检、报告与处理前后对比

## 目标与边界

本步骤把定时巡检、故障/性能报告和处理前后对比接入既有 Run/Task/Artifact 内核。Scheduler 只负责确定“何时为哪些 Target 创建工作”，巡检仍复用 `SCOPE → OBSERVE → DIAGNOSE → REPORT`；它不直接访问监控工具、目标数据库或 LLM。

首期产出 `INCIDENT`、`PERFORMANCE`、`INSPECTION_DAILY`、`INSPECTION_WEEKLY` 和 `COMPARISON` 报告。Email/IM 仅保留 `ReportDeliveryPort` 与明确返回“不支持”的空 Adapter，不实现投递配置、队列或重试表。

## 对旧 Scheduler 的处理

已审计的 3.x Scheduler 仅作为需求样本，不进入 4.0；原代码直接删除，需要时从 Git 历史查阅：

- 移除代码内硬编码巡检规则、阈值和 PromQL；
- 不由 Scheduler 直接修改 HITL/Run 状态或提交数据库事务；
- 不通过 HTTP Webhook 模拟内部调度触发；
- 不在 Scheduler 中执行监控查询、诊断和报告生成；
- 不使用进程内列表记录触发历史。

HITL、Task、Execution 超时分别由其领域 Reconciler 处理。Scheduler 可以承载这些周期性扫描入口，但不能把不同状态机合并成一个 SQL 更新脚本。

## Inspection Plan 与 Fire

Plan 是可修改配置，Fire 是某个计划时点的不可变执行事实：

```text
Inspection Plan 1 ── N Inspection Target
        │
        └── N Inspection Fire 1 ── N Ops Run（每个 Target 一个）
```

新增 `KBOT_OPS_INSPECTION_FIRE`，避免只依赖 `PLAN.NEXT_RUN_AT` 推断某次调度是否已经展开：

```text
INSPECTION_FIRE_ID, INSPECTION_PLAN_ID
SCHEDULED_FOR
STATUS: QUEUED | RUNNING | COMPLETED | PARTIAL |
        FAILED | SKIPPED | CANCELLED
PLAN_ROW_VERSION, TEMPLATE_ID, TEMPLATE_VERSION
SCHEDULE_RESOLVER_VERSION
PLAN_SNAPSHOT_JSON, RESOLUTION_JSON
TARGET_COUNT, RUN_COUNT, COMPLETED_COUNT, FAILED_COUNT
SKIP_REASON, ERROR_CODE, ERROR_MESSAGE
STARTED_AT, COMPLETED_AT, CREATED_AT, UPDATED_AT, ROW_VERSION
```

唯一约束 `(INSPECTION_PLAN_ID, SCHEDULED_FOR)` 是调度幂等边界。`KBOT_OPS_RUN` 增加 `INSPECTION_FIRE_ID`，不再重复保存 `INSPECTION_PLAN_ID`；唯一约束 `(INSPECTION_FIRE_ID, TARGET_ID)` 保证同一 Fire 对一个 Target 只创建一个 Run。非 Schedule Run 的 Fire ID 为空。

Plan 增加：

```text
MISFIRE_POLICY: SKIP | LATEST_ONLY
SCHEDULE_RESOLVER_VERSION
LAST_SCHEDULED_FOR
```

`CRON_EXPRESSION` 对 DAILY、WEEKLY 和 CRON 均必填。API 将日报时间、周报星期与时间编译为规范五段 Cron；应用使用 IANA 时区解析，数据库不解析 Cron。所有实际时间以 UTC 保存。

时区解析必须确定性：不存在的本地时间顺延到下一个有效时刻；重复的本地时间选择第一次出现。解析决策和时区数据库版本写入 Fire 的 `RESOLUTION_JSON`。升级 Resolver 后只影响未来 Fire。

## 多副本调度与事务

Scheduler 使用数据库 `SYSTIMESTAMP` 判断到期，按 `(STATUS, NEXT_RUN_AT)` 小批量领取 Plan 租约。领取、展开和推进游标均使用租约 Token/Row Version fencing，进程本地时间不能决定所有权。

对无重叠的到期 Plan，一个短事务完成：

1. 校验 Plan 租约、状态和版本；
2. 插入唯一 Fire，并冻结 Plan、模板和 Target Binding 快照；
3. 为每个有效 Target 创建 Run、首个 Task、Run Event 和 Outbox；
4. 更新 Fire 计数及状态；
5. 将 `LAST_SCHEDULED_FOR/NEXT_RUN_AT` 推进到确定的下一时点；
6. 提交后才由 Worker 领取 Task。

激活 Plan 前限制有效 Target 数量，默认建议上限 100，实际值由部署配置决定；超过上限必须拆分 Plan，避免巨型事务。

### 错过调度与重叠

- `MISFIRE_POLICY=SKIP`：推进到未来时点，并为最近错过时点记录一个 `SKIPPED` Fire；
- `LATEST_ONLY`：只为最近错过时点创建一个 Fire，不回放全部历史时点；
- `OVERLAP_POLICY=SKIP`：上一 Fire 未终止时，新时点记录 `SKIPPED`；
- `QUEUE`：最多保留一个 `QUEUED` Fire；出现更新时点时，将旧排队 Fire 标记为 `SKIPPED/SUPERSEDED_BY_LATEST`，再创建新的排队 Fire，防止无限积压且不改写原 `SCHEDULED_FOR`。

Fire Reconciler 在前一 Fire 终止后展开排队 Fire，并聚合子 Run 终态。全部成功为 `COMPLETED`；存在成功/降级与失败混合为 `PARTIAL`；全部无法形成有效报告为 `FAILED`。暂停 Plan 只阻止未来 Fire，不取消已经创建的 Fire/Run。

## 版本化巡检模板

Inspection Template 是随代码发布、不可覆盖的版本化资产：

```text
template_id, template_version, supported_db_types
checks[]:
  check_id, required, capability, entitlement
  monitor_metric_codes[], diagnostic_tool_refs[]
  observation_window, baseline_window
  thresholds, timeout, cost, report_sections[]
```

`TEMPLATE_OVERRIDES_JSON` 只能启停可选 Check、调整模板声明为可覆盖的阈值或窗口；不能注入 SQL、PromQL、Provider Item、工具版本、安全等级或任意 Prompt。激活 Plan 时完成 Schema、能力、Target 类型和覆盖白名单校验，Fire 再冻结最终解析结果。

日报使用一个业务日的半开区间 `[start, end)`，展示健康摘要、异常、容量趋势和待处理事项。周报独立采集七日趋势、重复告警、未解决问题和前一周对比，不能拼接七份日报。所有窗口先按 Plan 时区确定边界，再转为 UTC；查询与报告同时记录本地边界和 UTC 边界。

## 巡检任务图

每个 Target Run 使用统一任务：

```text
SCOPE → OBSERVE → DIAGNOSE → REPORT
```

模板将 Check 展开为受限 Tool Plan，仍由步骤 5、6、7 的 Catalog、Adapter、Executor 和诊断状态机执行。可选 Check 失败时继续生成 `PARTIAL` 报告；必需数据源不可用但仍能生成合规内容时也为 `PARTIAL/INCONCLUSIVE`。只有报告契约无法构造、内容无法持久化或运行内核失败才标记 Report/Run 为 `FAILED`。

自动 Schedule Run 不进入人工 SQL 补证循环。证据不足时保留缺口并结束，不等待用户。

## 不可变报告与版本

`KBOT_OPS_REPORT` 是 APEX 可查询投影，权威内容是不可变 `REPORT_CONTENT.v1` Artifact。表结构调整为：

```text
REPORT_ID, OPS_RUN_ID, TARGET_ID
REPORT_KEY, REPORT_TYPE, REPORT_VERSION
SUPERSEDES_REPORT_ID, IS_CURRENT
TITLE, STATUS: GENERATING | READY | PARTIAL | FAILED
PERIOD_START, PERIOD_END
BASELINE_START, BASELINE_END, AFTER_START, AFTER_END
RESULT: IMPROVED | UNCHANGED | DEGRADED | INCONCLUSIVE
TEMPLATE_ID, TEMPLATE_VERSION, GENERATED_BY_TASK_ID
CONTENT_ARTIFACT_ID, CONTENT_HASH, SUMMARY, SECURITY_LEVEL
SCHEMA_VERSION, CREATED_AT, UPDATED_AT
```

`REPORT_KEY` 标识 Run 内的逻辑报告，例如 `incident`、`inspection.daily`、`comparison.execution.<execution_id>` 或 `comparison.solution.<group_key>`。唯一约束 `(OPS_RUN_ID, REPORT_KEY, REPORT_VERSION)`；函数唯一索引保证每个 `(OPS_RUN_ID, REPORT_KEY)` 仅一个 `IS_CURRENT=1`。

生成新版本时先插入非当前 `GENERATING` 行。内容验证并持久化后，在一个事务中将旧版本设为非当前、将新版本发布为 `READY/PARTIAL + IS_CURRENT=1`，并写 Event/Outbox。已经发布的内容、Hash、周期和结论不可修改；更正必须创建新版本并通过 `SUPERSEDES_REPORT_ID` 留痕。`FAILED` 草稿不能成为当前版本。

`REPORT_CONTENT.v1` 至少包含：

- Scope、时间窗口、Target/模板/工具版本；
- 已验证事实、假设、根因等级和 Evidence 引用；
- Check 覆盖率、失败项、数据缺口和截断说明；
- 建议动作、执行/审批状态及验证结果；
- 对比指标、主结果、护栏指标和结论依据；
- Model、Prompt、Catalog、Provider Mapping 与 Renderer Provenance。

LLM 只负责受 Schema 限制的叙述，不能修改数值、证据引用、根因等级、动作风险或对比结论。Grounding Verifier 拒绝无 Evidence 的断言。模型不可用时仍用确定性模板生成可读报告，并将叙述能力缺失记录为 `PARTIAL`。

完整报告通过授权 API 读取；APEX 视图只暴露安全摘要、状态、版本和内容引用。报告安全等级取 Target 与所有来源 Artifact 的最高值，正文不复制未脱敏原始 SQL 结果或秘密。

## Verification 与 Comparison

Verification 回答“命令是否产生预期直接效果”，Comparison 回答“处理后整体健康是否改善且没有明显副作用”，二者不能互相替代。

在动作执行前创建不可变 `COMPARISON_PLAN.v1`：

```text
action/execution/solution_group refs
primary_metrics[], guardrail_metrics[]
provider/tool/catalog/mapping versions
dimensions, aggregation, units
baseline_window, after_window, settle_delay
thresholds and result rules
required_evidence
```

系统执行时，Dispatcher 前完成基线采集；模板/Policy 声明为必需的基线缺失会阻止命令提交。Advisory 在 Proposal 就绪时采集可得基线。用户回填已人工执行但不存在处理前基线时，仍可生成 Comparison，但结论只能是 `INCONCLUSIVE`。

执行后以 `COMPARE` Task 的 `AVAILABLE_AT` 表示 settle delay，不让 Worker 睡眠占用租约。采集必须使用相同指标定义、Provider、Mapping、维度、单位、聚合方式和等长窗口。版本变化、数据缺口、采样截断或来源切换会将结论降为 `INCONCLUSIVE`。

数值计算和结论由确定性规则产生：

- 主指标达到改善阈值且护栏无显著退化：`IMPROVED`；
- 变化在容差内：`UNCHANGED`；
- 主指标恶化或任一严重护栏退化：`DEGRADED`；
- 数据不可比或不足：`INCONCLUSIVE`。

LLM 只能解释结果，不能重新判级。每个已执行动作生成动作级 Comparison；多动作方案结束后再生成一个 Solution Group 级 Comparison。报告必须说明观察窗口和因果限制，时间相关性不能表述为已证明的因果关系。

## API、APEX 与投递边界

- `GET /v4/ops/reports` 默认只返回当前版本，可按类型、Target、周期筛选；
- `GET /v4/ops/reports/{report_id}` 返回授权后的结构化内容引用和派生渲染；
- `GET /v4/ops/reports/{report_id}/versions` 返回同一 Report Key 的历史；
- `GET /v4/ops/inspection-fires` 和单 Fire 查询用于调度审计；
- SSE `report.ready` 只携带 Report ID、Key、Type、Version、Status 和摘要，不推送完整正文。

新增 `KBOT_V_OPS_INSPECTION_FIRE`；`KBOT_V_OPS_REPORT` 只投影 `IS_CURRENT=1`，历史版本通过 API 查询。`ReportDeliveryPort` 的 4.0 实现只允许 `SystemStoreDelivery`；Email/IM 配置即使误传也返回稳定 `OPS_DELIVERY_CHANNEL_UNSUPPORTED`，不能静默丢弃。

## 代码布局

```text
aiops_agent/
  scheduling/
    resolver.py
    template_registry.py
    overlap.py
  application/inspections/
    claim_due_plans.py
    create_fire.py
    expand_fire.py
    reconcile_fire.py
  application/reports/
    build_report.py
    publish_report.py
    compare.py
  contracts/
    inspection_template_v1.py
    report_content_v1.py
    comparison_plan_v1.py
    comparison_result_v1.py
  ports/report_delivery.py
  adapters/report_delivery/system_store.py
```

Repository 增加 `inspection_fires`，但计划领取、Fire 展开、报告发布和 Fire 汇总仍由 Application Service + UoW 定义事务。

## 验收场景

- 两个 Scheduler 同时领取同一 Plan，只产生一个 Fire 和每 Target 一个 Run；
- 在创建 Fire、部分 Run、推进 `NEXT_RUN_AT` 等崩溃点恢复后无重复或漏跑；
- 覆盖 UTC、IANA 时区、夏令时缺失/重复时刻、暂停恢复和 Resolver 升级；
- 验证 `SKIP/LATEST_ONLY`、`SKIP/QUEUE`，且排队数量有界；
- Target 在 Plan 激活后被停用时记录跳过原因，不创建不可执行 Run；
- Template/Override 非法、能力不足或超出 Target 上限时不能激活；
- 日报/周报使用正确半开窗口，周报包含趋势与上周对比；
- 可选/必需 Check、Provider/LLM 故障按规则形成 `PARTIAL/FAILED`；
- 报告发布不可变、并发发布单 Current、更正版本和跨 Domain 读取受控；
- 动作级与方案级 Comparison 可共存，settle delay 不占 Worker；
- 不同 Mapping/单位/窗口不可比较，严重护栏退化不会被主指标改善掩盖；
- Schedule Run 永不进入人工补证，Email/IM 不产生任何外发行为。

## 完成定义

步骤 10 完成时，多副本 Scheduler 可恢复地创建 Fire/Run；日报、周报、故障、性能和多实例 Comparison 报告都以不可变 Artifact 生成并由版本化投影发布；APEX 可安全查询当前报告与调度状态；所有诊断、执行和对比仍由统一 Task 内核驱动。
