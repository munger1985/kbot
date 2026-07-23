# 4.0 AIOps 监控、巡检与报告

## 能力边界

AIOps Agent 支持 Oracle 和 MySQL，通过可扩展 Adapter 保留其他数据库、监控工具和通知渠道。第一版接入 Prometheus、Zabbix 和 OEM，只执行受控数据库 SQL；Shell、OEM Job 和 Zabbix Remote Command 只能作为人工建议。

数据库连接是可选的：未连接数据库时，Agent 仅基于监控数据和 SOP 诊断；配置只读凭据后可执行深度诊断；只有另外配置变更凭据且 `execution_mode=AGENT_EXECUTE` 时才可申请执行。

未配置只读连接或连接不可用时，只有 Chat Run 可在监控信息不足时进入交互式诊断循环：生成只读 SQL，请用户执行并回贴结果，再恢复同一 Ops Run。Alert、Schedule 和其他自动 Run 不等待用户输入，而是生成 `PARTIAL/INCONCLUSIVE` 报告并正常结束。详见 [21_aiops_interactive_diagnosis.md](21_aiops_interactive_diagnosis.md)。

## 监控源与 Target 绑定

`KBOT_OPS_MONITOR_SOURCE` 保存 domain 内的监控系统：`SOURCE_TYPE`、`ENDPOINT`、`SECRET_REF`、`CAPABILITIES_JSON`、`STATUS` 和审计字段。凭据仅保存引用。

`KBOT_OPS_TARGET_MONITOR` 实现 Target 与 Monitor Source 多对多绑定：

```text
TARGET_MONITOR_ID, TARGET_ID, MONITOR_SOURCE_ID
EXTERNAL_TARGET_KEY, ROLE: PRIMARY | SUPPLEMENTARY
PRIORITY, METRIC_SCOPE_JSON, MAPPING_OVERRIDES_JSON
STATUS, LAST_HEALTH_CHECK_AT, ROW_VERSION
```

同一 Target 可同时使用多个监控源。指标注册表定义稳定 `metric_code`、单位、数据类型、聚合方式和 Oracle/MySQL 支持范围；Adapter 把 PromQL、Zabbix Item 或 OEM Metric 映射为统一 `MetricObservation`。Agent 不复制整个时序库，只保存来源、查询窗口、映射版本、摘要、关键样本和完整结果 URI/Hash。

Provider Port、Metric Catalog、标准 Observation、Webhook 信任链、Target 精确映射和只观测 Run 的完整设计见 [34_aiops_step5_monitoring_observe_loop.md](34_aiops_step5_monitoring_observe_loop.md)。

## 统一触发模型

```text
CHAT | ALERT | SCHEDULE | API | IM | EMAIL
                         ↓
                  Ops Trigger Intake
                         ↓
                 Event / Alert / Ops Run
```

4.0 实现 Chat、Prometheus AlertManager/Zabbix Webhook、Scheduler 和内部 API。IM/Email 只保留 `OpsTriggerPort` 和 `ReportDeliveryPort`，不实现 Adapter。Webhook 完成验签、Target 映射、幂等落库后立即返回，诊断由 Worker 异步执行。`critical` 等级是默认自动诊断阈值，可按 Target 覆盖。收到 resolved/recovery 事件后，关联当前 Alert，触发验证并在满足条件时关闭。

## 定时巡检

`KBOT_OPS_INSPECTION_PLAN` 定义 `DAILY/WEEKLY/CRON`、时区、Cron、巡检模板及版本、超时、错过调度和重叠策略。`KBOT_OPS_INSPECTION_TARGET` 保存 Plan 与 Target 的绑定；`KBOT_OPS_INSPECTION_FIRE` 固化每个计划时点及其展开结果，Schedule Run 通过 Fire 归组。

每个 Fire 为每个有效 Target 创建独立 Ops Run；Plan 的 `SKIP/QUEUE` 策略控制重叠，队列最多保留一个 Fire。日报展示当日健康、异常和建议；周报必须聚合趋势、反复告警、未解决问题和上周对比，不只是七份日报的拼接。多副本领取、时区/DST、Misfire 和 Fire 状态详见 [39_aiops_step10_inspection_reporting_and_comparison.md](39_aiops_step10_inspection_reporting_and_comparison.md)。

## 报告模型

`KBOT_OPS_REPORT` 是面向 APEX/前端列表和渲染的稳定投影：

```text
REPORT_ID, OPS_RUN_ID, TARGET_ID
REPORT_KEY, REPORT_VERSION, SUPERSEDES_REPORT_ID, IS_CURRENT
REPORT_TYPE: INCIDENT | PERFORMANCE | INSPECTION_DAILY |
             INSPECTION_WEEKLY | COMPARISON
TITLE, STATUS: GENERATING | READY | PARTIAL | FAILED
PERIOD_START, PERIOD_END, BASELINE_START, BASELINE_END
AFTER_START, AFTER_END, RESULT: IMPROVED | UNCHANGED |
                              DEGRADED | INCONCLUSIVE
CONTENT_ARTIFACT_ID, CONTENT_HASH, SUMMARY, CREATED_AT
```

真实内容是不可变 `OpsArtifact`，Report 表只保存可查询元数据和当前内容引用。`REPORT_KEY` 允许同一 Run 存在多个动作级 Comparison，`REPORT_VERSION/IS_CURRENT` 保留更正历史并为 APEX 提供唯一当前投影。内容使用版本化 JSON schema，同时可保存 Markdown 渲染工件。

## 处理前后对比

所有已执行变更和用户标记为已人工处理的问题都创建 Comparison Task。Verification 验证命令的直接效果，Comparison 判断整体健康与副作用，两者不能合并：

1. 冻结处理前基线窗口、指标集、数据源和映射版本；
2. 按动作配置的 settle delay 等待系统稳定；
3. 用相同时长、单位、聚合方式采集处理后数据；
4. 计算前后值、绝对差、变化率、告警状态和副作用；
5. 生成 `COMPARISON_REPORT`，不足以证明改善时返回 `INCONCLUSIVE`。

人工模式允许用户回填 `EXECUTED/FAILED/CANCELLED`、执行时间、备注和可选结果。若配置只读诊断或监控连接，Agent 自动采集后续指标；不存在预先冻结的基线或数据不可比时必须返回 `INCONCLUSIVE`，不能只凭用户回填证明改善。确定性判级和护栏指标见步骤 10 详细设计。

## 执行与扩展接口

- `MonitorProvider`：目标发现、即时/时间范围指标和告警标准化；
- `DatabaseDialect`：Oracle/MySQL 的诊断模板、变更模板、参数 schema 和能力声明；
- `OpsTriggerPort`：Chat、Webhook、Scheduler，以及未来 IM/Email 入站；
- `ReportDeliveryPort`：未来 IM/Email 投递，4.0 仅实现系统内存储和查询；
- `CommandExecutor`：只接受已批准的类型化 SQL 命令。

新增数据库或监控工具时增加 Adapter 和能力映射，不修改 AIOps Planner、Run 状态机或 Report schema。
