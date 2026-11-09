# 空间变化趋势历史取证与监控证据引用修复验收

日期：2026-11-09。项目：Ammolite / KBot4，部署环境：kbotdev。

## 故障与原因

用户在 2026-11-09 14:07:04 Asia/Shanghai 对 MES (demo) 请求最近 7 天空间变化趋势。实际执行只取得数据库身份、当前容量和累计 IO 统计，未读取容量时序。随后回答保存因“回答引用的证据尚未投影到当前 Turn”失败。

1. 空间趋势图表档案仅配置通用 storage 指标，未包含目录中已有的 PostgreSQL 数据库容量指标。通用指标不支持 PostgreSQL，按数据库类型及绑定授权求交集后没有可查询指标。当前容量快照采集本身正常；不是数据库只采集过一次，也不是 Prometheus 没有历史。
2. 监控结果投影仅登记标准时间序列。成功完成但没有事件行的告警查询仍有可核验查询回执，评估器会引用它；该来源没有登记为 Turn 证据，导致终态回答引用校验失败。原会话两轮 source-events 引用均受影响。

## 修改范围

- 两项目的空间趋势、巡检空间档案补充 postgresql.database.size_bytes，继续按数据库类型、绑定授权和供应商能力求交集；不使用 CPU 等无关指标兜底。
- PostgreSQL 容量图表明确命名为“PostgreSQL 数据库逻辑容量趋势”，不将其解释为主机磁盘余量或表空间剩余容量。
- 监控投影登记标准时序、原生历史、指标目录和事件查询来源，并保留相应窗口与观测语义。成功零行事件查询允许引用，但不证明完整历史无告警；失败或未执行查询不伪造成功证据。
- 保留回答引用必须属于当前 Turn 的校验；重复登记保持证据幂等。
- 未修改 Prompt、数据库 Schema 或源系统采集配置，无需 Prompt 同步或 DDL。

功能提交：Ammolite 35d045a4（dev）；KBot4 955e92bd（kbot4.1.1）。

## 本地验证

隔离各仓库 PYTHONPATH，使用 cube 环境 Python 运行：

- test_aiops_turn_answer.py
- test_aiops_visualization_policy.py
- test_aiops_monitoring_snapshot.py
- test_aiops_monitoring.py
- test_aiops_monitoring_precision.py
- test_aiops_oem_events.py
- test_aiops_dba_playbook_planning.py

结果：Ammolite 262 passed / 24 subtests passed；KBot4 255 passed / 24 subtests passed。

Ammolite check_architecture_boundaries.py、KBot4 check_4_0_boundaries.py 均通过；定向 diff 检查通过。没有运行全量 Schema 验收，也没有全库初始化。

新增回归覆盖：七天查询窗口、PostgreSQL/Oracle/MySQL 指标选择、绑定权限限制、逻辑容量图表命名、成功空事件查询、多轮引用、重复投影、原生历史与空目录引用、失败查询不生成观测证据、未知 artifact 引用仍拒绝。

## 线上部署与验收

源码在本地修改、提交、推送后，线上使用 fast-forward pull。保留原有未跟踪配置备份；未在线编辑仓库源码。

- Ammolite 定向重启 aiops-agent 与 main-api，六个相关进程重新就绪。
- KBot4 使用官方 restart_kbot.sh 重启；SSH 输出连接中断后，重新检查确认 AIOps 四个进程和 Main API 均为新 PID，并非仅依据中断前脚本输出判断成功。
- 两项目 Main API readyz、各 AIOps API / DB Executor / Worker / Scheduler ready：共 10 项 HTTP 200。
- 线上开发日志 API 在部署后检查范围内未返回新的 ERROR / CRITICAL 记录。

实际取证使用线上已配置目标、绑定、标准 MonitoringSnapshotBuilder、EvidenceObserveHandler 和 Prometheus Adapter，未以手写 PromQL 替代产品查询链路。沿用用户请求窗口：

2026-11-02T06:07:04Z ～ 2026-11-09T06:07:04Z。

两项目均正确选中 postgresql.database.size_bytes，无初始指标缺口。aiops_demo 返回 120 个有界重采样点，coverage_ratio=1，无来源查询 gap，生成“PostgreSQL 数据库逻辑容量趋势”图。该查询每序列限额为 120 点；限额采样末点为 2026-11-09T04:43:04Z，不将其描述为请求终点的实时值。排查时直接核对 Prometheus，最近七天容量原始样本有 40,320 个；无需重新等待七天积累数据。

另在 Ammolite 对原故障的真实两轮监控 artifact 做数据库事务内回放：两条 source-events 引用成功登记，回答投影达到 COMPLETED，执行 flush 无引用缺失错误。随后全部回滚，并复查原故障 Turn 状态、确认验收临时 artifact 不存在，历史会话未改写。

此次线上验收覆盖实际取证、图表与原故障回答持久化链路；没有冒用用户身份创建新的聊天或宣称完成浏览器交互验收。旧失败记录保留，用户可重新发起“空间变化趋势，统计天数 7”。
