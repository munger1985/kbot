# SQL 诊断证据与授权边界

## Oracle 授权声明

在 AIOps 的 Target 编辑页，核对合同后按实际授权勾选 Oracle Diagnostics Pack 和 Tuning Pack。
不勾选代表未知或未确认，AWR/ASH 不可用；SQL Monitor 同时要求两项声明。
数据库参数、账号权限和连接成功均不能证明许可证有效，系统不会自动核验合同。
声明保存在 Target 的 capabilities.entitlements 中，分别为
`ORACLE_DIAGNOSTICS_PACK`、`ORACLE_TUNING_PACK`。
声明变更影响后续冻结的诊断任务，新版本冻结的任务按其既有授权快照执行。
升级前已冻结的动态查询可能因策略快照变更而安全拒绝，需要重新发起。
普通 Oracle SQL 健康检查不包含 SQL Monitor；满足两项授权后可单独调用 SQL Monitor
相关工具，不因缺少授权阻断普通游标、计划及对象取证。

## Top SQL 与计划

周期 SQL 快照目前只采集累计排名的有限候选，两个可比较快照只能计算已采集候选
的增量排名，不能证明覆盖全部窗口热点；出现、淘汰、重置、采集缺失或窗口覆盖
不足时必须保留相应缺口。累计值、当前活动快照和已采集候选的窗口增量不能相互替代。

PostgreSQL `pg_stat_statements_info.stats_reset` 是全局重置时间，不是单条 SQL
的确定起点；单条重置和淘汰也可能影响统计。`plan_count=0` 不代表没有执行计划，
需结合 `pg_stat_statements.track_planning` 和计划证据判断。对象索引目录查询仅返回
指定对象的可见元数据，不能证明索引在历史执行中被使用；零行也可能是权限或对象不可见。
为兼容 MySQL 8.0 早期版本，索引查询不返回函数索引表达式。
缺少业务 SLO、基线或实际计划时，不以平均耗时或示例估算计划断言是否异常或实际扫描方式。
系统不会自动执行 EXPLAIN ANALYZE、创建索引或更改目标库配置。

## 调查收敛

`max_investigation_replans` 默认 2，`max_investigation_replan_seconds` 默认 180 秒，
用于决定是否启动新一轮补证规划；它们不是全流程的硬超时，不中断已进行的模型或
数据库调用。原有运行 deadline、模型超时和工具预算仍分别适用。预算耗尽后以已有证据
回答并保留缺口，不把重复 artifact 当成新信息；不同时点活动采样仍可提供新观测。
