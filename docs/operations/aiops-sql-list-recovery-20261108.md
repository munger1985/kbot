# Top SQL 无窗口结果与当前活跃 SQL 排查

## 2026-11-08 线上证据

取证入口为用户指定的 Ammolite 在线日志服务及目标库只读查询。没有修改服务器源码、持久配置或业务数据，没有执行 DDL、拉取代码或重启服务。

- 14:21:45（Asia/Shanghai）的 Top SQL 请求针对 MySQL，范围 15 分钟、数量 10、TOTAL_DURATION 倒序。窗口查询没有可比较快照；累计参考查询另报 MySQL 1267。
- 该 MySQL Target 已开启一分钟周期采集，但没有最近采集记录，计划时间仍停在 2026-11-06 01:43 UTC。Scheduler 在创建 RESOURCE_COLLECTION Run 时违反 aiops_run.agent_id 的 NOT NULL 约束，事务回滚，因而没有把错误写到 Target 的采集失败字段。
- 手工只读确认线上 aiops_run.agent_id、agent_version_id 均为 NOT NULL。规范实体及显式部署初始化器已经允许 NULL，线上尚未配套完成这项 Schema 发布。不能重新引入 Agent 绑定或伪造 Agent ID 绕过约束。
- Performance Schema 的 digest/schema_name 为 utf8mb4_0900_ai_ci，执行器连接为 utf8mb4_general_ci。原模板把拼接身份键与 JSON_TABLE 的 VARCHAR 列比较，复现 Illegal mix of collations。
- 14:21:10 的当前活跃 SQL 请求成功、无告警、未截断、返回 0 行；当前复查同样为 0 行。额外使用同一诊断身份执行一条两秒只读测试查询，原活跃 SQL 模板能在执行期间找到它。不能把空闲连接或历史累计 SQL 填充为当前活跃 SQL。

## 两仓修复

- MySQL 跟踪/排除键使用 VARBINARY，库内复合身份 CAST AS BINARY；过滤与优先排序采用相同的精确比较，避免排序规则冲突，也不合并大小写不同的库名。不修改目标库的字符集、排序规则或全局配置。
- Top SQL 返回采集状态的检查时间、计划延迟及宽限。宽限为两个配置采集周期加本轮执行预算；超过宽限且没有已知执行错误时，明确提示调度逾期、检查 Scheduler 和任务创建日志，不再仅提示继续等待。
- 延迟推断不伪造 last_error_code、失败计数或采集成功记录；已有具体执行错误优先显示，关闭采集不展示历史错误。正常排队、尚无计划的状态不据此判定逾期，历史有效窗口数据不被删除。
- 当前活跃 SQL 的成功空结果简洁说明采集时点未发现正在执行的 SQL；失败、缺证据仍保留为不可用，不显示为空闲。

## 验证

线上只读复现原 MySQL 错误；修复表达式返回 10 行。在 utf8mb4_general_ci 和 utf8mb4_0900_ai_ci 两种连接排序规则下，跟踪优先/排除逻辑及中文身份键精确匹配均通过。当前活跃 SQL 的空闲及短暂执行对照均符合预期。

回归覆盖二进制身份 SQL AST、目录 Hash、参数绑定、调度宽限边界、两天未启动、禁用策略、缺失计划、具体错误优先、空结果/失败区分，以及三引擎采集授权、窗口增量、执行围栏、资源 Run Schema 和独立 Run 生命周期。测试使用各仓自己的源码路径，避免共用 Python 环境的 editable install 串仓。Ammolite 238 项、KBot 240 项通过，两仓架构边界检查通过。

## 线上发布步骤

1. Ammolite 需要在明确的目标 Schema 内，将 aiops_run 的 agent_id、agent_version_id 两列 DROP NOT NULL。现有 scripts/deployment/init_postgresql.py 的 _align_aiops_resource_collection 已定义精确变更。不要为了这两列运行可能清理其他对象的全库初始化。
2. 经授权后拉取对应提交、重启受影响 AIOps 进程并核验实际目录 Hash 及新 Run 生命周期；不处理无关服务或此前失败的历史数据。
3. 确认周期快照真实落库且连续，随后验证 15 分钟窗口的真实覆盖。不能补造历史快照，也不能以累计榜单代替窗口增量。
4. KBot 同步代码语义；本文的 PostgreSQL 约束取证仅针对 Ammolite，不表示已对 KBot 的 Oracle 运行库执行验收。

仅提交推送代码不等于线上采集已经恢复。

## 授权后上线验收（2026-11-08）

用户明确授权上述定向 Schema 变更、拉取和重启后，在 kbotdev 的 /home/ubuntu/Ammolite 执行：

- 工作区已跟踪文件干净，原有三个未跟踪配置备份保留。只快进到代码提交 25b8448b，没有修改服务器源码。
- 仅停止并重新启动 Ammolite 的 AIOps API、DB Executor、Worker、Scheduler。KBot 的同名进程及其他服务未重启；使用现有进程拓扑和启动脚本，内部包预检通过，没有运行全库初始化。
- 在单一事务中调用规范部署函数 _align_aiops_resource_collection，只解除 agent_id、agent_version_id 两列 NOT NULL。执行前两列 is_nullable=NO，提交后均为 YES；未改写历史业务数据。
- 端口 19130–19133 的 /ready 均返回 HTTP 200，实际进程、数据库连接及 Worker 后台组件正常。此前尝试 /readyz 返回 404，本服务的正确就绪路径是 /ready。
- MySQL、PostgreSQL 在 06:34 和 06:35 UTC 连续两轮产生 READY 快照，周期 Run 均已 COMPLETED，采集错误为空且连续失败为 0。MySQL 每轮语句采集返回 100 行，执行产物中的模板 Hash 与本次修复一致。
- 按用户参数调用同一 WorkloadService：最近 15 分钟、TOTAL_DURATION、DESC、limit=10。读取两条快照，实际可比较区间为 06:34:02.558461–06:35:02.646850 UTC，匹配 21 条并按总耗时倒序返回 10 条。恢复前缺失的窗口仍如实标记 WINDOW_NOT_FULLY_COVERED，不以这一分钟证明完整 15 分钟已覆盖；热点候选集与新增/淘汰限制仍保留。
- 在线日志显示启动后的开发环境默认 Mutation Grant 密钥警告；未再出现此前的采集 Run 非空约束错误或 MySQL 1267。就绪状态不能替代上述实际落库与窗口计算验收。

独立遗留项：两个 Oracle Target 的能力列表为空，db.oracle.workload.identity、db.oracle.workload.statements 因缺少 dynamic_performance_views 被过滤，快照仍报 ORACLE_SNAPSHOT_IDENTITY_UNAVAILABLE。没有擅自补写能力声明、扩大数据库权限或重置失败历史；本次不声称 Oracle SQL 快照已恢复。
