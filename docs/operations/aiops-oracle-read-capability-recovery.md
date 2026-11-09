# Oracle 只读能力探测与 SQL 快照恢复

## 根因与范围

2026-11-09 在 kbotdev 使用目标原有诊断凭据验证：两个受影响 Oracle 19c 目标均可读取
V$DATABASE、V$INSTANCE、V$SQLSTATS、V$SESSION、GV$SQL、GV$SQL_PLAN 和结构目录。
连接、认证和只读权限本身正常，不能把空能力声明归因为权限不足。

KBot 的 Oracle 测连返回 oracle-controlled-options.v1，却没有填充发现的能力/权限，
同步或后台测连保存时因而用空列表替换原能力。Ammolite 的 Oracle 测连尚未提供这项能力
探测，同样无法为新目标生成可供周期采集使用的能力声明。两仓均需从实际只读查询产出能力，
不能手工写入 dynamic_performance_views 来绕过验证。

## 修复合同

- 共用语义的 Oracle 适配器使用十项固定系统视图零行查询，仅验证诊断身份的读取能力；
  不扫描业务表、原始 SQL 或计划内容，不执行 DDL、DML、GRANT 或受控动作。
- 只有 V$DATABASE、V$INSTANCE、V$SESSION 全部可读才发现 dynamic_performance_views；
  只有 DBA_TABLES、DBA_INDEXES、DBA_TAB_COLUMNS 全部可读才发现 dba_catalog_views。
  这两项表示基础视图族可用，不保证该族所有视图均可访问；每个成功探测的视图单独记录
  规范权限名称。V$SQLSTATS、GV$SESSION、GV$SQL、GV$SQL_PLAN 分别记录读取结果，
  不从基础连通性推断它们可用。
- 成功执行但零行仍为 AVAILABLE。ORA-00942、ORA-01031 只将对应视图标记为
  UNAVAILABLE；超时、断连和取消不发布“权限已消失”的空能力快照。
- 测连返回 oracle-capabilities.v1、发现的能力、逐项权限及 read_access 回执。
  同步保存和后台测连继续使用现有版本围栏；保留其他配置元数据和 KBot 已有动作候选项。
- 不探测或开启商业许可，不从只读凭据推断写权限，不改变 Target/Agent 开关或业务授权。
  架构验收仅明确登记这个外部目标库适配器，不扩大应用自有 Schema 检查的运行时例外。

## 线上验收步骤

本地测试并提交推送后，授权操作者只快进拉取代码，按原解释器、参数与环境定向重启
受影响 AIOps API 和 Worker。对已确认的两个目标走现有测连应用用例，用真实只读结果
刷新能力并保留幂等审计；不要直接 UPDATE 能力字段，也不要伪造探测回执。

检查实际持久化的探测版本、能力、逐项权限和服务就绪，然后观察自然调度的工具清单、
执行状态、READY 快照及语句行落库；至少两条同实例可比较快照才能证明窗口增量可用。
已有失败快照和调度退避保留，成功后由正常采集流程更新错误状态；不补造历史窗口。

## 本地验证

两仓各自使用本仓源码路径运行：Oracle 只读探测、三引擎测连、同步/后台能力保存、
连通性恢复、配置用例、资源采集授权、Oracle 快照、执行围栏及窗口查询共 216 项和 12 个子测试通过。
两仓架构边界检查通过。
补充报告入口和动作授权回归：45 项及 57 个子测试通过。

额外运行 test_aiops_existing_plan_boundary.py 时，两仓均有 3 个执行计划展示断言失败。
已分别从修改前 HEAD 导出隔离工作区复跑，均为相同的 38 通过、3 失败；
失败涉及展示文案及 PostgreSQL/MySQL 只读估算策略，不是本次 Oracle 探测引入。
本项未更改这些策略或测试，不将其计为通过。

## KBot 线上发布与已确认结果

- 本地修复提交 KBot 3e6d5b55、Ammolite a8452275 均已推送；kbotdev 的 KBot
  仓库只快进拉取，保留已有未跟踪配置备份，没有在线修改源码。
- 仅恢复 KBot AIOps API、Worker 两个受影响进程，Scheduler、DB Executor、
  Main API 及 Ammolite 进程保持不变。四个 AIOps /ready、Main API 18099 /readyz、
  Ammolite 19090 /readyz 复查均为 200。
- 使用现有 request_target_connectivity_check 应用用例及原诊断凭据，在版本条件、
  幂等审计下刷新两个目标；没有直接写入能力列表。两者均为 CONNECTED，
  oracle-capabilities.v1 的十项视图检查全部 AVAILABLE，能力为
  dba_catalog_views、dynamic_performance_views。
- 前后能力与运行态保存在私有目录 /home/ubuntu/kbot4-oracle-capability-1yyese5u；
  两个目标的既有失败次数及下一次调度时间未改变，不删除历史失败快照。
- 额外通过相同诊断身份，在只读事务中执行现有目录的工作负载身份和 SQL 模板，
  分别各返回 1 行和 10 行；不在日志或运维文档保留 SQL 正文、诊断用户名、密码或连接串。
- 本次没有拉取或重启 Ammolite，也没有修改其线上目标配置；Ammolite 同类产品修复
  已同步提交，线上发布状态不能据此宣称已更新。
- 首个目标在 03:16:48、03:17:49 UTC 连续生成 READY 快照；由正常成功采集流程清除
  采集错误、将连续失败计数归零。读取 15 分钟窗口、TOTAL_DURATION、DESC、limit=10，
  基于上述两条可比较快照匹配 33 条 SQL，按耗时倒序返回 10 条。
  修复前的历史窗口缺口仍如实报告，不把约一分钟增量冒称完整 15 分钟。
- 第二个目标按原退避在 03:25:25、03:26:27 UTC 连续生成 READY 快照，未人为重置
  计划时间或失败状态。最终复查两个目标分别已有 10 条、2 条新 READY 快照，
  workload_last_error_code 均为空、连续失败均为 0，持久化探测版本仍为 oracle-capabilities.v1。
- 最终按相同 15 分钟请求、TOTAL_DURATION / DESC / limit=10 读取窗口：
  首个目标匹配 44 条、第二个匹配 19 条，均返回 10 条，排序检查通过。
  实际可比较区间分别为 03:16:48–03:25:56、03:25:25–03:26:27 UTC。
  查询仍如实标记修复前窗口缺口和有限热点候选集，不承诺全库完整榜单。
- 四个 AIOps /ready 最终均为 200；03:14 UTC 之后的在线 ERROR 日志查询为空。
  这些就绪结果之外，以上实际快照、语句行和窗口排名共同构成本轮端到端验收。
