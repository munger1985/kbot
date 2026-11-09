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
连通性恢复、配置用例、资源采集授权、Oracle 快照及执行围栏共 192 项和 12 个子测试通过。
两仓架构边界检查通过。

额外运行 test_aiops_existing_plan_boundary.py 时，两仓均有 3 个执行计划展示断言失败。
已分别从修改前 HEAD 导出隔离工作区复跑，均为相同的 38 通过、3 失败；
失败涉及展示文案及 PostgreSQL/MySQL 只读估算策略，不是本次 Oracle 探测引入。
本项未更改这些策略或测试，不将其计为通过。
