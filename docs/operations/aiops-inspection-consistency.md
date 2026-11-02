# 日常巡检一致性与上线验收

Ammolite 与 KBot4 使用相同的巡检产品语义。平台数据库差异不应改变模板取证、覆盖率、报告状态与失败重试行为。

## 固定取证与备份策略

日巡检启用备份策略时，按数据库工具合同解析检查目录中可执行的检查项：

| 数据库 | 工具合同 | 当前检查项 |
| --- | --- | --- |
| Oracle | `db.backup.recent_jobs` | `oracle.backup.recent_jobs` |
| PostgreSQL | `db.postgresql.archiver.status` | `postgresql.wal.archive` |
| MySQL | `db.mysql.binlog.status` | `mysql.binlog.status` |

代码不再引用已删除的 `postgresql.backup.archive_chain` 与 `mysql.backup.binlog_chain`。旧代码会在追加策略取证时抛出未知检查项的 ValueError；即使冻结模板已有同一工具也会先编译错误 ID，从而使整批目标的巡检启动事务失败。

冻结步骤已有工具时不重新读取目录、不重复追加或重新编译。缺少工具时才从同数据库的 READY 目录项解析并编译。Oracle 按策略最大成功间隔扩大查询窗口，新增策略窗口限制为 7–90 天，不缩短冻结模板已有的更长窗口，也不原地改写模板定义。

冻结模板与备份策略准备错误分别产生 `AIOPS_INSPECTION_TEMPLATE_INVALID`、`AIOPS_INSPECTION_BACKUP_POLICY_INVALID`。调度日志记录安全的阶段与原因类型，不记录原始异常中的潜在敏感内容；此类确定性配置错误不重复投递，真正的瞬时故障继续按既有规则重试。

PostgreSQL WAL 归档、MySQL Binlog 状态只证明归档链可观测，不能替代外部备份平台的成功记录。缺少备份成功证据时仍应报告真实缺口。

## 报告统计与状态

- 检查项证据覆盖率按去重的 `check_ids` 计算；未带检查 ID 的冻结步骤按工具标识计数。
- 工具执行覆盖率按取证步骤及其证据引用计算。一份观测可支持多个语义检查项。
- 有引用但缺字段、观测不完整或存在对应步骤缺口时，工具可算已执行，语义证据不算完整。
- 监控采样覆盖率与上述两种覆盖率独立；监控缺口不能使成功执行的数据库工具被算作未执行。
- 纯调查评估提示不冒充模板数据缺口；真实数据源、字段及报告合同缺口继续保留。
- 完整证据与数据库健康是两个维度：报告 READY 不表示所有检查正常，发现需关注项也可形成完整报告。
- 摘要中的取证步骤数不冒充勾选的语义检查项数。报告发布后的历史工件保持冻结，查看或下载不会按新代码重算。

本修复不变更 Schema、Prompt 或 API 字段。界面的“版本 1 / 模板 v1”不是部署代码版本，也不能证明不同日期使用相同模板快照、策略及监控窗口。

## 离线回归

在各自仓库根目录，使用对应环境运行，避免直接执行脚本导致仓库测试模块导入路径缺失：

```bash
python -m unittest \
  tests.unit.aiops_agent.test_aiops_inspection_policy \
  tests.unit.aiops_agent.test_aiops_inspection_scheduler \
  tests.unit.aiops_agent.test_aiops_check_catalog \
  tests.unit.aiops_agent.test_aiops_reporting \
  tests.unit.aiops_agent.test_aiops_formal_reporting \
  tests.unit.aiops_agent.test_aiops_readiness \
  tests.unit.aiops_agent.test_aiops_dba_playbook_planning \
  tests.unit.aiops_agent.test_aiops_turn_answer
```

Ammolite 使用 `conda run -n cube`，并运行 `tests/acceptance/check_architecture_boundaries.py`；KBot4 使用 `conda run -n kbot4`，并运行 `tests/acceptance/check_4_0_boundaries.py`。单元测试使用模拟依赖，不触发真实巡检。

## 部署后验收

部署、服务重启和真实巡检需单独获得操作授权。代码回归通过不能当作线上恢复证明。

1. 核对两边实际部署提交、冻结模板内容/Hash、三个目标、备份策略、时区、巡检窗口与监控数据来源。
2. 经授权触发一轮新巡检，核对三个目标分别生成会话/运行，固定取证实际执行，随后发布报告；按计划及运行关联线上日志，不仅检查 healthz 或 HTTP 200。
3. 比较新报告的语义证据覆盖率、工具执行覆盖率、监控采样及真实缺口；不能要求不同时间窗口的数值或告警条数机械相等。
4. 历史失败的 Outbox 事件不自动清理或重入队；如需补跑，先核对事件状态、巡检周期及幂等边界，获得授权后处理。

旧报告具体每条缺口与检查数量变化仍需原始冻结工件逐项核对。在线日志中的未分类 ValueError 与上述缺陷症状相符，但缺少失败事件的实际策略/模板载荷时，不将离线复现宣称为该线上事件的完整因果证明。
