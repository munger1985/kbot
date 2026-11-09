# Oracle 资源采集 Run 的定向对齐

资源采集不依赖 Agent。规范 DDL、ORM 已允许 `KBOT_OPS_RUN.AGENT_ID`、
`AGENT_VERSION_ID` 为 NULL，但已部署数据库需要显式对齐；只更新代码会使
Scheduler 创建 `RESOURCE_COLLECTION` Run 时继续触发 `ORA-01400`。

## 固定范围与执行

工具 `scripts/deployment/align_aiops_resource_collection.py` 使用 AIOps 的现有配置，
默认只读；只对实际 NOT NULL 的上述两列执行一条 `ALTER TABLE ... MODIFY (... NULL)`。
不改业务行、主外键、索引或其他列，不改 Schema 版本，不执行全库初始化，
不创建虚拟 Agent，不在启动、readiness 或请求中增加 Schema 校验。

仅由已获授权的操作者在目标仓库执行。先停止本仓库的 AIOps API、Worker、Scheduler、
DB Executor，保留其他服务；确认目标身份、已有 Run 库存和当前备份可用。

```bash
python scripts/deployment/align_aiops_resource_collection.py
mkdir -m 700 /明确的私有审计目录
python scripts/deployment/align_aiops_resource_collection.py \
  --apply --expected-plan-hash <上一步的plan_hash> \
  --report-dir /明确的私有审计目录
```

指纹包含数据库/容器/Schema 身份、列/约束/索引及历史 Run 主键库存；目标或库存变化后，
旧指纹不能执行，必须重新只读核对。目录需归当前用户所有、无公共权限且不能是符号链接；
审计文件为 0600，不含凭据、连接字符串或诊断正文。已可空的列不重复修改。

Oracle DDL 隐式提交，不能依赖事务回滚。工具保存 before、plan、after、完成或部分完成记录，
并验证两列可空、历史 Run 主键保留、其他列/约束/索引不变。失败后先查私有审计记录；
不要盲目恢复 NOT NULL：恢复采集后新 Run 合法包含 NULL，重新收紧会失败或阻断采集。

## 业务验收

恢复四个 AIOps 进程，逐一检查进程、端口及 `/ready`，再观察至少两轮自然调度。
必须确认新资源采集 Run 创建并进入执行、不再触发此 `ORA-01400`，再分别核验实际快照及
活动采样。就绪成功不等于采集成功；连接、能力声明或目标授权问题须单独报告，不自动改写。
历史窗口缺口不补造，已有失败记录不重写。Ammolite 的 PostgreSQL 同类两列已由规范
初始化器覆盖，本工具不连接或修改 PostgreSQL。

## 2026-11-09 已授权上线记录

- 只读确认线上两列均为 RAW(16)、NULLABLE=N，已有 11,245 条 Run，资源采集 Run 为 0。
  当前在线 Scheduler 日志每约 30 秒出现 IntegrityError，事件详情确认 `ORA-01400` 指向
  `KBOT_OPS_RUN.AGENT_ID`。
- 本地完成部署工具后提交 `97ba3ce6` 并推送，kbotdev 仓库只快进拉取；保留未跟踪
  `configuration/kbot.toml.bak`，没有在线编辑源码、初始化数据库或修改 Prompt。
- 03:00 UTC 仅停止本仓库四个 AIOps 进程，用只读计划指纹核对相同目标后执行一条 DDL。
  结果两列均为 NULLABLE=Y；前后历史 Run 数、主键指纹、其他列、约束和索引完全一致。
  私有审计目录为 `/home/ubuntu/kbot4-resource-collection-9pq49qt5`。
- 使用各进程原有解释器、参数和环境恢复同一四个进程；18110–18113 的 `/ready` 均为 200。
  Main API 18099 `/readyz` 复查为 200，Ammolite 19090 `/readyz` 为 200，未重启它们。
- 03:00:55 UTC 自然调度创建两条无 Agent 的资源采集 Run，均进入执行并最终 COMPLETED。
  03:03:27 UTC 下一轮自然调度再次创建资源采集 Run 并进入 RUNNING，证明后续任务也能创建。
  失败快照按既有退避策略重试，没有人为重置调度时间或失败次数。
  启动后在线 ERROR 日志没有再次出现非空约束错误。
- 独立遗留项：上述两个 Oracle Target 的能力名称集合为空，目录要求
  `dynamic_performance_views` 的 `db.oracle.workload.identity`、
  `db.oracle.workload.statements` 被禁用，03:01 UTC 快照仍是 FAILED /
  `ORACLE_SNAPSHOT_IDENTITY_UNAVAILABLE`。Run 完成不能当作快照成功；未自动补写能力、
  提权或清除失败历史。实际数据库授权尚未据此认定为不足。
- KBot 定向部署工具及资源采集 Schema、授权、执行围栏、独立生命周期共 153 项回归通过；
  Ammolite 同类语义回归 130 项通过。两仓架构边界及 KBot Oracle 建库工件检查通过。
  没有修改 Ammolite 的 PostgreSQL 数据库。
