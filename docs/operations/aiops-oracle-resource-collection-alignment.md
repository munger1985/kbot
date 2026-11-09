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
