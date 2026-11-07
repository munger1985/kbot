# KBot AIOps观测栈自动化部署资产

本目录是生产自动化部署包的受控资产目录，不是用户直接执行入口。

- 唯一入口：`scripts/aiops-stack`
- 唯一用户配置：`var/aiops-stack/aiops-stack.ini`
- 生产部署说明：`docs/operations/aiops-observability-production-deployment.md`
- 人工安装与运维：`docs/operations/aiops-observability-manual-deployment.md`

KBot 容器整机交付时，人工只填写 `installation/dev/deployment.ini`；发布器将其中
`[observability.*]` 配置转换为生成目录下权限为 `0600` 的 `aiops-stack.ini`，再通过
`KBOT_AIOPS_STACK_CONFIG_FILE` 调用同一个零参数入口。该环境变量属于整机编排器接口，
不用于维护第二份人工配置。独立部署 Central 或 Collector 时仍使用上述 `var/` 配置。

`compose.yaml`只定义稳定的中心组件和主机采集组件。数据库Exporter、Oracle Alert
Collector、MySQL/PostgreSQL Docker日志发现、逐目标Secret、Volume和KBot Webhook签名桥由入口脚本生成到
`var/aiops-stack/generated/compose.generated.yaml`。不要手工维护生成文件，也不要从
本目录直接执行裸`docker compose up`。

同时启用`[logs]`和`[containers]`时，Alloy只采集显式设置
`aiops_target_key`以及`aiops_log_source=mysql_error|postgresql_log`的容器。Docker
Socket以只读方式挂载；不要给无关容器设置日志采集标签。

Oracle DBA在目标PDB创建专用数据库诊断用户时，使用
`oracle/create_kbot_monitor.sql`。该脚本必须以SYSDBA执行，并会拒绝在`CDB$ROOT`
创建用户。用户已经存在但授权不完整时，使用
`oracle/grant_kbot_monitor.sql`补齐并验证完整授权，不要重复创建用户。
两份脚本都只授予`CREATE SESSION`和`SELECT ANY DICTIONARY`，使AIOps可以读取
`V$`/`GV$`、`DBA_`/`CDB_`以及AWR/ASH等系统诊断视图。AIOps不根据Oracle许可证
裁剪数据库诊断能力；许可证管理由部署和使用该数据库的组织负责。

### Oracle PDB采集与空结果

Exporter显式加载受版本管理的`configuration/oracle/default-metrics.toml`，不依赖
镜像内置默认SQL。两份指标文件的内容Hash均参与容器配置，修改后会重建受影响Exporter。

PDB的`V$RESOURCE_LIMIT`、`V$WAITCLASSMETRIC`可能成功查询但返回零行；
仅这些可选基础视图允许空结果，ASM仍沿用未使用ASM时允许为空的语义。
连接、权限、SQL错误不会被忽略，`OracleExporterScrapeFailed`告警保持启用。
`oracledb_kbot_source_row_count{source="resource|wait_time|cpu_window"}`
显式记录各原生数据源本轮行数；零行不代表健康，也不会补成零使用率。

连接使用率继续使用PDB会话数和有效会话上限。非空闲等待通过
`oracledb_kbot_wait_seconds_total{wait_class="..."}`提供当前容器累计秒数，
使用`rate(...[1m])`计算窗口速率，不与原生等待窗口值混用。
CPU优先使用最近两分钟内的原生PDB窗口；无窗口时，使用当前容器
`DB CPU`累计秒数的一分钟速率除以`V$OSSTAT.NUM_CPUS`，乘100得到
相对主机CPU容量的百分比，不使用可能不同的`CPU_COUNT`作分母。
缺少有效分母或采样不足时不产生虚假数值；连续两分钟仍缺少CPU使用率时，
`OracleCpuMetricUnavailable`单独报告覆盖缺口。

OEM不属于此安装包，部署完成后在AIOps App内配置。Grafana Server及其Dashboard资产
不再随观测栈部署；Prometheus、Zabbix的标准时序统一由AIOps原生实时监控页面展示。
Grafana Alloy仍作为日志采集器保留，与Grafana Server没有运行依赖。
