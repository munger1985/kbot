# KBot AIOps观测栈自动化部署资产

本目录是生产自动化部署包的受控资产目录，不是用户直接执行入口。

- 唯一入口：`scripts/aiops-stack`
- 唯一用户配置：`var/aiops-stack/aiops-stack.ini`
- 生产部署说明：`docs/operations/aiops-observability-production-deployment.md`
- 人工安装与运维：`docs/operations/aiops-observability-manual-deployment.md`

`compose.yaml`只定义稳定的中心组件和主机采集组件。数据库Exporter、Oracle Alert
Collector、逐目标Secret、Volume和KBot Webhook签名桥由入口脚本生成到
`var/aiops-stack/generated/compose.generated.yaml`。不要手工维护生成文件，也不要从
本目录直接执行裸`docker compose up`。

Oracle DBA在目标PDB创建专用数据库诊断用户时，使用
`oracle/create_kbot_monitor.sql`。该脚本必须以SYSDBA执行，并会拒绝在`CDB$ROOT`
创建用户。用户已经存在但授权不完整时，使用
`oracle/grant_kbot_monitor.sql`补齐并验证完整授权，不要重复创建用户。
两份脚本都只授予`CREATE SESSION`和`SELECT ANY DICTIONARY`，使AIOps可以读取
`V$`/`GV$`、`DBA_`/`CDB_`以及AWR/ASH等系统诊断视图。AIOps不根据Oracle许可证
裁剪数据库诊断能力；许可证管理由部署和使用该数据库的组织负责。

OEM不属于此安装包，部署完成后在AIOps App内配置。

启用`[dashboard]`且`local_access = true`时，Grafana默认只发布到
`127.0.0.1:3000`。需要从管理网直接访问时，通过唯一INI中的
`grafana_bind_address`填写本机管理网IPv4地址；不要手工修改Compose文件，也不建议
在客户生产环境使用`0.0.0.0`。

Grafana启用后会自动配置固定UID的Prometheus、Loki和Alertmanager数据源，并从
`configuration/grafana/dashboards`加载KBot AIOps只读看板。看板以`target_key`作为
数据库或主机切换变量；不要在Grafana UI中复制一套客户私有版本，通用修改应回写仓库
中的JSON并通过部署脚本发布。
