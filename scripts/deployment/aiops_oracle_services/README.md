# Oracle 基础服务的可选只读采集

AIOps 为 ASM、RAC、Listener 提供独立资源和33项共用语义。OEM 默认映射依据真实导出；Prometheus/Zabbix 必须配置明确的原生映射。缺失采集不能靠 AIOps 拼接主机或普通数据库数据补齐。

`collector.py` 是独立、可选工件，不改变现有 Exporter/Prometheus/Zabbix/OEM 配置，不自动安装或授权。需要在实际 Oracle/Grid 环境验证后启用。只运行固定 SELECT、tnsping 和 lsnrctl services；不运行变更、crsctl 操作、用户 SQL 或 Shell。普通数据库测试容器不能代替 RAC/ASM 验收环境。

## 配置与运行

配置 JSON 必须由运行用户持有，权限0600。每进程绑定一个服务身份；禁止将数据库密码写入命令行或示例。ASM/RAC 使用已有只读账号及 `username/password/dsn`，不自动获取 SYSDBA/SYSASM；如果现场 ASM 不允许最小只读授权，应继续使用 OEM/已批准 Exporter，不扩大权限。

- 共用：`resource_type`（ORACLE_ASM / ORACLE_RAC / ORACLE_LISTENER）、`target_key`。
- ASM：`instance_name`，精确核对 V$INSTANCE；读取 V$ASM_DISKGROUP_STAT、V$ASM_DISK_STAT、GV$ASM_DISKGROUP_STAT、GV$ASM_DISK_STAT。
- RAC：`database_name`、`services`数组；精确核对 V$DATABASE、cluster_database=TRUE；读取 GV$INSTANCE、GV$ACTIVE_SERVICES、DBA_SERVICES。未配置名称不会作为停止的服务输出。
- Listener：`oracle_home`绝对路径、`tns_alias`、`listener_name`、`services`数组；由本地 Oracle 配置解析。原厂工具输出强制英文，只识别审核过的语法；静态 UNKNOWN 不当作动态 READY。

```bash
python collector.py --config /absolute/private/service.json --listen 9217
python collector.py --config /absolute/private/service.json --format json
```

HTTP 仅监听127.0.0.1，部署方负责明确的网络访问控制；每次 scrape 现采，失败503，不返回旧缓存。不要把一次性输出长期放入 textfile collector 或 Pushgateway 后当作持续现采。JSON 为 Zabbix External Check 的主 Item 格式：按 `samples` 中 code+完整dimensions 建立 dependent Items/LLD，定位固定 Host+精确 Item key；AIOps mapping 使用 fixed_dimensions。此工件不自动导入特定版本的 Zabbix Template。

每个固定 SQL／原厂工具超时10秒；ASM多次身份与指标读取顺序执行，整体最坏延迟可能超过默认 scrape timeout，部署时需设置合理的 scrape_timeout（小于 scrape_interval）并实测。ASM另核对 instance_type=ASM，RAC另核对 cluster_database=TRUE，不能只因账号能登录就当作已验证服务类型。

## 映射和能力边界

Prometheus 指标名由 `NAMES` 固定，绑定labels必须包含实际 `target_key`、`resource_scope` 及抓取时的 job/instance 等全部身份标签；剩余标签逐个映射为目录维度。Zabbix 每个 Item 的单位与原生类型必须与映射一致。状态字典显式配置：ASM磁盘1在线/0离线，RAC实例1 OPEN/2 MOUNTED/3 STARTED，RAC服务1已运行/0未运行，Listener响应1成功/0已审核TNS失败，动态注册1 READY/0未READY。后三者不能合并为整机健康。

已提供采集：ASM容量4项、磁盘模式、6项累计I/O；RAC实例状态与所选服务运行状态；Listener响应、TNS延迟和逐服务动态注册。累计I/O对应原生ASM计数器，不自动换算速率。OEM原厂osm_common.xmlp已核实其聚合读取reads/writes、bytes/1048576、read_time/write_time乘1000，单位保持一致。

Clusterware综合状态、OS进程、TCP独立状态、OEM interval服务性能、ASM安全总容量/安全使用率/不均衡/重平衡/同步和Listener interval连接计数不由该最小采集器伪造。已安装的 OEM/Exporter/Zabbix 插件若提供这些列，使用显式映射；否则需要针对实际版本补充审核采集，页面明确未映射，不能宣称33项均已采集。

OEM确实缺原生列时，可由OEM管理员使用受控 Metric Extension 调用上述本地只读采集，而后在 AIOps 配置其真实 metric group/column、原生类型/单位/完整keys。必须经过测试目标采集、OMS上传和REST/EMCLI实际读取验收；不写OEM内部表，不修改原厂插件，也不在AIOps绕过OEM直接执行采集器。该扩展需要现场安装审批，此仓库不虚构通用的 Metric Extension GUID 或版本兼容包。
