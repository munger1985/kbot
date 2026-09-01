# AIOps观测栈生产自动化部署

## 1. 交付定位

本交付物是客户环境的标准安装入口。运维人员只维护
`var/aiops-stack/aiops-stack.ini`，并始终使用零参数命令：

```bash
scripts/aiops-stack
```

首次执行只生成配置模板，不启动服务。填写必选项、取消所需模块的
`enabled = true` 注释后再次执行，脚本会校验配置、生成逐服务Secret、动态Compose、
Prometheus目标文件、规则和部署清单，再应用选中的服务。配置文件必须为`0600`，不得
提交到Git。OEM不属于本安装包，部署后在AIOps App内配置。

人工逐项安装、已有监控系统接入和故障恢复请参考
[人工安装与运维手册](aiops-observability-manual-deployment.md)。

## 2. 生产拓扑选择

开发、演示或明确接受单点风险的小规模环境使用`all-in-one`。正式客户环境推荐分别
部署：

```text
Collector节点                         Central节点
Oracle/MySQL/PostgreSQL               Prometheus / Alertmanager
        │                             Loki / 可选Grafana
独立Exporter ─── 内部受控网络 ───────▶ Prometheus
Oracle Alert Collector ── Alloy ─────▶ Loki TLS入口
                                      Loki Ruler ────────▶ Alertmanager
                                      Alertmanager ── 签名桥 ──▶ KBot
```

- `central`：运行Prometheus、Alertmanager、Loki和可选Grafana；
- `collector`：运行Exporter、Oracle Alert Collector、Alloy和Node Exporter；
- 每个节点使用自己的INI和`deployment_id`，各执行一次同一个脚本；
- Collector到Central只能经过客户批准的受控网络；远程Loki必须使用TLS和Token；
- Exporter监听地址必须显式配置，并通过防火墙只允许Central访问。

单机文件存储不是高可用方案。关键生产环境仍需在实施设计中确认Prometheus冗余、
Alertmanager集群、Loki对象存储与副本、容量、备份、证书入口和恢复演练。

## 3. 多数据库监控模型

一个Prometheus可以监控多个数据库，不需要为每个数据库部署一套Prometheus。正确的
隔离单元是“数据库目标”：

- 每个数据库一个Exporter实例；
- Oracle每个数据库再有一个Alert Collector实例和独立Checkpoint Volume；
- 每个实例使用独立Secret和最小权限数据库账号；
- 每个目标拥有稳定且唯一的`target_key`；
- Prometheus使用`target_key`标签区分时序数据，AIOps Source Binding使用同一个值；
- 不同数据库不能共享Exporter进程、DSN、Checkpoint或日志文件。

例如两个Oracle和一个PostgreSQL会生成三个Exporter，但中心端仍只有一套Prometheus。

## 4. all-in-one多数据库示例

首次生成模板后，配置可写为：

```ini
[deployment]
deployment_id = customer-a-observability
role = all-in-one
local_access = true

[metrics]
enabled = true
prometheus_retention = 30d
alertmanager_retention = 120h

[logs]
enabled = true
loki_retention = 720h

[dashboard]
enabled = true
# 仅允许本机访问时使用127.0.0.1；管理网直连时填写部署节点的管理网IPv4地址。
grafana_bind_address = 10.10.1.10
grafana_port = 3000
grafana_admin_user = kbot-admin
grafana_admin_password = 填写实际强密码

# 启用后自动加载Oracle总览、存储、异常、Alert Log和主机资源看板，
# 无需在Grafana中手工创建Dashboard。

[oracle:oracle-prod-01]
enabled = true
host = 10.10.1.21
port = 1521
service = PROD1
username = kbot_monitor
password = 填写实际密码

[oracle:oracle-prod-02]
enabled = true
host = 10.10.1.22
port = 1521
service = PROD2
username = kbot_monitor
password = 填写实际密码

[postgres:postgres-prod-01]
enabled = true
uri = 10.10.1.31:5432/postgres?sslmode=require
username = kbot_monitor
password = 填写实际密码
```

脚本生成类似以下Prometheus目标：

```json
[
  {
    "targets": ["oracle-oracle-prod-01-exporter:9161"],
    "labels": {
      "job": "oracle",
      "instance": "oracle-prod-01",
      "target_key": "oracle-prod-01"
    }
  },
  {
    "targets": ["oracle-oracle-prod-02-exporter:9161"],
    "labels": {
      "job": "oracle",
      "instance": "oracle-prod-02",
      "target_key": "oracle-prod-02"
    }
  }
]
```

配置段可以继续复制。冒号后的Target Key只允许小写字母、数字、连字符和下划线。

Oracle补充指标读取`V$SYSMETRIC`中的最近一分钟主机CPU使用率，以及`V$SYSSTAT`中的
SQL解析失败累计数。必须使用独立的`kbot_monitor`诊断账号，不得使用SYS、SYSTEM或
应用账号。该账号同时供AIOps动态只读诊断使用。

首次创建监控账号时，以SYSDBA连接目标PDB后执行：

```sql
ALTER SESSION SET CONTAINER = PDB01;
SHOW CON_NAME;
@scripts/deployment/aiops_observability/oracle/create_kbot_monitor.sql
```

脚本会拒绝在`CDB$ROOT`运行，交互隐藏输入密码，并授予及验证以下两个系统权限：

```sql
GRANT CREATE SESSION TO kbot_monitor;
GRANT SELECT ANY DICTIONARY TO kbot_monitor;
```

`SELECT ANY DICTIONARY`用于覆盖实例/PDB身份、RAC会话与锁、SQL文本与执行计划、等待与
I/O、SGA/PGA、存储、恢复、Data Guard、维护任务以及AWR/ASH等数据库诊断信息。
AIOps不根据Oracle许可证对AWR/ASH查询进行能力门控；许可证管理不属于AIOps运行时职责。
数据库查询仍由AIOps只读SQL策略、执行超时和结果上限约束。用户已存在时不要重复执行
初始化脚本，改为执行完整授权脚本：

```sql
ALTER SESSION SET CONTAINER = PDB01;
SHOW CON_NAME;
@scripts/deployment/aiops_observability/oracle/grant_kbot_monitor.sql
```

## 5. Central与Collector配置示例

数据库侧Collector节点：

```ini
[deployment]
deployment_id = customer-a-db-zone-01
role = collector
local_access = false

[oracle:oracle-prod-01]
enabled = true
host = 10.10.1.21
port = 1521
service = PROD1
username = kbot_monitor
password = 填写实际密码
exporter_bind_address = 10.20.1.11
exporter_port = 19161
```

中心节点：

```ini
[deployment]
deployment_id = customer-a-central
role = central
local_access = true

[metrics]
enabled = true
prometheus_retention = 30d

[prometheus_target:oracle-prod-01]
enabled = true
engine = oracle
address = 10.20.1.11:19161
environment = production

[prometheus_target:oracle-prod-02]
enabled = true
engine = oracle
address = 10.20.1.12:19161
environment = production
```

Node Exporter也通过相同配置登记，只需使用`engine = node`和对应的`host:9100`地址。

Collector的`exporter_bind_address:exporter_port`必须与Central的`address`一致。新增目标
时先部署Collector，再更新Central目标段并重新执行脚本。Prometheus通过`file_sd`每
30秒刷新目标文件，不需要复制Prometheus实例。

## 6. Alertmanager到KBot

Webhook必须先在AIOps App中建立接收身份，再写入唯一部署配置。完整步骤如下。

1. 进入“诊断源”，新增`ALERTMANAGER`类型诊断源；只接收Webhook时访问地址可以留空。
2. 系统固定从告警的`target_key`标签读取目标标识，页面无需配置标签名称。该标签值与
   本文件中的数据库Target Key一致。
3. 点击“创建并生成接入凭据”。页面会一次完成诊断源创建、Webhook Secret生成和
   Webhook Key生成，并在同一弹窗集中显示只显示一次的Secret、Key和INI配置片段。
   复制配置后再关闭弹窗，不需要保存后重新进入编辑页面。
4. 启用该诊断源。在Agent编辑页选择它并与数据库Target绑定；Locator填写对应的
   Target Key，例如`oracle-prod-01`。
5. 将刚才得到的Key和Secret写入`var/aiops-stack/aiops-stack.ini`的`[metrics]`：

页面生成Webhook Key依赖KBot标准部署必选项`KBOT_MASTER_KEY`。KBot会按用途自动派生
签名密钥，不需要额外执行`openssl`，也不需要单独配置
`KBOT_AIOPS_WEBHOOK_KEY_SECRET`。

```ini
[metrics]
enabled = true
kbot_webhook_url = https://kbot.customer.example
kbot_webhook_key = 填写页面生成的Webhook Key
kbot_webhook_secret = 填写页面生成并已保存到诊断源的Webhook Secret
```

`kbot_webhook_url`只填写Main API根地址，不得追加
`/api/v1/integrations/aiops/signals/...`。同机`all-in-one`测试环境可以填写KBot主机内网
地址，例如`http://10.0.0.190:18099`；`central`生产角色必须使用客户HTTPS入口。

6. 不带参数重新执行`./scripts/aiops-stack`。脚本会保留现有数据卷，增加或重建
   `kbot-webhook-signer`，并把Alertmanager Receiver从`discard`切换为`kbot`。
7. 验证签名桥和Alertmanager均健康：

```bash
docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'alertmanager|webhook-signer'
sed -n '1,80p' var/aiops-stack/generated/alertmanager/alertmanager.yml
```

生成配置必须包含`receiver: kbot`和
`url: "http://kbot-webhook-signer:8080/alertmanager"`。随后发送一条带正确
`target_key`的受控测试告警，并在“告警诊断”确认事件已经关联到对应Target。

安装包会启动内部`kbot-webhook-signer`。Alertmanager把原始JSON发送给它；签名桥按
`timestamp + "." + raw_body`计算HMAC-SHA256，添加`X-KBot-Timestamp`和
`X-KBot-Signature`后转发。签名桥不映射宿主机端口，不使用静态Bearer代替验签。

同一Central或`all-in-one`节点同时启用`metrics`和`logs`时，脚本还会启用Loki
Ruler。Oracle Alert Collector依据`V$DIAG_ALERT_EXT.MESSAGE_TYPE`和
`MESSAGE_LEVEL`生成统一严重度，Incident Error、Error、Critical和Severe归为
`critical`，Warning归为`warning`。Ruler转发所有这两类结构化异常，不维护ORA错误码
白名单。部分Oracle错误会被ADR标为普通Notification或Important，因此Collector还会
识别Oracle通用的“组件前缀-数字”标准诊断码格式；它同样不枚举ORA编号。因此新出现
的ORA、TNS或其他Oracle组件错误无需修改规则。没有诊断码的普通Notification、Trace
和Dump只保留在Loki供诊断查询，避免正常启动信息造成告警风暴。

Prometheus侧仍由所有处于firing状态的Alerting Rule进入同一个Alertmanager，不按
告警名称设置白名单。新增监控指标时必须同时定义对应的告警条件；仅存在一条指标时序
不代表发生了异常。

## 7. 生成产物

所有运行产物位于`var/aiops-stack/generated/`：

| 产物 | 作用 |
| --- | --- |
| `stack.env` | 固定镜像和非敏感运行变量 |
| `compose.generated.yaml` | 按数据库目标生成的服务、Secret和Volume声明 |
| `secrets/` | 各Exporter、Collector和Webhook签名桥独享Secret |
| `prometheus/prometheus.yml` | Prometheus主配置 |
| `prometheus/targets/kbot.json` | 所有本地及远程数据库抓取目标 |
| `prometheus/rules/` | Oracle语义Recording Rules和告警规则 |
| `prometheus/kbot-aiops-query-overrides.json` | 完整AIOps指标语义到PromQL映射 |
| `alertmanager/alertmanager.yml` | 告警路由配置 |
| `loki/loki.yml` | 单机Loki配置 |
| `loki/rules/fake/kbot-oracle-alerts.yml` | Oracle Alert Log通用异常规则 |
| `deployment.json` | 不含密码的部署清单和Target列表 |

这些文件由脚本生成，不得手工编辑。源配置仍只有INI一份。

## 8. 验收

部署后至少检查：

```bash
docker compose \
  --env-file var/aiops-stack/generated/stack.env \
  -f scripts/deployment/aiops_observability/compose.yaml \
  -f var/aiops-stack/generated/compose.generated.yaml ps

curl -fsS http://127.0.0.1:9090/-/ready
curl -fsS http://127.0.0.1:9090/api/v1/targets
curl -fsS http://127.0.0.1:9093/-/ready
sed -n '1,120p' var/aiops-stack/generated/loki/rules/fake/kbot-oracle-alerts.yml
```

验收标准是每个配置的`target_key`各有一个预期Exporter Target且为`up`，查询结果不
跨Target，Oracle日志带正确`target_key`和`severity`进入Loki，结构化异常日志及
Prometheus测试告警都经过Alertmanager和签名桥被KBot接受。
不要通过停止生产数据库制造测试告警。

自动化部署已经挂载受控Oracle自定义指标，并统一生成Recording/Alerting Rules和
`kbot-aiops-query-overrides.json`，不需要额外执行脚本。只有接入客户已有的systemd或
人工安装Prometheus时，才使用
[configure_prometheus_aiops_oracle.sh](../../scripts/deployment/configure_prometheus_aiops_oracle.sh)。
Target绑定和`mapping_overrides`仍需在AIOps App中逐库登记。

AIOps App中的Prometheus Source Binding需要分别填写数据库指标的`instance`值和该
数据库所在主机的Node Exporter `target_key`；例如数据库为`oracle-dev-190`、主机为
`dev-db-host-190`。Alertmanager继续使用数据库`target_key`。数据库`instance`与主机
`target_key`不得默认视为相同值。

## 9. 当前生产边界

本包已经提供固定镜像Digest、无默认公网端口、角色拆分、逐目标Secret、多数据库目标
发现、持久化Volume、Healthcheck和KBot HMAC签名桥。客户正式上线前仍必须由实施方案
确定TLS入口、网络ACL、资源限制、容量、高可用、对象存储、备份恢复、升级回滚和离线
镜像供应链策略；未完成这些客户级事项时，不得把单机Compose宣称为高可用生产集群。

## 10. MySQL和PostgreSQL指标策略

MySQL和PostgreSQL也需要统一指标，但不建议默认增加任意自定义SQL。

MySQL生产基线应覆盖可用性、当前连接/连接上限、事务提交与回滚、InnoDB Buffer Pool、
锁等待、慢查询、复制状态、磁盘容量和Exporter抓取状态。`mysqld_exporter`默认已提供
Global Status和Global Variables；Processlist、InnoDB Metrics、复制等Collector按
客户版本和权限启用。高基数的用户、SQL文本、表级统计不作为默认采集项。

PostgreSQL生产基线应覆盖可用性、连接数/上限、事务提交与回滚、死锁、缓存命中率、
临时文件、数据库大小、WAL/复制延迟、锁和Exporter抓取状态。默认Exporter指标优先；
`pg_stat_statements`只在客户已安装扩展并接受额外开销时启用。当前
postgres_exporter的自定义`extend.query-path`已被上游标记为弃用，因此需要通用
自定义SQL时应单独评估SQL Exporter，不能把弃用接口作为KBot生产默认能力。

当前自动化包先完整交付Oracle语义指标。MySQL下一阶段可直接基于官方Exporter指标生成
同名`kbot_db_*` Recording Rules；PostgreSQL还需要先把AIOps指标目录的支持数据库类型
扩展到`POSTGRESQL`，再交付规则，避免监控侧有指标而诊断域拒绝使用。
