# AIOps Oracle观测栈人工安装与运维

## 1. 目的与范围

本文是人工安装、已有环境接入、故障排查和应急恢复手册。客户标准安装优先使用
[生产自动化部署包](aiops-observability-production-deployment.md)。本文指导运维人员在
开发设备上人工安装、配置和验证以下组件：

- Oracle Database：模拟被监控 Target；
- Oracle Exporter：输出 Oracle Prometheus 指标；
- Prometheus：抓取、存储和查询指标；
- Prometheus Recording/Alerting Rules：形成 AIOps 标准指标和告警；
- Alertmanager：聚合、抑制和路由 Prometheus 告警；
- Node Exporter：补充数据库主机 CPU、内存、磁盘和网络指标；
- Oracle Alert Collector：增量读取 `V$DIAG_ALERT_EXT`；
- Grafana Alloy：采集 Collector 输出的 JSONL；
- Loki：保存并查询 Oracle Alert Log。

Zabbix、OEM 和 Grafana不在本文范围内。OEM只在 AIOps App内配置，不能由本文步骤
安装。本文以单机开发拓扑为基线，生产环境还必须补充 TLS反向代理、认证、容量、备份、
对象存储和高可用设计。

## 2. 当前开发设备基线

本文编写时对 Ubuntu 24.04 x86_64开发设备进行了只读检查，结果如下：

| 项目 | 当前状态 | 处理要求 |
| --- | --- | --- |
| Prometheus | systemd运行，实际监听 `19090` | 保留安装，修正自抓取地址 |
| Node Exporter | systemd运行，监听 `9100`，Target为 `up` | 不重复安装，只验证 |
| Oracle Exporter | Prometheus已配置 `127.0.0.1:9161` | 当前抓取超时，必须先恢复 |
| Prometheus自身 Target | 配置为 `localhost:9090` | 与实际端口不一致，必须改为 `19090` |
| Alertmanager | Prometheus已指向 `localhost:9093` | 服务尚未运行，需要安装 |
| Loki、Alloy、Oracle Alert Collector | 未运行 | 按本文安装 |

这是一份时间点快照。正式操作前必须重新执行预检，不能把表中的状态视为长期事实。

## 3. 最终开发拓扑

```text
Oracle Database
├── Oracle Exporter :9161 ──▶ Prometheus :19090
│                                  │
│                                  ├── Recording/Alerting Rules
│                                  └── Alertmanager :9093
│
├── V$DIAG_ALERT_EXT
│       └── Oracle Alert Collector ──▶ JSONL ──▶ Alloy ──▶ Loki :3100
│
└── Node Exporter :9100 ──▶ Prometheus :19090

KBot AIOps ──query──▶ Prometheus / Loki / Oracle Target
```

开发机端口只应绑定到回环地址或受控网络。Loki没有内置认证能力，不能直接暴露到公网。

## 4. 操作前填写参数表

执行前由运维人员确认以下值。本文示例值不能原样用于其他环境：

| 参数 | 开发示例 | 说明 |
| --- | --- | --- |
| `PROMETHEUS_URL` | `http://127.0.0.1:19090` | 当前设备实际地址 |
| `ORACLE_EXPORTER_URL` | `http://127.0.0.1:9161` | Exporter指标入口 |
| `ORACLE_JOB` | `oracle_db_monitor` | Prometheus Job名称 |
| `ORACLE_INSTANCE` | `oracle-dev-01` | 与 AIOps Source Binding一致 |
| `ORACLE_HOST` | 实际 Oracle地址 | 不要默认写 `localhost` |
| `ORACLE_PORT` | `1521` | Oracle Listener端口 |
| `ORACLE_SERVICE` | 实际 PDB Service | 例如 `pdb01` 或 `FREEPDB1` |
| `ORACLE_USER` | `kbot_monitor` | 独立最小权限监控账号 |
| `LOKI_URL` | `http://127.0.0.1:3100` | 同机开发地址 |
| `AIOPS_TARGET_KEY` | `oracle-dev-01` | KBot Target外部标识 |

不要把真实数据库密码、Webhook Secret、完整 DSN或 API Key写进本文、Git提交、Shell
历史或普通日志。

## 5. 基础预检

在仓库根目录执行：

```bash
systemctl is-active prometheus
systemctl is-active prometheus-node-exporter
systemctl show prometheus --property=ExecStart,User,FragmentPath --no-pager
curl -fsS http://127.0.0.1:19090/-/ready
curl -fsS http://127.0.0.1:9100/metrics >/dev/null
curl -fsS --max-time 15 http://127.0.0.1:9161/metrics >/tmp/oracle-exporter.metrics
```

检查 Prometheus Target：

```bash
curl -fsS http://127.0.0.1:19090/api/v1/targets
```

必须满足：

- Prometheus返回 `Prometheus Server is Ready.`；
- Node Exporter Target为 `up`；
- Oracle Exporter `/metrics` 在15秒内返回；
- Oracle Exporter Target为 `up`；
- `/tmp/oracle-exporter.metrics` 中存在 `oracledb_up`。

当前设备的 Oracle Exporter抓取为超时状态。在它恢复以前，不要执行扩展指标脚本，
因为脚本会检查必要指标并正确地拒绝安装不完整规则。

## 6. 修正并验证现有 Prometheus

当前设备已经安装 Prometheus。仅当 `systemctl status prometheus` 返回 Unit不存在时，
才执行安装：

```bash
sudo apt-get update
sudo apt-get install prometheus
sudoedit /etc/default/prometheus
```

同机开发环境可设置：

```text
ARGS="--web.listen-address=127.0.0.1:19090 --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus/"
```

然后启动并设置开机启动：

```bash
sudo systemctl enable --now prometheus
```

备份配置后编辑 `/etc/prometheus/prometheus.yml`：

```bash
sudo cp -a /etc/prometheus/prometheus.yml \
  /etc/prometheus/prometheus.yml.before-aiops.bak
sudoedit /etc/prometheus/prometheus.yml
```

确认 Alertmanager、Prometheus自身、Node Exporter和 Oracle Exporter配置如下：

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["127.0.0.1:9093"]

scrape_configs:
  - job_name: prometheus
    static_configs:
      - targets: ["127.0.0.1:19090"]

  - job_name: node
    static_configs:
      - targets: ["127.0.0.1:9100"]

  - job_name: oracle_db_monitor
    static_configs:
      - targets: ["127.0.0.1:9161"]
        labels:
          instance: oracle-dev-01
```

`instance` 必须稳定，并与 AIOps App中的 Target Source Binding Locator一致。配置变更
前先检查，检查失败时不得重载：

```bash
sudo promtool check config /etc/prometheus/prometheus.yml
sudo systemctl reload prometheus
curl -fsS http://127.0.0.1:19090/-/ready
```

### 6.1 一个Prometheus监控多个数据库

不要为每个数据库复制Prometheus。每个数据库运行独立Exporter并使用不同监听端口，
Prometheus在同一个Job中登记所有目标：

```yaml
scrape_configs:
  - job_name: oracle_db_monitor
    static_configs:
      - targets: ["10.20.1.11:19161"]
        labels:
          target_key: oracle-prod-01
          environment: production
      - targets: ["10.20.1.12:19161"]
        labels:
          target_key: oracle-prod-02
          environment: production
```

每个Exporter必须使用对应数据库的独立DSN和最小权限账号。Oracle Alert Collector也
必须逐库部署，使用独立Checkpoint和JSONL目录。`target_key`必须唯一且稳定，并与
AIOps App中各数据库Target的Source Binding Locator完全一致。

数据库数量较多时应改用Prometheus `file_sd_configs`维护JSON目标文件，避免频繁修改
主配置。生产自动化部署包已经按此方式生成
`var/aiops-stack/generated/prometheus/targets/kbot.json`。

## 7. 核验或安装 Oracle Exporter

首次准备监控账号时，以SYSDBA连接实际PDB并执行仓库脚本：

```sql
ALTER SESSION SET CONTAINER = PDB01;
SHOW CON_NAME;
@scripts/deployment/aiops_observability/oracle/create_kbot_monitor.sql
```

脚本会隐藏密码输入，拒绝在`CDB$ROOT`运行，并创建固定用户`kbot_monitor`及当前
Exporter、补充指标、Alert Collector和AIOps诊断所需的逐对象最小授权。当前累计Top SQL
读取`V_$INSTANCE`、`V_$DATABASE`和`V_$SQLSTATS`；当前活跃会话和阻塞链读取
`GV_$SESSION`；长事务读取`GV_$TRANSACTION`；表空间容量读取`DBA_DATA_FILES`和
`DBA_FREE_SPACE`；实时性能读取`V_$SYSMETRIC`和`V_$RSRCPDBMETRIC`；内存读取`V_$SGA`和
`V_$PGASTAT`；归档/FRA容量读取`V_$RECOVERY_FILE_DEST`。脚本不授予AWR
历史视图，也不隐含Diagnostics Pack许可。
建用户脚本只用于首次创建。用户已存在时不要重复执行`CREATE USER`，应执行以下完整
授权脚本补齐并验证授权：

```sql
ALTER SESSION SET CONTAINER = PDB01;
SHOW CON_NAME;
@scripts/deployment/aiops_observability/oracle/grant_kbot_monitor.sql
```

当前设备已经在 Prometheus中登记 Oracle Exporter，但 Target抓取超时。先检查现有
进程、端口和日志；只有确认服务不存在时才安装，不能并行启动第二个 `9161` 实例：

```bash
ss -lntp | grep ':9161'
curl -fsS --max-time 15 http://127.0.0.1:9161/metrics | head
```

如果需要重新安装，使用仓库固定的上游镜像和 Secret Wrapper。先创建 DSN文件：

```bash
source scripts/deployment/aiops_observability/images.env
sudo install -d -m 0700 /etc/kbot-aiops/oracle-exporter
sudoedit /etc/kbot-aiops/oracle-exporter/dsn
sudo install -m 0644 \
  scripts/deployment/aiops_observability/configuration/oracle/custom-metrics.yaml \
  /etc/kbot-aiops/oracle-exporter/kbot-custom-metrics.yaml
sudo chown 1000:1000 /etc/kbot-aiops/oracle-exporter/dsn
sudo chmod 0400 /etc/kbot-aiops/oracle-exporter/dsn
```

DSN文件只包含一行：

```text
oracle://kbot_monitor:URL编码后的密码@10.0.0.20:1521/FREEPDB1
```

密码中的 `@`、`:`、`/`、`%` 等字符必须进行 URL编码。地址必须是容器能够访问的
Oracle地址，不能在数据库位于宿主机时误写容器自身的 `127.0.0.1`。

构建并启动 Wrapper：

```bash
docker build \
  --build-arg ORACLE_EXPORTER_IMAGE="${ORACLE_EXPORTER_IMAGE}" \
  -t kbot/oracledb-exporter-secret-wrapper:0.6.0 \
  scripts/deployment/aiops_observability/oracle_exporter_wrapper

docker run -d \
  --name kbot-oracle-exporter \
  --restart unless-stopped \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:9161:9161 \
  -v /etc/kbot-aiops/oracle-exporter/dsn:/run/secrets/oracle_exporter_dsn:ro \
  -v /etc/kbot-aiops/oracle-exporter/kbot-custom-metrics.yaml:/etc/oracledb_exporter/kbot-custom-metrics.yaml:ro \
  kbot/oracledb-exporter-secret-wrapper:0.6.0 \
  --query.timeout=15 \
  --custom.metrics=/etc/oracledb_exporter/kbot-custom-metrics.yaml
```

验证：

```bash
docker logs --tail 50 kbot-oracle-exporter
curl -fsS --max-time 15 http://127.0.0.1:9161/metrics \
  >/tmp/oracle-exporter.metrics
grep '^oracledb_up' /tmp/oracle-exporter.metrics
grep '^oracledb_kbot_cpu_utilization_percent' /tmp/oracle-exporter.metrics
grep '^oracledb_kbot_errors_total' /tmp/oracle-exporter.metrics
```

CPU记录指标没有样本时，先确认`oracledb_kbot_cpu_utilization_percent`是否存在。该指标读取
`V_$RSRCPDBMETRIC`，因此必须在被监控PDB中执行最新授权脚本，并在替换采集配置后重启
Exporter。连接使用率读取Exporter默认的`V_$RESOURCE_LIMIT`指标，可用以下查询区分原因：

```bash
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=oracledb_resource_current_utilization{instance="oracle-dev-190",resource_name="sessions"}'
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=oracledb_resource_limit_value{instance="oracle-dev-190",resource_name="sessions"}'
```

`limit_value=-1`表示Oracle返回`UNLIMITED`，此时不存在可解释的连接使用百分比，记录规则
会有意不生成`kbot_db_connection_utilization_percent`；数值上限大于0但记录指标仍为空时，
再检查Prometheus规则状态和规则文件是否已重载。

只有 `oracledb_up=1` 且 Prometheus Target为 `up` 才算恢复。Exporter监控账号所需
数据字典和动态性能视图权限应按实际启用指标逐项授权，不能为省事使用 `SYS`、
`SYSTEM` 或应用账号。

KBot补充指标还需要读取`SYS.V_$RSRCPDBMETRIC`；`SYS.V_$SYSSTAT`已属于Exporter默认指标
依赖。补充指标分别提供当前PDB最近一分钟相对数据库主机总CPU容量的平均使用率，以及实例启动以来SQL解析失败累计数，
不采集SQL文本、用户名或业务数据。

## 8. 核验或安装 Node Exporter

当前开发设备已经通过 Ubuntu包安装并运行 `prometheus-node-exporter`，通常只需核验：

```bash
systemctl status prometheus-node-exporter --no-pager
curl -fsS http://127.0.0.1:9100/metrics | head
```

仅当服务不存在时安装：

```bash
sudo apt-get update
sudo apt-get install prometheus-node-exporter
sudo systemctl enable --now prometheus-node-exporter
```

安装完成后回到 Prometheus `/api/v1/targets`，确认 `job="node"` 为 `up`。Node
Exporter 默认监听所有接口；开发机防火墙应只允许 Prometheus所在主机访问 `9100`。

## 9. 安装 Alertmanager

本文使用仓库受控镜像清单中的固定版本和 Digest，避免使用 `latest`：

```bash
source scripts/deployment/aiops_observability/images.env
docker pull "${ALERTMANAGER_IMAGE}"
sudo install -d -m 0750 /etc/kbot-aiops/alertmanager
docker volume create kbot-alertmanager-data
```

使用 `sudoedit /etc/kbot-aiops/alertmanager/alertmanager.yml` 创建初始配置：

```yaml
route:
  receiver: kbot-disabled
  group_by: [alertname, instance, severity]
  group_wait: 10s
  group_interval: 1m
  repeat_interval: 4h

receivers:
  - name: kbot-disabled
```

先验证配置：

```bash
docker run --rm \
  --entrypoint /bin/amtool \
  -v /etc/kbot-aiops/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro \
  "${ALERTMANAGER_IMAGE}" \
  check-config /etc/alertmanager/alertmanager.yml
```

启动：

```bash
docker run -d \
  --name kbot-alertmanager \
  --restart unless-stopped \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:9093:9093 \
  -v /etc/kbot-aiops/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro \
  -v kbot-alertmanager-data:/alertmanager \
  "${ALERTMANAGER_IMAGE}" \
  --config.file=/etc/alertmanager/alertmanager.yml \
  --storage.path=/alertmanager \
  --data.retention=120h
```

验证：

```bash
curl -fsS http://127.0.0.1:9093/-/ready
curl -fsS http://127.0.0.1:19090/api/v1/alertmanagers
```

### 9.1 安装KBot Webhook签名桥

先在AIOps App完成接收身份配置：

1. 在“诊断源”新增`ALERTMANAGER`诊断源；Webhook-only场景可以不填访问地址。系统
   固定使用告警中的`target_key`标签，无需配置标签名称。
2. 点击“创建并生成接入凭据”。页面会一次完成创建并集中显示只显示一次的Webhook
   Secret、Webhook Key和INI配置片段，不需要保存后再次进入编辑页面。
3. 复制Secret和Key，分别写入签名桥的`kbot_webhook_secret`和
   `kbot_webhook_key` Secret文件；遗失时只能在编辑页面轮换，不能查询历史明文。
4. 启用诊断源，并建立数据库Target到该诊断源的绑定；Locator必须等于告警中的
   `target_key`。

Webhook Key使用KBot标准必选的`KBOT_MASTER_KEY`按用途派生，不需要执行`openssl`或
增加独立环境变量。若页面提示“Webhook Key 派生密钥未配置”，应先修复KBot主密钥
配置并重启Main API，而不是在监控主机临时生成另一套Key。

KBot公开接收地址是：

```text
POST /api/v1/integrations/aiops/signals/{webhook_key}
```

它要求每次请求携带动态生成的 `X-KBot-Timestamp` 和
`X-KBot-Signature: sha256=<HMAC>`，签名内容是 `timestamp + "." + raw_body`。
Alertmanager原生Generic Webhook不能按请求正文动态计算这两个Header。仓库提供
`scripts/deployment/aiops_observability/webhook_signer/`，标准自动化部署会按INI中的
Webhook URL、Key和Secret自动构建并启动。人工部署时也必须使用这个桥接器，不得用
静态Bearer Token代替HMAC，也不得关闭KBot验签。

手工部署时，分别使用页面生成的Webhook Key和Secret创建只读文件，将它们挂载为
`/run/secrets/kbot_webhook_key`和`/run/secrets/kbot_webhook_secret`，然后构建并运行：

```bash
docker build \
  -t kbot/alertmanager-webhook-signer:4.0.0 \
  scripts/deployment/aiops_observability/webhook_signer

docker run -d \
  --name kbot-webhook-signer \
  --restart unless-stopped \
  --security-opt no-new-privileges:true \
  --network kbot-aiops-observability \
  -e KBOT_WEBHOOK_BASE_URL=https://kbot.customer.example \
  -v /etc/kbot-aiops/webhook/key:/run/secrets/kbot_webhook_key:ro \
  -v /etc/kbot-aiops/webhook/secret:/run/secrets/kbot_webhook_secret:ro \
  kbot/alertmanager-webhook-signer:4.0.0
```

Alertmanager Receiver配置为内部地址：

```yaml
receivers:
  - name: kbot
    webhook_configs:
      - url: http://kbot-webhook-signer:8080/alertmanager
        send_resolved: true
```

签名桥不映射宿主机端口。验证必须包含一条受控测试告警，并在KBot事件收件箱确认返回
`202`且事件被关联到正确Target。

## 10. 安装 Prometheus扩展指标和告警规则

已有systemd或人工安装的Prometheus使用仓库专用脚本：
[configure_prometheus_aiops_oracle.sh](../../scripts/deployment/configure_prometheus_aiops_oracle.sh)。
自动化Compose部署不需要执行此脚本，`scripts/aiops-stack`已经生成相同规则和完整查询
映射。人工部署时脚本会：

1. 检查 Oracle Exporter是否提供必需指标；
2. 安装 AIOps Recording Rules；
3. 安装 Oracle Exporter、数据库、表空间和主机告警规则；
4. 生成 Target Monitor `mapping_overrides`；
5. 使用 `promtool` 检查配置；
6. 备份原配置并重载 Prometheus；
7. 通过 Rules API确认新规则已经加载。

当前设备 Prometheus使用 `19090`，不能使用脚本默认的 `9090`。确认 Oracle Exporter
已恢复后执行：

```bash
sudo \
  PROMETHEUS_CONFIG=/etc/prometheus/prometheus.yml \
  PROMETHEUS_RULE_DIR=/etc/prometheus/rules \
  PROMETHEUS_SERVICE=prometheus \
  PROMETHEUS_URL=http://127.0.0.1:19090 \
  ORACLE_EXPORTER_URL=http://127.0.0.1:9161 \
  ORACLE_JOB=oracle_db_monitor \
  ORACLE_INSTANCE=oracle-dev-01 \
  bash scripts/deployment/configure_prometheus_aiops_oracle.sh
```

预期产物：

```text
/etc/prometheus/rules/kbot-aiops-oracle.yml
/etc/prometheus/rules/kbot-aiops-query-overrides.json
```

验证：

```bash
sudo promtool check rules /etc/prometheus/rules/kbot-aiops-oracle.yml
curl -fsS http://127.0.0.1:19090/api/v1/rules
curl -fsSG http://127.0.0.1:19090/api/v1/query \
  --data-urlencode 'query=kbot_db_active_connections{instance="oracle-dev-01"}'
```

`kbot-aiops-query-overrides.json` 不是自动写入 KBot数据库的文件。创建 Target Source
Binding时，应审核后把其中 `prometheus_queries` 内容配置为该绑定的
`mapping_overrides`。

## 11. 准备 Oracle Alert Collector账号

由 DBA在实际 PDB中创建独立监控账号，并审核后授予：

```sql
GRANT CREATE SESSION TO kbot_monitor;
GRANT SELECT ON SYS.V_$DIAG_ALERT_EXT TO kbot_monitor;
GRANT SELECT ON SYS.V_$SYSMETRIC TO kbot_monitor;
GRANT SELECT ON SYS.V_$RSRCPDBMETRIC TO kbot_monitor;
```

验证查询：

```sql
SELECT ORIGINATING_TIMESTAMP, RECORD_ID, MESSAGE_TYPE, MESSAGE_LEVEL,
       MESSAGE_TEXT, PROBLEM_KEY, COMPONENT_ID, HOST_ID,
       CONTAINER_NAME, DATABASE_ID, SQL_ID, SESSION_ID
FROM V$DIAG_ALERT_EXT
WHERE ROWNUM <= 10
ORDER BY ORIGINATING_TIMESTAMP DESC, RECORD_ID DESC;
```

不要使用 `SYS`、`SYSTEM`、KBot Schema账号或业务应用账号代替监控账号。

## 12. 安装 Oracle Alert Collector

构建仓库内 Collector镜像并创建共享 Volume：

```bash
source scripts/deployment/aiops_observability/images.env
docker build \
  -t "${ORACLE_ALERT_COLLECTOR_IMAGE}" \
  scripts/deployment/aiops_observability/oracle_alert_collector
docker network create kbot-aiops-observability
docker volume create kbot-oracle-alert-log
```

网络或 Volume已存在时，Docker会报告已存在；先检查现有对象，不要为了重跑命令删除
已有数据。

准备仅容器 UID `10001` 可读的凭据文件：

```bash
sudo install -d -m 0700 /etc/kbot-aiops/oracle
sudoedit /etc/kbot-aiops/oracle/username
sudoedit /etc/kbot-aiops/oracle/password
sudo chown 10001:10001 \
  /etc/kbot-aiops/oracle/username \
  /etc/kbot-aiops/oracle/password
sudo chmod 0400 \
  /etc/kbot-aiops/oracle/username \
  /etc/kbot-aiops/oracle/password
```

两个文件分别只写用户名和密码，不加引号。启动前将示例 Oracle地址和 Service替换为
实际值：

```bash
docker run -d \
  --name kbot-oracle-alert-collector \
  --restart unless-stopped \
  --security-opt no-new-privileges:true \
  --network kbot-aiops-observability \
  -e ORACLE_HOST=10.0.0.20 \
  -e ORACLE_PORT=1521 \
  -e ORACLE_SERVICE=FREEPDB1 \
  -e ORACLE_TARGET_KEY=oracle-dev-01 \
  -e ORACLE_POLL_SECONDS=15 \
  -e ORACLE_INITIAL_LOOKBACK_SECONDS=900 \
  -e ORACLE_MAX_ROWS=1000 \
  -v /etc/kbot-aiops/oracle/username:/run/secrets/oracle_username:ro \
  -v /etc/kbot-aiops/oracle/password:/run/secrets/oracle_password:ro \
  -v kbot-oracle-alert-log:/var/lib/kbot/oracle-alert \
  --health-cmd='python /app/healthcheck.py' \
  --health-interval=30s \
  --health-timeout=5s \
  --health-retries=5 \
  "${ORACLE_ALERT_COLLECTOR_IMAGE}"
```

验证：

```bash
docker ps --filter name=kbot-oracle-alert-collector
docker inspect --format '{{.State.Health.Status}}' kbot-oracle-alert-collector
docker logs --tail 50 kbot-oracle-alert-collector
docker run --rm -v kbot-oracle-alert-log:/data:ro busybox \
  sh -c 'ls -l /data && tail -n 5 /data/alert.jsonl'
```

日志中不得出现用户名、密码或完整 DSN。

## 13. 安装单机 Loki

创建 `/etc/kbot-aiops/loki/loki.yml`：

```bash
sudo install -d -m 0750 /etc/kbot-aiops/loki
sudoedit /etc/kbot-aiops/loki/loki.yml
```

开发环境使用单副本文件存储和30天保留期：

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2024-01-01
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h

compactor:
  working_directory: /loki/compactor
  retention_enabled: true
  delete_request_store: filesystem

limits_config:
  retention_period: 720h
  max_query_lookback: 720h
  max_query_length: 744h
  max_entries_limit_per_query: 5000
  allow_structured_metadata: true
```

拉取固定镜像并启动：

```bash
source scripts/deployment/aiops_observability/images.env
docker pull "${LOKI_IMAGE}"
docker volume create kbot-loki-data
docker run -d \
  --name kbot-loki \
  --restart unless-stopped \
  --security-opt no-new-privileges:true \
  --network kbot-aiops-observability \
  -p 127.0.0.1:3100:3100 \
  -v /etc/kbot-aiops/loki/loki.yml:/etc/loki/loki.yml:ro \
  -v kbot-loki-data:/loki \
  "${LOKI_IMAGE}" \
  -config.file=/etc/loki/loki.yml
```

验证：

```bash
curl -fsS http://127.0.0.1:3100/ready
docker logs --tail 50 kbot-loki
```

Loki自身不是安全边界。本文的 `auth_enabled: false` 只允许用于绑定到
`127.0.0.1` 的开发环境。

## 14. 安装 Alloy并采集 Oracle Alert Log

复制仓库维护的 Alloy Pipeline：

```bash
sudo install -d -m 0750 /etc/kbot-aiops/alloy
sudo install -m 0644 \
  scripts/deployment/aiops_observability/configuration/alloy/config.alloy \
  /etc/kbot-aiops/alloy/config.alloy
printf '%s\n' 'local-internal-only' | \
  sudo tee /etc/kbot-aiops/alloy/loki_authorization >/dev/null
sudo chmod 0644 /etc/kbot-aiops/alloy/loki_authorization
```

`local-internal-only` 不是远程认证凭据，只是当前 Alloy配置读取的本地占位值。生产或
远程 Loki必须改为真实短期 Token，并使用受限文件权限和 TLS入口。

启动：

```bash
source scripts/deployment/aiops_observability/images.env
docker pull "${ALLOY_IMAGE}"
docker volume create kbot-alloy-data
docker run -d \
  --name kbot-alloy \
  --restart unless-stopped \
  --security-opt no-new-privileges:true \
  --network kbot-aiops-observability \
  -e AIOPS_LOKI_PUSH_URL=http://kbot-loki:3100/loki/api/v1/push \
  -e AIOPS_LOKI_TENANT=kbot \
  -e AIOPS_TARGET_KEY=oracle-dev-01 \
  -v /etc/kbot-aiops/alloy/config.alloy:/etc/alloy/config.alloy:ro \
  -v /etc/kbot-aiops/alloy/loki_authorization:/run/secrets/loki_authorization:ro \
  -v kbot-oracle-alert-log:/var/lib/kbot/oracle-alert:ro \
  -v kbot-alloy-data:/var/lib/alloy \
  "${ALLOY_IMAGE}" \
  run --storage.path=/var/lib/alloy /etc/alloy/config.alloy
```

验证：

```bash
docker ps --filter name=kbot-alloy
docker logs --tail 50 kbot-alloy
curl -fsSG http://127.0.0.1:3100/loki/api/v1/query_range \
  --data-urlencode 'query={job="oracle_alert",target_key="oracle-dev-01"}' \
  --data-urlencode 'limit=10'
```

查询返回 `status="success"` 且有 Stream，才证明 Collector、Alloy和 Loki链路完整。
Oracle在测试时间窗没有 Alert Log时，可以得到成功但空结果；此时应先确认共享 Volume中
是否有 JSONL，再判断是采集问题还是确实没有新日志。

## 15. 在 AIOps App内完成接入

工具安装完成不等于 KBot已经能使用。管理员还需在 AIOps App执行：

1. 在“运维目标”创建 Oracle Target，填写实际 Host、Port、Service和只读诊断凭据；
2. 在“监控接入”创建 Prometheus Diagnostic Source，Endpoint填写
   `http://127.0.0.1:19090`，或填写 AIOps服务实际可达地址；
3. 创建 Loki Diagnostic Source，Endpoint填写 `http://127.0.0.1:3100`，或填写受控
   反向代理地址；
4. 将 Prometheus Source绑定到 Oracle Target，数据库Locator填写`oracle-dev-01`，
   Node Exporter target_key填写该数据库所在主机的Prometheus标签值；两者可以不同；
5. 审核 `/etc/prometheus/rules/kbot-aiops-query-overrides.json`，将其中
   `prometheus_queries` 写入该 Source Binding的 `mapping_overrides`；
6. 将 Loki Source绑定到同一 Target，绑定标签使用
   `target_key="oracle-dev-01"`；
7. 分别执行 Prometheus、Loki和 Oracle Target的“测试连接”；
8. 创建Alertmanager Diagnostic Source，在页面生成Webhook Secret和Webhook Key，
   绑定Target并配置签名桥后启用事件推送；目标标签统一使用`target_key`。

连接测试 HTTP为 `200` 时仍需检查响应正文中的 `ok`；`ok=false` 仍表示测试失败。

## 16. 端到端验收

### 16.1 指标链路

```bash
python3 tests/acceptance/check_prometheus_metrics.py \
  --url http://127.0.0.1:9161/metrics

curl -fsSG http://127.0.0.1:19090/api/v1/query \
  --data-urlencode 'query=oracledb_up{job="oracle_db_monitor",instance="oracle-dev-01"}'

curl -fsSG http://127.0.0.1:19090/api/v1/query \
  --data-urlencode 'query=up{job="node"}'
```

验收标准：Oracle指标校验脚本通过，`oracledb_up=1`，Node Exporter `up=1`。

### 16.2 告警计算与 Alertmanager链路

```bash
curl -fsS http://127.0.0.1:19090/api/v1/rules
curl -fsS http://127.0.0.1:19090/api/v1/alerts
curl -fsS http://127.0.0.1:9093/api/v2/status
```

使用受控方法造成一个测试规则进入 Firing后，Prometheus和 Alertmanager均应看到该
告警。不要通过停止生产数据库来制造告警。

### 16.3 日志链路

```bash
docker inspect --format '{{.State.Health.Status}}' kbot-oracle-alert-collector
docker logs --tail 50 kbot-alloy
curl -fsSG http://127.0.0.1:3100/loki/api/v1/query_range \
  --data-urlencode 'query={job="oracle_alert",target_key="oracle-dev-01"}' \
  --data-urlencode 'limit=10'
```

### 16.4 AIOps链路

- Prometheus、Loki、Oracle Target连接测试均为 `ok=true`；
- 同一个 Target能够查询指标和 Oracle Alert Log证据；
- `mapping_overrides` 能解析 `db.availability`；
- 手工诊断或巡检 Run形成 Evidence，不出现跨 Target数据；
- Alertmanager到KBot的事件推送必须经过HMAC签名桥，并以KBot返回`202`作为验收证据。

## 17. 运维与故障定位顺序

出现异常时按数据流顺序检查：

1. Oracle Listener、Service和监控账号；
2. Oracle Exporter `/metrics`；
3. Prometheus Target、Rules和 Query；
4. Alertmanager Ready和 Alerts；
5. Collector Health、Checkpoint和 JSONL；
6. Alloy日志；
7. Loki Ready和 LogQL；
8. AIOps Diagnostic Source、Source Binding和连接测试正文。

不要先重启全部组件。只重启确认异常的组件，并在操作后验证进程、端口、Readiness和
最近日志。停止或删除容器时不得附带删除 Volume；`kbot-loki-data`、
`kbot-alertmanager-data`、`kbot-alloy-data` 和 `kbot-oracle-alert-log` 都包含需要保留的
运行数据。

## 18. 官方参考

- Prometheus：<https://prometheus.io/docs/introduction/overview/>
- Prometheus Alertmanager：<https://prometheus.io/docs/alerting/latest/alertmanager/>
- Prometheus Alerting Rules：<https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/>
- Node Exporter：<https://github.com/prometheus/node_exporter>
- Oracle Database Exporter：<https://github.com/iamseth/oracledb_exporter>
- Loki安装：<https://grafana.com/docs/loki/latest/setup/install/>
- Alloy Linux安装：<https://grafana.com/docs/alloy/latest/set-up/install/linux/>
- Alloy Linux权限：<https://grafana.com/docs/alloy/latest/access_permissions/linux/>
