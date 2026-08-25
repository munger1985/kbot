# AIOps 观测工具选型与 Docker Compose 部署基线

## 状态与范围

本文既是已经确认的目标态设计基线，也是轻量观测栈当前实现说明。仓库已经交付
`metrics`、`logs`、`dashboard`、`host`、`oracle`、`mysql` 和 `postgres` Profile，
以及 `configure`、`up`、`status`、`validate` 四类入口。托管 Zabbix Overlay、生产
级多节点 Loki、备份/升级/卸载和端到端告警注入仍属于后续阶段。

本设计只解决 AIOps 诊断数据工具的选型、组合和部署，不替代 KBot AIOps 的数据库
目标、Diagnostic Source、Evidence、Situation 和 Diagnostic Run 领域设计。工具
负责提供事件和证据，KBot 负责调查、诊断、建议、受控执行、验证与主动分享。

## 已确认决策

1. KBot 提供一个模块化 Docker Compose 项目，通过 Profile 和部署预设按需启动
   观测组件，不要求所有客户安装同一套工具。
2. 默认推荐栈是 Prometheus、Alertmanager、Loki 和 Grafana Alloy。
3. Grafana、Node Exporter 和各数据库 Exporter 均为可选组件。
4. Prometheus 是可选 Metrics/Event Provider，不是 KBot AIOps 的运行时强依赖。
5. Alertmanager 只负责 Prometheus 告警链路，不作为 Zabbix、OEM 和其他来源的
   通用事件总线。
6. Loki 是默认日志证据库，Alloy 是默认日志采集器；已有 OpenSearch、OCI Logging
   或其他日志平台时允许只配置外部 Diagnostic Source。
7. 已有 Zabbix 时直接接入其 Webhook/API；只有用户明确选择 KBot 托管 Zabbix 时
   才启动独立的 Zabbix Compose Overlay。
8. OEM 独立安装和运维。KBot 只接入已有 OEM 的 Incident/Event、Metrics、Target
   等接口，不提供、不封装也不自动安装 OEM、OMS、Management Repository 或
   Management Agent。
9. 没有 OEM 或 Zabbix 不影响 KBot 使用 Oracle/MySQL/PostgreSQL Exporter、日志
   采集和数据库直连诊断。
10. 单机测试环境可以一套 Compose 同时启动 Central 和 Collector；生产环境必须
    支持中央观测栈与数据库侧采集栈分离部署。

## 组件职责与部署归属

| 组件 | 诊断职责 | KBot 随包部署 | 默认状态 |
| --- | --- | --- | --- |
| Prometheus | 指标存储、范围查询、记录规则和告警规则 | 是 | 推荐启用 |
| Alertmanager | Prometheus 告警分组、抑制和路由 | 是 | 随 Prometheus 启用 |
| Loki | 日志存储和按时间窗检索 | 是 | 推荐启用 |
| Grafana Alloy | 文件、容器和遥测数据采集 | 是 | 随日志能力启用 |
| Grafana | 人工 Dashboard、指标和日志探索 | 是 | 可选 |
| Node Exporter | 主机 CPU、内存、磁盘、网络和进程基础指标 | 是 | 可选 |
| Oracle Exporter | Oracle 标准/自定义 Prometheus 指标 | 是 | Oracle Profile 可选 |
| Oracle Alert Collector | 查询诊断视图并输出可断点续采的 JSONL | 是 | 随 Oracle 日志能力启用 |
| MySQL Exporter | MySQL 指标 | 是 | MySQL Profile 可选 |
| PostgreSQL Exporter | PostgreSQL 指标 | 是 | PostgreSQL Profile 可选 |
| Zabbix | 事件、指标、主机和传统基础设施监控 | 独立 Overlay 可选 | 默认不启动 |
| OEM | Oracle Incident、指标、Target 和拓扑 | 否，只接入外部实例 | 默认不要求 |
| OpenSearch/OCI Logging | Loki 的外部日志证据替代源 | 否，只接入外部实例 | 可选 |

所有镜像名称、版本和 Digest 在实现阶段通过受控版本清单固定。Compose 和脚本不得
使用浮动的 `latest` 标签作为正式部署默认值。

## 部署拓扑

### 单机测试拓扑

单机模式用于开发、测试和演示，可以一次启动所有选中组件：

```text
Docker Host
├── KBot AIOps
├── Prometheus / Alertmanager
├── Loki / Alloy
├── Grafana
├── Node Exporter
└── Selected Database Exporter
        └── Remote or Local Database
```

容器之间使用内部 Docker Network 和服务名访问。需要连接目标数据库、远程 Central
或 KBot Webhook的 Collector/Alertmanager额外加入无端口映射的 outbound Network；
这只提供出站连接，不公开监听端口。除经过认证的入口外，Prometheus、Loki、
Alertmanager 和 Exporter默认不绑定公网地址。

### 生产分离拓扑

生产部署分为 Central Stack 和 Collector Stack：

```text
数据库或相邻采集节点                    KBot / 中央观测节点
┌─────────────────────┐                ┌─────────────────────┐
│ Alloy                │── logs ──────▶│ Loki                │
│ Node Exporter        │── metrics ───▶│ Prometheus          │
│ Database Exporter    │── metrics ───▶│ Alertmanager        │
└─────────────────────┘                │ KBot Event Intake   │
                                       │ KBot AIOps          │
                                       └─────────────────────┘
```

`Central Stack` 承担持久化、查询、告警路由和 KBot 接入；`Collector Stack` 靠近数据
库部署，承担本地文件读取和 Exporter。一个 Compose 项目不能直接跨多台远程主机
管理容器，因此标准化的一键部署定义为在每个目标节点执行一次相应角色的安装命令。

已有 Prometheus、Loki、Zabbix、OEM 或云监控时，Central/Collector 中对应组件均可
省略，KBot 通过托管凭据和 Diagnostic Source 配置连接外部平台。

## Compose Profile

目标 Compose 提供以下 Profile：

| Profile | 启动内容 |
| --- | --- |
| `metrics` | Prometheus、Alertmanager及规则配置 |
| `logs` | Loki、Alloy及日志采集配置 |
| `dashboard` | Grafana及预置数据源/Dashboard |
| `host` | Node Exporter |
| `oracle` | Oracle Exporter及 Oracle采集配置 |
| `mysql` | MySQL Exporter及 MySQL采集配置 |
| `postgres` | PostgreSQL Exporter及 PostgreSQL采集配置 |
| `zabbix-managed` | 独立 Zabbix Server、Web和数据库 Overlay |
| `demo` | 仅用于测试的示例数据、规则和 Dashboard |

Profile 表示组件集合，不能承载环境 Secret。数据库密码、API Token、Wallet 和 TLS
私钥由独立 Secret 文件或外部 Secret Manager 注入。

## 部署预设

安装脚本在 Profile 之上提供易用预设：

| Preset | Profile/外部接入组合 |
| --- | --- |
| `oracle-lite` | `metrics + logs + host + oracle`，Grafana可选 |
| `mysql-lite` | `metrics + logs + host + mysql`，Grafana可选 |
| `postgres-lite` | `metrics + logs + host + postgres`，Grafana可选 |
| `oracle-oem` | 外部 OEM，加可选 `logs`；不安装 OEM |
| `external-zabbix` | 外部 Zabbix Webhook/API，按需增加 `logs` |
| `zabbix-managed` | Zabbix独立 Overlay，按需增加 `logs` |
| `external-all` | 不启动观测工具，只登记已有 Diagnostic Source |
| `custom` | 用户显式选择 Profile和外部数据源 |

当前脚本的交互形式为：

```bash
scripts/aiops-stack configure --preset oracle-lite \
  --target-key oracle-dev-01 \
  --oracle-host 10.0.0.20 \
  --oracle-service ORCLPDB1 \
  --oracle-user kbot_monitor
scripts/aiops-stack up --role all-in-one
scripts/aiops-stack status
scripts/aiops-stack validate
```

生产分离部署的目标形式为：

```bash
scripts/aiops-stack up --role central
scripts/aiops-collector
```

`configure` 只接收非敏感参数；密码默认通过无回显终端输入，也可以通过
`--oracle-password-file` 等 Secret 文件参数注入，脚本不提供密码命令行参数。
`--local-access` 只把维护端口绑定到 `127.0.0.1`，不提供公网监听。分离 Collector
写入远程 Loki 时必须把 `--loki-url` 指向已有 TLS/认证入口并提供 Token 文件。

## OEM 外部接入约束

OEM 部署完全独立于 KBot Release：

```text
Existing OEM
├── Incident / Event API ──▶ KBot Event Source Adapter
├── Metrics API ───────────▶ KBot Metrics Evidence Adapter
└── Target API ────────────▶ KBot Topology Evidence Adapter
```

KBot 只保存 OEM Endpoint、TLS Profile、托管凭据引用、Target映射和 Capability，
不保存明文 OEM 密码。OEM不可用时，依赖其证据的任务应形成明确数据缺口；若同一
Target还绑定 Prometheus、Loki或数据库直连，则调查计划可以使用这些来源降级取证。

OEM事件保留 Incident ID、Event ID、Target GUID、原始状态和严重级别。KBot可以
将其与其他来源事件关联为 Situation，但不能覆盖 OEM原始事件，也不能默认向 OEM
执行 Clear、Suppress等写操作。此类能力以后必须通过独立 `SourceActionPort`、权限
和审批策略设计。

## Zabbix 外部与托管模式

外部模式是首选：Zabbix Action Webhook将事件发送给 KBot，KBot通过 Zabbix API
按需查询 Problem、Event、History和 Trend证据。KBot不要求外部 Zabbix改变自己的
存储、Trigger和运维方式。

托管模式使用独立 Compose Overlay，避免 Zabbix Server、Web和数据库成为轻量默认
栈的隐式依赖。托管模式必须独立配置持久化、管理员初始化、备份、升级和资源限制；
停止 KBot AIOps不能自动删除 Zabbix历史数据。

## Oracle Alert Log链路

默认 Oracle日志链路为：

```text
Oracle V_$DIAG_ALERT_EXT
        │
        ▼
KBot Oracle Alert Collector输出结构化 JSONL
        │ shared volume
        ▼
Alloy只读采集 → Loki → KBot Log Evidence Adapter
```

通用 Oracle Exporter只负责 Prometheus指标，并不输出 Alert Log JSON。独立的
Collector和 Alloy共享专用 Volume；Alloy只读挂载。Collector以
`ORIGINATING_TIMESTAMP + RECORD_ID` 为断点顺序，先持久化 JSONL并执行 `fsync`，再
原子更新 checkpoint，重启后继续采集。默认不把整个 Oracle ADR目录暴露给容器。

监控账号使用独立最小权限凭据，不能使用 SYS、SYSTEM或应用账号代替。典型授权由
数据库管理员在目标库审核后执行：

```sql
GRANT CREATE SESSION TO kbot_monitor;
GRANT SELECT ON SYS.V_$DIAG_ALERT_EXT TO kbot_monitor;
```

不同 Oracle版本、CDB/PDB连接位置及客户许可证策略必须在部署前校验。Collector不会
自动提权、创建数据库用户或修改目标数据库。

如果客户已有日志平台，Alloy可以直接发送到该平台，或省略 Alloy并由 KBot连接现有
日志源。告警只携带事件和证据定位信息，原始日志保留在日志平台中；KBot仅把本次
诊断实际使用的有限片段固化为 Evidence Artifact。

## 配置输入与 Secret

数据库 Collector至少需要：

- 数据库类型、主机、端口和数据库/Service标识；
- 实例、集群、环境和 KBot Target映射；
- 最小权限监控用户名；
- 密码、Wallet、TLS证书或外部 Secret引用；
- 启用的指标组、自定义指标和采集周期；
- 是否采集数据库日志及日志来源；
- Central Endpoint、认证和 TLS配置。

安装向导可以自动生成非敏感配置，但不得把 Secret写入 Compose、仓库、普通 `.env`、
命令行参数或容器日志。正式部署优先使用外部 Secret Manager；本地 Compose使用权限
受限且不进入 Git的 Secret文件，并通过 Compose Secrets挂载。

## 网络、持久化与保留

实现必须遵守以下约束：

- Prometheus、Alertmanager、Loki和 Exporter默认仅在内部 Network可见；
- 需要人工访问时统一通过带 TLS、认证和授权的反向代理；
- Collector到 Central的连接必须经过身份认证和传输加密；
- Prometheus、Loki、Grafana和托管 Zabbix使用独立持久化 Volume；
- 指标、日志和告警历史分别配置保留周期，不能无限增长；
- 删除容器不删除持久化数据，清除数据必须使用显式的破坏性命令并二次确认；
- 单机文件存储只用于测试或明确接受单点风险的小规模环境；
- 生产容量、对象存储、备份和恢复策略在部署前显式确认。

Loki本身不作为安全边界。即使没有映射公网端口，KBot查询也必须经过受控网络、
Target绑定、Domain权限、查询时间窗、最大行数/字节数和敏感字段脱敏。

## 安装与运维脚本职责

后续生成的脚本必须至少完成：

1. 检查操作系统、CPU架构、Docker和 Compose版本；
2. 根据 Preset解析实际 Profile、外部依赖和端口；
3. 收集并校验数据库、Central、TLS和 Secret配置；
4. 检查端口冲突、目录权限、磁盘空间和网络连通性；
5. 拉取固定版本/Digest镜像并生成可审计部署清单；
6. 启动组件并等待 Healthcheck和 Readiness；
7. 验证 Prometheus Target、Alertmanager、Loki、Alloy和选中 Exporter；
8. 验证一次测试指标查询、日志查询和告警到 KBot的完整链路；
9. 输出需要登记到 KBot的 Diagnostic Source和 Target绑定信息；
10. 提供幂等重配置、状态、日志、备份、升级、回滚和卸载入口。

升级不能隐式删除 Volume或重建历史数据。卸载默认只停止和删除容器/网络，保留数据；
删除持久化数据必须使用单独命令并明确列出目标。

## 当前实现位置

- Compose与固定镜像清单：
  `services/aiops_agent/deployment/observability/`；
- Oracle Alert Collector：同目录的 `oracle_alert_collector/`；
- 安全入口：`scripts/aiops-stack` 和 `scripts/aiops-collector`；
- 运行时配置与 Secret：默认写入 Git忽略的 `var/aiops-stack/`，也可以通过
  `--state-dir` 或 `KBOT_AIOPS_STACK_HOME` 指向受控目录；
- 静态验收：`python3 tests/acceptance/check_aiops_observability_stack.py`。

当前实现不会自动启动容器；只有显式执行 `up` 才会构建/启动所选服务。OEM没有任何
Compose Service。`external-all` 不启动组件，只保留外部 Diagnostic Source登记模式。

## 后续实现边界

在生成 Compose前，仍需逐项确定：

1. 上游镜像更新审批、许可证清单和部署时镜像签名验证；
2. Prometheus/Loki测试、单机生产、分布式生产容量档位及对象存储；
3. Central入口证书签发、轮换与 Collector身份登记协议；
4. MySQL/PostgreSQL日志采集及三类数据库完整最小授权模板；
5. Exporter自定义指标、数据库规则包和版本兼容矩阵；
6. Oracle Alert Log文件轮转、归档和更大规模重复数据治理；
7. 外部 Zabbix/OEM能力探测、Target映射和托管 Zabbix Overlay；
8. 备份、升级、回滚、卸载和真实依赖端到端验收。

这些细节可以改变具体配置文件，但不能推翻本文“工具可选、OEM独立、外部平台优先
复用、Central/Collector可分离、Secret不落明文”的已确认边界。
