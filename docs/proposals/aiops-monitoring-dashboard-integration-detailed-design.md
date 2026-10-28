# AIOps 原生实时监控与统一 Profile 详细设计

版本：2.0

状态：已批准并实施

基准日期：2026-10-11

2026-10-28 设计复审补充：既有标准监控能力状态不变；第 5.4 节及相关诊断扩展是待实施
目标，实施与验收以 [SQL 诊断详细设计](aiops-sql-diagnosis-detailed-design.md) 为准。

## 1. 决策摘要

AIOps 统一使用原生实时监控页面，不再部署或集成 Grafana Server。观测栈保留
Prometheus、Alertmanager、Loki、Grafana Alloy、Node Exporter 和数据库 Exporter。
其中 Grafana Alloy 仅是日志采集器，与 Grafana Server、Dashboard 或登录体系没有运行依赖。

实时监控与 Agent 智能诊断的标准监控概览共用唯一 Monitoring Profile Catalog 和唯一 Metric
Catalog。标准概览调用方不得按页面、告警类型或数据库版本另写指标清单；必须使用
`profile_id + db_type + version_code` 解析一个确定的 Profile Variant，再从 Variant 取得
Metric Code。问题驱动的临时取证不以 Profile 清单为能力上限，但必须走第 5.4 节的
受控发现与查询，不能复制标准定义或绕过授权/版本策略。

## 2. 设计目标

1. 实时监控与智能诊断使用相同的指标语义、Provider 映射和数据库版本边界；
2. 数据库版本差异只在 Profile Variant 中表达，不散落在 API、页面、告警处理器或查询代码中；
3. 新增数据库版本时只增加并验证 Variant；共用 Variant 只能基于系统视图和采集合同确实未变化；
4. 浏览器只消费标准化 Monitoring View，不直连 Prometheus、Zabbix 或 Loki；
5. Grafana 删除后，不保留隐藏入口、兼容字段、双写配置或 Dashboard 资产；
6. KBot4 与 Ammolite 保持相同产品合同、Catalog 和解析行为。

## 3. 唯一资源模型

### 3.1 Metric Catalog

规范文件：

`services/aiops_agent/src/aiops_agent/resources/metrics/baseline.v1.json`

Metric Catalog 定义稳定 Metric Code、单位、值类型、维度、查询预算，以及 Prometheus、
Zabbix 等 Provider 的受控查询定位。它是标准指标定义的唯一来源，实时监控和诊断都不得复制
已有 Provider 查询模板。诊断发现的原生指标保留真实来源、单位/类型、标签、时间与质量，
不能冒称已有 Metric Code 的等价指标；通用化后需评审纳入同一 Catalog。

### 3.2 Monitoring Profile Catalog

规范文件：

`services/aiops_agent/src/aiops_agent/resources/monitoring/monitoring_profiles.v1.json`

Profile 表达一个运维场景，例如数据库总览、复制、容量或主机资源。每个 Profile 包含一个或多个
Variant：

- `variant_id`：稳定的版本变体标识；
- `db_type`：`ORACLE`、`MYSQL` 或 `POSTGRESQL`；
- `version_codes`：已经完成端到端验证的精确版本集合；
- `metric_codes`：该数据库类型与版本允许查询的标准指标。

当前明确拆分 Oracle 19c、Oracle 26ai、MySQL 8.4 和 PostgreSQL 16。即使两个版本当前使用相同
Metric Code，也保留不同 Variant，以便后续版本演进和审计。只有数据字典、Exporter 输出合同、
单位和 Provider 映射均一致时，才能在同一 Variant 的 `version_codes` 中复用多个版本。

### 3.3 Visualization Profile

图表 Profile 只定义展示槽位、顺序、标题和可视化策略。它不能决定数据库版本，也不能包含另一份
Provider 查询。诊断需要实时概览时，必须先解析 Monitoring Profile，再把解析后的 Metric Code
交给证据采集和图表编译。

## 4. 解析规则

统一解析入口接收：

```text
profile_id + [(db_type, version_code), ...]
```

规则如下：

1. 每个 Target 必须恰好命中一个 Variant；
2. 未配置版本、未知版本、类型不匹配或多个 Variant 同时命中均拒绝；
3. 多实例查询只允许所有实例解析到相同 Metric Code 集合；
4. 返回的 `MonitoringProfileSummary` 包含实际
   `supported_database_versions`、`resolved_variant_ids` 和 `metric_codes`；
5. 缓存键包含 Profile 版本和 Variant ID，Target 版本或 Profile 内容变化后不得复用旧结果；
6. 不允许回退到“仅按 db_type”或“未知版本使用最新版”的隐式选择。

错误使用稳定代码：

- Profile ID 不存在：`MONITORING_PROFILE_NOT_FOUND`；
- 类型、版本或多实例口径不兼容：`MONITORING_PROFILE_INCOMPATIBLE`。

## 5. 调用链

### 5.1 原生实时监控

```text
Target(db_type, version_code)
  -> Monitoring Profile Catalog.resolve
  -> Metric Catalog
  -> Target Source Binding
  -> Prometheus/Zabbix Adapter
  -> Monitoring View
  -> AIOps 原生页面
```

Profile 列表必须按实际 Target 的类型和版本过滤。View 请求仍只提交受控
`profile_id`、实例、来源和时间窗，服务端重新解析并校验，不能信任前端选择结果。

### 5.2 告警与智能诊断

告警诊断、健康总览及确需标准实时概览的 Agent 调查使用
`overview_for_database(db_type, version_code)`。诊断快照记录已解析的 Profile 版本、Variant ID
和 Metric Code，保证证据可回放、可审计。

这不要求每次 SQL 调查先查完概览、复制等所有指标；是否需要概览由问题与证据需求决定。

数据库直连 SQL 诊断工具仍使用 Diagnostic Tool Catalog 的版本 Variant。它和 Monitoring
Profile 职责不同，但必须使用同一个 Target `db_type/version_code`，不得自行猜测版本。

### 5.3 日常巡检

巡检需要趋势证据时，先根据 Target 类型和版本解析 Monitoring Profile，再由检查项或可视化策略
选择该 Profile 内允许的指标。检查项不得直接注入未进入统一 Metric Catalog 的查询。

### 5.4 问题驱动的临时诊断查询（待实施）

Agent 可基于具体问题/假设，在授权 Target 绑定内发现真实可用的指标、标签、日志与
数据库视图，选择已有目录或受控动态查询回答长尾问题。不要求每个临时查询先修改
Metric Catalog；也不把 Monitoring Profile 版本不支持等同于所有直连诊断能力不可用。
各能力仍须满足自身引擎/版本、实际权限与安全策略，不能隐式匹配最近版本。

临时查询必须记录证据需求、源身份与血缘、元数据/测量语义、实际时间/覆盖、变换、
成本/截断与引用；沿用 AST、精确 Target selector、行数/字节/序列/点数/超时配额。
没有真实指标元数据或可解释的单位时，不猜名字、不把空值当 0、不推导因果。
监控与数据库直连按信息价值组合，不使用“监控失败才查库”的串行门禁。

该路径只供有审计的本轮诊断，不作为浏览器任意查询接口，不自动加入日常巡检或
Dashboard，不修改标准 Metric Code 的语义，也不生成第二套指标定义。稳定需求仍走
Catalog/Profile Variant 评审发布；新 Adapter 必须实际注册、绑定并验证后才能调用。

## 6. Grafana 删除后的部署边界

部署中删除：

- Grafana Server 服务、镜像、端口、Volume 和管理员 Secret；
- `[dashboard]` 模块及 `grafana_*` 配置；
- Grafana provisioning、datasource 和 Dashboard JSON；
- API 中的 Dashboard UID、LINK/DISABLED 集成对象；
- Portal 或静态页面中的 Grafana 下钻入口。

继续保留：

- Prometheus 时序采集与告警规则；
- Alertmanager 告警路由；
- Loki 日志存储；
- Grafana Alloy 日志采集；
- Node/Oracle/MySQL/PostgreSQL Exporter；
- AIOps 原生 Monitoring API、页面和 Agent 诊断取证。

已有部署升级后重新执行无参数 `scripts/aiops-stack`。Compose 期望状态将不再包含 Grafana；
Ammolite 的运行时 Secret 清理会删除过期 Grafana Secret。持久化的旧 Grafana Volume 不由应用
脚本主动销毁，确需释放磁盘时应在备份和变更审批后单独处理。

## 7. 新版本接入流程

新增数据库版本必须完成：

1. 在 Target 支持版本合同中登记 `db_type/version_code`；
2. 验证系统视图、Exporter 原始 Series、单位、Label 和空数据语义；
3. 在 Monitoring Profile Catalog 新增或扩展 Variant；
4. 若查询定位变化，在唯一 Metric Catalog 增加版本化 Provider 定义，不在调用方分支；
5. 验证 Prometheus 与 Zabbix 的等价语义和 Gap；
6. 验证实时页面、告警诊断、手工继续诊断和巡检取证得到同一 Variant；
7. 更新合同测试、OpenAPI 和本文档的支持矩阵。

未经验证的版本不能匹配最近版本，也不能显示为“有数据”。

## 8. 验收标准

1. 两个仓库的部署渲染均不含 Grafana 服务、镜像、Secret、端口、Volume 和配置目录；
2. Loki 和 Alloy 仍可启用，`grafana/loki`、`grafana/alloy` 镜像名不视为 Grafana Server；
3. Profile Catalog 对类型和版本精确匹配，未知版本稳定拒绝；
4. Oracle 19c 与 26ai 返回可审计的不同 Variant ID；
5. 实时监控和 Agent 告警诊断调用同一 Profile Catalog；
6. Monitoring View 不再返回 `dashboard` 或 `grafana_dashboard_uid`；
7. Profile 版本和 Variant ID 进入缓存隔离；
8. KBot4 与 Ammolite 的资源、合同和核心测试等价；
9. Portal 与 KBot4 静态页面继续只消费原生 Monitoring View；
10. OpenAPI 快照、部署说明和跨项目实施方案不再承诺 Grafana 能力。

## 9. 后续演进

缺少的专业监控能力直接补到 AIOps 原生页面和统一 Profile，包括更细的容量、复制、锁、等待、
SQL、日志关联、跨实例对比与导出。新增标准页面能力先进入 Metric Catalog 和版本 Variant，
再进入页面；不得重新引入另一套 Dashboard 指标定义。Agent 临时调查按第 5.4 节处理，
不以页面指标发布流程限制其在已授权资源内的证据发现。
