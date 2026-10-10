# AIOps 实时监控 Dashboard 集成详细设计

版本：1.3

状态：待实施，第一阶段范围已确认

基准日期：2026-10-09

## 1. 决策摘要

本设计在现有 AIOps `Dashboard` 的异常决策能力之外，新增由监控源驱动的“实时监控”工作区。
现有 Dashboard 继续回答“现在最需要处理什么”，新工作区回答“这个监控源下的数据库实例现在
发生了什么、指标如何变化、数据来自哪里”。两者不合并成一个无限扩张的首屏。

实时监控具有明确前置条件：管理员必须先在“监控接入”配置并启用 Prometheus 或 Zabbix，完成
连接验证和数据库实例映射。未配置监控源时，页面只展示接入引导；监控源未启用、连接失败或没有
已映射实例时，页面展示对应空状态，不发起指标查询，也不伪造示例数据。

一个监控源可以包含多个数据库实例。第一阶段复用现有一对多关系：同一个 Diagnostic Source 通过
多条 `TargetSourceBinding` 关联多个 Target；每条 Binding 保存该实例在监控产品内的受控定位符。
实时监控支持该来源下的多实例图表总览，并可下钻到单实例详情。

第一阶段只实施以下监控来源：

- Prometheus：支持 Oracle、MySQL、PostgreSQL 的第一阶段标准指标和诊断接入；
- Zabbix：在本次实时监控改造中补齐 Oracle、MySQL、PostgreSQL 的同等指标和诊断接入能力，并
  只连接客户已有的外部 Zabbix；
- OEM：暂缓，不进入新页面、实时监控 API、选源逻辑、验收范围或第一阶段产品承诺。

“能力对齐”以统一产品合同为准，不要求 Provider 协议相同：Prometheus 通过 PromQL、HTTP API 和
配套 Alertmanager 工作，Zabbix 通过 JSON-RPC、Problem API 和原生 Webhook 工作；但两者必须向
AIOps 提供相同的连接验证、实例发现、Target 映射、标准时序、活动事件、告警触发诊断、手工诊断
取证、恢复事件、数据质量和审计语义。任一能力未通过真实环境验收，该来源不得标记为
`diagnostic_readiness=READY`。

第一阶段采用“原生多实例实时监控视图 + Prometheus Grafana 高级下钻”的组合：

1. AIOps 原生页面通过 Main API 读取标准化时序数据，统一 Prometheus 与 Zabbix 的显示、权限、
   数据质量和 Agent 上下文；
2. 已部署的 Grafana 看板作为 Prometheus 专业下钻入口，只有完成认证、Domain/Target 隔离后才可
   在 AIOps 内嵌；不满足隔离条件时只能从受控入口在新窗口打开；
3. 浏览器不得直接访问 Prometheus、Zabbix API，不得获得监控凭据，也不得提交任意 PromQL 或
   Zabbix Item 查询。

## 2. 背景与现状

现有 AIOps Dashboard 是由 Target、Situation、Run、Inspection Fire 和 Finding 批量投影得到的
运维态势页面。它默认每分钟刷新，但不会在首屏对每个 Target 实时调用 Prometheus、Zabbix 或
数据库。这一边界可以防止首屏产生 N+1 查询和监控平台放大流量，应继续保留。

当前仓库已经具备以下基础：

- Prometheus、Alertmanager、Loki、Grafana 和 Exporter 的可选部署资产；
- 固定 UID 的数据库、主机和日志 Grafana Dashboard；
- Prometheus、Zabbix、OEM Diagnostic Source Adapter；
- Target Source Binding、Capability、查询预算和标准 Metric Catalog；
- Prometheus `query_range` 与 Zabbix `history.get` 的标准化时序结果；
- AIOps 静态 UI、Domain 权限校验和 Main API 公共 BFF。

当前 `scripts/aiops-stack` 只声明 `metrics`、`logs`、`dashboard`、`host` 四个模块 Profile，固定
Compose 只包含 Prometheus、Alertmanager、Loki、Alloy、Grafana 和 Node Exporter；数据库段会生成
Exporter，但没有 `[zabbix]` 配置、Zabbix 镜像、Compose Profile、数据库 Volume 或初始化流程。
这一边界继续保持：本设计不向 `aiops-stack.ini`、Compose 或生成脚本增加Zabbix部署能力。

当前模型已经允许一个 Diagnostic Source 通过多条 `TargetSourceBinding` 关联多个 Target，其中
Prometheus Binding 的 `source_locator_key` 对应 Metric Catalog 模板声明的数据库实例定位标签值
（例如 `instance` 或 `target_key`），Zabbix Binding 的 `source_locator_key` 对应 Host 技术名称。
因此一期不需要为“一源多实例”新增业务表，但当前仍缺少：

- 面向浏览器的、按 Domain、Source 与 Target 授权的监控源和实例目录 API；
- 面向浏览器的多实例实时指标公共 API；
- AIOps 原生多实例与单实例时序图表页面；
- Prometheus、Zabbix 的受控实例发现及批量映射流程；
- Grafana 与 AIOps 登录态之间的受控认证桥；
- Grafana 数据源的 Domain/Target 隔离保证；
- Zabbix 专用健康检查、Item Value Type 处理和等价于 Prometheus 的 Adapter 测试；
- 能明确区分“无数据、部分数据、数据过期、来源不可达”的页面合同。

### 2.1 当前支持审计

本次审计基于 2026-10-09 仓库代码、合同、Metric Catalog 与部署资产，不把“已有 Adapter”视为
生产环境已经验收，也没有用真实客户 Prometheus、Zabbix 或 OEM 环境做在线验证。

| 来源 | 仓库当前能力 | 进入实时看板前的主要缺口 | 第一期结论 |
| --- | --- | --- | --- |
| Prometheus | 已注册 `health.check`、`event.query`、`metric.query_range`；有专用 API 健康检查、`query_range` 标准化、Oracle/MySQL/PostgreSQL Metric Catalog、Exporter、规则和固定 UID Grafana Dashboard；告警入站由 Alertmanager Adapter 承担 | 缺少浏览器公共 BFF、来源实例目录、多实例 View、原生图表和 Grafana 登录/租户隔离桥；监控源与 Alertmanager 的逐实例诊断就绪状态尚未统一投影；尚无真实客户环境闭环证据 | 作为对齐基准，但仍须与 Zabbix 一起通过统一验收 |
| Zabbix | 已注册 `health.check`、`event.receive`、`event.query`、`metric.query_range`；可按精确 Host 查询 Problem，并按精确 Item Key 调用 `history.get`；Catalog 当前已声明 Oracle/MySQL 八项统一指标 | 当前健康检查仍是通用 HTTP GET；`history` 类型固定；缺少 PostgreSQL Provider Definition、完整指标映射覆盖、实例发现、批量查询、统一事件语义和真实环境 Smoke；部署栈没有 Zabbix Profile 或容器 | 在本次改造中补齐第 13 节外部接入与能力对齐项后，与 Prometheus 同批验收；不实施容器化部署 |
| OEM | 已有只读 Incident/Metric Adapter 与 Catalog 模板，合同层声明事件和时序能力 | 使用通用健康检查；真实 OEM 版本、认证、Target/Metric 路径、分页与错误语义均未验收；没有本看板所需实例目录与页面合同 | 保留现有代码，不进入第一期实施或验收 |

审计证据锚点：

- Adapter 注册与能力：`services/aiops_agent/src/aiops_agent/adapters/diagnostic_sources/registry.py`；
- Provider 行为：`services/aiops_agent/src/aiops_agent/adapters/diagnostic_sources/prometheus.py`、
  `zabbix.py`、`oem.py`；
- 指标口径：`services/aiops_agent/src/aiops_agent/resources/metrics/baseline.v1.json`；
- 一源多实例关系：`services/aiops_agent/src/aiops_agent/entities/monitoring.py` 中的
  `TargetSourceBindingEntity`；
- Grafana 资产：`scripts/deployment/aiops_observability/configuration/grafana/dashboards/`。
- 当前部署入口与服务清单：`scripts/aiops-stack`、
  `scripts/deployment/aiops_observability/compose.yaml`。

审计结论中的“仓库当前能力”仅说明静态实现存在；只有完成连接、受控实例映射、真实查询与页面
验收后，才能对某个部署环境声称“实时监控可用”。

## 3. 目标与非目标

### 3.1 第一阶段目标

1. 只有至少一个已配置、已启用且具备 `metric.query_range` 能力的 Prometheus 或 Zabbix 监控源时，
   用户才能进入实时数据查询；
2. 用户先选择监控源，再选择该来源下一个或多个已映射数据库实例，查看最近 15 分钟到 24 小时
   的标准实时指标；
3. 多实例总览以图表展示实例状态和趋势，每条曲线具有明确实例图例，并支持下钻到单实例详情；
4. 每个指标明确显示来源、实例、最后采样时间、覆盖率、单位和数据缺口；
5. 用户可以把选定实例、时间窗和异常指标带入智能运维会话；
6. Prometheus 用户可以从同一上下文进入固定 UID 的 Grafana 专业看板；
7. 所有查询由服务端根据 Metric Catalog 和 Source Binding 生成，保持 Domain、Source、Target、权限、
   预算与审计边界；
8. 页面与现有 AIOps 的靛蓝主题、高密度布局、状态语义和中文交互保持一致。
9. Prometheus 与 Zabbix 对 Oracle、MySQL、PostgreSQL 提供相同的第一阶段 Dashboard Profile、
   Metric Code、单位、图形、Gap 和查询预算；
10. 两类监控接入均可查询活动事件、接收故障与恢复告警、映射到同一 Target，并触发同一套自动
    诊断、证据补采、Situation 关联和报告流程。
11. Zabbix和OEM均保持外部接入边界，不进入 `aiops-stack.ini`、Compose、镜像清单或部署脚本。

### 3.2 第一阶段非目标

- 不实施 OEM 实时监控 Dashboard；
- 不新增 OEM Webhook、Topology、Target Discovery 或 OEM Job；
- 不由 KBot 部署或托管 Zabbix Server、Web、数据库、Proxy、Agent或相关容器；
- 不让部署脚本修改、升级或接管客户外部 Zabbix；
- 不要求 Prometheus 与 Zabbix 使用相同查询、告警传输或认证协议；
- 不因监控与诊断取证对齐而扩大数据库变更权限；PostgreSQL 本阶段仍保持只读诊断边界；
- 不开放任意 PromQL、Zabbix Item Key 或 Grafana Explore；
- 不把 Grafana 管理员账号、API Key、Service Account Token 或监控源凭据发送给浏览器；
- 不用实时监控页面替代告警诊断、巡检、报告或智能运维；
- 不在现有 Dashboard 上为所有 Target 同时拉取实时曲线；
- 不展示未映射到当前 Domain Target 的 Prometheus Series 或 Zabbix Host；
- 不允许普通用户通过实时监控页面发现监控平台中的任意 Host、Label 或 Locator；
- 不在第一阶段提供超过 24 小时的 Zabbix Trend 查询，但这一限制不得缩减 24 小时内与
  Prometheus 对齐的指标和诊断能力；
- 不把 Prometheus 与 Zabbix 的同名曲线静默拼接成一条时序。

## 4. 用户与核心场景

### 4.1 运维操作员

- 从 Dashboard 的异常 Target 进入对应监控源与实例，也可从实时监控先选来源再选择多个实例；
- 查看告警前后 CPU、连接、吞吐、延迟和容量变化；
- 在同源多实例曲线中识别异常实例并下钻；
- 判断异常仍在持续、已经恢复，还是监控数据本身不可用；
- 把当前时间窗交给 Agent 继续分析。

### 4.2 DBA

- 在监控源下选择多个数据库实例，比较同一指标趋势；
- 从多实例总览下钻到一个实例，查看数据库运行趋势；
- 同一 Target 同时绑定 Prometheus 与 Zabbix 时，从单实例详情切换来源并核对口径；
- 进入 Grafana 查看更细的表空间、复制、主机或数据库引擎专属指标。

### 4.3 AIOps 管理员

- 在“监控接入”配置、测试并启用 Prometheus 或 Zabbix；
- 从监控产品发现数据库实例并批量映射到 AIOps Target；
- 在 Target Source Binding 中配置来源角色、优先级、受控 Locator、指标范围和映射覆盖；
- 查看来源不可达、指标不支持和映射缺失，不在实时监控页面临时录入凭据。

## 5. 信息架构

```text
业务工作区
├── Dashboard             异常与待办决策
├── 实时监控              同源多实例总览与单实例下钻
├── 智能运维
├── 告警诊断
├── 日常巡检
├── 恢复演练
└── 报告中心

资源配置
├── 运维目标
├── 监控接入             来源配置、连接验证、实例发现与映射
└── 现有其他配置页面
```

新增页面建议为：

```text
ui/aiops/monitoring.html
ui/aiops/js/aiops-monitoring.js
```

公共布局、颜色、按钮、表格和状态样式继续使用 `ui/aiops/css/aiops.css`。只有图表画布、图例、
数据质量和时间轴需要新增监控页面样式，不建立第二套 Design Token。

现有入口调整：

- 未配置任何可用监控源时，“实时监控”展示“前往监控接入”，不展示实例选择器和空图表；
- Dashboard 数据库清单新增“实时指标”，深链到该 Target 的首选来源与实例；
- 优先处理队列新增“查看指标”，同时保留“查看处理”；
- Target 详情的“观测与采集”区域新增“进入实时监控”；
- 告警诊断详情新增“查看告警时间窗指标”；
- 智能运维保持 `target_id` 深链，并接收可选监控上下文。

## 6. 页面详细设计

### 6.1 页面头部

页面标题为“实时监控”，说明文案为“从已接入的监控源查看一个或多个数据库实例的实时指标”。

头部操作区包含：

- 自动刷新：关闭、15 秒、30 秒、1 分钟；
- 立即刷新；
- 进入智能运维；
- Prometheus 来源可用且 Grafana 集成已启用时显示“打开专业大盘”。

刷新按钮附近必须显示：

- 本次生成时间；
- 最新采样时间；
- 当前数据来源；
- 已选实例数；
- 正常、部分成功或不可用状态。

### 6.2 接入门禁与空状态

页面加载后先读取监控源目录，不直接查询指标：

1. 没有 Prometheus/Zabbix 监控源：展示“尚未配置监控源”和“前往监控接入”；
2. 只有 `DISABLED` 监控源：展示“监控源未启用”，有管理权限时提供配置入口；
3. 来源已启用但连接验证失败：展示最后检查时间和稳定错误码，不展示为“无数据”；
4. 来源已启用且可连接但无 ACTIVE 实例映射：展示“尚未关联数据库实例”和“管理实例映射”；
5. 来源和实例满足条件：才加载 Profile，并在用户选择实例后查询时序数据。

普通用户只看到有权使用的来源摘要和已映射实例，不看到 Endpoint、凭据、完整 Locator 或未映射
发现结果。无配置、无映射、连接失败和指标无采样是四种不同状态，必须分别表达。

### 6.3 上下文筛选

```text
监控源 | 数据库实例（多选） | 时间范围 | 指标视图
```

- `监控源`：只列出当前 Domain 内已配置、`ENABLED` 且第一阶段支持的来源；
- `数据库实例`：只列出所选来源下通过 ACTIVE Binding 映射的、当前用户有权查看的已启用 Target；
  默认选择从深链带入的实例，否则按名称稳定排序选择第一个；
- 实例选择支持搜索、全选当前筛选结果和清空，第一阶段一次最多选择 12 个实例；
- `时间范围`：15 分钟、1 小时、6 小时、24 小时；
- `指标视图`：多实例总览、性能、连接、容量、高可用；不可用视图不显示。

筛选发生变化时取消尚未完成的旧请求。连续操作使用 250 毫秒防抖，避免每个下拉动作都产生重复
查询。

### 6.4 实例选择与多实例总览

实例选择和查询状态使用同一张表，不得先展示复选卡片、再重复展示一张实例状态矩阵：

```text
┌ 监控源：生产 Prometheus ─ 实例：核心库、订单库、报表库 ─ 1 小时 ┐
├ 实例选择表：选择 | 实例 | 接入状态 | 查询状态 | 最新采样 | 操作    ┤
├ CPU 使用率趋势                                                   ┤
│ ─ 核心库   ─ 订单库   ─ 报表库                                  │
├ 活动连接趋势                                                     ┤
│ ─ 核心库   ─ 订单库   ─ 报表库                                  │
└ 数据质量：3 个实例正常 · 0 个部分成功 · 0 个不可用               ┘
```

- 实例表每行一个有权访问的 ACTIVE Binding 实例；复选框直接控制下方查询范围；
- 接入状态只表达监控与诊断能力，查询状态和最新采样只取当前 Profile 的实际返回，不使用固定
  CPU 或连接列填充其他 Profile；
- “仅看此实例”把选择收敛到当前行并保留来源、时间范围和指标视图，不再进入第二张实例表；
- 趋势图每个指标一个 Panel，每个已选实例一条 Series，图例使用 Target 展示名；同名时追加短 ID；
- 图例支持显隐单条实例曲线，悬浮提示同时显示实例、时间、值、单位与采样来源；
- 颜色只用于区分曲线，不承担健康状态的唯一语义；
- 超过 6 个实例时默认突出异常或用户固定的 6 条曲线，其余仍可从图例开启；
- 点击图例或曲线进入单实例下钻，并保留来源、时间范围和指标视图。

不同数据库类型只有在所选 Profile 对它们均有统一 Metric Code 时才能同图比较；否则按数据库类型
拆分 Panel，并明确标记“不支持”，不得把缺失数据绘制为零。

### 6.5 单实例当前状态摘要

首行最多展示当前指标视图中前六个有业务意义的指标，不填充装饰性统计。Oracle 与 MySQL
实时总览依次展示：

1. 数据库可用性；
2. CPU 使用率；
3. 活动连接；
4. 连接使用率；
5. 事务吞吐；
6. 响应延迟。

容量视图不使用三张相互割裂的指标卡；它按表空间逐行展示，并在同一张表中并列已用、可用、
最大容量及容量构成，让用户可以直接比较同一对象的容量关系。主机资源或引擎专属视图必须展示
该 Profile 实际返回的指标，不得用未查询的总览指标生成 `NO_DATA` 卡片。每个摘要显示最后值、
单位、相对前一个有效采样的变化方向、采样时间和来源。已查询但没有数据时显示“无有效采样”
与 Gap Code，不显示 `0`。

### 6.6 单实例趋势区

趋势图按业务问题分组，而不是按监控产品分组：

| 分组 | 第一阶段标准指标 | 默认图形 |
| --- | --- | --- |
| 运行状态 | `db.availability` | 状态时间线 |
| 性能 | `db.cpu.utilization`、`db.response.latency` | 双图，不共用单位轴 |
| 连接 | `db.connection.active`、`db.connection.utilization` | 数量与百分比分图 |
| 负载 | `db.transaction.throughput`、`db.error.rate` | 时序图 |
| 容量 | `db.storage.utilization` | 百分比时序图 |

不能因为布局方便而把百分比、毫秒、连接数和每秒事务放在同一纵轴。引擎专属指标由 Profile
声明后追加，不改变统一指标语义。

### 6.7 数据质量与缺口

页面右侧或趋势区下方展示“数据质量”：

- 来源、实例与 Binding；
- 采样窗口与实际覆盖率；
- 期望点数与有效点数；
- 是否截断；
- 是否使用 Prometheus 回退查询；
- 不支持、无数据、限流、认证失败、不可达或响应无效等 Gap；
- Zabbix Item Key 或 Prometheus Query Template 只展示稳定引用，不返回完整凭据或敏感查询内容。

状态必须使用中文文字、边框与颜色共同表达，不能只依赖红绿颜色。

多实例请求中某个实例失败时，其他实例曲线仍可展示；Gap 必须带稳定的 `instance_id`，并在状态
矩阵对应行显示。整源失败与单实例失败不得合并为同一错误。

### 6.8 单实例多来源对比

多来源对比只在单实例下钻中提供。用户主动开启后，页面可以同时查询该 Target 已绑定的
Prometheus 与 Zabbix，但必须：

- 各自保留来源名称和独立图例；
- 不自动平均、拼接或相互补点；
- 单独计算覆盖率和数据时间；
- 口径或单位不一致时显示“不可直接比较”；
- 任一来源失败时保留另一来源结果，并明确显示部分成功。

### 6.9 图表实现约束

第一阶段采用仓库本地固定版本的 Apache ECharts，不从 CDN 动态加载。依赖放在 `ui/vendor/`，记录
版本和许可证；页面脚本只消费标准化 `MonitoringPanel`，不感知 Prometheus 或 Zabbix 查询语法。

- 时间轴统一使用 UTC 数据并按浏览器时区显示，Tooltip 同时给出带时区的完整时间；
- 折线不跨越缺失区间，`null` 保持断点，不使用零值补点；
- Series、点数和动画遵守查询预算，自动刷新时不重建整页 DOM；
- Canvas 图表必须配套可访问的摘要和可切换数据表，键盘可以操作图例与下钻入口；
- 图例、Tooltip 和轴标签全部以文本方式写入，不接受 Provider 返回的 HTML；
- 页面打印或导出只导出当前授权结果，不提供浏览器直连监控源的导出接口。

## 7. 第一阶段指标范围

### 7.1 Prometheus

Prometheus 使用现有 Metric Catalog 中声明的 Provider Template，是本次对齐的当前实现基准。
第一阶段包含 Oracle、MySQL、PostgreSQL 已进入 Dashboard Profile 的所有标准指标，时间范围最大
24 小时，单序列最大 240 个点，最终上限仍以 Metric Definition 和查询预算中更严格者为准。

页面不得接收或回显任意 PromQL。Binding 的 `prometheus_queries` 覆盖仍由管理员配置，并在执行前
通过现有 PromQL AST 策略验证 Target Scope。

### 7.2 Zabbix

Zabbix 不再只实现 Oracle/MySQL 八项统一指标。本次改造必须为第一阶段 Dashboard Profile 中每个
Prometheus Metric Definition 增加等价 Zabbix Provider Definition，并支持相同数据库范围：

| 指标组 | 数据库范围 | 对齐要求 |
| --- | --- | --- |
| 数据库统一指标 | Oracle、MySQL | `db.availability`、CPU、活动连接、连接使用率、事务吞吐、响应延迟、存储使用率、错误率 |
| 数据库容量 | Oracle、MySQL | `db.storage.used_bytes`、`db.storage.free_bytes`、`db.storage.max_bytes` |
| 主机资源 | Oracle、MySQL | CPU、内存、文件系统、磁盘 IO、网络吞吐五项 `host.*` 指标 |
| MySQL 专属指标 | MySQL | 可用性、连接使用率、事务吞吐、慢查询率、行锁等待率、Buffer Pool 命中率六项 `mysql.*` 指标 |
| PostgreSQL 专属指标 | PostgreSQL | 可用性、活动连接、事务吞吐、复制延迟、Slot WAL 保留、数据库容量六项 `postgresql.*` 指标 |

Zabbix Template 可以使用原生采集或依赖 Agent/UserParameter，但 Metric Catalog 中的 Item 必须具有
明确的精确 Item Key、`value_type`、单位和取值语义。不能仅凭 Item 名称相似就声明等价；计算型
指标必须说明采集端公式，并用相同输入样本验证与 PromQL 结果的允许误差。

默认 Item Key 来自 Metric Catalog。客户现有 Template 与默认 Key 不一致时，通过 Source Binding
的映射覆盖配置，Oracle、MySQL 与 PostgreSQL 使用同一合同：

```json
{
  "zabbix_item_keys": {
    "db.cpu.utilization": "customer.db.cpu.utilization"
  }
}
```

映射必须是“标准指标代码到精确 Item Key”的对象，不允许前端提交通配符、正则表达式或自由
JSON-RPC。第一阶段查询 `history.get`，最大时间范围 24 小时；`trend.get` 留到后续阶段，但 24 小时
以内的 Profile、时间窗、单位、点数、Gap 和图表必须与 Prometheus 对齐。

### 7.3 对齐准入规则

第一阶段不允许同一标准 Profile 因来源不同而静默减少指标。每个 Metric Code 只有同时满足以下条件
才能标记为 `ALIGNED`：

1. Prometheus 与 Zabbix Provider Definition 均通过启动时合同校验；
2. 两端的单位、值类型、聚合语义、最小步长、最大点数和 Series 维度兼容；
3. Oracle、MySQL、PostgreSQL 适用范围一致；
4. 空数据、目标不存在、认证失败、超时、限流和响应超限映射为同一 Gap Code；
5. 使用同一受控测试样本完成结果误差、时间戳和缺口对比；
6. 在真实 Prometheus 与 Zabbix 环境分别完成 Smoke。

未通过的指标在管理员页面显示“待对齐”，不能只在某一来源的统一 Profile 中出现。Grafana 中的
Prometheus 专业扩展指标可以继续存在，但不计入 AIOps 原生看板的对齐承诺。

### 7.4 OEM

OEM 在第一阶段明确暂缓：

- 不出现在实时监控页面的数据来源选项；
- 不返回 OEM Dashboard Profile；
- 不把 OEM 纳入第一阶段公共 API 示例和验收；
- 不增加 OEM Grafana 数据源或 OEM 专属页面；
- 不删除现有 OEM Diagnostic Source Adapter，避免扩大本设计的变更范围；
- 后续必须在真实 OEM 环境验证认证、Incident、Metric、Target 和版本差异后另立设计与验收。

## 8. 监控源、实例发现与映射规则

### 8.1 监控源可用条件

监控源目录返回当前 Domain 内用户可见的第一阶段来源，并给出 `monitoring_readiness`：`READY`、
`DISABLED`、`DISCONNECTED`、`CAPABILITY_MISSING` 或 `NO_MAPPED_INSTANCE`。只有同时满足以下条件
的来源才是 `READY`，也只有 `READY` 来源可以调用 View API：

1. 属于当前 Domain；
2. Source 状态为 `ENABLED`；
3. Source Type 为第一阶段的 `PROMETHEUS` 或 `ZABBIX`；
4. 声明或连接验证发现 `metric.query_range` 能力；
5. 最近一次连接验证为 `CONNECTED`；
6. 当前用户至少能查看该来源下一个 ACTIVE Binding 对应的 Target。

管理员可以在目录中看到无映射来源以完成配置；普通用户只看到有授权实例的来源。连接快照过期时
先触发受预算约束的连接复核，再决定是否查询指标。`DISCONNECTED` 来源不能被静默隐藏成
“未配置”，也不能发起多实例指标查询。

### 8.2 一源多实例模型

一期不新增“监控实例”业务表，实例目录是以下数据的授权投影：

```text
Diagnostic Source 1
  ├── Target Source Binding A → Target A（数据库实例 A）
  ├── Target Source Binding B → Target B（数据库实例 B）
  └── Target Source Binding C → Target C（数据库实例 C）
```

目录中的稳定 `instance_id` 使用 `target_id`，`binding_id` 用于查询与审计，展示名来自 Target。
Prometheus 使用 Binding 中已验证、且由 Metric Catalog 模板引用的数据库实例标签值，Zabbix 使用
已验证的 Host 技术名称。
同一来源下 `(source_id, source_locator_key)` 只能存在一个 ACTIVE 映射，该业务唯一性由应用服务在
事务内检查，不新增数据库 `UNIQUE` 约束。

### 8.3 发现与映射

实例发现只属于“监控接入”的管理员流程，不属于实时看板：

- Prometheus 通过服务端白名单元数据查询发现 Metric Catalog 所需指标中出现的数据库实例标签；
- Zabbix 通过服务端 `host.get` 与受控 Template/Tag 过滤发现数据库 Host；
- 发现结果只返回脱敏候选摘要和服务端生成的短期 `candidate_ref`；
- 管理员将候选映射到已有 Target，或先在“运维目标”创建 Target，再完成 Binding；
- 保存映射时服务端解析 `candidate_ref`，重新验证候选仍存在、类型匹配且未被当前来源重复绑定；
- 普通用户和实时看板不得提交原始 PromQL、Zabbix Host、任意 Label Matcher 或 Locator；
- 未映射候选不进入实时监控实例目录，也不能被批量查询。

发现失败不删除已有 Binding。管理员仍可看到现有映射及其健康状态，但新增映射必须在来源重新连接
并完成发现后进行。

### 8.4 深链与单实例来源选择

从 Dashboard、告警或 Target 详情进入时，服务端按该 Target 的 ACTIVE Binding 解析来源：先按
`PRIMARY`、`SUPPLEMENTARY`、`FALLBACK`，同一 Role 再按 `priority` 与稳定 ID 排序。深链最终仍
落到“来源 + 实例”上下文。

用户在实时监控页显式选择来源后，不跨来源自动回退；来源失败时直接展示该来源的 Gap。只有单实例
详情可主动开启第二来源对比，且两种来源各自保留口径与数据质量。

### 8.5 监控与诊断能力对齐

“监控接入”对 Prometheus 和 Zabbix 展示同一组逐实例能力，不以 Adapter 数量定义产品差异：

| 产品能力 | Prometheus 路径 | Zabbix 路径 | 统一验收结果 |
| --- | --- | --- | --- |
| 连接验证 | Prometheus Build Info/API | Zabbix JSON-RPC 最小查询 | `health_check=READY` |
| 实例发现 | 白名单 Label/Series 元数据 | `host.get` + Template/Tag | 候选可映射为同一 Target |
| 时序指标 | `query_range` | `item.get` + `history.get` | 同一 Metric Code、单位、窗口和 Gap |
| 活动事件 | Prometheus `/api/v1/alerts` | `problem.get` | 同一事件查询合同 |
| 告警入站 | 关联的 Alertmanager Diagnostic Source | Zabbix 原生 Webhook | 同一 `SignalEvent` 合同 |
| 故障与恢复 | Alertmanager `firing/resolved` | Zabbix `PROBLEM/OK` | 同一状态迁移、幂等和恢复闭环 |
| Target 关联 | Alertmanager `target_key` Binding | Zabbix Host Binding | 服务端解析到同一授权 Target |
| 自动诊断 | Signal → Situation → Run | Signal → Situation → Run | 相同严重度门槛、冷却、预算和报告 |
| 手工诊断 | Agent 主动补采 Prometheus 时序 | Agent 主动补采 Zabbix 时序 | 相同 Evidence、Gap 和引用合同 |
| PostgreSQL | Prometheus PostgreSQL Provider | Zabbix PostgreSQL Template/Item | 相同 Profile 与诊断流程 |

Prometheus 的“监控接入”在产品层由 Prometheus 指标源和关联的 Alertmanager 事件源共同完成；
Zabbix 可以由同一个 Diagnostic Source 同时承担查询与 Webhook。页面仍以一张监控接入卡展示
`metrics`、`event_query`、`alert_ingress`、`target_mapping` 四项 readiness，不能因为内部资源数量
不同而给用户两套配置心智。

逐实例状态定义为：

- `monitoring_readiness=READY`：来源已启用且连接正常，Target 映射有效，所有必需 Dashboard 指标
  可查询；
- `diagnostic_readiness=READY`：在监控就绪基础上，活动事件查询、告警入站、事件类别映射、
  故障/恢复归一化和 Agent `monitor.query_range` 取证均已配置；
- `diagnostic_readiness=PARTIAL`：至少一项缺失，页面必须列出缺口，不能笼统显示“接入成功”。

Readiness 是配置与运行态投影，不是 Webhook 路由开关。只要事件来源仍为 `ENABLED` 且验签通过，
即使当前指标连接失败也要接收入站事件并创建带 Gap 的诊断；不能因为一次连通性波动丢弃故障或恢复
事件。

Prometheus 与 Alertmanager 的内部关联不改变 Target Source Binding 的精确定位规则；二者可以使用
不同 Locator，但必须分别映射到同一 Target。Zabbix Trigger 名称或 Prometheus告警文本都不能用于
猜测 Target 或事件类别；Target 由 Binding 定位，规范事件类别由受控标签、Trigger Tag 或
`event_class_map` 显式映射。

## 9. API 与合同设计

### 9.1 公共 Main API

```http
GET /api/v1/apps/aiops/monitoring/sources

GET /api/v1/apps/aiops/monitoring/sources/{source_id}/instances

GET /api/v1/apps/aiops/monitoring/sources/{source_id}/profiles

POST /api/v1/apps/aiops/monitoring/sources/{source_id}/views
Content-Type: application/json

{
  "profile_id": "database-overview",
  "instance_ids": ["target-uuid-1", "target-uuid-2"],
  "window": "1h"
}
```

多实例查询使用 `POST`，避免把实例数组和未来筛选条件塞入 URL。请求体只接受来源目录返回的
`instance_id`，每个 ID 都必须在服务端重新校验 Domain、Target 和 ACTIVE Binding。浏览器不能
提交 `binding_id`、`source_locator_key` 或 Provider 查询。

“监控接入”的管理员实例发现与批量映射扩展现有诊断源和 Source Binding API：

```http
POST /api/v1/apps/aiops/diagnostic-sources/{source_id}/instance-discoveries
POST /api/v1/apps/aiops/diagnostic-sources/{source_id}/instance-mappings
```

发现接口只接受服务端定义的分页和数据库类型筛选；映射接口接收 `candidate_ref` 与 `target_id`，
不接受原始 PromQL。Zabbix Host 技术名称和 Prometheus Locator 仅在具有配置权限的映射确认页按
脱敏规则展示，保存时仍由服务端从 `candidate_ref` 解析并复核。

单实例下钻和跨来源对比使用：

```http
GET /api/v1/apps/aiops/targets/{target_id}/monitoring/profiles

GET /api/v1/apps/aiops/targets/{target_id}/monitoring/view
    ?profile_id=database-overview
    &window=1h
    &source_id=<uuid>
    &compare_source_id=<optional-uuid>
```

`source_id` 在单实例查询中为必填；它必须与该 Target 存在 ACTIVE Binding。`window` 只能取服务端
枚举值，不接收任意起止时间。告警诊断需要精确时间窗时，由受控深链使用：

```http
GET /api/v1/apps/aiops/targets/{target_id}/monitoring/view
    ?profile_id=database-overview
    &source_id=<uuid>
    &window_start=<utc>
    &window_end=<utc>
```

精确时间窗必须同时提供，跨度不超过 24 小时，结束时间不得位于未来。

### 9.2 内部 AIOps API

Main API 使用现有 Service Credential 与 audience-bound AuthContext JWT 调用：

```http
GET /internal/v1/aiops/monitoring/sources
GET /internal/v1/aiops/monitoring/sources/{source_id}/instances
GET /internal/v1/aiops/monitoring/sources/{source_id}/profiles
POST /internal/v1/aiops/monitoring/sources/{source_id}/views
GET /internal/v1/aiops/targets/{target_id}/monitoring/profiles
GET /internal/v1/aiops/targets/{target_id}/monitoring/view
```

内部 API 不对浏览器暴露，不信任调用方自行填写的 Domain 或 Actor Header。

### 9.3 响应模型

```json
{
  "schema_version": "aiops.public.v1",
  "generated_at": "2026-10-09T10:00:00Z",
  "refresh_after_seconds": 30,
  "source": {
    "source_id": "uuid",
    "display_name": "生产 Prometheus",
    "source_type": "PROMETHEUS",
    "monitoring_readiness": "READY",
    "diagnostic_readiness": "READY"
  },
  "window": {
    "start": "2026-10-09T09:00:00Z",
    "end": "2026-10-09T10:00:00Z"
  },
  "instances": [
    {
      "instance_id": "target-uuid-1",
      "display_name": "核心库",
      "db_type": "ORACLE",
      "status": "AVAILABLE",
      "monitoring_readiness": "READY",
      "diagnostic_readiness": "READY",
      "capability_gaps": []
    }
  ],
  "panels": [],
  "gaps": [],
  "partial": false
}
```

核心合同建议：

```text
MonitoringProfileSummary
MonitoringView
MonitoringSourceSummary
MonitoringSourceReadiness
MonitoringDiagnosticReadiness
MonitoringInstanceSummary
MonitoringWindow
MonitoringPanel
MonitoringSeries
MonitoringPoint
MonitoringGap
```

`MonitoringPanel` 只描述经审核的展示语义：

- `panel_id`、`title`、`description`；
- `visualization`：`STAT`、`STATE_TIMELINE`、`TIME_SERIES`、`GAUGE`；
- `unit`、`value_kind`；
- `series`；
- `summary`；
- `quality`。

API 不返回 Provider 原始响应、不返回凭据、不返回可执行查询文本。

`MonitoringSeries` 必须包含 `instance_id`、`instance_display_name` 和稳定 `series_key`。Provider Label
只经过服务端白名单投影，不能原样透传。`MonitoringGap` 可以位于来源、实例或指标层级，并通过
`scope` 与可选 `instance_id` 明确影响范围。

## 10. 应用与持久化边界

实时监控属于 AIOps 应用用例，API Adapter 只解析参数并返回合同。实现必须保持：

```text
HTTP API
  → Monitoring Catalog / View Application Service
  → Unit of Work / Repository 读取 Source、Target、Binding
  → Monitoring Snapshot Builder 冻结来源、实例、权限、窗口、指标和预算
  → Managed Credential Resolver
  → Diagnostic Source Adapter
  → 标准化 Monitoring View
```

禁止：

- API 层直接持有 Session 或执行 SQL；
- API 层直接调用 Prometheus/Zabbix；
- Repository 调用 `commit()`；
- 前端依据 Source Type 拼接监控查询；
- 前端使用未进入实例目录的 Locator 发起查询；
- 为实时页面新建第二套 Metric Catalog；
- 把查询结果先写入数据库再读回，只为绘制即时图表。

第一阶段无需新增业务表：一个 Source 的多个实例由现有多条 `TargetSourceBinding` 表达，实例目录
由 Source、Binding 与 Target 实时投影。Profile 使用版本化资源文件，例如：

```text
services/aiops_agent/src/aiops_agent/resources/monitoring/
└── dashboard_profiles.v1.json
```

资源文件引用稳定 Metric Code 和可选 Grafana UID，由 Pydantic 合同启动时校验。Target、Source、
Binding 和凭据继续使用现有规范实体和 Repository。

## 11. 查询执行、缓存与限流

### 11.1 查询预算

每次页面请求遵守以下上限：

- 多实例视图固定查询一个显式来源；
- 一次最多 12 个实例，Profile 可以设置更低值；
- 单次最多 8 个指标；
- 每个实例、每个指标最多 8 个 Provider Series；标准总览原则上归一为一个实例一条 Series；
- 单次响应最多 96 条标准化 Series；
- 每个 Series 最多 240 个点；
- 总响应字节预算沿用 AIOps 监控配置；
- Provider 超时沿用 Diagnostic Source 请求超时；
- 单实例对比模式最多两个来源；
- 一个用户对同一 Source 的并发实时查询最多一个，旧请求可取消；
- Provider 支持正则或集合匹配时，服务端可将同一指标的多个已授权 Locator 合并为一次查询；否则按
  实例分批执行，但必须服从总调用数、超时和响应字节预算。

超过实例、Series 或响应预算时，接口返回稳定的 `MONITORING_QUERY_BUDGET_EXCEEDED`，不静默漏掉
用户已选实例。服务端不得为减少查询次数而放宽 Label/Host 范围到未授权实例。

### 11.2 短缓存

允许使用 15 秒进程内短缓存，缓存 Key 至少包含：

```text
domain_id + source_id + sorted(instance_ids) + profile_id
+ compare_source_id + window + end_time_bucket
+ sorted(binding_versions) + source_config_version
```

缓存只保存标准化结果，不保存明文凭据或 Provider 原始 Payload。配置版本变化后自然失效。

### 11.3 部分成功

单个指标或单个实例失败不使整个请求返回 500。只要仍有一个实例的有效 Panel，接口返回 200、
`partial=true` 和带实例范围的结构化 Gap。以下情况才返回请求级错误：

- Source 不存在、不属于当前 Domain、未启用或不在第一阶段范围；
- 任一请求实例不存在、越权，或不属于该 Source 的 ACTIVE Binding；
- 用户无权使用 AIOps；
- 参数或时间窗无效；
- Profile 与所有选定实例的数据库类型均不兼容；
- 查询预算在执行前已超限。

## 12. Prometheus Grafana 高级下钻

### 12.1 固定 Dashboard 映射

第一阶段只允许仓库内受控 UID：

| 场景 | Grafana UID |
| --- | --- |
| 数据库 Fleet | `kbot-database-fleet` |
| Oracle 总览 | `kbot-oracle-overview` |
| Oracle 存储 | `kbot-oracle-storage` |
| Oracle 异常 | `kbot-oracle-alerts` |
| Oracle Alert Log | `kbot-oracle-alert-log` |
| MySQL 总览 | `kbot-mysql-overview` |
| PostgreSQL 总览 | `kbot-postgresql-overview` |
| 主机总览 | `kbot-host-overview` |

多实例总览只允许进入 `kbot-database-fleet`，服务端根据所选实例的 ACTIVE Binding 写入受控实例
变量；单实例页面根据 Target DB Type 和 Profile 选择 UID，并由服务端写入 `var-target_key`、
`from`、`to`。浏览器不能提交任意 Grafana 路径、UID、变量名或原始 Locator。若现有 Grafana
Dashboard 不能安全锁定多个实例，则多实例页面不显示 Grafana 入口，仅保留 AIOps 原生图表。

### 12.2 安全前置条件

当前 Grafana 只配置管理员用户和密码，不能直接用于 AIOps iframe。启用内嵌前必须同时满足：

1. AIOps 登录态可以交换短期、HttpOnly、限定路径的 Embed Session；
2. Grafana 不使用管理员身份向普通用户提供页面；
3. 禁用 Explore、Dashboard 编辑、数据源配置和任意查询入口；
4. 只允许白名单 UID；
5. 单实例或多实例 Target Key 均由服务端从已授权 Target Source Binding 解析；
6. Grafana 数据源实现 Domain/Target 隔离，或部署被明确证明为单 Domain 专用；
7. 网关转发 WebSocket、静态资源和必要 Header，并配置精确 CSP `frame-src`；
8. 审计记录用户、Domain、Target、Dashboard UID 和时间窗，不记录凭据。

任一条件不满足时，`integration_mode` 必须为 `LINK` 或 `DISABLED`，不得退化为匿名访问、共享
管理员账号、URL 内 Token 或把 Grafana 暴露到 `0.0.0.0`。

### 12.3 单一部署配置

Grafana 集成继续使用唯一 `aiops-stack.ini` 的 `[dashboard]`，不增加第二份人工配置。建议未来
扩展以下非敏感项：

```ini
[dashboard]
enabled = true
integration_mode = link
grafana_public_base_url = https://monitor.customer.example
```

密码和 Token 仍由入口脚本转换为逐服务 Secret，不写入生成的前端 Runtime Config。

## 13. Prometheus 与 Zabbix 能力对齐实施

### 13.1 Zabbix 外部接入边界

Zabbix 与 OEM 一样只作为客户已有的外部平台接入。第一阶段固定以下边界：

- `scripts/aiops-stack`、`aiops-stack.ini`、Compose、`images.env` 和Role服务集合均不增加Zabbix；
- KBot不安装、启动、升级或备份Zabbix Server、Web、数据库、Proxy、Agent及其容器；
- Zabbix Endpoint、API凭据、TLS配置和Webhook Secret只在AIOps App“监控接入”中配置，并保存为
  Managed Credential引用；
- 连接测试、Host发现、Template/Item校验、实例映射和Webhook测试均由AIOps服务端通过外部Zabbix
  API完成，浏览器不直接访问Zabbix；
- KBot不自动导入Template、创建Host、修改Trigger或变更客户Zabbix配置；管理员根据KBot提供的
  版本化接入说明在外部Zabbix完成必要准备；
- 外部Zabbix的容量、保留、备份、升级、Agent和Proxy运维由客户负责，不纳入KBot部署验收；
- Prometheus部署能力保持现状，不因Zabbix外部接入而增加目标段Provider开关或改变Exporter生成。

Zabbix是否完整接入只看外部链路的能力验证结果，不看任何本地容器状态。只有连接、实例发现、
标准指标、活动事件、Webhook及诊断闭环都通过，才返回 `monitoring_readiness=READY` 和
`diagnostic_readiness=READY`。

### 13.2 Zabbix 基础 Adapter 加固

Zabbix 进入第一阶段验收前必须完成：

1. 实现 Zabbix 专用 `health_check`，通过 JSON-RPC 验证 API、认证和最小查询，而不是对 Endpoint
   做通用 HTTP GET；
2. 明确支持的认证模式并形成合同，不能把所有 Token 都模糊处理为同一种认证；
3. `item.get` 同时读取 `value_type` 和单位，为 `history.get` 选择正确 History Type；
4. 支持 Binding `zabbix_item_keys` 精确映射覆盖；
5. 对重复 Host、缺失 Item、多个同 Key Item、无历史采样、限流、认证失败和响应过大返回与
   Prometheus 一致的稳定 Gap；
6. 为事件查询、指标查询、Webhook 验签、健康检查和错误归一化增加独立单元测试；
7. 第一阶段页面时间窗限制为 24 小时，不声称已支持 Trend。

### 13.3 PostgreSQL 与指标范围对齐

本次改造必须同步交付版本化 Zabbix Template 和 Metric Catalog Provider Definition：

- 为第 7.2 节列出的 Oracle/MySQL 全部第一阶段指标补齐精确 Item Key；
- 为六项 `postgresql.*` 指标补齐 PostgreSQL Template、Item、Value Type、单位和采样周期；
- PostgreSQL Host 发现必须能区分实例、集群和数据库维度，Binding Locator 固定为经验证的 Host
  技术名称，不使用展示名；
- 复制延迟和 Slot WAL 保留必须明确主库、备库、Slot 不存在等语义，空集合不能转换为零；
- 客户 Template 覆盖使用同一 `zabbix_item_keys` 合同，并在保存 Mapping 时验证 Item 确实存在；
- 提供可重复导入的参考Template、版本说明和升级规则，不要求管理员手工创建零散Item；导入动作由
  客户Zabbix管理员审核后在外部平台执行。

参考Template、Metric Mapping清单和校验说明放在 `configuration/zabbix/`，不放入
`scripts/deployment/aiops_observability/`，也不由部署脚本自动导入。

### 13.4 诊断接入对齐

两类监控必须同时通过以下诊断链路，不能只验收实时图表：

```text
Prometheus → Alertmanager Webhook ┐
                                  ├→ SignalEvent → Target Binding → Situation
Zabbix → Zabbix Webhook ──────────┘                 → Diagnostic Run
                                                     → Metrics/Event Evidence
                                                     → Finding/Report/Recovery
```

统一要求：

1. 入站事件通过 HMAC、时间戳和防重放校验；不把 Provider Token 交给浏览器；
2. 故障和恢复事件使用稳定 Provider Event ID 归一化并保证幂等；
3. 严重度、状态、`event_class`、可选 `metric_code` 和诊断时间窗进入统一 `SignalEvent`；
4. `source_locator_key` 必须精确命中 ACTIVE Target Source Binding，不能用 Host/Label 名称模糊匹配；
5. 相同规范事件类别进入同一 Situation 关联、最小严重度、冷却和自动诊断策略；
6. Diagnostic Run 使用触发前后相同窗口主动补采该来源的标准 Metric Code，并保留来源、单位、
   覆盖率、Gap 和不可变 Evidence 引用；
7. PostgreSQL 告警与 Oracle/MySQL 使用同一触发、取证和报告流程，不因数据库类型退回只展示告警；
8. 数据库不可直连时，两类监控都可以单独提供监控证据，但必须降低结论等级并记录证据缺口；
9. 恢复事件关闭或更新 Situation 后，使用同一指标口径执行恢复验证并生成 Before/After 对比。

Prometheus告警入站继续由 Alertmanager Adapter 承担，Zabbix由自身Webhook承担；不得为了表面
“实现一致”而复制两个 Adapter。对齐的是标准事件、Target、安全、诊断和验收合同。

### 13.5 真实环境对齐验收

使用 Oracle、MySQL、PostgreSQL 各至少一个受控实例，对 Prometheus 与 Zabbix 分别执行：

- 连接验证、实例发现、映射、24 小时时序和多实例图表；
- 每个 `ALIGNED` Metric Code 的单位、时间戳、趋势方向和允许误差对比；
- 可控故障告警、重复投递、严重度门槛、自动诊断、恢复事件和恢复验证；
- 来源不可达、认证失败、缺失 Item/Series、部分指标失败和响应预算超限；
- PostgreSQL 可用性、连接、事务、复制延迟、Slot WAL 保留和容量完整闭环。

只有两类来源都通过同一验收清单，才可以在产品中显示“Prometheus/Zabbix能力已对齐”。

## 14. 权限与安全

- 页面访问使用现有 `aiops:use`；
- Target、Source 和 Binding 均按当前 Domain 查询；
- Main API 继续认证 Portal Bearer Token，再向内部服务签发短期 AuthContext JWT；
- 只有 AIOps 配置管理员可以执行实例发现、Mapping、Item Key 覆盖、Alertmanager 关联和 Webhook
  测试；
- 浏览器不能指定 Domain、Actor、Endpoint、Credential、PromQL 或 Zabbix JSON-RPC 方法；
- Diagnostic Source Credential 仅在服务端调用期解密并保留于内存；
- 日志只记录 Source ID、Binding ID、Target ID、稳定错误码、耗时和响应大小；
- 不记录用户名、密码、Token、完整 DSN、Authorization Header 或原始监控响应；
- Profile、Binding 和 Provider Template 版本必须进入结果 Provenance，便于复现和审计；
- 用户从实时监控进入 Agent 时只传递 Target、时间窗、标准 Metric Code 与 Evidence 引用，不复制
  未经预算控制的完整时序到对话正文。

## 15. 可观测性

新增服务指标建议：

- `kbot_aiops_monitoring_view_requests_total`；
- `kbot_aiops_monitoring_view_duration_seconds`；
- `kbot_aiops_monitoring_provider_requests_total{source_type,result}`；
- `kbot_aiops_monitoring_provider_duration_seconds{source_type}`；
- `kbot_aiops_monitoring_cache_hits_total`；
- `kbot_aiops_monitoring_gaps_total{source_type,code}`；
- `kbot_aiops_monitoring_response_bytes`。

指标 Label 不包含 Domain ID、Target ID、Source ID 或用户 ID，避免高基数和租户信息泄露。请求级
关联通过日志中的 Request ID 与 Trace ID 完成。

## 16. 实施拆分

### 16.1 第一阶段 A：统一实时监控

- 将“监控接入”明确为实时监控的前置流程，补齐来源状态、实例发现与批量映射；
- 新增按 Source 投影的授权实例目录，不新增实例业务表；
- 新增 Monitoring Profile 合同与版本化资源；
- 新增多实例 Monitoring View 应用服务和单实例下钻；
- 复用 Monitoring Snapshot、Metric Catalog、Credential Resolver 和 Adapter；
- 新增内部 AIOps 路由、Main API BFF、Platform Client 和公开合同；
- 引入本地固定版本 Apache ECharts，并保留版本与许可证；
- 新增 `monitoring.html`、多实例图表、可访问数据表、页面脚本和现有主题内样式；
- Dashboard、Target、Situation 和 Chat 增加上下文深链；
- Prometheus 完成现有能力收口，并形成统一 Profile、事件和诊断验收基准；
- 为 Zabbix 补齐全部第一阶段 Metric Code、PostgreSQL Template、实例发现、查询和 Gap 合同；
- 保持Zabbix外部接入边界，不修改 `aiops-stack.ini`、Compose、镜像清单或部署脚本；
- 统一投影 `monitoring_readiness`、`diagnostic_readiness` 与逐项缺口；
- Prometheus/Alertmanager 与 Zabbix 分别完成故障、恢复、自动诊断和恢复验证闭环；
- 两类 Provider 同批通过对齐准入后发布，不交付长期停留在“部分对齐”的产品状态；
- OEM 明确排除。

### 16.2 第一阶段 B：Grafana 高级下钻

- 固定 UID 与 Profile 映射；
- 单一 INI 增加集成模式与公开入口；
- 建立受控认证网关或短期 Embed Session；
- 完成单 Domain 专用或数据源租户隔离验收；
- 不满足内嵌条件时只提供受控新窗口链接。

### 16.3 后续阶段

- Zabbix Trend 长周期查询；
- 跨监控源的多实例对比；
- 管理员审核后的自定义 Dashboard Profile；
- OEM 真实环境契约、认证、指标、Incident 与 Target 映射设计；
- 其他监控平台 Adapter。

## 17. 测试计划

### 17.1 单元测试

- Profile 解析、数据库类型过滤和稳定排序；
- 监控源可用门禁、Source 深链解析和单实例来源对比；
- 一个 Source 投影多个 ACTIVE Binding，禁用 Source 或 Binding 不进入实例目录；
- 实例发现候选验证、重复 Locator 拒绝、Target 类型校验和映射事务；
- 时间窗、实例数、点数、Series、字节预算和部分成功；
- Metric Observation 到 Panel 的单位、摘要和数据质量映射；
- 多实例 Series 带实例标识、同名实例图例消歧、跨数据库类型 Panel 拆分；
- Prometheus 查询覆盖与 Target Scope；
- Metric Catalog 中每个第一阶段 Prometheus Definition 都存在兼容的 Zabbix Definition；
- Zabbix 健康检查、Value Type、Item 映射、History 和 PostgreSQL 六项指标；
- 部署配置解析和Compose服务清单拒绝新增 `[zabbix]` 模块或Zabbix镜像；
- Prometheus/Alertmanager 与 Zabbix 的故障、恢复、严重度、事件类别和幂等归一化；
- 两类来源生成相同 Evidence/Gap/Provenance 合同并进入相同诊断策略；
- 指标连接失败不会阻断已启用来源的合法 Webhook，诊断以结构化 Gap 继续；
- OEM 不进入第一阶段 Profile 与选源；
- 缓存 Key 包含 Domain 与版本，不缓存凭据。

### 17.2 合同测试

- Main API 与 AIOps Internal OpenAPI；
- 公共 DTO 不暴露 Endpoint、凭据、查询文本或 Provider 原始 Payload；
- 多实例请求不接受 `source_locator_key`、PromQL、Zabbix Host 或任意 Provider 参数；
- 无来源、来源禁用、连接失败、无实例映射和无采样返回不同状态；
- Prometheus 与 Zabbix 返回相同的 Oracle/MySQL/PostgreSQL Profile 和 Metric Code 集合；
- 监控接入返回 `monitoring_readiness`、`diagnostic_readiness` 与逐项缺口；
- Zabbix Endpoint和凭据只出现在受控的AIOps配置合同与Credential引用中，不进入部署配置；
- 图表依赖从本地加载且有许可证，不引用外部 CDN；
- 静态页面导航、权限、中文文案和脚本语法；
- 不新增 V1 兼容路由或无版本业务路由。

### 17.3 集成与 Smoke

- Prometheus 真实 `query_range`；
- Zabbix 真实 `host.get`、`problem.get`、`item.get`、`history.get`；
- 一个 Prometheus Source 映射多个数据库实例并在同一 Panel 返回多条实例曲线；
- 一个 Zabbix Source 映射多个 Oracle/MySQL/PostgreSQL Host，并在同一 Panel 返回多条实例曲线；
- 未映射或其他 Domain 实例不能通过篡改 `instance_ids` 被查询；
- 同一 Target 绑定 Prometheus 与 Zabbix 的手工对比；
- Oracle、MySQL、PostgreSQL 的同指标 Prometheus/Zabbix 结果对齐；
- Alertmanager 与 Zabbix Webhook 分别触发故障、重复投递、自动诊断、恢复和恢复验证；
- 指标端点不可达时，两种 Webhook 仍被接收并形成 `PARTIAL/INCONCLUSIVE` 诊断；
- Source 不可达、认证失败、无数据和部分指标失败；
- Domain A 用户不能读取 Domain B Target；
- Grafana 白名单、Target 锁定、权限拒绝、Session 过期和退出登录；
- 从 Dashboard、Situation、Target 详情到实时监控，再进入 Chat 的完整深链。

## 18. 第一阶段验收标准

1. AIOps 导航存在“实时监控”，现有 Dashboard 职责和性能不被改变；
2. 没有已配置并启用的 Prometheus/Zabbix 来源时不发起指标查询，并展示监控接入引导；
3. 来源禁用、连接失败、无实例映射和指标无采样显示为四种不同状态；
4. 一个监控源可以映射并列出多个当前 Domain 内有权访问的数据库实例；
5. 用户可以一次选择最多 12 个同源实例，图表按实例展示独立曲线和可辨识图例；
6. 用户可以从状态矩阵或曲线下钻到单实例，并保留来源、时间窗和指标上下文；
7. Prometheus 可以展示 Oracle、MySQL、PostgreSQL 的第一阶段标准实时指标；
8. Zabbix 可以展示与 Prometheus 相同的 Oracle、MySQL、PostgreSQL Profile 和 Metric Code；
9. OEM 不出现在页面选项、API Profile、选源或验收结果中；
10. 页面支持 15 分钟、1 小时、6 小时和 24 小时时间窗；
11. 每个 Panel 展示来源、实例、采样时间、覆盖率、单位和 Gap；
12. 单实例失败不影响其他实例展示，无数据不会显示为 `0`、健康或成功；
13. 实时监控只查询用户明确选择的一个来源及其授权实例，不在 Dashboard 首屏批量拉取曲线；
14. 篡改 `source_id` 或 `instance_ids` 不能越权读取其他 Domain、未映射或已停用实例；
15. 浏览器网络请求中没有 Prometheus/Zabbix 凭据、Locator、任意查询接口或内部 AIOps 路由；
16. 用户可以把实例、来源、时间窗和标准指标上下文带入智能运维；
17. Grafana 未满足安全前置条件时不会以内嵌或匿名方式暴露；
18. Prometheus 和 Zabbix 均可查询活动事件，并通过各自告警入站路径触发同一自动诊断流程；
19. 两类来源对故障、恢复、重复事件、严重度、Target 映射、证据补采和恢复验证具有相同语义；
20. PostgreSQL 在 Zabbix 下通过时序图表、事件查询、告警触发诊断和恢复验证完整验收；
21. Zabbix 专用健康检查和 Adapter 行为具有单元测试及真实环境 Smoke 证据；
22. 只有 `monitoring_readiness=READY` 且 `diagnostic_readiness=READY` 的实例可以标记“完整接入”；
23. `aiops-stack.ini`、Compose、镜像清单和生成脚本不包含Zabbix模块、容器、Volume或部署凭据；
24. Zabbix只通过AIOps App配置外部Endpoint、Managed Credential、实例映射和Webhook；
25. KBot不自动导入Template、创建Host或修改客户外部Zabbix配置；
26. 公开与内部 OpenAPI、静态页面合同、AIOps 单元测试和边界检查全部通过；
27. 实施报告分别说明 Prometheus、Zabbix、Grafana 和 OEM 的真实状态，不把配置成功当成运行验收。

## 19. 需要保持的产品表述

第一阶段可以表述为：

> 管理员配置并启用 Prometheus 或 Zabbix 监控源、完成实例映射和诊断接入后，用户可以在 AIOps
> 统一实时监控页面以相同 Profile 查看 Oracle、MySQL、PostgreSQL 多实例标准指标。两类来源都可
> 查询活动事件、触发自动诊断、补采时序证据并处理恢复闭环；Prometheus 环境还可以进入受控
> Grafana 专业看板。

不得表述为：

- “已统一支持 Prometheus、Zabbix、OEM 全部实时指标”；
- “任意 Grafana 看板都可以安全嵌入”；
- “Zabbix 已支持长期 Trend”；
- “配置诊断源即代表真实链路已验收”；
- “只要实时图表有数据就代表诊断接入完成”；
- “KBot会安装、托管或升级Zabbix”；
- “未映射的监控平台实例也会自动展示”；
- “OEM Incident Webhook、Topology 或自动执行已交付”。

## 20. KBot4 与 Ammolite Cube 同步实施规则

本设计同时适用于 `/home/chris/kbot4` 与 `/home/chris/ammolite_cube`。后续实现必须以一个功能变更集
同步推进，不允许一个项目形成新的AIOps业务分支或长期落后于另一个项目。

### 20.1 必须保持一致的服务层

以下AIOps服务层代码、行为和版本化资源在两个项目中必须保持同构；除导入路径等纯机械差异外，
业务逻辑、合同字段、状态机、错误码和测试场景应一致：

- Monitoring Profile、Monitoring View、Source/Instance目录及Readiness合同；
- Prometheus、Alertmanager、Zabbix Diagnostic Source Adapter及统一端口；
- Metric Catalog、Zabbix Provider Definition、Gap、Provenance和查询预算；
- 实例发现、Target Source Binding、来源选择和多实例查询应用服务；
- SignalEvent归一化、Situation关联、自动诊断、证据补采、恢复验证和报告投影；
- Main API公共合同、AIOps内部合同、Platform Client方法和OpenAPI语义；
- Provider合同测试、诊断闭环测试、安全边界测试和受控Smoke场景。

不允许只在某一项目中修复Provider语义、增加Metric Code、修改事件映射或调整诊断策略。共享服务层
变更必须在同一实施批次进入两个项目，并分别通过项目自己的测试门禁。

### 20.2 允许不同的实现边界

两个项目只允许以下项目形态差异，不得渗入共享业务逻辑：

| 边界 | KBot4 | Ammolite Cube | 同步要求 |
| --- | --- | --- | --- |
| Meta Database | Oracle实体、DDL、Repository和UoW | PostgreSQL规范ORM、初始化、Repository和UoW | 表达相同实体、字段、关系、事务和查询语义；只适配方言、类型和身份存储 |
| 前端 | KBot4维护的AIOps静态页面与脚本 | `apps/portal-web`中的Vue AIOps页面 | 页面结构可按各自Design System实现，但菜单、状态、筛选、图表、错误和权限语义一致 |

入口认证、Tenant/Domain解析和服务启动装配属于各项目外围适配层；进入AIOps应用服务前必须归一为
同一可信上下文，不能导致服务层出现两套业务分支。

### 20.3 同步交付门禁

每个实施批次必须同时完成：

1. 在两个仓库记录对应基准提交和变更清单；
2. 对服务层文件、符号、合同、资源和测试做差异审计；
3. 每项差异只允许分类为 `DIRECT_COPY`、`META_DATABASE_ADAPTATION` 或
   `FRONTEND_ADAPTATION`；其他差异必须消除；
4. KBot4同步更新Oracle规范结构，Ammolite同步更新PostgreSQL规范初始化，不增加兼容层、双读写或
   补丁式迁移；
5. 两边同步生成OpenAPI并运行Provider、诊断、权限和文档合同测试；
6. 分别使用真实Meta Database完成持久化验证，并使用同一组Prometheus/Zabbix受控场景完成Smoke；
7. 交付报告同时列出两个仓库的提交、测试、未执行项和在线验收状态；任一项目未完成时，整体能力
   不得标记为已交付。

文档同步本身不代表代码已经同步。后续实施必须依据本节逐批执行代码、数据库、前端和测试变更。
