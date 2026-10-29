# 数据库原生身份见证协议

这些是 sql_exporter 的三库 Collector 配置，不含连接凭据，不由 aiops-stack 自动启用。
必须由运维审核实际采集器版本、部署配置、权限和连接方式后，纳入现有采集器的部署流程；
本次交付不部署采集器、不授予数据库权限，也不新增第二个部署入口或可编辑凭据文件。

## 接入与授权

1. 按数据库类型选择一个 Collector，并通过采集器现有的受保护连接配置执行；
   不能用另一个 DSN 的探针证明既有 Exporter 的数据来自同一数据库。
2. 采集器暴露 aiops_database_identity_info。原生身份必须是标签中的原始字符串，
   大整数标识不能转换成 Prometheus 浮点样本值。
3. Prometheus 既有绑定须以精确 target_key 定位，并保留 job、instance；
   联邦/多集群部署还需保留 cluster、prometheus、replica 等作用域标签，
   不应把彼此不同的采集端点重写成相同标签。
4. 在已授权 Target 的 Source Binding 指标范围中显式加入 db.identity.info。
   查询仍使用 metric.query_range；不自动扩大绑定指标范围、来源或数据库权限。
   此指标不在默认基线中，未配置见证的既有监控继续保持身份未知。
5. 普通诊断可随该绑定采集；有明确指标/Profile限制的调查，须在既有授权范围内
   显式请求该指标，不绕过原有 Profile 和预算。

canonical 查询为：

```promql
timestamp(aiops_database_identity_info{target_key="绑定的精确Locator",identity_version="1"})
```

结果值是原始 Prometheus 抓取时刻，不是范围查询的求值时刻。生产投影拒绝未来时间、
求值时点前超过120秒的陈旧抓取及无效数值，比较直连与监控时同样要求120秒邻近。
Prometheus抓取新鲜不代表Exporter内部没有缓存，也不是数据库原生执行时刻证明。
原始响应摘要、模板版本、source/binding/config版本、序列路径和抓取时点均保留。

## 三库边界

- PostgreSQL：system_identifier、database_oid及启动时间。同族主备可能共享系统标识；
  即使启动时间相同，也只给出数据库族的来源自报匹配，不宣称同一运行节点。
  pg_control_system不可读时由实际取证形成未知，不自动授予权限。
- MySQL：server_uuid支持来源自报节点匹配，但没有可靠启动代际；不能据此跨重启做差。
  标准mysqld_exporter通常不提供字符串UUID，版本号或server_id不能替代。
- Oracle：DBID、INSTANCE_NUMBER、CON_ID及原始启动时间。
  RAC/Data Guard可能共享DBID；STARTUP_TIME没有时区，不擅自补UTC，也不跨源比较该未知时钟。
  根容器与PDB范围不同不能无条件合并。

同一Collector作用域在一次原始结果中出现不同数据库族标识时，关系保持未知；
明确的近时原生身份字段冲突单独展示，不能被相同Target或地址掩盖。
缺失/截断/不合法见证不抹掉其他有效SQL或监控证据。

## 能证明什么

PHYSICAL_IDENTITY_GRAPH.v1保留原始来源节点及有界关系，最多64节点/128关系；
超预算明确标记，不冒充完整拓扑。它区分来源自报数据库族匹配、MySQL节点匹配、
冲突和未知，不把匹配升级成物理认证。

身份信息序列只见证它自身的查询连接。相同job/instance、target_key或单独部署的身份采集器，
均不足以把其他Exporter的全部历史指标归属到该节点。实际指标归属还需真实Exporter
配置及同连接/同代际证据；本批没有把metric_series_assignment_verified置为真。
窗口连续性、SQL因果和同源指标独立互证同样保持未证明。

诊断回答和SQL Evidence Bundle使用同一图；调查记录冻结见证和图；
报告/案例发布与检索通过共享来源校验重读OBSERVATION_SET和DBA_TOOL_RESULT，
重新计算图。删除见证、图或来源标记不能绕过原始来源复核。

## 可选的同查询指标身份（2026-10-29）

三库新增 `postgresql-metrics.collector.yml`、`mysql-metrics.collector.yml`、
`oracle-metrics.collector.yml`，每条SQL同时返回原生身份字符串和active_sessions。
沿上面的同一受保护采集器流程接入；不创建新凭据文件，不自动加载或授予权限。
在既有Binding显式授权 `db.native.active_sessions` 后才能查询，默认基线不启用。

Prometheus必须保留metric_identity_version="1"、target_key及job/instance/集群作用域。
协议DATABASE_METRIC_ROW.v1同时保留数值及timestamp(同一指标)抓取时点与响应Hash；
陈旧超过120秒、未来、重复序列/求值时点不升级身份关联。
PG只统计当前库活跃后端；MySQL统计server前台非Sleep线程；
Oracle统计当前实例、当前容器活跃USER会话，三者都排除当前采集会话。
这三种活跃定义不是完全等价的数据库负载，不据此跨引擎直接排名。

此资产只把本条原生数值与来源自报身份放在同一查询；
不能认证其他Exporter指标、运行节点、窗口连续性或SQL因果。
Prometheus抓取新鲜也不排除Exporter缓存，两个HTTP响应Hash不构成数据库原子时钟证明。
部署者须实际审核sql_exporter版本、列类型/大小写、原生目录权限及真实抓取；本批未部署验收。
