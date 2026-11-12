# OEM 审计缺口补齐实施记录

依据 `aiops-oem-audit-remediation-20261112.md` 的 F01/F05/G01/G02/G03；两平台同步。普通 Prometheus 数据库、主机的监控与诊断路径保持原样。ASM/RAC/Listener 是独立资源能力，可绑定 OEM、Prometheus 或 Zabbix，不隐式跨源补数。

## 实施合同

1. 普通 OEM 另设 `native_metric_transport`、`native_emcli_timezone`、`native_emcli_certificate_sha256`。默认 REST，只有显式选 EMCLI 的有维度数值历史采用原厂固定命令；元数据、目标 GUID/type、完整维度、单位与半开窗口仍必须核验。7天之外明确不支持，不冒充月基线。Exadata 通道不变。
2. 从真实定义补充42项 Cell 与13项 PDU模块/相位指标。状态、属性、窗口统计、累计能耗分别处理；4项仅空值的 Cell 指标继续显示无采样。目录接入、组件身份接入和原厂实际产生采样分别验收。
3. 子组件归属优先使用原厂可证明的目录。若现场插件没有可用公开关系接口，允许管理员显式导入组件清单，保存父子关系和来源证据，并逐个在线核验 GUID/type/name。导入确认不得伪称自动发现；必须经管理员确认，后续每次查询仍重新核验，不按名称或同主机扩大成员。
4. ASM、RAC、Listener 使用独立资源类型和严格来源定位；指标采用共用语义定义、来源原生映射，显式区分可达、进程、服务注册、服务状态、性能及容量。未知指标不补零；未映射不能执行；累计量不直接平均或换算速率。Agent只在当轮授权资源内分析，有证据才给结论。
5. 配置、表单、资源目录、采样、图形、Agent结构化路由与分析共同贯通，不把新增JSON或问题词表视为能力交付。产品语义两仓一致，平台持久化差异保留。

## 验证状态

### 已实现并通过本地验证

- F01：组件清单从发现、候选签名、保存前复核、持久化到每次查询复核贯通。身份由 OEM GUID/type/name 验证，父子归属明确记录为管理员确认。最多64项导入、128项总成员，不宣称无限自动发现；53项旧子组件定义因此有显式可执行入口。
- F05：普通 OEM 带维度数值历史有独立显式 EMCLI 配置、证书固定与私有会话；不受 Exadata 开关影响。当前原厂通道最多7天，30天明确不支持。错误分类仍区分原厂5xx与来源连接失败。
- G01/G02：补55项真实定义，目录合计347项、场景可执行345项；父目标两个无采样的 ComponentState 不恢复。38项状态定义经精确原值字典解释；属性、累计量及窗口统计不计算均值、不伪造性能曲线。
- G03：ASM16、RAC9、Listener8项共用语义；独立资源、OEM/Prometheus/Zabbix精确映射、发布与绑定、实时读取、状态/属性行、趋势、7/30天基线及授权Agent只读分析共用同一来源身份。未映射项目明确不可查询，不能补成0。来源是否支持目标历史窗口、采样量和预算仍须分别验证。
- 高频基线不再依赖千点展示上限：Prometheus按完整标签读取受控 sum_over_time/count_over_time；OEM分页汇总；Zabbix独立服务基线最多16页并核验重叠边界。均统一半开窗口，截断或超预算不发布部分均值。Zabbix不隐式改用趋势表；来源不保证跨请求一致快照，留存覆盖率未知。
- 可选只读采集工件见 `scripts/deployment/aiops_oracle_services/README.md`，包含固定Oracle查询、原厂Listener协议解释、现采OpenMetrics/JSON与身份复核；未自动安装、未扩大数据库权限。不是33项全指标的通用采集器。

最终两仓扩大后端回归及完整OpenAPI快照合同各693项通过，另13个subtests通过；其中采集器8项通过。两端普通OEM24面板、Exadata345项展示决策及响应式页面、观测设置切换、服务状态/属性/趋势/数据入口/过期处理浏览器合同通过。Ammolite `npm run build`通过；两仓架构边界通过。新增服务9组发布与持久化贯通使用真实SQLite UoW，不冒充线上Oracle/PostgreSQL并发验收。

### 真实 OEM 模拟环境取证

- 2026-11-12，模拟器原位更新，保留bind mount inode；未重启。备份 `/opt/aiops-oem-lab/state/exadata-sim-update-3t47nd40`，新SHA256 `3aeea0d0dc6dfc547bd707f67a8984efa85f88094ba7d303e4947c56c9816833`。
- 已定向触发 Cell 性能、IORM、容量集合并成功上传OMS。原厂GUID REST实际返回 Top_CPU_Activity/Activity=`12`（13:05:02Z）、Sparse虚拟容量=`4096`（13:04:24Z）；不是只验证模拟器自身输出。
- PDU相位GUID REST实读电压=`230.0`、电流=`2.7`（12:57:14Z）。子组件latest/history/baseline均按GUID REST，不送入原厂不支持子组件target name的EMCLI。原厂GUID历史接口若仍失败，保留来源错误，不用latest冒充历史。
- XRM协议已实现，但原厂XA插件的ValidIf要求23Ver1/23Ver2且networkCategory=roce；现有模拟Cell上报IB。刷新动态属性后仍被原厂拒绝为invalid Metric；不修改原厂插件或把IB资产强改为RoCE以伪造支持。需另建一致的RoCE模拟目标或真实目标才能验收XRM采样。

### 发布与现场验收

以下事项必须以实际发布记录补充，不能由本地测试代替：两端提交/推送与6个相关进程定向重启；通过公开配置确认模拟PDU清单；显式启用普通OEM EMCLI后实读3项历史；真实ASM/RAC/Listener的现场采集、页面和Agent问答验收。当前测试环境没有真实ASM/RAC，未创建假集群资源冒充验收。OEM缺少原生指标时，通过管理员批准的Metric Extension或现有Exporter/Zabbix采集器补采，再显式映射；不写OEM内部表、不绕过所选来源补数。
