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

最终两仓扩大后端回归及完整OpenAPI快照合同各754项通过，另13个subtests通过；其中采集器8项通过。两端普通OEM24面板、Exadata345项展示决策及响应式页面、观测设置切换、服务状态/属性/趋势/数据入口/过期处理浏览器合同通过。Ammolite `npm run build`通过；两仓架构边界通过。新增服务9组发布与持久化贯通使用真实SQLite UoW，不冒充线上Oracle/PostgreSQL并发验收。

### 线上验收发现并修复的合同差异

- GUID子目标：真实归档定义保留ComponentId/PortId，但对应采样的KEY_VALUE为空；原厂SI插件以键列的COMPONENT_TYPE声明独立组件，GUID REST元数据因此可省略已提升为目标身份的键。新增 `guid_transport.v1.json`，仅正向审核的7个目标类型/指标组允许这种表示；目录仍保留归档原键。latest、history、baseline均验证同一GUID和精确类型，不补造ComponentId=GUID，不放宽其他组的完整维度校验，不合并设备。规则同时纳入目录证据Hash。
- 普通OEM等待类别：原厂一天的JSON实测612348字节，固定半预算会在总预算尚有余量时错误截断。现在按身份、元数据和CLI原始输出的实际字节扣减；当前趋势优先，基线只使用剩余预算。总冻结上限不增加，完整基线超预算仅返回基线不可用，保留当前趋势，不能发布部分均值。
- 普通OEM CLI超预算规范为公开支持的SOURCE_RESULT_TRUNCATED，不再被Main API吞成通用503；保留Exadata原有CLI错误合同。原厂帮助仅有last_n_days，最多7天，没有小时或维度过滤参数，不能宣称已支持30天带维度历史。
- PDU新增6项物理量的均值比较审核：相位有功功率、电流、视在功率、电压，模块不平衡电流、模块总输入有功功率。累计能耗、相位/模块编号、阈值不参与均值；每个模块/相位GUID分别计算，不做跨设备均值。
- 应用层基线及其缓存同步采用GUID维度合同。允许审核过的键提升表示，但快照、历史和缓存之间不得混用不同维度表示；缓存也要复核当前维度，而非只比较单位和绑定版本。

### 真实 OEM 模拟环境取证

- 2026-11-12，模拟器原位更新，保留bind mount inode；未重启。备份 `/opt/aiops-oem-lab/state/exadata-sim-update-3t47nd40`，新SHA256 `3aeea0d0dc6dfc547bd707f67a8984efa85f88094ba7d303e4947c56c9816833`。
- 已定向触发 Cell 性能、IORM、容量集合并成功上传OMS。原厂GUID REST实际返回 Top_CPU_Activity/Activity=`12`（13:05:02Z）、Sparse虚拟容量=`4096`（13:04:24Z）；不是只验证模拟器自身输出。
- PDU相位GUID REST实读电压=`230.0`、电流=`2.7`（13:12:14Z）。两端AIOps公开view接口已实读相同数值及4个真实历史时刻；页面实际显示电压、电流、图表与数据入口，1440/390宽度无横向溢出、无pageerror。子组件latest/history/baseline均按GUID REST，不送入原厂不支持子组件target name的EMCLI。
- 单个Phase目标有8项相位采样；5项模块指标在该Phase目标返回空采样，是原厂按子目标身份分开的结果，不补成0、不用Module目标数值冒充Phase数值。Module0和Phase1/2/3已分别经管理员确认导入，两端OEM来源均恢复为ENABLED/CONNECTED。
- 普通OEM两端通过公开metric-samples接口实读：等待类别65点、13个完整class维度且未截断；Recovery_Area为3至4点；UserBlock为真实0条，不制造阻塞值。样本数随窗口滚动，不作为固定验收常量。
- 两端开启7天基线复验：等待类别仍返回65个当前窗口采样点，基线因完整结果超出剩余预算而明确不可用；Recovery_Area返回1条完整维度基线；UserBlock仍为空采样。未用不完整均值掩盖预算限制，未因基线失败抹掉当前趋势。
- 两端PDU Phase1电压基线实读通过：7天窗口中26条留存采样，均值230.0V，当前230.0V，差值0.0V、偏离0%，未截断。样本最早时间及覆盖率由OEM实际留存决定，不把26点说成完整覆盖7天。
- XRM协议已实现，但原厂XA插件的ValidIf要求23Ver1/23Ver2且networkCategory=roce；现有模拟Cell上报IB。刷新动态属性后仍被原厂拒绝为invalid Metric；不修改原厂插件或把IB资产强改为RoCE以伪造支持。需另建一致的RoCE模拟目标或真实目标才能验收XRM采样。

### 发布与现场验收

代码已提交、推送并发布到kbotdev的两套服务：

| 阶段 | KBot | Ammolite | 定向重启 |
| --- | --- | --- | --- |
| 能力补齐 | c012235f | d9e8f85a | 两端Main API、AIOps API/Worker，共6进程 |
| GUID维度与历史预算 | d15d0014 | 045d612b | 两端AIOps API/Worker，共4进程 |
| 当前趋势优先与PDU基线 | 7ed104db | 39cc6996 | 两端AIOps API/Worker，共4进程 |
| 基线应用与缓存维度校验 | 3bee4b96 | 2e7fc10d | 两端AIOps API/Worker，共4进程 |

发布证据目录依次为 `/tmp/aiops-oem-closure-release-tJJlrkEA`、`/tmp/aiops-oem-guid-release-LyHNOhXi`、`/tmp/aiops-oem-baseline-release-4xs5n8D2`、`/tmp/aiops-oem-dimensions-release-T7eHfIDP`。第一阶段45个、后续各阶段47个无关进程身份不变；配置Hash不变，8个页面/服务就绪入口全200。最终代码发布核验时间为2026-11-12T13:32:37Z。仅重启实际依赖变更的服务，不以全系统重启代替定向验收。

### 尚需真实环境或管理员决策的部分

1. ASM/RAC现场验收：测试环境没有真实ASM/RAC。33项是共用指标语义和来源映射能力，不代表当前环境已生成33项采样。可选采集器只覆盖已实现的固定查询与Listener协议；未覆盖的项目保持未映射，详见其README。不得创建假集群资源冒充真实验收。
2. OEM原生列缺失：由OEM管理员批准Metric Extension或现有Exporter/Zabbix采集器补采，再配置明确映射。扩展必须完成实际采集、OMS上传和REST/EMCLI实读；此轮没有擅自安装扩展、授予SYSASM/SYSDBA或绕过OEM直接读其他来源。
3. XRM：需另外部署版本与RoCE网络属性一致的模拟目标或真实设备；现有IB目标不满足原厂ValidIf，不能靠补JSON、改插件或改设备标签伪称采样成功。
4. 历史能力：原厂EMCLI最长7天及响应预算是明确能力边界；超预算或30天不输出虚假的完整均值。GUID REST仍可独立支持其实际留存，必须逐项读取验收。
5. 独立环境异常：当前线上development logs仍有KBot工作负载SQL快照落库ORA-01653，目标为KBOT_OPS_WORKLOAD_STATEMENT、共享Oracle表空间APEX_9500769379212461。与OEM读取无关；未清理数据、扩容共享库或停用无关任务，需要另行批准处理。
