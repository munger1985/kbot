# OEM 监控图表审核与实现

日期：2026-11-12。以用户最后确认的范围为准：仅 OEM，包含 Exadata 与非 Exadata；非 OEM 的图形、采集与诊断保持原状。此前未完成的非 OEM 展示策略已撤回，不纳入交付。

## CPU 与内存的统一规则

- Cell 的 `Cell_Metric/cpu_utilization`、`Cell_Metric/memory_utilization`，以及 DB Server/普通 OEM 主机的 `CPUUsageSummary/OverAllUsagePct`、`Load/memUsedPct` 均使用折线，不能以环图代替趋势。
- Exadata 在最新快照模式下保留快照摘要，复用现有 `/view` 历史接口，为当前场景、组件与分页补读过去 60 分钟真实采样；一页最多增加一个历史请求，不扩大成员或翻页范围。不改变服务端合同、目录和采集协议。
- 当前快照、历史曲线、7 天/30 天均值分别取证。资源、绑定版本、来源版本与目录摘要必须一致；曲线及基线按组件、指标、原生单位与完整维度匹配，不计算跨设备均值。
- 实线为真实采样，虚线为所选历史窗口留存采样均值，标注周期及两位小数数值；卡片保留均值偏离摘要。历史均值不是故障阈值。坐标范围包含基准值，不因基准在当前曲线范围外而消失。
- 只有一个历史采样时显示单点；空值保留断点，零值有效。历史失败或无采样时显示缺口，不能复制最新快照生成曲线。切换场景/资源、刷新或卸载时取消旧请求，切换基线后不显示旧周期基准线。
- “查看数据”是唯一证据入口，区分当前快照与历史采样，保留精确原值、维度、时间、质量和均值窗口。

## 非 Exadata OEM 总览逐项决策

现有审核目录共 16 个总览位置，其中 15 项有明确原生选择器；表空间保留原有缺口，不补造指标。

| 指标 | 图形 | 依据与边界 |
|---|---|---|
| Response/Status | 原始状态 | 不将未经审核的数字状态码画成性能曲线或推断健康 |
| Response/DatabaseStatus | 原始状态 | 保留原始文本与采集时间 |
| CPUUsageSummary/OverAllUsagePct | 折线＋均值虚线 | CPU 使用率趋势 |
| Load/memUsedPct | 折线＋均值虚线 | 内存使用率趋势 |
| Load/cpuLoad_1min | 折线＋均值虚线 | 每 CPU 运行队列，不伪装成百分比 |
| Load/swapUtil | 占用环图 | 仅单维、0–100%；未使用部分标为计算值 |
| Filesystems/pctAvailable | 挂载点水平柱图＋均值标记 | 剩余空间比例，不将其标成已用比例 |
| DiskActivitySummary/maxavserv | 折线＋均值虚线 | 原生磁盘服务时间，保留最大值口径 |
| NetworkSummary/totalNetworkThroughPutRate | 非堆叠面积趋势＋均值虚线 | 原生吞吐速率，不解释为链路占比 |
| instance_throughput/avg_active_sessions | 折线＋均值虚线 | AAS 负载趋势 |
| Database_Resource_Usage/session_usage | 占用环图 | 会话配额占用，不解释为活跃会话数量 |
| UserBlock/count | 按 SID 水平柱图＋均值标记 | 不跨 SID 求和冒充全局阻塞计数 |
| wait_sess_cls/dbtime_waitclass_pct | 等待类别水平柱图＋均值标记 | 不把不同维度/时刻的百分比求和画饼图 |
| instance_throughput/physreads_ps | 非堆叠面积趋势＋均值虚线 | 每秒速率，不推算窗口事件总数 |
| 表空间容量 | 原有缺口 | 无审核指标，不以数据库大小替代 |
| Recovery_Area/Free_Space | 占用环图 | 审核元数据为 Used Space (%)，不是名称字面含义的空闲率 |

环图只对明确占用语义、同一维度、有效 0–100% 使用率绘制；多维或越界改水平柱图，不截断原值。柱图超过 6 个维度可滚动，基线用短竖线标记；数值对比不代表健康。OEM 总览的折线保留真实空点，不用旧的非空采样冒充当前值。

Exadata 其余 286 个指标继续遵循 [Exadata 逐项图表审核](aiops-exadata-oem-chart-audit.md)，本次不重写已确认的状态字典、诊断、模拟器、授权或数据库。颜色沿用监控卡片现有六色序列与 surface/line/tooltip 令牌，不用序列颜色冒充健康等级。

## 验证入口

两仓分别运行 `node tests/contract/aiops_oem_charts.mjs`、`node tests/contract/aiops_exadata_cards.mjs`；同步检查为 `node tests/contract/sync_aiops_oem_presentation.mjs`。固定替身覆盖生产组件、真实 ECharts 画布、基准线、两位小数、空/零值、维度隔离、周期切换、原始数据入口和响应式布局；它们不等于线上 OEM 采样恢复验收。Portal 另运行 `npm run build`。

本次本地两仓的 OEM/Exadata 展示合同、同步检查与 `aiops_monitoring_request_encoding.mjs` 均通过；Portal 类型检查与生产构建通过，构建仍有既有大 chunk 提示。

## 线上部署与验收

2026-11-12 已推送并部署到 kbotdev 的 KBot 与 Ammolite。主功能提交分别为 `df2b3869`、`03f232d6`；随后补齐 KBot“查看数据”的快照/历史采样明细，并同步两仓证据表格回归合同。只更新前端：KBot 静态文件、Ammolite 在线 Vite 页面直接加载新代码，不重启后端、OEM、模拟器或数据库。

- 两端真实页面均逐一通过 8 个 Exadata 场景、磁盘历史折线与 390px 窄屏验收，无页面脚本异常或监控区域横向溢出。
- Cell 与 DB Server 的 CPU/内存均为折线；7 天和 30 天均值虚线已与线上 `/baselines` 的同组件、同指标均值逐项核对，展示两位小数。取证时 DB Server CPU 过去一小时只有 1 个真实采样，按单点展示，不补造连续趋势；所选窗口不代表已有完整 7/30 天覆盖。
- 两端普通 OEM 主机页面均通过真实验收：CPU/内存/运行队列/磁盘服务时间折线，Swap 环图，文件系统余量水平柱图，网络吞吐面积趋势；“查看数据”可打开。原有主机可用性无采样仍显示缺口，没有以其他设备或旧值替代。
- 备份与部署证据保留在服务器 `/home/ubuntu/aiops-oem-ui-20261112-HUzqFe`。部署后 8 个服务/前端就绪地址均返回 HTTP 200；51 个既有服务进程及配置摘要未变化，无服务重启。

额外整页对照 `AIOPS_PAGES=monitoring node tests/contract/aiops_page_alignment.mjs` 未通过：其旧替身将监控业务系统目录也匹配到普通业务系统的 `{items: ...}` 响应，未匹配当前监控目录的数组合同，产生 `state.systems.flatMap is not a function`，在图表之前已停止。此次未修改该替身，也未修改非 OEM 生产导航代码；不得将此项记录为通过。

## Exadata 卡片留白修正

针对 2026-11-12 16:59 截图，移除整机旧资源样式将“查看数据”左右内边距清零的覆盖。标题、数值、时间、图表和说明统一左右 16px，底部入口上下各 12px；概览嵌套卡片与设备边框间距为 16px，左右卡片等高。间距复用组件级令牌，限定在 `.exadata-oem` 内；不修改指标值、图形选择、采集、诊断或其他监控页面。KBot 静态 CSS 更新缓存版本。

两仓合同增加桌面/窄屏下的实际边距与证据入口检查，覆盖设备详情和整机嵌套卡片。本节为本轮样式修改，尚未部署；上方线上验收记录属于此前已部署的图表功能。
