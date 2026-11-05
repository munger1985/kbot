# OEM 原生指标监控

日期：2026-11-05。Ammolite 与 KBot 使用相同目录、采样、授权和质量边界。

## 产品行为

监控顺序保持“业务系统 → 资源 → 实际绑定的监控源”。选择 OEM 后直接使用该资源的
原生指标目录，不把 OEM 裁剪成 Prometheus 的五张主机图或八张数据库图，也不要求用户填写
标准指标映射 JSON。数据库和主机使用各自已核验的 Target GUID，不借用承载主机或 CDB 身份。

目录读取实际插件的全部指标组，支持搜索、每页 100 项显示和逐指标按需采样。
展示原始名称、类型、单位、维度、时间、精确十进制值及截断状态。目录可见不证明该指标
已启用采集；不批量预取所有历史时序，不把“完整目录”解释为“全部指标有完整采样”。
OEM 历史时序接口只采样 NUMBER；NUMBER 和 STRING 都可读取官方 `latestData` 最近采集快照。
页面提供“历史时序 / 最近采集快照”切换，文本指标默认快照。快照不声明历史时间窗，保留实际
采集时间与空值；即使快照早于所选历史窗口也不改写时间或冒充新采样。
无值不补零，不将未知状态报告为宕机或健康。

默认选项由 `resources/monitoring/oem_native.v1.json` 维护；只有真实元数据的名称、单位和类型
均匹配才优先选择。它只是原生面板默认值，不是跨 Provider 的语义等价认证。
其他数值指标不受默认选项限制。标准指标仍需审核精确的插件语义、单位、维度和版本，
本次没有为缺少依据的标准总览指标生成自动映射。

## 查询合同与安全

- 公开入口：`GET …/monitoring/sources/{source_id}/resources/{instance_id}/metrics`。
  接受受限 `search`、`offset`、`limit`，offset 仅对服务端发现结果分页，不冒充 OEM 原生分页。
- 采样入口：`POST …/monitoring/sources/{source_id}/resources/{instance_id}/metric-samples`，
  请求为 `metric_name`、`window` 与 `mode`（HISTORY/LATEST）；不接受原始 GUID、URL、查询语句或来源凭据。
- Main API 经现有客户端转交内部监控应用。采集前后均核验当前租户/业务空间、已发布系统、
  活跃资源、绑定及来源版本。缓存仅保存 30 秒目录，命中缓存仍重新授权；采集中撤销绑定
  或改变配置则丢弃结果。
- 目录最多 32 个 Provider 页面；历史最多 16 页、1000 个样本，并受配置的总响应字节预算约束。
  续页只提取同一固定端点的 OEM `page` 令牌；允许原样回显的冻结参数，拒绝外域 URL、
  改变目标/过滤条件的查询和重复令牌。每次完整目录/采样操作还受 Provider 总超时约束。
- OEM 24ai 的 `metricTimeSeries` 页面最大 limit 为 10；原先请求 20 会收到 HTTP 400
  `LimitExceeded`。修复为 10，并支持受控续取。JSON 小数直接解析为十进制文本，不先转浮点。
- OEM 自动选择原始、小时或日聚合；不声称原始留存粒度、覆盖完整率、counter/gauge 或速率。
  图表为近似趋势，精确值以原始采样表为准。截断、无数据、类型不支持、权限拒绝和来源超时
  分别展示，不合并成资源故障。

## 本次只读实测

在现有 OEM24ai 的已绑定 Oracle 数据库与 host 上，修复代码仅加载到临时 Python 进程，
未写入服务器文件、配置或数据库，未重启服务：

- 数据库：464 项目录，其中 343 项 NUMBER，目录未截断。
- 主机：294 项目录，其中 229 项 NUMBER，目录未截断。
- `Database_Resource_Usage/session_usage` 在一小时窗口返回 6 条采样，原生单位 `%`。
- `CPUUsageSummary/OverAllUsagePct` 返回 1 条采样，原生单位 `%`。
- `DATABASE_SIZE/USED_GB` 在该窗口返回 0 条，不据此补零或推断数据库容量。
- 文件系统 `metricTimeSeries` 返回 OEM 自身 HTTP 500 `InternalServiceError`；同一指标的
  `latestData` 返回 HTTP 200，包含 1 条真实 `mountPoint` 维度的最近采集值。
  页面支持显式选择快照，但不会悄悄把它当成已恢复的历史数据。
- 修复后的 Adapter 也已读到 `Response/DatabaseStatus` 的 1 条 STRING 快照，以及
  `Filesystems/available` 的 1 条 NUMBER 快照，原生单位分别为 `n/a` 和 `MB`。
  状态快照的实际采集时间早于当天，仍原样保留，不标注为最新实时健康。

这些结果不是所有数值指标均有采样的证明；当前实现提供完整可见目录和按需查询入口。
页面本地回归使用真实生产组件/脚本加模拟接口；没有将它等同于线上页面发布验收。
ODA/Exadata 等新资源类型、文本历史接口、标准指标自动语义映射和硬件诊断编排仍未交付。
文件系统的 OEM 历史接口 500 仍需从 OEM 服务端日志继续排查，不能宣称所有历史采样均已恢复。

官方契约参考：
https://docs.oracle.com/en/enterprise-manager/cloud-control/enterprise-manager-cloud-control/24.1/emrest/openapi.json
