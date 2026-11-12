# Exadata OEM 指标图表逐项审核

日期：2026-11-12。来源为当前真实导出审核目录 `metrics.v1.json`、`device_metrics.v1.json`、状态字典与8个监控场景；不是线上采样齐全的声明。

## 展示规则

- 整机概览、DB Server、Cell、磁盘 IO、Flash / Smart IO、IORM、互联设备、容量均复用相同展示决策；原有状态行保留。
- 最新快照：各维度独立柱状对比；文件系统/容量使用率在单维、原生百分比、0–100%时用使用/未使用环图。CPU、内存单独补读过去1小时真实历史，使用折线及7天/30天均值虚线；缺历史不复制快照生成趋势。不把其他百分比、IO负载、平均值、速率当作可加总的饼图扇区。
- 历史：同设备、同指标、同原生单位、相同完整维度独立折线，至少2个有效时间点；缺值不补零、不跨空值连线，不把最新快照复制成趋势。
- 柱图完整保留维度，超过6个提供滚动；零值保留，空值无柱。数值只用于近似绘制，原始十进制文本不变；不推断阈值、健康、跨组件均值或精确计数增量。
- 卡片默认只显示名称、设备、质量、当前值、时间、图表、精简均值摘要与一个“查看数据”入口；原厂选择器、单位、完整采样/维度/质量、缺口、均值对比窗口及两位小数的明细均在该入口内。
- 使用Prometheus卡片的surface/surface-subtle、line、accent及六色序列色板，tooltip使用surface-raised/line-strong；环图剩余部分用accent-soft。序列颜色不代表健康，错误/过期通过质量标签区别，配置/状态不强行数值化。
- 本表只覆盖Exadata OEM展示；普通OEM总览见[OEM图表审核](aiops-oem-chart-audit.md)。非OEM页面、授权、目录、OEM模拟器、诊断与数据库不变；CPU/内存通过既有历史接口补读，不新增服务端合同。

## 审核范围

共 290 个独立指标；性能/容量 233、状态 28、配置/标识 29。同一指标可在多个场景出现，以下每个代码只列一次。运行时仍以实际采样、类型、单位及质量决定是否绘图；无数据和读取失败不画空健康图。

| 指标代码 | 场景 | 原厂类型 | 最新快照 | 历史模式 | 语义依据/边界 |
|---|---|---|---|---|---|
| `oracle_exadata.Aggregated_Exadata_Capacity_Metric.Usage` | capacity | NUMBER | 占用环图 | 折线（至少2个有效时间点） | 仅单维且0–100%；多维或越界改柱图；未使用=100−使用 |
| `oracle_exadata.Aggregated_Exadata_Capacity_Metric.flashdisktype` | capacity | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Aggregated_Exadata_Capacity_Metric.size` | capacity | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_Exadata_Diskgroup_Capacity_Metric.count` | capacity | STRING | 维度柱图 | 不生成数值历史 | 明确容量选择器允许近似绘制；原厂类型与精确文本不变 |
| `oracle_exadata.Aggregated_Exadata_Diskgroup_Capacity_Metric.size` | capacity | STRING | 维度柱图 | 不生成数值历史 | 明确容量选择器允许近似绘制；原厂类型与精确文本不变 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_io_load` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_io_util` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_large_read_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_large_read_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_large_write_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_large_write_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_lg_r_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_lg_w_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_read_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_read_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_read_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_sm_r_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_sm_w_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_small_read_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_small_read_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_small_write_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_small_write_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_write_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_write_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.avg_cd_write_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.flash_type` | disk | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.max_cd_sm_r_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.max_cd_sm_w_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.sum_cd_io_load` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.sum_cd_io_util` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.sum_cd_read_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.sum_cd_read_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.sum_cd_write_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Aggregated_HardNFlashDisk_Metric.sum_cd_write_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_io_load` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_io_util` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_lg_r_bytes` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_lg_r_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_lg_r_reqs` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_lg_w_bytes` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_lg_w_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_lg_w_reqs` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_read_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_read_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_read_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_sm_r_bytes` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_sm_r_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_sm_r_reqs` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_sm_w_bytes` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_sm_w_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_sm_w_reqs` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_write_iops` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_write_latency` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.cd_write_throughput` | disk | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.CellDisk_Metric.celldisk_type` | disk | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.CellDisk_Metric.flash_type` | disk | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.CellSrv_Status_Metric.Status` | overview, cell | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_exadata.CellSrv_Status_Metric.msStatus` | overview, cell | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_exadata.CellSrv_Status_Metric.rsStatus` | overview, cell | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_exadata.Cell_Metric.cell_name` | cell | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Cell_Metric.cl_iorm_boost` | cell | NUMBER | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Cell_Metric.cl_runq` | cell | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Cell_Metric.cpu_utilization` | overview, cell | NUMBER | 过去1小时真实折线＋均值虚线 | 折线＋均值虚线 | 同Cell同维度；单点仅画点，空值断开；不复制快照 |
| `oracle_exadata.Cell_Metric.iorm_mode` | cell | NUMBER | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Cell_Metric.led_status` | cell | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Cell_Metric.memory_utilization` | overview, cell | NUMBER | 过去1小时真实折线＋均值虚线 | 折线＋均值虚线 | 同Cell同维度；单点仅画点，空值断开；不复制快照 |
| `oracle_exadata.Cell_Metric.network_received` | cell | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Cell_Metric.network_sent` | cell | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Cell_Metric.temperature` | cell | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Cell_Metric.temperature_lower` | cell | NUMBER | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Cell_Metric.temperature_upper` | cell | NUMBER | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Exadata_Disk_Status_Metric.status` | disk | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_exadata.Filesystem_Utilization_Metric.cell_name` | capacity | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Filesystem_Utilization_Metric.utilization` | capacity | NUMBER | 占用环图 | 折线（至少2个有效时间点） | 仅单维且0–100%；多维或越界改柱图；未使用=100−使用 |
| `oracle_exadata.FlashCache_Metric.all_bytes` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.all_reqs` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.cell_name` | flash | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.FlashCache_Metric.default_hits` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.default_hits_rate` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.default_misses` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.default_misses_rate` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_bykeep_overwr_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_default_read_iops` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_default_read_throughput` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_default_used` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_by_r_dw_per_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_by_r_miss_dw_per_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_rq_disk_write_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_rq_r_dw_per_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_rq_w_first` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_rq_w_first_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_rq_w_overwrite` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_rq_w_overwrite_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_rq_w_populate_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_io_rq_w_skip` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_keeppool_read_iops` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_keeppool_read_throughput` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_keeppool_used` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_read_hit_rate_oltp` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_read_iops` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_read_throughput` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.fc_used` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.keep_bytes` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.keep_hits` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.keep_miss_reqs` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.keep_misses` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.keep_misses_rate` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.keep_reqs` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.miss_bytes` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashCache_Metric.miss_reqs` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.cell_name` | flash | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.FlashLog_Metric.flashlog_actual_outliers` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_by_keep` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_disk_io_errs` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_efficiency_percentage` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_efficiency_percentage_hour` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_flash_io_errs` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_flashlog_io_w_skip_busy_min` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_io_db_by_w_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_io_fl_by_w_sec` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_io_w` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_io_w_skip_busy` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_io_w_skip_large` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_io_w_skip_no_buffer` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.FlashLog_Metric.flashlog_prevented_outliers` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cell_name` | iorm | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_io_by_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_io_rq_lg_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_io_rq_sm_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_io_tm_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_io_util` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_io_wt_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_io_wt_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_io_wt_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_CG_Metric.cg_fd_iops` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.cell_name` | iorm | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_by_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_rq_lg_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_rq_sm_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_tm` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_tm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_tm_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_util` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_wt_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_wt_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_io_wt_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_DB_Metric.db_fd_iops` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.cell_name` | iorm | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_io_by_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_io_rq_lg_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_io_rq_sm_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_io_tm_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_io_util` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_io_wt_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_io_wt_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_io_wt_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.Flash_Disk_IORM_PDB_Metric.pdb_fd_iops` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.HostInterConnect_Statistics.c_mb_dropped` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.HostInterConnect_Statistics.c_mb_rdma_dropped` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.HostInterConnect_Statistics.c_mb_received` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.HostInterConnect_Statistics.c_mb_resent` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.HostInterConnect_Statistics.c_mb_sent` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.HostInterConnect_Statistics.c_rdma_retry_latency` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.HostInterConnect_Statistics.cell_name` | network | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.IORMDB_Metric.cell_name` | iorm | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.IORMDB_Metric.db_fd_io_tm` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_fd_io_tm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_by_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_load` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_tm_lg` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_tm_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_tm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_tm_sm` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_tm_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_util` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_util_lg` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_util_sm` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_wt_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_io_wt_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_iops` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.db_rq_wait` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.num_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.num_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.wait_time_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORMDB_Metric.wait_time_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_CONSUMER_GROUP_Metric.cell_name` | iorm | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.IORM_CONSUMER_GROUP_Metric.cg_io_by_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_CONSUMER_GROUP_Metric.cg_io_util` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_CONSUMER_GROUP_Metric.cg_iops` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_CONSUMER_GROUP_Metric.cg_req_wait` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.cell_name` | iorm | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.IORM_PDB_Metric.num_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.num_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_io_by_sec` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_io_load` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_io_util` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_io_util_lg` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_io_util_sm` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_io_wt_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_io_wt_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_iops` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.pdb_rq_wait` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.wait_time_lg_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.IORM_PDB_Metric.wait_time_sm_rq` | iorm | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.SmartIO_Metric.cell_name` | flash | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_exadata.SmartIO_Metric.smartio_passthru` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.SmartIO_Metric.smartio_read_flashcache` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.SmartIO_Metric.smartio_read_harddisk` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_exadata.SmartIO_Metric.smartio_storage_index` | flash | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_hw_component_server.ComponentState.ComponentState` | overview, capacity | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_hw_component_server.ComponentState.ErrorDetail` | capacity | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_si_hw_component_switch.ComponentState.ComponentState` | overview, capacity | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_hw_component_switch.ComponentState.ErrorDetail` | capacity | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_si_netswitch.PfcIfPriorityEntryMetric.DeltaNumIndications` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_netswitch.PfcIfPriorityEntryMetric.DeltaNumRequests` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_netswitch.PfcIfPriorityEntryMetric.NumIndications` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_netswitch.PfcIfPriorityEntryMetric.NumRequests` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.AdministrationState` | network | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_si_network_port_server.NetworkPortPerformance.InDiscard` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.InErrors` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.InNUcastPackets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.InOctets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.InOctetsRate` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.InUcastPackets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.MTU` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.OperStatus` | overview, network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_network_port_server.NetworkPortPerformance.OperStatusDown` | network | NUMBER | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_network_port_server.NetworkPortPerformance.OutDiscards` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.OutErrors` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.OutNUcastPackets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.OutOctets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.OutOctetsRate` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.OutUcastPackets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_server.NetworkPortPerformance.TotalOctetsRate` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.AdministrationState` | network | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_si_network_port_switch.NetworkPortPerformance.InDiscard` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.InErrors` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.InNUcastPackets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.InOctets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.InOctetsRate` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.InUcastPackets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.InUnknownProto` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.MTU` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.OperStatus` | overview, network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_network_port_switch.NetworkPortPerformance.OperStatusDown` | network | NUMBER | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_network_port_switch.NetworkPortPerformance.OutDiscards` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.OutErrors` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.OutNUcastPackets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.OutOctets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.OutOctetsRate` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.OutUcastPackets` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.Speed` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.NetworkPortPerformance.SpeedUnits` | network | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_si_network_port_switch.NetworkPortPerformance.TotalOctetsRate` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.AdminStatus` | network | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.Bandwidth` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.NetworkRecdRate` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.NetworkSentRate` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.OperStatus` | overview, network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.PercentBandwidthUsedIncoming` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.PercentBandwidthUsedOutgoing` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.PercentDiscardsIn` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.PercentDiscardsOut` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.PercentErrorsIn` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.PercentErrorsOut` | network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_network_port_switch.SwitchEthPortsMonitor.PortStatusError` | network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `host.CPUUsageSummary.OverAllUsagePct` | overview, dbserver | NUMBER | 过去1小时真实折线＋均值虚线 | 折线＋均值虚线 | 同DB Server同维度；单点仅画点，空值断开 |
| `host.Load.memUsedPct` | overview, dbserver | NUMBER | 过去1小时真实折线＋均值虚线 | 折线＋均值虚线 | 同DB Server同维度；单点仅画点，空值断开 |
| `host.OEMTarget.status` | overview, dbserver | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.AccessPointResponse.statusMsg` | overview, network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_server_map.AccessPointResponse.statusMsg` | overview, dbserver | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_server_map.ComponentState.ComponentState` | overview, dbserver | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_server_map.SystemStatus.HealthStatus` | overview, dbserver | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_server_map.SystemStatus.PowerStatus` | overview, dbserver | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_si_server_map.SystemStatus.OSStatus` | overview, dbserver | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_pdu.PduHttpResponse.Status` | overview | NUMBER | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SwitchSystemStatus.HealthStatus` | network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SwitchSystemStatus.PowerState` | network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SwitchSystemStatus.CoolingState` | network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SwitchSystemStatus.TemperatureState` | network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SwitchSystemStatus.CoolingRedundancy` | network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SwitchSystemStatus.PowerRedundancy` | network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SwitchStatus.Status` | overview, network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SensorStatus.Status` | overview, network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_netswitch.SensorStatus.Value` | overview, network | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_netswitch.SensorStatus.Unit` | overview, network | STRING | 配置/原始值 | 不生成数值历史 | 配置、标识、阈值或指示值，折叠查看 |
| `oracle_si_netswitch.PowerSupplyStatus.Status` | overview, network | STRING | 紧凑状态 | 不生成数值历史 | 按审核字典逐维度显示，不绘数值图 |
| `oracle_si_pdu.PduSummaryStatus.TotalPduActivePower` | overview, capacity | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
| `oracle_si_pdu.PduSummaryStatus.TotalPduEnergy` | overview, capacity | NUMBER | 维度柱图 | 折线（至少2个有效时间点） | 原生数值逐维度比较；单位原样，不推导速率或健康 |
