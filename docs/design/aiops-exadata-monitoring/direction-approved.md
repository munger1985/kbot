# Exadata OEM 监控方向确认

确认日期：2026-11-12。

用户先选择“使用方案A”，随后明确：“建议采用‘A 做整机首页＋B 做设备详情’，按你的建议来做”。

最终采用 A＋B：A 类型分区作为整机首页，B 对象聚焦作为设备详情；不采用 C 监测矩阵。已展示初稿及截图：

- A：design-demos/direction-a-type-sections.html、direction-a-type-sections.png。
- B：design-demos/direction-b-device-focus.html、direction-b-device-focus.png。
- C：design-demos/direction-c-observation-matrix.html、direction-c-observation-matrix.png。
- 三版入口：design-demos/index.html；对比：design-demos/comparison.png。

实施沿用现有 AppShell、主题 tokens 和共享按钮；真实公共监控 API 驱动，不将原型固定快照、演示提示或假趋势带入正式页面。仅修改 Exadata＋OEM，两端同步；保留分页、权限边界、原始证据、历史和7天/30天均值偏离。硬件可达与硬件健康分开，缺口不填健康。

本文件记录方向授权，不代表实施、验证或部署已完成。
