# Exadata OEM 指标与页面重设计

当前状态：用户已确认 A 做整机首页＋B 做设备详情，进入正式实施；实施与部署结果另行记录。

打开 [三版对比](design-demos/index.html)，或分别预览：

- [A：类型分区](design-demos/direction-a-type-sections.html)（推荐），按 DB Server、Cell、互联与供电设备对照节点。
- [B：设备详情](design-demos/direction-b-device-focus.html)，按单个对象深入查看，保留性能卡、配置和证据。
- [C：监测矩阵](design-demos/direction-c-observation-matrix.html)，逐设备对照，并在旁栏核查证据。

三版均使用真实模拟 OEM 固定快照，并明确标记为离线样稿。状态和数值来自 2026-11-12 14:25 的正常公共 API 取证，不是物理 Exadata 的生产值。只在样稿里调整呈现，不掩盖现场数据缺口。

## 核查结论与正式修复范围

1. 交换机与 PDU：管理可达有值，但父 ComponentState 的读取返回 SOURCE_QUERY_UNSUPPORTED。展示模型目前把所有 SOURCE_ 非空错误统一降成“读取失败/读取错误”，应按结构化原因区分“不支持读取、未采样、查询失败”。不能修改错误码、补采样或视为健康。
2. 真实硬件状态：导出中有效 ComponentState 主要位于 oracle_si_hw_component_server / oracle_si_hw_component_switch，分别有 82 条 OK+3 条空值、12 条 OK。当前模拟没有这些可靠子组件身份；正式修复必须建立官方可验证的父子组件取证，不可在父目标上合成 OK、编 GUID 或借管理 UP 替代。
3. DBM、Grid、Rack：属于资产结构对象，不应逐项占用状态区并重复长说明。不能因为将它们收起，就声称已取得独立状态。
4. Cell：本次资源场景十二项均 OBSERVED，未复现读取失败。其 CPU/内存重复显示，以及配置、模式、温度上下限、状态与性能混排，是明确的呈现问题。温度上下限与读数相等在真实导出也存在，不据此臆造新阈值；未声明单位不补写摄氏度。
5. 原始定义、完整维度、时间、质量、错误码和证据必须保留在详情中。7 天/30 天均值比较保留，不跨节点合并，不绘制固定快照的假趋势；月历史原厂限制不改。

## 验证

三版各在 1440×1000、768×1024、390×844 验证，共九组；检查切换设备/场景、展开资产与缺口、证据弹窗、主题切换、完整标识加载及整页不横向溢出。浏览器 pageerror=0，远端请求=0；截图和机器结果见 design-demos/verification.json。样稿不发请求，因此这些验证不能替代正式产品 API、采集或线上验收。

设计与实施约束见 [design-spec.md](design-spec.md)，事实核查见 [product-facts.md](product-facts.md)，用户确认见 [direction-approved.md](direction-approved.md)。
