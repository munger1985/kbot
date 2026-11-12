# 事实与只读核查 · 2026-11-12

本任务不是 Exadata 产品发布物料，不判断厂商最新版本/型号。技术事实以用户真实导出、当前产品源码和在线 OEM 读取为依据。

1. 真实归档只有六 CSV，外部 README 单独提供。README 计数为50目标、88680定义、28651最新记录，无历史。当前重新以 csv.DictReader 解析相关定义与最新值。
2. ComponentState/ComponentState 实际最新样本：oracle_si_hw_component_server：82 OK、3空值；oracle_si_hw_component_switch：12 OK；父 oracle_si_netswitch、oracle_si_pdu、oracle_si_server_map 无该组最新样本。仅定义存在不能证明可在父接口直接读取。
3. 整机/Grid 的 Response 为合成定义，Rack 没有审核独立可读状态，不能当普通 NUMBER/STRING 值硬读，更不能借用 Cell 状态。
4. 当前正常公共 API：14:25:14 概览七对象/十二查询/九采样；交换机和PDU父 ComponentState 均 SOURCE_QUERY_UNSUPPORTED，DB Server Response 为 NO_DATA；Cell 资源14:25:57十二指标均 OBSERVED。
5. 原厂 EMCLI 独立查询固定 SIM 交换机 ComponentState 返回 {}，成功返回不代表有状态值。不向 OEM 内部表写入数据。
6. 当前父错误不能证明风扇/电源等子组件全部无监控；现有模拟只有管理访问可达，产品尚未建立可读子组件身份。
7. Cell 的温度、上限、下限当前均32.0；真实导出中同节点这三列也有22/22/22、23/23/23、24/24/24。不能将这种相等本身宣判为模拟器错误或修改为臆造阈值。单位未声明，不补写°C。
8. A 的外部设计参照核实：2026-11-12 通过官方 Carbon 页面 https://www.carbondesignsystem.com/building-blocks/core/components/data-table/guidelines 实际读到 Data table、toolbar、dense、expandable；不宣称获奖或复制第三方品牌。
9. B 是设计哲学锚点，不宣称由相关设计师制作或背书。C 的轮盘为秒03，3%20+1=4；只取几何交互表达，产品配色约束优先。
10. 私有正常浏览器证据：/tmp/exadata-hardware-render-ammolite-Dl9HK6；Cell原始响应：/tmp/exadata-redesign-audit-lxBJ9A/cell.json。不把浏览器会话、凭据或真实客户主机名放入设计稿。
