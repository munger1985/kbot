# OEM 指标审计整改（2026-11-12）

依据真实归档 `oem-exadata-export-20261110.tar.gz`，SHA256 为 `8c6c9b7e55903eaeed3cc46551ca3be2b36f939a94d34fd8f7edd858c8e02c6f`。本次只改 OEM，不改 Prometheus、不新增数据库结构、不修改 Agent 授权、监控绑定或线上读取通道。

## 已实施的修正

| 审计项 | 整改 |
| --- | --- |
| F02 PDB 默认总览不匹配 | 根据已冻结绑定的 `target_type=oracle_pdb` 选择独立8面板：State、CPU/I/O/其他活跃会话分项、普通/临时表空间使用率、已分配/已用容量。仍精确核验类型、单位与完整维度，不借用 CDB、其他 PDB 或主机数据。 |
| F03 来源配置冲突与映射维护缺口 | 同一来源分别校验 Exadata 通道和标准映射；公开来源配置及两端表单允许维护 `standard_metric_mappings`。未知字段仍拒绝。只改通道时保留既有映射；明确提交新映射或 `{}` 时允许更新或清空，未配置不自动猜测。 |
| F04 Agent 选择已下架指标 | Agent 合同和参数验证仅接受审核场景内290项指标，不广告两项已下架的交换机/PDU父目标 ComponentState。完整原厂定义目录仍保留292项，不把定义存在当成可执行。 |
| F05 原厂错误被当网络不通 | OEM 原生 HTTP 5xx 使用 `SOURCE_PROVIDER_ERROR`；网络超时/连接失败保留 `SOURCE_UNREACHABLE`，权限、空样本、截断继续独立。Main API 仅在原生目录/采样两个入口保留已知来源错误码、状态码及固定安全详情，不再统一抹为 `AIOPS_UNAVAILABLE`；其他入口和共享 Provider 行为不变。 |
| F06 旧值/截断参与当前比较 | 服务端返回逐点 quality 与 current_eligible。审核总览采用保守展示时效：HOST 2小时、数据库/PDB 30分钟，写入审核资源；这是展示策略而非 OEM 在线调度。未知指标时效未审核时不参与当前比较。历史旧点仍可展示其历史趋势；超期快照/空值/截断不充当有效当前值，原文可在数据入口核对。 |
| F07 累计语义不准确 | 精确修正审计列出的16项为 CUMULATIVE；不生成均值偏离、增量或自动速率。保留原值、单位、维度。 |

PDB 的 REST 显示单位已经在现场核验：`Response/State` 无 unitDisplayName，waitcnt 三项 n/a，problemTbsp 百分比，tbspAllocation MB，与真实归档编码对应。数据库版本缺少阻塞/恢复区选择器时继续显示目标自身缺口，不自动套相似指标。

## 尚未闭环，不得称为全部修复

### F01：组件身份的原厂发现与父子归属

新只读取证更正了此前范围过大的判断：原厂 `/targets/{组件GUID}`、其 `/metricGroups` 和 `/latestData` 能读取 PDU 相位组件，并返回原始电压样本。组件没有出现在普通 `/targets?typeName=oracle_si_hw_component_pdu` 中，也不等于不能按 GUID 读取。

但是当前 OEM 的父/子 `/members`、`/memberof`、`/immediateRelationships`、`/properties`、`/configMetricGroups` 相关试读返回404，尚未获得可用于自动发现组件GUID并证明父归属的公开接口。不能将取证得到的固定GUID写入产品、凭相同组件名推断归属，或直接让应用访问 OEM 内部表。

因此53项服务器/交换机硬件与端口指标的自动成员覆盖仍未完成。继续保留既有已确认成员，不能为部署无提示扩大绑定或 Agent 授权。后续需先取得可核验的 OEM 父组件目录，再通过现有发现/确认流程更新整机成员。

### F05：原厂带维度历史查询仍失败

现场对 `wait_sess_cls/dbtime_waitclass_pct`、`UserBlock/count`、`Recovery_Area/Free_Space` 再测：不带 sort 以及按真实键显式 ASC 排序，均返回HTTP500，与此前 OMS ORA-01791 证据一致。错误分类已修正不等于原厂历史已恢复。

本次未静默启用普通 OEM EMCLI、未以最新快照冒充历史，也未修改原厂仓库或二进制。若改用 EMCLI，必须有独立显式配置、已审核的原生元数据校验与真实验收；7天上限不能冒充30天。

### G01/G02/G03：覆盖扩展

PDU 模块/相位电气量、额外 Cell 指标以及 RAC/ASM/Listener 完整资源监控仍属于待接入能力，不因本次修错自动纳入。尤其 PDU 电流阈值状态不等同综合硬件健康；累计能耗不做均值偏离，编号不画性能趋势。

## 验证与发布

定向回归覆盖精确PDB元数据、合法共享配置、明确更新/清空映射、Agent目录、16项累计语义、逐点时效/截断门禁。两端页面合同使用生产组件验证24面板及旧值、空值、截断、均值虚线与响应式布局。发布结果以实际测试和线上验收记录为准，不以构建或readiness代替全部指标成功。

发布前验证：两仓各423项OEM/Exadata定向单元和公开HTTP合同通过；两端OEM24面板及Exadata生产组件浏览器合同通过；Ammolite来源表单合同、类型检查/生产构建通过（保留既有chunk体积提示）；两仓架构边界通过。未运行全仓测试，未宣称原厂每项历史接口成功。

### 实际发布与追加回归

| 仓库 | 首批功能整改提交 | Main API 错误分类补充提交 |
| --- | --- | --- |
| Ammolite / dev | `5c8b9bf887dd41a75da1e692364af50eb517634b` | `ae6e28c51909250cdde346db1bf4cce25e2bb69a` |
| KBot / kbot4.1.1 | `fbe48e346251f6d4bf3b369274dff54e44541ab2` | `bfddc58dadf496c50c3a3131a7abffbdc3050056` |

以上已推送并部署至 kbotdev。首批定向重启两端 Main API、AIOps API、AIOps Worker，共6个进程；第二批只重启两个 Main API。第二批8个就绪检查全部200，49个无关服务进程及配置摘要未变。未运行全栈重启、未修改数据库或监控绑定。

首批备份/发布记录：`/home/ubuntu/aiops-oem-remediation-20261112-zHI0MxeJ/`；第二批：`/home/ubuntu/aiops-oem-error-boundary-20261112-Nc9h4dgK/`。第二批优雅退出超过30秒，Ammolite旧Main API二次优雅退出仍未完成，经PID/启动时刻/模块核验后仅终止该旧进程并恢复；实际恢复与就绪通过时间为2026-11-12 11:40:30 UTC。发布存在短时主API中断，不能记录为零中断发布。

追加扩大回归到16个OEM/Exadata单元测试文件及原生HTTP合同，两仓均 **473 passed**。同步修正旧接入测试仍禁止合法 `standard_metric_mappings` 的断言，保留禁止来源配置绑定数据库和任意查询的断言，并新增显式空映射合法用例。此前423项结果不代表包含该旧测试；以追加回归结果为准。

### 线上验收事实

- 两端普通OEM主机页面均已检查8个面板、7个图表，无页面脚本异常或横向溢出。主机状态原厂返回空采样，不伪造为可用。
- 两端普通数据库页面均有4项成功采样；`UserBlock/count`、`wait_sess_cls/dbtime_waitclass_pct`、`Recovery_Area/Free_Space` 的历史仍失败。补充发布后均为HTTP502 / `SOURCE_PROVIDER_ERROR`，不再是HTTP503 / `AIOPS_UNAVAILABLE`，这只证明错误分类整改成功。
- 两端普通主机8项请求全部HTTP200，实际响应带 `current_max_age_seconds=7200` 和逐点质量字段；文件系统旧快照真实返回 `STALE` / `current_eligible=false`，没有参与当前比较。其带维度历史基线仍失败，当前原始快照保留，不把无基线当成零偏离。
- 两端Exadata线上8场景、历史切换、7天均值、设备过滤、普通资源切换及5种宽度验收均通过，无页面脚本异常。KBot记录位于本地 `/tmp/exadata-status-table-online-kbot-BcOUm7/result.json`，Ammolite位于 `/tmp/exadata-status-table-online-ammolite-he56LB/result.json`。该验收验证当前场景与页面行为，不代表292项指标全部有线上采样。
- 第二批发布后在线开发日志在核对窗口中两端Main API各记录3条 `SOURCE_PROVIDER_ERROR`，对应上述已知原厂历史错误；两端AIOps服务未发现新增ERROR。没有据此宣称整个OEM监控无错误。

线上PDB没有现成已授权绑定，本轮不为验收新增绑定；PDB整改依据真实导出、现场元数据和回归验证，不能宣称已经完成线上PDB全流程验收。F01、原厂历史读取及G类扩展仍以“尚未闭环”章节为准。
