# Exadata → 真实 OEM 调试夹具

本目录只在隔离实验环境模拟 CellCLI 与只读 SNMP 交换机。OEM 使用未修改的 XA/SI 插件采集、上传并保存指标，AIOps 仍访问真实 OEM REST，不直读模拟器，不写 OEM 内部表。

全部名称含 SIM，全部数值是合成数据。字段映射依据真实 OEM 导出目录及 XA 24.1.1 原厂采集脚本；不包含客户主机、凭据或真实业务值，也不承诺覆盖目录全部 267 项。

## 现场状态（2026-11-11）

- OEM：既有 `aiops-oem`，OMS/Agent 24.1.0.0.0、XA 24.1.1.0.0。
- 新标准 Agent：`aiops-exadata-sim-monitor`，主机 `aiops-exadata-monitor.lab`，独立卷 `aiops_exadata_sim_monitor_data`。
- 新 CellCLI：`aiops-exadata-cellcli-sim`，主机 `aiops-exadata-cellcli.lab`，实验网络地址 `172.22.0.8`；无宿主端口发布。
- 原厂目标：`AIOPS_SIM_EXADATA_DBM01`（oracle_dbmachine）、`AIOPS_SIM_EXADATA_GRID01`（oracle_exadata_grid）、`AIOPS_SIM_EXADATA_CELL01`（oracle_exadata）。
- 原厂允许的结构是 DB Machine → Grid → Cell，不能直接把 Cell 加入 DB Machine。三者通过原厂 create/modify_system 命令建立，现已通过官方 Data Service 包含关系和 GUID 目录双重独立核验，并与原厂 `get_system_members -depth=2` 结果一致。
- overview 的 5 项最新值、cell 的 12 项最新值以及 CPU/内存数值历史已被实际 AIOps OEMAdapter 读取。采样不足时不能推断趋势。
- KBot/Ammolite 分别使用本仓库 Adapter 完成严格整机 latest、cell、history smoke，输出 `inventory_verified=true`。最后相关回归为两端各 10 文件 **316 passed + 13 subtests passed**；KBot 加 Oracle 验收生成器为 **324 passed + 18 subtests passed**，干净 Git archive 加部署配置环境复跑也通过；两端架构边界检查通过。不是全仓或全部指标覆盖声明。
- 一分钟调度已在 Agent 集合配置落地；手工仅刷新上传后，OEM 历史包含 09:49、09:50、09:51 UTC 的连续自动采样，CPU 为 26.0/26.2/26.88%，内存为 57.0/57.07/57.29%，不是只保存一条手工采样。上传批次延迟仍存在。
- **整机发现已修复**：初版 `/members` 为 404；现统一使用官方 GET `/em/websvcs/restful/emws/core/v0/list/dataservice`，固定资源 `TargetAssociations` 和 `internal_contains`，每个父目标精确过滤，要求 `exceedsMaxRows=false`，再逐项查询唯一名称/类型对应的 GUID 并复核详情。只展开根及直接 Grid；歧义、截断、未应用过滤、循环和超预算均失败关闭。没有双路径降级，没有 SQL、直连 OEM 仓库或额外服务。
- **KBot 页面已验收**：已按授权部署两端、无损新增 Exadata 两表并定向重启三个 API。KBot 经正常登录、公共发现和系统创建接口新增独立 SIM 业务系统及硬件绑定；Chromium 实际核对 5 行概览、12 行 Cell 最新值、450 行历史样本，CPU/内存值与采样时间一致，无 pageerror。Ammolite 现用 OEM 凭据严格读取通过，尚缺正常 App 登录，未创建绑定或验收其页面。不得把 Adapter smoke 等同于两端页面全部通过。


## 场景修复状态（2026-11-12）

下述为当前修复结果；上方及“页面联调部署与入口（2026-11-11）”是当时的验收记录，不能作为当前覆盖或部署版本的声明。完整变更、证据与限制见[实施设计 10.10](../../../docs/operations/aiops-exadata-oem-implementation-design.md#1010-场景无数据修复与覆盖复核2026-11-12)。

- 已在线原位更新 CellCLI 源码并备份，没有重启容器、OEM 或既有数据库 Agent，也没有修改原厂插件或写 OEM 内部表。
- FlashCache、FlashLog、DB/PDB/消费组磁盘与 Flash IORM 六组、Cell 互联及部分容量已由真实原厂插件采集上传。当前 Cell 的 Flash 场景 56/56、IORM 72/72、Cell 互联 7/7 目录项在 OEM 最新快照有值；不代表整机 267 项已覆盖。
- 修复了带下划线 objectType 的过滤及容量单位；Filesystem 旧 5 行混入非文件系统对象，当前仅 /、/opt/oracle 两个正确维度，不能把旧行视为有效百分比。
- 应用端同步修复维度名顺序误判，按名取值并保留严格身份校验；专用指标 HTTP 5xx 明确显示 OEM 服务错误，不再冒充无数据；删除指定顶部提示。两端代码已完成回归，**尚未部署应用或重启应用服务**。
- **未恢复**：OEM 带维度 metricTimeSeries 仍因原厂 ORA-01791 返回 500；没有交换机/端口模拟目标；磁盘状态与 ASM 容量虽可由 Agent getmetric 算出，OEM latestData 仍为空。不得以快照替代历史或以 Cell 互联冒充交换机。
- 本次手动采集/上传成功，不等于全部新增集合按一分钟调度；IORM 仍为 15 分钟、容量 1 小时、Filesystem 24 小时。原厂拒绝提高频率后未使用 -force。

### 更新既有模拟器

仅用于已经部署、核对过的本实验容器；不要为更新源码重新执行创建/注册流程。将审核后的本目录工件放到独立临时目录，先只读核对现场文件摘要，再将该旧摘要作为必填参数：

```bash
sudo sha256sum /opt/aiops-oem-lab/exadata-sim/cell_simulator.py
sudo python3 update_cellcli.py --source ./cell_simulator.py --expected-sha256 已核对的旧文件SHA256
```

更新器限定 aiops-exadata-cellcli-sim 的唯一只读 bind mount、固定源路径及运行状态；拒绝漂移和符号链接，编译检查后备份到 state/exadata-sim-update-*，原位写入保留绑定 inode，并在容器内复核 SHA256，失败恢复旧内容。不重启容器，不自动运行原厂采集或提高采集频率。备份和 receipt 应保留。

本次最终源码 SHA256 为 f59daf83690ba1066b59aa8cbaa98ecb7b5f197dcbb0993c6278af2117999bea。只对独立 SIM Agent 手工执行过的有效集合名称为 Exadata_Performance_Metrics_All、Exadata_IORM_Performance_Metrics_All、Exadata_Capacity_Metrics、Filesystem_Utilization_Metrics；必须等待成功后再上传，不能使用子指标组名称或把失败命令当作成功。


## 剩余三项与交换机模拟（2026-11-12 续）

本节及[实施设计 10.11](../../../docs/operations/aiops-exadata-oem-implementation-design.md#1011-三项剩余问题的原厂证据交换机模拟与后续边界2026-11-12)是当前复核结果，替代上述“应用未部署”“状态/ASM 上送留存不明”的旧状态：

- 两端应用修复已按用户确认部署，只重启两个 AIOps API；Ammolite 正常会话浏览器已显示五场景首页卡片。带维度历史仍准确显示 OEM 读取失败，不是历史已修复；没有宣称 KBot 本轮浏览器也验收通过。
- OEM 原厂 get_metric_data 已能读到两块磁盘 normal、SIMDATA01/SIMDATA 的 size=8192/count=1，以及聚合磁盘多个历史采样。REST 对纯 STRING 带键组错误使用数值键集合过滤，历史维度键 DISTINCT 子查询又存在排序错误。不能再把这三组归因于模拟器没有上传，不能改原厂 STRING 为 NUMBER 或用最新值替代历史。
- 新增只读 SNMP 交换机 aiops-exadata-switch-sim，仅实验网 172.22.0.9:1161/UDP，无宿主端口；两个 100Gbps SIM Ethernet 端口、PFC priority 0..7，原始累计值由原厂 SI 计算 rate/Delta。机箱序列号等身份明确含 SIM，没有复制客户身份、GUID 或业务快照。
- 原厂 SI 插件已在独立 Agent 部署 Success；原厂 SNMP AP 注册后，AIOPS_SIM_EXADATA_SWITCH01 已在 OEM get_targets 中为 Up。端口配置、性能、PFC 手动采集及上传成功，但尚未把这些结果等同于 OEM REST 全部值、自动采集调度或页面验收。
- 独立原厂管理员 REST 已核对 PfcIfPriorityEntryMetric/latestData：16 行、64 个非空值、两个端口与 priority 0..7；未使用现用应用监控账号验收。NetworkPortPerformance 只有 Agent 本地两行性能值，父交换机对应 REST 为 404；NetworkPortsTargetComponent 单项仍报 Empty snmp response，原厂所需 VTP VLAN/FDB 等附加配置 MIB 尚未模拟。PfcIfEntryMetric 本地单项也未成功，因此不声明端口组件或全部 PFC 已覆盖。
- 真实导出的 138 个交换机端口是 3 个父交换机下的组件，不是独立 Target。现有 AIOps 尚需原厂组件身份支持；未伪造端口目标，也未把交换机加入现有 DBM 而破坏两端已确认的 inventory digest。
- 完成页面读取还需原厂补丁，或另行明确批准仅为 Exadata 增加原厂 EMCLI 只读通道和组件发现。当前应用读取方式、授权、业务绑定均未改，普通监控诊断不动。

### 仅首次创建交换机

既有模拟交换机和注册记录不可重复执行以下命令。确认独立 SIM Agent 的 oracle.sysman.si:24.1.1.0.0 原厂部署为 Success，再把本目录三个交换机脚本置于现场部署目录：

```bash
sudo python3 deploy_switch.py
sudo python3 register_switch.py
```

注册器检查固定容器身份、网络、源码 SHA、同名目标和已有提交记录；失败保留现场，不使用 force。SNMPVersion 必须 v2c，原厂实际 monitoring_cred 为单数；通过原厂 argfile 的临时 0600 文件传凭据，不放进进程参数，完成后删除临时文件。容器 private community 保留在 state/exadata-sim-switch-snmp，不输出或提交。receipt 中 command_accepted=true 仅证明命令成功，inventory_verified 仍为 false，不能作为 DBM 成员或页面覆盖凭证。

已对新 AP 成功手动执行的集合为 Response、_Capability_、correlation_identifier、oracle_si_net_port_config、oracle_si_net_port_perf、oracle_si_switch_pfc；原厂命令格式：

```bash
emctl control agent runCollection AIOPS_SIM_EXADATA_SWITCH01_SNMP:oracle_si_switch_cisco_ios_snmp oracle_si_net_port_perf
emctl upload agent
```

仅在独立 SIM Agent 以原厂要求的用户执行，先等待采集完成再上传；不适用于既有数据库 Agent。本轮经原厂 preview 后，仅对新交换机启用 oracle_si_switch_pfc 的五分钟周期；OMS 命令成功、Agent 专属 collection XML 已核对，尚未验收连续自动采样。其他默认禁用集合没有一并开启或提高频率。当前交换机源码 SHA256 为 5916b7c0ca1b305ac3c21f13a19bebcb881f620cf3ceaadd57310f2b849e6efa。只重启过新交换机容器；源码备份及各次预检失败记录留在其私有 state，旧目标、卷和数据未删除。

定向测试两端各 28 passed + 13 subtests passed，两端架构边界通过。没有扩大为整机或全部指标覆盖声明。

## 仅在相同实验环境部署

这些是现场专用工件，不是通用 OEM 安装器。不下载或分发 Oracle 软件，不改通用 observability INI，不把 OEM 来源配置移出 AIOps App。

宿主机须已有 Docker 网络 `aiops-oem-lab`、实验镜像 `aiops-lab/oem-remote-agent:ol8`、`/opt/aiops-oem-lab/staging/agent19/24.1.0.0.0_AgentCore_*.zip`、私有 `/opt/aiops-oem-lab/secrets/` 和既有安全包装器 `scripts/oem_cli.py`、`scripts/oem_install_status.py`。镜像须已准备 Python 3.12、SSH、Oracle 用户及真实 SysVinit 3.14 服务入口；本目录不自动安装系统依赖或绕过原厂 prerequisite。

部署文件放在宿主机 `/opt/aiops-oem-lab/exadata-sim/`。以下命令仅用于**全新**模拟容器/卷/目标，不能在已有成功或失败记录上盲目重跑：

```bash
cd /opt/aiops-oem-lab/exadata-sim
sudo python3 create_agent.py
sudo python3 deploy_agent.py prepare
sudo python3 deploy_agent.py install
sudo python3 deploy_agent.py root
sudo python3 deploy_agent.py status
```

安装器检查 11 项原厂 prerequisite，全部 Passed 才安装；保留非空卷、原厂日志和失败记录。`prepare` 的原厂 prerequisite 可能留下 OUI/JDK；`install` 仅在严格核验本次前置目录形态后归档它，不使用覆盖安装。中断后的恢复开关只供人工核对后的同一现场恢复，不作默认步骤。

随后通过 OEM 原厂插件管理为此新 Agent 部署 `oracle.sysman.xa`，用原厂 `get_plugin_deployment_status` 确认该 Agent 为 Success 后，执行：

```bash
sudo python3 deploy_cellcli.py
sudo python3 register_targets.py cell
sudo python3 register_targets.py grid
sudo python3 register_targets.py system
```

注册器拒绝同名目标、重复提交、未完成依赖。返回码 0 伴随 warning/error 也不签发命令成功凭证；即使命令成功，`inventory_verified` 仍为 false，直至只读成员接口独立验证。整理后的三阶段注册器只做静态/单元验证；已有现场来自同等原厂命令的分阶段排障操作，未为测试新注册器而删除重建目标。

Cell 属性为 `MgmtIPAddr=172.22.0.8;SupportExacli=0`。采集用独立 ed25519 身份，私钥只在新 Agent 的持久卷；从本次生成的服务器公钥固定 known_hosts。SSH ForceCommand 仅接受 `cellcli -xml -e` 的有限只读 LIST；关闭密码、交互 TTY、转发、root 登录和用户环境覆盖。未知命令/对象/静态属性明确 fault，不用健康零值填补。

可按需对**新模拟目标**用原厂 `modify_collection_schedule` 把 `Exadata_Status_Metrics_All`、`Exadata_Performance_Metrics_All` 改为一分钟；现场已执行。调度配置成功不代表 OEM REST 每分钟立即更新，原厂 Agent 的批量上传和仓库处理仍可能延迟。

可手工触发的原厂集合名称是 `Exadata_Status_Metrics_All`、`Exadata_Performance_Metrics_All`、`Response`，不是 `Cell_Metric` 等子指标组名。只能对新 Agent 执行 `emctl control agent runCollection`，等它完成后再 `emctl upload agent`，不能并发把未完成的上传说成成功。

## 只读 smoke

在任一仓库使用该仓库自己的 PYTHONPATH 和 Python 环境，凭据通过已有安全工具的管道传入 JSON `username/password`，不要将明文放在命令行、日志、版本库或示例文件中。

```bash
python tests/smoke/check_exadata_oem_sim.py --endpoint https://127.0.0.1:17804/em --insecure-lab-tls
python tests/smoke/check_exadata_oem_sim.py --endpoint https://127.0.0.1:17804/em --insecure-lab-tls --scene cell
python tests/smoke/check_exadata_oem_sim.py --endpoint https://127.0.0.1:17804/em --insecure-lab-tls --history
```

示例依赖受控 SSH 隧道，17804 不是公开 OEM 地址；自签名开关仅限本次实验。SSH 使用用户提供的私钥及 `KexAlgorithms=curve25519-sha256`。

默认模式严格验证根/成员，当前两端已通过，输出 `acceptance_scope=VERIFIED_EXADATA_INVENTORY_AND_METRICS`、`inventory_verified=true`。排障选项 `--cell-only` 仍只读唯一 SIM Cell，输出 `acceptance_scope=METRIC_TRANSPORT_ONLY`、`inventory_verified=false`；不要将该选项用于整机验收。两种模式都不持久化、不生成绑定，不绕过应用授权/绑定流程。

## 页面联调部署与入口（2026-11-11）

应用环境为 kbotdev（私网 `10.0.0.190`），OEM 位于 aiopsdb（私网 `10.0.0.88`）。完整数据库核验、备份及验收记录见[实施设计 10.3](../../../docs/operations/aiops-exadata-oem-implementation-design.md)。

- KBot 已部署 `/home/ubuntu/kbot4.0`，`kbot4.1.1@4aa7f4ce`；Ammolite 已部署 `/home/ubuntu/Ammolite`，`dev@51174b7b`。
- KBot 先完整核对已有对象，再用规范 SQL 新增两表，最后同步版本视图为 Schema 60；没有直接套用不匹配的 59→60 脚本。完整 Oracle 目录验收通过。Ammolite 仅用规范 Entity 定向新增两表，既存知识库 Schema 漂移未处理，不宣称全库验收通过。
- 仅重启 KBot AIOps API（18110）、Ammolite AIOps API（19130）、Ammolite Main API（19090），各自实际 readiness 为 200。其余服务、Portal、既有 OEM 和数据库 Agent 均未重启。配置及对象元数据备份保留在 kbotdev 的私有 `/home/ubuntu/aiops-exadata-deploy-20261111-6rj5zwyh`，会话/凭据不入库。
- 两端沿用既有 `OEM24ai` 来源及凭据；仅用官方 EMCLI 给现用监控账号增加三个固定 SIM 目标的 `VIEW_TARGET`，不授管理员或全主机权限，不改既有目标权限。

KBot 正常登录后进入[实时监控](http://140.238.44.208:8080/ui/aiops/monitoring.html?target_id=01a1ca77-0b23-7cb1-84de-5c806521749a&businessSystemId=01a1ca77-0ad7-7f92-8cd4-30224c93e76b&source_id=01a1a702-71fe-7793-8e47-8cf23f290df8)，选择业务系统 `Exadata OEM 模拟联调（SIM）` → 硬件 `SIM - Exadata DBM01` → 来源 `OEM24ai`。专用场景选“Cell 资源”，可切换“最新快照”和“过去 1 小时历史（仅数值）”。该系统经公共 API 原子创建，未改旧系统/绑定，无直接写业务表或伪造鉴权。

浏览器核验时 CPU 47.95%、内存 64.32%，采样时间 2026-11-11 11:42 UTC；CPU/内存历史各 45 个样本。另经正常公共诊断 API 保存并读回报告，证据一致、结论为 `INCONCLUSIVE`；未将其冒充诊断按钮浏览器验收。所有值均为合成调试数据；OEM 批次上传仍有延迟，不能承诺秒级或一分钟页面更新。

Ammolite Portal 为 `http://140.238.44.208:19080/`，监控路由 `/apps/aiops/:domainCode/monitoring`。其部署进程使用现有托管 OEM 凭据已读到 5 项概览、12 项 Cell 最新值及 CPU/内存各 44 个历史样本，但尚无正常 App 登录会话；未创建模拟系统/绑定，未验证公共 App API 或浏览器数据展示。需通过正常登录并获得领域授权后，在硬件层新建 EXADATA、选择现有 OEM 来源、发现 DBM01 并确认 Grid/Cell 成员；不得重置密码、伪造 SSO 或直接写绑定表。

## 限制与保留现场

已实现 Cell/PhysicalDisk/CellDisk/FlashCache/FlashLog/GridDisk/三类 IORM/Cell 互联/Filesystem 的只读子集，覆盖仍以真实 OEM 已留存结果为准；不声明多 Cell、交换机、数据库节点、ASM 容量及全部状态规则已覆盖。OEM 目录、维度和采样仍由真实插件决定，未支持项必须作为缺口。

早期未采用的 REST Cell 原型 `aiops-exadata-sim`、Remote Agent 原型 `aiops-exadata-sim-agent` 已停止，保留私有 state/卷供排障；仓库只保留最终 CellCLI 流程，不将失败凭据路径作为部署方案。既有 `aiops-oem-remote-agent` 与这些新建原型不同，不得停止、重装或清理它。

本次没有删除 OEM 目标、卷或 state，没有改 OEM 内部表，没有重启既有 OEM/数据库 Agent。若要清理，仅在核对 SIM 名称、标签、卷和原厂成员关系后逐项处理；不得对整个实验根目录或 Docker 环境进行批量清理。
