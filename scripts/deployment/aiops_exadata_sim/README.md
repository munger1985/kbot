# Exadata → 真实 OEM 调试夹具

本目录只在隔离实验环境模拟 CellCLI。OEM 使用未修改的 XA 插件采集、上传并保存指标，AIOps 仍访问真实 OEM REST，不直读模拟器，不写 OEM 内部表。

全部名称含 SIM，全部数值是合成数据。字段映射依据真实 OEM 导出目录及 XA 24.1.1 原厂采集脚本；不包含客户主机、凭据或真实业务值，也不承诺覆盖目录全部 267 项。

## 现场状态（2026-11-11）

- OEM：既有 `aiops-oem`，OMS/Agent 24.1.0.0.0、XA 24.1.1.0.0。
- 新标准 Agent：`aiops-exadata-sim-monitor`，主机 `aiops-exadata-monitor.lab`，独立卷 `aiops_exadata_sim_monitor_data`。
- 新 CellCLI：`aiops-exadata-cellcli-sim`，主机 `aiops-exadata-cellcli.lab`，实验网络地址 `172.22.0.8`；无宿主端口发布。
- 原厂目标：`AIOPS_SIM_EXADATA_DBM01`（oracle_dbmachine）、`AIOPS_SIM_EXADATA_GRID01`（oracle_exadata_grid）、`AIOPS_SIM_EXADATA_CELL01`（oracle_exadata）。
- 原厂允许的结构是 DB Machine → Grid → Cell，不能直接把 Cell 加入 DB Machine。三者通过原厂 create/modify_system 命令建立，现已通过官方 Data Service 包含关系和 GUID 目录双重独立核验，并与原厂 `get_system_members -depth=2` 结果一致。
- overview 的 5 项最新值、cell 的 12 项最新值以及 CPU/内存数值历史已被实际 AIOps OEMAdapter 读取。采样不足时不能推断趋势。
- KBot/Ammolite 分别使用本仓库 Adapter 完成严格整机 latest、cell、history smoke，输出 `inventory_verified=true`；CPU/内存各读到 40 个历史样本。两端各 10 个相关测试文件共 **291 passed + 13 subtests passed**（其中新增成员关系安全合同 38 项、模拟器/SSH/注册安全合同 8 项、Exadata/HTTP 合同 47 项、既有普通 OEM/资源监控回归 198 项）；两端架构边界检查通过。不是全仓测试或页面部署通过声明。
- 一分钟调度已在 Agent 集合配置落地；手工仅刷新上传后，OEM 历史包含 09:49、09:50、09:51 UTC 的连续自动采样，CPU 为 26.0/26.2/26.88%，内存为 57.0/57.07/57.29%，不是只保存一条手工采样。上传批次延迟仍存在。
- **整机发现已修复**：初版 `/members` 为 404；现统一使用官方 GET `/em/websvcs/restful/emws/core/v0/list/dataservice`，固定资源 `TargetAssociations` 和 `internal_contains`，每个父目标精确过滤，要求 `exceedsMaxRows=false`，再逐项查询唯一名称/类型对应的 GUID 并复核详情。只展开根及直接 Grid；歧义、截断、未应用过滤、循环和超预算均失败关闭。没有双路径降级，没有 SQL、直连 OEM 仓库或额外服务。
- **页面未验收**：未部署/重启 AIOps、未应用 AIOps Schema、未新增业务系统资源绑定。不得把 `--cell-only` 的通过说成页面或整机验收。

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

## 页面联调前提

已确认应用环境为 kbotdev（私网 `10.0.0.190`），OEM 位于 aiopsdb（私网 `10.0.0.88`）。2026-11-11 只读核查结果：

- KBot 运行目录 `/home/ubuntu/kbot4.0`，分支 `kbot4.1.1`、代码 `6cc14557`；已包含 Exadata 基础功能，尚无本次成员读取修复。监控页面 `http://140.238.44.208:8080/ui/aiops/monitoring.html` 返回 HTML 200，仅证明静态页面可达，不是登录后数据验收。
- Ammolite 运行目录 `/home/ubuntu/Ammolite`，分支 `dev`、代码 `109138b2`；尚未包含 Exadata 基础功能。Portal `http://140.238.44.208:19080/` 返回 HTML 200；实际监控路由为 `/apps/aiops/:domainCode/monitoring`，需要正常登录及领域授权。
- 两端均已有启用且连接状态为 CONNECTED 的 `OEM24ai` 来源，`aiops-oem.lab` 在应用主机解析为 `10.0.0.88`；不新增或覆盖既有来源凭据。两端均无 EXADATA 资源和 Exadata 专用两表。
- KBot 版本视图返回 AIOPS / Schema 57，不能直接运行仅适用于完整 Schema 59 的 `apply_aiops_schema_60.sql`。必须先做部署期只读差异核对，再确定经授权的无损处理范围；不得仅抬高版本视图绕过前置条件。Ammolite 采用规范 Entity/初始化流程处理新增表，不重建或清空现有库。

本轮未拉取代码、未部署或重启应用、未执行 DDL、未修改来源和绑定。应用更新、必要数据库处理及定向重启需先取得明确授权。部署条件满足后，在正常业务系统硬件层新增 EXADATA，选择既有 OEM 来源，发现并选择 `AIOPS_SIM_EXADATA_DBM01`、确认 Grid/Cell 成员后保存，再验收页面最新值和历史。不得绕过正常授权、套用不匹配的升级脚本或复用普通主机绑定。

## 限制与保留现场

只实现 Cell/PhysicalDisk/CellDisk 及部分 CPU、内存、网络、磁盘、Flash、SmartIO 原生字段；不声明 IORM、多 Cell、交换机、数据库节点、容量及全部状态规则已覆盖。OEM 目录、维度和采样仍由真实插件决定，未支持项必须作为缺口。

早期未采用的 REST Cell 原型 `aiops-exadata-sim`、Remote Agent 原型 `aiops-exadata-sim-agent` 已停止，保留私有 state/卷供排障；仓库只保留最终 CellCLI 流程，不将失败凭据路径作为部署方案。既有 `aiops-oem-remote-agent` 与这些新建原型不同，不得停止、重装或清理它。

本次没有删除 OEM 目标、卷或 state，没有改 OEM 内部表，没有重启既有 OEM/数据库 Agent。若要清理，仅在核对 SIM 名称、标签、卷和原厂成员关系后逐项处理；不得对整个实验根目录或 Docker 环境进行批量清理。
