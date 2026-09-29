# AIOps 数据库实施文档中心详细设计

版本：1.1
状态：已实施
基准日期：2026-09-28

实施结果：Runbook v3、Profile Registry、统一校验器、Turn Artifact、JSON/ZIP 下载以及本文定义的
九个新增实施档案均已进入代码；历史 v1/v2 ADG 文档继续按固化 Block 展示和导出。

## 1. 目标与范围

本文定义 ADG 之后的数据库实施文档能力，使 KBot 能根据当前数据库、主机、基础设施和运维策略事实，
确定性生成完整、可审核、可下载的实施操作文档与脚本包。

目标能力包括：

1. Oracle RAC 从零建设及现有数据库迁移；
2. RMAN 备份体系建设和可直接部署的日常脚本；
3. RMAN 恢复、时间点恢复和灾难恢复操作文档；
4. GI/数据库 RU 补丁；
5. 数据库跨版本升级；
6. 数据库迁移；
7. 测试库克隆和刷新；
8. Data Pump 逻辑迁移；
9. ADG Switchover、Failover 和 Reinstate 演练文档。

这些能力统一属于 `IMPLEMENTATION_RUNBOOK`，不属于普通 Advisory，也不表示系统立即执行变更。
模型只识别实施档案；前置取证、参数解析、命令、脚本、验证和回退均由版本控制的确定性组件生成。

## 2. 核心设计决策

### 2.1 一个文档中心，多套独立实施档案

所有档案复用当前 `ImplementationRunbook` 展示和下载框架，但每个档案必须拥有独立的：

- `ImplementationProfile`；
- 前置 Tool 和 Playbook；
- 归一化事实模型；
- 参数解析器；
- 确定性编译器；
- 脚本和配置文件生成器；
- 验证、回退和停止条件；
- 单元、契约和 Golden File 测试。

禁止建立一个由模型自由扩写命令的“通用实施方案生成器”。

### 2.2 结构化 Runbook 是唯一真相

权威链路保持为：

```text
数据库/主机/拓扑/策略事实
  → 归一化实施事实
  → 确定性 Runbook Compiler
  → ImplementationRunbook JSON
      ├─ 页面正式文档渲染
      ├─ PDF 下载
      ├─ Markdown 下载
      └─ Scripts ZIP 下载
```

结构化 JSON 只作为程序内部唯一真相，不提供用户下载。PDF、Markdown 和脚本包都是结构化
Runbook 的派生产物，不重复持久化另一份可漂移的正文。

### 2.3 零聊天补参不等于伪造基础设施

数据库名称、路径、版本、Oracle Home、实例名等可以从 Target 事实确定或安全派生；IP、VIP、SCAN、
共享磁盘 WWID 和补丁介质路径不能从数据库参数安全猜测。

系统不得输出 `<NODE2_IP>`、`${SCAN_NAME}` 等伪可执行占位符，也不得编造网络地址或磁盘设备。
这些值必须来自 Target 运维事实、主机自动发现或已登记的部署拓扑。缺少事实时：

- 仍生成完整阶段、判断、验证和数据库侧内容；
- 依赖真实基础设施值的步骤标记为 `BLOCKED`；
- 不在聊天中反复要求用户逐项输入；
- 在 Target 配置中引导登记部署拓扑，重新生成后转为 `READY`。

RAC 从零建设的 OS、网络检查、共享盘发现、GI 安装和 ASM 初建不适用上述阻断方式。该 Profile
假设用户已把 Oracle GI grid home ZIP 放入策略约定的 `/stage/oracle`，按源库版本派生标准
`GRID_HOME`，直接输出两个节点的预安装包、用户组、目录、GI 解压命令，并调用安装包自带的
`gridSetup.sh`、`root.sh`、`cluvfy`、`crsctl`、`olsnodes`、`srvctl` 和 `asmca`。节点、VIP、SCAN、
网卡用途和 ASM 设备在 Oracle 安装器的交互及预检查过程中登记，不得因此把这些建设阶段标记为
`BLOCKED`；只有后续数据库资源注册或业务服务发布确实需要固化值时才允许阻断相应后续步骤。

### 2.4 文档生成与实际执行继续分离

Runbook 阶段不创建 Proposal，不执行脚本。用户明确要求执行某一步时，必须重新核验实时状态，并把
该步骤映射到已登记 Action Template。Restore、Failover、共享磁盘初始化、GI 安装等高风险动作保持
人工执行或独立受控流程，不能因为命令来自 Runbook 就绕过审批。

## 3. 实施档案目录

| 优先级 | `ImplementationProfile` | 默认业务场景 | 主要交付物 |
| --- | --- | --- | --- |
| P0 | `ORACLE_RAC_BUILD` | 当前单实例迁移到全新两节点 RAC | RAC 文档、响应文件、Shell/SQL/RMAN 脚本 |
| P0 | `ORACLE_RMAN_BACKUP_BUILD` | 建设标准 RMAN 备份体系 | 策略文档、RMAN 脚本、调度和监控脚本 |
| P0 | `ORACLE_RMAN_RECOVERY` | 恢复演练和故障恢复 | 场景化恢复文档、校验脚本、人工检查表 |
| P1 | `ORACLE_RU_PATCH` | GI 和数据库 Home 安装批准 RU/OJVM | 补丁文档、静默响应、验证和回退 |
| P1 | `ORACLE_DATABASE_UPGRADE` | 升级到平台批准目标版本 | 预检查、升级、组件验证和回退 |
| P1 | `ORACLE_DATABASE_MIGRATION` | 迁移到已登记目标环境 | 方法选择、预同步、切换和校验 |
| P2 | `ORACLE_CLONE_REFRESH` | 生产到测试的克隆/刷新 | 克隆脚本、重命名、脱敏交接清单 |
| P2 | `ORACLE_DATAPUMP_MIGRATION` | Schema 或 PDB 逻辑迁移 | parfile、目录、导入导出和校验脚本 |
| P2 | `ORACLE_ADG_DRILL` | 已有 ADG 环境的切换演练 | Precheck、Switchover/Failover/Reinstate 文档 |

`ORACLE_RMAN_BACKUP_BUILD` 和 `ORACLE_RMAN_RECOVERY` 必须分离。前者是可重复执行的日常运维建设，
后者包含破坏性恢复流程、时间点选择和业务确认，风险等级和执行边界完全不同。

## 4. 通用契约详细设计

目标结构化版本为 `AIOPS_IMPLEMENTATION_RUNBOOK.v3`。新生成只写 v3，不双写旧版本；PDF、Markdown
和页面继续读取历史 v1/v2 Block，保证已生成文档可以下载。

### 4.1 `ImplementationProfile`

目标枚举：

```text
NONE
ORACLE_ADG_BUILD
ORACLE_RAC_BUILD
ORACLE_RMAN_BACKUP_BUILD
ORACLE_RMAN_RECOVERY
ORACLE_RU_PATCH
ORACLE_DATABASE_UPGRADE
ORACLE_DATABASE_MIGRATION
ORACLE_CLONE_REFRESH
ORACLE_DATAPUMP_MIGRATION
ORACLE_ADG_DRILL
```

规划器只选择枚举值，不生成拓扑、路径、版本、脚本或命令参数。服务端 Registry 负责把 Profile 映射到
固定 Playbook、事实模型和 Compiler。

### 4.2 `RunbookCommand` 扩展

现有字段继续保留，新增：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `executor` | Enum | `SQLPLUS`、`RMAN`、`DGMGRL`、`BASH`、`CRSCTL`、`SRVCTL`、`ASMCMD`、`DBCA`、`OPATCH`、`DATAPUMP`、`MANUAL` |
| `run_as` | String | `root`、`grid`、`oracle`、`SYSDBA` 等执行身份 |
| `node_scope` | tuple[String] | `source`、`node1`、`node2`、`all_rac_nodes` 等逻辑节点 |
| `container_name` | String/null | `CDB$ROOT` 或确定的 PDB 名称 |
| `working_directory` | String/null | 命令运行目录 |
| `target_path` | String/null | 配置文件或脚本落盘位置 |
| `expected_result` | tuple[String] | 执行后必须观察到的结果 |
| `artifact_ref` | String/null | 命令对应的脚本或配置 Artifact |
| `risk_level` | Enum | `LOW`、`MEDIUM`、`HIGH`、`CRITICAL` |

`command_type` 保留用于兼容历史 Runbook；新 Compiler 同时填充更精确的 `executor`。

### 4.3 `RunbookArtifactDescriptor`

Runbook 新增脚本和配置文件描述契约：

```text
artifact_id
file_name
relative_path
media_type
sha256
file_mode
run_as
target_path
description
contains_secret=false
```

约束：

- 只允许 UTF-8 文本脚本和配置；
- 禁止包含密码、私钥、Wallet 内容和 SBT 凭据；
- `sha256` 由服务端按最终字节计算；
- PDF 和 Markdown 引用 `file_name`、目标路径和 SHA256；
- ZIP 按 `relative_path` 确定性排序，同一 Runbook 重复下载字节一致。

脚本正文作为现有 Turn Artifact 的 UTF-8 Payload 保存；Answer Block 只保存
`RunbookArtifactDescriptor`，避免数据库行无限增长和正文双写。

### 4.4 部署拓扑事实

新增 Target 级 `DeploymentTopologyFact`，通过 Target 运维记忆或独立配置页维护：

```text
topology_kind
environment
node_name
host_name
public_ip
vip_name
vip_ip
private_host_name
private_ip
scan_name
scan_ips
network_interface
interconnect_interface
asm_disks
disk_groups
grid_base
grid_home
oracle_base
oracle_home
inventory_location
patch_stage_path
source
verified_at
```

数据库 Target 不直接拥有第二节点时，RAC Compiler 从同一部署拓扑中的逻辑节点读取事实。聊天会话不
临时保存基础设施地址，避免同一 Target 在不同 Turn 中产生互相矛盾的命令。

持久化复用现有 Target Fact/运维记忆能力，以版本化 JSON Fact 保存，不新增一套平行拓扑表。每次
生成 Runbook 时把使用到的 Fact 值、来源和时间固化到参数附录，后续修改 Target Fact 不改写历史文档。

### 4.5 策略模板

不能从数据库推导的业务选择由版本化策略模板提供，而不是模型猜测：

| 模板 | 关键默认值 |
| --- | --- |
| `oracle.rac.standard-2node.v1` | 两节点、ASM、SCAN、独立 Grid/DB Home、现有数据库迁移 |
| `oracle.rman.standard-disk.v1` | 周 L0、日 L1 Cumulative、小时归档、7 天恢复窗口 |
| `oracle.patch.approved-ru.v1` | 使用平台批准 RU，不自动选择互联网最新版本 |
| `oracle.upgrade.approved-target.v1` | 目标版本来自平台批准矩阵 |
| `oracle.migration.standard.v1` | 根据停机窗口、平台和许可选择确定方法 |

模板版本、来源和覆盖值进入 `resolved_parameters`，状态标记为 `DERIVED`。

非秘密基线模板放在版本控制的实施策略目录并校验 Hash；Domain/环境选择哪一个模板由 AIOps 配置
保存。批准 RU、升级目标版本等经常变化的值只存配置，不写入 Prompt 或 Compiler 源码。

### 4.6 `ImplementationRequestContext`

RAC、恢复、迁移和克隆需要比 Profile 更具体的结构化范围。Task Frame 新增：

```text
profile
primary_target_id
destination_ref
policy_template_id
scenario
requested_target_version
recovery_target_time
recovery_target_scn
explicit_options
```

规则：

- `primary_target_id` 仍是 Conversation 冻结的单一主 Target；
- `destination_ref` 只能引用用户有权读取的 Target 或已登记部署拓扑，不开放任意跨 Target 调查；
- Planner 只能提取用户明确提供的恢复时间、目标版本、目标环境等值，不能补全关键业务参数；
- 用户没有明确选择时，服务端使用版本化策略模板；没有可用策略时标记缺失事实；
- Context 与最终参数表一起固化，历史 Runbook 不随 Target 或策略变更。

### 4.7 缺失事实与状态

新契约增加 `missing_facts`，每项包含 `fact_key`、`resolution_source`、`reason` 和 `blocking_steps`。
`resolution_source` 只能是 `TARGET_FACT`、`DEPLOYMENT_TOPOLOGY`、`HOST_COLLECTOR`、`POLICY_TEMPLATE`
或 `EXPLICIT_USER_DECISION`。

目标状态增加 `BLOCKED_BY_REQUIRED_FACTS`。历史 `BLOCKED_BY_REQUIRED_INPUTS` 仍可渲染，但新档案不把
基础设施事实包装成聊天输入请求。页面直接显示应在哪个 Target/拓扑/策略位置补齐。

## 5. 通用生成流程

```text
用户要求生成实施文档
  → Planner 选择 ImplementationProfile
  → 服务端 Registry 选择固定 Playbook
  → 采集数据库、主机、拓扑和策略事实
  → Fact Normalizer 统一单位、名称和来源
  → Parameter Resolver 生成 VERIFIED/DERIVED 参数
  → Profile Compiler 生成阶段、步骤、命令、脚本和停止条件
  → Contract Validator 拒绝占位符、秘密和未引用 Artifact
  → 固化 IMPLEMENTATION_RUNBOOK Answer Block 与 Artifact
  → 页面/PDF/Markdown/JSON/ZIP 投影
```

每个 Compiler 必须在证据缺失时仍生成完整阶段，但不能把未知事实描述成已确认，也不能生成包含未知值
的伪命令。通用编译器不得为缺少专用命令的阶段自动生成中文 `MANUAL` 兜底；阻断步骤的命令集合
必须为空，真正的人工决策门禁由投影层独立显示为“人工确认项”。

## 6. `ORACLE_RAC_BUILD` 详细设计

### 6.1 默认范围

默认场景为：

> 以当前单实例 Oracle CDB/NON-CDB 为源库，在已登记的全新两节点基础设施上建设 RAC，并通过 RMAN
> Backup/Restore 或 Active Duplicate 把当前数据库迁移到 RAC。

规则：

- 当前 Target 是 PDB 时，通过已登记父 CDB Target 关系提升到 CDB；没有父 CDB 关系时标记缺失事实，
  不使用 PDB 连接生成 RAC 文档；RAC 不是 PDB 级部署能力；
- 当前数据库已是 RAC 时，不生成“从单实例迁移”文档，后续由独立节点扩容档案处理；
- GI 和 DB Home 版本与当前数据库兼容，RU 使用平台批准版本；
- 目标使用 ASM/OMF，默认 `+DATA` 和 `+RECO`；OCR/Voting 的磁盘组按拓扑事实决定；
- 不采用在生产主机上直接把原实例原地改成 RAC 的默认路径。

### 6.2 前置 Tool 与 Playbook

新增 Playbook：`oracle.ha.rac_build@1.0.0`。

固定 Tool：

| Tool | 作用 |
| --- | --- |
| `db.instance.identity` | 数据库版本、实例、角色和时间基准 |
| `db.ha.rac_precheck` | RAC 参数、实例、线程、Undo、Redo、服务和文件布局 |
| `db.storage.database_footprint` | 数据库、FRA、归档和临时文件容量 |
| `db.backup.rman_configuration` | RMAN 配置和可迁移备份能力 |
| `host.oracle.inventory` | Oracle/Grid Home、Inventory、RU 和 OS 用户组 |
| `host.rac.precheck` | CPU、内存、内核、包、limits、HugePages、时间同步 |
| `host.network.rac_topology` | 公网、VIP、私网、SCAN、DNS、MTU 和互通性 |
| `host.storage.asm_candidates` | Multipath、WWID、UDEV 和 ASM 候选盘 |

主机 Tool 必须由后续 Host Runner 或受控 SSH Collector 提供，不能通过 DB Executor 执行 Shell。

### 6.3 归一化事实

RAC Compiler 至少需要：

- 数据库名、`DB_UNIQUE_NAME`、CDB/PDB、版本、RU、字符集；
- `cluster_database`、实例数量、线程、Undo、Redo 和服务；
- 数据库总量、最大文件、FRA 和归档生成速度；
- 源端 Oracle Home、密码文件、SPFILE、监听和服务；
- 两节点主机名、公网、VIP、私网和 SCAN；
- CPU、内存、HugePages、内核、用户组和时钟同步；
- 共享磁盘 WWID、容量、冗余和目标 ASM Diskgroup；
- Grid Base/Home、Oracle Base/Home、Inventory 和补丁目录；
- 目标迁移方法和实施窗口策略。

### 6.4 参数派生

可确定性派生：

- RAC 数据库唯一名默认沿用源 `DB_UNIQUE_NAME`；
- 实例名为 `<DB_NAME>1`、`<DB_NAME>2`，遵循 Oracle 长度限制；
- Undo 为 `UNDOTBS1`、`UNDOTBS2`；
- 每个实例独立 Redo Thread；
- Oracle Home 与版本从源端和批准策略派生；
- 服务名从当前服务清单生成 RAC 服务，并绑定首选/可用实例；
- ASM 路径根据 `+DATA`、`+RECO` 和 OMF 生成，不生成文件名转换占位符；
- 密码文件目标使用 ASM 或集群标准路径；
- RMAN 并行度根据数据库大小、CPU 和存储策略分档确定。

禁止派生 IP、VIP、SCAN IP、WWID 和实际补丁文件名；这些必须为 `VERIFIED` 拓扑事实。

### 6.5 文档阶段

| 阶段 | 内容 | 主要执行器 |
| --- | --- | --- |
| 1. 范围与拓扑 | 当前状态、目标节点、版本、路径、网络和存储摘要 | MANUAL |
| 2. 源库保护 | 备份、恢复点、归档、强制日志和回退边界 | SQLPLUS/RMAN |
| 3. 操作系统准备 | 用户组、目录、包、内核、limits、HugePages | BASH |
| 4. 网络与名称解析 | 公网、VIP、私网、SCAN、DNS、MTU、连通性 | BASH |
| 5. 共享存储 | Multipath、UDEV、ASM Label/Filter、容量验证 | BASH/ASMCMD |
| 6. Grid 安装 | 响应文件、root 脚本、CRS 和资源验证 | BASH/CRSCTL |
| 7. ASM 建设 | Diskgroup、兼容性、容量和挂载验证 | ASMCMD/SQLPLUS |
| 8. DB Home 安装 | 软件、RU/OJVM、Inventory 和 OPatch 验证 | BASH/OPATCH |
| 9. 数据库迁移 | RMAN Restore/Duplicate、控制文件和数据文件恢复 | RMAN |
| 10. RAC 化 | `cluster_database`、线程、Undo、Redo、实例参数 | SQLPLUS |
| 11. 集群注册 | `srvctl add database/instance/service` | SRVCTL |
| 12. CDB/PDB 服务 | PDB 保存状态、服务和启动策略 | SQLPLUS/SRVCTL |
| 13. 验收 | 实例、CRS、ASM、服务、故障漂移和性能基线 | CRSCTL/SRVCTL/SQLPLUS |
| 14. 运维与回退 | 启停、节点维护、备份接管和回退步骤 | 多执行器 |

### 6.6 脚本包

```text
oracle-rac-build/
├── 00-manifest.json
├── 01-topology-summary.md
├── os/
│   ├── node1-root-precheck.sh
│   ├── node2-root-precheck.sh
│   ├── configure-kernel-limits.sh
│   ├── configure-network.sh
│   └── configure-asm-devices.sh
├── grid/
│   ├── gridsetup.rsp
│   ├── install-grid.sh
│   └── verify-cluster.sh
├── database/
│   ├── db_install.rsp
│   ├── prepare-source.sql
│   ├── restore-to-rac.rman
│   ├── configure-rac.sql
│   ├── register-resources.sh
│   └── configure-services.sh
└── validation/
    ├── validate-rac.sql
    ├── validate-services.sh
    └── failover-test-checklist.md
```

响应文件不得包含密码；需要密码的安装步骤通过运行时安全输入或 Oracle 支持的钱包/凭据机制完成。

### 6.7 状态和停止条件

`READY` 要求网络、节点、共享存储和版本事实齐全。以下情况必须停止：

- 源库版本、字符集或组件与目标版本不兼容；
- SCAN/VIP/DNS 解析或私网互通失败；
- 两节点时间偏差、MTU 或内核配置不一致；
- 共享磁盘 WWID、容量或多路径映射不一致；
- OCR/Voting 或 `+DATA`/`+RECO` 容量不足；
- 源库无可验证备份或无法建立恢复点；
- GI、DB Home、RU 或 OPatch 版本不一致；
- 实际主机、设备或路径与 Runbook 固化事实不一致。

## 7. `ORACLE_RMAN_BACKUP_BUILD` 详细设计

### 7.1 默认策略

未配置专用策略时使用 `oracle.rman.standard-disk.v1`：

- 每周一次 Level 0；
- 每日一次 Level 1 Cumulative；
- 每小时归档日志备份；
- 控制文件和 SPFILE 自动备份；
- 7 天 Recovery Window；
- 每日 Crosscheck 和过期记录清理；
- 每周 `RESTORE DATABASE VALIDATE`；
- 每月执行一次隔离环境恢复演练；
- Systemd Timer 优先，目标系统不支持时使用 Cron；
- Disk Channel 为默认类型，只有已验证 SBT 配置时才生成 SBT 分支。

这些值必须在参数附录中标记为策略派生，而不是伪装成用户确认值。

### 7.2 前置 Tool

新增 Playbook：`oracle.backup.rman_build@1.0.0`。

| Tool | 作用 |
| --- | --- |
| `db.instance.identity` | 数据库和版本身份 |
| `db.backup.recent_jobs` | 最近任务、成功率、耗时和压缩比 |
| `db.backup.rman_configuration` | `V$RMAN_CONFIGURATION` 与当前策略 |
| `db.backup.inventory_summary` | Backup Set/Piece、控制文件和归档覆盖 |
| `db.storage.database_footprint` | 数据量、FRA、数据文件和临时文件规模 |
| `db.archive.generation_history` | 最近 7/30 天归档生成量与峰值 |
| `db.recovery.capabilities` | ARCHIVELOG、BCT、Flashback、TDE 和 CDB/PDB |
| `host.backup.destination` | 备份路径容量、挂载、权限和文件系统类型 |

### 7.3 策略计算

备份空间建议按以下事实计算并在文档中展示：

```text
Level 0 需求 = 数据库已分配量 × 估算压缩系数
Level 1 需求 = 日变化量 × 保留天数
归档需求    = P95 每日归档量 × 保留天数
校验余量    = 上述总量 × 20%
```

没有历史变化率时不伪造压缩比和日增量，使用未压缩数据库分配量作为保守基线并标记
`PARTIAL_EVIDENCE`。

并行度依据 CPU、数据库大小和目标存储吞吐策略分档，不直接使用固定数字。压缩、加密和 SBT 只有在
许可、Wallet 和介质管理事实已确认时才启用。

### 7.4 文档阶段

1. 当前备份状态和恢复风险；
2. 目标 RPO/RTO 与默认策略来源；
3. ARCHIVELOG、BCT、FRA 和控制文件自动备份整改；
4. RMAN `CONFIGURE` 基线；
5. Level 0、Level 1 和归档备份脚本；
6. Crosscheck、Catalog 和清理脚本；
7. 数据库、控制文件、SPFILE 和归档校验；
8. Systemd/Cron 调度；
9. 日志、退出码和监控接入；
10. 恢复演练计划；
11. 容量、失败处理和日常运维命令。

### 7.5 脚本包

```text
oracle-rman-backup/
├── 00-manifest.json
├── env.conf
├── rman/
│   ├── configure.rman
│   ├── backup_level0.rman
│   ├── backup_level1.rman
│   ├── backup_archivelog.rman
│   ├── backup_controlfile_spfile.rman
│   ├── crosscheck_cleanup.rman
│   ├── validate_database.rman
│   └── restore_validate.rman
├── bin/
│   ├── run-rman-job.sh
│   ├── check-last-backup.sh
│   └── install-schedule.sh
├── systemd/
│   ├── oracle-rman@.service
│   ├── oracle-rman-level0.timer
│   ├── oracle-rman-level1.timer
│   └── oracle-rman-archivelog.timer
└── monitoring/
    ├── prometheus-textfile.sh
    └── alert-rules.yml
```

`env.conf` 只保存 SID、备份目录和日志路径，不保存 SYS 密码。默认通过本机 OS 认证执行
`rman target /`。当前实现不要求用户提供 `ORACLE_HOME`：脚本使用 `ORACLE_SID` 调用主机
`oraenv` 初始化 Oracle 环境，`oraenv` 不可用时从对应 PMON 进程解析 Home。备份目录按显式 Target 配置、现有 RMAN Disk Channel FORMAT、文件系统
FRA 的顺序复用；均不存在时按 `DB_UNIQUE_NAME` 派生 `/u01/app/oracle/backup/<db_unique_name>`，先生成
目录创建与 `CONFIGURE CHANNEL DEVICE TYPE DISK FORMAT` 命令，并提示在实施前按真实独立挂载点调整。

### 7.6 脚本行为规范

- Shell 使用 `set -euo pipefail`；
- 每次任务生成独立日志、开始时间、结束时间和退出码；
- 同类备份使用锁文件防止并发；
- RMAN 错误码和 ORA 错误导致非零退出；
- 清理脚本只执行由策略确定的 `DELETE OBSOLETE`，不执行任意日期删除；
- `DELETE INPUT` 只能在备份成功且归档保留策略满足时使用；
- 每个脚本先输出数据库身份和当前角色，角色不符立即停止；
- Data Guard 环境必须根据备份职责策略决定在哪个角色执行。

## 8. `ORACLE_RMAN_RECOVERY` 详细设计

恢复文档不作为备份脚本的附录，按场景独立生成：

| 场景 | 交付范围 |
| --- | --- |
| Controlfile/SPFILE 丢失 | Autobackup 定位、启动状态、恢复和验证 |
| 单数据文件恢复 | Offline/Restore/Recover/Online 和对象验证 |
| 全库恢复 | 控制文件、Catalog、Restore、Recover、Open |
| 时间点恢复 | SCN/时间确认、恢复点、`RESETLOGS` 和后续备份 |
| PDB PITR | Auxiliary Destination、PDB 状态和恢复验证 |
| ASM/主机灾难恢复 | 新环境目录、密码文件、控制文件和全库重建 |

每份恢复文档必须冻结恢复目标时间/SCN、可用备份链和归档覆盖；未确认恢复目标时只生成演练模板，
不得输出可直接执行的 `SET UNTIL`。所有恢复档案默认 `MANUAL_ONLY`，不提供聊天内自动执行按钮。

## 9. 其他常用档案详细边界

### 9.1 `ORACLE_RU_PATCH`

- 区分单实例、RAC、GI 和 DB Home；
- 补丁版本只来自平台批准补丁目录；
- 检查冲突、空间、OPatch、Inventory、Data Guard 和 RAC Rolling 能力；
- 输出 `opatchauto`/`opatch`、停启、`datapatch`、组件验证和回退；
- 补丁介质不存在时标记 `BLOCKED`，不伪造文件名。

### 9.2 `ORACLE_DATABASE_UPGRADE`

- 目标版本来自批准升级矩阵；
- 采集组件、字符集、时区、参数、无效对象、PDB、应用容器和选件；
- 使用 AutoUpgrade 为默认路径，生成配置文件和 analyze/fixups/deploy 流程；
- 输出升级前备份、Guaranteed Restore Point、升级、`datapatch`、组件和业务验证；
- 不支持的跨平台或直接跨版本路径转入迁移档案。

### 9.3 `ORACLE_DATABASE_MIGRATION`

确定性方法选择顺序：

1. 同平台、停机可接受：RMAN Backup/Restore；
2. 同平台、低停机：Data Guard 或 RMAN 增量滚动；
3. 跨平台大库：XTTS；
4. Schema/PDB 级：Data Pump；
5. 异构或近零停机：只有已确认 GoldenGate 许可时才选择 GoldenGate。

目标平台、版本、字符集、字节序、停机窗口和许可事实缺失时输出方法比较，不声称已选定可执行路径。

### 9.4 `ORACLE_CLONE_REFRESH`

- 默认源为生产、目标为已登记非生产 Target；
- 支持 RMAN Duplicate、PDB Clone 和 Snapshot Copy 的确定性选择；
- 输出数据库重命名、服务隔离、DB Link/Job 禁用和脱敏交接点；
- KBot 不生成业务脱敏规则，脱敏包必须来自已批准 Artifact。

### 9.5 `ORACLE_DATAPUMP_MIGRATION`

- 生成 `expdp`/`impdp` parfile，不把超长参数放在单行 Shell；
- 读取对象量、LOB、分区、目录、字符集、时区和无效对象；
- 输出表空间映射、Schema 映射、统计信息、对象计数和无效对象验证；
- 密码通过 Wallet 或运行时安全输入，不写入 parfile。

### 9.6 `ORACLE_ADG_DRILL`

- 仅适用于已验证 Broker 配置；
- 区分 Switchover、计划 Failover、非计划 Failover 和 Reinstate；
- 默认只生成演练文档，不注册自动执行模板；
- 必须包含应用停写、连接排空、日志零缺口、业务验证和回切计划；
- 任一成员存在错误、延迟超标或 Flashback 不满足时停止。

## 10. 模块和文件规划

现有 `application/implementation/runbooks.py` 已包含完整 ADG/DGPDB 编译器。扩展前应按档案拆分，
不再继续把所有 Compiler 堆入单文件：

```text
services/aiops_agent/src/aiops_agent/application/implementation/
├── registry.py
├── facts.py
├── parameters.py
├── artifacts.py
├── validation.py
├── markdown.py
├── pdf.py
└── profiles/
    ├── adg.py
    ├── rac.py
    ├── rman_backup.py
    ├── rman_recovery.py
    ├── patch.py
    ├── upgrade.py
    ├── migration.py
    ├── clone.py
    ├── datapump.py
    └── adg_drill.py
```

ADG 现有行为迁入 `profiles/adg.py`，不保留两套编译入口。公共 Registry 是唯一的 Profile 分发点。

诊断目录新增相应 SQL/Host Tool，Playbook 目录按 `oracle.<domain>.<profile>` 命名。Prompt 只补充枚举
选择规则，不复制 Compiler 的业务规则和命令。

## 11. API 和下载设计

保留当前：

```text
GET .../implementation-runbook.pdf
GET .../implementation-runbook.md
GET .../implementation-runbook.zip
```

ZIP 不存在 Artifact 时返回 404，而不是生成空压缩包。所有下载继续复用 Conversation 所有者鉴权、
`private, no-store` 和 `nosniff`。文件名使用 Profile 和 Turn ID，例如：

```text
oracle-rac-build-<turn_id>.pdf
oracle-rman-backup-build-<turn_id>.zip
```

结构化 Runbook JSON 是程序内部合同，不提供用户下载路由。ZIP 自动加入 `README.md`，说明 Markdown、
PDF 和 ZIP 的用途、实际制品清单、使用前检查和 Profile 对应的安全执行入口；`00-manifest.json`
记录 README 的 SHA256，确保说明文件也属于可校验交付物。

只要实施命令引用 ZIP Artifact，Runbook 的第一个阶段必须是“配套 ZIP 制品部署”：明确提示用户从
当前 Turn 下载脚本包、传输并保存为固定暂存文件，然后提供解压到 `/var/tmp/kbot-runbooks`、恢复清单
权限以及逐文件 SHA256 校验的真实命令。后续 SQL、RMAN 或 Shell 命令只有在该阶段校验通过后才可执行。

## 12. 页面设计

实施文档页面继续使用连续正式文档，不使用折叠容器和内部滚动区。新增：

- Profile 中文名称和版本；
- 证据状态：已验证、策略派生、缺失阻断；
- 执行身份、节点和容器标签；
- 脚本清单、目标路径、文件权限和 SHA256；
- PDF、Markdown 和脚本 ZIP 下载；
- `BLOCKED` 步骤明确指向 Target 拓扑配置，不在聊天正文要求用户复制大量参数；
- 历史 Turn 永远下载当时固化版本，不按新 Compiler 重新生成。

## 13. 校验和安全规则

固化 Runbook 前执行统一 Validator：

1. 拒绝 `${...}`、`<...>`、`TODO`、`CHANGEME` 等未解析占位符；
2. 拒绝密码、私钥、连接串明文和 Wallet 内容；
3. 命令中的每个外部值必须能追溯到 `VERIFIED` 或 `DERIVED` 参数；
4. Artifact 引用必须存在且 SHA256 一致；
5. `run_as`、`node_scope` 和 `executor` 必填；
6. 高风险步骤必须有验证、回退或明确“不可回滚”；
7. 恢复、Failover、共享磁盘初始化和 GI 安装不得标记为自动执行；
8. 当前环境不满足适用条件时返回明确状态，不生成错误类型的文档。

## 14. 测试和验收

每个 Profile 至少覆盖：

- Planner 选择和错误路由测试；
- Tool Manifest、SQL Hash 和只读权限测试；
- 事实缺失、事实冲突和完整事实测试；
- 参数来源和禁止占位符测试；
- 当前已满足步骤只保留验证的测试；
- Script Artifact 文件名、权限、SHA256 和 ZIP 确定性测试；
- PDF 长命令、跨页、中文和目录测试；
- Markdown fenced code block 测试；
- 历史 schema 下载兼容测试；
- OpenAPI、Main API BFF 和 UI 静态契约测试。

Golden File 至少包含：

- 单实例文件系统到 RAC ASM；
- 单实例 ASM 到 RAC ASM；
- CDB 多 PDB 到 RAC；
- RMAN 无配置、已有配置、FRA 不足、SBT 已配置；
- 完整恢复、PDB PITR 和控制文件恢复；
- 补丁/升级/迁移的 READY、PARTIAL_EVIDENCE 和 BLOCKED。

## 15. 实施顺序

### 阶段 A：公共框架

1. 扩展 Profile、Command 和 Artifact 契约；
2. 建立 Compiler Registry 和模块化目录；
3. 增加 JSON/ZIP 下载；
4. 增加统一 Validator；
5. 保持 ADG/DGPDB 行为和历史下载不变。

### 阶段 B：RAC

1. 数据库 RAC Precheck；
2. 部署拓扑事实；
3. Host Runner/Collector 只读取证；
4. RAC Compiler 和脚本包；
5. PDF/Markdown/ZIP 验收。

### 阶段 C：RMAN

1. RMAN 配置、备份链、归档和容量 Tool；
2. Backup Build Compiler；
3. 调度、监控和脚本包；
4. Recovery Compiler；
5. 隔离恢复演练验收。

### 阶段 D：常用变更文档

按 RU Patch、Upgrade、Migration、Clone、Data Pump、ADG Drill 顺序实施。每个档案完成后独立交付，
不等待所有档案一次性上线。

## 16. 完成标准

数据库实施文档中心达到可交付状态必须同时满足：

- RAC 和 RMAN 文档基于真实事实生成，不以模型文本作为命令来源；
- READY 文档没有占位符和未解释参数；
- ZIP 脚本可解压、可校验、路径和权限明确；
- PDF、Markdown、JSON 和 ZIP 内容一致；
- 缺少基础设施事实时不伪造值，也不退化成只让用户补充信息；
- 所有高风险操作仍受现有审批和执行边界约束；
- 产品、架构、OpenAPI、测试和实施档案文档同步更新。
