# Exadata＋OEM 专用监控与诊断实施设计

日期：2026-11-11。状态：代码实现与本地相关回归完成；未部署，真实 OEM REST 与在线数据库验收未完成。

## 1. 目标与不变边界

在业务系统硬件层新增 `EXADATA`（Exadata 一体机），所属架构层为 `INFRASTRUCTURE`。仅当资源类型为 EXADATA 且监控源类型为 OEM 时启用本设计；普通数据库/主机的 OEM 行为、全部其他监控来源、现有 SQL 诊断、Exacheck 报告解析和默认指标目录保持原样。

不新增 OEM_EXADATA 监控源类型，不修改 OEM 来源的全局配置，不向现有主机类型集合加入 EXADATA，不把资源 ID 当数据库 Target ID。上轮审计发现的普通 OEM/PDB 总览失配不纳入本次修改。

## 2. 资源、发现与绑定

- 一个 EXADATA ManagedResource 表示一台整机，复用现有业务系统资源生命周期、责任组及版本控制。
- OEM 根目标必须为 `oracle_dbmachine`；`oracle_exadata` 是 Storage Cell，不能作为整机候选。
- 提供专属发现入口，返回服务端签发、绑定租户/Domain/来源/操作者/EXADATA 类型及版本的候选引用；保存时另行检查资源范围。
- 按可验证 OEM 关系发现成员；确认成员后才能读取。不能按名称、IP 网段或来源中所有目标推断成员。
- Cell、计算节点、交换机、端口、PDU、ASM 等保存到专用成员目录，不批量创建业务系统卡片。
- 绑定保存前重读根目标和成员身份，核验候选签名、成员归属、资源权限、来源状态和版本。
- 根资源与监控源绑定复用 ResourceSourceBinding；子成员及诊断结果采用专属持久化模型。仓储不提交事务。
- 新增数据库到 Exadata 的显式 `USES_EXADATA` 关系，保持现有 RUNS_ON、USES_STORAGE 和数据库拓扑语义不变。
- 未绑定 OEM 的 Exadata 可维护资产；不自动回退到主机指标或其他来源。

## 3. 指标契约与目录

新增独立 `resources/exadata_oem/`，包含 metrics、monitoring_profiles、diagnosis_profiles 三份版本化目录。

监控和诊断引用同一份精确指标定义：指标编码、OEM 类型、组/列、REST 类型、原生单位与审核换算、维度键、状态/速率/累计量语义、快照/历史模式、聚合规则、阈值和新鲜度依据。

首批覆盖真实导出中已有采样的 Cell 状态/资源、磁盘 IO、Flash Cache/Log、Smart IO、IORM、互联/RDMA、交换机与端口、容量/ASM/硬件。不是将 580 个 Cell 定义列全部声明为可用数值时序。

- IOSEC、MBPS 等已是速率，不重复求 rate。
- 保留 Cell/组件 GUID＋全部维度；不同 Cell 的同名磁盘不合并。
- 状态不求均值，累计量不凭快照求增量，延迟不做无权重跨盘平均。
- 仓库 UNIT 编码不等于 REST unitDisplayName；真实 REST 元数据不匹配时给出缺口。
- 字符串状态、原值、空值和采集时间保留；无数据、禁用、旧快照、读取错误及截断不解释为健康。
- OEM 阈值未读取或不可验证时不伪造默认健康阈值。
- 现有 baseline.v1.json、HOST/DATABASE OEM 默认面板内容不变。

## 4. 实时监控

新增 ExadataOEMService 和独立页面组件。现有入口仅对 EXADATA＋OEM 增加分派，其他输入返回原路径。

视图按整机概览、Cell、磁盘/Flash、IORM、互联/交换机、容量/硬件分组；支持组件筛选与下钻。展示实际采集时间、查询模式、来源单位、缺口和截断，不把较慢的 OEM 采集包装成秒级实时。

组件/指标查询必须有界，读前与返回前重新校验资源归属、绑定版本和确认成员。历史与最新快照区分，证据保留元数据和响应摘要。

## 5. 专用资源诊断

新增以 resource_id 为主体的同步有界只读诊断入口和结果合同，独立于数据库 Target 诊断。复用认证、UoW、幂等和配置审计设施，不投递旧数据库任务；资源主体和数据库主体明确区分，旧数据库租约的 target_id 不放宽或伪造。

诊断步骤：资源/成员权限核验 → 冻结目录与绑定版本 → 场景指标选择 → 有界只读取证 → 确定性计算 → 证据结论与缺口 → 持久化报告。

场景包含整机健康、磁盘 IO、Flash、IORM、互联网络、容量及硬件。所有发现必须包含真实指标、组件、时间和证据。缺少历史不能生成趋势结论；缺少阈值或状态字典不自动判健康。专用状态规则/字典必须附审核依据。

关联数据库的既有诊断入口、指标请求和 SQL 执行不改变；只有运行明确包含 Exadata 且已授权时才能引用专用硬件证据。本期不新增 CellCLI、Shell、OEM Job 或硬件变更动作。

## 6. 安全、数据库和同步

来源凭据继续通过既有托管凭据机制解析，根目标仅接受服务端签发候选，不接受任意目标 GUID、查询 URL 或 SQL；组件筛选只接受已确认成员 ID。成员关系失效、跨租户、资源移除、来源停用和绑定变更均拒绝读取。

KBot4 与 Ammolite 同步维护合同、业务行为、UI、模型、规范 Schema、初始化/部署工件和测试；只允许 Oracle/PostgreSQL 设施差异。不得主动在 readiness 或请求中校验自身 Schema，不改写已有绑定或用户数据。

## 7. 实施顺序与验收

1. 资源类型和整机候选/成员绑定。
2. 独立指标目录、读取服务、监控入口及 UI。
3. 资源诊断入口、证据、结果与持久化。
4. 两端合同、Schema、架构边界、前后端与隔离回归。
5. 真实 OEM REST 逐项验收，验证根/成员/组件、单位、类型、latest/history 和事件能力。

隔离验收包括：非 Exadata、不属于 OEM、混合来源、跨系统、撤销绑定、未知成员、源变更、部分失败、空值、旧快照、目录失配和查询预算；现有数据库/主机测试保持原预期。

本地归档 `oem-exadata-export-20261110.tar.gz` 与配套 README 是目录和快照审计依据，不复制真实地址、GUID 或业务值到源码。归档不能证明 REST 在线可读，不自动切换为 OEM 仓库数据库直连。

## 8. 实施记录

### 8.1 已实现范围

- 两端同步新增 EXADATA 资产类型、硬件层编辑/展示、专属整机发现与成员确认、OEM 绑定、USES_EXADATA 关系校验及监控入口。
- 共享合同：`packages/platform_core/src/platform_core/contracts/aiops/exadata.py`。应用入口：`application/exadata.py` 中的 `ExadataOEMService`，同时承担专用监控和资源诊断；不另建通用主机指标路径。
- OEM 身份与指标适配：`adapters/diagnostic_sources/oem_exadata.py`；绑定确认：`application/configuration/exadata_binding.py`；目录加载：`application/exadata_catalog.py`。以上相对路径均位于 `services/aiops_agent/src/aiops_agent/`。
- KBot 页面为 `ui/aiops/js/aiops-exadata-metrics.js`；Ammolite 页面为 `apps/portal-web/src/features/aiops/business/AIOpsExadataMetrics.vue`。只有 EXADATA＋OEM 进入专用页面，普通 OEM 和其他监控来源沿用原路径。
- 诊断是同步、有预算的资源级只读取证与报告保存，不投递数据库 Worker，不改变 TaskLease 的数据库 Target 外键。当前只输出缺口、采样质量及同指标同单位的精确 Decimal 最小/最大观察样本和证据索引；不把观察范围当趋势或故障结论。报告冻结指标定义、目录 Hash、绑定/来源版本、原值、维度、单位、时间及响应摘要。
- 尚未取得可验证的在线 OEM 阈值和状态字典，因此本期报告结论为 `INCONCLUSIVE`，不能自动给出健康、故障严重度或根因。这不是完整的硬件健康规则验收。

### 8.2 真实导出统计与首批目录

归档审计口径：50 个目标；1,350 个定义 GUID；88,680 行定义；9,303 个非键定义列；28,651 行最新采样，其中 19,217 行非空。Cell 子集为 3 个目标、67 个指标组、580 个非键定义列；252 列有最新行，248 列存在非空值。定义数、维度采样行数和非空列数不可混用。

专用目录收录真实导出中存在非空样本的 **267 个审核定义**，不是承诺所有 OEM 环境在线可读。七个场景分别为 overview 5、cell 12、disk 53、flash 56、iorm 72、network 60、capacity 11；overview 与 cell 共享 2 个定义，因此场景计数合计不等于去重定义数。

审核纠正包括：Aggregated_HardNFlashDisk 保留 `celldisk_type`；HostInterConnect/Filesystem 保留 `name`、`cell_Id`；磁盘/Flash/IORM 保留 `object_name`；聚合 Diskgroup 的 size/count 保持 STRING；容量 Usage 标签为 Allocated (%)；SmartIO MBPS 不当百分比；IOSEC/MBPS 不再次求 rate。禁用的 Exadata_KPI/EXA_ROCE 不声明可用。单位不从仓库编码猜测为 REST unitDisplayName，不做未审核换算。

`max_age_seconds=1800` 是首期审核的显示新鲜度策略，不是导出证明的 OEM 调度周期；页面保留实际采集时间，旧值标为 STALE，不能据此判硬件健康。

### 8.3 请求合同、成员范围与预算

内部入口为 `/internal/v1/aiops/exadata` 下的发现、视图、诊断和报告读取。公共前缀分别为 KBot 的 `/api/v1/apps/aiops`、Ammolite 的 `/api/v1/apps/aiops/domains/{domain_code}/ops`：

- `POST /diagnostic-sources/{source_id}/exadata-discoveries`
- `POST /business-systems/{system_id}/exadata/{resource_id}/sources/{source_id}/view`
- `POST /business-systems/{system_id}/exadata/{resource_id}/sources/{source_id}/diagnoses`
- `GET /business-systems/{system_id}/exadata/{resource_id}/sources/{source_id}/reports/{report_id}`

请求只接受审核场景、已确认组件 ID、分页和查询模式。默认 `LATEST_SNAPSHOT`；`HISTORY` 支持过去 1—60 分钟的数值历史，分辨率标为 `PROVIDER_MANAGED`，文本历史返回明确缺口，不回退快照。页面提供最新快照/过去一小时历史及当前页只读诊断。报告以 Idempotency-Key 幂等保存。

每页默认 12、最多 24 个“组件×指标”查询，每项最多 256 个样本、262,144 字节；指标采集 30 秒，成员复核与采集合计 55 秒。发现最多扫描 2,000 个目标、16 台整机，每台最多确认 128 个成员。预算耗尽或分页均明确标为部分覆盖，不作整机健康结论。

根目标必须为 `oracle_dbmachine`。成员必须有 OEM 原生包含关系证明，并逐项复核 GUID、名称和类型；不得从名称/IP、同 Agent 或指标目录推断归属。初版使用 `/members`，现场验证后的正式关系读取设计见第 10 节。可用目标类型的目录定义不代表这些组件已经属于该整机。

候选签名绑定租户（平台适用时）、Domain、来源、操作者、EXADATA 候选类型、来源版本和成员摘要；**不预先绑定某个资源 UUID**。资源与系统的权限、归属和类型由保存入口另行检查。采集前后重读成员并检查资源、系统、绑定、来源版本；成员变化要求重新发现确认。

初版成员接口依据为 [Oracle Enterprise Manager 24.1 members REST 文档](https://docs.oracle.com/en/enterprise-manager/cloud-control/enterprise-manager-cloud-control/24.1/emrest/op-https-em_host-em_console_https_port-em-api-targets-targetid-members-get.html)。这不证明现场相同版本号即支持该接口；第 10 节改用已验证的官方 Data Service，仍不猜名称、不直连 OEM 仓库补成员。

### 8.4 Schema 与部署工件

两端新增独立成员快照表、资源诊断报告表：Inventory 以 binding_id 为主键/外键；Report 关联 resource_id 与 binding_id 并保存冻结 JSON。只由 UoW 提交事务，不改已有绑定数据，不执行启动时自 Schema 检查。

KBot 规范为 Schema 60（`aiops-oracle-v60-exadata-oem`），102 表、10 视图、15 份规范 SQL。新增 `database/oracle/aiops_agent/015_ops_exadata.sql`；Manifest、版本视图、静态验收及两份重建/验证生成工件同步。保留配置的同版重建也保留已确认成员，但不保留运行报告。

`database/oracle/operations/apply_aiops_schema_60.sql` 仅供已核对完整 Schema 59 基线的发布窗口新增两表和更新版本视图；执行前检查版本。它不是从旧 Schema 57 或其他漂移环境直接升级的通用脚本。Oracle DDL 隐式提交，失败时须保持停机并检查已创建对象，不能假设回滚或盲目重跑。Ammolite 只同步规范 Entity、初始化模块登记和静态验收，没有本地迁移、回填或双读路径。

### 8.5 实际验证结果（2026-11-11）

- 19 个相关单元/HTTP 合同测试文件：KBot **447 passed + 21 subtests passed**；Ammolite **445 passed + 12 subtests passed**。覆盖各端 38 项新 Exadata 单测、9 项新 HTTP 合同，以及普通 OEM、Exacheck、资源监控、业务系统等隔离回归；不是全仓测试通过声明。
- 两端硬件绑定/Exadata 独立发现/USES_EXADATA 的 Node 合同及拓扑布局合同通过。KBot 新增和修改的监控 JS 语法检查通过。
- 两端架构边界检查通过；KBot Oracle 规范静态检查与生成工件一致性通过；Ammolite PostgreSQL 规范/初始化静态检查及 AIOps metadata 检查通过（106 表）。
- 两端最新快照/历史合同已重新生成并校验 OpenAPI：KBot 16 个快照、855 条路径；Ammolite 18 个快照、1,005 条路径。
- Ammolite Portal 生产构建通过；仍有现存大包告警，不视为运行时验证或浏览器验收。
- 旧 `test_aiops_monitoring_precision.py`：KBot 为 24 passed。Ammolite 初次验收为 16 failed / 8 passed，已按后续用户要求补齐 SimpleNamespace 测试替身的 `_monitoring_profile_catalog=None`，与真实构造器默认值及 KBot 测试一致；未修改生产代码或原有断言。修复后该文件为 **24 passed**；与监控快照、巡检主机、专业巡检报告、巡检时间质量共 5 个文件联合回归为 **129 passed + 5 subtests passed**（Ammolite）。

### 8.6 未完成验收与发布条件

1. **整机在线 OEM REST 验收未完成**。后续模拟环境已完成部分指标传输验证，详见第 9 节；仍须以现场版本、受控凭据和已确认整机核验成员、完整元数据类型/单位/维度、文本缺口、权限和预算。未新增或宣称 Exadata 事件诊断能力。
2. **没有应用 Schema 或部署**。只读真实 Oracle Schema 验收因本机 127.0.0.1:1521 无监听而不能完成；真实 PostgreSQL 全实体只读验收因新增两表未应用及既有大量 Schema 漂移而未通过，涉及知识入库、输出任务、资源绑定、通知及旧 KC 对象等。静态工件通过不等于现有数据库可运行。
3. 发布前先单独核对数据库完整基线和安排相应初始化/切换，不能在业务请求中补表，也不能直接对漂移环境套用 Schema 60 两表脚本。
4. 实施验收阶段未执行数据库 DDL/初始化、部署或服务重启；后续代码提交/推送不代表上线或在线验收通过。无关工作区修改保持独立，不纳入本功能提交。

## 9. 模拟 Cell 与真实 OEM 第一阶段联调（2026-11-11）

用户授权在 aiopsdb 实验主机新建模拟容器，用真实 OEM 采集并保存，再由 AIOps 正常 OEM 接口读取。采用 **CellCLI XML/独立 SSH → 标准 OEM Agent/XA 原厂插件 → 真实 OEM latest/history → AIOps OEMAdapter**；不模拟 OEM REST，不写仓库内部表，不注入客户真实业务数值。

工件和操作说明见 [Exadata 模拟夹具 README](../../scripts/deployment/aiops_exadata_sim/README.md)。新增 `aiops-exadata-sim-monitor`、`aiops-exadata-cellcli-sim` 及 `AIOPS_SIM_EXADATA_*` 目标，原有 OEM、数据库和 Agent 不重启、不接管。标准 Agent 原厂 11 项 prerequisite 全部通过，XA 插件部署、心跳与上传成功；模拟目标的状态/性能集合调度已设一分钟，不承诺仓库即时刷新。

原厂系统模型要求 DB Machine → Grid → Cell；现场通过 create/modify_system 建立该结构。命令接受不等于成员读验收：现场根/Grid 的 `/members` 均为 404 HTML，`expand=associates` 返回空，严格整机 smoke 返回 `SOURCE_QUERY_UNSUPPORTED`。只读检查现场 TargetsController 未检出 members 路由，但尚未查明支持接口所需的确切 RU，不自动升级 OEM、不以名称/IP猜归属、不把 Cell 当整机根，不放宽已有失败关闭行为。

实际 KBot 和 Ammolite 的 OEMAdapter 分别 已从真实 OEM 读到 overview 5 项、cell 12 项最新值以及 CPU/内存数值历史。最初样本中 CPU 为 42.801%、内存为 62.601%，采样时间是 2026-11-11 09:45:00 UTC；这些是模拟器按原生字段生成、由真实插件计算保存的合成数据，不是用户导出包中的业务值。部分单位为 n/a/null 时保留 OEM 原值，不擅自补单位。

两端各 9 个相关测试文件通过 **253 passed + 13 subtests passed**：模拟器/SSH/注册安全合同 8 项、Exadata/HTTP 合同 47 项、既有普通 OEM/资源监控回归 198 项。部署工件与 smoke 两仓库同步；整理后的注册器未删除重建现场目标进行复跑，因此不声明全新部署全自动验收。CellCLI 仅用于隔离实验模拟，不新增产品内的 Shell/CellCLI 执行能力。

一分钟原厂调度写入 Agent 配置后，未重新手工触发集合，仅调用新 Agent 的 `emctl upload agent` 刷新上传，即在 OEM history 读取到 09:49、09:50、09:51 UTC 的连续自动采样；CPU 为 26.0/26.2/26.88%，内存为 57.0/57.07/57.29%。因此已验证持续合成采集，但仍存在原厂批次上传/入库延迟，不能把一分钟调度说成一分钟页面刷新保证。

只读 smoke 的 `--cell-only` 明确标为 `METRIC_TRANSPORT_ONLY`、`inventory_verified=false`，不持久化或进入产品绑定路径。**目前未完成页面端到端验收**，未部署/重启 AIOps、未执行 Schema 变更、未创建或更改业务系统绑定。需先解决现场 OEM 可验证成员接口能力，再按原有安全绑定流程接入页面。此次没有修改非 Exadata/非 OEM 监控和诊断，也没有修改应用生产适配器。

## 10. 现场成员关系读取修订

用户确认继续处理成员发现。现场官方 REST `members`、`immediateRelationships` 和 `configMetricGroups` 均为 404；原厂 `get_system_members -depth=2` 则明确返回 DB Machine → Grid → Cell。已验证官方 EMCLI `invoke_ws` 文档中的 GET `/em/websvcs/restful/emws/core/v0/list/dataservice`，使用固定 `name=TargetAssociations` 及 `TARGET_NAME/TARGET_TYPE/ASSOC_TYPE` 精确过滤，能读到同样的真实关系。

将 Exadata 专用成员读取**统一改为**该 Data Service；不保留先试 members 再降级的双路径，不修改普通 OEM Adapter 读取。不增加客户端软件、SSH、直连 OEM 仓库、新服务或凭据种类。

1. 仍以 `/api/targets/{rootGuid}` 核验 `oracle_dbmachine`。Data Service 固定在同一 OEM `/em` 基址，只读取 `TargetAssociations`，固定过滤 `ASSOC_TYPE='internal_contains'`；不开放 SQL、资源名、查询字段、端点或关联类型给用户/模型。
2. 返回的关系必须逐行精确匹配当前父目标名称、类型和关联方向。这里名称/类型来自 **OEM 显式关系**，仅用于查回唯一 GUID，不是按相似名称猜归属。再通过 `/api/targets?name=...&typeName=...&limit=2` 唯一定位，以及 GUID detail 复核身份；缺失、歧义、目录分页或类型/名称漂移均失败关闭。
3. 只遍历根整机及其直接 `oracle_exadata_grid` 的包含边；保留 Grid 自身以及实际返回的直接组件。Grid → Cell 必须有独立显式包含关系；不把 `hosted_by`、`monitored_by`、`stores_on`、部署关系或数据库关系混入硬件成员。不遍历任意间接端口/ASM/其他系统，不允许循环、重复成员或跨整机根；至少须有一个可核验 Cell。
4. Data Service 不提供此资源的分页合同，故必须有布尔 `exceedsMaxRows=false`、完整列名和合法数据数组。缺失标志、截断、超预算或服务未正确应用过滤时拒绝整批结果，不把已收到部分当完整 Inventory。
5. 保留既有 128 成员、4 MiB、30 秒 Inventory 预算，发现 2,000 目标/16 根预算以及读前读后的成员摘要、来源/绑定/权限复核。全部 HTTP GET 禁止重定向和跟随响应地址；沿用来源托管凭据和字节预算。无新持久化模型或对现有绑定的自动改写。

依据：[Oracle EMCLI invoke_ws](https://docs.oracle.com/en/enterprise-manager/cloud-control/enterprise-manager-cloud-control/24.1/emcli/invoke_ws.html)、[list](https://docs.oracle.com/en/enterprise-manager/cloud-control/enterprise-manager-cloud-control/24.1/emcli/list.html)、[get_system_members](https://docs.oracle.com/en/enterprise-manager/cloud-control/enterprise-manager-cloud-control/24.1/emcli/get_system_members.html)；现场资源 help 明确五列及可搜索字段，过滤后的 `internal_contains` 与原厂系统成员命令一致。官方 REST `/targets` 的精确名称/类型查询也已现场验证。

### 10.1 修订实现与验证结果

- 两端同步新增 `adapters/diagnostic_sources/oem_exadata_inventory.py`，只被 Exadata 专用入口调用；普通 OEM/其他来源监控、诊断、认证和 Schema 未修改。专用页面的成员范围提示同步改为实际证明范围。
- 两端各新增 38 项关系安全测试，覆盖两层包含、固定同源端点、精确过滤、重复/循环、唯一 GUID、详情漂移、截断/格式错误、认证/限流/服务错误、禁止重定向和任意地址。连同已有相关合同及隔离回归，各 **291 passed + 13 subtests passed**（10 个文件）；两端架构边界检查通过。
- 使用各仓库自身实际 OEMAdapter，在真实 OEM 上分别通过默认严格整机 overview、cell、数值 history 三种 smoke。结果均为 `inventory_verified=true`、`VERIFIED_EXADATA_INVENTORY_AND_METRICS`，确认 DBM01、GRID01、CELL01；overview 5 项和 cell 12 项最新值非空，CPU/内存各返回 40 个历史样本。成员摘要一致；没有使用 `--cell-only` 或假绑定。
- 历史末三项为 2026-11-11 10:25/10:26/10:27 UTC，CPU 30.56/32.58/34.81%，内存 58.52/59.19/59.94%。这些仍是合成调试值，由真实 OEM 原厂插件采集保存；存在批次上传延迟，不保证秒级或一分钟可见。
- 应用环境已确认为 kbotdev（私网 `10.0.0.190`），2026-11-11 完成只读核查，详见下节；尚未部署/重启应用、执行 DDL 或创建业务系统绑定，未完成浏览器端到端验收。源码和工件同步不等于已推送或已上线。

### 10.2 kbotdev 联调部署前核查

- KBot 当前运行 `/home/ubuntu/kbot4.0`，`kbot4.1.1@6cc14557`，已有 Exadata 基础功能但未包含本次成员修复；`http://140.238.44.208:8080/ui/aiops/monitoring.html` 返回 HTML 200。
- Ammolite 当前运行 `/home/ubuntu/Ammolite`，`dev@109138b2`，尚未包含 Exadata 基础功能；`http://140.238.44.208:19080/` 返回 HTML 200。监控路由为 `/apps/aiops/:domainCode/monitoring`，必须正常登录并使用授权领域。
- 两端均已有启用、CONNECTED 的 `OEM24ai` 来源，应用主机将 `aiops-oem.lab` 解析为 OEM 私网地址 `10.0.0.88`。两端均无 EXADATA 资源、Exadata 专用两表；不能仅更新代码就宣称页面可用。
- KBot 版本视图为 AIOPS / Schema 57，不能直接执行只适用于完整 Schema 59 的 Exadata 增量脚本；先核对实际对象差异并确定经授权的无损方案，不抬高版本号绕过检查，不重建或清空现有库。Ammolite 新表按规范 Entity/初始化流程处理。
- 本轮只读检查不涉及服务器拉取、DDL、来源/绑定修改或任何重启。后续部署必须明确授权、限定相关服务并保留现有配置备份；既有 OEM/数据库 Agent 和非 Exadata/非 OEM 监控保持不变。HTML 200、来源 CONNECTED 和适配器 smoke 均不能代替页面数据验收。
