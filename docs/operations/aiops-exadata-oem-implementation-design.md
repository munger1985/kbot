# Exadata＋OEM 专用监控与诊断实施设计

日期：2026-11-11。状态：已部署 kbotdev 并无损补齐 Exadata 两表；两端均已正常创建模拟系统、绑定现有 OEM，并通过真实公共 API 和浏览器最新值/历史验收。Ammolite Agent 列表及业务系统选择器也已在正常会话下验收，既有 Agent 授权未改；最新记录见 10.8。

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

### 8.6 首次实现阶段的未完成验收与发布条件（历史记录）

本节保留首次实现时的验收边界；后续模拟器、成员修订及经授权部署的实际结果以第 9、10 节为准，不代表当前仍未部署。

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

只读 smoke 的 `--cell-only` 明确标为 `METRIC_TRANSPORT_ONLY`、`inventory_verified=false`，不持久化或进入产品绑定路径。**第一阶段结束时尚未完成页面端到端验收**，当时未部署/重启 AIOps、未执行 Schema 变更、未创建或更改业务系统绑定。后续成员接口修订和经授权部署见第 10 节。第一阶段没有修改非 Exadata/非 OEM 监控和诊断，也没有修改应用生产适配器。

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

### 10.2 kbotdev 联调部署前核查（历史快照）

- KBot 当前运行 `/home/ubuntu/kbot4.0`，`kbot4.1.1@6cc14557`，已有 Exadata 基础功能但未包含本次成员修复；`http://140.238.44.208:8080/ui/aiops/monitoring.html` 返回 HTML 200。
- Ammolite 当前运行 `/home/ubuntu/Ammolite`，`dev@109138b2`，尚未包含 Exadata 基础功能；`http://140.238.44.208:19080/` 返回 HTML 200。监控路由为 `/apps/aiops/:domainCode/monitoring`，必须正常登录并使用授权领域。
- 两端均已有启用、CONNECTED 的 `OEM24ai` 来源，应用主机将 `aiops-oem.lab` 解析为 OEM 私网地址 `10.0.0.88`。两端均无 EXADATA 资源、Exadata 专用两表；不能仅更新代码就宣称页面可用。
- KBot 版本视图为 AIOPS / Schema 57，不能直接执行只适用于完整 Schema 59 的 Exadata 增量脚本；先核对实际对象差异并确定经授权的无损方案，不抬高版本号绕过检查，不重建或清空现有库。Ammolite 新表按规范 Entity/初始化流程处理。
- 本轮只读检查不涉及服务器拉取、DDL、来源/绑定修改或任何重启。后续部署必须明确授权、限定相关服务并保留现有配置备份；既有 OEM/数据库 Agent 和非 Exadata/非 OEM 监控保持不变。HTML 200、来源 CONNECTED 和适配器 smoke 均不能代替页面数据验收。

### 10.3 经授权部署与实际页面验收（2026-11-11）

用户明确授权部署两端、无损补齐必要数据库对象及仅重启相关服务。没有清库、运行全库初始化、重建业务表或处理无关知识库漂移。部署代码为 KBot `kbot4.1.1@4aa7f4ce`、Ammolite `dev@51174b7b`；均先确认目标提交，再 fast-forward，保留既有配置。配置及数据库元数据备份位于 kbotdev 的 `/home/ubuntu/aiops-exadata-deploy-20261111-6rj5zwyh`（私有目录，不提交会话或凭据）。

数据库实际处理：

- KBot 的版本视图虽为 57，完整部署期差异核对证明原有 100 表的列/类型/nullable、375 命名索引、295 命名约束、10 视图、1 属性图已符合当前基线，仅缺 Exadata 两表及对应约束、版本视图。没有直接套用只接受完整 59 的升级脚本，也没有先抬高版本绕过检查。先用规范 `015_ops_exadata.sql` 新增 `KBOT_OPS_EXADATA_INVENTORY`、`KBOT_OPS_EXADATA_REPORT`，复核原有对象未改变后，才同步规范 Schema 60 版本视图。最终全目录验收为 102 表、10 视图、1 图、375 索引、298 命名约束、1,564 列，合同 `aiops-oracle-v60-exadata-oem`。
- 现场完整验收暴露 Oracle 单次 `ODCIVARCHAR2LIST` 超过 999 参数的 `ORA-00939`；生成器已改为分批并用子查询固定双向 MINUS 的运算边界，规范验收工件与回归同步，不缩小验收集合。
- Ammolite 仅通过规范 Entity 定向创建 `aiops_exadata_inventory`、`aiops_exadata_report`：11 列、3 外键、2 主键已核验，已有表列/约束元数据前后完全一致。既存的 19 个 `aiops_knowledge_*` 缺表及旧 `kc_*` 等知识库漂移未处理，不能宣称 PostgreSQL 全库 Schema 验收通过。

仅定向重启 KBot AIOps API（18110）、Ammolite AIOps API（19130）、Ammolite Main API（19090）。实际 readiness 分别为 `/ready`、`/ready`、`/readyz`，均 200；AIOps API 的 `/readyz` 并非有效探针。KBot Main、两端 Worker/Scheduler/Executor/通知、Ammolite Portal、既有 OEM/数据库 Agent 未重启。Portal 为 Vite 开发服务，自动加载新源码；远端生产构建也已通过（954 modules，仅大 chunk 告警）。

两端沿用原 `OEM24ai` 来源和托管凭据。正常发现最初为空，核验确认现用非管理员 OEM 账号没有三个 SIM 目标的可见权限；经官方 EMCLI 仅对 DBM01/GRID01/CELL01 三个固定目标新增 `VIEW_TARGET`，未更换账号/密码、未授管理员或全主机权限、未修改既有目标权限。

KBot 真实业务流程：

- 使用正常登录接口取得真实会话，通过公共发现接口选定 DBM01 与其 Grid/Cell，再经公共业务系统 API 原子创建独立 `Exadata OEM 模拟联调（SIM）`，没有改任何既有业务系统或旧绑定。
- 系统 ID `01a1ca77-0ad7-7f92-8cd4-30224c93e76b`；EXADATA 硬件资源 ID `01a1ca77-0b23-7cb1-84de-5c806521749a`，显示名 `SIM - Exadata DBM01`。资产描述/属性明确标为合成联调数据。
- 公共专用 view API 实际读到 overview 5 项、cell 12 项最新值；一次过去一小时查询返回 450 个样本，CPU/内存各 45 个。该次最新样本为 2026-11-11 11:42 UTC，CPU 47.95%、内存 64.32%；仍是合成值，不是客户业务值，也不代表秒级刷新。
- Chromium 加载真实监控页面，使用正常登录结果，无路由 mock、无伪造 AuthContext。核对专用组件标题、SIM 资源、三次真实响应、DOM 行数（5/12/450）及 CPU/内存值和采集时间；最新快照和 HISTORY 切换均通过，无 pageerror。
- 公共诊断 API 返回 200，保存报告后通过公共 GET 读回，冻结证据完全一致；报告为 `INCONCLUSIVE`，含 440 个证据样本。这是 API/持久化验收，不是诊断按钮浏览器验收，也不是硬件健康规则验收。

Ammolite 已用其部署进程的配置及现用托管 OEM 凭据，分别严格核验整机成员、5 项 overview、12 项 cell 和 10 项数值历史；CPU/内存各 44 个历史样本。尚无正常 App 登录会话，**未创建 Ammolite 模拟系统/绑定、未验收其公共 App API 或浏览器数据页**；没有伪造 SSO、重置密码或直接写业务绑定表。仍需用户提供正常登录配合后完成此部分。

最后联调相关回归：两仓库各 10 个文件 **316 passed + 13 subtests passed**；KBot 加 Oracle 验收生成器为 **324 passed + 18 subtests passed**，并在干净 Git archive 加部署配置环境下复跑通过。两端架构边界检查通过。所有结果均为上述限定范围，不是全仓测试、267 项指标全覆盖或两端页面全部通过声明。

页面仍诚实显示 `THRESHOLDS_UNVERIFIED`、文本历史不支持、OEM 留存分辨率、成员范围和快照/历史差异等缺口，不把部分覆盖当整机健康。既有非 Exadata、非 OEM 监控及诊断保持原路径。

### 10.4 Exadata 专用指标卡片（2026-11-11）

针对页面只有裸表格的问题，两端专用组件改为复用现有监控令牌与共享按钮的响应式卡片网格，不改普通 OEM、数据库、主机或其他来源组件，也不改变指标目录、后端查询/诊断合同和数据库。

- 最新快照按组件、指标、原生单位及类型分卡。CPU/内存等明确以 `%` 返回且在 0—100 内的数值用比例条；文本与原始状态用数值/状态卡，不把状态码 1 翻译成健康。缺失、旧快照、截断和不支持的历史分别标注，不补零、不回填旧值、不虚构进度或健康阈值。
- HISTORY 仅为数值绘制 ECharts；相同指标的不同维度分别成序列，不同 Cell、单位和类型不合并，不求和、不计算跨组件均值或二次 rate。末次空值不回填；图表允许数值近似，卡片和“查看数据”保留精确原文、维度、单位、质量与采集时间。
- 原始表格移入每卡“查看数据”；全局缺口收进“数据质量与采集边界”，同时始终保留阈值/状态字典未验证的可见提示。保留有界分页、组件筛选、刷新和只读诊断；切换查询时清除旧场景卡片，取消旧请求并释放图表/ResizeObserver。
- 纯展示规则以 Ammolite `exadataPresentation.ts` 为基准，同步 KBot 的浏览器脚本；两端 CSS 相同，合同测试检查生成脚本全文与 CSS 一致。所有样式限定在 `.exadata-oem` 内，不增加全局表格、按钮或其他监控覆盖。
- 浏览器合同发现 Ammolite 专用 POST 未声明 JSON 请求头，已仅在该组件补齐；两端诊断幂等键改用各平台已有 UUID 工具，兼容当前 HTTP 调试地址，不修改通用请求封装或鉴权。

验证：两端 `tests/contract/aiops_exadata_cards.mjs` 均通过，使用真实生产组件与 ECharts、隔离 API 夹具，覆盖 1920/1440/1000/780/390 宽度、最新值/历史、空值/真零/旧值/截断、组件/维度/单位隔离、原始精度、分页、原始证据、错误状态及诊断按钮请求。既有 `aiops_monitoring_request_encoding.mjs`（六种非 Exadata 资源）及 `aiops_resource_monitoring.mjs` 两端通过；两端架构边界、KBot JS 语法检查、Ammolite Portal 生产构建通过（957 modules，仍有既有大包告警）。

另使用真实 KBot 登录会话，在完整监控页面中仅由浏览器替换本地修改的 HTML/JS/CSS，所有指标请求仍走真实公共 API/OEM，无指标响应 mock：5 张 overview 快照卡、12 张 Cell 快照卡、12 张历史卡（10 张数值图、520 条采样）均通过；1600 宽度下三列、网格无横向溢出，无 pageerror。Ammolite 本轮仍只有隔离浏览器合同验收，不宣称完成其现场正常登录验收。

本节首次检查时仅完成本地预览；经用户后续明确确认，现已同步 kbotdev，实际部署与线上验收见 10.5。

### 10.5 卡片前端部署与线上验收（2026-11-11）

用户确认同步前端。kbotdev 的 KBot 已 fast-forward 至功能提交 `2b4b1bd8`，Ammolite 至 `1c14fa57`；同时带入前次验收文档和无行为变化的模拟器 EOF 空白清理，没有修改后端代码或模拟容器。部署前核对准确提交、分支、文件差异和干净的已跟踪工作区，原有未跟踪配置备份全部保留；`.env` 及 TOML 配置 Hash 前后相同。

- 旧前端文件、提交、配置摘要及进程基线保存在 kbotdev 私有目录 `/home/ubuntu/aiops-exadata-cards-20261111-5t7fud88`。本次没有 DDL、绑定修改或任何服务重启；51 个 API/Worker/前端等进程的 PID 与启动时间保持不变，两个 AIOps API 和 Ammolite Main 的实际 readiness 均 200。
- KBot 在线 HTML、Exadata CSS、展示脚本、组件脚本均 200，响应 Hash 与本地提交文件一致。Chromium 直接访问已部署页面，使用正常会话；**没有静态资源替换或 API 响应拦截**。真实 OEM overview 显示 5 张快照卡，Cell 显示 12 张快照卡，HISTORY 显示 12 张卡（10 张数值图、470 条采样，CPU/内存各 47 条）。CPU/内存末次原值为 26.17% / 57.06%，采集时间 2026-11-11 12:27 UTC；DOM 原值与采样时间和真实响应一致，三列网格无横向溢出，无 pageerror。
- Ammolite 现有 Vite 进程自动读取新版主组件、指标卡、展示模块和 CSS，线上模块请求均 200；在原 Portal 进程环境下远端生产构建也通过（957 modules，仍有既有大 chunk 告警）。没有把模块可达/构建通过冒充登录后业务页面验收；Ammolite 仍缺正常应用会话，尚未建立其模拟资源绑定。

以上仅是卡片前端发布，不扩大既有模拟指标覆盖，不更改 OEM 采样/上传周期，也不将未知阈值、旧采样或不支持的历史解释为健康。浏览器若保留旧 HTML，可强制刷新后选择 SIM 硬件和对应场景。

### 10.6 聊天具体指标查询与授权隔离（2026-11-11）

针对“Exadata cell 温度是多少”却返回普通数据库总览的问题，用户确认补齐两端聊天链路。线上本轮请求为 `01a1caa4-7823-72b8-a7d5-92bc0a828550`，原监控适配器只调用数据库默认 Profile，未消费具体指标意图；当轮 Agent 的授权资源中也没有 Exadata。不得通过扩大 Agent 授权、修改绑定或用无关数据库的 NO_DATA 总览来代答。

设计与实现：

- Turn Router 继续使用现行 `turn_router@1.0.0`，在结构化输入中提供普通监控与 Exadata 专用参数的 `oneOf` 合同，以及从真实 267 项目录生成的编码、显示名、组件类型和场景。没有新增关键词匹配表、温度特判、自由查询语言或 Prompt 数据库版本。普通分支保留 `{}` / `window`；只有 Exadata 意图选择专用分支。
- 专用参数保存 `resource_type_key=EXADATA`、指标编码、明确的系统/资源/组件/来源名称及快照或历史意图。具体指标、概览和无法映射三种意图分开；读数与上下限、累计与速率不能互相替代。模型输出的指标必须属于审核目录，多个同能力步骤不得被静默忽略。
- 范围解析只在当前 Agent 已授权的活跃业务系统中筛选 Exadata。明确名称不匹配时拒绝，不退回整个系统、可信页面中的普通数据库或其他类型资源。无匹配 Exadata 时给出专用范围提示；专用适配器再与公开解析资源求交集，随后只从当前资源的 OEM 绑定中选来源，多个来源要求明确指定，不选择列表首项。
- Worker 装配既有 `ExadataOEMService`，复用标准托管凭据和采集前后整机成员、资源、来源、绑定版本复核。聊天在内部仅缩小审核场景的指标与组件选集；页面 HTTP 合同、场景默认选集、诊断选集和普通监控/诊断路径均不变。问温度只读取 `oracle_exadata.Cell_Metric.temperature`，不采集或输出未询问的 CPU、温度上下限、数据库总览。
- 多 Cell、不同维度、原生单位和类型分别显示，保留完整原值、采样时间、质量和响应摘要。单位为空明确提示，不擅自补 ℃；真零不判缺失，末次空值及同时间末条空值不回填，旧采样不冒充当前值，不做跨 Cell 聚合或健康阈值推断。回答证据冻结专用采集快照，可核对资源、绑定版本、目录摘要和原始值。
- 实时读取最多 24 项，且受本轮 `max_live_queries`、场景分页及总计不超过 55 秒/本轮超时预算约束；未完成部分明确标为缺口。历史只支持最近一小时内的整分钟相对窗口，不把绝对时段、超长窗口或文本历史偷偷替换成较短窗口/最新快照，不新增预测或诊断能力。

验证与交付边界：

- 新增两端相同的 `test_exadata_chat.py`，覆盖目录和参数合同、授权交集、具体编码取数、公开资源复核、来源隔离、组件隔离、精度/单位/真零/空值/旧值、多维度、预算、历史和诊断选集不可扩展等边界。Ammolite 既有普通监控测试替身仅补充实际 TurnPlan 必有的空 `steps`，未改变原断言或普通监控实现。
- 本项限定回归为 KBot 12 个文件 **251 passed + 25 subtests passed**，Ammolite 11 个文件 **253 passed + 25 subtests passed**，其中两端各新增 30 项专项测试；覆盖既有普通 OEM、资源监控、Turn 路由、共享资源和巡检范围，两端架构边界检查通过。不是全仓测试或线上聊天端到端验收声明。
- 在 kbotdev 使用现行模型及数据库中激活的 `turn_router@1.0.0`，只传入待验收参数合同进行实际模型调用：温度映射到 `Cell_Metric.temperature`，CPU 利用率映射到 `Cell_Metric.cpu_utilization`，普通数据库总览保持空参数 `{}`。当时沿用的真实授权摘要只有数据库/容器/虚拟机，模型仍保留 Exadata 意图供服务端拒绝，没有改查普通数据库。这是模型参数合同验证，不是新代码上线或聊天页面端到端验收。
- 实现提交阶段未执行 DDL、未同步 Prompt 数据库、未修改 Agent/业务系统/OEM 绑定，也未拉取线上代码或重启服务。实际读取温度必须已有模拟 Exadata 系统授权；用户随后确认部署，执行及验收见 10.7。

### 10.7 聊天修复部署与线上验收（2026-11-11）

用户明确同意同步到 kbotdev，且只重启两端 AIOps Worker。KBot 功能提交为 `4d385a78aa36e084ed490be147f154c3df8f980c`，Ammolite 为 `1449904f03725c694b9735e3a3e8a2c25090db94`；分别在原分支精确 fetch 并 fast-forward，没有合入本地其他未提交修改。

- 北京时间 21:04、21:05 分别完成 KBot、Ammolite Worker 优雅退出和启动，无强杀。各 Worker 的数据库、4 个任务执行组件、reconciler、outbox、retention、output-jobs readiness 连续三次成功；稍后再次检查两端 Worker、AIOps API、Main API 均 200/ready。其余 **49 个基线进程的 PID 和启动时间不变**，没有重启 API、前端、Scheduler 或 Executor。
- `.env` 和实际 TOML 配置 Hash 前后相同，原未跟踪配置备份保留。没有执行 DDL、初始化、Prompt 同步、权限更新或绑定修改。旧代码、配置摘要、进程基线、重启结果和验收记录保存在远端私有目录 `/home/ubuntu/aiops-exadata-chat-20261111-rr5zna1r`，不包含登录令牌或进程环境转储。
- 验收前通过正常会话只读查询发现，现行“OEM 测试用 Agent”已经同时授权测试数据库和模拟 Exadata 系统，与原失败轮的授权快照不同；本次没有修改该授权。使用正常会话、公共聊天 API 和独立验收会话，实际执行结果如下。

| 验收问题与范围 | 结果 | 证据 |
| --- | --- | --- |
| 已授权 Agent：“Exadata cell温度是多少” | `COMPLETED`；只读取 `oracle_exadata.Cell_Metric.temperature`，一次查询；回答模拟 Cell 原值 `32.0`、北京时间 `20:57:00` 采样、OEM 未返回单位，不输出数据库总览 | Turn `01a1cab6-b0b0-7130-99f8-e4ee433df12b`；公共 Exadata 页面 view API 的值、时间、单位及响应摘要均一致 |
| 未授权 Exadata 的现有 Agent：同一温度问题 | 范围 `DENIED`、Turn `PARTIAL`，明确“当前 Agent 授权范围内没有匹配的 Exadata 资源”，无指标读数、不改查其他数据库 | Turn `01a1cab7-71ef-7307-b273-2e47a832b82f` |
| 原 OEM Agent：“查看测试数据库（OEM监控）的当前数据库监控总览” | `COMPLETED`；步骤参数仍为 `{}`，只解析原数据库业务系统并返回原 Oracle 实时总览；原数据仍为 `NO_DATA`，没有宣称修复该既有数据问题 | Turn `01a1cab7-7204-76d8-9eb2-bcb9cf6b718c` |

两端当前在线 development-logs API 已核对：KBot 有上述温度轮的模型路由及新 Worker 启动事件，Ammolite 在重启后继续完成 `database.scope`、`database.diagnostic`、`workload.snapshot.collect` 和 Outbox 投递。日志中的退出码 143/signal 15 对应本次明确授权的 SIGTERM。以上为 **KBot 公共聊天 API 端到端验收**，不是聊天浏览器页面验收；Ammolite 已部署并验证进程/readiness，但仍缺正常 App 会话及模拟系统绑定，不将其健康检查或本地回归冒充线上指标聊天端到端通过。模拟原值不是物理 Exadata 的实测温度，未补写单位或推断健康阈值。

待排查观察：KBot 在线日志在北京时间 21:05:49、21:06:48 各出现一次“Conversation Turn 排队提升失败 / ORA-00060”。上述三个验收 Turn 均已到达预期终态，但不能据此宣称队列无异常；本次未修改队列并发逻辑，也未把该日志归因为本次 Exadata 变更。Ammolite 本次查询的部署后 ERROR 事件仅有授权重启的 SIGTERM 记录。

### 10.8 Ammolite 模拟系统与正常登录页面验收（2026-11-11）

用户提供测试账号后，经标准 SSO 与 AIOps BUSINESS 会话，在 `aiops-default` 通过公共发现及原子创建接口建立 `Exadata OEM 模拟联调（SIM）`，环境 DEV、硬件 `SIM - Exadata DBM01`、监控源为现有 `OEM24ai`，确认 `AIOPS_SIM_EXADATA_DBM01` 及其 Grid/Cell。系统 ID `01a1cad9-2904-7892-8428-fc333bd1f7ec`，资源 ID `01a1cad9-2906-7c00-baa6-1306ba23b359`；没有复制 KBot 主键、凭据或业务表行。

Chromium 加载已部署 Portal，无静态资源替换和 API mock：Agent 页显示现有 `Cyber DBA`，新建向导可选择原有系统及模拟系统（共 5 个），选中后取消，未保存 Agent。监控页实际显示 overview 5 张卡、Cell 快照 12 张卡，Cell HISTORY 12 张卡（10 张数值图、570 条采样）；CPU `48.3%`、内存 `64.43%`、温度原值 `32.0`（单位为空），采样时间北京时间 `21:42:00`，DOM 原值/时间与实际响应一致，三列无横向溢出，无 pageerror 或失败请求。

原有 4 个系统详情、Agent 列表/授权和 OEM 来源均经公共 API 复核不变。本次没有修改现有 Agent 授权、DDL、Prompt、后端代码、模拟容器或服务进程；尚未验收 Ammolite 聊天或生成新的诊断报告。模拟采样、OEM 留存分辨率、文本历史不支持和阈值未验证等边界保持可见，不解释为物理设备实测或健康结论。

本轮配置与浏览器证据、目录故障修复和部署记录详见 [Agent 目录故障隔离与 Ammolite 定向修复](aiops-agent-catalog-isolation.md)。历史章节中的“缺正常会话/未创建”均为当时状态，不是当前交付状态。

### 10.9 场景无数据只读排查与顶部提示删除（2026-11-12）

用户要求删除顶部整段 OEM 留存分辨率/阈值提示，并先确认磁盘 IO、Flash IO、IORM、互联交换机场景无数据的来源。本轮删除两端专用组件在最新快照与历史模式下的顶部提示及其未使用样式；第 10.4 节所述“始终保留”该顶部提示的要求由本节替代。保留“数据质量与采集边界”折叠区、每项样本质量和原始证据，不改变后台阈值、状态字典或诊断结论。

#### 10.9.1 在线证据与归因

使用 Ammolite 正常 BUSINESS 会话、现有模拟业务系统及托管 OEM 来源，通过公共 view API 复现；再用同一来源托管凭据只读核对真实 OEM 原始元数据、latestData、metricTimeSeries、现场模拟器代码与 OMS 日志。没有 mock 指标响应，没有直写 OEM 仓库或业务表。

| 场景 | 最新快照证据 | 历史证据与归因 |
| --- | --- | --- |
| 磁盘 IO | 公共接口首页 12 项、24 条非空样本；原始组全量与现行目录比对，53 项中 52 项有值，合计 104 个指标/维度值，缺磁盘状态项 | 首页 12 项均报 SOURCE_UNREACHABLE；直接读取聚合磁盘历史接口返回 HTTP 500，不是全部磁盘数据未采集 |
| Flash IO | 公共接口首页 12 条非空样本；原始组全量与现行目录比对，56 项中 41 项有值，15 项无值 | 首页 11 项 SOURCE_UNREACHABLE，另 1 项文本历史不支持；FlashCache 数值历史原始请求返回 HTTP 500；FlashLog 原始快照组为空 |
| IORM | 公共接口首页为空；6 个 IORM 原始组均为空，当前 Cell 适用 72 项均无样本 | Flash Disk IORM CG 数值历史原始请求也返回 HTTP 500。最新值缺失主要是模拟覆盖不足，不能以该历史错误解释全部快照缺失 |
| 互联/交换机 | HostInterConnect_Statistics 原始组为空，当前 Cell 适用 7 项无样本；已确认成员只有 DB Machine → Grid → Cell，没有交换机/端口目标 | HostInterConnect 数值历史原始请求返回 HTTP 500；交换机缺少目标与指标覆盖，不能用 Cell 的 HCA 吞吐冒充交换机指标 |
| 容量 | 两个 Aggregated 容量组为空；Filesystem_Utilization_Metric 实有 5 行，但其两项目录指标被拒绝 | OEM 维度名顺序为 cell_Id/name，目录为 name/cell_Id；适配器按 tuple 顺序比较，返回 SOURCE_METRIC_DIMENSION_MISMATCH。这是独立的采集适配问题，不是这两项在 OEM 中没有数据 |

表中全量快照统计来自已读取的组级原始数据与目录比对，不等同于逐页公共 API/浏览器全场景验收；适用项以当前已确认 Cell 为范围，不代表整机 267 项或交换机已覆盖。未将非空数值等同于健康或物理设备实测。

OEM 历史错误的反证与服务器证据：

- 同一来源、同一时段，Cell_Metric/cpu_utilization 的 metricTimeSeries 返回 HTTP 200；聚合磁盘 avg_cd_io_load、FlashCache fc_read_iops、HostInterConnect c_mb_received、Flash Disk IORM CG cg_fd_io_by_sec 的原始请求返回 HTTP 500，响应 code 为 InternalServiceError。
- 真实 OEM 容器 OMS 日志 /opt/oem/gc_inst/em/EMGC_OMS1/sysman/log/emoms.log 在 2026-11-12 01:43:08 UTC 连续记录 MetricsTimeSeriesRestController / MetricValueRepositoryImpl.getMetricTimeSeriesList 的 ORA-01791: not a SELECTed expression。因此当前失败发生于 OEM REST 后端 SQL，不是主机网络不通，也不能简单归结为 AIOps 未发起采集。尚未验证 OEM 补丁或请求规避方案。
- 现有 BaseDiagnosticSourceAdapter._response_json 把上游 HTTP >=500 统一转成 SOURCE_UNREACHABLE，页面缺口未区分 OEM 内部 SQL 失败与连接失败；本轮没有修改这个共享错误映射。

原始快照组覆盖复核：

- 有行：CellSrv_Status_Metric 1、Cell_Metric 1、Aggregated_HardNFlashDisk_Metric 2、CellDisk_Metric 2、FlashCache_Metric 1、SmartIO_Metric 1、Filesystem_Utilization_Metric 5。
- 空组：Exadata_Disk_Status_Metric、FlashLog_Metric、Flash_Disk_IORM_CG_Metric、Flash_Disk_IORM_DB_Metric、Flash_Disk_IORM_PDB_Metric、IORMDB_Metric、IORM_CONSUMER_GROUP_Metric、IORM_PDB_Metric、HostInterConnect_Statistics、Aggregated_Exadata_Capacity_Metric、Aggregated_Exadata_Diskgroup_Capacity_Metric；均未返回下一页链接。
- 现场 CellCLI 模拟器的 values_at 提供 CELL 6 项、两块 CELLDISK 各 14 项、FLASHCACHE 6 项、SMARTIO 4 项；有 IORM_MODE 和 HCA 吞吐不代表提供 DB/PDB/consumer-group IORM 指标或完整互联组。没有交换机/端口模拟目标。
- 现场模拟器文件与仓库文件 Hash 不同，values_at 主体相同；本轮按现场实际文件核验，未覆盖、对齐或重启模拟器，不把仓库文件存在当作现场已部署证明。

#### 10.9.2 修改、验证与未实施边界

- 两端真实生产组件/ECharts 的 aiops_exadata_cards.mjs 浏览器合同通过，新增最新快照/历史均不显示指定顶部提示的断言；原有布局、分页、精度、维度隔离和质量展示合同保持通过。
- Ammolite Portal typecheck、生产 build 与 KBot 专用 JS 语法检查通过；生产 build 仍有既有大 chunk 告警。
- 本轮仅修改专用前端、回归测试及本节文档。尚未修复维度顺序比较、修改历史请求或错误映射、扩充模拟器、改变 OEM 配置，也未改变 Agent 授权、业务绑定、数据库、普通 OEM/非 Exadata/非 OEM 监控诊断。
- 本轮未拉取线上代码、部署前端或重启任何服务。在线证据保存在 kbotdev 私有诊断目录 exadata-scenes-audit-20261112-blncunzf；包含正常会话的私有文件不得提交或输出。
- 后续如获授权，分开处理：维度名称集合与按名取值的严格身份验证、OEM 历史接口错误及准确展示、原厂字段驱动的 IORM/互联/交换机模拟覆盖；不得补零、伪造 OEM 返回或关闭维度校验来制造“有数据”。

### 10.10 场景无数据修复与覆盖复核（2026-11-12）

本节是用户确认“修复”后的实施记录，承接第 10.9 节诊断；不是所有历史查询、交换机或整机目录均已恢复的声明。普通 OEM、非 Exadata 与非 OEM 的监控诊断路径保持不变，没有修改数据库模型、Schema、授权或业务系统绑定。

#### 10.10.1 应用修改

- 专用 latest/history 接口按维度名称集合核验元数据，不再把 OEM 返回的字段排列顺序当作维度身份；按名称取值，仍严格拒绝缺失、额外、重名及错误维度，不合并不同组件或维度。
- 仅 Exadata 指标读取入口把 OEM HTTP 5xx 区分为 SOURCE_PROVIDER_ERROR，页面明确显示“读取失败 / OEM 服务错误”，不再归为“无数据”。实际网络异常仍为 SOURCE_UNREACHABLE；普通 OEM 和整机成员发现的错误合同不变。不展示上游原始报错或凭据。
- 删除两端最新快照及历史模式顶部指定提示；保留数据质量折叠区和原始证据，不改变状态字典、阈值或诊断结论边界。
- 没有把 latest 快照冒充历史、补零或改写 OEM 返回值。当前带维度历史仍可能读取失败，准确展示故障不等于恢复历史。

#### 10.10.2 模拟器补全与在线更新

依据 XA 24.1.1 原厂采集脚本扩充 CellCLI 只读输出，不修改原厂插件、不写 OEM 内部表：

- 增补 FlashCache 请求/字节累计及写入字段、14 项 FlashLog 原生字段。
- 分别输出 IORM_DATABASE、IORM_PLUGGABLE_DATABASE、IORM_CONSUMER_GROUP，各 24 个原生字段，由原厂插件生成磁盘/Flash IORM 六组指标；不使用 IORM_MODE 代替 DB/PDB/消费组 IO。
- 补充 HOST_INTERCONNECT 六个累计计数器、固定计数器起点和 XML context 时间字段；适配原厂解析器所需的 collectionTime 字段顺序，不用 CELL 的 HCA 速率替代互联统计。
- 修复 objectType 正则不接受下划线的问题。旧 Filesystem 5 行混入磁盘/Cache 等其他对象数值，不可用作文件系统利用率；新采样严格只有 / 和 /opt/oracle 两个文件系统维度。
- 补充 FlashCache/GridDisk 容量属性，磁盘容量使用原厂解析器支持的 G 单位，防止 T 单位被错误解析为近零；合成原值由真实插件计算后上传。
- 所有数值仍为 SIM 调试数据，未知命令和属性仍显式 fault，不代表物理设备实测或健康。

现场只更新 aiops-exadata-cellcli-sim 唯一只读 bind mount 的源码。新增 update_cellcli.py 要求核对旧 SHA256、验证容器及固定路径、备份并保留绑定 inode，容器内摘要匹配才成功；失败恢复原内容，不重启容器。本次最终源码 SHA256 为 f59daf83690ba1066b59aa8cbaa98ecb7b5f197dcbb0993c6278af2117999bea；两次备份保留在 aiopsdb 的 /opt/aiops-oem-lab/state/exadata-sim-update-qx1j9wm_ 和 exadata-sim-update-br2dds8e。

只对独立 SIM Agent 成功执行 Exadata_Performance_Metrics_All、Exadata_IORM_Performance_Metrics_All、Exadata_Capacity_Metrics、Filesystem_Utilization_Metrics 的原厂 runCollection 和上传；没有重启 OEM、CellCLI 或既有数据库 Agent。性能/状态原已是一分钟调度；IORM 仍按原厂 15 分钟、容量 1 小时、Filesystem 24 小时调度。本次提高频率的请求被原厂警告拒绝，未使用 -force，因此不能声称所有新增组已一分钟自动刷新。

#### 10.10.3 真实 OEM 最新快照结果

更新后读取真实 OEM latestData 全组，与当前审核目录逐项比对：

| 场景 | 当前 Cell 适用目录项 | 有值目录项 | 指标/维度值数量 |
| --- | ---: | ---: | ---: |
| 概览 | 5 | 5 | 5 |
| Cell | 12 | 12 | 12 |
| 磁盘 IO | 53 | 52 | 104 |
| Flash IO | 56 | 56 | 56 |
| IORM | 72 | 72 | 72 |
| Cell 互联 | 7 | 7 | 7 |
| 容量 | 7 | 5 | 16 |

这些是单个已发现 Cell 的 raw 组级快照统计，不是 267 项整机覆盖，也不是公共 API 逐页或浏览器全场景验收。文件系统样本已替换为两个正确维度；分别在 Ammolite 与 KBot 的运行环境和现用托管来源下，以修复版适配器的临时只读进程从真实 OEM 读取聚合磁盘、FlashCache、FlashLog、IORM、Cell 互联、容量和 Filesystem 八项代表指标，均有值且具名维度一致。互联接收速率原值 128、DB IORM 吞吐原值 72、Flash Disk 消费组吞吐原值 64；不擅自改单位、跨维度平均或补趋势。临时进程验证不代表线上应用代码已部署。

#### 10.10.4 尚未恢复的边界

1. **带维度历史**：OEM metricTimeSeries 对磁盘、FlashCache、IORM、HostInterConnect 等请求仍返回 HTTP 500；OMS 记录 ORA-01791: not a SELECTed expression。无维度 CPU 历史可读。已核对官方参数 sort、维度 key_ 过滤、目标 GUID/名称、分页及时间窗口，均未找到已验证的规避方式；尚未安装或验证原厂补丁。不能通过写 OEM 仓库、私改原厂二进制或替换成快照声称修复。
2. **交换机**：当前仍无交换机/端口模拟目标及正式成员，Cell 互联最新值已补齐不等于交换机指标已恢复。
3. **磁盘状态与 ASM 容量**：Agent 原厂 getmetric 可返回两块正常磁盘、SIMDATA 的 size/count，但 OEM latestData 的 Exadata_Disk_Status_Metric/status 和 Aggregated_Exadata_Diskgroup_Capacity_Metric/count,size 仍为空；尚未查明上送/留存原因，不以 Agent 本地结果冒充 OEM 已留存。

#### 10.10.5 回归与交付边界

- 两端分别执行维度/HTTP 故障、模拟器、Exadata OEM/成员发现/聊天、普通 OEM 原生/标准指标及监控、Exadata HTTP 合同，共 10 个测试文件，各 292 passed + 13 subtests passed。
- 两端真实生产组件/ECharts 卡片浏览器合同通过 1920/1440/1000/780/390 宽度；包含提示删除、上游故障卡片不补零/不画图、原有精度与维度隔离检查。
- Ammolite Portal typecheck、生产 build 及两端架构边界检查通过；build 保留既有大 chunk 告警。没有执行全仓测试或 DDL。
- 本轮模拟器已原位在线更新；应用代码和前端未在线拉取、部署或重启。真实证据保存在 kbotdev 私有目录 exadata-scenes-audit-20261112-blncunzf，会话及凭据文件不得输出或提交。

### 10.11 三项剩余问题的原厂证据、交换机模拟与后续边界（2026-11-12）

本节替代第 10.10 节中的“尚未部署应用”及“状态/ASM 上送留存原因未明”两项旧结论；不代表全部指标已能在页面读取。本轮继续限定真实 OEM、原厂插件及 SIM 目标，不修改 Oracle 二进制、内部表或指标类型，不把快照伪装成历史。

#### 10.11.1 已批准的应用部署与页面复核

两端已在 kbotdev 精确 fast-forward：Ammolite dev 至 6b1e5df096c4b9b1041ef466bceb2eee041cca90，KBot kbot4.1.1 至 050c1c248a044b6df156fb17e954ae42b38c436e。按此前用户确认，只重启两端 AIOps API：Ammolite 19130 的 /ready 与 KBot 18110 的 /ready 均 200；其余 48 个基线进程及配置 Hash 不变。没有 DDL、Prompt、授权或业务绑定修改。备份及 receipt 保留于 kbotdev 私有目录 /home/ubuntu/aiops-exadata-repair-20261112-oexnsi8c。

Ammolite 使用正常 BUSINESS 会话与已部署生产组件、真实公共 API/OEM，无响应 mock：磁盘、Flash、IORM、互联、容量首页分别有 24、12、12、7、16 条非空样本；对应 12、12、12、7、7 张卡。磁盘历史首页 12 张卡准确显示 SOURCE_PROVIDER_ERROR/读取失败，不补零、不以快照代替历史；无 pageerror，顶部指定提示已删除。证据目录 /tmp/exadata-online-deploy-wPjuMP。这是首页验收，不是逐页全量、全部组件或历史修复声明。Ammolite 业务链接必须带 domainId；不带时返回首页是路由上下文缺失，不能归因为鉴权故障。KBot 本轮未获得可用正常登录会话，未再次验收其浏览器页面。

#### 10.11.2 真实导出重新核对

重新完整读取用户 README，并按支持引号跨行的 CSV 解析原始压缩包：

- 定义仍为 88,680 条。oracle_si_network_port_switch 定义 17,940 条，归属于 138 个端口组件和 3 个父交换机；这 138 个端口 GUID 均不在 targets.csv 的 50 个独立目标内。
- 因此端口是原厂 TargetComponent，不是可人为注册成独立 Target 的同名对象。后续必须维护父交换机、原生组件 GUID、PortId/其他维度及授权关系，不能制造 Target GUID 或仅靠名称推断成员。
- 原始归档只有最新快照，没有历史时序。模拟历史必须由新采样经原厂 Agent 上传并由 OEM 留存，不能把客户导出的快照复制成过去的曲线。
- 不复制客户主机、IP、业务值或 GUID 入模拟系统；所有身份含 SIM，指标组、列名、类型、维度及单位仍以真实目录和现场原厂元数据为准。

#### 10.11.3 状态/ASM 不是“未入库”，历史不是“缺少模拟采样”

使用原厂 get_metric_data -json，独立于 AIOps REST 读取 OEM 已留存数据：

| 指标 | 原厂 EMCLI 留存证据 | 同一来源 OEM REST |
| --- | --- | --- |
| Exadata_Disk_Status_Metric/status | CD_00_sim_cell01、FD_00_sim_cell01 两个维度均有 normal | latestData items 仍为空 |
| Aggregated_Exadata_Diskgroup_Capacity_Metric | SIMDATA01/SIMDATA 的 size=8192、count=1；原厂类型均为 STRING | latestData items 仍为空 |
| Aggregated_HardNFlashDisk_Metric/avg_cd_io_load | 最近一天 HardDisk/FlashDisk 多个历史采样可由原厂命令读取 | metricTimeSeries 返回 500，OMS 为 ORA-01791 |

这证明相应模拟数据已经经过原厂上传和留存，不能再把 REST 空结果归因为模拟器没造数据。STRING/stateful 不得改成 NUMBER 来迁就接口；不能从组标签擅自补单位或把正常文本当作整机健康。

只读核对现场原厂 MetricValueRepositoryImpl 的查询构造后，确认两种独立问题：

1. 历史维度键子查询 SELECT DISTINCT metric_key_value，却按未投影的 KEY_PART_1..7 排序并分页，触发 ORA-01791；显式 sort 或 key 过滤不消除该子查询。无维度 CPU 历史仍可读。
2. latestData 虽为 STRING 选择 gc_metric_latest，却对带键组无条件以 GC_METRIC_VALUES_LATEST 的数值键集合过滤；纯 STRING 组因而被过滤为空。get_metric_data 可读与 REST 空结果并存，不代表上传丢失。

未私改原厂 jar、视图、metadata 或接口响应，也没有新增任意 SQL/Shell 代理。继续增加模拟数值不能修复上述 REST 查询逻辑。

#### 10.11.4 隔离交换机模拟

新增 switch_simulator.py、deploy_switch.py、register_switch.py，与两仓同名测试同步。仅在 aiopsdb 既有隔离实验网络创建 aiops-exadata-switch-sim（172.22.0.9:1161/UDP，无宿主端口发布），只读容器、非 root UID、cap-drop ALL、独立随机私有 community。

- MIB-II/IF-MIB 两个 100Gbps SIM Ethernet 端口；ENTITY-MIB 明确 SIM 的机箱、型号及序列号，用于原厂相关性身份。CISCO-PFC-EXT-MIB 提供两个端口、priority 0..7 的原始计数。速率、Delta 由原厂 SI 计算，不在 AIOps 或模拟器内伪造。
- 使用独立 SIM Agent 的 oracle.sysman.si 24.1.1.0.0，原厂部署状态 Success；不修改原厂插件。原厂 add_target 经 SNMP Access Point 创建 AIOPS_SIM_EXADATA_SWITCH01，OEM get_targets 已独立核对为 Up，Agent 中 AP 存在。
- 现场原厂实际选项为 monitoring_cred 单数；帮助文本的复数写法不生效。input_file 标签不替换这个选项内的凭据，故使用原厂 argfile 与临时 0600 文件，community 不进入进程参数或输出，结束删除临时文件。SNMPVersion 使用 v2c，不使用 2c。
- 已对新 AP 成功执行 Response、_Capability_、correlation_identifier、oracle_si_net_port_config、oracle_si_net_port_perf、oracle_si_switch_pfc 的手动采集及上传。首次 PFC 遍历到模拟 MIB 尾部的失败已保留；补充标准 SNMP 引擎标量后重新采集成功。原厂命令成功不等于 OEM 最新值或 AIOps 页面已验收。
- 独立验证结果：原厂 Agent 的 NetworkPortPerformance 确实返回两个端口，接收速率分别为 126000000、127000000，发送速率 96000000、97000000；这些是原厂按原始计数计算的值，不能冒充已从 OEM 仓库读到。原厂管理员只读 REST 已读到新交换机 PfcIfPriorityEntryMetric/latestData 的 16 行、64 个非空值，维度为 PortId/Priority。这是 OEM 传输与留存验证，不是现用 AIOps 监控账号授权或页面验收。
- 原厂 modify_collection_schedule 经 preview 核对后，仅对新交换机的 oracle_si_switch_pfc 启用原厂五分钟周期；OMS 返回成功，独立 Agent 的目标专属 collection XML 已核对 IntervalSchedule=5 Min，未使用 force。只是调度配置落地，尚不声明连续自动采样验收；其他新交换机集合没有提高频率或一并开启。
- 端口组件配置仍不完整：runCollection 返回成功后，单独 getmetric NetworkPortsTargetComponent 仍报 Empty snmp response；只读核对原厂 Cisco 实现发现其端口配置还依赖 VTP VLAN/FDB 等额外 MIB，当前夹具未覆盖。父交换机 NetworkPortPerformance REST 返回 404，不应以 Agent 本地性能值冒充 OEM 组件已建成或页面已读到；PfcIfEntryMetric 的本地单项调用也未成功，不声明完整 PFC 组覆盖。
- 只为本轮新建交换机容器加载源码重启；没有重启 OEM、既有数据库 Agent、CellCLI 或 SIM Monitor Agent。各失败注册记录和源码备份均保留，没有 force、删除重建目标或卷。
- 交换机尚未加入现有 DBM 成员，避免两端绑定的 live inventory digest 突变而导致原有监控 409。后续应通过正常发现/确认接口同步两端绑定，不能直接写业务表或构造用户身份。
- 当前 AIOps 发现只展开独立 Target，没有完成原厂 TargetComponent 端口身份支持。不能宣称新增容器或交换机 Up 就已让页面显示交换机端口指标。

#### 10.11.5 需要另行确认的接入修复

若要让以上原厂已留存值进入页面，需要选择原厂修复补丁，或明确批准仅为 Exadata 增加原厂 EMCLI 只读指标通道；不是继续补模拟数据即可实现。拟议只读通道必须限制审核指标、目标及组件身份，保留原厂采样时间/值类型/维度/单位，区分快照与数值历史、设置输出及时间预算，沿用资源授权和绑定前后复核，不接受任意命令、SQL、路径或模型拼接参数。普通 OEM、非 Exadata 及非 OEM 的监控诊断路径不动。此接入改造尚未实施，不能作为已交付能力。

两端模拟器与定向安全测试各 28 passed + 13 subtests passed，两端架构边界检查通过；没有执行全仓测试或 DDL。当前补充提交只包含隔离交换机部署工件、测试和本节文档，不改变应用读取通道、授权或既有绑定。

### 10.12 Exadata 专用原厂 EMCLI 读取通道实施设计（2026-11-12）

用户明确同意本节方案。只扩展 Exadata 专用入口；普通 OEM/非 Exadata/非 OEM 的查询和诊断仍走既有代码，不新增通用 Shell/SQL 服务，不修改 OEM 二进制或内部表。

- 原厂 EMCLI 客户端部署在应用服务器，运行文件路径/JVM/私有会话目录属于服务端技术依赖，不接受页面或模型提供的程序路径、参数或命令。OEM Endpoint、账号和密码仍只来自 AIOps App 内现有监控来源及托管凭据。
- OEM 来源新增显式 Exadata 读取方式 OEM_REST/EMCLI，默认保留 OEM_REST；选择 EMCLI 时要求配置已核对的 OEM CLI 时间区。没有异常自动切换，也不把 EMCLI 失败伪装成无数据或返回旧缓存。
- 每次读取仍由现有资源授权、来源绑定及采集前后成员/版本复核控制。EMCLI 使用同一来源账号，不借用 SYSMAN。只读取审核指标目录中匹配目标类型的组/列，独立核验原厂目标名称、类型、组名、键名和返回列；账号无权限、客户端缺失、超时、截断及格式错误分别失败关闭。
- 客户端使用无 Shell 的参数数组启动原厂程序，密码只经标准输入；隔离私有会话、限制会话数量和有效期，凭据变化不复用旧会话。原始错误、凭据和会话文件不进入日志或响应。进程超时/取消/输出超预算时回收本次子进程。
- 默认最新值调用原厂 get_metric_data -json，历史仅 NUMBER 使用原厂 -last_n_days=1 后严格裁剪到请求窗口，不增加未来点、扩展状态区间或生成缺失采样。使用经确认的时区解析原厂时间，保留十进制文本/STRING/真零/空值和不同维度，不改单位、不跨组件聚合，文本历史仍不支持。证据 Hash 来源于实际原厂响应。
- 补齐原厂交换机配置所需的最小只读 MIB，使端口组件由 SI 原厂采集生成。组件发现必须来自原厂可验证的父子元数据，明确区分 Target 与 TargetComponent，保持原生 GUID 和父目标授权链；不制造端口 Target，不按名称猜关系。
- 两端同步维护合同、实现、部署工件、测试和来源配置 UI。无需为来源 JSON 配置新增业务表；若组件身份合同需要字段，规范合同和保存的成员快照同步更新，不增加兼容路由或双写。
- 部署前备份源代码/配置/进程及原有绑定。仅定向重启需要加载新代码的 AIOps API/Worker；加入新交换机成员后通过正常公共发现/确认流程重绑，两端都完成确认后再验收页面。不伪造登录身份，不直接写绑定表，不影响原有业务系统。

#### 10.12.1 已实施的读取合同与安全边界

- 新增 oem_emcli.py 有界原厂客户端与 oem_exadata_emcli.py 严格响应解析，仅 Exadata latest/history 在来源显式选择 EMCLI 时使用；未选择时保持 OEM_REST，不进行错误自动降级。普通 OEM 标准指标、其他硬件和非 OEM 读取不切换。
- 来源配置新增 exadata_metric_transport、exadata_emcli_timezone、exadata_emcli_certificate_sha256，要求有效 IANA 时区与 64 位小写 SHA256 叶证书指纹。两端来源表单均可配置；编辑仍保留由独立用例维护的 standard_metric_mappings，既有连接预检/启用流程不变。
- 原厂客户端固定安装位置 /opt/aiops/emcli/emcli；API/Worker 各持有单一执行器，关闭时回收子进程与私有会话。每次读取（含缓存会话）先通过不携带凭据的 TLS 连接核对证书指纹，指纹不符不发送监控凭据。未使用 trustall、nocertvalidate 或全局 JVM 安全降级。
- 会话目录 0700，最多 4 个缓存会话、TTL 300 秒，HMAC 缓存键包含来源/版本/Endpoint/凭据/时区/证书指纹；子进程过滤平台环境，密码只经 stdin，超时/取消/输出超限回收独立进程组，异常不返回原厂原始错误。
- 原厂 setup 已建立认证会话，不再重复 login。针对 Java 21 默认 PKCS12 与原厂空密码证书库的差异，只在本子进程设置 keystore.type=JKS。原厂 setup/sync 必须写 bindings，因此每个私有会话仅从可信安装目录复制 jar、emcli、version.txt，不复制其他来源 bindings、cookie 或凭据；不要求应用用户写共享安装目录。已按实际 ubuntu UID/GID 验证，不以 root 成功代替应用运行用户验证。
- 专用读取预算为 2 MiB，native stdout 占其一半；每指标最多 256 点并明确标注截断。保留原厂末次采样时间、十进制文本、STRING、真零、空值、单位和全部具名维度；不展开 First/Last 区间，不推算增量，不跨组件平均。文本状态不读取历史；数值历史仅原厂 last_n_days=1 后裁剪至用户不超过一小时的窗口。无效时间/JSON、TLS 超时均转为受控来源错误，不泄漏原始响应。

#### 10.12.2 定向部署与公共页面验收

已从现有 OEM 取得原厂 24.1.0.0.0 advanced kit，核对 SHA256 后安装至 kbotdev 的 /opt/aiops/emcli；没有下载第三方客户端或覆盖原厂 jar。部署前备份代码、Hash 与进程，记录位于 kbotdev 私有目录 /home/ubuntu/aiops-emcli-deploy-0xud9qyx。仅重启 Ammolite/KBot 两个 AIOps API 和两个 Worker，没有重启 Executor、Main API、Portal、OEM、既有数据库 Agent 或 CellCLI。两端 API 19130/18110、Worker 19132/18112 的 /ready 均已复核为 200；近期 KBot 在线日志有正常任务完成。未执行 DDL 或直接写业务绑定表。

Ammolite 使用正常 BUSINESS 会话和公共 API 保存 EMCLI/UTC/已核对证书指纹，经连接预检后启用来源；原有标准指标映射前后完整一致。没有借用管理员读取应用指标或伪造登录身份。真实浏览器结果：

| 项目 | 已验证结果 |
| --- | --- |
| 磁盘 IO 首页 | 12 张卡、24 条非空样本 |
| Flash 首页 | 12 张卡、12 条非空样本 |
| IORM 首页 | 12 张卡、12 条非空样本 |
| 容量首页 | 7 张卡、18 条非空样本，包含 ASM size/count |
| 磁盘状态 | 正常页面翻至第五页，两块磁盘均为 STRING normal，object_name 分别保留 |
| 磁盘一小时历史 | 首次验收 1248 条采样，无 SOURCE_PROVIDER_ERROR；另等图表完成后核对 12 个真实 ECharts canvas，无页面异常 |

这些是指定场景/分页的验收，不代表全部 267 项或交换机端口已接入。历史由 OEM 实际留存采样构成，不是导出快照回填。原厂 REST 的 ORA-01791 与 STRING 过滤问题本身未打补丁；EMCLI 为经用户同意的专用读取路径。浏览器证据保留于本地私有 /tmp/aiops-emcli-public-6zJPEP、/tmp/exadata-history-render-J4RMTS、/tmp/exadata-disk-status-WvANPc；最终异常处理修订定向部署重启后再次通过 12 个历史 canvas 验证，证据为 /tmp/exadata-history-render-KsBdjM。会话文件不提交。

#### 10.12.3 交换机新进展与尚未交付边界

- 补齐原厂 Cisco 配置读取需要的最小 VTP VLAN1 状态、Ethernet 类型、MTU、SAID、TypeExt BITS 和 SVI。仅同一私有 community 或其唯一 @1 VLAN context 可读取，错误 secret 和其他 VLAN context 均拒绝；不填造 TokenRing/FDDI、MAC 邻居或拓扑。现场最终 switch_simulator.py SHA256 为 dd324522157a0408fd1eb1527e31480d64f239bc4093a79ef055d100c92b8a04。
- 独立 Agent 的原厂 NetworkPortsTargetComponent 已从此前 Empty snmp response 恢复，返回 Ethernet1/1、Ethernet1/2 两行真实 SI 识别的 SIM 配置；随后仅对此新 AP 执行 oracle_si_net_port_config 和 upload，原厂均成功。各次源码更新先核旧摘要、留备份，只重启新增交换机容器。
- 现用 Ammolite 来源监控账号最初无新交换机权限，已仅为该目标授 VIEW_TARGET，未授管理员/全主机权限。随后同一来源账号核对父目标 REST 身份，并通过原厂 EMCLI 读到 PfcIfPriorityEntryMetric/DeltaNumRequests 的 16 个 PortId/Priority 采样；不是仅管理员可读的证明。
- **仍未完成端口 TargetComponent 的可靠身份目录与应用合同**：原厂 get_targets 的独立目标目录未列出端口组件；TargetAssociations/父目标 associates 没有给出可用的组件身份。组件不能伪装成独立 Target，不能生成或猜测 GUID。因此尚不宣称端口性能或交换机页面已修复。
- **交换机尚未加入共享 DBM，也未更改两端业务绑定**。加入成员会改变 inventory digest；需两端均通过正常公共发现/确认流程重绑。当前 KBot 没有可用正常会话，用户此前提供的同名账号登录返回 401，未猜测密码、重置账号、构造用户 token 或直接写绑定表。
- **KBot 代码与来源配置 UI 已部署，但 OEM 来源仍保持 OEM_REST，尚未启用/验收 EMCLI 页面**。不能把 Ammolite 的页面成功等同于 KBot 也已恢复。继续需要 KBot 当前可用的正常登录账号或由用户在页面完成来源配置；同时还需解决原厂端口组件目录，二者是独立剩余事项。
- 后续以 KBot 实际运行用户和现用托管 OEM 凭据只读复核，新交换机 GUID/名称/类型也已核验通过；没有额外为 KBot 授管理员权限。该检查仍不等于应用来源已切换或业务成员已确认。

#### 10.12.4 本轮验证

两仓分别执行 EMCLI、Exadata OEM、成员发现、交换机 SNMP、普通 OEM 原生/标准指标、来源连接、资源监控共 8 个定向文件，**各 251 passed**；包含专用通道显式启用、普通 REST 不降级、原始精度/状态/空值/时间、严格身份/类型/维度、证书指纹、超时/取消/输出预算、会话隔离、私有 native bindings、VLAN context 和标准指标映射保留。两端架构边界检查通过；Ammolite Portal 生产 build 通过（保留既有 chunk 体积提示），KBot 配置 JS 语法及真实 OEM 来源表单脚本合同通过，隐藏的未完成指纹字段不会阻塞其他来源表单。未运行全仓测试，未执行 Schema 变更。交换机公共绑定及 KBot 正常页面验收不在通过项内。
