# AIOps 数据库实施方案 Runbook

版本：2.1
状态：数据库实施文档中心已实现
基准日期：2026-09-29

## 1. 产品定位

实施方案 Runbook 是聊天 Turn 内的独立任务模式，用于根据当前环境事实生成可审核、可复制、
可分步执行的数据库建设或改造方案。它不属于普通诊断建议，也不等于让系统立即执行变更。

三类能力边界如下：

| 能力 | 目的 | `action_intent` | 是否创建 Proposal |
| --- | --- | --- | --- |
| 调查 / 诊断 | 回答现状、趋势和根因 | `NONE` | 否 |
| Advisory | 展示一条已登记 Action 模板的命令 | `ADVISORY` | 可生成仅建议 Proposal |
| Implementation Runbook | 生成完整建设、升级、迁移或改造步骤 | `NONE` | 否 |
| Execute | 执行用户明确选择的具体动作 | `EXECUTE` | 是，必须审批 |

Runbook 生成阶段只允许执行固定只读前置核验。用户后续明确选择“执行第 N 步”时，系统必须按
当时环境重新核验对象和参数，再进入现有受控 Action、审批、执行和验证链；不能把整份
Runbook 一次性批准或批量执行。

## 2. Oracle ADG 建设

首个实施档案为 `ORACLE_ADG_BUILD`。用户提出“结合当前数据库参数生成完整 ADG 建设/实施/
改造方案”时，系统进入 `IMPLEMENTATION_RUNBOOK`，固定读取数据库身份和 ADG 前置事实，再
确定性生成完整方案。

方案覆盖：

1. 主备拓扑、版本、网络、存储、保护模式和实施窗口确认；
2. 主库 `ARCHIVELOG`、`FORCE LOGGING`、`REMOTE_LOGIN_PASSWORDFILE`、FRA 和本地归档目标整改；
3. 按每线程“联机日志组数 + 1”检查并补建 Standby Redo Log；
4. 把目标环境视为全新主机，从操作系统用户、目录、Oracle 软件和 RU 开始建设；
5. 主备 `DB_UNIQUE_NAME`、`LOG_ARCHIVE_CONFIG`、归档目标、FAL 和文件管理参数；
6. 复制主库密码文件内容，以目标 SID 命名，并配置静态监听和双向 TNS；
7. CDB Root/NON-CDB Target 使用 RMAN Active Duplicate 建设整库物理备库；
8. Oracle 26ai PDB Target 使用 DGPDB：创建独立目标 CDB、Broker 配置组和 standby PDB，不执行整库 Duplicate；
9. 日志传输、实时应用和可选 Active Data Guard 只读；
10. Data Guard Broker、延迟、日志缺口、RFS/MRP 或 PDB 级应用健康验收；
11. 每个阶段的风险、验证、回退、停止条件和日常运维命令。

SQL、RMAN、DGMGRL、Shell、配置片段和人工确认必须按真实命令类型分开展示。系统不能把 RMAN
或操作系统命令伪装为 SQL，也不能让模型自由生成未受版本控制的执行命令。
人工确认内容显示在“人工确认项”，不提供复制命令按钮；缺失事实时只展示阻断原因和补齐位置，
不能用“按检查表实施”之类中文说明填充“实施命令”。

## 3. 全新目标环境的自动派生规则

系统先把主库证据转换成一组确定的实施参数。备库 `DB_UNIQUE_NAME`、SID、主备 TNS Alias、
Broker 配置名、保护模式和 Redo 传输方式等可安全派生的值直接显示并代入命令。例如主库
`DB_UNIQUE_NAME=db26ai` 时，默认生成备库名和 TNS Alias `db26ai_stby`，命令直接包含
`DG_CONFIG=(db26ai,db26ai_stby)`，不再输出待替换模板。

系统不再把备库主机、Oracle Home 和存储路径作为必须由用户补充的输入。目标环境固定按“全新环境”
处理，并采用以下可审计默认值：

- 目标主机名按 `<主库短主机名>-stby` 派生；
- Oracle Home 优先从主库密码文件路径还原，无法还原时按数据库大版本使用标准目录；
- 数据文件、redo、FRA、审计目录和 ASM/OMF 磁盘组沿用主库布局；
- 整库备库名、SID、TNS Alias、Broker 配置名，以及 DGPDB 的目标 CDB/PDB 和两端配置名按固定规则派生；
- 密码文件复制主库原文件内容，只把目标文件名替换为目标 SID；
- 所有派生值进入“已解析参数”附录并直接代入命令，禁止输出 `${...}` 或 `<...>` 占位符。

基础设施实际命名规范与派生值不一致时，应在变更评审中调整生成参数；系统不会因为尚未提供目标环境
事实而只返回补充信息或阻断整份文档。

只有数据库前置取证本身失败时，状态才是 `PARTIAL_EVIDENCE`。这表示当前状态标记可能不完整，
不是拒绝生成方案。取得前置证据后，整库 ADG 与 PDB DGPDB 文档均为零必填外部输入。

## 4. 展示与交互

回答只给出一句执行边界提示，随后直接展示权威 `IMPLEMENTATION_RUNBOOK` 操作区块：

- 采用正式实施文档版式连续展示阶段和步骤，不使用折叠容器；
- 文档顶部提供可跳转的阶段和步骤目录；
- 命令类型、复制按钮、验证命令、风险和回退；
- 当前已满足的步骤只展示验证命令，不重复输出实施命令；
- 当前环境、解析参数、缺失事实和脚本 SHA256 清单作为连续文档附录展示；
- 页面随文档内容自然增长，代码块自动换行，聊天窗口不再嵌套独立滚动区；
- 支持当前及历史 Runbook 下载为 PDF 和 Markdown；包含脚本 Artifact 时可下载确定性 ZIP；内部结构化
  JSON 不作为用户下载项；
- 明确提示“仅生成方案，不会执行”。

从功能菜单生成实施文档时，先展示档案对应的可选参数表单：

- 参数按“常用参数”和“高级参数”分组，所有字段均可留空；
- “使用自动配置生成”完全依赖数据库事实、Target 运维事实和默认策略；
- “按当前参数生成”只提交用户实际填写的字段，用户输入具有最高优先级；
- RMAN 时间点恢复的目标时间使用浏览器日期时间选择器，提交后由服务端校验真实日历日期并统一保存为
  `YYYY-MM-DD HH24:MI:SS`；
- 聊天框中明确写出的目录、Schema、节点名、版本和场景也可进入同一白名单参数链；
- 密码、密钥、Wallet 和自由命令不允许作为生成参数；
- 文档生成后可点击“调整参数并重新生成”，系统回填本轮参数并创建新的 Turn，不修改历史文档。

结构化 `IMPLEMENTATION_RUNBOOK` JSON 是唯一权威数据，不额外持久化一份可能漂移的 Markdown。
下载时由确定性投影器从同一 Block 生成 Markdown，内容包含目录、阶段、步骤、命令 fenced code block、
风险、停止条件和参数附录，可直接进入版本库或继续编辑。PDF 也从同一 Block 渲染，不重新调用模型，
因此历史版本生成的文档同样可以下载。PDF 包含带页码的阶段/步骤目录，命令标题和正文采用 Markdown
风格代码块，保留原始换行并按页面宽度安全换行，不能再把多条命令拼成一条超长行或裁切到页面外。

## 5. 已实现实施档案

通用 v3 契约、Profile Registry、只读取证 Playbook、确定性编译器和脚本 Artifact 已建立。当前
数据库实施文档中心包含：

| 档案 | 目标范围 |
| --- | --- |
| `ORACLE_RAC_BUILD` | 两节点 GI/ASM/RAC 从零建设及现有数据库迁移 |
| `ORACLE_RMAN_BACKUP_BUILD` | RMAN 策略、备份脚本、调度、校验和监控 |
| `ORACLE_RMAN_RECOVERY` | 控制文件、数据文件、全库、PITR 和 PDB 恢复 |
| `ORACLE_RU_PATCH` | GI、DB Home、RU/OJVM 补丁、验证和回退 |
| `ORACLE_DATABASE_UPGRADE` | 版本/补丁兼容检查、升级路径、预检查、升级、字典与组件验证、回退 |
| `ORACLE_DATABASE_MIGRATION` | 源目标盘点、迁移方式选择、预同步、停机切换、校验和回退 |
| `ORACLE_CLONE_REFRESH` | 非生产克隆、刷新、隔离和脱敏交接 |
| `ORACLE_DATAPUMP_MIGRATION` | Schema/PDB 逻辑迁移及对象校验 |
| `ORACLE_ADG_DRILL` | Switchover、Failover 和 Reinstate 演练文档 |

新增档案必须复用同一 `ImplementationRunbook` 契约、命令类型、安全边界和执行升级规则，不能
为每类方案另建一套自由文本功能。

RMAN 备份和恢复档案都不要求用户补充 `ORACLE_HOME`，运行脚本通过 `ORACLE_SID + oraenv`
自动初始化环境，必要时从对应 PMON 进程解析 Home。用户填写的恢复环境 Oracle Home 只作为可选
覆盖提示；自动解析失败或本机 OS 认证不可用属于执行停止条件，不会把静态文档降为“等待必要事实”。
`BACKUP_DEST` 优先读取现有 RMAN Disk Channel 或文件系统 FRA；没有现有配置时按数据库唯一名派生
标准路径，并在正式配置前输出创建目录、容量核验和 RMAN Channel FORMAT 命令，提醒用户按实际挂载点调整。

RAC、升级等档案遇到无法安全派生的 IP、VIP、SCAN、WWID、目标版本或目标环境事实时，状态为
`BLOCKED_BY_REQUIRED_FACTS`：文档仍完整展示所有阶段，但相关步骤不会生成猜测值或占位符命令。
事实必须来自 Target、部署拓扑、Host Collector、策略模板或用户明确的业务决策。

RU 补丁静态文档不要求用户补录 `ORACLE_HOME`、`APPROVED_RU_ID` 和 `PATCH_STAGE_PATH`。
脚本运行时依次通过 `/etc/oratab`、PMON 和 `oraenv` 定位 DB Home，通过
`/etc/oracle/olr.loc` 定位 GI Home；补丁暂存目录优先使用显式配置，否则按数据库发行版和
`DB_UNIQUE_NAME` 派生。已审批补丁编号不明确时不伪造编号，而是在执行前要求暂存目录中只有一套
RU 和至多一套 OJVM，通过官方 SHA-256、补丁 Inventory 和 Analyze 校验后才允许实施。
介质、摘要、空间、Inventory 或冲突分析失败属于执行停止条件，不会把已生成文档降为“等待必要事实”。

Data Pump 文档允许在生成前可选填写 Directory 对象名、操作系统路径、业务 Schema、并行度和转储
文件前缀。留空时优先读取现有 `DATA_PUMP_DIR` 和当前容器非 Oracle 维护 Schema；没有可用目录时按
`DB_UNIQUE_NAME` 派生标准目录，没有可用业务 Schema 清单时生成当前容器 `FULL=YES` 方案。
Directory 对象名未填写时使用 `KBOT_DATAPUMP_DIR`；目标导入在目标主机使用本机 OS 认证执行，避免在
文档中伪造 TNS、主机名或凭据。派生目录会在参数附录标记为需执行前审阅，但不会把后续阶段降为
“等待必要事实”。

整库迁移静态文档也不把 `DESTINATION_REF`、迁移方法、源目标 TNS 和切换日期作为生成阶段的
阻断输入。未提供已登记目标拓扑时，默认生成“新目标主机、同平台同版本同拓扑同文件布局、批准
停机窗口内执行”的 RMAN Backup Location Duplicate 方案；源端和目标端均使用本机 OS 认证。
目标主机地址、传输凭据和具体窗口仍由正式变更单控制，文档只把它们列为执行条件和审阅项，不会
伪造参数，也不会用“等待必要事实”替代方法选择、目标准备、备份、复制、验证和回退命令。

RAC、RMAN 和其他常用档案的业务默认值、交付脚本、事实来源、阶段、风险和验收标准见
[AIOps 数据库实施文档中心详细设计](../proposals/aiops-database-implementation-library-detailed-design.md)。
