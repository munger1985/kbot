# AIOps 数据库实施方案 Runbook

版本：1.4
状态：首期已实现
基准日期：2026-09-28

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

## 2. 首期能力：Oracle ADG 建设

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
- 当前环境、解析参数和待补齐输入作为连续文档附录展示；
- 页面随文档内容自然增长，代码块自动换行，聊天窗口不再嵌套独立滚动区；
- 支持当前及历史 ADG Runbook 下载为 PDF；
- 明确提示“仅生成方案，不会执行”。

PDF 下载按会话和 Turn 鉴权，直接渲染已固化的 `IMPLEMENTATION_RUNBOOK` Block，不重新调用模型，
因此历史版本生成的文档也可以下载。PDF 包含带页码的阶段/步骤目录，命令块保留原始换行并按页面
宽度安全换行，不能再把多条命令拼成一条超长行或裁切到页面外。

## 5. 后续实施档案

通用契约和展示框架已经建立，后续按独立档案增加：

| 档案 | 目标范围 |
| --- | --- |
| `ORACLE_DATABASE_UPGRADE` | 版本/补丁兼容检查、升级路径、预检查、升级、字典与组件验证、回退 |
| `ORACLE_DATABASE_MIGRATION` | 源目标盘点、迁移方式选择、预同步、停机切换、校验和回退 |
| `ORACLE_RAC_BUILD` | GI/ASM/网络/SCAN/VIP 前置、集群安装、数据库转换、服务和故障验证 |
| `DATABASE_BACKUP_STRATEGY` | RPO/RTO、RMAN 策略、保留、归档、异地副本、恢复演练和监控 |

新增档案必须复用同一 `ImplementationRunbook` 契约、命令类型、安全边界和执行升级规则，不能
为每类方案另建一套自由文本功能。
