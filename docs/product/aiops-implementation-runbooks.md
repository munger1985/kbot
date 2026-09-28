# AIOps 数据库实施方案 Runbook

版本：1.0
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
2. 主库 `ARCHIVELOG`、`FORCE LOGGING`、FRA 和本地归档目标整改；
3. 按每线程“联机日志组数 + 1”检查并补建 Standby Redo Log；
4. 主备 `DB_UNIQUE_NAME`、`LOG_ARCHIVE_CONFIG`、归档目标、FAL 和文件管理参数；
5. 密码文件、静态监听和双向 TNS；
6. RMAN Active Duplicate 或经评审后替换为备份恢复路径；
7. 日志传输、Managed Recovery 和可选 Active Data Guard 只读；
8. Data Guard Broker 配置和验证；
9. 传输延迟、应用延迟、日志缺口、RFS/MRP 和 Broker 健康验收；
10. 每个阶段的风险、验证、回退和整体停止条件。

SQL、RMAN、DGMGRL、Shell、配置片段和人工确认必须按真实命令类型分开展示。系统不能把 RMAN
或操作系统命令伪装为 SQL，也不能让模型自由生成未受版本控制的执行命令。

## 3. 缺少外部输入时的行为

备库主机、Oracle Home、SID、TNS Alias、ASM/OMF 或文件系统路径、保护模式等事实通常不在主库
参数中。缺少这些输入时，系统仍交付完整 Runbook，状态为 `READY_WITH_REQUIRED_INPUTS`，并在
命令中使用 `${STANDBY_HOST}`、`${STANDBY_DB_UNIQUE_NAME}` 等显式占位符。

只有数据库前置取证本身失败时，状态才是 `PARTIAL_EVIDENCE`。这表示当前状态标记可能不完整，
不是拒绝生成方案。Runbook 不应仅返回“请先补充信息”。

## 4. 展示与交互

回答先给出简短摘要，随后展示权威 `IMPLEMENTATION_RUNBOOK` 区块：

- 当前环境摘要和整改状态；
- 实施前必须确认的输入；
- 可折叠的阶段和步骤；
- 命令类型、复制按钮、验证命令、风险和回退；
- 固定最大高度和区块内滚动，避免长方案拉高整轮回答；
- 明确提示“仅生成方案，不会执行”。

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
