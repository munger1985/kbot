# AIOps 数据库实施方案 Runbook 技术设计

版本：1.0
状态：首期已实现
基准日期：2026-09-28

## 1. 架构边界

Runbook 是 Conversation Turn 内的独立规划模式，不新增顶层 WorkflowKind：

```text
用户问题
  → Compact Planner 语义识别
  → IMPLEMENTATION_RUNBOOK + ImplementationProfile
  → 服务端固定展开只读 Tool / Playbook
  → Evidence Assessment
  → 确定性 Runbook Compiler
  → IMPLEMENTATION_RUNBOOK Answer Block
```

规划模型只负责识别实施档案，不负责生成最终命令。数据库前置事实来自固定目录 Tool；完整步骤和
命令由版本控制的编译器生成，避免自由 SQL、命令注入、遗漏整改步骤和不同模型输出漂移。

## 2. 契约

`CompactPlanningMode` 增加 `IMPLEMENTATION_RUNBOOK`，`TaskFrame` 和
`CompactPlanningOutput` 增加 `implementation_profile`。首期枚举为：

```text
NONE
ORACLE_ADG_BUILD
```

ADG Runbook 路由必须满足：

- `objectives=(PLAN,)`；
- `action_intent=NONE`；
- `requires_change=false`；
- `diagnostic_profile=GENERAL`；
- 不创建 Proposal，不进入审批或执行链。

结构化产物 `ImplementationRunbook` 包含：

- `status`：`READY`、`READY_WITH_REQUIRED_INPUTS`、`PARTIAL_EVIDENCE`；
- `current_state` 与 `required_inputs`；
- `phases[].steps[]`；
- 步骤适用性：`REQUIRED`、`ALREADY_SATISFIED`、`CONDITIONAL`；
- 命令类型：`SQLPLUS`、`RMAN`、`DGMGRL`、`SHELL`、`CONFIG`、`MANUAL`；
- `verification_commands`、`rollback`、`risks` 和 `stop_conditions`；
- 本轮证据引用。

## 3. ADG 前置取证

固定 Tool `db.ha.adg_precheck@1.0.0` 使用单条只读 Oracle SQL 汇总：

- 数据库名、`DB_UNIQUE_NAME`、平台、CDB、角色和打开模式；
- `LOG_MODE`、`FORCE_LOGGING`、Flashback、保护模式和切换状态；
- 密码文件、归档目标、FAL、文件名转换、OMF/FRA 和 Broker 参数；
- FRA 分配与已用空间；
- 联机日志线程、组数、大小；
- Standby Redo Log 线程、组数、大小；
- 每线程 `online group + 1` 的 SRL 需求和缺口。

Playbook `oracle.ha.adg_build@1.0.0` 固定执行：

```text
db.instance.identity → db.ha.adg_precheck
```

Tool SQL、Manifest、SHA256、输出列和 Playbook 引用一起版本化。目录加载时会校验 SQL Hash、只读
策略和 Playbook 只能引用允许的 Tool。

## 4. 编译规则

编译器位于 `application/implementation/runbooks.py`。它只消费已归一的
`TurnEvidenceFact`，不读取模型自然语言结论。

关键规则：

- `LOG_MODE != ARCHIVELOG`：保留并标记启用归档步骤为 `REQUIRED`；
- `FORCE_LOGGING != YES`：加入 `ALTER DATABASE FORCE LOGGING`；
- 未配置 FRA：加入 FRA 和本地归档目标配置；
- SRL 缺口大于零：按线程和缺口数生成补建命令；
- 已满足的前置条件不删除步骤，标记为 `ALREADY_SATISFIED`，便于审计和复核；
- 缺少备库主机、路径和保护模式时保留完整步骤并使用占位符；
- 前置证据缺失时产物为 `PARTIAL_EVIDENCE`，但仍包含完整阶段；
- Runbook 中不包含明文密码、密钥或自动切换/failover 命令。

## 5. 回答和 UI

`DbaAnswerComposeHandler` 根据 `task_frame.implementation_profile` 编译
`AnswerBlockType.IMPLEMENTATION_RUNBOOK`。模型 Markdown 只提供摘要，结构化 Block 是权威方案。
Runbook 模式显式禁止 `PROPOSAL_SUMMARY`。

前端按结构化字段渲染当前状态、必填输入、阶段、步骤和命令，并复用代码复制能力。阶段默认折叠，
Runbook 主体设置最大高度和内部滚动；页面不把长方案当作普通 Markdown 一次性铺开。

## 6. 从方案升级到执行

当前实现只交付方案。未来用户选择具体步骤时，执行链必须：

1. 将步骤中的占位符解析为用户确认或 Target 运维事实；
2. 重新读取当前数据库角色、状态和对象；
3. 把单条命令映射为已登记 Action Template；
4. 按风险策略创建一个 Proposal；
5. 人工审批后执行并验证；
6. 成功后再允许进入下一步。

不得直接执行 Runbook 自带文本，不得批量批准整份 Runbook，也不得因为命令来自确定性编译器就
绕过现有审批和执行边界。

## 7. 扩展规则

数据库升级、迁移、RAC 建设和备份策略通过新增 `ImplementationProfile` 和独立编译器扩展。
每个档案必须提供固定前置 Tool、确定性条件规则、完整阶段、验证/回退、测试和文档；共用现有
Task Frame、Evidence、Answer Block 和 UI，不新增平行的自由文本方案系统。
