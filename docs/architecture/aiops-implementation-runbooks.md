# AIOps 数据库实施方案 Runbook 技术设计

版本：1.4
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

- `status`：`READY`、`BLOCKED_BY_REQUIRED_INPUTS`、`PARTIAL_EVIDENCE`；
- `current_state`、`resolved_parameters` 与 `required_inputs`；
- `phases[].steps[]`；
- 步骤适用性：`REQUIRED`、`ALREADY_SATISFIED`、`CONDITIONAL`、`BLOCKED`；
- 命令类型：`SQLPLUS`、`RMAN`、`DGMGRL`、`SHELL`、`CONFIG`、`MANUAL`；
- `verification_commands`、`rollback`、`risks` 和 `stop_conditions`；
- 本轮证据引用。

## 3. ADG 前置取证

固定 Tool `db.ha.adg_precheck@1.0.0` 使用单条只读 Oracle SQL 汇总：

- 数据库名、`DB_UNIQUE_NAME`、实例名、主机名、服务名、平台、CDB、角色和打开模式；
- `LOG_MODE`、`FORCE_LOGGING`、Flashback、保护模式和切换状态；
- 密码文件、归档目标、FAL、文件名转换、OMF/FRA 和 Broker 参数；
- FRA 分配与已用空间；
- 联机日志线程、组数、大小；
- Standby Redo Log 线程、组数、大小；
- 每线程 `online group + 1` 的 SRL 需求和缺口。
- `COMPATIBLE`、字符集、进程和内存参数；
- 数据文件、临时文件、redo、密码文件、审计、诊断和控制文件路径样例；
- 数据文件总量和当前最大 redo group number。

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
- `REMOTE_LOGIN_PASSWORDFILE != EXCLUSIVE`：加入 SPFILE 参数整改和重启提示；
- 未配置 FRA：加入 FRA 和本地归档目标配置；
- SRL 缺口大于零：按线程和缺口数生成补建命令；
- 已满足的前置条件不删除步骤，标记为 `ALREADY_SATISFIED`，只保留验证命令；
- 主库 Data Guard 参数逐项比较，仅为缺失或不一致项生成 `ALTER SYSTEM`；
- 备库 `DB_UNIQUE_NAME`、SID、主备 TNS Alias 和 Broker 配置名按固定命名规则派生；
- 目标保护模式沿用当前保护模式，并确定性映射为 `ASYNC NOAFFIRM` 或 `SYNC AFFIRM`；
- 主库查询值、Target 连接事实、用户确认值和派生值统一进入结构化参数解析表；
- 备库主机按主机名后缀、Oracle Home 按密码文件路径或版本标准目录自动派生；
- 数据文件、redo、FRA、审计和密码文件目标路径沿用主库布局，目标环境按全新主机从零建设；
- 主库密码文件内容复制到目标环境，并使用目标 SID 文件名；
- `container_id > 1` 时路由到 Oracle 26ai DGPDB 编译器，创建独立目标 CDB、组合两端 Broker 配置并建立 standby PDB；
- CDB Root/NON-CDB 才生成整库物理备库和 RMAN Duplicate；
- 所有可复制命令必须已经代入解析值，禁止出现 `${...}` 或 `<...>` 伪可执行占位符；
- 前置证据缺失时产物为 `PARTIAL_EVIDENCE`，但仍包含完整阶段；
- Runbook 中不包含明文密码、密钥或自动切换/failover 命令。

取得前置证据后，编译器不再为备库主机、Oracle Home、存储或密码文件目标名生成
`required_inputs`。所有默认值在文档附录中标记为 `DERIVED`，便于变更评审追溯。

## 5. 回答和 UI

`DbaAnswerComposeHandler` 根据 `task_frame.implementation_profile` 编译
`AnswerBlockType.IMPLEMENTATION_RUNBOOK`。该模式不再调用模型扩写背景说明，只输出一句执行边界，
结构化 Block 是权威方案。Runbook 模式显式禁止 `PROPOSAL_SUMMARY`。

前端按正式 Markdown 文档视觉连续渲染封面信息、执行边界、目录、阶段、步骤、命令、停止条件和附录，
不再使用 `details` 折叠容器、固定最大高度或区块内滚动。目录链接定位到阶段和步骤，命令块在页面
宽度内自动换行，但复制按钮始终复制结构化 Block 中保存的原始命令文本。

结构化 Runbook JSON 是唯一真相。`application/implementation/markdown.py` 负责无状态、确定性的
Markdown 投影；页面继续读取结构化 Block 以保留复制按钮和命令类型元数据，PDF 与 Markdown 下载
均从同一 Block 即时生成，不把派生文本写回数据库。

导出接口为：

- 内部：`GET /internal/v1/aiops/conversations/{conversation_id}/turns/{turn_id}/implementation-runbook.pdf`；
- 内部：`GET /internal/v1/aiops/conversations/{conversation_id}/turns/{turn_id}/implementation-runbook.md`；
- 公共：`GET /api/v1/apps/aiops/conversations/{conversation_id}/turns/{turn_id}/implementation-runbook.pdf`；
- 公共：`GET /api/v1/apps/aiops/conversations/{conversation_id}/turns/{turn_id}/implementation-runbook.md`。

导出服务读取所属 Turn 的活动 `IMPLEMENTATION_RUNBOOK` Answer Block，并从持久化字典直接渲染 PDF，
不强制套用当前 v2 Pydantic 契约，从而兼容历史 v1 文档。PDF 使用两遍构建生成带页码和书签的
阶段/步骤目录；命令标题栏和浅色代码正文组成 Markdown 风格代码块，使用独立字符清洗逻辑保留
换行，并在空白或 SQL 分隔符处按页面宽度换行，避免越界裁切。Markdown 投影使用与命令正文不冲突
的动态 fenced code block，完整保留原始命令。两类接口复用会话所有者鉴权并禁止缓存。

## 6. 从方案升级到执行

当前实现只交付方案。未来用户选择具体步骤时，执行链必须：

1. 重新确认 Runbook 中的派生参数和外部基础设施事实仍然有效；
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
