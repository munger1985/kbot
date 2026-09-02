# AIOps 已登记受控动作设计与实施计划

版本：1.0

状态：阶段 0、1、阶段 2 的普通/分区 Index Rebuild，以及阶段 3 的会话、对象编译、表统计信息、Scheduler 和本地应用用户账号状态首批代码切片已实施；其余动作和真实环境验收待实施

基准日期：2026-09-02

## 1. 目标

把当前仅支持 `db.session.terminate` 的变更能力扩展为覆盖 DBA 日常运维领域的受控动作
平台。Agent 只能从随代码发布的 Action Catalog 中选择动作；非破坏性动作在一位有权用户
明确批准后执行，破坏性动作只生成命令、风险和验证说明，由 DBA 离开 KBot 执行。

本方案复用现有 Change Proposal、HITL、一次性 Approval Token、Mutation Grant、独立执行
凭据、Outbox、DB Executor、执行回调和处理后验证链路，不新增 OA、变更单、双人审批或逐级
审批流程。

## 2. 已确认的业务规则

1. “允许修改数据库”统一改称“允许审批后执行已登记受控动作”。
2. 一位拥有 `aiops:proposal:approve` 权限的用户显式批准一次，就是 KBot 内的最高业务授权；
   不要求发起人与审批人分离，也不增加第二审批人。
3. 每条实际执行命令单独形成 Proposal。多动作方案严格串行，上一条执行并验证完成后，才
   生成下一条待审批 Proposal；不支持整批批准。
4. 审批只能来自 Proposal 上的明确按钮和版本化命令，聊天中的“同意”“执行吧”等文本不
   构成审批。
5. 审批通过后仍必须验证 Catalog、Proposal Hash、Target 版本、对象、参数、执行凭据和执行
   前条件。这些是执行安全围栏，不是额外审批流程。
6. Agent、模型、用户粘贴内容和回答 Markdown 都不能提供自由 SQL 给 Executor。模型只能
   推荐动作意图，确定性 Compiler 只能把可信数据库事实绑定到已登记模板。
7. `DROP`、`TRUNCATE`、归档清理及其他会删除数据、对象、恢复介质或不可逆替换数据库状态的
   破坏性操作不支持自动执行。系统只展示人工执行命令、风险、前置检查和恢复建议。
8. DBA 人工执行后可以回填结果；Agent 使用只读诊断凭据重新取证、验证效果并生成对比报告。

## 3. 能力边界

### 3.1 动作执行模式

Action Catalog 中每个动作必须固定声明一种执行模式：

| 执行模式 | 含义 | 页面行为 |
| --- | --- | --- |
| `EXECUTABLE_AFTER_APPROVAL` | 非破坏性且已有受控执行器 | 展示“批准并执行”和“拒绝” |
| `MANUAL_ONLY` | 破坏性操作，或当前没有安全执行器 | 只展示命令、风险和人工结果回填 |
| `UNSUPPORTED` | 数据库类型、版本或能力不支持 | 只给出原因，不生成可执行命令 |

审批模式固定为单人审批，不在动作模板中配置审批人数、审批层级或 OA 流程。

### 3.2 破坏性的结构化判定

不能只用 SQL 关键字判断是否破坏性。Action 定义必须声明 `effect_class`，动作专用 Validator
还要解析命令语义并确认两者一致。以下效果类别只能使用 `MANUAL_ONLY`：

- `DATA_DELETION`：删除或截断业务数据；
- `OBJECT_DELETION`：删除表、索引、用户、表空间、数据文件或其他数据库对象；
- `RECOVERY_MATERIAL_DELETION`：删除归档日志、备份或恢复所需介质；
- `STATE_REPLACEMENT`：恢复、`RESETLOGS`、强制故障切换等会不可逆替换数据库状态的操作；
- `ARBITRARY_MUTATION`：不能由结构化动作准确限定范围的 DDL、DML 或 PL/SQL。

Catalog 加载、Proposal 创建和 Executor Claim 三个阶段都必须拒绝把上述动作转换为
`EXECUTABLE_AFTER_APPROVAL`。即使配置、数据库记录或请求被篡改，也不能进入 Mutation Driver。

## 4. DBA 日常运维动作目录

所有日常运维领域都进入统一目录，但“纳入目录”不等于“全部自动执行”。Oracle 先实现完整
纵向切片，MySQL 使用相同语义提供数据库方言 Variant；PostgreSQL 在成为公开可管理 Target
之前只保留只读诊断能力。

| 运维领域 | 首批动作 | 执行方式 |
| --- | --- | --- |
| 会话与事务 | 取消当前 SQL、断开会话 | 单人审批后执行 |
| 索引与统计信息 | rebuild/coalesce index、分区索引修复、收集/锁定/解锁统计信息 | 单人审批后执行；不支持删除索引 |
| 对象维护 | 编译指定对象、编译 Schema 无效对象 | 单人审批后执行；对象检查保持只读 |
| Scheduler | 启用、停用、运行、停止指定 Job | 单人审批后执行 |
| 存储容量 | 增加或扩容 datafile/tempfile、调整 autoextend | 单人审批后执行；删除文件只供人工执行 |
| 参数与资源 | 修改明确 Allowlist 内的动态参数、切换已登记 Resource Manager Plan | 单人审批后执行 |
| 用户与权限 | 锁定/解锁用户、过期密码、明确对象和权限的 grant/revoke | 单人审批后执行；删除用户只供人工执行 |
| 备份维护 | 发起备份、校验、crosscheck | 单人审批后调用登记的执行器；删除备份只供人工执行 |
| 恢复与清理 | restore/recover、`RESETLOGS`、删除备份、清理归档 | `MANUAL_ONLY`，只展示命令和恢复说明 |
| 高可用 | 启停日志应用、计划内 switchover | 单人审批后调用登记的执行器；强制 failover 只供人工执行 |
| 实例和服务 | PDB、数据库服务、实例和监听器启停 | 单人审批后调用对应数据库或主机执行器 |
| 补丁升级 | 预检查、补丁应用、升级 | 单人审批后调用已登记的 OEM、OCI 或自动化执行器；破坏性回退只供人工执行 |
| 破坏性数据操作 | `DROP`、`TRUNCATE`、任意 DML、删除分区或清空数据 | `MANUAL_ONLY`，永不进入 Executor |

当前 Catalog 共登记 53 个数据库 Variant：16 个审批后可执行 Variant、7 个只展示命令的
`MANUAL_ONLY` Variant，以及 30 个明确失败关闭的 `PLANNED/UNSUPPORTED` Variant。后者已经
覆盖 Schema 批量编译、datafile/tempfile 增加/扩容/autoextend、参数和 Resource Manager、
精确 grant/revoke、备份发起/校验/crosscheck、日志应用、PDB/服务/实例/监听器和补丁升级等
目录项，并记录各自缺失的可信事实或外部执行协议；它们不会出现在可勾选执行动作中。

破坏性目录除表 `DROP`/`TRUNCATE` 和归档清理外，还登记了 restore、recover、备份删除和物理
备库 failover 的固定人工命令。以上动作的 `executor_kind` 均为 `NONE`，Catalog 加载和 Executor
Claim 会双重阻止其进入自动执行链。任意 DML 和“其他 SQL”仍不提供通用入口。

每个动作都必须有独立的数据库类型和版本 Variant。不能用一个“执行 SQL”动作代替上述目录。

## 5. Agent 与 Target 配置

### 5.1 配置契约

删除单一的 `allow_change_execution` 布尔语义，改为每个 Agent–Target 绑定的显式动作策略：

```json
{
  "controlled_action_execution": {
    "enabled": true,
    "allowed_action_ids": [
      "db.index.rebuild",
      "db.statistics.gather",
      "db.scheduler.job.run"
    ],
    "object_scopes": {
      "schemas": ["APP_SCHEMA"],
      "exclude_system_objects": true
    },
    "max_daily_executions": 10
  }
}
```

`allowed_action_ids` 必须由用户从当前 Target 的兼容动作中显式选择。Catalog 新增动作时不能自动
进入已发布 Agent 版本，避免一次总开关隐式授权未来全部动作。Policy 和 Agent 版本继续不可变；
修改动作范围必须生成新版本，已经创建的 Run 继续使用冻结快照。

不配置审批层级、OA 单号或双人审批。可保留对象范围、并发数、每日执行上限和全局 Kill Switch
作为技术安全约束。

### 5.2 有效能力计算

Run 创建时按以下交集冻结有效动作：

```text
Agent–Target 显式选择
∩ Catalog 中当前启用的动作
∩ 数据库类型、版本、环境和能力
∩ Target 已配置的执行凭据能力
∩ 部署级 Mutation 开关
```

页面同时展示“已配置动作”和“当前可执行动作”，并逐项说明不可执行原因。

## 6. Action Catalog 改造

### 6.1 契约扩展

当前参数只支持 `integer` 和 `enum`，需要增加受约束的结构化类型：

- `database_object_ref`：数据库、容器、Schema、对象类型和对象名；
- `identifier`：经过方言 Validator 校验的单个标识符；
- `boolean`、`size`、`duration`、`timestamp`；
- 受限字符串：必须声明长度、格式和允许字符，不能接收 SQL 片段；
- 同一动作所需的 Evidence 字段、对象范围和参数来源。

Action 定义还要增加：

- `action_family`、`effect_class`、`execution_mode`和`executor_kind`；
- 执行前检查、执行后验证和稳定等待时间；
- 锁影响、预计时长、幂等等级、是否可取消；
- 所需凭据能力和最小数据库权限；
- 对应的 Compiler、Renderer、Validator 和 Verifier 标识。

### 6.2 渲染与校验

移除面向所有动作的字符串替换和单一正则表达式，改为按动作注册的方言 Renderer。值参数优先
使用数据库 Bind；DDL 中不能 Bind 的对象标识符必须来自已经验证的 `database_object_ref`，由
Renderer 按数据库方言安全引用。

每个 Renderer 输出规范命令、参数 Hash 和命令 Hash。Catalog Manifest、SQL 文件和实现版本
继续参与 Catalog Hash，审批后任何变化都会使原 Proposal 失效。

## 7. 从诊断到动作的确定性编译

把当前硬编码的阻塞会话提取器替换为 `ActionCompilerRegistry`。每个动作 Compiler 声明所需
Tool、Evidence 类型和参数绑定规则，只消费当前 Turn 中 `SOURCE_VERIFIED` 的数据库事实。

例如 `db.index.rebuild` 在形成 Proposal 前必须取得并冻结：

- Target、容器、Schema、索引、基表和可选分区的精确身份；
- 索引状态、类型、大小、分区状态和当前 DDL；
- 数据库版本以及 ONLINE、并行和压缩能力；
- 所需临时空间、可用空间、活动会话和锁影响；
- 推荐 Variant、执行前条件和执行后验证基线。

用户输入的索引名只能作为调查线索。Agent 必须先从数据库重新读取对象事实，不能直接把用户文本
拼入命令。

## 8. Proposal、审批和多动作计划

Proposal 增加动作族、执行模式、效果类别、规范对象引用、执行器类型、锁影响和预计时长。审批
页面必须展示精确对象、规范命令、风险、执行前条件、预期效果、验证方案和不可逆说明。

`EXECUTABLE_AFTER_APPROVAL` 的状态流为：

```text
PENDING_APPROVAL
  → 单人 APPROVED
  → CREATED
  → SUBMITTED
  → RUNNING
  → SUCCEEDED / FAILED / UNKNOWN
  → VERIFYING
  → VERIFIED / NOT_ACHIEVED / ADVERSE / INCONCLUSIVE
```

`MANUAL_ONLY` 不创建 Approval Token、Execution 或 Mutation Grant，也不显示批准按钮：

```text
ADVISORY_READY
  → DBA 人工执行
  → 回填 EXECUTED / FAILED / CANCELLED
  → 只读验证
  → 对比报告
```

Action Plan 可以包含多条动作，但运行时一次只创建当前 ordinal 的 Proposal。当前动作验证成功后
才释放下一条；失败、结果未知或验证不通过时停止后续动作，由用户重新诊断。

## 9. Executor 与凭据

### 9.1 执行器注册表

把 Oracle/MySQL Driver 中对 `db.session.terminate` 的硬编码替换为动作执行器注册表：

- 数据库内动作使用 `DATABASE` Executor；
- RMAN、监听器、实例、补丁和高可用动作使用显式登记的 `EXTERNAL` Adapter；
- Adapter 只接受签名 Grant 中的类型化动作和参数，不接受 Shell、脚本正文或自由命令。

长时间动作需要异步租约、心跳、进度、取消和超时后的 `UNKNOWN` 处理。非幂等动作在网络断开后
禁止自动重试，必须先通过只读工具确认数据库实际状态。

### 9.2 最小权限凭据

一个 Target 可以按能力配置多组独立执行凭据，例如会话管理、对象维护、空间管理、账号权限和
备份高可用。运行时按动作所需能力签发一次性凭据；不能让所有动作共用一个高权限账号，也不能
回退使用只读诊断凭据。

## 10. 动作专用验证

把当前只适用于会话终止的验证器替换为 `ActionVerifierRegistry`。每个动作必须登记执行前和执行
后只读 Tool，并保存 Before/After Evidence。例如：

- Index Rebuild：对象状态、分区状态、大小、可用性和无效标记；
- Statistics Gather：统计信息时间、采样设置和目标对象状态；
- Datafile Resize：文件大小、表空间容量和告警状态；
- Parameter Set：内存值、SPFILE 值、实例范围和是否需要重启；
- Scheduler：Job 启停状态和最新运行结果；
- Switchover：数据库角色、保护模式、传输和应用状态。

执行器返回成功但验证失败时，结果必须是 `NOT_ACHIEVED` 或 `ADVERSE`，不能报告处理成功。

## 11. Schema、API 与 UI 改造

### 11.1 Schema 和 Repository

- Target Binding 或 Agent Version–Target Policy 保存显式动作和对象范围；
- Change Proposal 保存 `effect_class`、`execution_mode`、`executor_kind`和规范对象引用；
- 现有 `solution_group_key + command_ordinal` 承担多动作顺序，Action Plan Artifact 保存完整计划；
- 单人 Approval Token、Execution 和结果 Artifact 继续复用；
- Repository 和 Unit of Work 继续拥有事务，API/Application 不直接访问 Session 或 SQL。

KBot 4.0 直接更新规范 Oracle DDL、Entity、初始化、重建文件和 Manifest，不增加兼容列、双写或
迁移适配层。

### 11.2 API

- 增加只读 Action Catalog 和 Target 有效动作查询；
- Agent 创建/修改使用按 Target 的 `controlled_action_execution`；
- Proposal 返回执行模式、效果类别、对象、影响和验证计划；
- 审批 API 只接受 `EXECUTABLE_AFTER_APPROVAL`；
- `MANUAL_ONLY` 继续复用人工结果回填 API，不增加“批准但不执行”状态。

### 11.3 UI

- Agent 页面按 Target 展示并选择动作，而不是一个总开关；
- 新动作不会自动勾选；
- 可执行 Proposal 显示“批准并执行”；
- 破坏性 Proposal 显示“仅供人工执行”，提供复制命令、风险确认和结果回填，不显示批准按钮；
- Run 时间线统一展示建议、审批、执行、验证和报告，但不模拟 OA 流程。

## 12. 实施阶段

截至 2026-09-02，阶段 0、阶段 1 的通用契约、Agent–Target 显式授权、单人审批、人工结果
回填、多动作逐条释放、Schema/API/UI，以及阶段 2 的普通和分区 Oracle Index Rebuild
纵向切片已经完成代码改造和离线回归。两类重建均从只读事实校验对象类型、表空间余量和
活动表锁，并在执行前重新确认精确对象；coalesce 和真实 Oracle 故障场景仍属于阶段 2 的
扩展实现与环境验收范围；
阶段 3 已完成 `db.session.cancel_sql`、`db.object.compile`、`db.statistics.gather`、
`db.statistics.lock`、`db.statistics.unlock`、`db.scheduler.job.run`、
`db.scheduler.job.enable`、`db.scheduler.job.disable`、`db.scheduler.job.stop`、
`db.user.lock`、`db.user.unlock` 和 `db.user.password.expire` 的离线代码切片。
对象编译支持 Oracle 19c+ 的 `PROCEDURE`、`FUNCTION`、`PACKAGE`、`PACKAGE BODY`、
`TRIGGER`、`VIEW`、`TYPE` 和 `TYPE BODY`；用户动作排除 Oracle 维护用户和 common user。
其余阶段 3、4 目录条目明确标为 `PLANNED/UNSUPPORTED`，不能配置成可执行动作。
阶段 5 未开始。

建议由两名后端、一名前端和一名兼职 DBA 按以下顺序投入。排期是设计评估，不是交付承诺；
真实数据库和外部系统联调情况会直接影响阶段 3、4 的时长。

| 阶段 | 建议周期 | 主要交付 |
| --- | --- | --- |
| 阶段 0 | 1～2 周 | 动作清单、破坏性分类、契约和验收样例 |
| 阶段 1 | 3～5 周 | 通用 Catalog、授权、Compiler、Executor、Verifier、Schema/API/UI |
| 阶段 2 | 2～3 周 | Oracle Index Rebuild 真实纵向切片 |
| 阶段 3 | 6～10 周 | Oracle 常用 DBA 动作 |
| 阶段 4 | 6～10 周 | MySQL Variant 与 OEM/OCI/RMAN 等 Adapter |
| 阶段 5 | 2～4 周 | 非生产验收、生产逐动作启用和运行手册 |

完整范围预计为 4～6 个月。若先只交付通用平台、Index Rebuild 和第一批 Oracle 常用动作，
可以把首个生产候选版本控制在约 8～12 周；不能以 Mock 测试通过压缩真实数据库验收阶段。

### 阶段 0：动作盘点与契约冻结

- 与 DBA 确认 Oracle/MySQL 动作清单、破坏性效果分类和首批优先级；
- 冻结 Action、Agent–Target Policy、Proposal 和执行结果契约；
- 为每个动作记录版本、权限、前置 Tool、验证 Tool、执行器和人工/自动边界。

门槛：全部日常运维领域都有明确动作或 `MANUAL_ONLY` 条目；不存在“其他 SQL”兜底动作。

### 阶段 1：通用动作平台

- 改造 Action 契约、Registry、Renderer、Validator；
- Agent 按 Target 显式选择动作，修复 Catalog 扩容后隐式授权风险；
- 引入 Compiler、Executor 和 Verifier 注册表；
- 完成单人审批、人工动作和多动作串行状态机；
- 更新 Schema、OpenAPI、UI 和配置文档。

门槛：未知动作、未选动作、Catalog 变化、对象越界和破坏性动作均 Fail Closed。

### 阶段 2：`db.index.rebuild` 纵向切片

- 增加索引详情、空间和锁影响只读 Tool；
- 支持普通、ONLINE、分区和 coalesce Variant；
- 完成可信事实编译、逐条审批、执行、结果回调和只读验证；
- 对 DROP INDEX 和其他对象删除进行 `MANUAL_ONLY` 回归验证。

门槛：真实 Oracle 环境完成正常、前置条件变化、权限不足、空间不足、超时、断线结果未知和验证
失败场景。

当前进度：普通及分区重建、ONLINE 选择、空间/锁事实、执行前复核和执行后状态验证已完成离线
代码切片。`db.index.coalesce` 已登记为 `PLANNED/UNSUPPORTED`；在取得可通用、只读且可信的
碎片判定依据前，不生成 coalesce 命令或审批入口。

### 阶段 3：Oracle 常用动作

- 会话取消/终止；
- 统计信息、对象编译和 Scheduler；
- datafile/tempfile 扩容与 autoextend；
- Allowlist 动态参数、用户锁定/解锁和精确 grant/revoke；
- 备份、校验、crosscheck、日志应用和计划内 switchover；
- PDB、服务、实例和监听器管理 Adapter。

门槛：每个动作有独立的参数注入、权限、陈旧前置条件、幂等、未知结果和验证测试。

当前进度：`db.session.cancel_sql` 已使用专用 `db.session.current_sql` 事实完成 Compiler、精确命令、
执行前 SQL_ID 复核和执行后“SQL 消失但会话仍存在”验证。`db.object.compile` 已使用专用
`db.object.status` 事实完成离线切片：仅把状态为 `INVALID` 的 Oracle `PROCEDURE`、
`FUNCTION`、`PACKAGE`、`PACKAGE BODY`、`TRIGGER`、`VIEW`、`TYPE` 或 `TYPE BODY`
编译成精确 `ALTER ... COMPILE` 命令，执行前重查对象身份、类型和
`INVALID` 状态，执行后要求同一对象为 `VALID`；对象消失记为 `ADVERSE`。
`db.statistics.gather` 已完成单表统计信息收集切片：只接受非临时、统计未锁定且统计缺失或
`STALE_STATS=YES` 的表，固定使用 `AUTO_SAMPLE_SIZE`、列直方图 AUTO、级联索引和
`AUTO_INVALIDATE`，不接受模型提供采样参数或 PL/SQL；执行后要求 `LAST_ANALYZED` 非空且统计
不再过期。`db.scheduler.job.run` 已完成指定 Job 运行切片：只接受已启用且处于 `SCHEDULED`
的 Job，冻结执行前运行/失败计数，执行后要求 Job 正在运行，或运行计数增长且失败计数未增长。
Scheduler enable/disable/stop 使用各自的状态候选 Tool，不从同一状态猜测相反意图；同一对象、
同一动作族出现多个候选时以 `AMBIGUOUS_ACTION_INTENT` 失败关闭。统计锁定/解锁同样使用独立候选
事实并在执行前、执行后复核状态。用户锁定、解锁和密码过期仅接受专用 `DBA_USERS` 事实，排除
`ORACLE_MAINTAINED='Y'` 及 `COMMON='YES'`，且不接收、生成或记录任何密码。Schema 批量编译
仍不提供通用入口。以上阶段 3 切片均尚未完成真实 Oracle 权限、资源消耗、锁等待、运行失败和
断线场景验收。

尚未激活的动作不是普通编码遗漏：datafile/tempfile 扩容需要数据库外存储余量和文件系统/ASM
事实，参数和授权动作需要可信的目标值/权限意图契约，RMAN、Data Guard、实例、监听器和补丁
动作需要已确定的 OEM、OCI、Ansible 或其他外部执行协议、Target 定位和最小权限凭据。上述
依赖未登记前保持 `PLANNED/UNSUPPORTED`；不得用自由 SQL、自由 Shell、用户文本回显或 Mock
成功结果替代这些契约。

### 阶段 4：MySQL Variant 与外部 Adapter

- 为相同业务动作增加 MySQL 方言和能力 Variant；
- 接入已登记的 OEM、OCI、RMAN 或自动化 Adapter；
- 保持相同的 Proposal、单人审批、一次性 Grant 和验证语义。

门槛：数据库方言不能串用，外部 Adapter 不能执行请求携带的自由 Shell 或 SQL。

### 阶段 5：生产启用

- 默认保持部署级 `agent_execution_enabled=false` 和 `mutation_enabled=false`；
- 先在非生产 Target 对首批动作逐项验收；
- 生产只启用已经完成真实依赖测试的动作；
- 核对执行凭据最小权限、Kill Switch、审计、监控和 `UNKNOWN` 人工处置说明；
- 分动作启用，不因 Catalog 发布自动扩大任何 Agent 的权限。

## 13. 测试与验收矩阵

每个动作至少覆盖：

1. Catalog、命令文件和实现 Hash 一致；
2. 数据库类型、版本、环境、能力和执行凭据匹配；
3. 只有 `SOURCE_VERIFIED` 事实能够绑定参数；
4. 标识符、对象范围和所有参数注入攻击被拒绝；
5. 未经一次明确审批不能执行，审批只能消费一次；
6. Target、Policy、Catalog、参数或前置条件变化使审批失效；
7. 破坏性效果只能产生 `MANUAL_ONLY`，任何路径都不能进入 Executor；
8. 执行成功、失败、超时、网络中断和结果未知状态准确；
9. 执行后重新取证，报告区分执行成功和问题解决；
10. 多动作逐条审批、验证和失败停止；
11. Domain、Agent、Target 和审批权限隔离；
12. Oracle/MySQL 真实依赖 Smoke，不用 Mock 结果冒充上线验收。

## 14. 完成定义

- 页面和 API 不再把一个总开关描述为任意数据库修改权限；
- Agent–Target 只拥有用户显式选择且当前兼容的动作；
- DBA 日常运维领域均有可执行、只供人工执行或不支持的明确状态；
- 所有可执行动作都经过一次人工审批、一次性授权、真实执行和动作专用验证；
- `DROP`、`TRUNCATE`、归档清理等破坏性操作只有人工命令和结果回填路径；
- 不存在任意 SQL、任意 Shell、聊天文本审批或 Catalog 新增即自动授权；
- Schema、初始化、Entity、Repository、API、OpenAPI、UI、测试和文档保持一致；
- 真实 Oracle/MySQL 和外部 Adapter 验收完成后，才允许逐动作启用生产执行。
