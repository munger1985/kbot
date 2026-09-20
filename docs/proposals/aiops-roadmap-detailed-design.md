# AIOps 诊断内核与后续能力详细设计

版本：1.1
状态：部分落地（P0/P1 主链已实施）
基准日期：2026-09-20
依据：

- 产品 Roadmap [`docs/product/aiops-roadmap.md`](../product/aiops-roadmap.md) v1.4
- 现有调查设计 [`docs/product/aiops-agent-chat-diagnosis.md`](../product/aiops-agent-chat-diagnosis.md)
- 现有受控动作计划 [`docs/proposals/aiops-controlled-actions-implementation-plan.md`](aiops-controlled-actions-implementation-plan.md)
- 当前代码：Finding Compiler、Check Catalog、SQL Monitor / SQLHC / ExaCheck 近端报告、容量约束、AWR Fact、Fleet、诊断四段与逐条审批

本文把 Roadmap 落成可实施设计：先冻结共用诊断内核，再按现有实现给出改造方案。
EICC 与 ADG 演练不在本期设计范围。

实施状态：

- 已落地：共用诊断内核、Finding Card 第一屏、三入口同一套 Agent 逻辑、诊断页逐条审批、
  Check Catalog 勾选、P1 Finding 五类、SQL Monitor 官方 HTML、SQLHC Playbook、
  ExaCheck 上传解析、AWR 对比/趋势 Fact、Fleet / leadership briefing、容量约束。
- 本轮补齐：Cube 告警/巡检/运行详情只读渲染四段 blocks；P1 Finding
  `INVALID_OBJECT` / `ARCHIVE_HEADROOM` / `BACKUP_FAILED` / `LONG_TRANSACTION` / `TOP_SQL`；
  PLANNED 检查项 UI 标「规划中」且不可勾选。归档 Finding 只挂 FRA 余量，不假装有归档生成量。
- 未做：飞书、ADG 切换演练、Host Runner、真实审批联调。

## 1. 目标

把三个入口背后已经共用的调查循环，改成 DBA 可验收的诊断产品：

```text
智能诊断 / 告警诊断 / 日常巡检
        ↓  入口只负责送入问题与取证范围
同一套 Agent 诊断内核
        ↓
Finding Card → 指标/趋势/根因分析 → 解决方案 → 可选逐条动手
```

近期要解决的不是“再做一个入口”，而是：

1. 诊断第一屏先出结构化 Finding Card，不再把关键事实埋进自由 Markdown；
2. 锁、会话等取证字段补齐到能写 Finding；
3. 动手只出现在方案之后，一条 SQL 审批一条、执行一条、验证后再给下一条；
4. 自动告警/巡检默认停在方案，不进入动手；
5. 官方 SQL Monitor 报告、SQLHC 语义、ExaCheck 报告按三种方式接入同一内核，不新开脚本入口。

## 2. 与现有设计的关系

| 现有文档 | 继续有效 | 本期覆盖 |
| --- | --- | --- |
| 调查循环 Collect–Assess–Replan | 保留 | 不另做第二套规划器 |
| Tool 原子、Playbook 可选、无 Playbook 仍可调查 | 保留 | SQLHC 做成新 Playbook，不是强制门槛 |
| 受控动作：单人审批、不整批、聊天“同意”不是审批 | 保留 | 只改呈现位置和自动入口的动手开关 |
| 聊天“不要套固定根因报告模板” | 仅非诊断 Turn 有效 | **诊断 Turn 改为固定四段** |
| AWR/ASH 原生 HTML 下载 | 保留 | 分析是叠加，不是替换 |
| 动态只读 SQL、用户补证、控制面与 Executor 分离 | 保留 | 网络边界做成可验收部署约束 |

诊断 Turn 的判定不靠关键字，而靠 Task Frame：

- 含 `DIAGNOSE` / `ASSESS` / `COMPARE` / `VERIFY`，或来源是告警/巡检：走四段诊断契约；
- 仅 `UNDERSTAND` / `EXPLAIN` 且无告警巡检来源：保持自然回答，不出空 Finding、不生成 Proposal。

## 3. 现有实现基线

以下只记代码现状，不把目标态写成已交付。

### 3.1 已经能复用

- 三入口已存在：`CHAT_TURN`、`ALERT_DIAGNOSIS`、`INSPECTION`；Conversation `source_type` 为 `CHAT` / `SITUATION` / `INSPECTION`。
- 调查循环、Playbook 目录、Tool 编译执行、Evidence Artifact、Turn Answer Block、SSE 已落地。
- Oracle 只读目录 51 个 Tool，含阻塞链、活动会话、表空间、复制延迟、等待、Top SQL、XPLAN、`db.sql.plan_monitor`、AWR/ASH HTML。
- AWR/ASH/SQL Monitor HTML 已按动作实例下载：`GET .../workload-reports/{tool_id}`，tool_id 仅允许 `db.oracle.awr.report|awr.diff_report|ash.report|sql_monitor.report`。
- 用户上传 HTML 已有 `HTML_TEXT_EXTRACT`；SQLHC 作为补证，ExaCheck/ORAchk 作为主路径导入 Finding，均不能当在线取证。
- 受控动作 Catalog、Compiler、单条 Proposal、审批 Token、Executor、验证、`aiops.action-sequencer` 追加下一条 `PROPOSAL_SUMMARY` 已落地。
- `ActionIntent.NONE | ADVISORY | EXECUTE` 已区分“只诊断 / 只给语句 / 申请执行”。
- `DiagnosticProfile.SINGLE_SQL_PERFORMANCE` 已挂 `oracle.sql.healthcheck`；Finding 核心三件套仍走固定基线，官方 SQL Monitor 报告在 Playbook DAG 中。

### 3.2 必须改的缺口

| 缺口 | 代码事实 |
| --- | --- |
| 诊断输出 | `DbaAnswerDraft` 只有自由 `markdown`；`AnswerBlockType` 无 Finding；UI 把 Markdown 拼成第一屏 |
| 提示词 | `aiops_agent.answer_compose` / `answer_stream` 明确要求不要套“根因/事实/建议”模板 |
| 锁取证 | `session_blocking.sql` 只有 waiter/blocker 的 inst/sid/username、event、seconds_in_wait；无 serial#、sql_id、type、lmode、ctime |
| 锁 Playbook | `oracle.session.blocking_chain` 只跑 identity + blocking_chain |
| 巡检范围 | 计划只绑 `template_id=database_daily` 的固定 7 步；`optional_checks` 只是 override key，页面不能勾选 |
| 巡检结论 | 有行也写成“结果正常”；没有越界 Finding |
| 动手呈现 | Proposal 可出现在回答任意位置；自动入口未强制 `ActionIntent.NONE` |
| SQL Monitor | `db.oracle.sql_monitor.report` 已用 `REPORT_SQL_MONITOR` 生成官方 HTML；`db.sql.plan_monitor` 仍是 `GV$SQL_PLAN_MONITOR` 行级 Fact |
| SQLHC | 已有 `oracle.sql.healthcheck` DAG；bind/histogram 专项 Tool 仍缺，记 Gap 不中断；上传 HTML 已解析为 USER_PROVIDED 证据和可选 `SQL_STATS_STALE` Finding，不替代在线 DAG |
| ExaCheck | 上传 HTML 已解析 FAIL/WARNING 为 USER_PROVIDED Finding；INFO 留在事实表；巡检 Check Catalog 可绑 Exadata 健康包，Fire 不执行官方脚本 |
| 容量记忆 | Target 无 ASM/路径等运维事实表 |
| AWR 分析 | 能出 HTML，没有 Load Profile / Top Wait / Top SQL 的结构化对比 Tool |

## 4. 目标契约：共用诊断内核

入口、规划、取证仍走现有 Turn 状态机。改变的是 **ASSESSING 之后的回答合成**，以及诊断专用的 Finding 编译。

```text
现有：Evidence Assessor
        ↓
   DbaAnswerDraft.markdown     + 可选 PROPOSAL_SUMMARY
        ↓
   UI 先渲染 Markdown

目标：Evidence Assessor
        ↓
   Finding Compiler（确定性，不经模型编字段）
        ↓
   DiagnosisCompose
     1. FINDING_CARDS
     2. ANALYSIS_MARKDOWN     （模型只写这一段和方案段）
     3. SOLUTION_MARKDOWN
     4. PROPOSAL_SUMMARY      （仅当前一条，且仅允许动手时）
     5. HTML_REPORT_LINKS     （AWR / SQL Monitor 等原生报告）
     6. EVIDENCE_REFERENCES   （折叠）
```

模型不再负责“先出什么”。它只能在已经编译好的 Finding 和批准证据上写分析与方案，
不得改写 Finding 字段，不得编造 SQL。

### 4.1 新 Answer Block

在 `AnswerBlockType` 增加，不保留旧“纯 Markdown 诊断”双路径。历史 Turn 仍按原 block 渲染。

| block_type | schema_version | 谁生成 | 出现条件 |
| --- | --- | --- | --- |
| `FINDING_CARDS` | `AIOPS_FINDING_CARDS_BLOCK.v1` | Finding Compiler | 诊断 Turn 必有；零 Finding 也要出空集合和原因 |
| `ANALYSIS_MARKDOWN` | `AIOPS_ANALYSIS_BLOCK.v1` | 模型草稿 + 服务端校验引用 | 诊断 Turn 必有 |
| `SOLUTION_MARKDOWN` | `AIOPS_SOLUTION_BLOCK.v1` | 模型草稿 + 服务端校验 | 诊断 Turn 必有 |
| `HTML_REPORT_LINKS` | `AIOPS_HTML_REPORT_LINKS_BLOCK.v1` | 运行时从 Tool 结果投影 | 本轮生成了原生 HTML 才有 |
| `PROPOSAL_SUMMARY` | 现有 `AIOPS_PROPOSAL_SUMMARY_BLOCK.v1` | 现有 Compiler | 仅第 4 段 |
| `MARKDOWN` | 现有 | 模型 | 仅非诊断 Turn |

UI 顺序写死为上表 1→2→3→4→5，不允许 CSS/前端重排。调查计划、原始表、图表继续放在可折叠“诊断依据”。

### 4.2 Finding Card

Finding 是结构化对象，不是模型段落。

```text
AIOPS_FINDING.v1
  finding_id            本轮稳定 ID
  finding_type          目录枚举，如 LOCK_WAIT
  severity              CRITICAL / HIGH / MEDIUM / LOW / INFO
  confirmation          CONFIRMED / LIKELY / POSSIBLE / UNKNOWN
  object_ref            {target_id, instance_id, object_kind, object_name, sql_id, ...}
  fields                由 Tool 列映射的 typed map，禁止自由文本冒充字段
  threshold             {metric, current, operator, limit} 可选
  impact                短中文，来自模板，可引用 fields
  evidence_refs         本轮真实 evidence_ref
  playbook_id           可选，进入该 Finding 的专项调查
```

编译规则：

1. Finding Type Catalog 随代码发布，声明 `source_tool_id`、列映射、越界谓词、缺字段时的 `UNKNOWN`。
2. 谓词是列比较和阈值合同，不是告警名/SQL 文本关键字。
3. 零行表示“当前没有该对象”，编译为无 Finding 或 INFO“未观察到阻塞”，不得写成未取证。
4. 缺 serial# / sql_id / 阈值时，卡片仍出，对应字段显式 `null`，并记 Evidence Gap。
5. 模型不能新增 Finding，也不能改 `fields`。

P0 先落地这些类型：

| finding_type | 主 Tool | 必填字段 |
| --- | --- | --- |
| `LOCK_WAIT` | `db.session.blocking_chain` | holder/waiter 的 inst、sid、serial#、username、sql_id、lock_type、lmode、ctime、wait_event、wait_seconds、chain_depth |
| `LONG_SESSION` | `db.session.active` | sid、serial#、username、status、wait_event、wait_seconds、sql_id |
| `DG_LAG` | `db.replication.lag` | metric_name、metric_value、unit、threshold |
| `WAIT_CLASS` | `db.wait.class_summary` | wait_class、time_waited_seconds、threshold |
| `TABLESPACE` | `db.storage.capacity` | tablespace_name、used_percent、free_mb、maximum_headroom_mb、file_count、threshold |

P1 起扩展：无效对象、归档量、备份失败、长事务、Top SQL 越界。P2 扩展：SQL Monitor 超时步骤、SQLHC 统计过期、ExaCheck FAIL/WARNING。

### 4.3 分析段

服务端把 Finding 列表、证据摘要、测量语义、已确认快照/sql_id 交给模型。模型只输出分析 Markdown：

- 解释每张卡片为什么成立或为什么还不能下结论；
- 需要历史时做趋势，需要对照时做对比，不默认出全量 AWR；
- 根因只用已确认 / 很可能 / 可能 / 无法判断，且必须引用 Finding 或 evidence_ref；
- 未确认快照号、sql_id、sql_exec_id 时禁止猜测。

AWR 原生 HTML 继续按现有 Report Tool 生成；分析引用的是结构化事实或已抽出结论，不把整页 HTML 贴进对话。

### 4.4 方案段

只给可执行路径：缓解、修复、回滚、验证，或不做的理由。没有 Finding 时允许“继续观察”。
自动告警和巡检在这里结束，提供“生成正式报告 / 进入续聊”。

方案中的变更意图只能指向 Action Catalog 的 `action_template_id`，不能出现自由 SQL。

### 4.5 受控动手循环

沿用现有 Proposal 状态机和 sequencer，改三处产品规则：

```text
解决方案
  → 仅当全部成立才进入第 4 段：
       诊断来源是续聊或智能诊断
       且 TaskFrame.action_intent = EXECUTE
       且 Agent–Target 允许该动作
       且执行凭据与 Catalog 可用
  → 只渲染当前这一条 PROPOSAL_SUMMARY
  → 批准 → Executor 执行 → 同口径验证
       ├─ 成功 → sequencer 追加下一条（旧卡片变为只读状态）
       ├─ 失败/拒绝 → 停止后续
       └─ 没有下一条 → 结束，必要时 VERIFICATION_COMPARISON
```

自动告警 Run、巡检 Run 创建时冻结 `action_intent=NONE`。用户从结果页点“续聊”后，新 Turn 才允许 `EXECUTE`。

已有约束一律不改：不整批批准；聊天“同意”不是审批；命令/对象/Target 版本变化后原审批失效；`DROP` / `TRUNCATE` / switchover / failover / restore 不进执行器。

UI：诊断结果里审批卡只能出现在方案段之后；一次只展示当前可批的一条。

## 5. 三入口如何共用内核

| 入口 | 现有触发 | 改造后差异 | 相同点 |
| --- | --- | --- | --- |
| 智能诊断 | `CHAT_TURN` | 用户问题 + 可选材料；可进入动手 | 四段输出 |
| 告警诊断 | `ALERT` → Situation | SignalEvent 作为调查起点；自动 Run 停在方案 | 四段输出 |
| 日常巡检 | `SCHEDULE` + 模板步骤 | 只改取证范围：执行勾选检查项；越界行变 Finding | 四段输出 |

禁止：按入口再写一套 Planner、一套 Prompt、一套 Finding 规则。巡检报告模板继续从同一 `AIOPS_TURN_RESULT` 投影，不再在前端把“有行”写成“结果正常”。

## 6. 模块设计与改造

### 6.1 锁 / 会话取证（P0，改 Tool）

改 `db.session.blocking_chain`，不另做锁产品。

目标 SQL 语义（Oracle 19+ `GV$SESSION` + `GV$LOCK`）：

- waiter / holder：`inst_id`、`sid`、`serial#`、`username`、`sql_id`、`prev_sql_id`、`event`、`seconds_in_wait`、`status`
- 锁：`type`、`lmode`、`request`、`ctime`、`id1`、`id2`
- 链：`chain_depth`、`is_holder`

现有 `output_columns` 不兼容时，**直接升级该 Tool 的 version 和 template hash**，KBot 4.0 不保留旧列双读。Playbook `oracle.session.blocking_chain` 的 `required_privileges` 补 `GV_$LOCK`。
活动会话 Tool 补 `sql_id`，供 `LONG_SESSION` 卡片使用。

Finding Compiler 用新列映射 `LOCK_WAIT`。字段缺失则卡片仍出，值为 `null`。

### 6.2 回答合成与 UI（P0，改）

| 落点 | 改造 |
| --- | --- |
| `AnswerBlockType` | 增 `FINDING_CARDS` / `ANALYSIS_MARKDOWN` / `SOLUTION_MARKDOWN` / `HTML_REPORT_LINKS` |
| `DbaAnswerDraft` | 诊断 Turn 改为两段 Markdown：`analysis_markdown`、`solution_markdown`；Finding 不在草稿里 |
| `turn_answer_handlers.py` | 先编译 Finding，再调模型，再按固定顺序组 block；Proposal 插到方案之后 |
| `aiops_agent.answer_compose` / `answer_stream` | 诊断 Turn 取消“不要套模板”；改为“必须基于给定 Finding 写分析和方案，不得改字段” |
| `ui/aiops/js/aiops-workspaces.js` | 卡片组件；分析/方案分区；Proposal 只跟在方案后；巡检结果复用同一渲染 |
| 自动 Run | 告警/巡检创建时 `action_intent=NONE`，不跑 change handler |

非诊断 Turn 仍用现有单段 Markdown，避免把“什么是 enqueue”做成空巡检报告。

### 6.3 网络边界（P0，改部署与验收，少改业务代码）

现有控制面与 DB Executor 已分离。本期做成客户可验收约束：

- Portal / Main API / LLM 配置禁止出现数据库地址、账号、口令；
- 只有库网 Executor / Collector 持有诊断和执行凭据；
- 前端永不展示或缓存 DSN；
- 不能直连时只走监控、日志、用户补证，并在 Finding / Gap 中写明。

验收用部署清单和负向测试，不靠口头说明。

### 6.4 巡检 Check Catalog（P1，改计划模型 + 加目录）

现状：`InspectionPlan` 绑死 `database_daily` 的 7 个 `evidence_steps`。

目标：

```text
Check Catalog（随代码发布）
  check_id
  tool_id / playbook_id
  finding_types
  default_for: DAILY / WEEKLY
  trend_required: bool
        ↓
InspectionPlan.selected_check_ids   用户勾选，不能自由写 SQL
        ↓
Fire 时只编译勾选检查项为调查 DAG
        ↓
同一套 Finding Compiler + 四段输出
```

日检默认当前态；`schedule_type=WEEKLY` 自动打开趋势窗口，复用现有监控趋势字段（first/latest/change_per_day 等），不让模型重算。

客户点名的 Oracle 项按 Roadmap 第 6.2.H 分组进入 Catalog。页面从 Catalog 勾选，禁止手填 Tool SQL。

Schema：计划表增加 `selected_checks_json`（或子表），与 contracts、UI、Fire 投影、测试一起升级，不做兼容读旧 7 步。

### 6.5 Target 运维记忆与容量（P1，加表 + 改容量决策）

新表 `KBOT_OPS_TARGET_FACT`：

| 列 | 含义 |
| --- | --- |
| target_id / fact_type / fact_key | 如 `ASM_DISKGROUP` / `DATAFILE_PATH` / `TABLESPACE_PLACEMENT` |
| fact_value | 结构化 JSON |
| source | `MANUAL_CONFIRMED` / `DISCOVERED` |
| status | `ACTIVE` / `RETIRED` |

容量 Finding 用 `db.storage.capacity` + 历史趋势 + 上述事实。方案只能是观察 / `AUTOEXTEND` / `RESIZE`；动手走现有动作。没有磁盘组或路径事实时，允许聊天写入并经 DBA 确认后固化，不自动 `ADD DATAFILE`。

### 6.6 AWR 按需分析（P1，加 Fact Tool，保留 HTML）

不解析 HTML 当事实源。新增只读 Fact Tool，从 `DBA_HIST_*` 取已确认快照上的 Load Profile、Top Wait、Top SQL。

流程：

1. `db.oracle.awr.snapshots` 列出可见快照，用户或时间窗确认后才用；
2. 单次报告：现有 `db.oracle.awr.report` HTML + Fact 结论；
3. 对比两点：现有 Diff HTML + 两段 Fact 对比；
4. 一段时间趋势：快照序列上的 Fact，不猜 snapshot_id。

分析段引用 Fact 和 Finding；HTML 走 `HTML_REPORT_LINKS`。

### 6.7 官方脚本三种方式（P2）

三种方式不能混用，详见 Roadmap 第 5 节。设计落点如下。

#### 方式 A：SQL Monitor 官方报告

新增 Report Tool `db.oracle.sql_monitor.report`：

- 调用 `DBMS_SQLTUNE.REPORT_SQL_MONITOR`（或等价只读包），输出单列 `output` HTML/TEXT；
- 参数：已确认 `sql_id`，可选 `sql_exec_id` / `sql_exec_start`；未确认时先列可见执行，不猜测；
- 固定模板 + hash，模型不得把包名写进动态 SQL；
- 下载 API 从现有 workload-reports 扩展白名单，文件名 `oracle-sql-monitor-{action_id}.html`；
- 保留 `db.sql.plan_monitor` 为 Fact Tool，**不**把它当成官方报告。

用户不跑、不上传 SQL Monitor。

#### 方式 B：SQLHC 语义 Playbook

新增 `oracle.sql.healthcheck`，挂到已有 `DiagnosticProfile.SINGLE_SQL_PERFORMANCE`。

DAG（版本化，停止条件写在 Manifest）：

```text
identity
  → cursor_details
  → display_cursor
  → object_statistics
  → plan_monitor
  → sql_monitor.report
  → bind/histogram 等补齐 Tool（缺则记 Gap，不中断）
```

停止条件写在现有 Manifest 字段，不扩展 `PlaybookToolStep`：

- 某步 FAILED / 空结果：记 Gap，后续步骤继续；
- bind/histogram 等专项 Tool 尚未入目录：不写入 DAG，视为能力缺口；
- Finding 核心三件套仍由 `_single_sql_investigation_output` 固定基线取证；
- `db.oracle.sql_monitor.report` 只出现在 Playbook DAG，不进入固定基线。

不把 MOS SQLHC 放进仓库，不在目标库执行/安装 SQLT。客户已有 SQLHC HTML 时走现有上传 + `HTML_TEXT_EXTRACT`，解析为 `USER_PROVIDED` 证据和可选 Finding，**不替代**在线 DAG。

三个入口共用：有 `sql_id`、粘贴 SQL、或从锁/长会话 Finding 点进某条 SQL 时选用该 Playbook。

#### 方式 C：ExaCheck / ORAchk / TFA

近端只消费已有报告：上传或 Collector 落盘 → 结构化解析 FAIL/WARNING/INFO → Finding Card。
这是唯一以上传为主路径的官方脚本。Host Runner 与 ADG 同类，不进 P0/P1。

不能把方式 C 套到 SQL Monitor 或 SQLHC。

### 6.8 本地模型绑定（P1，改绑定校验 + Provider）

生产环境的规划/诊断模型必须绑定客户近端 DeepSeek；开发环境可以绑定 GPT。隔离只做在 AIOps Agent 的绑定和解析，不在 Model Serving 全局禁用 GPT。三个入口仍走同一套 Agent 诊断内核，Finding Card、指标分析、根因、方案和逐条审批执行都不改。

新增 Provider `local_deepseek`：

- 表示客户侧 vLLM / SGLang 等 OpenAI 兼容 HTTP 推理服务，不是进程内加载权重，也不是云端 `api_deepseek`；
- 必填 `api_endpoint` 与 `api_key`（密钥可以是占位值）；仓库不预置模型文件或真实密钥；
- 与其它 OpenAI 兼容 LLM（`api_deepseek` / `api_qwen` / `chatgpt`）走同一套 HTTP 适配器。

校验时机与错误：

- 仅 Agent 状态为 `ACTIVE` 时强制本地模型；`DRAFT` 不强制，便于先保存再补近端模型；
- 生产环境名与 `Settings.is_production()` 一致：`prod` / `production` / `live`；
- 生产启用时若模型目录客户端不可用 → `503 AIOPS_AGENT_MODEL_DIRECTORY_UNAVAILABLE`；
- 生产启用时若规划或诊断模型不是 `local_deepseek` → `422 AIOPS_AGENT_PRODUCTION_LOCAL_MODEL_REQUIRED`；
- 运行时解析规划/诊断模型再校验一次，避免目录被改成云端模型后继续跑生产诊断。

UI：Agent 模型下拉展示 Provider 中文标签，`local_deepseek` 置顶；新建时预选第一个本地 DeepSeek。Ammolite Portal 的 `local_* → runtime_type=LOCAL` 对 `local_deepseek` 必须例外，其运行方式是 `API`（`PLATFORM_BOUNDARY_ADAPTATION`）：近端推理服务仍走 HTTP，不是进程内 LOCAL 权重。

## 7. 分阶段改造方案

原则：KBot 4.0 不写兼容分支；契约、目录、UI、测试一起改。UI 只动 `ui/aiops`，不动 `integrations/apex/**`。

### P0｜诊断主链成型（先做，约 4–6 周）

目标：任意入口先看到 Finding Card；允许动手时方案后一条一条批。

| 顺序 | 工作项 | 类型 | 主要落点 |
| --- | --- | --- | --- |
| P0-1 | Finding 契约与 Compiler | 加 | `packages/platform_core/.../conversation.py`、`contracts/turn_answer.py`、新 `application/diagnosis/findings.py` |
| P0-2 | 锁/会话 Tool 列升级 | 改 | `diagnostics/catalog/oracle/sql/session_blocking.sql`、`session_active.sql`、manifest hash、单测 |
| P0-3 | 阻塞链 Playbook 权限 | 改 | `playbooks/catalog/oracle/oracle.session.blocking_chain/manifest.json` |
| P0-4 | 诊断四段合成 | 改 | `workers/turn_answer_handlers.py`、`prompts.toml`、`runtime/service.py` 组 block 顺序 |
| P0-5 | 自动入口禁止动手 | 改 | 告警/巡检 Run 冻结 `ActionIntent.NONE`；续聊才允许 `EXECUTE` |
| P0-6 | UI 卡片与审批位置 | 改 | `ui/aiops/js/aiops-workspaces.js`、`ui/aiops/css` |
| P0-7 | 网络边界验收 | 改/文档 | 部署约束、负向测试；业务链不改 Executor 分离模型 |
| P0-8 | 回归 | 测 | 锁场景智能诊断+告警诊断；零行阻塞；非诊断问答；逐条审批 |

P0 验收：

- 锁场景第一屏能看到 Holder/Waiter 的 SID、Serial、SQL_ID、锁类型、持有时间；
- 分析在卡片之后，方案在分析之后；
- 自动告警/巡检无审批按钮；续聊且授权后，方案后只出现当前一条 SQL；
- 非诊断问题仍是自然回答。

### P1｜巡检、容量、AWR 分析（约 6–10 周）

| 顺序 | 工作项 | 类型 | 主要落点 |
| --- | --- | --- | --- |
| P1-1 | Check Catalog | 加 | 新 catalog JSON + 加载器；计划 contracts/entity/UI 勾选 |
| P1-2 | 巡检 Fire 按勾选取证 | 改 | inspection Fire → 调查 DAG；取消“有行即正常” |
| P1-3 | 周趋势 | 改 | `WEEKLY` 打开趋势窗口，分析段用服务端趋势字段 |
| P1-4 | Target Fact | 加 | 新表 + API + 聊天确认写入 |
| P1-5 | 容量方案 | 改 | Finding + 事实 → AUTOEXTEND/RESIZE；无余量事实不加文件 |
| P1-6 | AWR Fact Tool | 加 | Load Profile / Top Wait / Top SQL；HTML 保留 |
| P1-7 | 本地模型绑定 | 改 | 生产默认 DeepSeek V4，开发隔离 |

### P2｜SQL 诊断包与 Exadata 报告（约 8–12 周）

| 顺序 | 工作项 | 类型 | 主要落点 |
| --- | --- | --- | --- |
| P2-1 | SQL Monitor 官方报告 Tool | 加 | 新 SQL 模板、manifest、workload-reports 白名单 |
| P2-2 | SQLHC Playbook | 加 | `oracle.sql.healthcheck` 挂 `SINGLE_SQL_PERFORMANCE` |
| P2-3 | SQLHC HTML 补证 | 改 | 现有上传解析 → USER_PROVIDED Finding，不替代 DAG |
| P2-4 | ExaCheck 报告导入 | 加 | FAIL/WARNING 解析器 → Finding；巡检可绑该包 |
| P2-5 | Dashboard / 飞书 / PG·MySQL | 加 | 不改诊断内核 |

### P3｜独立功能（不回头改内核）

ADG 切换演练、Host Runner、`ADD DATAFILE` 余量契约完整后再开放。不在本设计展开。

## 8. 数据与契约变更

P0 尽量不改表：Finding 放在 Answer Block `payload_json`，Compiler 纯内存。

P1 必改 Schema（与代码同发，不写迁移兼容）：

- `KBOT_OPS_INSPECTION_PLAN` 增加勾选检查项；
- 新增 `KBOT_OPS_TARGET_FACT`；
- Manifest / contract_version 随表结构递增。

P2 不强制新表：SQL Monitor 沿用 Tool Artifact；ExaCheck 沿用上传/Collector 证据。若巡检要绑定“Exadata 健康包”，复用 Check Catalog，不另做产品表。

## 9. 测试要求

不靠关键字列表伪装 Finding。至少覆盖：

1. 阻塞链有完整字段 → `LOCK_WAIT` 卡片字段与行一致；
2. 阻塞链零行 → 无锁 Finding，说明当前无阻塞，不是未取证；
3. 缺 `GV_$LOCK` → 卡片仍出，锁类型/LMODE 为 `null`，有 Gap；
4. 告警自动诊断无 `PROPOSAL_SUMMARY`；续聊授权后方案后才出现；
5. 上一条未验证前 API 不能放出下一条 Proposal；
6. 解释类问题不出现空 Finding 段；
7. AWR 未确认快照只列目录，不调用 report；
8. SQL Monitor 未确认 `sql_id` 不调用 `REPORT_SQL_MONITOR`；
9. 巡检勾选子集时，未勾选项不得出现在 DAG 和报告；
10. 上传 ExaCheck 可出 Finding；上传 SQLHC 只补证，在线 DAG 仍执行。

## 10. 明确不做

1. 不拆三套诊断内核，不新开“跑官方脚本”入口。
2. 不取消 AWR/ASH HTML；不把 HTML 当唯一事实源。
3. 不把 MOS SQLHC/SQLT/ExaCheck 脚本放进仓库或目标库执行。
4. 不把“用户跑完再上传”套到 SQL Monitor / AWR / SQLHC。
5. 不把 `GV$SQL_PLAN_MONITOR` 行数据当成官方 SQL Monitor 报告。
6. 不自动执行 switchover / failover / restore；ADG 不进 P0/P1。
7. 不支持整批批准；聊天文本不是审批。
8. 不做 EICC。
9. 不改 `integrations/apex/**`。

## 11. 建议实施切面

P0/P1 主链与官方报告接入已经落地。后续只补飞书、ADG 演练、Host Runner 和真实审批联调，不再回头改内核。
