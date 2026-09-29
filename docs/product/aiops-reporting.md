# KBot 4.0 AIOps 正式报告与导出设计

## 目标与边界

正式报告服务于用户汇报和可追溯复盘，而不是替代三个业务入口中的实时
诊断界面。智能运维、告警诊断和日常巡检继续是唯一的业务入口；报告模板
属于资源配置能力，不增加第四个业务入口。

三个入口必须复用同一条报告链路：来源适配器取得经过授权的业务事实，报告
生成器冻结 `ReportContext`，模板解析器选择版本化模板，文档渲染器提供预览
和 PDF 下载。入口层不得自行拼接 Markdown、HTML 或 PDF。

自动告警诊断只创建 Situation、Run、证据和诊断结论，绝不自动创建 READY 或
PARTIAL 报告。用户可直接选择生成报告，或者进入智能运维继续取证后再生成。
本版本所有正式报告均由用户明确发起：Run 完成只冻结诊断结果，用户选择模板后才
创建正式报告。告警诊断绝不自动生成报告；巡检月报、季报和年报也只汇总已完整闭合的
自然时间窗，不能用单次 Run 冒充。

## 统一模型

```text
Chat Run / Alert Situation + Run / Inspection 时间窗
                    ↓
             ReportContext
                    ↓
       TemplateResolver + ReportAssembler
                    ↓
  REPORT_CONTENT.v1（内容、来源、模板均冻结） + ReportSource
                    ↓
       ReportPresentation / PDF ReportExport
```

`ReportContext` 至少包含单一 Target、入口类型、报告时间窗、来源链路、结论、
事实、证据引用、数据缺口、建议、处置和验证状态。每项证据保留来源定位符、
采集时间、可信等级与内容哈希；禁止写入凭据、完整连接串、令牌和未授权原始
日志。

月度、季度和年度报告不是重命名单次巡检。它们按报告时区汇总同一 Target 在
自然月、自然季度或自然年中的巡检、关联告警、诊断、处置和验证结果。巡检
执行频率与报告周期独立，例如每天巡检、每月出一份月报。

## 两类模板的职责边界

模板管理明确拆成两类，二者不能复用同一实体或同一用途：

- 巡检模板决定“检查什么”。用户只能从受控 Check Catalog 勾选检查项；发布时冻结目录哈希、
  检查项 ID 和固定取证步骤。巡检计划只引用一个确定的模板版本，不再保存模板字符串、解析器
  版本或自行勾选检查项。模板发布新版本不会静默改变既有计划，只有用户重新选择模板时计划才
  切换版本。
- 会话报告模板决定“怎么展示”。它只适用于完整 Session 生成的正式报告，用户可选择和排列
  受控章节；执行摘要与证据边界始终保留。它不决定诊断工具、取证步骤、巡检范围或调度规则。

告警和巡检正式报告使用只读系统版式，不允许把会话展示模板用于自动诊断或巡检执行。

## 系统报告版式

系统预设模板如下：

| 模板引用 | 适用入口 | 周期 |
| --- | --- | --- |
| `system:diagnosis.standard` | 智能运维、告警诊断 | 单次诊断 |
| `system:inspection.daily` | 日常巡检 | 当日或当次 |
| `system:inspection.monthly` | 日常巡检 | 自然月 |
| `system:inspection.quarterly` | 日常巡检 | 自然季度 |
| `system:inspection.annual` | 日常巡检 | 自然年 |

系统版式不可修改。Session 自定义报告模板可版本化，使用受控章节 DSL，只能排列摘要、
范围、风险、发现、根因、建议、处置与验证和证据附录等固定章节；不接受 SQL、脚本、任意
HTML、任意表达式或外部 URL。根因等级、数据缺口、证据边界和内容哈希为强制信息，任何
模板都不能隐藏。

生成报告时，系统将解析后的模板定义、模板引用、版本和模板哈希写入报告内容。
模板之后被更新或停用不影响已发布报告的预览与再次导出。

## 用户流程

### 智能运维

会话中没有进行中的 Turn，且至少一个 Turn 已形成终态诊断结果时，会话底部显示唯一的
“生成正式报告”入口。用户选择适用模板后，系统冻结该 Session 全部已结束 Turn 的诊断
结果、证据引用和数据缺口；不会为每个 Turn 单独生成报告。任何 Turn 未形成可报告终态
结果时，报告明确记录相应的数据缺口并以 PARTIAL 状态发布。

### 告警诊断

已完成的自动诊断显示“生成报告”和“进入智能运维”。前者由用户显式发起，
后者通过 `source_situation_id`、`source_run_id` 继承原告警的 Target、时间窗、
事实和证据。自动告警结果未完成时，报告按钮不可用；后台不得创建报告。

### 日常巡检

巡检详情在结果结束后显示“生成正式报告”。用户选择日常、月度、季度或年度模板；
周期模板以该结果所属 Target 和计划时区计算最近一个已经闭合的自然周期，汇总该期
巡检结果、失败数和数据缺口，再生成、预览和下载报告。

## API、状态和权限

模板管理与同步生成、下载 API 如下：

```text
GET  /api/v1/apps/aiops/inspection-templates
POST /api/v1/apps/aiops/inspection-templates
POST /api/v1/apps/aiops/inspection-templates/{template_id}/versions
GET  /api/v1/apps/aiops/session-report-templates
POST /api/v1/apps/aiops/session-report-templates
POST /api/v1/apps/aiops/session-report-templates/{template_id}/versions
GET  /api/v1/apps/aiops/report-layouts
POST /api/v1/apps/aiops/reports:generate
GET  /api/v1/apps/aiops/reports/{report_id}/presentation
GET  /api/v1/apps/aiops/reports/{report_id}/pdf
```

生成请求仅可选择来源 Session 或来源 Run 和模板，不可提交事实正文。智能运维使用
`conversation_id`，告警和巡检继续使用 `ops_run_id`；二者必须且只能选择其一。所有写请求带
`Idempotency-Key`；模板新版本带并发版本校验。当前报告状态为 `READY`、`PARTIAL`
或 `FAILED`，PDF 根据冻结内容同步渲染，导出不会修改正式报告。

读取、生成、预览和下载均校验 Domain、Target 与来源 Agent 的 Domain 归属。报告
生成、读取和下载使用既有 `aiops:use`，自定义模板配置沿用 `aiops:plan_manage`。

## 持久化与审计

正式报告使用不可变 Artifact 和 Report 投影；生成时冻结模板定义与哈希。`ReportSource`
逐条记录产生报告的来源 Run、来源产物哈希和采集时间，周期报告可关联多个 Run；Report
投影中的 `OPS_RUN_ID` 仅是用户发起生成时的锚点。PDF 依据冻结内容同步生成，不保存
正文副本。所有规范 DDL、实体、Repository、OpenAPI 和测试同步更新，不保留兼容读写。

## 验收

1. 三入口均调用同一个报告生成器和文档渲染器。
2. 告警自动结束后没有新增报告或导出，只有用户操作才会生成。
3. 告警续聊生成的报告可追溯 Situation、自动 Run、Chat Run 和证据。
4. 月、季、年窗口按 IANA 时区稳定计算，覆盖数、失败数和缺口可复现。
5. 模板变更不改变历史报告；PDF 正确呈现中文、分页、证据索引和内容哈希。
6. 跨 Domain、来源 Agent 不属于当前 Domain、来源未结束、模板不适用和重复请求均有确定行为。
