# AIOps DBA Dashboard 详细设计

版本：1.0
状态：已实施
基准日期：2026-09-29

## 1. 目标与边界

Dashboard 是 DBA 登录 AIOps 后的运维态势入口，替代原“库群总览”。它不复制监控系统，
也不在首屏展开 SID、SQL_ID 或原始时序明细，而是让 DBA 在十秒内回答：

1. 现在是否有必须立即处理的数据库；
2. 哪些高重要程度 Target 风险最高；
3. 是否存在不可达、数据过期或完全无证据的监控盲区；
4. 容量、复制、备份、会话与锁的已确认风险集中在哪里；
5. 最近 24 小时自动诊断和巡检是否可靠完成；
6. 下一步应进入告警、诊断详情还是智能运维会话。

本功能使用现有 Target、Situation、Run、Inspection Fire 和 Finding 数据，不新增数据库表，
不改变诊断主链，也不提供 `/fleet` 路由或静态页面兼容层。

## 2. 现状审计与重构结论

原库群总览主要是 Target 列表和摘要，存在以下问题：

- 页面命名偏资产盘点，不能表达 DBA 的值班决策入口；
- 没有把严重告警、不可达和数据盲区形成统一优先队列；
- 缺少明确的新鲜度语义，容易把无数据误读为健康；
- 容量与复制延迟容易被混成一个泛化风险值；
- 最近自动化结果和可下钻活动没有形成闭环；
- `/fleet` 及 Fleet 契约把旧信息架构固化在接口层。

因此采用完整替换：删除 Fleet 页面、脚本、契约和路由，建立独立 Dashboard 投影与页面。

## 3. 信息架构

```text
Dashboard
├── 清单筛选：名称、环境、数据库类型、最低 Level、健康状态、仅关注项
├── 当前态势：一句结论 + 5 个可下钻 KPI
├── 优先处理队列：最多 8 项，直接进入处理页面
├── 库群健康分布：健康语义的完整分布
├── 风险热点：容量 / 复制 / 备份 / 会话与锁
├── 最近 24 小时运维结果：自动诊断与日常巡检分开统计
├── 最近活动：最多 12 条可下钻状态变化
└── 数据库清单：当前筛选下的可操作明细表
```

页面采用高密度运维布局。异常、证据时间和下一步动作优先于装饰性图形；表格保留横向滚动，
小屏改为单列信息区，不通过折叠容器隐藏核心状态。
清单筛选只影响数据库明细表；态势、优先队列和风险热点始终覆盖当前 Domain 全部 Target，
避免低 Level Target 的严重异常被默认清单筛选隐藏。

## 4. 健康与数据新鲜度语义

### 4.1 健康状态

| 状态 | 含义 | 判定优先级 |
| --- | --- | ---: |
| `CRITICAL` | 存在严重告警，或最新 Finding 中存在高严重度关键风险 | 1 |
| `UNREACHABLE` | 连接状态不可达/配置错误，或启用只读连接后探活为 DOWN | 2 |
| `WARNING` | 有普通未关闭告警、最近自动诊断失败或中高风险 Finding | 3 |
| `STALE` | 有证据，但最后证据时间超过 24 小时 | 4 |
| `UNKNOWN` | 没有可用于判断的观测、诊断或告警证据 | 5 |
| `DISABLED` | Target 已停用 | 6 |
| `HEALTHY` | 有新鲜证据且没有上述异常 | 7 |

不可达优先于告警和 Finding。只有 `CRITICAL` 告警会把 Target 提升为 `CRITICAL`；普通未关闭
告警进入 `WARNING`。`UNKNOWN` 永远不能计入健康。

### 4.2 数据新鲜度

数据新鲜度独立输出 `CURRENT`、`STALE`、`UNKNOWN`。证据时间取 Target 最近观测、最新完成
诊断和未关闭 Situation 最近观测中的最大值。默认过期阈值为 24 小时，并通过
`stale_after_seconds` 返回，避免前端自行猜测。

“数据盲区”同时包含 `STALE` 和 `UNKNOWN`。Dashboard 必须分别展示二者，但汇总筛选可一次查看
全部盲区。

## 5. 排序与决策规则

优先处理队列按以下稳定顺序排序：

1. 健康状态优先级；
2. Target `importance_level` 从 L5 到 L1；
3. 告警或 Finding 严重度；
4. 异常开始时间，持续更久的在前；
5. Target 名称与 ID，用于稳定分页和测试。

每个 Target 只生成一个首要处理项，来源依次为不可达、最高告警、最近自动化失败、最高风险、
数据新鲜度。全量风险仍保留在风险热点表中。

## 6. 风险语义

风险热点只消费最新完成诊断的结构化 Finding，不从自由 Markdown 提取数字：

| 分类 | Finding |
| --- | --- |
| 容量 | `TABLESPACE`、`CONNECTION_USAGE`、`ARCHIVE_HEADROOM` |
| 复制 | `DG_LAG`、`REPLICATION_LAG` |
| 备份 | `BACKUP_FAILED` |
| 会话与锁 | `LOCK_WAIT`、`LONG_SESSION`、`LONG_TRANSACTION` |

容量百分比和复制延迟保持独立字段与单位。无法安全解析具体值时显示“已确认”，不得构造估算值。

## 7. API 与投影契约

公开入口：

```http
GET /api/v1/apps/aiops/dashboard
```

内部入口：

```http
GET /internal/v1/aiops/dashboard
```

返回 `OpsDashboard`，主要结构为：

- `generated_at`、`stale_after_seconds`；
- `summary`：首屏汇总；
- `health_distribution`：完整健康分布；
- `attention_items`：最多 8 项；
- `risk_items`：最多 20 项；
- `automation`：最近 24 小时精确状态计数；
- `recent_activities`：最多 12 项；
- `targets`：域内全部 Target 行。

接口只按当前 Domain 隔离数据，不按当前 Agent 过滤。Agent 绑定仍在进入智能运维会话后校验。

## 8. 数据访问与性能

Dashboard 是数据库批量投影，不对每个 Target 发起实时 Prometheus、Alertmanager 或数据库连接：

1. 一次读取域内 Target；
2. 一次读取未关闭 Situation；
3. 使用窗口函数一次取得每个 Target 最新完成 Run；
4. 一次读取最近 24 小时有限 Run，供最近活动展示；
5. 使用窗口函数一次取得每个 Target 最近 24 小时的最新失败 Run，失败提示不受活动条数截断；
6. 分别按状态聚合 Run 和 Inspection Fire，统计不受活动列表条数限制；
7. 一次批量读取最新 Run 的 Finding Blocks；
8. 在应用层完成健康、优先队列、风险与活动投影。

该规则用于避免 N+1 和首屏加载时对监控系统形成放大流量。页面默认每 60 秒刷新，也允许关闭或
改为 30 秒、5 分钟。

## 9. 页面交互与深链

- KPI 和健康分布直接设置全量 Target 表筛选，并滚动到列表位置；
- “数据盲区”筛选同时匹配 `STALE` 和 `UNKNOWN`；
- 告警链接使用 `situations.html?target_id=...&situation=...`，页面预设 Target 并打开指定告警；
- 诊断链接使用 `run-detail.html?id=...`；
- 智能运维链接使用 `chat.html?target_id=...`，页面预选 Target 后按现有 Agent 绑定继续操作；
- 自动化失败 KPI 定位到最近 24 小时运维结果，不伪造一个不存在的失败列表接口。

页面禁止使用 `scrollIntoView`，统一根据页面坐标调用 `window.scrollTo`，避免破坏嵌套滚动区。

## 10. 安全与可访问性

- 页面只调用 Main API 公共 BFF，不出现内部路由、数据库凭据或监控凭据；
- Dashboard 不返回 SID、SQL_ID、SQL 文本或连接串；
- 所有动态文本在插入 HTML 前转义；
- 健康状态同时使用中文文本、边框和颜色，不只依赖颜色；
- 表单、状态区和表格保留语义标签，自动刷新按钮可手动关闭；
- 无数据、读取失败和无匹配筛选分别显示真实状态，不用示例数据填充。

## 11. 验收标准

1. 导航只有 `Dashboard`，不存在 Fleet 页面、脚本、路由或兼容别名；
2. 没有证据的 Target 显示 `UNKNOWN`，不计入健康；
3. 超过 24 小时的证据显示 `STALE`；
4. 不可达优先于告警，普通告警只产生 `WARNING`；
5. 优先队列遵守健康、Level、严重度、持续时间排序；
6. 容量和复制风险不共用字段或单位；
7. 最近 24 小时统计来自聚合查询，不受活动列表上限影响；
8. 告警、诊断和智能运维深链可定位到对应上下文；
9. 后端单元测试覆盖健康、新鲜度、排序、风险和批量查询；
10. 静态页面契约、JavaScript 语法、OpenAPI 快照和 Python 编译检查通过。
