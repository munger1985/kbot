# 三库 SQL 深入诊断执行计划一致性交付记录（2026-11-09）

## 交付行为

Oracle、PostgreSQL、MySQL 的单条 SQL 深入诊断统一保留模型正文，在正文后展示执行计划卡片。成功采集时默认显示图形，支持原始计划切换、节点详情、缩放与全屏；采集失败或没有匹配计划时，同一位置显示真实状态，不画空图，不要求用户重复提供本可采集的材料。

图形组件保持两项目一致，保留各数据库原生算子、成本与估算/运行统计的差异。统一展示不意味着把估算计划声明为已发生的执行，也不改写历史回答。正文流出后不因附加计划卡片而被撤回或替换。

## 已定位并修复的采集与投影问题

- PG 固定计划工具补齐安全的 EXTRACT 与只读主备/WAL 状态函数。归一化统计文本中的 EXTRACT($n FROM ...) 仅按 AST 和词法位置转换为 PostgreSQL 14+ 原生 pg_catalog."extract"($n, ...)，保持 numeric 返回类型，不猜字段参数、不全量重写 SQL、不更改原文哈希。
- PG 的实际 PREPARE 语法失败返回 SQL_PLAN_STATEMENT_UNSUPPORTED，不再统一误报内部错误。
- MySQL 优先读取活动连接计划；活动结束时，读取 performance_schema 真实完整 QUERY_SAMPLE_TEXT，在数据库内复核 STATEMENT_DIGEST、Schema、完整性和前后样本一致性，进行只读 EXPLAIN FORMAT=JSON。不执行原 SQL、不执行 ANALYZE、不猜参数、不放宽动态 SQL 工具权限。
- MySQL 真实身份仅包含 server_uuid 而没有启动代次。仅对可信本轮固定工具的 DATABASE_ESTIMATE、无运行统计、主体与原文哈希一致且近时点同服务器的材料使用 SOURCE_SERVER_MATCH，保留启动代次未知说明。不将它用于历史执行证明、运行统计、人工材料或精确对象解析。
- MySQL EXPLAIN 产生的 1003 Note 可能含完整 SQL 和字面值。固定计划游标只记录警告编号，不输出原文，数据库异常仍正常抛出。
- MySQL nested_loop 数组只投影一个连接节点，保留各输入的原生顺序和估算值，不伪造连接类型、时间或运行统计。补充 1/2/3 输入、物化子查询、隐私、重放、非法成员和节点/深度上限测试。

## 真实数据库复测

两个项目均使用现有目标诊断凭据和正式工具，经过 Driver、单元值归一化、证据投影、主体匹配及最终回答块生成：

| 数据库 | 实际 SQL 主体 | 节点数 | 计划来源 |
|---|---|---:|---|
| PostgreSQL | -1330044169105867050 | 1 | DATABASE_ESTIMATE |
| PostgreSQL | -6589067333046971749 | 16 | DATABASE_ESTIMATE |
| MySQL | ba848b025362483ff8381d9dd5078163f45d05a35f88febb02aec028baeb5aa4 | 5 | DATABASE_ESTIMATE |
| Oracle | 49fbb8nm4fkr2 | 24 | DATABASE_CURSOR |

全部生成 PRIMARY / SQL_EXECUTION_PLAN / plan_graph。MySQL 物理来源范围为 SOURCE_SERVER_MATCH，其余为 SOURCE_MATCH。

该复测不调用模型、不保存新会话、不重放业务 SQL。它验证真实工具到最终回答图块的链路，不等同于已完成线上浏览器与 LLM 全会话验收。历史已保存的回答不会自动补图。

## 验证记录与边界

- 最终代码：两个项目各 239 项计划专项单元测试通过。
- 两项目架构边界检查通过。
- 两端执行计划浏览器专项 43 项通过。
- 将真实数据库的四份最终图块分别加载到两端浏览器，8 次图形/原始切换与节点数核验通过，无页面脚本错误，并审阅真实 MySQL 连接图。
- KBot 执行计划及 Top SQL 表格/筛选 JavaScript 合同通过。
- 本轮前序 Ammolite 本地与线上 Portal 构建均通过；后续只有计划领域投影和测试变化。
- 全页面对照检查仍报告案例资源状态文案和 runbook 人工步骤等差异，未独立做修改前基线复现，不能宣称整套页面回归通过。本次不改动这些无关功能。

## 发布范围

本地开发后提交并推送，服务器仅快进拉取，不在线修改源码。保留两仓用户既有文档修改和服务器未跟踪配置备份。

两个项目的 19 个 AIOps Prompt 已按开发基线 1.0.0 定向同步并核验运行时哈希，Ammolite 来源 BUILTIN、KBot 来源 FILE_SEED；不覆盖网页人工维护版本、不改写冻结证据。

代码提交：
- Ammolite：0de01c76、1523c637、96e0ad3e。
- KBot：521643de、2227be72、9e422841。

Ammolite 使用既有 aiops-agent 定向重启；KBot 使用既有全量重启脚本。最终进程、端口和当前在线日志以发布后健康核验结果为准。
