# 诊断前主动补证与建议文档内容类型修复（2026-11-08）

## 线上现象与原因

SQL `1285382359565720997` 对应调查 `01a1b9f3-01a1-7ea4-8c12-a40fc2e534aa`：
本轮只调用身份探测与 PostgreSQL 语句详情工具，另复用上一轮窗口统计；没有发起计划、索引定义等采集。
模型正常生成一次，并非再次超时。最终建议中的 VERIFY 补证项被 Runbook 编译器过滤，
验证与回退说明被包装为 MANUAL 命令，页面与导出又显示为代码和复制按钮。
Runbook 在没有缺失参数时固定标为 PARTIAL_EVIDENCE，也未使用真实调查完成状态。

## 本轮变更

- 所有诊断入口（含 SQL 深入分析、健康诊断快捷入口及普通调查）在最终正文前评估补证路径。
  评估读取当前授权目录、数据库只读目录查询、监控/日志能力及本轮尝试回执。
  关键证据不足且存在尚未尝试的路径时先补充调查；不把“上下文没有”解释为“数据源没有”。
  完整零行仍是已完成检查，不为可选基线扩大健康诊断目标。
- SQL 计划优先采集从 Oracle 扩展到 PostgreSQL 和 MySQL，机械续取仍经过正式编译、Grant、
  只读凭据兑换、Executor、原始观察落库和来源重放；不让模型直接提交任意业务 SQL。
- PostgreSQL 14–19：按 Query ID、数据库 OID、用户 OID、top_level 和原文本哈希重新定位统计主体。
  在同一只读事务中 PREPARE 原语句，并强制 force_generic_plan 后只执行 EXPLAIN EXECUTE。
  EXECUTE 的 NULL 仅满足接口形式，不能参与特化；没有执行原 SQL，也没有 ANALYZE。
  只接受单条可解析 SELECT、已限定对象及安全函数；原业务 search_path、缺失绑定类型不猜测。
  权限不足、文本变化、跨库、无法安全解析、超时等均保留实际采集回执，不自动授权。
- MySQL 8–9：按 Digest、Schema 和文本哈希匹配当前活动执行，读取 EXPLAIN FORMAT=JSON FOR CONNECTION，
  前后核对 thread/event/connection 身份；执行结束或改变时拒绝关联。没有活动、多个未唯一绑定活动、
  权限不足等明确返回不可得；不执行 Digest 文本，不用问号猜参数。
- 新计划来源 DATABASE_ESTIMATE 与 Oracle 缓存、用户上传、历史独立审批估算分开。
  只保留安全节点、估算数值及同连接物理来源，不保留谓词、绑定值、SQL 文本或实际运行统计。
  报告/案例从原始固定工具 Artifact 独立重建；节点解释引用真实 plan_nodes_json 原始字段。
- 计划获取状态区分 NOT_ATTEMPTED、FAILED、NO_MATCHING_PLAN、SUCCEEDED；
  对实际失败保留错误码，对未执行保留“尚无采集回执”，不把这两种情况混同。
- Runbook 保留 VERIFY/OBSERVE 步骤，验证说明和停止/回退条件使用独立文字字段。
  根据实际 GOAL_ANSWERED 或缺失参数设置文档状态，不再固定标为证据不完整。
  诊断建议文档不等于执行命令包；真实 SQL 草案仍可复制，但不携带执行授权。
  Portal、KBot 聊天/非聊天、Markdown/PDF 使用一致语义；现有 MANUAL+MANUAL 内容也按说明呈现。
- 保留上一轮正文交付终点规则：补证评估不输出正文；最终正文开始后不再重规划、重写或撤回。

## 验证与发布边界

最终定向回归：Ammolite 607 项及 89 项子测试通过；KBot4 598 项及 78 项子测试通过；
两端浏览器 38 项通过。两仓架构边界、KBot 内容合同/JS 检查、Portal 类型检查及生产构建通过。
生产构建仍提示已有大 chunk 警告，未作为本次任务扩大拆包改动。

定向测试覆盖：三库计划优先采集、主体与文本绑定、零行/失败去重、安全语句、MySQL 执行竞争、
估算计划来源与篡改重放、计划解释字段定位、建议文档状态与内容类型、
健康检查空结果、快捷入口、流式终态保护、诊断 Executor 和取消/超时边界。

浏览器验证覆盖 Portal 与 KBot 的说明/SQL 草案、真实复制按钮、计划图/原始树、
桌面/移动布局及既有流式正文不撤回。
另运行两仓架构边界、Portal 类型检查/生产构建、KBot JS 与文档内容合同及 diff 检查。

本地隔离 PostgreSQL 18 使用真实 pg_stat_statements 和正式只读 Driver 实测：
取得 Aggregate + Index Only Scan 两节点；采集前后原 SQL 调用次数不变，文本哈希及物理身份均可核验。
MySQL 本轮仅进行受控 Driver/竞争场景回归，没有连接线上 MySQL 做实测。

无应用数据库 Schema 或 Prompt 基线变更，不覆盖管理员 Prompt，不改写历史冻结调查。
只提交本任务相关代码、测试及本文；未在 kbotdev 修改仓库、拉取代码或重启服务。
