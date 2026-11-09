# SQL 深入诊断计划采集修复（2026-11-09）

## 真实问题

MES (demo) 的 PostgreSQL SQL `-6589067333046971749` 在 16:30（Asia/Shanghai）的 Ammolite 深入分析中，已调用一次固定计划工具，但返回 `SQL_PLAN_FUNCTION_UNSUPPORTED`。安全解析器将 `AND` 连接条件归为函数校验，导致尚未执行 EXPLAIN 就拒绝。最终回答未解释实际失败，转而要求 DBA 提供执行计划。

## 修复范围

- 固定计划工具区分布尔、CASE/WHEN 语法节点和真正函数调用。函数白名单及子节点检查保持有效，`"and"(...)`、`"if"(...)` 等同名函数不能绕过。
- 线上继续验证发现，归一化统计 SQL 经 PREPARE 重新解析后产生不同 Query Identifier，导致有效计划被拒收。固定工具依同连接复核的完整统计主体与原文本哈希绑定来源，重新解析标识不能覆盖原 Query ID；人工上传原生标识仍严格核对。
- 分区表用 NULL 作为协议占位时，即使强制 generic 也可能在 EXPLAIN 初始化阶段裁剪分区子计划。PG16+ 使用原生 `GENERIC_PLAN TRUE` 保留完整子计划；PG14/15 如发生裁剪，返回明确失败，不猜实际参数、不展示残缺计划冒充完整计划。
- 计划采集状态及具体失败回执在模型生成前进入诊断输入，并保留在回答元数据。不将工具失败描述为未尝试、没有计划或无权限。
- 统一规划及 SQL 诊断 Prompt：优先使用已授权固定工具；PostgreSQL 可取得不执行原 SQL 的通用估算计划，MySQL 只读匹配活动连接计划，Oracle 读取匹配缓存计划。
- 已有估算计划应先分析其节点，不能要求先提供生产实际计划才进行分析。仍禁止 ANALYZE、重放业务 SQL、猜测业务参数、自动提权及模型自拼 EXPLAIN/PREPARE。
- 不增加生成后改写、模型重试或流式正文撤回。普通诊断不自动生成操作稿；测试显式请求 PLAN 后才验证操作稿。

## 验证边界

本轮无 Schema、权限或目标业务数据修改。Prompt 文件基线保持 `1.0.0`，开发环境定向原地同步且保留管理员版本，不改写历史运行。

最终专项回归覆盖安全解析、失败回执、计划导入/展示/优先取证、诊断、流式输出、动态查询及 Prompt：Ammolite 366 项通过（另 14 个子测试），KBot 360 项通过（另 9 个子测试）；两仓架构边界和 `git diff --check` 通过。

更大范围 AIOps 回归未全部通过：两仓各 8 项既有失败已在修改前快照复现（Top SQL 计算测试、MySQL 原生字段模板哈希、旧 SQL 测试上下文缺 run_id）；KBot 另 2 项既有 Schema 工件断言失败也已在修改前复现。当前本地 Python 环境缺 pytest-asyncio，另 22 个异步通知测试不能执行。本轮未将这些结果计为通过，未扩展修改其生产逻辑。

## kbotdev 发布与真实取证

- 两项目均从已推送提交快进更新，未在线修改仓库源码。Ammolite 定向重启 AIOps；KBot 使用现有全量重启脚本，均返回成功。
- 两项目本轮 5 个 Prompt 基线定向原地更新到 `1.0.0`，19 个 AIOps Prompt 均经运行时 Hash 核验；无管理员版本被覆盖。
- 两项目分别使用现有 MES (demo) 目标诊断凭据，经正式 PostgreSQLDiagnosticDriver 和固定目录工具读取相同 SQL 的计划，均成功返回 **16 个节点**：Limit、Sort、Gather、Append 与 12 个分区 Seq Scan。来源为 `DATABASE_ESTIMATE`，有同连接物理身份和原 SQL 哈希，`runtime_statistics=NONE`。
- 未执行原 SQL、ANALYZE、猜测真实参数或修改历史 Artifact。这是数据库通用估算计划，不是生产代表参数的实际行源统计。未把直接工具验收冒充完整浏览器 LLM 会话验收。
- 本地 Ammolite Prompt 同步成功；本地 KBot 的 Oracle 连接不可达，未完成本地 Prompt 数据库同步。线上 KBot 同步及运行时解析核验已完成。
- 发布后 10 个 Main API / AIOps 就绪端点均为 200；两项目各 4 个 AIOps 进程及 Main API 存活。当前在线日志中，KBot 未见近期业务 ERROR；Ammolite Main API 在 08:52:21 UTC 有一条 `aiops.report.ready` 通知 Outbox 隔离记录，本轮计划验收不产生报告通知，未将此记录宣称为已修复或无错误。Ammolite 重启产生的 4 个 SIGTERM 退出已单独识别。
