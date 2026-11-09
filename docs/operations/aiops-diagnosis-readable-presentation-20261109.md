# 诊断目标名称与状态语义展示

## 展示规则

- 新建 Run 时冻结 Target 的 display_name；回答模型通过 target_presentation 使用该名称，
  不在正文输出 Target UUID、内部资源或绑定标识。运行重试、后续编译不刷新原目标名。
- 缺少名称或名称所属 Target 不匹配时使用“所选数据库”，不猜测名称、不用 UUID 充当名称。
  实际数据库名、实例名、容器名及 SQL ID、query_id、digest 等定位信息仍保留其原始语义。
- 监控展示值逐证据行读取 semantic_role、value_kind、unit；
  AVAILABILITY 的 1/0 显示“可用/不可用”，BACKUP_LAST_RUN_STATUS 显示“成功/失败”。
  不以状态单位或 0/1 数值推断未知指标的业务含义。
- 非法或缺失状态显示“未知状态/未取得有效状态”，不得视为正常。
  状态编码均值、分位数不翻译成状态或自动换算可用率；
  已观测状态仅描述有效采样，不将最新可用扩大为全窗口连续健康。
- 统一提供指标中文名及展示字段标签；原有精确数值、单位换算和窗口口径保持不变。

## 验证与边界

两平台同步覆盖目标名称冻结、名称不匹配、状态语义、混合可用状态、未知值、精确证据不变、
单次模型生成与流式原文一致，以及真实 Run 创建到 Top SQL 编译保留目标名的回归。
结构化回答、普通流式回答与完整诊断回答均传入目标展示及数值展示策略。

不使用 UUID 正则替换或事后正文改写，不清空或重新生成已交付回复，
不改写原始证据及历史回答，也不改变诊断状态或执行授权。
实现提交阶段仅修改仓库代码和 1.0.0 Prompt 基线；后续线上部署记录如下。

## kbotdev 部署验收（2026-11-09）

- 两仓功能提交均已推送并在线上快进拉取：Ammolite `072f2f47`（dev），
  KBot `58eb3275`（kbot4.1.1）。未在线修改源码，未重置或清理工作区；配置备份全部保留。
- 两套开发数据库各核对 19 个 AIOps Prompt，其中各有 8 个文件基线落后，已定向原地同步并激活：
  `turn_router`、`investigation_planner`、`investigation_policy_repair`、
  `investigation_replanner`、`answer_compose`、`answer_stream`、
  `diagnosis_answer_compose`、`sql_diagnosis`（均为 `aiops_agent.` 前缀）。
- 版本维持 `1.0.0`，Ammolite 来源为 `BUILTIN`，KBot 来源为 `FILE_SEED`；
  本次未发现需要跳过的手工维护版本。未执行全库初始化、DDL、清除历史或改写历史运行。
  数据库激活指针与文件 Hash 一致，分别通过正常 PromptResolver 核验全部 19 项实际解析结果。
- KBot 使用标准 `restart_kbot.sh` 全量重启，新 AIOps/Main API 进程启动于 `05:01:42 UTC`；
  Ammolite 使用 `restart_ammolite.sh aiops-agent main-api` 定向重启，实际 Python 进程启动于
  `05:03:35 UTC`。Ammolite Main API 停止超过脚本的 10 秒等待，由标准脚本强制结束后重新拉起。
- 两套 Main API 的 `/readyz`，以及 AIOps API、DB Executor、Worker、Scheduler 的 `/ready`
  共 10 项均返回 HTTP 200 / ready；核对了新进程 PID、所属仓库和监听端口。
- 验收使用线上 development/logs API。本次重启后的 KBot AIOps 错误日志样本未见新错误，
  但 Main API 有 Webhook 路由不存在及运行投影对账警告，不将这些链路视为已验收通过。
- Ammolite 仍存在重启前已出现的自动采集失败：`UntranslatableCharacterError` /
  `unsupported Unicode escape sequence`；同时通知 Outbox 写入 operation watcher 时引用的用户不存在，
  触发 `ForeignKeyViolationError`。这些问题未在本次部署中修改或清理，不能据 readiness 宣称业务全链路正常。
- 未以真实用户发起新的模型对话，未宣称已完成浏览器端完整回答验收。
  资料目录修复不等于历史全文索引缺列已处理，本次未执行相关 AIOps Schema 变更。
