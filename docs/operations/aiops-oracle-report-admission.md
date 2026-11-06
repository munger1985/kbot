# Oracle 原生报告的执行边界

AWR、AWR Diff 和 ASH 仅支持 Oracle，实例选择限定为 Agent 已授权业务系统内的
Oracle 目标，不能通过客户端参数扩大授权范围。

## 职责边界

商业许可不属于 AIOps 的管理职责。AIOps 不维护 Oracle Pack 授权声明，不要求
管理员勾选 Diagnostics/Tuning Pack，也不根据声明缺失禁用功能或阻止执行。
固定工具、Playbook、动态 SQL 和受控变更模板均不设置商业许可门禁；模型可见
能力目录与执行器使用同一套技术能力边界，不伪造或自动添加任何许可声明。

以下执行边界继续保留：

- Agent 资源授权、Oracle 数据库类型、目标启用状态、连接及受管凭据。
- 工具版本、目标库能力、参数、只读 SQL AST、对象和函数范围。
- 数据库实际账号权限；缺少视图或包执行权限时按真实错误反馈。
- 签名 Grant、资源额度、执行超时、结果上限，以及受控变更的人工审批。

AIOPS_STARTER_TOOLS_UNAVAILABLE 表示当前技术能力目录不能执行所选功能。
检查数据库版本、只读连接和工具目录，不要求维护商业许可声明。该业务错误
不可重试，关联 Turn/Run 收敛为失败终态，安全日志不展开用户输入或凭据。

## 报告时间绑定

AWR 快照查询在 SQL 中使用 `SYS_EXTRACT_UTC` 明确返回真实 UTC 时间，不依赖
驱动为无时区值推断偏移。绑定器按实际时刻比较快照边界，不忽略 UTC 偏移，也
不把不同秒的时间按分钟截断。补全报告动作时，同一窗口的 `Z`、毫秒和 `+08:00`
写法按实际时刻去重，避免重复 AWR 动作干扰 AWR Diff 的窗口配对。

ASH 的浏览器输入支持 UTC `Z`、毫秒时间及带偏移的 ISO 8601 时间，在参数校验时统一为
UTC、六位小数秒和显式偏移，保留亚秒边界。SQL 使用匹配的
`YYYY-MM-DD"T"HH24:MI:SS.FFTZH:TZM` 格式；不再把浏览器时间直接交给只接受
秒级加空格偏移的旧模板。规划与 Executor 重复校验必须得到同一参数和哈希。

## 验证范围

回归测试位于：

- `tests/unit/aiops_agent/test_aiops_starter_admission.py`
- `tests/unit/aiops_agent/test_aiops_conversation_starters.py`
- `tests/unit/aiops_agent/test_aiops_discovery_binding.py`
- `tests/unit/aiops_agent/test_aiops_turn_retry.py`

覆盖三种报告在没有任何许可声明时的入口开放、规划和执行，以及能力目录缺失、
自然语言和 UI 提交、Oracle 类型筛选、多目标汇总、Outbox 单次失败及 Turn/Run
终态。正常路径加载真实目录，覆盖快照绑定、Tool DAG 编译、冻结工具版本、
签名 Grant、Executor 参数校验、HTML 结果归一化和产物下载。数据库边界使用模拟
驱动，不连接 Oracle、不调用原生报告包；线上产出仍需部署后在实际目标上验收。

本修复不涉及持久化 Schema、初始化、数据库迁移或 Prompt 版本变更。
