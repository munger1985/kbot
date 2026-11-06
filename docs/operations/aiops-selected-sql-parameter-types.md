# 已选 SQL 续问的参数类型边界

日期：2026-11-06。

## 现象与根因

线上 08:24:22 的 PostgreSQL SQL 续问已沿用原数据库资源和 SQL 行引用，
但调查计划在执行工具前被拒绝：`database_oid 必须是整数`。
冻结主体中的 `database_oid`、`user_oid` 为精确保真的十进制字符串。
首轮选行调查已规范化这些值，后续 Target 绑定却用原始证据值重新覆盖工具参数。

## 修复边界

- Target 绑定复用 `normalize_sql_subject`，按数据库引擎验证 SQL 主体，
  仅用规范化后的值覆盖工具输入。
- PostgreSQL 的 OID、Oracle 的游标数值维度恢复为整数；
  PostgreSQL `query_id` 等 SQL 标识继续保留字符串，不经浮点转换。
- 冻结选择、来源引用、原始证据和任务主体的精确传输表示不改写，
  不重新指定数据库，不放宽工具目录、授权、来源完整性或行引用校验。
- 布尔 OID、小数、负数、越界数值和跨引擎主体仍拒绝。
- 无 Schema、公开接口或 Prompt 变更。

## 回归与上线

`tests/unit/aiops_agent/test_aiops_sql_followup_budget.py` 使用真实任务编译和工具目录，
覆盖三种数据库、整数与精确字符串表示、完整 Oracle 游标，以及无效维度拒绝。
本轮同时运行 SQL 来源快照、统计主体、诊断及开放查询相关回归。

部署代码并重启受影响的 AIOps Worker 后，应从原 SQL 结果再次选择并诊断。
已有失败轮次不改写；单元回归通过不代表线上 SQL 分析已经完成。
