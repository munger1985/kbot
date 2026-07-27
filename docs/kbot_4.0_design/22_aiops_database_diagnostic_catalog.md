# 4.0 AIOps 数据库诊断目录与 Dialect

步骤 6 的可实施级组件、Grant、Driver、限界和验收设计见 [35_aiops_step6_readonly_database_diagnostics.md](35_aiops_step6_readonly_database_diagnostics.md)。

## 核心决策

直连数据库时，LLM 只判断“需要查什么”，输出类型化 `DiagnosticAction`；`DatabaseDialect` 决定“实际执行哪条 SQL”。系统不把 LLM 自由生成的 SQL 交给直连数据库自动执行。

```text
Diagnosis Planner
      ↓ DiagnosticAction(tool_id, parameters)
Diagnostic Catalog
      ↓ db_type + version + capabilities
Oracle/MySQL DatabaseDialect
      ↓ versioned SQL template + bound parameters
Read-only DB Executor
      ↓
DatabaseObservation Artifact
```

## 诊断契约

LLM 只能输出：

```text
DiagnosticAction {
  action_id
  tool_id
  target_id
  parameters
  reason
  required_evidence
}
```

`tool_id` 必须来自当前 Target 可用目录，`parameters` 必须通过 JSON Schema。Planner 不输出 SQL、凭据、连接信息或未登记的工具名。

`DiagnosticToolDefinition` 至少包含：

```text
tool_id, version, title, purpose, category
supported_db_types, min/max_db_version
required_capabilities, required_privileges
parameter_schema, output_schema
template_ref, template_hash
timeout_seconds, max_rows, cost_level
sensitive_columns, fallback_tool_ids, status
```

## 存储与包结构

诊断目录是 AIOps Agent 的版本化部署资产，不是 APEX 可编辑业务数据。建议结构：

```text
services/aiops_agent/src/aiops_agent/
  diagnostics/
    contracts.py
    registry.py
    validator.py
    dialects/
      oracle/
        manifest.yaml
        sql/
        parsers.py
      mysql/
        manifest.yaml
        sql/
        parsers.py
```

Manifest 和 SQL 随代码评审、测试和发布。数据库只在 Run/Artifact 中记录 `tool_id + version + template_hash + parameters_hash`，便于审计和复现。

## Oracle/MySQL 能力集

首期诊断目录按能力组织，不按用户问句组织：

- 连接、实例状态和数据库角色；
- 会话、长事务、锁等待和阻塞链；
- CPU/DB Time、等待事件、Top SQL 和执行计划；
- 表空间/磁盘、内存、临时空间和容量趋势；
- Oracle 归档、RAC/Data Guard 以及 MySQL 复制/InnoDB；
- 错误日志、参数差异和关键能力可用性。

OEM 深度指标属于 `MonitorProvider`，不冒充数据库 SQL 诊断。Oracle/MySQL 目录可对同一 `tool_id` 提供不同模板和输出解析器。

## 模板渲染和执行护栏

- 数值、时间、文本参数使用数据库 bind variable，禁止 f-string 直接拼接；
- 表名、视图名和排序字段等不能 bind 的标识符只能从模板白名单选择；
- Executor 使用只读账号、只读会话/事务、statement timeout、行数上限和 Target 并发上限；
- 执行前校验 `template_hash`、Target 版本和所需能力，不接受请求传入的 SQL 文本；
- 权限不足、视图不存在或版本不支持时，Executor 返回结构化错误；Task Handler 按已冻结的 `fallback_tool_ids` 显式调度，Executor 不在内部隐式换 SQL；
- 原始结果先脱敏、截断和 schema 校验，再作为 Artifact 提供给 LLM。

诊断账号的只读性由目标数据库权限和 Executor 双重强制，不依赖 SQL 关键字检查。

## 结果契约

```text
DatabaseObservation {
  artifact_id, action_id, target_id
  tool_id, tool_version, template_hash
  db_type, db_version
  captured_at, duration_ms
  columns, rows, row_count, truncated
  summary_metrics, warnings
  provenance, sensitivity
}
```

Observation 不直接等于根因。Diagnosis Skill 必须将其与监控时间窗口、其他数据库 Observation 和 SOP Evidence 结合，并区分事实、相关性和根因推断。

## 四种 SQL 边界

| 场景 | SQL 来源 | 执行方式 |
| --- | --- | --- |
| 直连数据库诊断 | 版本化预置模板 | 只读 Executor 自动执行 |
| Alert/巡检 | 仅版本化预置模板 | 只读 Executor 自动执行 |
| 变更/修复 | 版本化动作模板 | 每条命令一次审批后执行 |
| Chat 中诊断模板不足 | LLM 生成并经只读安全校验 | 只供用户手工执行和回贴 |

Alert/巡检在目录无法补齐证据时输出 `PARTIAL/INCONCLUSIVE` 报告，不运行临时 SQL。诊断模板和变更模板分库管理，不能因为某条 SQL 在语法上只读就绕过动作审批。

## 从 3.x 迁移

3.x `DatabaseDiagnosticTools` 的工具意图、DBA SQL 和 Oracle/MySQL 差异可作为评审输入，但不直接复制嵌入 Python 的 `sql_registry`。迁移时必须逐条校验版本、权限、开销、bind 参数、输出 schema 和敏感字段。PostgreSQL 模板本轮不迁移，但 `DatabaseDialect` 保留新增数据库的注册协议。

## 验收

- 目录启动时通过 Manifest、SQL Hash、schema 和重复 `tool_id` 校验；
- 每个 Oracle/MySQL 工具有支持版本、缺权限、超时、空结果、超行数和脱敏测试；
- 契约测试证明 Planner 无法提交 SQL 文本或未注册 `tool_id`；
- 安全测试证明请求参数无法注入标识符、第二语句或变更命令；
- Artifact 能完整回溯 Target、工具/模板版本、参数和执行时间。
