# AIOps Agent

## 服务边界

AIOps Agent 是独立领域服务，当前面向 Oracle 和 MySQL 的监控、故障与性能诊断。
它拥有 `KBOT_OPS_*` 表、状态机、调度器、Worker 和 DB Executor，不把运维流程
塞入通用 Agent Runtime。Prometheus、Zabbix 和 OEM 通过 Monitor Provider
适配；后续监控或数据库类型通过注册协议扩展。

## 触发与闭环

入口包括用户聊天、监控告警和定期巡检。统一闭环为：

```text
Trigger → Observe → Diagnose → Evidence Assessment → Advisory
        → Approval → Execute → Verify → Report / Comparison
```

自动告警和巡检只能使用已配置的数据源；证据不足时生成不确定结论和后续建议，
不会等待用户。聊天场景可进入多轮 HITL：当数据库不可直连且监控证据不足时，
Agent 给出受控只读 SQL，用户手工执行并粘贴结果，系统持续补证直到形成根因判断
或达到预算。

## 监控与数据库诊断

监控采集保存来源、时间窗、单位和质量标记。数据库直连查询不允许模型生成任意
SQL：LLM 只能选择版本化诊断工具和参数，DB Executor 根据数据库方言使用预审计
模板，执行只读、超时、行数和敏感字段限制。Oracle/MySQL 均支持版本化只读诊断、
受控人工 SQL 和审批后变更。PostgreSQL 已有内部只读诊断扩展，但尚未进入公开
AIOps Target 契约。Target 启用时探测权限和能力，凭据只保存统一托管凭据引用。

## 建议、审批与执行

分析结果可以只给解决思路，也可以生成版本化 Change Proposal。每条命令必须单独
获得一次人工批准并留痕；批准绑定 Proposal Hash、Target 版本、参数、策略和过期
时间。配置允许 Agent 执行时，DB Executor 仍是最终校验点。多命令方案严格串行，
上一条执行并验证成功后才生成下一条待审命令。

用户也可选择自行执行并回填结果。无论自动或人工处理，系统都保留 Before/After
指标，生成验证结论和对比报告。

## 巡检与报告

Inspection Plan 支持日报、周报和 Cron。Scheduler 生成不可变 Fire，并按 Target
展开 Run。报告保存在系统中供前端渲染，包括诊断报告、巡检报告、处理结果和前后
对比报告。Email、IM 当前不实现，仅保留通知端口；未来接入不改变 Run、Report 和
审批模型。

精确 API 以 `docs/openapi/aiops_*.json` 为准，表结构以
`database/oracle/aiops_agent/` 为准。
