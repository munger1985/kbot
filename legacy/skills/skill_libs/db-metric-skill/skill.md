---
name: db-metric-skill
description: 【数据库运维专有工具 v2】Prometheus 优先获取常规监控指标（CPU/内存/存储/会话/锁/IO等），16个专家SQL工具兜底深度根因诊断（锁链/高负载SQL/表空间段/等待事件/变更审计/主备状态等）。Zabbix接口预留。
category: ops_diagnose
usage_example: 有一台 Oracle 生产库报锁等待，快看看哪个会话把别人卡住了，并捞取死锁头元数据。
---

# 输入参数约束
* context (OpsContextMemory, 必填): 包含全局总线上下文（内含 command_or_query、instance_id、db_type、monitor_type、prometheus_instance_label、zabbix_host_name 等参数）。

# 控制面规划与总线回填特性 (v2)
1. 监控指标查询（按 monitor_type 选择数据源）：
   技能内部首先尝试将用户任务匹配到 Prometheus 或 Zabbix 监控指标
   （通过 metrics_mapping.yaml 注册中心）。
   - monitor_type = "prometheus": 渲染 PromQL → Prometheus HTTP API → 标准 MetricResult
   - monitor_type = "zabbix": 渲染 Zabbix Item Key → Zabbix JSON-RPC API (host.get→item.get) → 标准 MetricResult
   结果沉淀至 monitor_results。

2. 专家SQL 兜底机制：
   当监控系统无法覆盖（需要深度诊断如锁链/高负载SQL文本/变更审计等），
   LLM 从 16 个专家诊断工具中做单选题，精准调用对应的 DatabaseDiagnosticTools 方法。
   执行结果沉淀至 metric_results。

3. 逆向总线变量回填 (Backfill Registry)：
   执行成功后，除了向流中推送 PacketType.SQL_RESULTS 结构化报表外，还会将故障源的核心标识逆向注入全局变量池。

4. Zabbix 原生支持（v2.1）：
   当 monitor_type == 'zabbix' 时，通过 ZabbixProvider 走 Zabbix JSON-RPC API
   （user.login → host.get → item.get → history.get），无需额外桥接工具。
