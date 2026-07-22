---
name: db-analysis-skill
description: 【数据库运维专有工具 v2】融合 Prometheus 监控时序数据 + 专家 SQL 诊断结果 + RAG 运维手册，由 LLM 扮演资深 Principal DBA 进行内核级故障根因分析（RCA），为后续下发自愈行动提供确定性底座。
category: ops_diagnose
usage_example: 前置探针已经通过 Prometheus 抓到了 CPU 使用率 99% + 活跃会话数暴增的监控数据，且专家 SQL 工具抓到了 Top CPU SQL 明细，请立刻融合这些多路素材进行专家级 RCA 根因推演。
---

# 输入参数约束
* context (OpsContextMemory, 必填): 包含全局专职自愈总线上下文（内含 command_or_query、instance_id、db_type、environment、llm_model，以及多源数据沉淀池 monitor_results、metric_results 和 doc_results）。

# 控制面规划与总线推理特性 (v2)
1. 多源证据链融合：
   诊断大脑同时消费三路数据源：
   - monitor_results: Prometheus 标准化时序数据（趋势、当前值、标签）
   - metric_results: 专家 SQL 诊断工具返回的明细数据（锁链、SQL文本、段分布等）
   - doc_results: RAG 检索到的 DBA 运维手册 / SOP 文档
   当三者存在交叉印证时，RCA 结论置信度最高。

2. 确定性低变异推理：
   技能内部将大模型流式推理的生成温度（Temperature）严格锁死在 0.1。拒绝任何高变异度、发散性的客服语气话术，确保输出的 RCA 诊断逻辑具备极高的 DBA 工业刚性与技术严密性。

3. 证据链前置熔断拦截：
   坚持"宁缺毋滥，安全第一"的自愈原则。如果 monitor_results 与 metric_results 同时为空，诊断大脑将立即触发安全熔断，通过 PacketType.WARNING 友好退场，绝对不允许大模型对生产环境进行无数据凭证的盲目猜测。

4. 自愈联动基石输出 (RCA Output Anchor)：
   输出格式为严格的 Markdown 结构化工业报告，完美闭环【内核现状概览】、【深度根因推演 (RCA)】和【预案自愈建议】。
