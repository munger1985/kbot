---
name: DBAnalysisSkill
description: 【数据库运维专有工具 v2】融合 Prometheus 监控时序数据、专家 SQL 诊断结果与运维手册, 进行全方位故障根因分析（RCA）并提供自愈建议
category: ops
domain: ops
run_mode: read_only
usage_example: 在 DBMetricSkill 采集完指标后, 由 Planner 自动调用此技能进行 RCA 融合分析
---

# DBAnalysisSkill v2

## 概述
AIOps 专属核心故障根因诊断大脑 (RCA Engine)。

## 功能
融合 Prometheus 监控时序数据 + 专家 SQL 诊断结果 + RAG 运维手册,
由 LLM 扮演资深 Principal DBA 进行确定性根因推理。

## 参数
* task_description (string, 必填): 诊断任务描述, 引用前序步骤的 {{metric_results}} 变量
