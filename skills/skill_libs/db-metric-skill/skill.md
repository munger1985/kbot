---
name: DBMetricSkill
description: 【数据库运维专有工具 v2】Prometheus 优先获取常规监控指标, 16 个专家 SQL 工具兜底深度根因诊断
category: ops
domain: ops
run_mode: read_only
usage_example: 用户输入 "查看实例 X 的 CPU 使用率" 或 "排查当前锁阻塞链"
---

# DBMetricSkill v2

## 概述
Prometheus 优先 · 专家 SQL 兜底的数据库运维指标听诊器。

## 架构
- 阶段一: Prometheus 监控指标查询（主路径）
- 阶段二: 16 个专家诊断 SQL 工具（兜底路径）

## 参数
* task_description (string, 必填): 自然语言描述的运维诊断任务
