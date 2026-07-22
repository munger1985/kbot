---
name: ask-data-skill
description: 用于将自然语言转换为 SQL 并查询企业关系型数据库，检索生产、质检、流水线、库存等结构化指标数据。
category: data_analysis
usage_example: 查询 A1 流水线在 {{start_date}} 期间的设备异常点数
---

# 输入参数约束
* query (string, 必填): 详细的自然语言数据查询诉求，例如“查询上个月异常的工单总数”。