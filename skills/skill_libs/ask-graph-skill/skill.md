---
name: ask-graph-skill
description: 用于在企业知识图谱（Oracle 26ai Graph Property）中执行实体拓扑下游走与关联性溯源。擅长挖掘复杂实体的依存关系、影响链路、因果血缘以及组织或物料的隶属网络，并反查出对应的规范化文本资产。
category: knowledge_retrieval
usage_example: 在设备图谱中以“A1流水线伺服电机”为实体进行2度游走，探寻其故障会导致哪些下游工序或产线发生级联停机。
---

# 输入参数约束
* vertex_names (array[string], 可选): 提炼出的核心图谱实体/节点名称列表。如未提供，系统将自动退化降级从 `search_keywords` 或当前查询词中动态提取。
* max_depth (integer, 可选): 知识图谱深度遍历的度数，默认为 2。
* graph_weight (float, 可选): 图检索所占的分数初始权重系数，默认为 1.2。