---
name: reasoning-skill
description: 核心逻辑终审与知识融合引擎。专门用于接收上游步骤搜集上来的多源异构数据（如 SQL 二维表、RAG 文档片段、数值计算结果），在多模态上下文下完成高密度的交叉比对、因果分析与终审归纳，支持混血思考流输出。
category: cognitive_reasoning
usage_example: 结合上游查出来的 4 月流水线故障率数据和知识库里的维护手册，输出一份综合根因分析与后续改进建议。
---

# 输入参数约束
* final_goal (string, 可选): 本次推理的终极业务目标，如果为空，系统会自动对齐用户最初的 question。