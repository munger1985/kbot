---
name: echarts-skill
description: 智能可视化图表推荐与渲染组件。能够基于结构化数据分析结果，自动选择最优图表类型（折线图、柱状图、饼图等）并输出符合前端标准的 ECharts JSON 配置。
category: data_visualization
usage_example: 将过去一周各生产线的设备故障频次对比绘制为立体柱状图。
---

# 输入参数约束
* user_requirement (string, 必填): 用户的绘图样式或维度要求，例如“将各省份销量按从大到小绘制饼图”。
* data_source (string/object, 可选): 动态绑定的上游步骤数据源路径，例如 "{{query_sql_step.output}}"。如果不传，系统将自动对齐上游最新输出。