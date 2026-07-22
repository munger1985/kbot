---
name: ops-heal-skill
description: 【运维愈合执行引擎】读取诊断报告，自动查询数据库获取真实数据，生成并执行变更 SQL，失败时自愈重试。支持多轮闭环直至问题解决。
category: ops
domain: ops
run_mode: mutation
usage_example: 在 DBAnalysisSkill 输出诊断报告后，由 Planner 调用此技能执行愈合操作。自动查库→生成SQL→执行→修正→循环，最多5轮。
---
