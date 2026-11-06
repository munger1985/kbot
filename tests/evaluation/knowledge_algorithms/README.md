# 知识基础能力评测工具

- run_baseline.py：只跑 G-A 合成 fixture，不加载真实模型或数据库。
- audit_quality.py：读取单 App、固定数据分割的评测清单及 JSONL 结果，计算逐查询/分任务 Recall@K、Precision@K、nDCG@K、引用边召回率、成功率、P95、越权/身份错误和静默零交付。失败、取消、无结果查询必须保留，不能只提交成功样例。
- 本轮不生成真实语料清单、不填真实阈值、不运行真实测试。只有 App 配套接入后，用经授权的文档与模型生成结果，再运行审计器。

清单字段：schema_version=knowledge.quality_manifest.v1、owner、dataset_revision、configuration_hash、split、cases、thresholds。
每个 case 必须包含 query_id、task、requested_count、relevant_keys；可声明 forbidden_keys、required_links（证据键对）、answer_required、expect_empty。
thresholds 包含 max_p95_ms 与按 task 登记的 by_task；每任务均须显式提供 recall_at_k、ndcg_at_k、link_recall、successful 最低值。阈值须在查看 holdout 前冻结。

每行结果包含 query_id、owner、status、duration_ms、primary_keys、evidence_keys、delivered_keys、returned_links、delivery_status、readable_within_budget。键必须为冻结 EvidenceKey 的稳定摘要，不放正文、密钥或签名 URL；跨 App 标识不可混用。规范身份审计还须配合原始 EvidenceSet 合同测试，不能靠自报 stable_key 代替授权验证。

执行：python tests/evaluation/knowledge_algorithms/audit_quality.py --manifest <已批准的清单> --results <已完成的结果.jsonl>。stdout 为 JSON，不自动覆盖工件或启动业务请求。比较方案时固定语料/预算/模型/分割，对齐逐查询结果；不得把合成基准或离线 PASS 当作 G-C 全部通过。
