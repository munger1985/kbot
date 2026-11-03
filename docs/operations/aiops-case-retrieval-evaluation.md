# 诊断案例检索评测

该工具复用 Main API 的 `operations-knowledge/search-preview`，不新建检索引擎，也不初始化数据库、
账号、Collection 或测试资产。两仓使用相同数据集、评分器和门禁；分别运行、分别记录结果。

## 1. 三种证据不得混用

- `--validate-dataset`：只校验 20 个场景的结构，输出 `retrieval_quality_evaluated=false`。
- `--responses`：对完整 JSONL 响应评分。每条明确标记 `OFFLINE_CONTRACT` 或
  `LIVE_DEPENDENCIES`；后者只是导出者声明，输出
  `DECLARED_LIVE_EXPORT_NOT_INDEPENDENTLY_VERIFIED`，不能证明本工具实际访问过真实环境。
- `--endpoint`：实际向显式地址发起检索预览，成功采集后才输出
  `live_requests_performed=true`。这仍只证明本次请求和评分，不替代来源复核、发布、
  撤回、并发、通知及完整 P7 验收；`production_acceptance_proven` 始终为 false。

单测的 KC／模型替身只验证评分器与真实应用返回合同兼容。不能把合成金标响应的满分写成真实召回率。

当前正文结果附 OPS_KNOWLEDGE_RANKING.v1 评分卡：KC 相关性为主，五项业务因子
最多增加 20%，版本年龄最多占 2%。记录评分卡可解释某次排序，不证明权重或语义判断正确；
真实质量结论仍取决于独立金标和实际返回顺序。尤其核对 older_relevant、mixed_manual_case、
negative_warning 与跨版本场景；不能为了使分数通过而回写金标、日期或适用性判断。

## 2. 独立准备固定资产与金标

需要单独授权后在指定测试 Domain 准备 `tests/evaluation/aiops_case_retrieval_cases.json`
列出的 15 份资产。清单中有已发布资料、四类用途的案例以及草稿、退役、外域、
来源撤权和被替代的旧版本。不存在的资产不能改成跳过；不能在评测器中自动创建、发布或修改环境。

要求：

- Oracle、MySQL、PostgreSQL 的案例记录各自数据库版本及拓扑；Oracle／PG 已验证案例有
  独立的受控验证原件及逐项正向效果，不能用案例总结果代替单项措施证据。
- `pg_old_relevant` 是较旧但正文相关的排查参考，保留未决事项，故整体复用状态为 UNKNOWN；
  `pg_recent_irrelevant` 是较新但正文不相关的参考。日期顺序与正文相关性必须独立准备。
- `pg_lock_copy` 与 `pg_lock` 有同一来源血缘、不同确切版本标识；
  `shared_lineage` 允许其中任一等价副本命中金标，但同时出现即失败，不能算两份独立证据。
- `pg_effect_unverified` 的环境和条件可以匹配，但至少一项措施缺少对应正向受控效果。
  该场景可以 MATCHED，仍不可推荐措施；不通过人为把环境写成 UNKNOWN 来规避逐项效果检查。
- `pg_optimization` 为受控测量的优化经验；失败教训保留未知条件或未决事项，只能作为 WARNING_ONLY。
- 自然语言前提应选用当前 Target 元数据确实能证明的条件；业务负载、备份、维护窗口、
  故障恢复和安全风险不能被数据库类型“证明”。这些条件缺少当前可信证据时应为 UNKNOWN。
- 金标及当前上下文指纹由独立核查确认，不能直接从待评分响应反抄，否则无法检验错误判断。

场景当前目标：

| 场景 | Target 要求 |
| --- | --- |
| oracle_context／mysql_context／postgres_context | 分别匹配案例数据库、主版本和拓扑 |
| version_conflict／query_injection | 与 PG 案例明确主版本或数据库冲突；查询文字不能改变当前 Target |
| topology_conflict | 同数据库和主版本，拓扑不同 |
| unknown_target | 不绑定 Target |
| same_symptom_other_mechanism／older_relevant | 至少一项必要环境或自然语言条件无法核验，保持 UNKNOWN |
| same_mechanism_different_words／mixed_manual_case／publication_scope／exact_revision／shared_lineage | 匹配 PG 案例；按真实正文相关性召回 |
| negative_warning | 保留失败教训的未知前提；CHANGE 用途只允许警告 |
| effects_closure | 匹配环境和条件，但单项措施没有对应正向受控效果 |
| verified_optimization／plan_reuse／verify_reuse | 匹配对应已验证案例，仍不授予执行权限 |
| prohibited_condition | 当前 Oracle 命中 PG 案例“禁止在 Oracle 复用”的明确禁忌，状态 PROHIBITED |

这些是可重建质量场景，不是生产故障关键词表；产品代码不得按场景 ID 或查询词特殊处理。

## 3. 环境清单与响应格式

环境清单使用 `OPS_CASE_RETRIEVAL_FIXTURES.v1`，两个必填字段：

- `assets`：键必须精确等于数据集 `asset_specs`；每项包含真实
  `asset_version_id`、`bundle_revision_id`、`document_version_id` UUID、
  `lineage_id`，案例还需要规范正文的 `body_sha256`。不按标题猜版本。
- `case_bindings`：键必须精确覆盖全部场景；每项包含真实 `agent_id`、
  可空的 `target_id` 及独立读取当前 Agent／Target 后核对的 `context_fingerprint`。
  使用应用层 `load_case_search_context` 的安全投影核对；不采集主机、账号、端点或凭据。

JSONL 每行结构为 `{"case_id": "...", "evidence_kind": "OFFLINE_CONTRACT", "response": {...}}`；
真实导出者可声明 `LIVE_DEPENDENCIES`，但声明不自动升级为已验证真实执行。
必须覆盖全部场景，缺失、重复、未知 ID、未声明来源或无效清单直接报输入错误。

评分包括 Recall、nDCG、倒数排名，以及以下独立安全门禁：

- 正文相关性和顺序；未知资产占据原始排名，不删除后虚增 nDCG。
- 发布／Domain／来源状态、类型与权威标签、同源去重、范围完整性和数量上限。
- 每份 Citation 的确切 Revision、正文版本、正文哈希、知识单元引文／位置／Hash 与必要闭包。
- 结果、复用上下文和 Citation 中的适用性判断一致，绑定独立核对的当前上下文。
- UNKNOWN／CONFLICT／PROHIBITED 和排查参考、失败教训不可获得处置推荐；
  每项可推荐措施还必须有正向受控效果链接；任何结果都不能授予执行权限。
- `action_allowed` 是推荐资格的上限，不要求模型必须建议全部有资格的动作。

## 4. 运行

先做不访问环境的结构检查：

```bash
conda run --no-capture-output -n cube python tests/evaluation/evaluate_aiops_case_retrieval.py --validate-dataset
```

授权环境准备好后，评分已经采集的响应：

```bash
conda run --no-capture-output -n cube python tests/evaluation/evaluate_aiops_case_retrieval.py --manifest /tmp/aiops-evaluation/manifest.json --responses /tmp/aiops-evaluation/responses.jsonl
```

或显式指定已授权测试 Main API 的完整检索预览 URL：

```bash
conda run --no-capture-output -n cube python tests/evaluation/evaluate_aiops_case_retrieval.py --manifest /tmp/aiops-evaluation/manifest.json --endpoint "$AIOPS_CASE_EVAL_ENDPOINT"
```

KBot 将环境名 `cube` 换成 `kbot4`，地址采用其 Main API 路由。不要使用默认线上地址。
认证头从已有安全注入的 `AIOPS_CASE_EVAL_HEADERS_JSON` 环境变量读取；不在命令行或清单中放密钥。
端点不得含凭据、查询串或 fragment；拒绝重定向，避免认证信息转发到其他地址。
每次请求最多等待 60 秒，正文读取预算 4 MB；端点或输入错误不打印认证头和原始错误正文。

退出码：0 为本次全部评分通过，1 为完成评分但有失败项，2 为输入或端点问题、未完成评测。
保留运行时间、版本、环境身份和独立资产／Target 核对记录；不要把原始凭据加入 Git。
