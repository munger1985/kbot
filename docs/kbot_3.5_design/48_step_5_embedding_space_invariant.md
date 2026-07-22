# 步骤 5 详细设计：Embedding 维度与模型一致性

## 决策

3.5 不增加 `KBOT_KC_INDEX_GENERATION` 或 `KBOT_KC_EVIDENCE_INDEX`。Evidence 与 Discovery Object 直接保存 `retrieval/profile_text + embedding`，并由 Oracle 建立 Text/Vector 物理索引。

Embedding 采用“全局维度、Collection 绑定模型”两层约束：

- `base.toml` 只锁定 Oracle VECTOR 的统一维度；
- `KBOT_MD_MODELS` 为 embedding 类模型增加非空 `embedding_dimension`；该值必须等于 `base.toml` 的维度；
- 每个 Collection 通过 `embedding_model_id` 绑定一个 embedding 模型，继承 V1 `KB.models.txt_embedding_model` 的职责；
- 同一 Collection 的索引向量和查询向量必须由该绑定模型生成。

相同维度只保证可以写入同一物理列，不代表向量可比较。模型 A 生成的查询向量绝不能用于检索模型 B 生成的内容向量。

## 所有权与调用链

Parser 负责 Atom/Structure IR、定位、来源和确定性 `retrieval_text`。检索向量在 `INDEX` 阶段生成；若开发期暂时仍由 Parser Worker 调用 embedding 服务，模型参数也必须来自 Collection 的 Job 快照，不能读取 Worker 默认值。

```text
Collection.embedding_model_id ─┬→ INDEX Job → Evidence/Discovery embedding
                               └→ Retrieval → query embedding
```

`KnowledgeRetrievalSkillV2` 和 `DocumentAgentV2` 只传查询文本与 Collection Scope，不自行选择模型或传 `query_vector`。KC 根据 Collection 解析模型并调用模型托管服务。

## 配置与校验

`base.toml` 仅提供统一维度，例如：

```toml
[embed]
dimensions = 1536
```

创建或修改 embedding 模型时，模型类别、启用状态和 `embedding_dimension` 必须校验；维度不一致的模型不能保存为可用状态。创建 Collection 时必须绑定一个可用 embedding 模型。

Evidence 和 Discovery Object 保存 `embedding_model_id`、模型稳定 key/revision、输入 hash 和 `indexed_at`。Embedding 服务响应必须回显实际模型身份和向量长度；与 Job 快照或全局维度不一致时，INDEX 失败并报警。

## 多 Collection 检索

一个 Agent 绑定的 Collection 可以使用不同 embedding 模型。KC 先按 `embedding_model_id` 分组，每个不同模型只生成一次 query embedding，再仅检索该模型对应的 Collection：

```text
query
  ├─ model A → collections 1, 2 → 各自 Text/Vector RRF
  └─ model B → collection 3    → Text/Vector RRF
                         ↓
             跨 Collection 公平合并/LLM Selector
```

跨模型不比较原始 cosine 分数。先在 Collection 内进行 Text/Vector 排名融合，再通过归一化名次、平权合并和 LLM 内容判断形成候选集。

## 模型变更

Collection 更换 embedding 模型是该 Collection 的维护操作，不影响其他 Collection：停用检索和新索引写入，绑定维度合法的新模型，重建其全部 ACTIVE Evidence/Discovery 向量，验证模型身份、空向量和召回基线后重新启用。重建完成前不得让新旧模型向量同时参与该 Collection 的查询。

全局维度变化才需要修改 `base.toml`、Oracle VECTOR DDL/索引并执行全应用重建。

## 验收

- `KBOT_MD_MODELS.embedding_dimension` 与 `base.toml` 维度不一致时不可启用或绑定。
- Collection 的 INDEX 与 Query 使用同一 `embedding_model_id` 和模型 revision。
- 同维度、不同模型的向量只能在各自 Collection 范围检索，不能交叉计算。
- 多模型 Collection 查询按模型分组生成向量，并在局部召回后做跨 Collection 排名融合。
