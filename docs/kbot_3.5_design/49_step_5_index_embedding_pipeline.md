# 步骤 5 详细设计：Evidence INDEX 向量流水线

## 本步边界

解析完成只代表结构化 Parse View 和 Evidence 通过质量门，不代表已经可以召回。解析 Worker 不调用 Embedding 服务，也不能从 parser policy、请求参数或进程默认配置读取模型。KC 为每个 Parse View 投递一个幂等的 `INDEX` Job；该 Job 是本步唯一的文本向量生成入口。

## 单一模型来源

1. `KBOT_KC_COLLECTION.embedding_model_id` 是 Collection 唯一绑定。
2. KC 的模型目录解析器读取 `AIModel.embedding_dimension`、稳定技术名和配置 revision，并校验模型类别为文本 Embedding、状态启用、维度等于 `base.toml [embed].dimensions`。
3. INDEX 执行时保存 `embedding_model_id`、`embedding_model_key`、配置指纹、输入文本 hash 和生成时间。Provider 返回的模型身份、向量数量和维度必须逐项核对；任何不一致都不写库。
4. `retrieval_text` 变化、Collection 更换模型或模型 revision 变化都会使 Evidence 重新进入 INDEX。相同文本、同一模型和同一配置指纹重复执行必须幂等跳过。

Evidence 只有一份文本 embedding。图片的 VLM 描述是 Parse View 的来源增强文本，不是第二个向量；视觉检索向量若以后需要，必须属于独立 `VISUAL` 索引字段和明确的跨模态查询协议，不能覆盖文本向量。

## 状态转换

```text
PARSE SUCCEEDED
  → Evidence ACTIVE + Member INDEXING + INDEX Job PENDING
  → INDEX claim/lease → 批量生成并校验向量
  → 全部 Evidence 完成 → Member READY → Revision READY/PARTIAL
```

解析失败不会创建 INDEX Job。INDEX 的暂时性失败只重试当前 Job；旧的 ACTIVE Parse View 不因新 View 的 INDEX 失败而被删除或替换。只有同一 View 的所有可见 Evidence 都具备匹配 Collection 模型的向量后，Revision 才能进入可检索状态。

## 已落地实现

- `knowledge_core/application/indexing.py`：独立 INDEX 租约、模型快照、provider 响应校验、输入 hash 和 Evidence 批量索引服务。
- `knowledge_core/api/index_task_router.py`：INDEX claim、heartbeat、run 内部协议；与 Parser 租约端点分离。
- `KBOT_KC_EVIDENCE` 增加向量及模型身份字段；`migrations/kc/005_kc_retrieval_index.sql` 为已存在数据库提供增量迁移。
- Parse completion 已改为创建独立 `INDEX` Job，并将成员置为 `INDEXING`；Parse Worker 不再生成向量。

本步暂不实现 Discovery Object、Oracle 检索 SQL、LLM 候选选择和问文 Skill；它们在向量写入闭环通过后继续实施。
