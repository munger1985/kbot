# Model Serving 身份与 OpenAI 兼容契约

## 决策

Model Serving 拥有 `KBOT_AI_MODEL`、模型 Provider、模型池和推理契约。
数据库身份、服务名称和厂商名称必须分离：

| 字段 | 语义 |
| --- | --- |
| `model_id` | UUIDv7 内部主键；Oracle `RAW(16)`，未来 PG `uuid` |
| `served_model_name` | `/api/v1` 请求中稳定且全局唯一的 `model` 值 |
| `display_name` | APEX/UI 可修改显示名称 |
| `provider_model_name` | 传给上游 Provider 或本地引擎的真实名称 |

`served_model_name` 不是 API Key，也不是数据库主键。它创建后不可修改；
Provider、Provider 模型名、类别、Embedding 维度和模型参数同样属于模型
身份。需要改变这些字段时创建新的 `model_id`。显示名、端点、凭据、状态
和说明可以更新。

## API 边界

内部管理和 KBot 服务调用使用 `/internal/v1`，要求服务凭证与短期
AuthContext JWT：

- `/internal/v1/models/{model_id}` 使用 UUIDv7；
- 内部推理 DTO 使用 `served_model_name`，不接受显示名或 UUID；
- 管理响应永不返回上游 API Key。

外部模型调用使用独立的 `security.model_api_keys`：

- `GET /api/v1/models`；
- `POST /api/v1/chat/completions`；
- `POST /api/v1/embeddings`。

请求/响应及 SSE 使用 OpenAI 字段 `model=served_model_name`。Portal Key
不能直接调用模型接口，Model API Key 也不携带或声明 Domain。Visual
Embedding 没有对应的 OpenAI 标准，当前只保留内部接口。

## 模型池

模型池以 `served_model_name` 为缓存键，以 `provider_model_name` 构造
Provider Client。相同模型的并发冷启动只执行一次，不同模型使用独立锁；
加载失败和卸载后回收对应的生命周期锁，避免无效模型名造成内存增长。加载前
校验 App、类别和启用状态。配置更新或归档提交后立即失效内存实例，避免旧
凭据、旧端点或已禁用模型继续提供服务。预热保持串行，避免多个本地大模型
同时占用 GPU 峰值。

## Knowledge Core 约束

Collection 保存具体的 UUIDv7 `embedding_model_id`，INDEX 与查询阶段均
通过 Model Serving 解析同一模型。Evidence、Discovery 和 INDEX Job 同时
冻结 `model_id`、`served_model_name`、维度与配置指纹。KC 不绑定显示名或
动态别名；未来即使增加 `MODEL_ALIAS`，Embedding Collection 也只能绑定
具体模型身份，防止向量空间静默漂移。
