# Model Serving

`model_serving` 是模型托管服务包，拥有 Provider Adapter、模型池、
`KBOT_AI_MODEL` 配置目录和分类推理服务，不拥有 Agent/Skill 或
Knowledge Core 业务数据。

`common/entities/ai_model.py` 和 `common/model_repository.py` 是模型目录的
唯一 Owner。其他服务只能通过 `platform_clients.AIModelConfigClient`
读取脱敏 DTO，不能导入 Entity 或 Repository。当前仍连接同一 Oracle
Schema，后续拆库只需替换 Model Serving 的数据库配置。

可部署进程位于 `model_serving/entrypoints/`：

- `embedding`：文本向量与相似度；
- `llm`：对话补全与工具调用；
- `vlm`：视觉语言推理；
- `visual`：图像向量。

四个进程都提供按类别隔离的 `/internal/v1/models` 管理接口。`DELETE`
只归档模型，保留 Collection 引用和审计历史。外部 LLM、VLM 和文本
Embedding 进程还提供 `/api/v1` OpenAI 兼容接口，使用独立 Model API
Key；数据库 UUID 和上游凭据不会进入公开推理契约。

模型字段语义：

- `model_id`：UUIDv7 内部身份；
- `served_model_name`：公开 `model` 参数和模型池缓存键；
- `display_name`：可修改显示名称；
- `provider_model_name`：上游厂商或本地引擎名称。
