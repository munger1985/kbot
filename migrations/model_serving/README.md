# Model Serving migrations

按编号通过发布迁移器执行。`001_model_registry.sql` 创建 KBot 4.0
模型目录 `KBOT_AI_MODEL`；旧 `KBOT_MD_MODELS` 不再被 4.0 代码读取，
但迁移脚本不会删除旧表。

`MODEL_ID` 是应用生成的 UUIDv7，Oracle 保存为 `RAW(16)`；未来
PostgreSQL 版本映射为原生 `uuid`。`SERVED_MODEL_NAME` 是推理 API
接受的稳定名称，`PROVIDER_MODEL_NAME` 仅用于调用上游模型。两者不能
混用。

Embedding 模型的维度必须与 `base.toml` 一致。模型目录先部署，随后
才能执行 Knowledge Core Schema，因为 Collection 保存模型服务拥有的
UUID 引用。
