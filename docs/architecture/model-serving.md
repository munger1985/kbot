# Model Serving

## 模型身份

`KBOT_AI_MODEL.MODEL_ID` 使用 UUIDv7 主键；`SERVED_MODEL_NAME` 是调用方和
OpenAI 兼容接口使用的稳定名称。模型名称不是数据库主键，允许底层 Provider、
路径和部署参数在受控更新后变化。`MODEL_PARAMS` 使用 Oracle 原生 JSON，
本地模型路径写入其中的 `model_path`。

模型目录记录类别、Provider、服务名、维度、加密连接配置和参数。文本 Embedding
模型的维度必须等于全局 `embedding_dimension`；相同维度不代表相同向量空间。

## 进程与接口

Model Serving 是一个服务包，按资源类型运行独立进程：

- LLM；
- VLM；
- Text Embedding；
- Visual Embedding。

各进程复用模型目录、Model Pool、配置和管理逻辑，但只加载自己的模型类别。内部
管理与推理接口使用 `/internal/v1`；明确开放给非 KBot 调用方的推理接口使用
独立 Model API Key，并提供 `/api/v1/models` 等 OpenAI 兼容契约。

## Model Pool

Model Pool 按 `served_model_name` 缓存已加载实例，负责并发加载、健康检查、空闲
卸载和失败隔离。启动预热只处理配置为预热且具备完整路径/Provider 参数的模型；
单个模型预热失败不能伪装为服务健康。

## 功能绑定

Collection 的 `models_json` 保存模型 UUID，因为 KC 必须固化索引模型身份；Agent
的 `models_json` 保存 `served_model_name`，便于按功能调用并兼容模型 API。
Collection 的文本 Embedding 是必选且不可原地替换；VLM 和视觉 Embedding 可选。
Agent 可为路由、上下文、回答和记忆分别绑定不同模型。

DeepSeek OCR 是 Parser 的独立 OpenAI 兼容依赖，不登记到 Model Serving，也不由
Model Pool 托管。
