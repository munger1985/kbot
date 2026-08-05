# Model Serving

## 模型身份

`KBOT_AI_MODEL.MODEL_ID` 使用 UUIDv7 主键；`SERVED_MODEL_NAME` 是调用方和
OpenAI 兼容接口使用的稳定名称。模型名称不是数据库主键，允许底层 Provider、
路径和部署参数在受控更新后变化。`MODEL_PARAMS` 使用 Oracle 原生 JSON，
本地模型路径写入其中的 `model_path`。

模型目录记录类别、Provider、服务名、连接配置和参数。API 响应、引用摘要和失效事件
均不得返回 `API_KEY`。文本 Embedding 模型将
`embedding_dimension` 存入 `MODEL_PARAMS`，其值必须等于全局
`embedding_dimension`；相同维度不代表相同向量空间。

## 进程与接口

Model Serving 是一个服务包，按资源类型运行独立进程：

- LLM；
- VLM；
- Text Embedding；
- Visual Embedding。

各进程复用模型目录、Model Pool、配置和管理逻辑，但只加载自己的模型类别。内部
管理与推理接口使用 `/internal/v1`；明确开放给非 KBot 调用方的推理接口使用
独立 Model API Key，并提供 `/api/v1/models` 等 OpenAI 兼容契约。

目录写入统一经过 `ModelServingUnitOfWork` 和 Model Repository。生命周期为
`DRAFT → ACTIVE → ARCHIVED`，归档不可逆；所有更新、状态变更和删除都携带
`expected_row_version`。只有没有 Agent Runtime、Knowledge Core、Data Query 引用，
引用服务全部可用，且本进程没有运行实例的归档模型才能物理删除。

Provider Options 由代码目录控制，按模型类别声明必要连接字段、Secret 字段和允许的
`model_params`。未知 Provider、未知参数、非法类别组合和不匹配的文本向量维度都会在
写入前被拒绝。

## Model Pool

Model Pool 按 `served_model_name` 缓存已加载实例，负责并发加载、健康检查、空闲
卸载和失败隔离。启动预热只处理配置为预热且具备完整路径/Provider 参数的模型；
单个模型预热失败不能伪装为服务健康。

目录创建、更新、状态变更、归档或删除提交后发送模型粒度失效事件，事件只包含
`model_id`、`served_model_name`、`category` 和 `row_version`。所属推理进程卸载旧实例，
下一次请求从 Oracle 读取新配置；进程重启同样从目录恢复，不保存长生命周期配置快照。

## 功能绑定

Collection 和 Agent 的 `models_json` 都保存模型 UUID；运行时快照再固化
`served_model_name`，用于调用 OpenAI 兼容推理接口。
Collection 的文本 Embedding 是必选且不可原地替换；VLM 和视觉 Embedding 可选。
Agent 可为路由、上下文、回答和记忆分别绑定不同模型。

DeepSeek OCR 是 Parser 的独立 OpenAI 兼容依赖，不登记到 Model Serving，也不由
Model Pool 托管。
