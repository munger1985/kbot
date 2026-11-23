# Model Serving

## 模型身份

`KBOT_AI_MODEL.MODEL_ID` 是管理、绑定、推理与实例缓存共用的 UUIDv7 标识。
OpenAI 兼容接口的 `model` 参数填写 UUID；`PROVIDER_MODEL_NAME` 只发送给厂商或本地引擎。
`MODEL_PARAMS` 使用 Oracle 原生 JSON，本地模型路径写入 `model_path`。

模型目录记录类别、Provider、厂商模型名称、连接配置和参数。API 响应、引用摘要和失效事件
均不得返回 `API_KEY`。文本 Embedding 模型将
`embedding_dimension` 存入 `MODEL_PARAMS`，其值必须等于全局
`embedding_dimension`；相同维度不代表相同向量空间。

## 进程与接口

Model Serving 是一个服务包，按资源类型运行独立进程：

- LLM；
- VLM；
- OCR；
- Text Embedding；
- Visual Embedding；
- Image Generation。

各进程复用模型目录、Model Pool、配置和管理逻辑，但只加载自己的模型类别。内部
管理与推理接口使用 `/internal/v1`；明确开放给非 KBot 调用方的推理接口使用
独立 Model API Key，并提供 `/api/v1/models` 等 OpenAI 兼容契约。

浏览器从工作台的“系统管理 → AI 模型管理”访问 `/api/v1/platform/models`。
Main API 要求有效平台会话和 `platform:model_manage`，再转发给模型所属进程。
全局模型配置不属于多媒体创作 App；业务 App 只维护自己的模型引用和绑定。

目录写入统一经过 `ModelServingUnitOfWork` 和 Model Repository。生命周期为
`DRAFT → ACTIVE → ARCHIVED`，归档不可逆；所有更新、状态变更和删除都携带
`expected_row_version`。只有没有 Knowledge Retrieval App、AIOps App、Knowledge
Core、Data Query 引用，
引用服务全部可用，且本进程没有运行实例的归档模型才能物理删除。

新建模型由系统生成 UUID，无需输入额外调用名。「厂商模型名称 / ID」对应 `provider_model_name`，可以按版本更新并触发模型缓存失效。
如果新型号对应不同向量空间，应创建新的模型配置并重建相关索引。
表单仅展示当前 Provider 声明的参数；OCI 签名认证材料仅在支持 OCI 的选项下展示。

已启用的模型可从列表的「测试」操作验证已保存配置。Main API 的
`POST /api/v1/platform/models/{category}/{model_id}:test-connection`
复用模型管理权限，并转发到该类别的 Model Serving 进程。测试使用固定小样本执行
真实推理，成功仅返回 `latency_ms`，不改动模型配置，也不回显凭据或推理内容。
图像生成测试会请求生成一张图，可能产生费用；测试产物不保存。

Provider Options 由代码目录控制，按模型类别声明必要连接字段、Secret 字段和允许的
`model_params`。未知 Provider、未知参数、非法类别组合和不匹配的文本向量维度都会在
写入前被拒绝。

`local_deepseek` 是 LLM 的客户近端 OpenAI 兼容 Provider，必填 `api_endpoint`
与 `api_key`，由客户侧 vLLM / SGLang 等推理服务提供，不在本进程加载权重。
它与云端 `api_deepseek` 分开登记。AIOps 生产环境的规划/诊断绑定只允许该
Provider；该约束属于 AIOps 绑定校验，不改变 Model Serving 对其它应用提供
GPT 的能力。


## Model Pool

Model Pool 按 `model_id` 缓存已加载实例，负责并发加载、健康检查、空闲
卸载和失败隔离。启动预热只处理配置为预热且具备完整路径/Provider 参数的模型；
单个模型预热失败不能伪装为服务健康。

目录创建、更新、状态变更、归档或删除提交后发送模型粒度失效事件，事件只包含
`model_id`、`category` 和 `row_version`。所属推理进程卸载旧实例，
下一次请求从 Oracle 读取新配置；进程重启同样从目录恢复，不保存长生命周期配置快照。

## 功能绑定

Collection 和 App 私有 Agent 的模型绑定保存模型 UUID；运行时 Execution Spec
使用同一个 UUID 执行推理。
Collection 的文本 Embedding 是必选且不可原地替换；VLM 和视觉 Embedding 可选。
Agent 可为路由、上下文、回答和记忆分别绑定不同模型。

### 现有本地库切换

- 使用 UUID 调用的客户端须和 Model Serving 同一维护窗口更新。内部推理请求用
  `model_id`；OpenAI 兼容接口仍用 `model` 字段，但值改为同一个 UUID。
- 备份并停写后，审阅并执行
  `database/oracle/operations/model_serving/use_model_uuid_for_inference.sql`。
  脚本校验原有 Agent 记忆索引画像的模型和配置摘要，替换为 UUID 后才删除旧列。
  保留模型 UUID、业务绑定、所有模型记录和向量；遇到无法核实的画像时应停止切换。
- 旧的 `knowledge.aiops.served_model_names` 配置不再读取，保留各表示空间的
  `model_ref` UUID。切换前完成正在执行的知识任务。旧版冻结的解析审批和已发布表示
  空间的推理配置摘要可能包含旧名字，不能直接当作新摘要继续运行；应在切换前盘点，
  切换后按业务流程重新审批或重建受影响的索引。系统不会自动改写历史审批或冻结任务，
  也不会因请求带有旧摘要而跳过版本校验。

`model_ocr` 是 AIOps 对话图片文字提取的独立 OCR 进程，目录类别为 `6`，支持
`local_tesseract`、`local_easyocr`、`local_rapidocr` 与 `api_deepseek_ocr`。它不调用
Knowledge Core 的 Docling 解析链路；部署可复用同一台主机的 Tesseract 二进制和语言包，
但模型目录、模型池、进程和调用链均独立。Knowledge Core Parser 仍按既有配置使用
Docling 及其独立的 DeepSeek OCR 增强端点。
