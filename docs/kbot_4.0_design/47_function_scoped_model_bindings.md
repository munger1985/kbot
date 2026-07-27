# 按功能角色绑定模型

## 设计目标

KBot 4.0 不为 Agent 或 Collection 配置一个“万能模型”。模型由业务归属对象
按功能角色绑定，使低成本模型承担路由、改写和记忆整理，高质量模型只用于最终
回答或高价值视觉理解。模型服务地址仍由各服务 TOML 配置。Agent 与
Collection 都保存模型目录的 UUIDv7 `model_id`，执行时解析并冻结
`model_id + served_model_name + category + 配置指纹`。Run、Parse View 和
Ingestion Job 均不依赖随后发生的非向量模型配置修改。

## Agent 模型角色

`KBOT_AGENT_DEFINITION.MODELS_JSON` 保存“功能角色 → 模型 UUIDv7”映射：

| 角色 | 作用 | 变更规则 |
| --- | --- | --- |
| `router_llm` | 多能力路由和计划 | 可选、可修改，新 Run 生效 |
| `context_llm` | 指代消解、问题改写 | 必选、可修改，新 Run 生效 |
| `composer_llm` | 最终回答和引用组织 | 必选、可修改，新 Run 生效 |
| `memory_llm` | 摘要、记忆提取和冲突判断 | 必选、可修改，新领取的 Job 生效 |
| `query_vlm` | 用户图片理解 | 可选、可修改 |
| `chart_llm` | ECharts 配置生成 | 可选、可修改 |
| `memory_embedding` | 长期及情景记忆向量 | 必选，创建后永久不可修改 |
| `do_rerank` | 是否启用 KC 对象级及 Evidence Group LLM 重排 | 可修改，新 Run 生效 |

模型映射和 `do_rerank` 不能藏在 `CONFIG_JSON`。新增能力可增加合法角色键，
不需要修改表结构；角色名使用小写 `snake_case`。`CONFIG_JSON.memory` 只保存
`shared_keys`、`episodic_enabled` 等行为策略。创建 Run 时把全部角色模型写入
`CONFIG_SNAPSHOT_JSON.agent.models`。每个角色的快照包含 `model_id`、
`served_model_name`、`category` 和 `config_fingerprint`，运行中的 Task
不读取 Agent 最新值。
Memory Job 在领取时读取 `models.memory_llm`；Memory Index Profile 再次核对
不可变的 `models.memory_embedding` 和全局维度。

## Knowledge Core 模型角色

Collection 独立拥有解析和检索模型，不继承 Agent：

| 角色 | 作用 | 变更规则 |
| --- | --- | --- |
| `parser_llm` | 结构修复、Manifest、关系提取 | 必选、可修改，新 Parse View 生效 |
| `parser_vlm` | 页面、表格、图片视觉解析 | 可选、可修改，新 Parse View 生效 |
| `embedding` | Evidence、Discovery 和查询向量 | 必选，创建后永久不可修改 |
| `visual_embedding` | 图片索引和图搜图查询 | 可选，首次启用后永久不可修改 |
| `retrieval_llm` | Bundle 选择、证据审阅等受控判断 | 必选、可修改，新检索请求生效 |

物理上只保存 `KBOT_KC_COLLECTION.MODELS_JSON`，值为模型目录中的 UUIDv7
字符串。创建接口及 `PUT .../models` 接收完整映射，应用层校验必选角色，并
禁止删除或替换已经设定的 `embedding`、`visual_embedding`。新增解析或检索
能力只需定义新角色，无需增加列。

KC 查询必须按 Collection 的模型空间分组；Agent 只提交自然语言或图片，不得
传入模型名称。多个 Collection 使用不同 Embedding 时分别生成查询向量，再在
各自空间内召回和融合排名，禁止跨模型直接计算向量距离。

## 不可变边界

不可变规则针对所有会产生持久化向量的模型，而不仅是维度：即使两个模型维度
相同也不能替换。全局 `kbot.toml` 的 `embedding_dimension` 约束所有文本
Embedding 定义；模型名称、类别、维度和配置指纹必须同时匹配。需要另一
Embedding 时创建新的 Agent 或 Collection，不能重建、双写或代际切换。
