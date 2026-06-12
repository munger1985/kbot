## 2026-05-15 kbot v3.3.0

# 数据库变更日志

## 新增表
### 📄 表名：kbot_md_parser_conf
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ | `parser_conf_id` | `Numeric(38, 0)` | 自增ID，主键 |
| ➕ | `domain_id` | `Numeric(38, 0)` | Domain ID, referencing domain table |
| ➕ | `engine` | `String(10)` | Parser engine：Enum：ParserEngine |
| ➕ | `parser_params` | `JSON` | Parser parameters |
| ➕ | `created_by` | `String(256)` | Creator user |
| ➕ | `created_time` | `Date` | Creation time (默认当前时间) |
| ➕ | `updated_by` | `String(256)` | Updater user |
| ➕ | `updated_time` | `Date` | Update time (默认及更新时自动更新) |

### 📄 表名：kbot_graph_knowledge_vertices
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ | `vertex_id` | `VARCHAR2(64)` | 顶点（实体）唯一标识，主键 |
| ➕ | `vertex_name` | `VARCHAR2(255)` | 实体或概念的实际名称 |
| ➕ | `vertex_type` | `VARCHAR2(64)` | 实体的业务大类分类 |
| ➕ | `description` | `CLOB` | LLM提炼的简要文本定义或描述 |
| ➕ | `attributes` | `JSON` | 动态扩展属性（如设备型号、指标单位等） |
| ➕ | `name_vector` | `VECTOR` | 实体名称/描述生成的向量嵌入，用于消歧 |
| ➕ | `created_at` | `TIMESTAMP` | 实体的首次创建或抽取时间 |

### 📄 表名：kbot_graph_knowledge_edges
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ | `edge_id` | `VARCHAR2(64)` | 边的唯一标识，主键 |
| ➕ | `source_id` | `VARCHAR2(64)` | 关系的源顶点ID（外键关联 vertices） |
| ➕ | `target_id` | `VARCHAR2(64)` | 关系的目标顶点ID（外键关联 vertices） |
| ➕ | `relation_type` | `VARCHAR2(128)` | 关系的语义类型（如属于、导致等） |
| ➕ | `weight` | `NUMBER(10)` | 关系权重值（提及频次），默认值为1 |
| ➕ | `attributes` | `JSON` | 该条边特有的附加信息 |
| ➕ | `updated_at` | `TIMESTAMP` | 关系的最后抽取或权重更新时间 |

### 📄 表名：kbot_graph_edge_chunk_map
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ | `edge_id` | `VARCHAR2(64)` | 边的唯一标识（联合主键，外键关联 edges） |
| ➕ | `chunk_id` | `VARCHAR2(64)` | 提取出该关系的原始文档切片ID（联合主键） |
| ➕ | `file_id` | `VARCHAR2(64)` | 冗余存储的文档唯一标识，便于级联清理 |
| ➕ | `created_at` | `TIMESTAMP` | 该条关系在当前切片下的提取时间 |

## 表字段变更
### 📄 表名：kbot_md_kb_files
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ | `batch` | `str` | `批次名称` |
| 🗑️ | `batch_id` | `int` | 旧字段 |

### 📄 表名：kbot_md_kb
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| 🗑️ | `txt_embed_model_id` | `int` | 旧字段 |
| 🗑️ | `img_embed_model_id` | `int` | 旧字段 |
| 🗑️ | `img2txt_model_id` | `int` | 旧字段 |
| 🗑️ | `llm_model_id` | `int` | 旧字段 |
| ➕ | `engine` | `str` | `知识库解析引擎类型` |
| ➕ | `models` | `json` | `知识库关联的模型配置参数` |
| ➕ | `dbconf` | `json` | `知识库关联的数据库配置参数` |

### 📄 表名：kbot_md_user_profile
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ | `entity_relations` | `JSON` | 轻量级实体关联，如产线-负责人 |
| ➕ | `correction_history` | `JSON` | 用户订正过的错误事实或偏好 |

### 📄 表名：kbot_md_conv_context
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ | `current_plan` | `JSON` | TaskPlanner 生成的当前待执行步骤 |
| ➕ | `step_outputs` | `JSON` | 上一个执行步骤的输出 |
| ➕ | `last_relevance_score` | `NUMBER(2,1)` | 上一个执行步骤的相关性评分 |
| ➕ | `active_topic` | `VARCHAR2(512)` | 当前活跃话题标签 |

### 📄 表名：kbot_md_memory_entry
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ | `user_id` | `VARCHAR2(256)` | 用户ID |
| ➕ | `thought` | `CLOB` | LLM 在改写阶段的思考过程 |
| ➕ | `current_plan` | `JSON` | TaskPlanner 生成的当前待执行步骤 |
| ➕ | `reasoning_path` | `JSON` | 推理路径 |
| ➕ | `turn_type` | `VARCHAR2(64)` | 轮次类型 |
| ➕ | `blocks` | `JSON` | 流式响应块 |
| 🗑️ | `retrieved_chunks` | `JSON` | 搜索到的chunk，移动到blocks方便前端渲染 |

### 📄 表名：kbot_md_agent
| 操作类型 | 字段名称 | 数据类型 | 备注 |
| :--- | :--- | :--- | :--- |
| ➕ 新增字段 | `models` | `JSON` | AI模型统一配置（整合原LLM、Embedding、Reranker配置） |
| ➕ 新增字段 | `category` | `NUMBER(1,0)` | 代理分类，0：通用代理，1：运维代理 |
| 🗑️ 删除字段 | `llm_id` | `NUMBER(38,0)` | 旧字段，已迁移至 `models` |
| 🗑️ 删除字段 | `llm_params` | `JSON` | 旧字段，已迁移至 `models` |
| 🗑️ 删除字段 | `embedding_model_id` | `NUMBER(38,0)` | 旧字段，已迁移至 `models` |
| 🗑️ 删除字段 | `feedback_similarity_flag` | `NUMBER(1,0)` | 旧字段，已删除 |
| 🗑️ 删除字段 | `synonym_similarity_flag` | `NUMBER(1,0)` | 旧字段，已删除 |
| 🗑️ 删除字段 | `reranker_model_id` | `NUMBER(38,0)` | 旧字段，已迁移至 `models` |
| 🗑️ 删除字段 | `reranker_topk` | `NUMBER(38,0)` | 旧字段，已迁移至 `models` |
| 🗑️ 删除字段 | `reranker_score_threshold` | `NUMBER(38,0)` | 旧字段，已迁移至 `models` |


## 删除表
### 🗑️ 表名：kbot_md_kb_batch

---
# APP 变更日志
1. 上传接口：参数batch_id, batch_name已删除，使用batch (str)替代
2. 删除kb接口：参数batch_id, batch_name， file_paths已删除，使用batch (str)替代

3. `kbot_md_agent` 表的 `models` 字段，已整合原LLM、Embedding、Reranker配置。JSON `key` 命名如下：
```json
{
    "do_rerank": false, 
    "llm_model": "deepseek-chat", 
    "llm_top_k": 50, 
    "llm_top_p": 0.8, 
    "rerank_model": "", 
    "rerank_top_k": 9, 
    "llm_max_tokens": 8192, 
    "llm_temperature": 0.2, 
    "txt_embedding_model": "Qwen/Qwen3-Embedding-4B"
}
```

4. `kbot_md_kb` 表的 `models` 字段，已整合原LLM、Embedding、Reranker配置。JSON `key` 命名如下：
```json
{
    "llm_model": "deepseek-chat", 
    "vlm_model": "qwen3-vl-plus", 
    "txt_embedding_model": "Qwen/Qwen3-Embedding-4B"
}
```