# Changelog — kbot3 功能新增

> **日期：** 2026-07-14
> **范围：** 技能系统、文档解析、视觉 Embedding、监控工具、utils 重构、自动运维、文档召回

---

## 1. 技能系统 (skills/)

### 覆盖的技能库（5 个）

| 技能 | 操作 | 说明 |
|---|---|---|
| db-analysis-skill | 覆盖 | `db_analysis_skill_core.py` + `sufficiency_checker.py` + `skill.md` |
| db-metric-skill | 覆盖 | `db_metric_skill_core.py`（含 Zabbix 分支）+ `skill.md` |
| image-search-skill | **新建** |  `__init__.py` / `image_search_core.py` / `skill.md` |
| ops-heal-skill | 覆盖 | `ops_heal_skill_core.py` + `skill.md` |
| reasoning-skill | 覆盖 | `reasoning_core.py` + `skill.md` |

### 未变更的技能库

| 技能 | 原因 |
|---|---|
| ask-doc-skill | 已保持一致 |
| ask-data-skill | 两边查询方式不同 |
| echarts-skill | 已保持一致 |
| chit-chat-skill | 已保持一致 |

---

## 2. 文档解析 (microservices/file_processor/)

### 新增解析器（12 个）

| 文件 | 大小 |
|---|---|
| `parsers/chunk_reflector.py` | 15.9 KB |
| `parsers/hierarchy_builder.py` | 7.4 KB |
| `parsers/hierarchy_merger.py` | 6.8 KB |
| `parsers/image_extractor.py` | 3.1 KB |
| `parsers/layout_cluster.py` | 4.0 KB |
| `parsers/metadata_extractor.py` | 4.6 KB |
| `parsers/page_span_stitcher.py` | 4.2 KB |
| `parsers/precision_analyzer.py` | 7.9 KB |
| `parsers/precision_engine.py` | 2.3 KB |
| `parsers/section_chunker.py` | 6.3 KB |
| `parsers/structure_quality_gate.py` | 9.8 KB |
| `parsers/structure_repairer.py` | 13.9 KB |


### 覆盖的服务层文件（5 个 + 1）

| 文件 | 说明 |
|---|---|
| `services/__init__.py` | 覆盖 |
| `services/docling_service.py` | 覆盖 |
| `services/engine.py` | 覆盖 |
| `services/file_processor.py` | 覆盖 |
| `services/txt_to_md.py` | 覆盖 |
| `parser_schema.py` | 覆盖 |

### 新建 DAO 文件（4 个）

| 文件 | 说明 |
|---|---|
| `dao/entities/doc_metadata.py` | 文档元数据实体（适配 Oracle 类型） |
| `dao/entities/doc_relation.py` | 文档引用关系实体 |
| `dao/repositories/doc_meta_repo.py` | 文档元数据仓库（Oracle MERGE INTO 语法） |
| `dao/repositories/doc_relation_repo.py` | 文档引用关系仓库 |

### 更新 DAO 文件（1 个）

| 文件 | 变更 |
|---|---|
| `dao/entities/txt_chunk.py` | 追加 6 个字段（hierarchy_path/depth/heading_level/parent_chunk_id/section_id/created_at） |
| `dao/repositories/txt_chunk_repo.py` | SQL 追加新字段到查询结果 |

---

## 3. 视觉 Embedding (Visual)

### 新增微服务层（5 个文件）

| 文件 | 说明 |
|---|---|
| `microservices/visual/visual_service.py` | 视觉 embedding 服务 |
| `microservices/visual/model.py` | 模型基类 |
| `microservices/visual/model_factory.py` | 模型工厂 |
| `microservices/visual/model_pool.py` | 模型池 |
| `microservices/visual/schema.py` | 输入输出 schema |

### 新增服务层（4 个文件）

| 文件 | 说明 |
|---|---|
| `services/visual/search_engine.py` | 视觉搜索引擎 |
| `services/visual/visual_indexer.py` | 视觉索引器 |
| `services/visual/visual_search.py` | 视觉搜索 |
| `services/visual/cleanup.py` | 清理工具 |

### 新增 DAO 文件（4 个）

| 文件 | 说明 |
|---|---|
| `dao/entities/extracted_image.py` | 提取图片实体 |
| `dao/entities/page_visual_index.py` | 页面视觉索引实体 |
| `dao/repositories/extracted_image_repo.py` | 提取图片仓库（Oracle 向量搜索语法） |
| `dao/repositories/page_visual_index_repo.py` | 页面视觉索引仓库 |

---

## 4. 监控工具 (monitor)

### 覆盖文件

| 文件 | 说明 |
|---|---|
| `utils/monitor/__init__.py` | 导出 ZabbixProvider |
| `utils/monitor/base.py` | 增加 `from_zabbix()` 解析方法 |
| `utils/monitor/registry.py` | 完善 Zabbix 分支 |
| `utils/monitor/zabbix.py` | **桩→完整实现**（JSON-RPC API） |
| `configuration/metrics_mapping.yaml` | 覆盖（含 23 个指标的 Zabbix Item Key） |

### 配置层变更

| 文件 | 变更 |
|---|---|
| `core/config/settings.py` | 新增 `ZabbixConfig` + `get_zabbix_config()` |
| `configuration/base.toml` | 新增 `[zabbix]` 配置段 |
| `configuration/example/base.toml.example` | 新增 `[zabbix]` 示例配置段 |

---

## 5. utils 工具层重构

### 覆盖的文件（11 个）

| 文件 | 说明 |
|---|---|
| `utils/__init__.py` | 覆盖 |
| `utils/sanitize.py` | 覆盖 |
| `utils/thread.py` | 覆盖 |
| `utils/clients/__init__.py` | 覆盖 |
| `utils/clients/model.py` | 覆盖 |
| `utils/clients/ops.py` | 覆盖 |
| `utils/clients/sql.py` | 覆盖 |
| `utils/codec/__init__.py` | 覆盖 |
| `utils/codec/encoder.py` | 覆盖 |
| `utils/codec/serializer.py` | 覆盖 |
| `utils/monitor/`（5 文件） | 在监控工具步骤中已覆盖 |

### 删除的文件

| 文件 | 原因 |
|---|---|
| `utils/sse.py` | SSE 由 FastAPI StreamingResponse 直接处理 |

### 保留的 kbot3 特有文件

| 文件 | 说明 |
|---|---|
| `utils/codec/oracle_vec_handler.py` | Oracle 23ai 向量处理适配 |

---

## 6. 自动运维 (Auto Operations)

### 覆盖的 agent/ 层文件（20+ 个）

| 目录 | 文件 |
|---|---|
| `agent/agent/` | `ops_agent.py` / `doc_agent.py` / `root_agent.py` / `dify_service.py` |
| `agent/common/` | `ops_context.py` / `diagnostic_tools.py` / `business_context.py` / `skill_context.py` / `mixin.py` |
| `agent/memory/` | `memory_service.py` / `context_manager.py` / `state_manager.py` |
| `agent/orchestrator/` | `ops_orchestrator.py` / `intent_router.py` / `root_orchestrator.py` / `doc_orchestrator.py` |
| `agent/planner/` | `ops_planner.py` / `decision_engine.py` / `llm_planner.py` |
| `agent/prompt/` | `default_prompt.py` |
| `agent/` | `__init__.py` |

### 新增的规划器文件（5 个）

| 文件 | 说明 |
|---|---|
| `agent/planner/execution_scheduler.py` | 执行调度器 |
| `agent/planner/plan_validator.py` | 计划验证器 |
| `agent/planner/skill_io_map.py` | 技能输入输出映射 |
| `agent/planner/workflow_compiler.py` | 工作流编译器 |
| `agent/planner/workflow_planner.py` | 工作流规划器 |

### 更新的 Ops DAO

| 文件 | 变更 |
|---|---|
| `dao/entities/ops_db_instance.py` | 追加 `zabbix_host_name` 字段 |
| `services/basic/ops_db_instance_service.py` | 返回 `zabbix_host_name` |
| `services/basic/ops_agent_conf_service.py` | 返回 `zabbix_host_name` |

---

## 7. 文档召回 (Search)

### 新建/覆盖的文件

| 文件 | 操作 | 说明 |
|---|---|---|
| `services/search/doc_search.py` | 新建 |  `doc/search.py`  |
| `services/search/doc_service.py` | 覆盖 | 文档检索服务 |
| `services/search/reranker.py` | 新建 | LLM 重排序（替代原 rerank.py） |
| `services/search/result.py` | 覆盖 | 搜索结果模型 |

### 删除的文件

| 文件 | 原因 |
|---|---|
| `services/search/rerank.py` | 被 reranker.py 替代 |
| `services/search/kb_search.py.old` | 旧版备份 |


---

## 8. 删除 Reranker 微服务

| 路径 | 操作 |
|---|---|
| `microservices/reranker/` | **整目录删除**（功能由 `services/search/reranker.py` 中的 LLM 重排序替代） |

---
