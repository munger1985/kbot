# 步骤 7 详细设计：V1 知识链路退役清单

## 目的与边界

本清单用于 V2 稳定验收后的**受控删除**，不是当前上线动作。删除前必须在生产 Schema、APEX 页面、定时任务、Portal、Agent 和外部调用方中完成依赖扫描；任何名称中含 `KB` 的表都不能仅凭名称自动删除。

## 可作为 V1 知识链路候选删除项

| 类别 | 当前实现/表 | 退役前验证 |
| --- | --- | --- |
| V1 上传 API | `api/routers/kb_router.py` 的 `/api/kb/*`，`KBController` | Portal、APEX、脚本和外部消费者均已停止调用 |
| V1 写服务 | `services/kb/file_service.py`、`chunk_service.py`、V1 `KBService` 的文件路径 | 无路由、Scheduler 或运维脚本导入/调用 |
| V1 检索 | `services/search/kb_search.py:TxtBaseSearch`、`doc_service.py:DocService`、`TxtBaseSearchResult`、旧 reranker | V2 Agent/Skill 完整替代，Graph/Visual 等消费者已迁移或退役 |
| V1 解析写入 | Parser 中 `_get_embeddings()`、`_save_chunks()`、V1 File 状态轮询 | V2 Parser Worker 已稳定运行且没有 V1 待处理任务 |
| V1 核心表 | `kbot_md_kb`、`kbot_md_kb_files`、`KBOT_BIZ_TXT_EMBEDDING` | 记录数已归档，所有 V1 查询/外键/视图/索引依赖为零 |
| V1 附属表 | `KBOT_MD_DOC_METADATA`、`KBOT_MD_DOC_RELATION`、`KBOT_MD_EXTRACTED_IMAGES` | 已确认只服务 V1 File/Chunk；V2 对应能力已由 Discovery/Relation/Evidence payload 取代 |

上表中的文档元数据、关系和图片表当前均以 `kb_id/file_id/chunk_id` 为锚点，因此在确认无其他产品消费者后应与 V1 核心表同批退役，而不是迁移其内容到 KC。V2 按来源重新解析并生成自己的 Version、Evidence、Relation 和工件。

## 必须保留或逐行拆分的对象

| 对象 | 原因与处理 |
| --- | --- |
| `kbot_md_domain` | APEX 的 Domain 隔离根，KC Collection 仍依赖它，必须保留。 |
| `kbot_md_agent`、App/用户/模型/认证表 | 非 V1 知识表，保留。 |
| `kbot_md_agent_conf` | `tool_id` 当前兼容 KB/其他工具。先删除或迁移其中的 KB Tool 行到 `KBOT_KC_COLLECTION_BINDING`，不能整表删除，除非已证明没有其他工具配置。 |
| `kbot_md_parser_conf` | 可能仍包含共享 Parser 配置；先将 V2 policy snapshot/配置迁出，再决定是否删除。 |
| `KBOT_GRAPH_*` 与 Graph 服务 | 当前与 Chunk 映射存在关联，但可能是独立图谱能力。必须先决定 V2 Relation 是否完全替代该产品功能；不能随 V1 KB 表自动删除。 |
| `kbot_ops_*`、审计/监控表 | 由其他 AIOps/平台功能使用，保留。 |

## 删除前门禁

1. 静态扫描：源码、部署脚本、APEX SQL、数据库视图/包/触发器、CI 和外部集成中不存在 V1 API、表名或 Repository 引用。
2. 运行扫描：稳定期内 V1 路由访问量为零，V1 Parser 队列为空，无活动 V1 Agent 配置行。
3. 数据验证：目标 Collection 的 V2 Bundle/Manifest/附件数量与需保留来源一致；评测和 `doc_results_v2` 指标达标。
4. 归档：导出 V1 表行数、DDL、必要审计和对象存储清单；明确归档保留期限与恢复责任人。
5. 演练：在非生产副本执行删除顺序、验证 KC 不受影响，并验证 APEX 关键页面和非知识 Agent 功能。

## 执行顺序

```text
停止 V1 路由和 Parser
  → 清理/迁移 AgentConf 中的 KB Tool 行
  → 删除 V1 API、Service、Repository、Entity 和测试
  → 删除 V1 视图/索引/触发器
  → 按依赖顺序删除 V1 附属表，再删除 File/Chunk/KB 核心表
  → 回收 V1 对象存储目录和配置
```

实际 DDL 必须由独立迁移脚本生成并在执行前列出精确对象名、依赖检查结果和归档位置；禁止在应用启动脚本或通用 `purge` 中隐式删除。删除完成后，从文档、监控告警和权限配置中移除 V1 概念，系统只暴露 KC V2。
