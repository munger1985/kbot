# Knowledge Core 文件职责与服务目录重组

## 1. `knowledge_core/` 文件职责

`knowledge_core/` 是 KC 的领域库与应用库，不等同于一个 HTTP 进程。HTTP API、Parser Worker、Projection Worker 都依赖它，但它不依赖 Agent、Skill 或 V1 `File/Chunk`。

### 根与适配器

| 路径 | 职责 |
| --- | --- |
| `__init__.py`、各子目录 `__init__.py` | Python 包边界和受控导出，不包含业务状态。 |
| `client.py` | KC V2 HTTP 客户端，供 Skill/Agent 调用 Discovery、Evidence 和 Citation 接口。 |
| `adapters/embedding.py` | 通过模型配置 HTTP DTO 和推理 Gateway 适配 Embedding；校验 Collection 绑定模型，不读取模型服务 ORM。 |
| `adapters/local_object_store.py` | 开发环境本地不可变对象存储；生产环境可替换 OCI/S3 实现。 |
| `adapters/local_parser_artifact_store.py` | 开发环境内容寻址的 Parser 工件存储，负责幂等写入和旧视图工件清理。 |

### API

| 路径 | 职责 |
| --- | --- |
| `api/intake_router.py` | KM Asset 与普通用户文件 multipart 接收，只做 DTO/HTTP 映射，不持有 DB Session。 |
| `api/collection_router.py` | Collection 创建、查询、启停、删除，以及 Agent 多对多绑定。 |
| `api/status_router.py` | Domain 范围内 Bundle、Revision、Member 状态查询。 |
| `api/parse_task_router.py` | Parser Worker 的 PARSE claim、租约、工件、Evidence 批次、完成/失败回调。 |
| `api/index_task_router.py` | Evidence/Discovery INDEX Worker 的租约和结果回调。 |
| `api/profile_task_router.py` | Bundle/Document PROFILE Worker 的租约和结果回调。 |
| `api/purge_task_router.py` | 未绑定 Collection 的异步物理清理协议。 |
| `api/discovery_router.py` | 第一阶段 Bundle/Document 画像召回 API。 |
| `api/evidence_router.py` | 第二阶段候选范围内 Evidence、上下文组和 Citation Pack API。 |

### Application

| 路径 | 职责 |
| --- | --- |
| `application/intake.py` | 接收后的最终事务：创建 Bundle/Revision/Document/Version/Parse View/Job。 |
| `application/multipart.py` | multipart 五阶段编排：校验、暂存、发布、数据库提交、补偿。 |
| `application/collections.py` | Collection 生命周期和 Binding 用例。 |
| `application/collection_purge.py` | PURGE Job 租约和后代数据/本地对象清理。 |
| `application/parse_tasks.py` | Parser 回调应用服务、Evidence 落库、质量门和 Parse View 激活。 |
| `application/discovery.py` | 构造确定性 Bundle/Document Profile，并投递 Discovery INDEX。 |
| `application/indexing.py` | 唯一的 Evidence/Discovery 文本向量生成路径；模型身份由 Collection 决定。 |
| `application/query_embeddings.py` | 按 Collection 绑定模型分组生成查询向量。 |
| `application/retrieval.py` | Discovery 文本/向量/RRF 召回和 Bundle 级聚合。 |
| `application/evidence_retrieval.py` | Evidence 候选、邻接上下文、Evidence Group 和引用单元。 |
| `application/retrieval_plan.py` | 版本化 QueryPlan、对象级候选选择和 Evidence 支持判断接口；含安全降级实现。 |
| `application/grounding.py` | 回答 Claim 覆盖校验和 `doc_results_v2` 投影。 |
| `application/answer_generation.py` | Root Agent V2 的回答模型边界，不负责检索和权限。 |
| `application/task_dto.py` | Agent→Skill 的版本化任务 DTO，为 4.0 多 Agent 委派预留。 |
| `application/scope.py` | Domain、Agent Binding、Collection 状态和安全等级校验。 |
| `application/status.py` | 入库进度和对象状态的只读查询。 |
| `application/sse_v2.py` | Grounded Answer 的 SSE 事件序列化。 |

### Domain、Parsing、Persistence、Ports

| 路径 | 职责 |
| --- | --- |
| `domain/intake.py` | 与 FastAPI/数据库无关的 Bundle、Document 声明和校验。 |
| `domain/manifest.py` | KC 拥有的 Bundle Manifest Markdown 渲染。 |
| `domain/parse_tasks.py` | 任务 claim、租约和过期规则。 |
| `domain/revision_status.py` | Member 状态聚合为 Revision/Bundle 可用状态。 |
| `parsing/converter.py` | Docling 转换适配器，只负责底层格式转换。 |
| `parsing/docling_adapter.py` | Docling 输出到 Atom IR 的无损映射。 |
| `parsing/ir.py` | Atom、Reading Order、Structure 的不可变中间表示。 |
| `parsing/reading_order.py` | 页面阅读顺序和跨页连续性。 |
| `parsing/structure_builder.py` | 标题树、章节和语义块构建。 |
| `parsing/evidence_planner.py` | 根据结构 IR 规划 SECTION/PARAGRAPH/TABLE/SHEET 等 Evidence。 |
| `parsing/quality.py` | 覆盖率、标题树、定位和短片段质量门。 |
| `parsing/spreadsheet_artifact.py` | Excel 单元格坐标/表头结构化工件，供未来 Data Query 使用。 |
| `parsing/pipeline.py` | 串联 Docling 后处理、质量评估、Evidence 和 Parser 工件。 |
| `parsing/contracts.py` | Evidence、Artifact、Locator、Hash 和输出指纹契约。 |
| `persistence/uow.py` | KC 显式事务边界和全部 KC Repository 组合。 |
| `repositories/` | KC 专属 Collection、Ingestion、Discovery、Evidence 和 Relation Repository；不再放在通用 `dao/repositories`。 |
| `ports/object_store.py` | 对象存储抽象接口。 |
| `ports/parser_artifact_store.py` | Parser 工件存储抽象接口。 |

### Workers

| 路径 | 职责 |
| --- | --- |
| `workers/parser/client.py` | Parser→KC 的 HTTP 租约/结果客户端。 |
| `workers/parser/worker.py` | Docling 转换、视觉增强、Evidence 批次提交的 Parser Worker。 |
| `workers/parser/visual_enricher.py` | 受策略控制的 VLM 图片描述增强。 |
| `workers/projection/client.py` | PROFILE/INDEX/PURGE Worker 的内部 HTTP 客户端。 |
| `workers/projection/worker.py` | 画像生成、向量索引和 Collection 清理任务的统一 Worker 循环。 |

## 2. Parser 是否仍然需要

需要，但它不是旧 `FileProcessor` 的延续。当前 Parser 是 KC 的异步下属 Worker：

```text
KC API 接收入库
  → KBOT_KC_INGESTION_JOB(PARSE)
  → Parser Worker claim/heartbeat
  → Docling + Atom/Structure/Quality/Evidence
  → KC 回调并激活 Parse View
  → PROFILE → INDEX
```

Parser 不应访问 KC 数据库，也不应写旧 `KB/File/TxtChunk` 表。旧的 `microservices/file_processor/services/file_processor.py`、`services/docling_service.py` 和 `parsers/*` 仍服务 V1 File/Chunk 链路，当前只作为 legacy 代码保留，不是 KC Parser 的依赖。

## 3. 当前服务打包方案

本次已将可独立运行的 KC 进程整理为：

```text
apps/
├── knowledge_core_api/main.py          # KC HTTP API
├── knowledge_core_parser/main.py       # KC Parser Worker
└── knowledge_core_projection/main.py   # PROFILE/INDEX/PURGE Worker
knowledge_core/                          # KC 共享领域/应用库
microservices/                            # 模型托管、DB Executor 和 V1 legacy
```

根目录 `kbot_app_knowledge.py`、`kbot_app_parser.py`、`kbot_app_kc_worker.py` 目前只是兼容启动包装器；`start_kbot.sh` 已直接启动 `apps/` 下的实现。旧 `microservices/file_processor/kc_*.py` 也只保留导入 shim，避免开发期测试和外部脚本突然失效。

## 4. 是否每个服务独立打包

推荐“每个可部署进程独立打包，但共享库按包复用”，而不是把所有代码都放进独立仓库：

- `knowledge_core` 是 KC 领域包，API、Parser、Projection 三个进程共享它；共享的是领域规则、DTO、端口和客户端，不共享进程状态。
- `apps/knowledge_core_*` 是独立部署单元，分别拥有启动、健康检查、依赖和资源配置。
- Embedding/LLM/VLM/Visual 继续作为模型托管服务；它们共享 `model_serving/common` 的认证边界、模型目录、Pool 和配置 Registry，但不共享 KC 业务代码。
- V1 `microservices/file_processor` 进入 `legacy` 维护边界，待 V2 稳定后整体删除，而不是继续向其中添加 KC 功能。

4.0 的多 Agent/Skill 应沿用同一模式：`apps/agent_runtime` 只负责运行时和路由，`agent/`、`skills/` 是可版本化能力包，KC、Data Query、模型服务通过明确 Client/DTO 通信。只有需要独立扩缩容、独立发布或独立安全边界的组件才拆成新的部署单元。

## 5. 模型服务重组

模型托管已采用同样的结构：`model_serving/common` 放置共享模型池、模型配置 Repository、CRUD Registry 和管理 DTO；`embedding`、`llm`、`vlm`、`visual` 各自只包含该类别的 Provider、Factory、Pool、Schema 和推理服务。四个 `apps/ai_models_*` 进程分别独立启动、扩缩容和发布。

`AIModelEntity` 已迁移到 `model_serving/common/entities/ai_model.py`；Repository 的规范实现位于 `model_serving/common/model_repository.py`，原 `dao/entities/ai_model.py` 和 `dao/repositories/ai_model_repo.py` 仅作为兼容导出。共享 ORM 基础类型位于 `platform_core/persistence/orm.py`，旧 `dao/entities/base.py` 只做兼容导出。

KC 的索引和查询只依赖 `platform_clients.AIModelConfigClient` 读取模型快照，再通过
`AIModelClient` 调用向量服务；KC 不再导入 `AIModelEntity` 或
`AIModelRepository`。因此模型目录迁移到独立数据库时，KC 的持久化代码和任务协议无需修改。

## 6. Core 复用与未来拆仓

不建议在单仓库中复制多份 `platform_core/` 源码。推荐将其演进为版本化的 `platform-core` 包，包含配置加载、日志、认证、数据库连接、ORM 基础类型和跨服务契约；服务自己的 Entity、Repository、Application Service 和 API 必须留在对应服务包中。

开发部署时，各服务可以从同一仓库构建独立镜像并安装同一版本的 `platform-core`。未来拆成多个仓库时，复制的是包依赖或构建产物，而不是手工复制目录。若某服务需要不同日志/数据库实现，应通过 Core 的 Port/Adapter 替换，不应复制后直接修改共享代码。

当前边界如下：

```text
platform-core distribution (`platform_core/`)
  配置、日志、认证、DB Session、ORM 基础类型

model_serving/
  AIModelEntity、模型 Repository、模型配置 CRUD、Provider/Pool

knowledge_core/
  entities、persistence、Repository/Application/Parser/Projection

agent/ + skills/
  Agent/Skill 领域和版本化任务 DTO
```

### `utils/` 的处理

`utils/` 当前确实被 Agent、KC、V1 服务和模型客户端共同使用，但它不是一个
清晰的领域边界。暂不直接改名为 `platform_utils`，因为这个名称仍会形成新的
通用“杂物箱”。后续应按职责拆分：通用配置/日志/数据库/ORM 放入
`platform_core`，跨服务 HTTP 客户端逐步收敛到 `platform_clients`，稳定的请求/响应
DTO 放入 `platform_core/contracts`（规模扩大后也可独立为 `platform_contracts`）；文本清理、旧 SSE 和 V1 专用工具则留在各自服务
或 legacy 包中。当前 `utils/` 作为迁移期兼容入口保留，新增跨服务能力不得继续
无边界地堆入其中。

KC 的 `KBOT_KC_*` Entity 已位于 `knowledge_core/entities/`；旧
`dao/entities/kc` 只保留兼容导出。KC 拆成独立服务时，不需要携带 V1
`dao/entities`。

## 7. `apps/` 的归属与拆分策略

当前顶层 `apps/` 是单仓库内的“可部署进程入口清单”，其中的每个
`main.py` 已经是独立进程，并不代表这些进程共享运行时状态。因此在当前
分布式单体阶段，保留一个顶层 `apps/` 是可行的，`start_kbot.sh` 也可以按
入口选择性启动和打包。

但从服务所有权和未来拆仓角度，扁平目录不是最终形态。下一阶段建议先收敛
为显式命名空间：

```text
apps/
├── knowledge_core/
│   ├── api/main.py
│   ├── parser/main.py
│   └── projection/main.py
└── model_serving/
    ├── embedding/main.py
    ├── llm/main.py
    ├── vlm/main.py
    └── visual/main.py
```

真正拆仓时，再将每个命名空间连同对应领域包、`pyproject.toml`、配置示例、
Dockerfile 和测试整体移入独立服务仓库；`platform-core` 作为版本化依赖保留。
这样既不会因为现在就大规模移动路径影响开发，又能保证每个入口、配置、依赖
和发布流程都有明确的服务归属。
