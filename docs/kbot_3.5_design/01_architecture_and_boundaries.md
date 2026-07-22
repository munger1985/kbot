# 3.5 架构与服务边界

## 服务拓扑

```text
KM Portal / Main API
       │ HTTP
       ▼
Knowledge Core ── claim/result ──► Parser Worker ──► Embedding / LLM / VLM
       │
       ├─ Discovery：找 Bundle / Document
       └─ Evidence：找可引用内容

Main API / Agent Router ──► Data Query ──受控 SQL──► DB Executor
```

KC 建议作为独立 FastAPI 进程（建议端口 18090），保持同一仓库、Schema 与部署编排。它拥有 Collection、Bundle、Document、版本、解析结果、任务和检索投影；外部系统只通过 API 操作，不直连 `KBOT_KC_*` 表。

Domain 是 APEX 的数据与权限隔离边界，Collection 只能在一个 Domain 内创建和查询；`app_id` 由 `base.toml` 固化并仅写入 Collection，以支持 APEX 直连数据库页面过滤。其他 KC 表通过 `collection_id` 自动关联 Scope，不作为客户端业务参数或用户权限 Scope。详细约束见[步骤 0 详细设计](07_step_0_scope_and_tenancy.md)。

## 职责划分

| 组件 | 负责 | 不负责 |
| --- | --- | --- |
| Knowledge Core | 入库编排、版本切换、任务状态、Discovery、Evidence、关系 | Agent/Skill、最终回答、SQL 执行 |
| Parser Worker | Docling/OCR/VLM、分块、结构化提取、Embedding、结果回传 | 发现来源、修改 Bundle 状态、直写 KC 表 |
| Portal | 来源增量、字段规范化、附件下载、一次 Bundle 投递 | KBot 数据库和解析状态机 |
| Data Query | 数据源与语义模型、NL2SQL、权限、审计、表格结果 | 文档解析和 Evidence 生命周期 |
| DB Executor | SQL 安全校验、限流、受控执行 | 数据语义、调用方权限决策 |

Collection 的消费者（首期为 V2 Agent）通过 KC 的多对多 Binding 契约登记引用；一个 Agent 的所有已启用 Binding 默认平权，显式选择 Collection 才收窄检索范围。KC 以 Binding 作为删除保护依据，而不直接依赖 Agent 表。存在有效 Binding 的 Collection 不可删除；未被引用的 Collection 删除时连同其 Bundle、Evidence 与对象存储内容异步物理清除。详见[步骤 0 详细设计](07_step_0_scope_and_tenancy.md)。

现有 `kbot_db_executor.py` 保持为底层执行服务。其当前接收 SQL 与连接配置，不能直接作为用户或 Agent 的问数入口；后续 Data Query 必须以注册数据源、凭据引用、行列权限、SQL AST/白名单、成本限制和审计来封装它。

## 问文 Skill 与 Agent 的接缝

3.5 将旧链路标记为 V1，将新 KC 链路标记为 V2。V1 的 `TxtBaseSearch` / `DocService`、旧问文 Skill 和旧 API 暂时保持不动；V2 问文 Skill 应重构为面向 KC 的检索编排器（建议命名 `KnowledgeRetrievalSkillV2`），直接消费 KC 的领域 DTO，而不是将 Evidence 压缩映射为旧 `TxtBaseSearchResult`。V2 不读旧 File/Chunk 表，也不设置请求内 V1 回退。

```text
用户问题 + Collection/权限上下文
  → Discovery：召回并排序 Bundle / Document
  → 选择候选范围与检索计划
  → Evidence：混合检索、主 View 去重、邻接扩展、关系扩展
  → Citation Pack：Evidence 内容 + 稳定定位 + 质量/覆盖信息
  → 回答模型 / 上层 Agent
```

V2 Skill 负责问题意图、候选范围策略、上下文预算和回答前的证据覆盖校验；KC 负责权限过滤后的召回、排序、版本/视图选择和引用定位。回答模型只能使用 Citation Pack 中的内容，不能绕过 KC 自行从文件或旧表取 Chunk。V2 Doc Orchestrator、Root Agent 与 SSE 输出使用 Bundle/Document/Evidence 引用结构；V1 的接口与 SSE 独立保留，调用方通过显式版本或路由选择，不能混合两种引用 DTO。

V2 对外 API 统一使用 `/api/v2/knowledge/*` 前缀，内部 Worker 协议使用 `/internal/v2/knowledge/*`；V1 路径不改。版本选择应固定在 App/Agent/路由配置或明确请求字段上，并记录在审计日志中，不能根据请求失败自动降级。

未来 Agent Router 可按意图调用 `retrieval`、`data_query` 或二者并行；混合回答必须分别保留 Evidence 引用或 `query_result_id`。
