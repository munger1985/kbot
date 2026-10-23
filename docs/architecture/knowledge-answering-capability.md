# Knowledge Answering 基础能力最终设计与一次性改造计划

> 状态：目标架构与实施基线。产品线分支已建立，KBot 4.1 的 KM 代码、Schema、路由、UI、部署工件和
> 测试已物理删除；Knowledge Answering 与 AIOps 接入改造尚未实施。本次状态不表示数据库升级、部署或
> 在线服务已经完成。
> AIOps 资产治理细节见
> [AIOps 运维知识库详细设计](../proposals/aiops-operations-knowledge-base-detailed-design.md)。

## 1. 强制决策

1. 知识问答由 Agent Runtime 与 Knowledge Core（KC）共同提供，不新增转发型
   `knowledge_answering` 服务，也不把 Knowledge Retrieval App 作为公共服务。
2. App 使用自己拥有的 Agent、服务端签发的 `AgentExecutionSpecV1` 和不可变知识范围接入；
   Conversation、Run、SSE、Document Skill、Composer 和引用校验只保留一套。
3. 一次性切换 API、DTO、Oracle canonical DDL/manifest、客户端、OpenAPI、页面和测试；不保留旧路由、
   兼容 DTO、双读、双写、回填或临时桥接。
4. `kbot4.1` 与 `km4.0` 已从同一基线提交分线；KM 只在 `km4.0` 维护，`kbot4.1` 已一次性
   物理删除全部 KM 代码、集成、Schema、测试和产品文档。终版标签后续只可补在共同基线上。
5. 浏览器不能提交 Agent、Collection、Revision、安全等级、模型或检索策略；无 Citation 时不得补写答案。

## 2. 目标与边界

目标是让 Knowledge Retrieval、AIOps 和未来 App 复用可信的多轮知识问答：上下文改写、精确范围、
全文/向量召回、流式回答、引用预览、断点重放和审计。用户可询问“Exadata 存储节点怎么维护”或
“查看某状态的命令是什么”，结果必须来自已发布、有权访问且能定位原文的资料。

基础化的是问答执行链路，不是各 App 的资产治理。KC 不判断 AIOps 发布态；Runtime 不拥有业务 Agent；
AIOps 知识不能代替当前 Target 证据、变更审批或实施后验证；未来 App 也不得调用 Knowledge Retrieval
App 的 BFF、私有表或私有服务。

## 3. 最终架构

```text
App 页面
  -> App BFF：鉴权、App-owned Agent、KnowledgeScopeProvider、ReferencePresenter
  -> Agent Runtime：Conversation / Turn / Run / SSE / Context Rewrite
       -> AppScopedSkill -> Document Skill -> KC Discovery / Evidence
       -> Response Composer -> Grounded Answer / References
  -> Knowledge Core：Agent Binding、精确 Bundle Revision、Citation Pack、原文读取
```

| 所有者 | 责任 |
| --- | --- |
| 业务 App | Agent 生命周期、资产发布、权限、安全级、适用范围、业务引用投影 |
| Main API / App BFF | 认证、签发 Scope/Spec、会话与 Turn、SSE 转发、错误映射 |
| Agent Runtime | Execution Spec、Run、任务、上下文、Document Skill、Composer、引用闭包 |
| KC | 入库解析、Agent—Collection Binding、精确 Revision 检索、Evidence、Citation、原文内容 |

## 4. App 接入合同

每个消费者提供 App-owned Agent、`KnowledgeScopeProvider` 与 `ReferencePresenter`。现有
`AgentExecutionSpecV1` 和 KC Agent Binding 是正式合同，不能替换成新的“Knowledge Consumer”身份，
也不重命名 `agent_id`。

知识问答执行规格至少包含：

```json
{
  "schema_version": "1",
  "owner_app_id": "aiops",
  "agent_kind": "KNOWLEDGE_RETRIEVAL",
  "consumer_agent_id": "uuid",
  "consumer_agent_version_id": "uuid",
  "enabled_capabilities": ["document"],
  "knowledge_scope": {"schema_version": "1", "scope_hash": "sha256"}
}
```

Runtime 校验 App、Agent 版本、能力、Binding 和 Scope 一致。AIOps 新建内部知识助手，不能用用户 ID，
不能复用现有诊断 Agent；Runtime 当前只接受 `KNOWLEDGE_RETRIEVAL` 的限制可以保留。

## 5. KnowledgeScopeSnapshotV1

Scope 由 App 服务端计算并冻结进 Run：

```json
{
  "schema_version": "1",
  "owner_app_id": "aiops",
  "domain_id": "uuid",
  "consumer_agent_id": "uuid",
  "consumer_agent_version_id": "uuid",
  "collection_ids": ["uuid"],
  "bundle_revision_ids": ["uuid"],
  "security_level": "INTERNAL",
  "business_context": {"database_type": "ORACLE", "database_major_version": "19"},
  "business_resource_refs": [{"kind": "OPERATIONS_KNOWLEDGE_VERSION", "id": "uuid"}],
  "policy_version": "aiops-knowledge-scope.v1",
  "issued_at": "2026-10-15T00:00:00Z",
  "scope_hash": "sha256:..."
}
```

`scope_hash` 对规范化 JSON 计算。Run 保存完整 Scope 或不可变引用、Revision 数、分片数、策略版本和最终
命中。浏览器、模型和 Prompt 都不能扩大范围；业务上下文的确定性字段不能被模型覆盖。

## 6. 精确 Revision 检索与分片

KC Discovery 必须使用请求级 `bundle_revision_ids`，并同时校验 Domain、Agent—Collection Binding、
Collection 与 Revision 归属。KBot4 当前单次范围上限为 1000，超过时：

1. 按 UUID 字节稳定升序排序并分片；
2. 每片使用同一 Query Plan、模型身份和 Top-K 规则调用 KC；
3. 合并后按统一归一化评分、权威级、适用性和稳定 ID tie-break 排序；
4. 保存分片数量、耗时、降级和命中，再执行 Evidence 覆盖诊断；
5. 禁止 Registry 先按创建时间或更新时间截断，否则旧而相关的正式手册会丢失。

视觉检索使用相同的 Revision Scope 和 1000 条稳定分片规则；每片结果先校验引用闭包，再按视觉分数和稳定
ID 合并。KC 返回任何不属于对应请求分片的候选时，Runtime 必须以
`KNOWLEDGE_SCOPE_CLOSURE_BROKEN` 失败，禁止静默过滤。

全文或向量单通道失败可以降级；所有通道失败才失败。有候选但无正文证据时返回知识缺口，不用业务摘要
冒充 Citation。

## 7. Conversation、SSE 与引用

执行模型为 `Conversation -> Turn -> Run`。Turn 使用幂等键；Run 冻结 Spec、Scope、模型和策略。事件
先持久化再发送，`Last-Event-ID` 只重放、不重复执行。公开事件包含 `run.created`、
`query.rewritten`、`retrieval.completed`、`answer.delta`、`answer.completed`、`run.failed` 和
`run.completed`，不得泄露隐藏推理或未授权候选。

统一回答返回 `answer_markdown`、`references` 和 `coverage`。每个 Reference 保留 Citation、Bundle
Revision、Document Version、页码/章节 Locator、逐字 Quote 与 `business_resource_ref`。Composer 完成前
校验标签存在、引用属于 Scope 且正文实际使用；Reference Presenter 只补充业务显示字段。引用内容读取
再次校验当前用户权限，权限收回后历史答案仍可见，但原文不可继续打开。

## 8. AIOps 接入

AIOps 创建平台管理的内部知识助手：

- `owner_app_id=aiops`；
- `agent_kind=KNOWLEDGE_RETRIEVAL`；
- `enabled_capabilities=[document]`；
- 不绑定 Target，不出现在用户 Agent 选择器中；
- 普通问答使用 `aiops:use`，治理使用 `aiops:knowledge_manage`。

Scope Provider 从 Registry 选择当前 `PUBLISHED` 版本，校验 Domain、安全级、来源权限和适用范围，生成
精确 Revision 集合。页面不提供 Agent、Collection、模型或策略选择。在“运维知识库”增加正式“知识问答”
及会话历史、流式回答、引用角标、原文预览、重试和证据不足状态；“检索测试”只保留为管理诊断工具。

当前 AIOps 代码读取 KC 响应的 `evidence/items`，而 KC 正式合同返回 `citations`；改造时必须删除旧解析，
只消费共享 Citation DTO。手册提炼使用版本化 Schema、Evidence Locator 和确定性校验，命令只能逐字来自
原文，覆盖 `cellcli`、`dcli` 等工具和多行命令，不能依赖命令关键词正则或让模型补参数。

## 9. Knowledge Retrieval App 接入

Knowledge Retrieval App 是基础能力的通用产品消费者，不是公共依赖。它继续拥有 Agent、Domain 业务规则、
上传审核、问文、问数与混合体验，但问文链路切换到本文合同；未来 App 直接接入 Runtime + KC。

## 10. KM 产品线分离与 4.1 删除

KM 以当前 4.0 完整技术栈为终版，使用 `km-v4.0` 标签和 `km4.0` 独立维护分支。KM 后续缺陷只在
`km4.0` 修复并发布 `km-v4.0.x`，不得进入 KBot 4.1 及后续版本。

`kbot4.1` 从同一批准提交的 `v4.0` 创建，并在本次 Knowledge Answering 原子改造中物理删除 KM 服务、
Main API/Slack 入口、Runtime 专属实现、KC/Data Query 特例、平台客户端与合同、`ui/km`、Oracle DDL/
bootstrap、部署配置、测试、OpenAPI 和产品文档。不保留 feature flag、兼容路由、dead code 或可选模块。

因此 KM 对本设计的影响只存在于分线前：`v4.0` 与 `km-v4.0` 必须在同一个通过 KM 全量回归的提交上冻结。
进入 `kbot4.1` 后，不再要求共享 Runtime/KC 对 KM 兼容，改为强制验证 KM 在代码、依赖、Schema、路由、
制品和文档中均不存在。完整规则和删除矩阵见
[KBot 4.0、KM 4.0 与 KBot 4.1 分线规则](release-4.0-4.1-boundary.md)。

## 11. API 与安全

共享内部 Runtime 合同：

```text
POST /internal/v1/agent-runtime/conversations
POST /internal/v1/agent-runtime/conversations/{conversation_id}/turns
GET  /internal/v1/agent-runtime/runs/{run_id}
GET  /internal/v1/agent-runtime/runs/{run_id}/events
POST /internal/v1/agent-runtime/runs/{run_id}:cancel
GET  /internal/v1/knowledge/citations/{citation_id}/content
```

AIOps 统一 Agent 公开业务合同：

```text
GET  /api/v1/apps/aiops/agents
GET  /api/v1/apps/aiops/conversations
POST /api/v1/apps/aiops/conversations
POST /api/v1/apps/aiops/conversations/{conversation_id}/turns
GET  /api/v1/apps/aiops/conversations/{conversation_id}/turns/{turn_id}/events
```

运维知识库路径只提供资产、标签、审核和检索治理，不再提供固定知识问答 Agent 的绑定、
独立 Conversation、Run 或引用读取入口。知识库范围只从 Agent 版本绑定读取。

请求体不出现内部范围字段。BFF 用用户 Token/App API Key 形成可信 Domain 和权限，再以内网 Service
Credential 与 audience-bound AuthContext 调用服务。安全级只能服务端计算；日志不得保存正文、Prompt、凭据、
完整 DSN 或未脱敏 SQL。

## 12. Schema 与文件级改造矩阵

优先复用 Runtime Conversation/Turn/Run/Task/Artifact/Event 与 KC Binding/Revision/Evidence/Citation。只有无法
冻结 Scope 时才修改规范 Oracle DDL、视图、实体和 `schema_manifest.json`；不为 AIOps 复制 Run 表。

| 区域 | 一次性改造内容 |
| --- | --- |
| `packages/platform_core` | Scope、回答/引用/SSE DTO、错误码、Schema 校验 |
| `packages/platform_clients` | Runtime、KC 精确范围与引用客户端；删除旧调用 |
| `services/knowledge_core` | Revision Scope、Binding/归属、稳定评分、引用内容授权 |
| `services/agent_runtime` | 通用 Document/Composer、AppScopedSkill、Scope 冻结、引用闭包、SSE |
| `services/knowledge_retrieval_app` | 作为消费者签发 Scope/Spec，保持自身产品治理 |
| `services/aiops_agent` | 统一 Agent 的知识范围、适用性与 Presenter；修复时间预截断和 Citation DTO |
| `services/main_api` | 统一 Agent BFF、权限、会话/Turn/SSE/预览；删除 KM BFF、登录与 Slack 路由 |
| `tools/dev_console` / `ui` | AIOps 统一 Agent UI；删除 `ui/km` 及共享导航/认证引用 |
| KM 删除面 | 删除 `km_asset_app`、Runtime/KC/Data Query 特例、客户端、合同、配置、部署与文档 |
| Oracle/OpenAPI/tests | canonical DDL/manifest、契约、权限、集成、旧路由与 KM 不存在性检查 |

## 13. 一次性原子改造与切换

以下是一个发布单元内的工作流，不是多个可独立上线的阶段：

1. 已从同一批准提交建立 `kbot4.1` 和 `km4.0`，并在 4.1 完成 KM 全量删除；终版标签后续补在共同基线上。
2. 冻结 Scope、Spec、SSE、引用、错误码和 App 扩展合同。
3. 同步完成 KC、Runtime、Knowledge Retrieval、AIOps、Main API、AIOps 页面、客户端、OpenAPI 和 Oracle
   canonical DDL/manifest。
4. 删除旧问答路由、重复 Composer、旧 KC 响应解析、兼容 DTO 和失效测试；KM 删除已作为 4.1 前置提交
   完成，并由不存在性合同持续锁定。
5. 按受控方式一次性应用正式 Schema 与初始化内部 AIOps Knowledge Assistant；不猜测或回填旧身份。
6. 在候选发布物运行 Knowledge Answering/AIOps 全量门禁和 KM 不存在性检查后同时切换前后端。
7. 失败时整体恢复上一发布物和数据库备份，不让新旧合同并行运行。

## 14. 测试与验收门禁

通用门禁覆盖 Scope 越权、伪造 Agent、精确 Revision、超过 1000 分片、旧而相关手册、稳定排序、零
Citation、多轮改写、标签闭包、命令逐字一致、Turn 幂等、Worker 接管、取消、SSE 重放和历史 Scope。

AIOps 门禁覆盖上传解析、发布、普通/治理权限、正式问答、引用预览、适用范围、诊断知识等级和旧 API
不存在。Knowledge Retrieval App 覆盖问文、问数、混合、会话和治理。

KM 只在分线前的 4.0 终版执行全量功能回归。4.1 不再运行 KM 功能测试，改为检查 workspace、服务拓扑、
路由/OpenAPI、Runtime Registry、KC/Data Query 类型、Oracle manifest、UI 制品、配置、文档和依赖图中
不存在 KM。

验收完成的定义是：AIOps 与 Knowledge Retrieval 只通过统一基础链路获得可追溯回答；KM 已从 KBot 4.1
物理删除并在 `km4.0` 独立维护；未来 App 无需依赖 Knowledge Retrieval App；系统不存在旧路由、双路径
或未受 Scope 约束的检索。截至 2026-10-15，4.1 代码、共享合同、KC、Runtime、Knowledge Retrieval、
AIOps、Main API、静态页面和 OpenAPI 已完成一次性切换；本地 `unit + contract` 门禁以
`1779 passed, 3 skipped` 通过。终版 tag、正式 Schema 应用、部署、重启和在线验收仍是独立发布动作，
不能由本地测试结果替代。
