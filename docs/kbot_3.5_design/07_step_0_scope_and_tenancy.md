# 步骤 0 详细设计：范围、Domain 与双轨基线

## 已冻结的租户与聚合边界

APEX 以 `Domain` 进行数据隔离、权限控制与用户可见范围管理。因此 V2 必须保留 `Domain` 作为 Knowledge Core 的顶层安全边界。其领域层次为：

```text
Domain（APEX 数据与权限边界）
  └─ Collection（Domain 内的知识/检索边界）
       └─ Bundle → Document → Document Version → Parse View → Evidence
```

`Collection` 不等同于 Domain：一个 Domain 可有多个 Collection，以区分资料类型、保留策略、默认安全等级或检索配置；Collection 不能跨 Domain。所有 Discovery、Evidence、Bundle 管理与 Parser 任务均在单一 Domain Scope 内执行，跨 Domain 查询首期一律拒绝。

Collection 由 APEX UI/管理端在指定 Domain 下创建，使用业务可读的 `collection_key`、名称、描述和默认安全等级；它不以来源类型命名或限制来源。`KM_ASSET` 是 Bundle 的 `source_type`，由 KM Portal 的受信任上传 Adapter 在服务端注入。UI、普通用户和通用 Bundle 请求不能自行声明或伪造该类型。

## Collection 生命周期与删除

Collection 支持 `ACTIVE`、`DISABLED`、`DELETING`、`DELETION_FAILED` 状态。UI 可以创建、编辑、停用和请求删除 Collection；不提供“仅删除 Collection 但保留内容”的语义。

删除前 KC 必须检查有效的 `KBOT_KC_COLLECTION_BINDING`。该表记录受控消费者对 Collection 的引用：`binding_id`、`app_id`、`domain_id`、`collection_id`、`consumer_type`（首期为 `AGENT`）、`consumer_id`、状态、可选展示备注和审计字段。Agent V2 配置通过 Binding API 创建/撤销引用，KC 不直接 import 或查询 Agent 的业务表。未来页面、工作流或数据服务可使用相同 Binding 模型。

Agent 与 Collection 是多对多关系，与 V1 `kbot_md_agent_conf` 的 Agent–KB 关系一致，但 V2 不复用该表：一个 Agent 可绑定多个 Collection，一个 Collection 也可绑定多个 Agent。每个 `(consumer_type, consumer_id, collection_id)` 只允许一条当前 Binding。所有 ACTIVE Binding 默认平权：未显式选择 Collection 时，Skill 只能在该 Agent 绑定的全部 Collection 范围内检索，不能在整个 Domain 盲检索；用户显式选择 Collection 时才收窄范围。

Binding 只表达“可检索范围”，不承载 V1 的 `tool_weight`、`reranker_flag`、`search_topk`、`search_score_threshold`、Primary 或优先级等检索策略字段；V2 的两阶段召回与上下文预算由 `KnowledgeRetrievalSkillV2` 和 Collection 检索策略决定。

Collection 停用不撤销 Binding；Skill 在构建检索范围时跳过 `DISABLED` Collection，重新启用后自动恢复。Agent 删除时应先解绑；重试耗尽后的孤儿 Binding 仅可由受限运维接口在验证 Agent 不存在后强制清理，并记录完整审计。

- 存在任一 ACTIVE Binding 时，`DELETE` 返回 `409 COLLECTION_IN_USE`，并返回不超过权限范围的引用摘要；不得删除。
- 无 ACTIVE Binding 时，KC 在事务中锁定 Collection、再次检查 Binding、置为 `DELETING` 并立即撤销其检索可见性和新入库权限。此后禁止新 Binding 和新 Job。
- 物理清除由 `COLLECTION_PURGE` 任务异步完成：取消/终止未完成 Job，删除 Evidence、Discovery、Relation、Parse View、Version、Document、Bundle、Binding 与对象存储内容，最后删除 Collection 行。
- 清除失败时保留 `DELETION_FAILED` 以便重试，不恢复可见性；成功后没有软删除副本。审计日志保存删除请求、执行者、引用检查结果和清除结果，但不保留可检索内容。

```text
DELETE Collection
  → ACTIVE Binding? ──是──► 409 COLLECTION_IN_USE
                   └─否──► lock + recheck → DELETING → COLLECTION_PURGE
                                                      → content/storage deleted → row deleted
```

## Domain 的建模与请求上下文

- `KBOT_KC_COLLECTION` 保存非空 `app_id`、`domain_id`，引用稳定的 APEX Scope；`collection_key` 的唯一约束为 `(app_id, domain_id, collection_key)`。
- 其他 KC 表不重复保存 `app_id`、`domain_id`；均保存或沿父链解析 `collection_id`。Collection 是唯一的 App/Domain Scope 事实来源。
- Bundle、Document、Version、Evidence 不接受客户端传入 Scope；其 Domain/App 通过 Collection 自动关联。APEX 直连页面通过 `KBOT_KC_V_*` 视图或 join Collection 后使用 `app_id + domain_id` 过滤。
- V2 API 以认证后的 Domain Scope 为准。面向用户的调用由认证中间件解析并校验 Domain 成员资格；Portal 等服务调用方提交来源 Domain 时，也必须由服务身份的 Domain allowlist 校验，不能信任裸请求字段。
- `domain_id` 可作为 V2 请求的显式选择参数或路径段，但它只是受校验的 Scope Selector；实际资源必须再次验证属于该 Domain。

## `app_id` 规则

`app_id` 是 APEX 直连数据库页面的必要过滤字段，后台运行时从 `base.toml` 固化读取。它不作为客户端可选择的业务 Scope，且只保存在 Collection：

- 在 `KBOT_KC_COLLECTION` 中为 `NOT NULL`，与 `domain_id` 共同构成 Collection 的唯一键和 APEX Scope 过滤条件。
- 不出现在 V2 Bundle、Discovery、Evidence 或 Parser API 的客户端请求中；KC 在创建 Collection 时从本地配置注入并校验。
- 不用于用户在 V1/V2 间选择路由；用户权限仍以 Domain 为准。APEX 的直连 SQL 必须同时用 `app_id` 与 `domain_id` 过滤。

为避免 APEX 页面散落直接表查询，优先提供 `KBOT_KC_V_*` 只读视图作为稳定渲染契约；视图通过 Collection join 暴露 `app_id`、`domain_id` 与页面所需字段。即使暂时直接查表，也必须 join Collection 并使用两个 Scope 条件。服务配置中的 app 标识不可由调用方覆盖。

## V1/V2 并行基线

V1 继续使用既有 `App → Domain → KB → File → TxtChunk`；V2 使用 `Domain → Collection → Bundle → Evidence`。两条链路在过渡期共享 Oracle Schema 和 Domain 主数据，但不共享知识事实表、API DTO、Skill 或请求内回退。

V2 路由以 `domain_id` 为最小切流单元，并可在该 Domain 内进一步细化到 Collection、Agent 或入口。所有路由决策必须显式配置和审计；V2 失败只报告 V2 错误或由运维显式切换，不能自动查询 V1 以拼接结果。KC 不跨 Domain 检索；未来跨 Domain 场景由上层 Agent 组合多个 Domain 作用域内的结果实现。

## 待本步骤确认的剩余项目

1. 首批接入 Domain 的清单，以及每个 Domain 的首个 `collection_key`。
2. 用户认证与 Portal 服务身份中可用的 Domain claim/allowlist 来源。
3. Domain 默认安全等级、允许的 Collection 数量，以及跨 Domain 管理员是否只允许管理而禁止检索。
4. Agent V2 配置保存时创建/撤销 Collection Binding 的调用方与审计主体。
5. 每个首批 Domain 的 V1/V2 切流入口、灰度比例和验收样本归属。
