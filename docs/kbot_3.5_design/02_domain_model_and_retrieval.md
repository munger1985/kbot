# 3.5 领域模型与检索设计

## 核心层次

```text
Collection
  └─ Bundle（一个来源业务对象）
       └─ Bundle Revision（不可变来源快照）
            ├─ Revision Document Member（该快照的附件清单）
            │    └─ Document（主信息、附件等逻辑文件）
            │         └─ Document Version（不可变内容）
            │              └─ Parse View（解析表示）
            │                   └─ Evidence（可检索、可引用片段）
            ├─ Discovery Object（Bundle/Document 检索画像）
            └─ Relation（有依据的关系）
```

| 表 | 事实职责 | 关键规则 |
| --- | --- | --- |
| `KBOT_KC_COLLECTION` | App/Domain 范围内知识边界、默认安全等级、元数据 | `(app_id, domain_id, collection_key)` 唯一；不能跨 Domain |
| `KBOT_KC_BUNDLE` | 稳定来源对象、当前可用 Revision 与整体状态 | 来源键在 Collection 内唯一 |
| `KBOT_KC_BUNDLE_REVISION` | 不可变来源 Manifest、Facet、附件清单指纹与处理状态 | 同一 Bundle 内来源修订唯一 |
| `KBOT_KC_BUNDLE_REVISION_DOCUMENT` | Revision 内文件声明、角色、顺序、必需性与接收/解析状态 | 附件漏传和补传均可追溯 |
| `KBOT_KC_DOCUMENT` | Bundle 内稳定逻辑文件身份 | 外部文档 ID 在 Bundle 内唯一 |
| `KBOT_KC_DOCUMENT_VERSION` | 不可变内容 hash、存储 URI、版本与内容状态 | 当前可检索版本由 Bundle Revision Member 指定 |
| `KBOT_KC_PARSE_VIEW` | TEXT/VISUAL/HYBRID/SPREADSHEET 表示和配置快照 | 同一类型/范围仅一个 Active；重解析成功后替换旧产物 |
| `KBOT_KC_EVIDENCE` | 片段、向量、页码、坐标、层级和质量 | 归属 Version 与 View；Revision 通过 Member 在查询时关联 |
| `KBOT_KC_DISCOVERY_OBJECT` | Bundle/Document 的全文、向量、Facet 投影 | 仅当前可见版本索引 |
| `KBOT_KC_RELATION` | 可回溯对象关系 | 保存类型、来源、置信度、Evidence 依据 |
| `KBOT_KC_INGESTION_JOB` | PARSE/PROFILE/INDEX/RELATE 执行状态 | 幂等键、租约、重试上限 |
| `KBOT_KC_INGESTION_RECEIPT` | HTTP 接收幂等、暂存/发布补偿与受理结果 | 不属于知识事实，不参与检索 |
| `KBOT_KC_COLLECTION_BINDING` | Agent 等消费者对 Collection 的多对多受控引用 | ACTIVE Binding 阻止删除；每个 Agent/Collection 一条当前 Binding |

## 不可变版本与可重建投影

Domain 是 APEX 的数据与权限边界，Collection 是其内部的知识边界。V2 API 从认证上下文获得并验证 Domain Scope；`app_id` 从 `base.toml` 固化写入 Collection，用于 APEX 数据库直连页面的物理过滤。其他 KC 表只通过 `collection_id` 关联 Scope，不作为客户端请求参数或用户权限 Scope。KC 不跨 Domain 检索，未来跨 Domain 仅由上层 Agent 组合实现。表的 Scope 写入与一致性策略见[步骤 1 表基础设计](09_step_1_table_foundations.md)。

内容、解析配置或来源修订变化时创建 Document Version，不能覆盖旧内容。新版本的 Evidence 与 Discovery 就绪后，才切换 current；历史版本可审计但首期不提供历史检索 UI。

Evidence 是精确回答和引用的事实投影，`retrieval_text` 只包含 Version 稳定的 MIME、类型、标题路径、结构标签和正文；Bundle 标题/Facet、Document Member 角色与声明名称属于 Revision 级上下文，写入 Discovery Object 并在排序、引用组装时动态附加。Discovery Object 是“找文件/找业务对象”的可重建投影，不能取代事实表。对两者建立 Oracle Text、Vector 和权限/状态/高频 Facet 索引。

Relation 首期只写显式链接、相同编号、引用/派生等可扩展的确定性关系；Manifest 附件归属已由 Revision Member 表达，Evidence 章节父子已由 `parent_evidence_id` 表达，不重复写入 Relation。模型推理关系必须记录模型/规则版本及可回溯依据，允许失效重建。

Collection 删除不是软删除：若存在 ACTIVE Binding，返回 `409 COLLECTION_IN_USE`；否则先置为 `DELETING` 并撤销可见性，再以 `COLLECTION_PURGE` 异步删除所有下游行和对象存储内容，最后删除 Collection。Binding 通过 KC API 管理，避免 KC 直接访问 Agent 表。

首期 Binding 的消费者为 `AGENT`。一个 Agent 可绑定多个 Collection，一个 Collection 可被多个 Agent 使用；所有 ACTIVE Binding 默认平权。Binding 不复制 V1 的 Chunk Top-K、权重和 rerank 参数；V2 检索策略属于 Collection 与 `KnowledgeRetrievalSkillV2`。

## 查询契约

- `POST /api/v2/knowledge/discovery`：自然语言、Collection、Facet、权限、`top_k` → Bundle/Document 卡片与命中理由。
- `POST /api/v2/knowledge/evidence`：问题、限定 `bundle_ids`/`document_ids`、权限、预算 → Evidence 与稳定定位。

Evidence 输出必须能回溯 `bundle_id → document_id → document_version_id → parse_view_id → section/page/bbox`。

## 问文检索质量链路

问文 Skill 采用“先 Discovery、后 Evidence”的两阶段流程，不能再以全库 Chunk Top-K 作为唯一检索策略：Discovery 先结合标题、Facet、Bundle/Document 画像与语义召回缩小候选；Evidence 再在该范围内进行混合检索、主视图去重、章节/页邻接扩展和有依据的 Relation 扩展。Skill 将结果组织为 Citation Pack，并校验回答所需主题、附件和页段覆盖情况。

这使 Bundle 级相关性、附件多样性、版本一致性和可定位引用成为检索目标，而不是仅提高单个文本片段的相似度。评测必须分别记录 Discovery Recall@K、Evidence Recall@K、引用定位准确率、跨附件覆盖率和最终答案的证据支撑率。

### 多 Collection 平权召回

当用户未显式选择 Collection 时，Skill 从 Agent 的 ACTIVE Binding 取得允许范围，并将全部 `collection_ids` 传给 Discovery。KC 不将所有 Evidence 混为一个大索引直接取 Top-K，而是先在每个 Collection 内独立执行关键词/向量混合召回，再做跨 Collection 融合：

1. 每个 Collection 使用相同的候选预算和最低质量阈值；无合格候选的 Collection 不强行占位。
2. Collection 内先融合关键词与向量排序；跨 Collection 使用秩融合或经校准的分数，避免不同 Collection 的向量分布、文档数量或术语密度不可比。
3. 在达到最低相关性要求的前提下，保留 Collection 多样性，再按 Bundle 相关性、精确 Facet 命中和冗余惩罚进行全局排序；不对 Collection 预设业务权重。
4. Evidence 只在 Discovery 选定的 Bundle/Document 范围内检索。Citation Pack 必须携带 `collection_id`，使回答、页面和审计可追溯来源。

用户在 UI 明确选择一个或多个 Collection 时，Binding 只用于权限校验，Discovery 严格以所选范围检索。
