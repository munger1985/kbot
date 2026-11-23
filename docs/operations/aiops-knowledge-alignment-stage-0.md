# AIOps 运维知识库对齐：阶段 0 基线与合同冻结

记录日期：2026-11-20。范围见[实施计划](aiops-knowledge-alignment-implementation-plan.md)。本文记录源码基线、目标合同和待审事项，不代表目标合同已上线。

## G0.1 基线与工作区

| 平台 | 工作区 | 基线 | 跟踪状态 |
|---|---|---|---|
| KBot4 | `/home/chris/kbot4` | `kbot4.1.1 @ 13b91596` | 相对本地记录的 `origin/kbot4.1.1` 领先 3、落后 3 |
| Ammolite | `/home/chris/Ammolite`（`/home/chris/ammolite_cube` 指向同一工作区） | `dev @ 73ba9b42` | 相对本地记录的 `origin/dev` 领先 2 |

两个工作区均有未提交修改。本轮只新增独立文档和离线合同用例，未切分支、拉取、清理或修改已有文件。后续实施重点检查重叠：两仓客户端、OpenAPI、启动装配、UoW、实体注册及跨仓检查；KBot4 的 Oracle 工件与 Ammolite 的 `ops.py`、权限注册、初始化器、Agent 应用层和仓储也已有工作区改动。本表是本地源码快照，不是服务器部署状态。

## G0.2 调用方与已知差异

| 链路 | KBot4 | Ammolite | 双仓下游 / 必改点 |
|---|---|---|---|
| 库卡片、编辑、删除 | `ui/aiops/js/aiops-knowledge-bases.js` → `main_api/api/aiops_app.py` | `AIOpsKnowledgeBasesView.vue` → `main_api/api/ops.py` | `platform_clients/aiops.py` → AIOps `api/operations_knowledge.py` → 同名应用服务、`repositories/knowledge.py`。两端编辑读旧库 `/models`，保存走库 CRUD；保留案例独立 `/case-collection/models` |
| 上传和审批 | AIOps 运维资料页面/API | `AIOpsOperationsKnowledgeView.vue` / Portal 业务 API | `_register_manual_documents` 在对象写入前、登记事务内复查旧角色；`approve_manual` 进入 `TAGGING`，`analyze_tags` 运行时再读库模型；应改为按阶段校验与冻结执行规格 |
| 模型引用保护 | AIOps Agent 模型引用入口 | AIOps Agent 模型引用入口 | `repositories/agent.py:knowledge_model_references` 分别读取手册库与案例配置；停用虚假库级角色时不能误删案例保护或漏掉冻结任务 |
| 预览与诊断 | `POST /api/v1/apps/aiops/operations-knowledge/search-preview`；`aiops-workspaces.js` 消费诊断引用 | `POST /api/v1/apps/aiops/domains/{domain_code}/ops/operations-knowledge/search-preview`；`AIOpsDiagnosisView.vue` 消费诊断引用 | 预览、`turn_capabilities.py`、`workers/diagnosis_handlers.py` 与 `knowledge_child.py` 下探同一搜索服务。页面源码未发现直接调用 `search-preview`，不能据此判断有无仓外消费者，也不因此新建页面 |
| Agent 绑定与删除 | 版本绑定由 `application/agents.py` 写入；Oracle 规范 DDL 已有资料和版本绑定到库的外键 | 同样按 Agent 版本写入；规范 ORM 的资料/版本绑定库字段未声明这两个外键 | 目前删除先锁库，引用汇总只连当前 Agent 指针；上传登记已持库锁，但绑定校验尚未用同一锁协议。Ammolite 实库约束未查证，不能当作已无外键 |

补充核对：

- 两仓建库与编辑转发使用 `exclude_none=True`；阶段 1 必须让显式 `null` 和字段省略在公开 API → 客户端 → 内部服务全链保持区别，特别是清除标签设置。
- 两仓 `knowledge_search.py` 从 `evidence` 顺序做局部排名，Repository 又只取 `READY` 索引候选；必须先检查资格，才能把零结果解释成没有资格资料或没有匹配。
- 两仓公共删除已使用 `If-Match: "rv-N"`，内部客户端传预计版本；缺失头的 HTTP 行为当前并不完全一致。阶段 3 按下文统一，不改变路由或授权模式。
- 无直接调用方只能作为本地证据。停用旧手册库模型入口前，需确认正式消费者；无论如何不增加旧角色的第二写入口，不退回兼容双轨。

## G0.3 目标合同（供阶段 1–3 消费）

合同用例在 `tests/contract/fixtures/aiops_knowledge_alignment_v1.json`，离线测试只校验合同行为及样例的自洽性，不调用运行 API。示例是相关字段切片：现有业务字段如资料标题、用途和业务排名继续按现有合同承载。字段名和结论码以下为准；任何变动先更新双仓示例和消费者约定。

### A. 有效能力与索引

- 建库和库编辑的配置字段仅 `display_name`、`description`、`tagging_model_id`（UUID 或 null）。编辑请求沿用 `expected_row_version`；模型字段省略表示保留，显式 null 表示清除，传 UUID 表示替换。请求不接受旧 `models` 字典、客户端指定的 `capabilities` 或索引身份；`parser_llm` 仅是现有运行链需核实的真实标签绑定角色，不是新的对外可写字段。
- 详情 `capabilities` 分别给出 `native_parse`、`ocr`、`visual_enrichment`、`lexical` 及实际配置的 `semantic_directions`。每项状态 `NOT_ENABLED / PENDING_VERIFICATION / AVAILABLE / DEGRADED`；`AVAILABLE` 来源须是实接有效方案及已验证的执行事实。语义列表不伪造未配置方向，能力状态不代表该库存在可检索资料。
- `index.state`：`EMPTY / NOT_READY / PARTIAL / READY / ERROR`。空库无资料不告警；库内有资料而无就绪覆盖不能标成空库或就绪。`tagging.state`：`NOT_CONFIGURED / PENDING_VERIFICATION / AVAILABLE / DEGRADED`，单独记录标签模型配置，不能反推解析/索引健康。
- 缺标签能力时允许上传并提示；进入标签处理前失败则保留待审，不登记标签任务。拒绝审批和已发布查询不受影响。执行规格冻结模型、Prompt、词表及来源；模型配置变更不重写已冻结任务。

### B. 主命中、关联与覆盖

- 公共响应保留业务 `results`、结果的业务排名和 `citation_pack`，另有 `retrieval_conclusion`（机器码见下）及 `coverage`。覆盖内独立表达 `scope`（`COMPLETE / PARTIAL / NOT_VERIFIED`）、`ranking`（`GLOBAL / LIMITED / NOT_APPLICABLE`）、`gaps` 和 `channel_failures`。正常且未启用的通道不算失败；关联缺口不自动改写召回覆盖或名次。覆盖摘要不披露无权资料或内部身份。
- `EVIDENCE_FOUND` 有可交付主证据，可伴部分覆盖。只有资格检查完成且确定无合格资料才使用 `NO_ELIGIBLE_RESOURCES`；只有有效范围充分覆盖、有效通道正常且零匹配才用 `NO_MATCH`；不满足前两个空结论又没有可交付主证据时为 `NOT_VERIFIED`。用途约束导致未交付应保留业务原因，不能伪称引擎没有匹配。鉴权拒绝仍沿用既有权限边界，不降级成零命中。
- 常见覆盖缺口：`CANDIDATE_BUDGET_EXCEEDED`、`INDEX_NOT_READY`、`AUTHORIZATION_CHANGED`、`GENERATION_CHANGED`、`RETRIEVAL_CHANNEL_FAILED`。真正影响有效范围时用部分覆盖；独立分片不具备全局名次，用 `LIMITED`，即便该次候选范围完整。只有同一次逻辑查询完成共同通道归并与融合且对应覆盖充分，才用 `GLOBAL`。
- 每个 `results` 项的 `evidence_groups` 含 `primary`（`evidence_key`、`document_version_id`、`rank`、`quote`、`locator`）及 `related`、`relation_gaps`。主命中以引擎身份和名次为准；关联按锚点分组，不设置主名次。对关联给出可读性及经授权的引用位置，不能用不可读内容补正文；无权内容不提供身份、原文或定位。产品不输出内部 Scope、Ticket、游标。父子执行应携带同一有界覆盖判断，不重搜排名来替换精确引用。

### C. 删除保护与并发

- 库详情和列表的 `usage` 使用 `assets`、`current_agent_versions`、`historical_agent_versions`，各计数互不重叠；`blockers` 提供 `ASSET_RECORDS / CURRENT_AGENT_VERSION / HISTORICAL_AGENT_VERSION` 和实际确认的其他保护原因。仍保留的资料记录不按发布状态排除。未确认的其他关系不能伪造计数。
- 公共 DELETE：缺少前置版本为 `428 PRECONDITION_REQUIRED`，格式无效为 `400`（沿用正式 ETag 格式），版本过期为 `412 STATE_VERSION_CONFLICT`，确有依赖为 `409 AIOPS_KNOWLEDGE_BASE_IN_USE` 并提供当前阻止原因，删除成功为无正文 `204`。核权、锁目标库后先检查版本再查依赖，越权不披露依赖；数据库只映射已知保护约束异常。
- 卡片用最近一次依赖汇总作提示，不能作为删除依据；禁止自动解绑、级联删除或把历史当作可清理对象。上传登记、Agent 版本写入、删除必须用一致事务门闩；保持停用可管理但不可新使用。

## G0.4 存量盘点与评审

| 授权后的只读项 | 必须形成的决定 |
|---|---|
| 手册库配置的字段/数量、实际方案与模型装配、模型引用来源 | 区分真实标签绑定、无效旧选择、真实运行依赖；标明是否可保留，不能把旧选择直接映射成已启用能力 |
| 已发布资料的精确文档/解析运行/表示身份、活动代次、一致性与覆盖 | 旧代次能否保持可用，需要另行验证或切换的范围；不自动回填、重建或回退 |
| 待处理标签任务及冻结规格完整性 | 不能证明执行身份的任务如何妥善处置，由上线评审确认；不静默以当前配置执行 |
| 资料和 Agent 当前/历史绑定、约束、孤立记录 | 分别确认 Oracle/PostgreSQL 非级联保护及需变更的最小范围；发现孤立记录先报告，不做自动删除 |
| 仓外或发布环境的旧 API 调用者 | 评审同切窗口、切换和回滚条件；不靠兼容双轨掩盖消费端差异 |

阶段 0 不连接数据库、模型服务或线上 API。存量数据与真实约束仍为待核查项；本轮只交付源码证据、合同和离线样例。

## G0.5 本轮验证记录

- KBot4：在仓库根执行 `conda run -n kbot4 python tests/contract/test_aiops_knowledge_alignment_stage0.py -v`，5 项通过。
- Ammolite：在仓库根执行 `conda run -n ammolite python tests/contract/test_aiops_knowledge_alignment_stage0.py -v`，5 项通过。
- 检查覆盖编辑省略与清空、拒绝旧模型角色、空库与能力异常区分、主命中与关联的边界、局部排序和部分覆盖、历史绑定及删除错误优先级。未运行服务合同、实库并发或真实资料检索质量验收。
