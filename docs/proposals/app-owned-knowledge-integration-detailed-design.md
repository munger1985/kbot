# App 独占知识能力接入与持久化详细设计

日期：2026-11-06。状态：正式 App／Runtime／DTO／双前端／AIOps 代码接入及旧路径退出已完成；真实场景和部署未执行。K0/K1 G-A 已通过；K2 已编码，知识检索 App 的 Oracle/PostgreSQL 定向受控验收各 18 项通过；本轮禁止修改 AIOps 数据库表，其 Oracle 验收保留，整体 K2 G-B 不标全量完成。适用：KBot4 与 Ammolite；两仓保持本文一致。

本文承接[基础能力设计](../architecture/knowledge-answering-capability.md)与[算法详细设计](app-owned-knowledge-algorithm-design.md)，补齐进程、传输、物理存储和 Runtime 交付合同。[实施计划](../operations/app-owned-knowledge-implementation-plan.md)负责代码落点、依赖和门禁。K2 数据库范围以[K2 实施记录](../operations/app-owned-knowledge-k2-progress.md)为准，组件、合同及产品代码接入见[基础编码记录](../operations/app-owned-knowledge-foundation-progress.md)；目标合同不表示线上 OpenAPI 或业务数据库已切换。

## 1. 运行单元与固定绑定

| 单元 | 部署/组合根 | 数据访问 |
|---|---|---|
| 纯解析执行器 | 复用 knowledge_core 计算进程位置，删除被替换的公共知识存储职责 | 仅计算任务元数据及按 App 隔离的临时制品，不写 App 业务表 |
| Knowledge Retrieval 知识模块 | knowledge_retrieval_app 的 application/adapters/workers | 仅本 App 表、目录、索引及受控模型 |
| AIOps 知识模块 | aiops_agent 的 application/adapters/workers | 仅 AIOps 表、目录、索引及受控模型 |
| Runtime Document/Composer | agent_runtime 的 Skill 与统一交付管线 | Runtime 运行事实；通过本 App 内容端口取证，不访问 App SQL |

App 组合根固定 BackendBinding、模型访问端口、SourceReader、ProjectionBuilder 和 RetrievalEngine。Runtime 组合根登记 AppRetrievalPort/AppContentPort 的可信目标服务与允许能力。只能从已验证 Execution Spec 的 owner App 选择登记项，不从请求接收 URL、Python 路径、表名、连接串或任意目标 App 列表。

这是可信 App 能力委托，不是跨 App 路由搜索：一次请求只有一个 owner App、一个授权范围，Runtime 不探测其他 App、不合并它们的候选、不在自身进程构造 App Repository。App 端点即使收到另一 App 的有效 JWT/同名证据 ID 也必须拒绝。

## 2. App 内部端口与认证

### 2.1 固定端点

两个 App 分别在自己的内部前缀实现相同相对资源，不新增公共搜索服务：

| App | 固定前缀 |
|---|---|
| knowledge_retrieval | /internal/v1/knowledge-retrieval/knowledge |
| aiops | /internal/v1/aiops/knowledge |

| 方法/相对路径 | 用途 |
|---|---|
| POST /scopes | 本 App 根据可信身份、Agent/业务资源冻结授权范围，返回范围引用/摘要，不接受任意 SQL/范围表达式 |
| POST /sources:read | 为受委托的解析任务读取精确文件版本；不接受路径/URL，不提供目录枚举 |
| POST /retrievals | 在本 App 执行 QueryPlan 与授权检索，返回 EvidenceSet 与 retrieval_ref |
| GET /retrievals/{retrieval_ref}/pages | 受控主命中/关联清单续取，不自动执行新搜索 |
| POST /evidence:read | 按精确 EvidenceKey 批量读取实际文本/图片，返回有界载荷流 |
| POST /evidence:revalidate | 核验当前发布、撤回、内容允许视图与权限，不检查数据库 Schema |
| POST /evidence-pins | 为确切 Run/EvidenceKey 建立有界审计保留引用，不授予读取权 |
| DELETE /evidence-pins/{pin_ref} | 解除该 Run 的保留引用，实际回收由 App 保留策略决定 |

目标公开 API 继续归各 App BFF：知识检索使用其 /api/v1/apps/knowledge-retrieval 下资源，AIOps 使用统一 Agent 会话/Turn 及既有知识治理入口。精确公开路由在实施同批更新 DTO/OpenAPI，不新建独立 knowledge-conversations、固定 Knowledge Assistant 或公共 /knowledge/search。直接检索使用 App 搜索权限，不强制经过 Runtime。

### 2.2 请求与委托

RetrievalRequest 必含 schema_version、request_id、scope_ref/hash、query、profile_ref、预算上限及 Runtime 场景的 run_ref/spec_hash。query 含文本/已授权图片句柄、输出类型、模式、用户数量/排序；不能携带待访问的 App 列表、数据库标识或任意文件路径。

scope_ref 是 App 预先冻结并持久化的范围引用。App 本地用例可直接使用类型化 AuthorizedScope，跨进程不可直接序列化 Python 对象或数据库过滤表达式。巨大允许版本集合存 App 范围明细/不可变清单，不把全部 ID 强塞进 JWT。

内部请求同时使用服务凭据和 audience-bound 短期 AuthContext。委托声明绑定 owner App、Tenant/Domain、用户/服务主体、Agent 版本、run_ref、scope_hash、允许操作、过期时间；scope_hash 不是签名。App 验证原始主体当前权限与服务委托的交集，不因 Runtime 是可信服务就使用管理员权限。

BFF 在创建正式 Run 前向所属 App 冻结范围与可信执行绑定。App 的 ScopeProvider 返回范围引用、摘要、策略与授权代次；Runtime 原样冻结。没有可验证的绑定、范围、模型能力或授权时拒绝，不能自动选其他资料库/模型。直接搜索也经本 App ScopeProvider，但不要求 Agent。

### 2.3 一致性与重试

同一 request_id、规范请求摘要和 scope/snapshot 组成幂等检索身份；同键异请求冲突。重试可复用仍有效的快照/回执，不复用跨 App 缓存。不承诺非确定性模型输出逐字相同，实际计划与查询表示指纹需冻结。

AppRetrievalPort 返回越界对象时 Runtime 以 SCOPE_CLOSURE_BROKEN 终止该委托，不静默丢弃；这与 App 内正常权限变化导致候选失效分开。返回前/读取时权限变化可减少有效集合，须记录 SCOPE_CHANGED 及缺口，不扩大到其他版本。

### 2.4 正式问答与 AIOps 调查的承接方式

知识检索 App 的正式问答复用既有 Runtime Conversation/Turn/Run API。AIOps 不创建用户可见的另一套知识会话：仅在当前统一 Agent 的业务 Turn 需要正式知识问答时，服务端通过既有 POST /internal/v1/agent-runtime/runs 创建受控内部执行，使用已授权 Agent 版本的 Spec/Scope，并绑定 parent_run_ref、parent_turn_ref 和幂等键。运行模板固定为批准的 Document/Composer 能力，不接受浏览器任意安装 Plan。

内部执行复用 Runtime 原表与 get run/result/events/cancel 合同；父业务 Turn 保存确切子执行引用并转发已持久化事件/结果。重复受理复用同一子执行，取消、失效与权限变化向下传播；不复制会话历史/Run 表、不创建固定助手、不让用户同时维护两条问答记录，也不把 AIOps 调查迁入知识检索 Root Planner。所需父引用/能力字段随同批 DTO 和契约测试冻结，不能假称现有代码已支持。

AIOps 调查中的 ops.knowledge.search 只需检索时，可在本 App 调用同一 RetrievalEngine，形成精确 KNOWLEDGE_CITATION，不额外启动 Composer 或模型。既有现场诊断回答、报告和案例提炼不属于本次统一知识 Composer 的替换面；如果这些消费者向自己的模型发送知识图片/正文，也必须遵守实际载荷、权限、预算与交付账本语义，不能把预览当作已读图。这不授权重新实现一套通用知识问答 Composer。

## 3. 远程解析任务与事件交付

### 3.1 计算 API 与 App 接收

纯解析执行器提供 /internal/v1/knowledge/parse-jobs 下的计算资源：

- POST 根资源：接受 parse_run_ref、attempt/fence、源内容哈希、受控 source_handle、冻结 ParseProfile/模型引用、App 隔离命名空间与资源预算，返回 202 计算回执。
- GET /{job_ref}/events：按 after_sequence 与 limit 拉取有界事件页，返回 next_cursor、完成状态与事件摘要。
- POST /{job_ref}/acknowledgements：App 持久接收成功后推进已确认 sequence/batch 摘要；仅同一 App 接收者可确认。
- POST /{job_ref}:cancel：撤销计算租约，拒绝旧 fence 的继续产出/确认。
- GET /{job_ref}/artifacts/{artifact_ref}：读取本次已授权、有期限、有摘要的字节流，不暴露实际路径或存储 URI。

不传 App 回调 URL、Python 回调或 DB Session。source_handle 仅可由可信 AppSourcePort 在原 App 内部端点解析，绑定文件版本、计算任务、允许操作及期限；执行器不能因此列举 App 目录。服务续期时必须重新校验任务与当前读取权，不接受任意 URL，防止 SSRF。

App 是业务 ingestion_job 的权威所有者。执行器只保存必要计算任务/已产出事件与制品保留状态，按 App 命名空间隔离，不成为公共解析结果库；完成持久接收后按确认与保留策略回收。计算 API 不等于新的必选消息中间件，复用现有持久任务/轮询机制。

计算任务的 App 命名空间与 AppSourcePort 由已认证委托和可信绑定确定；请求正文即使带相同字符串也不能选择/覆盖命名空间。source/artifact 读取、事件拉取及 ACK 都核验同一任务与 owner，防止伪造另一个 App 的 job_ref。

### 3.2 事件信封与可靠性

ParseEventEnvelope 的必需字段为 schema_version、job_ref、parse_run_ref、attempt、fence、sequence、batch_key、event_type、payload_hash、dependency_keys、payload/artifact_handles；与算法设计的事件族一致。原文/图片大载荷使用流式制品而非无界 JSON。

1. App 持久化任务和输入版本后提交计算；外部调用不持有业务事务。
2. 消费者先校验任务身份、租约、序号、摘要和预算；需要的二进制先复制到本 App 暂存对象区并校验哈希。
3. 通过 App Repository/UoW 幂等登记批次、对象引用和恢复点；事务成功后确认消费。对象发布使用 App 既有暂存/提交协议或 outbox，未提交孤儿制品可回收，不先 ACK 后落库。
4. NODE/ASSET/REFERENCE 批次可暂存前向引用；STRUCTURE_FINALIZED 与 RUN_COMPLETED 的最终 Manifest 校验全部依赖、对象计数、有序摘要和质量报告，未完成不能发布。
5. 同 batch_key/摘要重放无副作用；同键不同内容拒绝。并发消费使用 App 任务行锁/栅栏串行化，不能依赖业务 UNIQUE/CHECK。
6. 重新执行非确定性模型产生不同产物必须使用新 ParseRun。过期制品返回 ARTIFACT_EXPIRED，不能宣称可从已消失产物恢复。
7. 取消、崩溃、网络中断、迟到完成事件不能发布失效代次；恢复先重放确认范围之外的已保存事件，预算耗尽明确失败/部分成功。

流式传输不证明转换器低内存；执行器对不支持分段的适配器设进程级内存/时间边界。临时路径、任务正文和凭据不进入普通日志。

## 4. App 自有物理模型

### 4.1 命名空间与 owner

下表是待实施的固定物理映射，不按 Tenant、Domain、用户或安装动态建表：

| App | Oracle 前缀 | PostgreSQL 注册前缀 |
|---|---|---|
| knowledge_retrieval | KBOT_KR_ | knowledge_retrieval_ |
| aiops | KBOT_OPS_ | aiops_ |

每行逻辑表以对应前缀拼接后成为两套不同物理表。例如 knowledge_node 分别为 KBOT_KR_KNOWLEDGE_NODE、KBOT_OPS_KNOWLEDGE_NODE，以及 knowledge_retrieval_knowledge_node、aiops_knowledge_node。表名由代码静态登记，非请求拼接；PG 标识符不超过 63 字节。已有业务表如 aiops_knowledge_version 的资产版本含义保持，不与下列文档版本混淆。

### 4.2 表集合与关键字段

所有 App 表使用本 App UUIDv7 主键，带本平台适用的 Tenant/Domain、审计时间和行版本；算法 local_key 不作为全局主键。

| 逻辑表 | 关键字段/关系与用途 |
|---|---|
| knowledge_document | 稳定文档身份、业务资产引用、标题、状态 |
| knowledge_document_version | document_id、源哈希/MIME/字节数、App 对象引用、不可变业务版本 |
| knowledge_parse_run | document_version_id、输入/配置指纹、模型快照、attempt/fence、质量与状态 |
| knowledge_parse_batch | parse_run_id、batch_key/sequence/hash、依赖、接收恢复点 |
| knowledge_node | parse_run_id、local_key、类型、父对象、阅读顺序、原文/允许视图、locator/provenance |
| knowledge_visual_asset | parse_run_id、逻辑节点、原图/裁剪句柄、hash、尺寸/变换、来源 |
| knowledge_reference_occurrence | 源节点/跨度、原始引用、规范编号/作用域、歧义状态 |
| knowledge_relation | 源节点、目标节点或资产、关系类型、精确版本/解析代次、引用出现及解析依据 |
| knowledge_search_unit | 投影代次、unit_key、类型、允许检索文本/上下文、内容/配置指纹 |
| knowledge_unit_source | 单元到一个或多个原始节点/跨度/图片区域的映射，不丢跨页来源 |
| knowledge_representation | unit_id、space_id、形状/mask、输入摘要、向量或张量制品引用 |
| knowledge_index_generation | Manifest、各通道索引引用/状态、依赖代次、活动指针及并发行版本 |
| knowledge_ingestion_job | 持久任务、租约、重试/恢复点、输入与错误分类 |
| knowledge_outbox | 事务内投影/索引/回收事件、幂等键、投递/确认状态 |
| knowledge_retrieval_receipt | scope/snapshot/plan 指纹、候选与证据身份清单、覆盖、分页状态及期限 |
| knowledge_evidence_pin | App 内确切证据/解析代次到 owner Run 的保留引用、期限/释放状态 |

Scope 巨大时使用本 App 的 knowledge_scope_snapshot 与 knowledge_scope_member，前者保存不可变策略/摘要，后者保存精确允许文档版本；不在运行请求中动态建表。若复用 App 已有等价不可变范围表，须在 K0 的 D0 清单证明并固定字段映射，不能退回公共 KC 范围表。

节点关系源/目标只指本 App。结构节点/图题/图片使用统一内容对象键，HAS_ASSET 指向资产键；目标节点与资产互斥及关系类型组合由领域合同校验。跨文档引用只可指本 App 的精确版本，未解析或外部目标保留 occurrence，不造空 FK 或跨 App 引用。

本 App 内容对象外键不跨到其他 App。pin/回执中的 Runtime Run 引用是经可信委托核验的外部审计标识，不添加跨服务业务表 FK、级联删除或直接 Runtime Repository 查询；保留/释放通过版本化端口确认。

Representation 元数据与索引表示分开：单向量的数据库列/独立索引按 space 适配，token 矩阵使用有界张量制品/多向量引擎，不把 token 矩阵压成均值或伪装数据库单向量。各空间、外部引擎集合及对象根按 App 隔离。

### 4.3 事务、索引与保留

业务表只用 PK/FK/NOT NULL 保障结构；本设计不新增业务 CHECK/UNIQUE 或唯一索引。幂等键、活动版本、批次键及业务唯一性通过应用校验、固定根行锁、行版本与事务保证，测试必须覆盖竞态。

常规非唯一索引至少覆盖 owner 范围/状态、文档版本/解析代次、批次顺序、引用源与目标、单元来源、space、索引活动代次、任务领取/租约、outbox 待处理和 pin 回收。全文/ANN/多向量索引登记在 IndexManifest，不承诺各数据库天然支持所有多向量能力。

App UoW 提交规范结果及 outbox，索引器幂等写独立 generation 后切换活动指针。发布资格与索引就绪是不同门禁；授权读取不能只信任索引 payload。引用 pin 防止合法历史证据被过早回收，但不阻止依法删除，也不赋予越权读取。依法删除后返回明确缺口，不把旧 ID 映射到新资料。

## 5. 查询执行、排序与覆盖合同

### 5.1 查询模式

QueryInput 固定区分 RELEVANCE、FILTERED_ORDER、EXHAUSTIVE。RELEVANCE 是候选内相关性 Top-K；FILTERED_ORDER 在可执行的结构化资格条件内按指定日期/属性与稳定 ID 获取范围内前 N；EXHAUSTIVE 是可穷尽谓词的授权范围分页枚举。

FILTERED_ORDER/EXHAUSTIVE 的 order_by、谓词、计数/分页语义必须下推 App 后端，不先取相关性 Top-K。自然语言“所有相关文档”的语义全集不能证明时，不伪装穷尽；计划明确降为候选内排序/部分覆盖或请求缩小为可执行条件。稳定游标绑定模式、谓词/排序、scope/snapshot、最后排序键和期限，变更查询不能继续旧游标。

### 5.2 分片、信号族与排名

同一 space/相同模型评分合同的单通道在整个授权范围产生全局排名，使用原生分数与稳定身份归并分片；BM25 必须使用同一语料统计/评分规则，不能对不同分片做独立 min-max 归一化再混加。若执行引擎仅支持分片局部统计，必须实现一致评分或声明近似/不可用，不能保证分片不变性。

不同表示空间保留独立路线，不相加 cosine/MaxSim/BM25。多分片、多改写、多份同内容和全文+rg 不获得额外独立投票；先通道内/词法族内归并，再 RRF。分片数量和划分不能改变精确基线结果；ANN 误差、超时或截断另记覆盖与近似状态。

小规模质量基线采用范围内精确执行；Top-K 和补取必须支持全局融合的候选预算。只取各分片“融合后前 K”可能丢掉全局多通道共同命中，禁止据此宣称全局精确 RRF；先归并各原生通道的候选，再统一融合，受限候选集不冒充全文穷尽。

### 5.3 覆盖数据与预算重选

CoverageTarget 含 target_key、类型（比较对象/子问题/输入图/文档条件）、原始用户约束跨度、是否必需、来源计划版本。CandidateTargetMatch 含 target_key、evidence_key、匹配依据（结构字段、原生锚点、对应子查询命中或经评测的模型任务输出）、来源与状态 MATCHED/UNKNOWN/CONFLICT；相关向量命中只提供覆盖线索，不证明答案被支持。

先满足可判定的必需目标，再按融合排名补充。每次加入锚点时预估本组直接图文附件成本，并为其他必需目标的最小有效单元预留预算；若高位大组阻塞其他目标，允许有界替换为同目标可用的小组。实际 tokenizer/图片输入成本复核后再做有界重选，不截断到失去表头/单位/引用语义。不承诺求得全局最优。

无法满足全部目标时返回每目标状态及预算/召回/载荷原因；UNKNOWN 不当作未相关而删除，UNKNOWN 候选仍可按融合顺序进入剩余预算。覆盖映射不是新 reranker，不新增 LLM 逐候选 yes/no 门禁。数值预算和替换次数待黄金集实验冻结。

## 6. EvidenceSet、内容读取与 Citation

### 6.1 传输信封

EvidenceSetEnvelope 采用 knowledge.evidence_set.v1，至少包含 request_id/retrieval_ref、owner_binding_ref、scope_ref/hash、snapshot_ref、模式/排序/范围声明、primary_hits、evidence、groups、coverage、channel_status、trace_ref、continuation。

EvidenceKey 是 AppBoundBackend 身份加 document_version、parse_run、对象/来源跨度与投影身份的受控编码，不使用裸图号、页码或候选下标。线协议可用不透明键，但服务端必须能验证完整身份。示意：

~~~json
{
  "schema_version": "knowledge.evidence_set.v1",
  "retrieval_ref": "opaque-retrieval-ref",
  "scope_hash": "sha256:scope",
  "snapshot_ref": "opaque-snapshot-ref",
  "primary_hits": [{"evidence_key": "figure-key", "rank": 1, "group_key": "g1"}],
  "evidence": [{
    "evidence_key": "figure-key",
    "kind": "FIGURE",
    "document_version_ref": "opaque-document-version",
    "parse_run_ref": "opaque-parse-run",
    "locator": {"page": 12, "bbox": [0.1, 0.2, 0.8, 0.7]},
    "provenance": "NATIVE",
    "content_handle": "opaque-content-handle",
    "preview_handle": "opaque-preview-handle",
    "payload_status": "DEFERRED"
  }],
  "groups": [{"group_key": "g1", "anchor_key": "figure-key", "linked_keys": [], "relation_keys": []}],
  "coverage": {"manifest_status": "COMPLETE", "payload_status": "DEFERRED", "target_status": "UNKNOWN"}
}
~~~

示例省略必需通道/Trace 等字段，不是完整 API fixture。内容引用还需 hash、MIME、大小、允许视图版本及期限。图文关系类型、出现跨度、双方 locator、附件/反向正文续取保持；不因传输省略而丢失身份。

### 6.2 原文/图像批量读取

/evidence:read 接收当前 delegation、retrieval_ref、精确 evidence_keys 与受限载荷预算。响应元数据列出每项状态、hash/MIME/定位/字节成本和有界流分片；二进制按受控流传输，不能返回供模型自行访问的任意 URL 或全量 Base64 JSON。

App 读取器重新核权并选择冻结的允许视图。Runtime 校验内容哈希、身份、大小/媒体类型、模型输入支持和 scope；权限变化/撤回返回缺口，不换成同图号或当前新版。读取失败按项返回，不清空其余有效证据。预览由 Main API 转发本 App 内容端口，不能绕过授权或泄漏内部端点。

### 6.3 Artifact 与标签

| Runtime 输出 | 目标内容 |
|---|---|
| RETRIEVAL_EVIDENCE | EvidenceSet 有界快照、范围/排名/关联身份、来源及覆盖；不是可搜索文档库 |
| MODEL_EVIDENCE_DELIVERY | 实际文本/图像输入清单、证据键、hash、成本、状态与模型能力快照 |
| GROUNDED_ANSWER | 正文、实际使用的证据标签/引用、覆盖和缺口；不复制 App 治理数据 |

新语义一次性替换旧 CITATION_PACK/Bundle 单附件门禁，不双格式解析。CitationLabelMap 绑定 run_ref、label、evidence_key、角色及 locator；一个来源文档可有多个独立图/段落标签，相同载荷可复用但不同引用出现/关联边不能消失。

无正文的原始图片也可成为主命中和可引用证据；quote 对纯图可为空，必须有真实图像来源/定位及实际交付状态。图题、模型描述和图像本体分别记录。LinkedEvidenceGroup 不自动成为单个 Citation，也不把所有附件升级为 PRIMARY。

## 7. Runtime 统一交付与 Composer

唯一的正式知识问答模型调用位于 Runtime Composer 管线。AppContentPort 负责读取，Runtime Document Skill 负责材料化和身份校验，Composer 的模型输入构造/发送器负责最终预算核对与交付账本。App 不拥有另一套知识问答 Composer、Prompt 拼接器或 LLM 调用；现场诊断/报告等既有业务生成职责按第 2.4 节保持，不借本设计扩大改造范围。

交付顺序：

1. 冻结有效 EvidenceSet 与 CitationLabelMap，向 App 登记精确 Run 保留引用。
2. 按覆盖/预算计划加载文本、表格与实际图片；只将已校验材料交给 Composer。
3. 按实际 Model Serving 能力计算窗口/图像/字节成本，保留系统/用户/历史工具信息与输出预算；必要时有界重选并记录。
4. 发出请求前复核授权/撤回与载荷身份；记录准备的标签/证据清单。
5. 实际发送包含载荷的模型请求后登记 SENT；构造成功但未发送仅为 PREPARED，异常为 FAILED，不计已交付。网络状态不明标记 UNKNOWN_DELIVERY，不能据此保证未执行并无限重试。
6. 完成前核对正文标签、实际使用标签与公开 references 一致，全部落在本 Run 的可用且实际交付集合；输出引用内容不扩大读取权限。

provided_to_answer_model 由 Runtime 对实际发送载荷计算，App 检索响应不得自行声称为 true；SENT 只证明已发出包含载荷的请求，不证明模型阅读理解或结论有支持。模型不支持图片时不发送假的图像 URL，按场景返回能力不足/部分覆盖；纯搜索不调用模型，账本为 NOT_APPLICABLE。

每阶段记录数量、身份与原因，主命中和附件分开。合法可读且预算可容纳时禁止静默零交付；没有安全可用项则明确失败/不足，不强塞材料。来源中的提示词、HTML 或命令均为不可信数据，不能更改 Scope、模型、工具或系统指令；预览不执行主动内容，测试包含间接提示注入与恶意图片说明。

## 8. 状态、错误与续取

统一结果信封区分 COMPLETE、PARTIAL、EMPTY、FAILED，并独立声明能力、目标覆盖、关联清单、载荷和通道状态。完整执行但不足以回答可为 COMPLETE + 目标 UNKNOWN/不足，不能把回答支持状态等同执行状态。

| 原因 | 处理 |
|---|---|
| NO_MATCH | 通道正常且无候选，合法空结果 |
| NO_ACCESSIBLE_PAYLOAD | 候选全部不合法/不可读；仅披露允许的原因类别 |
| MODALITY_UNAVAILABLE / MODEL_SPACE_MISMATCH | 不伪装成零命中或完整看图 |
| CONTEXT_BUDGET_INSUFFICIENT | 无有效单元可装入，给可披露的缩小问题/续取建议 |
| EVIDENCE_MAPPING_FAILED / ASSEMBLY_FAILED | 有候选但后处理失败，保留安全部分结果或明确失败 |
| DELIVERY_FAILED / UNKNOWN_DELIVERY | 记录实际模型交付阶段，不归因为知识库无数据 |
| SCOPE_CHANGED / SNAPSHOT_STALE / CURSOR_EXPIRED | 拒绝过期/越界续取，必要时有界重新授权检索 |
| SCOPE_CLOSURE_BROKEN | 委托响应越出冻结范围，Runtime 失败并审计，不继续回答 |

最终具体错误码在合同工作包集中定义；语义不能由各 App 任意解释。续取游标不可篡改，绑定 App、当前主体、scope/snapshot、查询指纹、角色、排序键和期限；不得暴露未授权数量或其他 App 的存在。

## 9. App 领域适配

AIOps Registry 的资产/版本、来源资格、人工审核和 VERIFIED 规则保留。KnowledgeIndexRef 改为 AIOps 文档版本、解析/投影/索引代次与 Manifest 引用，不继续通过 KC Collection/Bundle/Revision 公共表查正文；旧字段随一次性切换移除，不加双读兼容。案例 Markdown/结构化投影进入同一算法核，业务适用性只产生经登记的确定性排名/提示，不启用模型重排。

Knowledge Retrieval 的资料库、资料包和上传审核可作为自身业务组织，不再成为基础算法前置门禁；Agent 版本绑定本 App 资料库和允许版本。问数与数据源边界保持，不因知识改造复制 Data Query。已有 App 范围共享授权不自动授权跨 App 文档检索；其他 App 必须独立拥有并解析其资料。

## 10. 验收与本次状态

合同测试覆盖无任意路由/路径、委托 audience、owner/scope 绑定、分片不变性、结构排序、覆盖重选、事件确认/重放、两 App 同号图、物理表/目录/缓存隔离、纯图 Citation、间接注入、逐阶段身份不丢失及 SENT/预览区分。

真实模型、数据库和页面门禁按实施计划分层执行。K0/K1 的中立引擎及正式线合同已通过 G-A，字段/版本、逐 App 存储映射和独立基准见[冻结清单](../operations/app-owned-knowledge-k0-freeze.md)，实际命令见[实施记录](../operations/app-owned-knowledge-k0-k1-progress.md)。K2 已实现独立 ASGI 计算/源端点、逐 App 存储、接收恢复和索引准备。前轮在一次性 PostgreSQL 合成库验证两个 App 的 36 张静态表；本轮根据明确权限，仅在开发 Oracle 和本地 Ammolite PostgreSQL 各新建知识检索 App 的 18 张规范表，并各通过 18 项真实事务/恢复/并发验收。AIOps 表不修改，其 Oracle 及跨 App 真库隔离证据保留，不以离线比较代替。前述为 K2 历史验证边界。本轮已替换源码正式入口、退出旧公共 KC／Citation 和生成本地 OpenAPI 快照；未运行全库初始化、修改真实业务表、注册模型、更新线上 API 或部署／重启服务；生产解析器和数值参数仍由 K5 受控实验冻结。K2 历史证据见[K2 实施记录](../operations/app-owned-knowledge-k2-progress.md)，六项代码接入和剩余真实门禁见[发布说明](../operations/app-owned-knowledge-release-readiness.md)。

## 11. 本次审计项闭合映射

| 审计问题 | 文档决定 | 实施验证 |
|---|---|---|
| 目标存储归属冲突 | 基础能力优先级、本文第 1/4 节逐 App 物理表 | K0/K2/K7 的 owner 与旧路径守卫 |
| Runtime 到 App/Composer 合同断层 | 本文第 2/6/7 节固定端口、Artifact、实际载荷 | K3/K4 认证、发送与引用闭包 |
| 固定助手与统一 Agent 互斥 | 第 2.4/9 节统一 Agent 和受控内部执行 | K4/K6 无独立知识会话/固定助手 |
| 日期排序只作用于相关性 Top-K | 第 5.1 节三模式与原生排序/穷尽 | K1/K3/K6 最新合资格项反例 |
| 覆盖目标缺少对应依据 | 第 5.3 节 CoverageTarget/匹配依据及重选 | K1/K3/K5 多目标预算 fixture/质量 |
| 分片融合顺序不明确 | 第 5.2 节单通道全局归并再 RRF | K1/K3/K5 分片不变性与近似披露 |
| rg 无名次但直接进入 RRF | 可选组件第 6 节词法族唯一名次 | R1 独有/重复命中、遍历顺序变换 |
| 早期退出依赖后续能力 | 实施计划 G-A/G-B/G-C 与 K0–K7 DAG | 各包只验已具备依赖，K4 后完整交付验收 |
