# 知识解析、检索与问答基础能力设计

日期：2026-11-06。状态：K0/K1 G-A 通过；K2–K4、K6 产品代码、旧路径退出和发布前工件已接入。真实质量／实库／部署门禁尚未执行。适用：KBot4 与 Ammolite。

本文取代原“Knowledge Answering 基础能力最终设计与一次性改造计划”中的目标架构和改造矩阵。旧版测试数量、KC 精确 Revision 接入和既有问答交付记录不是本次 App 独占数据改造的完成证据；源码和离线工件已按本版改造；本轮未修改真实数据库或在线服务。

## 1. 文档职责与生效顺序

| 文档 | 唯一负责的内容 |
|---|---|
| 本文 | 基础能力、所有权、正式问答接入与强制边界 |
| [解析与召回算法详细设计](../proposals/app-owned-knowledge-algorithm-design.md) | 解析、表示、召回、融合、覆盖、图文关系与算法验收 |
| [App 接入与持久化详细设计](../proposals/app-owned-knowledge-integration-detailed-design.md) | 进程调用、传输 DTO、App 物理表、证据交付及平台适配 |
| [实施计划](../operations/app-owned-knowledge-implementation-plan.md) | 文件级工作包、依赖、验收、切换与回退 |
| [可选 ripgrep 设计](../proposals/optional-ripgrep-text-scan-design.md) | 独立文本扫描组件，不改变核心必需能力 |

四份核心文档共同构成实施基线，不能选择性沿用旧架构。业务文档继续决定手册/案例审核、来源资格与 VERIFIED 等领域规则；其旧 KC 公共存储、固定助手、两阶段门禁和“只可引用正文”等技术假设由本套设计替代。历史实现说明必须注明适用批次；当前源码只采用本版路径，不作为另一套目标。

算法和接入合同已经确定；解析器、模型、索引规模路线、参数与质量阈值仍须实验后冻结。未验证选型不得写成已获最佳效果。

## 2. 强制决策

1. 只共享算法代码、模型推理和中立合同。每个 App 独占正文、图片、引用、解析批次、索引、缓存及对象目录，采用不同物理表，不能只增加 app_id 列。
2. 每个 App 仅检索自己的文档。不提供跨 App 搜索、联邦融合、公共候选池、引用跳转或跨 App 去重缓存。
3. ParseEngine 不写业务数据库。输出内存/有界流式事件与受控临时制品，App 接收后经 Repository/UoW 落库；不强制整文档常驻内存。
4. 每个 App 在自身进程组合根构造固定 AppBoundBackend 的 RetrievalEngine。通用算法不知道其他 App 的连接、表名或索引。
5. 解析与检索单元、表示构建解耦；以原生结构和局部质量修复为基线。图片、页面、表格可独立命中，不受 Bundle 画像或正文 Citation 的先验门禁。
6. 本版不实现独立 reranker、跨模态 reranker 或 LLM rerank，不新增对应模型类别、配置、绑定、接口或占位依赖。采用原生检索评分、RRF 和确定性覆盖/预算选择。
7. 正文到精确图片、图片到显式引用正文按已解析关系双向带出。权限、预算、未解析和不可读缺口必须可见。
8. Runtime 继续复用 Conversation/Turn/Run/Task/Artifact/Event、Document Skill 与 Composer，不复制 App 私有问答执行链路。
9. 检索结果不是答案支持证明。有有效候选不能静默丢失，但证据不足仍须明确说明，不强塞无关内容。
10. 从既有 KC 提取和替换，不建设长期并行算法，不保留旧接口 fallback、双读双写或自动历史回填。
11. 数据库 Schema 的准确性仅在规范工件、部署和验收检查，不在启动、readiness 或业务请求中主动探测自身表结构。

## 3. 基础能力的三个使用层次

| 层次 | 调用者 | 输出及运行依赖 |
|---|---|---|
| 解析/投影 | App 持久任务与执行器 | ParseEvent、ProjectionBatch；不依赖聊天 Agent 或 Composer |
| 直接检索 | App 自有检索用例 | EvidenceSet、原图/原文预览、排序及覆盖；不强制创建 Conversation 或回答 |
| 正式问答 | App-owned Agent 与可信执行规格 | Runtime 的可追溯回答、实际引用、事件与运行审计 |

直接检索不要求给每个使用者创建 Agent。正式问答才使用 App-owned Agent、不可变版本及服务端签发的执行规格。浏览器可按产品选择获授权的 Agent，并提交问题、图片和允许的业务筛选/数量/排序；不能提交内部 Scope、Collection/Revision 白名单、安全级、模型、Binding、任意 URL 或检索策略覆盖。

## 4. 进程调用与数据所有权

~~~text
App 上传/任务 -> 纯解析执行器 -> App 接收器 -> App 自有 Repository/UoW
                                         -> App ProjectionBuilder/索引

直接检索页面 -> App BFF -> 本 App RetrievalEngine -> 本 App 数据/索引
                          -> EvidenceSet -> 页面

正式问答页面 -> App BFF -> Runtime（冻结 Agent/Scope/策略）
                          -> 可信静态绑定的 App RetrievalPort
                             -> 本 App RetrievalEngine -> EvidenceSet
                          -> Runtime Document Skill / Composer
                             -> App 内容端口核权读取真实载荷
                             -> Model Serving -> Grounded Answer / SSE
~~~

Runtime 不访问 App 数据库，也不在 Runtime 内用 App Repository 实例执行检索。它仅通过受信配置登记的 App 内部端点调用，不调用 Knowledge Retrieval App 来检索 AIOps 数据。通用 Runtime 支持多个 App 的执行上下文，不代表存在跨 App 检索入口。

| 所有者 | 负责 | 不负责 |
|---|---|---|
| knowledge_algorithms | 解析、投影与检索算法、中立 DTO、融合和关联规则 | ORM、IAM、发布态、HTTP、会话、回答模型调用 |
| 纯解析执行器（复用 KC 进程位置） | 转换器/OCR/VLM 计算、受控制品、计算任务进度 | 公共文档库、业务发布、跨 App 搜索 |
| 各 App | 原文、解析、图文关系、自有物理表/索引、权限、检索执行、发布和内容读取 | 复制 Composer、读其他 App 数据 |
| Runtime | 冻结执行事实、委托 App 检索、统一载荷交付/Composer/引用校验、SSE | 业务文档库、App SQL、业务发布资格 |
| Main API / App BFF | 认证、App/Domain 上下文、公开接口、受控预览转发 | 未校验的范围拼接、第二套 Run、文档解析 |
| Model Serving | 已有模型类别的受控推理与真实能力上限 | 文档目录、跨 App 候选缓存、相关性过滤门禁 |

Runtime 只保存运行所需的最小证据快照、实际交付清单和引用关系；这是按 Run 与 owner App 隔离的审计事实，不建立可复用全文/向量语料库，不提供跨 App 检索。原始图片和历史解析由 App 按引用保留策略保存，Runtime 的固定引用不代表永久读取权限。

## 5. 授权、快照与端口绑定

正式问答冻结以下事实：

- Agent 稳定身份及不可变版本、owner App、Tenant/Domain（按平台适用）。
- App 颁发的 AuthorizedScope 不可变引用/摘要、精确允许版本、策略及撤回代次。
- 由可信组合根选定的 BackendBinding 与 App 内容端口，不接受请求任意选择。
- QueryPlan、检索/生成模型及索引/投影快照、预算和用户数量/排序。

Runtime 的 KnowledgeScopeSnapshot 不再以公共 KC Collection/Bundle Revision 作为基础能力必需字段。App 可在业务内部保留资料库、资料包和资产版本，但必须映射到自己的文档版本与授权范围；不得通过旧 KC ID 自动猜测新身份。

Scope Hash 用规范化序列化计算，不能以 Hash 代替授权或签名。当前权限、发布、撤回和内容视图在检索、续取、预览、交付模型前复核。历史 Run 不随新发布改写；撤权后阻断受限内容再次读取。

## 6. 多模态证据与正式回答

统一输出 EvidenceSet，支持文本、表格、图片、页面和区域；保留对象身份、精确版本、定位、来源、主命中与关联组。详细传输字段由接入详细设计负责。

Runtime 的证据 Artifact 保存 EvidenceSet 的有界快照。Citation 标签绑定精确证据，不再要求一个 Bundle 只有一个附件，也不要求图片必须先找到正文才能引用；已解析的直接图文关系仍按必需关联规则交付。

Document Skill 通过本 App 内容端口加载获授权的真实文本/图像，将有界证据载荷提交给唯一的 Runtime Composer 交付管线。App 提供内容适配器，不自行调用回答模型。Composer 对实际发出的文本、图像、证据身份及失败记账；预览 URL 和句柄不等于模型已收到图像。

模型不支持图像、图像加载失败或预算不足时，纯搜图仍可按真实能力展示；需要读图的问答须报告缺口，不用图片描述伪装看图。完成回答前检查标签闭包、Scope、精确证据身份、实际交付、正文实际使用与公开 Reference 一致；关联材料不得自动升级为答案支持。

知识库 Agent 在模型生成时通过现有 Run SSE 推送 `answer.preview.delta`，仅以纯文本展示生成预览，不显示可点击的引用。收到模型完整结束帧并通过原有引用闭包校验后，以 `answer.delta` 和 `answer.completed` 替换为正式回答及引用。预览不作为答案制品或会话历史；断流、取消或校验失败时不将预览转成正式回答。此流式阶段不改变检索或证据加载耗时。

KBot4 知识检索 App 中，Agent 的图文提问支持同轮问题和最多 4 张会话附件。Agent 版本分别配置可选的 `query_ocr`（提问图片文字识别）和 `query_vlm`（提问图片理解），不继承知识库解析模型。配置 OCR 时先识别文字：明确的转写请求且每张图片都取得完整的有界识别结果时，直接显示 OCR 文字；其他理解任务把原图和可用的识别文字一起交给 VLM。OCR 未读到字、失败或超时不阻断已配置的 VLM；只配置 OCR 时只给可用文字并说明限制，只配置 VLM 时直接看图，两项都不配置时明确提示。

OCR 推理端用请求中的 `expected_revision` 校验冻结配置指纹，响应的 `model_revision` 表示 OCR 引擎版本，不与配置指纹比较。发送给 VLM 的系统提示和用户消息均使用内容列表；纯文字 LLM 补全仍使用字符串。

知识检索 Agent 本轮没有图片附件时，仅检索正文、表格和已有的文字图注，使用该 Agent 版本的文本回答模型，不调度图片 Embedding 或 VLM；文字证据不足时说明缺口，不自动切到视觉路径。已有的关联图片仍可在检索结果查看。带图片附件的提问保留图文路径；直接检索的显式搜图不受这项问答规则约束。交付先根据冻结证据元数据及模型预算选组，再分批有界并发读取；首选证据读取失败或撤权时重新选组，按实际字节、预算与当前权限核对后才交付。检索结果和引用仍使用各自的正式证据身份。

知识库可以分别用文字和图片召回候选后融合排名，不声称使用联合向量或联合图片索引。提问图片及交付给 VLM 的 OCR 文本优先占用各自的预算，独立于知识库证据，也不生成知识库引用；未命中可访问资料时，仍可对图片中能观察到的事实作答。使用知识库资料的回答仍必须引用实际交付的资料。OCR 日志记录阶段、状态及识别数量，不记录图片或识别文本。

## 7. 排序、覆盖与失败语义

相关性检索、结构化筛选排序、全量枚举是不同任务。日期排序和“最新 N 条”必须参与后端候选生成；不能先取相关性 Top-K 再排序并声称全范围最新。不能证明语义全集时，只返回带覆盖声明的检索结果，不承诺列全。

RRF 在同一 App 内使用各通道全局名次；分片是执行细节，不是新增投票通道。词法与可选 rg 先按同一信号族形成可复现排序。覆盖目标及候选匹配依据必须显式，UNKNOWN 不冒充满足；预算不足允许有界重选但不恢复模型重排。

区分正常零命中、全部被合法过滤、能力不可用、载荷预算不足、组装/序列化失败及模型交付失败。合法可读且预算可容纳的证据不得被后处理静默清空；不为非空绕过权限、强塞无关材料或要求模型编造答案。阶段计数、身份流转及排除原因贯穿到实际模型请求。

## 8. AIOps 与知识检索 App

AIOps 统一使用当前业务 Agent 入口。知识库绑定属于具体 Agent 不可变版本；不新增 Domain 固定内部知识助手、不配置另一套 composer_llm、不新增独立知识问答 Conversation/Run 路径。既有 AIOps 调查状态机通过受控知识能力调用同一基础合同，不被强制迁入知识检索 Root Planner，也不复制 Composer 实现。

统一 Agent 的正式知识问答可委托一个绑定父业务 Turn 的内部 Runtime 执行，复用现有 Run/事件合同，不新建用户可见知识会话；具体父子执行、幂等、取消与结果承接见接入详细设计第 2.4 节。仅作诊断知识参考时直接使用本 App 检索，不强制再次生成知识答案。原有现场诊断回答、报告/案例提炼不在本次知识 Composer 替换范围；如使用知识载荷仍须记录真实交付，不因此另造通用问答实现。

运维资料与诊断案例同属 AIOps，分别定制投影、来源资格和业务排序，但检索只访问 AIOps 自有数据。规范、经验、失败教训和当前 Target 事实分开；知识引用不产生当前环境 VERIFIED，不绕过变更审批或实施后验证。审核/发布/退役与来源失效围栏继续有效。

知识检索 App 保留自己的 Agent、上传审核、业务空间、问文/问数/混合、文搜图和图搜图体验。页面向用户展示真实能力、关联缺口、处理状态和 Domain 切换结果，不暴露内部模型/表/Scope。其他 App 不依赖知识检索 App 的 BFF 或私有服务。

平台、Tenant 控制台与 App 的入口/成员授权仍遵守各自项目治理规范；本次不改变成员或 Agent 授权政策。

## 9. 平台适配与发布边界

| 事项 | KBot4 | Ammolite |
|---|---|---|
| 持久化 | Oracle、Domain、UUIDv7/RAW(16) | PostgreSQL、Tenant/ResourceDomain、native UUID |
| App 表归属 | KBOT_KR_*、KBOT_OPS_* 规范 DDL/manifest | 注册的 knowledge_retrieval_*、aiops_* ORM 命名空间 |
| 前端 | ui/knowledge_retrieval、ui/aiops | apps/portal-web |
| Schema 工件 | database/oracle 各 owner 目录 | canonical ORM、初始化器与 Schema 导出 |
| 运行行为 | 同一算法、DTO、隔离、图文关联、排序及验收语义 | 与 KBot4 相同，差异不扩展到产品行为 |

不为新设计恢复已分线产品或已删除集成。KBot 的 KM 产品线边界继续由本仓库分线规则约束，不属于本次实现面；不修改 integrations/apex。

开发分工作包验证，最终以同一批准发布清单原子切换，不长期部署新旧检索路径。已有数据如何保留、重新解析、切换或清理必须单独列出确切范围和批准；本设计不授权重建全库、回填历史、重启或部署。

## 10. 验收与交付状态

验收覆盖解析恢复、App 物理隔离、三类检索、图文双向带出、跨文档精确引用、排序/分片不变性、覆盖选择、授权竞争、实际上下文交付、引用闭包与两平台真实依赖端到端。

纯算法、App 适配器、Runtime 交付、前端和发布采用有向依赖门禁，见实施计划；不得用未来工作包的端到端结果阻塞早期纯算法退出，也不能用 fake 模型通过代替真实质量跑分。

K0/K1 的中立 ParseEngine、ProjectionBuilder、RetrievalEngine、严格线合同及独立合成基准已通过 G-A，详见[实施记录](../operations/app-owned-knowledge-k0-k1-progress.md)与[冻结清单](../operations/app-owned-knowledge-k0-freeze.md)。App 自有存储/接收恢复、内部端点、Runtime 字节交付与可选 rg 的基础编码及离线证据见[基础编码记录](../operations/app-owned-knowledge-foundation-progress.md)。K2 前轮定向实库结果仍单独保留；本轮不再进行真实数据库/文档/模型测试。正式 App、公开 DTO／客户端／OpenAPI、双前端、Runtime Worker／子执行及 AIOps 自有资料已接入；源码旧公共 KC／旧 Citation 路径已退出。真实质量和线上切换仍受 K5／K7 门禁约束，未部署或重启。详见[六项收尾与发布说明](../operations/app-owned-knowledge-release-readiness.md)。现有实现测试及历史交付记录不计入本版完成状态。
