# App 独占知识基础能力实施计划

日期：2026-11-06。状态：K0/K1 G-A 通过；K2–K4、K6 的六项剩余产品代码和 K7 源码退出／发布工件已经接入。K2 AIOps Oracle、K5 真实质量、K6 真依赖端到端及 K7 部署切换仍未验收。适用：KBot4 与 Ammolite；两仓保持本文一致。

依据：[基础能力设计](../architecture/knowledge-answering-capability.md)、[算法详细设计](../proposals/app-owned-knowledge-algorithm-design.md)、[接入与持久化详细设计](../proposals/app-owned-knowledge-integration-detailed-design.md)。本计划取代旧基础能力文件中的 KC 公共存储、固定助手和旧 Citation 改造矩阵，不把既有测试通过数记作本版进度。

## 本轮执行边界

按用户要求，本轮完成六项产品代码：App 业务、正式 Runtime、API／DTO／客户端／OpenAPI、双前端／Domain、AIOps 资料／案例、旧路径退出／发布工件。不执行真实数据库、模型、文档测试或部署。具体入口、离线证据与外部门禁见[六项发布说明](app-owned-knowledge-release-readiness.md)和[编码记录](app-owned-knowledge-foundation-progress.md)。

## 1. 范围与禁止事项

范围：提取既有 KC 解析/检索代码为算法核；Knowledge Retrieval 与 AIOps 两 App 各自接收、持久化、索引和召回；Runtime 统一消费多模态 EvidenceSet 与交付 Composer；同步双平台 BFF、页面、规范 Schema、生成 OpenAPI 和测试。

不做跨 App 检索、公共文档表、独立 reranker/LLM rerank、新模型类别、固定 Knowledge Assistant、独立知识问答会话产品、旧合同 fallback 或双读双写。rg 是独立可选工作包，不阻断不启用它的核心发布。

本计划不是数据库重建、数据处理或部署授权。不得擅自处理真实资料、清库、提交推送、更新 Prompt 数据库、重启或切换线上流量；每个外部动作单独确认范围。

## 2. 依赖图与门禁层次

~~~text
K0 范围/合同/数据处置清单
  -> K1 纯算法与确定性 fixture
     -> K2 App 自有存储、解析接收与恢复
        -> K3 App 完整召回、图文关系、快照/授权适配器
           -> K4 Runtime 委托、载荷交付、Composer/Citation
              -> K5 真实依赖与质量/规模评测
                 -> K6 双前端、Domain 与 App 端到端
                    -> K7 批准后的发布与退出旧路径

K3 -> R1 可选 rg（可独立延期；启用后才增加其专项门禁）
~~~

可并行准备工件，但不能越过依赖宣布通过。门禁分三层：G-A 纯算法/中立合同；G-B App/Runtime 适配器与受控故障注入；G-C 真实数据库、模型、前端及发布。早期 G-A 不要求尚未实现的模型发送或真实 UI，后续 G-C 不能被 fake 通过替代。

## 3. 工作包与退出条件

### K0：冻结范围、合同与基准

前置：无。已通过 G-A；范围/合同/基准见[冻结清单](app-owned-knowledge-k0-freeze.md)，实际验证见[实施记录](app-owned-knowledge-k0-k1-progress.md)。

- [x] 核对当前两仓 HEAD、脏文件、已暂存项和在途其他工作；形成只含本项的文件清单，不覆盖或清理它们。
- [x] 冻结 AppBackendBinding、委托/Scope、QueryInput 三模式、ParseEvent、EvidenceSet、错误/游标、Runtime Artifact 与 CitationLabelMap 的字段及 schema_version。
- [x] 逐 App 冻结接入详细设计中的全部表、既有业务版本映射、索引空间、对象/临时目录、缓存命名空间、Scope 持久化方式和模型角色；不新增 reranker。
- [x] 默认使用本 App knowledge_scope_snapshot/member 保存巨大精确范围，未使用公共 KC 表或声称复用已有等价范围表。
- [x] 建立 App 独立 G-A 合成黄金集、权限/故障集、确定性基线及实验预算；记录准入 fake/文本解析器 revision、许可证与资源约束。真实模型未准入，真实质量黄金集按 K5 冻结，不将合成基准冒充真实效果。
- [x] 列明真实文档、历史 Run、旧 KC 证据/对象和 App 资产映射的保留/待批准处理范围；当前无数据动作批准、未枚举真实数据，按本节退出规则阻止 K7。无法确定的身份不得按名称/Hash 猜测归属。

交付：D0 冻结清单，含 owner、文件、表、端点、合同、数据前提和测试映射；D1 版本化基准配置/数据分割。退出门禁：G-A 合同结构及静态无跨 App 检索字段检查通过；未批准数据动作可继续纯代码工作，但阻止 K7。

### K1：纯解析、投影与检索算法

前置：K0 的合同与基准，不要求业务数据库或真实模型。已通过 G-A，纯内存解析→投影→召回及故障门禁见实施记录。

- [x] 从既有 ir/pipeline/evidence_planner、检索与视觉匹配中提取中立结构、质量路由、ParseEvent 和 ProjectionBuilder。
- [x] 实现 local_key、来源/坐标映射、图号作用域/子图消歧及双向关系的纯算法合同。
- [x] 实现 RRF、全局通道归并、覆盖目标/依据、预算预留与有界重选；不新增模型重排模块。
- [x] 实现 RELEVANCE/FILTERED_ORDER/EXHAUSTIVE 的计划验证与 fake 后端合同；日期排序不允许伪装成候选后处理。
- [x] 实现 MaxSim/token mask/空间校验参考测试，禁止均值池化替代多向量。
- [x] 解析器/模型端口使用 fake，控制流、预算、取消、序列/摘要错误可复现。

退出门禁 G-A：算法无 ORM/FastAPI/IAM/App 导入；图文关系、分片不变性、结构排序反例、覆盖重选、身份守恒和预算选择测试通过。此时只验证“将交付的清单”，不宣称模型已收到载荷。

### K2：App 表、解析接收与可靠索引任务

前置：K1。

- [x] 两 App 分别实现规范实体、Repository/UoW、SourceReader、允许视图、暂存对象与解析接收器。
- [x] 实现计算任务受理、分页事件拉取、ACK、制品读取和取消的内部协议；不传回调 URL 或 Session。
- [x] 实现批次落库后 ACK、重放幂等、同键异内容拒绝、租约/栅栏、Manifest 最终化与取消恢复。
- [x] 实现 App 自有 outbox/索引任务、generation 准备和 pin 保留引用的存储，不依赖公共文档表。
- [x] Oracle DDL/manifest 与 PG canonical ORM/初始化导出已同步；知识检索 App 的真实 Oracle/PostgreSQL 定向验收各 18 项通过。
- [ ] AIOps Oracle 真库验收；本轮明确禁止修改其表，未创建、写入或清理其表，不以知识检索 App 或 PostgreSQL 的结果代替。
- [x] 以准入 UTF-8 及 synthetic 图文 fixture 验证背压、进程内存硬上限、制品过期与对象提交失败；不代表真实 Docling/模型质量通过。

以上勾选表示组件已编码和相应受控测试通过，不表示两个 App 的整体门禁都已通过。实际命令、39 项 K2 单测/每仓、前轮双 App PostgreSQL 集成，以及本轮知识检索 App 的 Oracle/PostgreSQL 各 18 项定向验收见[K2 实施记录](app-owned-knowledge-k2-progress.md)。定向验收只清理自己的合成数据；另一 App 只做离线映射比较，不计跨 App 真库隔离通过。新组件尚未替换线上入口，旧路径退出/进程启用按 K7 单独批准。

退出门禁 G-B：两数据库适配分别通过接收/恢复/竞态与物理隔离测试；App 原始/解析结果可按来源定位；不发布半成品，不读其他 App 数据；无运行时自身 Schema 探测。

### K3：App 完整检索与关系/快照安全

前置：K2。源码正式 App 入口已改绑；以下勾选限于编码与离线验证，不代表线上部署；当前使用有界全授权精确索引，模型方向/质量和大规模 ANN 尚未准入。

- [x] 实现各 App 的 AppBoundBackend 与内部检索/原文/预览/续取端口，固定组合根与鉴权，scope_ref/hash 可验证。
- [x] 建立词法、文本 dense、跨模态及适用的独立多向量/近重复投影，按 space 与 App 隔离。
- [x] 实现单通道跨分片全局归并、词法族排序、RRF、覆盖预算及稳定分页；原生 BM25 统计不可比时不得宣称精确分片不变。
- [x] 实现图文正反向读取、跨文档精确引用、关系 Manifest/载荷状态、受控续取和原图材料化。
- [x] 同批实现索引代次、发布指针、权限/版本复核、撤回读取门禁、源失效与历史 pin；不把这些延后到 Runtime 端到端之后。
- [x] AIOps Registry 改绑自有文档/索引代次，保留案例资格、脱敏、审核和来源围栏；知识检索保留本 App 资料组织。

退出门禁 G-B：直接检索的三类任务与第 13.4 节图文联动 fixture 通过；第 13.5 节中仅涉及 App/算法阶段的用例通过；越权诱饵零读取/零结果，版本/撤回竞争、游标篡改、分片变换与最新 N 排序反例通过。真实模型质量仍留 K5，不宣称此阶段最佳召回。

### K4：Runtime 委托、实际交付与引用闭包

前置：K3；不得将此包作为 K1/K3 的既有完成条件。正式 Worker、业务 Spec、父子 Run 和历史读取已经装配；勾选不代表真实依赖验收。

- [x] 注册可信 AppRetrievalPort/AppContentPort，服务凭据/audience/owner/Run/Scope 交叉校验；拒绝 App 间 scope/证据混用。
- [x] Document Skill 消费统一 EvidenceSet、冻结 RETRIEVAL_EVIDENCE Artifact 和精确 pin；统一 Composer 加载真实有界载荷。
- [x] 唯一 Runtime Composer 管线构造/发送文本/图像请求，记录 MODEL_EVIDENCE_DELIVERY；App 不自建模型调用。
- [x] 实现纯图 Citation、多对象标签、实际使用/实际交付/公开引用闭包，移除正文唯一门禁与 Bundle 单附件限制。
- [x] 实现无候选、权限失效、预算不足、材料化失败、序列化失败、发送失败/状态不明的不同语义。
- [x] 以记录实际请求体结构的隔离 fake ModelGateway 验证图片字节/文本而非预览 URL 确实进入发送路径；不记录生产正文。
- [x] AIOps 统一 Agent 接入，移除固定内部知识助手与独立知识会话目标，不改变其调查状态机或 VERIFIED 等级。
- [x] 按接入设计第 2.4 节承接必要的内部 Runtime 子执行：固定父 Turn/Run、幂等、取消/撤权传播、事件与结果映射；诊断只需检索时不额外生成答案，不重构已有现场诊断/报告生成器。

退出门禁 G-B：第 13.5 节完整故障注入通过；合法可读且预算可容纳时静默零交付为零；fake 发送清单与标签闭包正确，SENT/PREPARED/FAILED/UNKNOWN_DELIVERY 区分；不把该测试说成真实模型已理解图像。

### K5：真实依赖、质量与规模实验

前置：K4。需要批准的模型/测试语料/测试数据库访问，不得使用无权限生产原文。

- [ ] 各 App 独立运行真实文搜文、文搜图、图搜图、自然照片/截图、表格/复杂版面、近重复与多文档任务。
- [ ] 比较解析器局部修复、SearchUnit/表示、文本/跨模态/多向量、RRF、覆盖预算和引用扩展的独立收益；不引入 reranker 实验。
- [ ] 使用相同授权语料、候选/载荷预算、生成模型、任务权重和保留集，报告全部请求的质量/可靠性/成本，而非成功子集。
- [ ] 校准解析质量、模型方向能力、路由、候选预算、RRF 权重、重选次数及索引/压缩方案；未通过能力不能宣称支持。
- [ ] 在看保留集前登记各任务最低质量、关键组不退化、P95/内存/索引/成本预算及统计方法；报告逐查询对照和关键失败样例，不用一个综合分掩盖图像或精确术语退化。
- [ ] 测授权选择性、大范围分片、并发、超时、撤回/索引延迟和冷热缓存；给出可支持规模，不声称通用数据库天然支持多向量 ANN。

退出门禁 G-C：D2 评测报告包含冻结配置、结果、误差/失败分析和批准的能力矩阵；安全/身份/受控 fixture 的错误必须为零，质量与资源门槛全部满足。未满足则返回相关算法/适配器包修正，不以“以后加 reranker”替代验收。

### K6：公开 BFF、双前端与业务闭环

前置：K4；最终退出还要求 K5。

- [x] 同批修改 App 内外 DTO、客户端、API、生成 OpenAPI 与界面，拒绝浏览器的内部 Scope/策略覆盖。
- [x] 两 App 展示真实主命中、图片/图题/引用正文、关联缺口、反向引用续取与模型实际读图状态。
- [x] 知识检索验证文本/图片输入、结果类型、用户数量/排序、纯搜索与问答、上传/审核/解析/文件闭环。
- [x] Domain/业务空间切换时重新签发/校验上下文，清除旧 scope、候选、图片句柄、分页游标和请求；不能显示旧域结果或借并发迟到响应回填。
- [x] AIOps 从统一 Agent 的版本绑定使用资料库/案例库；配置治理、知识发布和知识调用权限不复制或放宽。
- [x] 双平台类型／构建／样式与隔离 fixture 浏览器回归；预览不执行原文主动内容。
- [ ] 单独执行真实依赖端到端，不以 fixture 冒充。

退出门禁 G-C：逐 App 的上传到发布、直接检索、正式问答、引用预览、撤回、Domain 切换和故障恢复通过；模型请求实际载荷与 UI 声明一致。浏览器 mock 通过不替代真实依赖验收。

### R1：独立可选 rg

前置：K3。不启用时不要求实现、安装或探测；不阻断核心发布。

- [x] 按[组件设计](../proposals/optional-ripgrep-text-scan-design.md)实现 TextScanPort、文本投影/字节映射、授权来源清单与受控执行。
- [x] 实现扫描唯一命中的词法族排序、去重不重复投票、预算未穷尽/故障状态；不把遍历顺序当 Top-K。
- [ ] 在 K5 固定核心配置上单独测增益与资源成本；在 K6 验证 EXACT_RECALL/WITHIN_CANDIDATES 的范围与续取提示。

启用退出门禁：组件关闭/未安装核心仍正常；source map、注入/路径/符号链接、退出码/取消、分批扫描及最终证据交付均通过。未通过则该 App 不启用，不伪装扫描成功。

### K7：发布、一次性切换与退出旧路径

前置：K0 数据动作批准、K1–K6 全部门禁；R1 仅对启用 App 要求通过。

- [x] 源码发布工件：规范 Schema／manifest、生成 SQL／OpenAPI、初始化、前端、安装／拓扑和依赖包已同步；发布说明提供固定指纹与批准范围要求。
- [ ] 实际发布时冻结两仓提交、镜像／模型／索引配置指纹；本轮未部署。
- [ ] 对确切目标 App 备份并核验可恢复性，冻结写入/任务领取窗口，确认无未完成旧任务被错误接管。
- [ ] 仅按批准方案应用 App 自有结构、构建索引、切换执行绑定与前后端；不执行全库初始化、不触及无关 App/通知/数据查询表。
- [x] 同一发布物删除公共 KC 知识存储运行路径、两阶段必经门禁、旧客户端/DTO/单附件逻辑和失效配置/测试；不以 feature flag 保留双路径。
- [ ] 运行受控上传/解析、三类检索、图文关联、问答实际交付、权限/撤回/Domain 及服务 readiness 场景，验证真实进程、端口、近期日志。
- [ ] 失败时停止新写入，整体恢复批准的应用/数据库/对象/索引一致快照；不只回退代码或混用旧 DB 与新索引。
- [ ] 保留审计与 pin 需要的对象；任何材料清理单独列明可恢复性和批准范围，不由安装/启动自动执行。

退出门禁：D3 发布记录与回退演练证据齐全，授权范围/真实质量/实际交付受控场景通过，旧路径不存在；此前不能标记整体完成。

## 4. 文件级改造矩阵

表中目录为既有落点，新增文件名为计划，实施前用 D0 核对实际布局。不要因路径变化复制成第二套模块。

| 区域 | 工作包 | 工件/目标 |
|---|---|---|
| packages/knowledge_algorithms（新增） | K1 | contracts、parsing、projections、retrieval、references、ports；无业务持久化依赖 |
| services/knowledge_core | K1/K2/K7 | 从 parsing/ir.py、pipeline.py、evidence_planner.py 等提取算法；仅保留批准的纯计算职责，删除被替换的公共存储/检索 |
| services/knowledge_retrieval_app | K2/K3/K6 | 新 knowledge 领域、实体/仓储/UoW、任务、内部检索/内容端口；不只修改 Agent 表 |
| services/aiops_agent | K2/K3/K4 | 自有解析/索引模块、Registry/KnowledgeIndexRef、来源/审核/撤回、统一 Agent 接入 |
| services/agent_runtime | K4 | AppScopedSkill、Document、Artifact、统一 Composer/交付账本、引用闭包与 Run 保留引用 |
| services/model_serving | K1/K5 | 已有 embedding/visual 多向量/方向/能力 DTO；不新增 reranker 类别/接口 |
| packages/platform_core | K0/K4/K6 | 可信 AuthContext、跨服务 wire DTO、Artifact/错误语义；不放 App ORM/Repository |
| packages/platform_clients | K2/K3/K4 | 纯计算客户端、静态 App 检索/内容客户端；不放 RRF/选择算法，不解析新旧双格式 |
| services/main_api | K6 | 两 App BFF、Scope 发起/预览转发、公开错误与会话入口 |
| 安装/配置/拓扑/工作区清单 | K2/K7 | 包发现、执行器/各 App worker、资源上限及目录；启动不检测/修复自身 Schema |
| 两仓 docs/openapi | K6 | 从实际 DTO/API 生成，不手工伪造目标快照 |

### 4.1 KBot4 专项

- database/oracle/knowledge_retrieval_app、aiops_agent：逐 App 增补详细设计的表与非唯一访问索引，更新各 manifest 和规范初始化。
- database/oracle/knowledge_core：按批准的删除面退出旧正文/索引职责；已有数据/审计先列清单，不能自动删库。
- scripts/db、安装 Schema 工件、tests/acceptance：同步规范导出/应用与离线 Schema 验收，不放运行时门禁。
- ui/knowledge_retrieval、ui/aiops 与相关 JS 客户端：保持 Ammolite 页面参考关系，更新多模态结果和 Domain 切换；不修改 integrations/apex。
- 保持已分线 KM 的不存在性规则，不恢复旧 KM UI/路由来实现知识检索。

### 4.2 Ammolite 专项

- 两 App canonical entities/repositories/persistence、packages/platform_core 的正式 Schema 导出及 scripts/deployment/init_postgresql.py 同批修改。
- 表使用注册的 knowledge_retrieval_ / aiops_ 前缀，不加 ammolite_，不以 core/kc 共用正文表承载 App 解析数据。
- 不新增本地 PostgreSQL migration、兼容表/列、双读双写或历史回填；任何开发测试 Schema 重建也要精确限定目标并获批准，不重建全库。
- apps/portal-web 的知识检索/AIOps、共享内容客户端与类型同批更新；保持 Tenant/ResourceDomain 与平台成员/入口治理规则。

## 5. 测试落点与执行说明

以下为测试分层落点。K0/K1 的算法、线合同和独立合成基准已通过，产品代码及离线测试已补齐，真实 G-C 仍未执行：

| 门禁 | 计划测试落点 |
|---|---|
| G-A | tests/unit/knowledge_algorithms、tests/contract 的 wire/schema、分片/排序/覆盖 fixture |
| G-B App | tests/unit/knowledge_retrieval_app、tests/unit/aiops_agent、tests/integration 的 DB/对象/事件/授权竞争 |
| G-B Runtime | tests/unit/agent_runtime、tests/contract 的载荷发送/引用/owner 闭包 |
| G-C | tests/evaluation 的版本化数据集/runner、tests/smoke 的真实依赖、前端 E2E 与 tests/acceptance |
| 发布 | 规范 Schema/manifest、安装/生成 OpenAPI、依赖/旧路径静态守卫及受控在线场景 |

既有边界检查在实际修改相应工件时运行：KBot 使用 python3 tests/acceptance/check_4_0_boundaries.py 与 check_oracle_schema.py；Ammolite 使用 conda run -n cube python tests/acceptance/check_architecture_boundaries.py 及对应 PG Schema acceptance。新增算法/合同测试采用可独立运行的 unittest/仓库既有测试风格，不假定新测试或模型环境已经配置。

每个结果注明仓库/提交、命令、环境、输入快照、通过/失败/跳过、依赖与限制。编译成功、HTTP 200、mock 通过、模型正常返回或关系表可写，均不能单独证明真实检索质量和实际交付成功。

## 6. 数据、历史运行与回退决策表

| 对象 | 发布前必须明确 | 禁止推断 |
|---|---|---|
| 当前源文件/业务资产 | App 归属、权限、版本、保留清单、是否授权重新解析 | 相同文件名/Hash 即可跨 App 搬运或复用 |
| 旧 KC 解析/索引 | 精确处理范围、退出顺序、可恢复备份和容量 | 新设计授权自动清库或全量回填 |
| 历史 Run/引用 | 保留冻结正文/身份；原文继续可读所需的经批准证据保留方案 | 按新版 ID 替换旧引用、为历史会话恢复旧检索 API |
| 新 generation/临时对象 | 有引用/pin、未完成任务、回收期限与孤儿判定 | 扫目录后直接批量删除 |
| 数据库/索引/对象备份 | 一致时间点、写入冻结窗口、恢复顺序与验证 | 仅代码回退即恢复完整一致状态 |

旧历史接口与新检索不长期共存。若历史原文可读要求尚无不依赖旧运行路径的可批准方案，阻止 K7 并请求用户决定；不擅自降级成断链或构造虚假新 Citation。

## 7. 进度与交付证据

K0、K1 的 G-A 退出门禁已通过；范围、冻结合同、独立基准和实际命令见[2026-11-05 实施记录](app-owned-knowledge-k0-k1-progress.md)。K2 知识检索 App 已通过 Oracle/PostgreSQL 定向受控验收，AIOps Oracle 及完整跨 App 真库隔离证据因本轮权限边界保留，整体 G-B 不标全量完成，见[K2 实施记录](app-owned-knowledge-k2-progress.md)；K3/K4 基础组件、R1 可选组件及真实评测审计工具已编码，离线证据见[基础编码记录](app-owned-knowledge-foundation-progress.md)。正式 App 业务接入、K5 真实测试、K6/K7 未完成。不沿用旧版本 1779/2222 等测试统计。本轮只批准 KC/知识检索 App 数据库相关表的定向操作，K7 的数据重处理/清理、入口切换、部署与重启仍须单独确认；组件测试通过不表示生产接入或真实质量已通过。

后续每包在本文附有日期的真实证据记录：文件/合同/规范工件、测试与失败、未执行依赖、数据动作批准及发布状态。只有上述退出条件实际满足才更新状态；未通过质量或未批准部署不算整体完成。
