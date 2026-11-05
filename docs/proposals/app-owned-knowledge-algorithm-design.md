# App 独占数据的解析、索引与多模态召回算法设计

日期：2026-11-05。状态：详细设计，尚未实施。适用：KBot4 与 Ammolite。

本文是底层算法模块的目标设计，不表示现有服务、接口或数据库已经完成替换。两仓库保持本文一致；平台鉴权、数据库、部署方式可以不同，算法合同与产品语义相同。

## 1. 已确定的边界

1. 仅复用算法代码、模型推理能力与标准合同，不建设公共文档库或跨 App 检索服务。
2. 每个 App 只检索自己拥有的文档，正文、图片、引用、检索单元及索引物理隔离。
3. 不提供 App 列表查询、跨 App 路由、联邦检索、跨 App 结果融合或引用跳转。
4. 通用解析器不访问业务数据库，不决定审批、发布、撤回或 App 数据表。
5. 解析结果由 App 接收，再通过其 Repository/UoW 落库；API 和应用编排层不得直接操作 SQLAlchemy Session 或执行 SQL。
6. 通用召回算法通过 App 绑定的存储端口读取本 App 数据，不掌握其他 App 的连接、表名或索引。
7. 不强制全部解析产物常驻内存；支持有界内存、流式交付和任务级临时制品。
8. 文档版本、原始证据、定位与模型快照必须可追溯；同一 App 内的跨文档引用仍然支持。
9. 目标是检索质量，不要求保留 Bundle 两阶段门禁、单向量接口或旧模型配置结构。
10. 新模块从现有 KC 中提取、重构，不复制为第二套长期并行实现，不保留旧接口兼容路径。

“App”指服务端登记的业务模块，不是一个用户、Domain 或每次安装。Tenant、Domain、Agent、资源权限继续在本 App 内生效。

## 2. 总体结构与依赖方向

~~~text
App 入库用例
  -> ParseEngine
  -> ParsedDocument / ParseEvent 流
  -> App 领域增强与合法性校验
  -> App Repository/UoW 保存规范解析数据
  -> ProjectionBuilder 构建检索单元与表示
  -> App 自有检索投影
  -> App 发布并切换可检索快照

App 检索用例
  -> App 授权得到 AuthorizedScope
  -> RetrievalEngine + 本 App 的 RetrievalBackend
  -> 词法 / 文本向量 / 跨模态向量 / 多向量
  -> 候选融合、细粒度重排、同 App 引用扩展
  -> EvidenceSet
  -> App 直接展示，或交给回答模块
~~~

算法模块对 App 的调用只经过注入的端口。端口由 App 组合根绑定，不能从用户提交的 App 名、表名、连接串或任意 Python 路径动态选择。

每个 App 单独构造自己的 RetrievalEngine 实例并固定 AppBoundBackend；不存在接收所有 App 搜索请求再选择数据库的公共检索实例。复用相同 Python 类不代表复用候选池、运行状态或缓存。

共享 GPU 进程不等于共享文档库。模型服务只处理本次获授权的输入，不建立面向所有 App 的搜索入口、全局候选缓存或文档目录。

## 3. 模块组织

新增独立、无业务持久化依赖的包：packages/knowledge_algorithms/src/knowledge_algorithms/。

~~~text
knowledge_algorithms/
  contracts/       # 文档、解析、表示、检索、证据、错误合同
  parsing/         # 路由、规范化、版面、结构、质量修复
  projections/     # 检索单元规划、表示构建、依赖指纹
  retrieval/       # 计划、通道编排、融合、重排、覆盖检查
  references/      # 引用出现提取、同 App 目标匹配规则
  evidence/        # 定位校验、证据集合及来源链
  ports/           # 转换、模型、临时制品、检索存储端口
~~~

依赖要求：

- contracts、retrieval、evidence 不导入 ORM、FastAPI、App、平台 IAM 或 Docling 对象。
- Docling、MinerU、PaddleOCR 等具体适配器位于执行解析的服务侧，转换为中立合同。
- SQL、全文表达式、向量库调用、事务、任务领取和重试状态均位于 App/基础设施侧。
- Model Serving 负责推理；算法包依赖 ModelGateway，不导入具体模型实现或密钥配置。
- 通用编排和重排不是 platform_clients 的职责；platform_clients 只承载实际存在的进程间客户端。
- 不把 App 审批、领域排名、运维业务、业务表实体移入算法包。
- 解析大依赖按执行器安装，不能因知识检索请求导入而强制加载转换器或 GPU 模型。
- 安装、发布清单和工作区包检查需要随实施同步更新；本设计没有创建可安装代码。

App 自有知识模块包括 application、domain、repositories、persistence、adapters、workers。App 不必拥有独立微服务：同一进程中的不同 App 模块也必须分别绑定数据适配器。

## 4. 中立合同与身份

### 4.1 文档与解析合同

| 合同 | 核心字段与约束 |
|---|---|
| SourceDescriptor | App 内文档版本引用、内容 SHA-256、真实 MIME、大小、受控读取句柄；不接受任意外网 URL |
| ParseProfile | 配置版本、语言、质量目标、转换/OCR/VLM 策略、资源预算；任务开始冻结 |
| ParseRun | run_id、输入指纹、解析器与模型快照、尝试代次；不等于业务发布版本 |
| Page | 页序、几何尺寸、旋转、原图引用；Office 逻辑定位可不伪造页码 |
| ContentNode | local_key、类型、父节点、阅读顺序、内容、定位、来源；与数据库主键无关 |
| VisualAsset | local_key、PAGE/FIGURE/PHOTO/REGION、原图引用、裁剪变换、图题与邻近节点 |
| ReferenceOccurrence | 源节点、精确位置、原始引用字符串、候选标识符；解析器不查其他 App |
| ParseManifest | 对象计数、有序分块摘要、模型与解析版本、质量报告、完成状态 |
| ParsedDocument | 规范结构、内容对象、视觉资产、引用出现和 ParseManifest；不包含数据库 Entity |

内容类型至少覆盖 DOCUMENT、SECTION、PARAGRAPH、LIST_ITEM、TABLE、TABLE_ROW、FIGURE、CAPTION、FORMULA、CODE、FOOTNOTE、IMAGE、REGION。PAGE 是物理视图，可与章节树同时存在，不能强制一段内容只有一种层级。

每项表示区分 NATIVE、OCR、MODEL_TRANSCRIPTION、MODEL_DESCRIPTION、APP_DERIVED；模型描述不是原文。结构推断也保存来源、生成器和置信信息。

所有 local_key 在一次确定的解析输出内稳定，基于原生引用/规范位置及算法版本生成；不承诺更换解析器后仍然相同。App 保存 local_key 到自身主键的映射。旧引用必须指向其原解析代次，不能按新主键猜测回填。

### 4.2 位置合同

- 物理页面：document_version、page_no、归一化左上角 bbox，并记录原始坐标、旋转与变换。
- 原生 Office：段落、形状、工作表、单元格等逻辑位置；渲染图另存映射。
- 表格：表、行列、合并单元格、跨页续表关系。
- 图片裁剪：原图身份、父页面和裁剪变换，允许精确返回区域或原图。
- 文本跨度：对应的原始内容节点和字符范围；规范化文本另存映射。

没有可靠页码或坐标时返回逻辑定位及警告，禁止生成假的定位。摘要、图片说明与模型补全只能通过其来源链回到证据。

### 4.3 检索与证据合同

| 合同 | 核心内容 |
|---|---|
| SearchUnit | unit_key、内容对象引用、单元类型、可检索文本、上下文、原始跨度、投影版本 |
| RepresentationSpec | space_id、模型/处理器/权重快照、输入类型、输出形状、归一化、相似度、任务能力 |
| Representation | 单向量或 token 矩阵、mask、来源单元、输入指纹；不把二者混成 list[float] |
| QueryInput | 文本、图片、可选区域、明确的组合方式、输出对象类型、用户指定数量/顺序 |
| QueryPlan | 通道、查询改写、预算、覆盖目标、重排与扩展策略；不能更改授权范围 |
| AuthorizedScope | App 端构造的不可扩大范围、授权快照、允许版本/集合条件与撤回代次 |
| Candidate | unit_key、精确对象身份、通道排名和原始分数、匹配区域、检索快照 |
| EvidenceSet | 有序证据、定位、关系来源、覆盖与通道健康信息；不强制先生成回答 |
| RetrievalTrace | 策略/模型/索引版本、各阶段耗时、预算截断和退化信息；不默认记录原文或图片 |

RepresentationSpec 必须声明支持哪些查询到对象方向，例如 TEXT_TO_TEXT、TEXT_TO_IMAGE、IMAGE_TO_IMAGE、TEXT_TO_PAGE。具有图片输入能力不自动证明适合图搜图；未评测或不支持的方向不能静默启用。

App 归属由服务端固定的 BackendBinding 确定。算法请求没有 app_ids 或目标 App 路由字段。算法包不规定平台必须新增 app_id 鉴权字段；复用各平台已有可信上下文。

AuthorizedScope 必须绑定签发它的服务端 BackendBinding，后端校验绑定及授权快照。即使不同 App 恰有相同的文档 ID，也不能将一个 App 的 scope 或 Candidate 交给另一个 App 的适配器读取。

### 4.4 核心接口

下面是待实现的类型化接口示意，不是现有 Python API；Referenced 类型均为 contracts 中的数据合同，方法不得接收 Session、表名或任意 App 路由。

~~~python
class ParseEngine:
    def parse_events(
        self, source: SourceReader, profile: ParseProfile, run: ParseRun,
    ) -> AsyncIterator[ParseEvent]: ...

class ProjectionBuilder:
    def build(
        self, document: AllowedDocumentReader, profile: ProjectionProfile,
    ) -> AsyncIterator[ProjectionBatch]: ...

class RetrievalEngine:
    def __init__(
        self, backend: AppBoundBackend, model_gateway: ModelGateway,
    ): ...

    async def retrieve(
        self, query: QueryInput, scope: AuthorizedScope,
        profile: RetrievalProfile,
    ) -> EvidenceSet: ...

class AppBoundBackend:
    async def snapshot(self, scope: AuthorizedScope) -> RetrievalSnapshot: ...

    async def search_lexical(
        self, request: LexicalRequest, scope: AuthorizedScope,
        snapshot: RetrievalSnapshot,
    ) -> CandidateBatch: ...

    async def search_dense(
        self, request: DenseRequest, scope: AuthorizedScope,
        snapshot: RetrievalSnapshot,
    ) -> CandidateBatch: ...

    async def search_multivector(
        self, request: MultiVectorRequest, scope: AuthorizedScope,
        snapshot: RetrievalSnapshot,
    ) -> CandidateBatch: ...

    async def search_near_duplicate(
        self, request: NearDuplicateRequest, scope: AuthorizedScope,
        snapshot: RetrievalSnapshot,
    ) -> CandidateBatch: ...

    async def expand(
        self, request: ExpansionRequest, scope: AuthorizedScope,
        snapshot: RetrievalSnapshot,
    ) -> CandidateBatch: ...

    async def load_evidence(
        self, keys: tuple[EvidenceKey, ...], scope: AuthorizedScope,
        snapshot: RetrievalSnapshot,
    ) -> EvidenceBatch: ...

    async def revalidate(
        self, keys: tuple[EvidenceKey, ...], scope: AuthorizedScope,
        snapshot: RetrievalSnapshot,
    ) -> ValidationResult: ...
~~~

SourceReader、AllowedDocumentReader 和制品读取都由 App 绑定且受限；AllowedDocumentReader 只提供本 App 已准许用于索引的视图。ProjectionBuilder 注入 ModelGateway，但不写存储。ProjectionBatch 由 App 的写入适配器持久化。

各 SearchRequest 包含对象类型、查询表示/空间、Top-K、超时和分页/预算条件。DenseRequest 明确文本或跨模态空间，MultiVectorRequest 明确矩阵和 mask。每个后端方法均强制传 scope 与 snapshot，禁止“加载上下文时省略权限”。

CandidateBatch 包含候选、通道状态、索引身份及是否穷尽；空列表不是故障或能力不足的替代。revalidate 是发布、授权及证据状态的复核，不查询或校验数据库 Schema。

ReferenceCatalog 和 Citation 预览也使用相同的 scope/snapshot 约束。App 接收端自行编排事务，解析器和 ProjectionBuilder 都不调用其 Repository。

### 4.5 策略预算与失败合同

| 配置 | 必须显式记录的预算 |
|---|---|
| ParseProfile | 输入大小/页数/像素、并发、内存/时间、修复次数、允许的局部失败策略 |
| ProjectionProfile | tokenizer、单元长度、重叠、表格行组、视觉分辨率、张量/token 上限 |
| RetrievalProfile | 每路线候选数、融合预算、重排数、总超时、扩展深度/扇出、证据载荷预算 |

用户结果数量与内部候选预算分开。无效或互相矛盾的参数应拒绝，而不是截断后假装满足；服务上限导致无法满足用户数量时明确说明。参数在实验后冻结，不以本文未验证的数字作为默认质量承诺。

最低错误类别：UNSUPPORTED_INPUT、RESOURCE_LIMIT、PARSE_QUALITY_INSUFFICIENT、ARTIFACT_EXPIRED、MODEL_SPACE_MISMATCH、MODALITY_UNAVAILABLE、SCOPE_CHANGED、SNAPSHOT_STALE、CHANNEL_FAILED。实施时固化完整错误码和重试属性。

检索整体状态区分完整成功、部分/退化、空结果和失败，并另行报告 capability、coverage、channel_status。合法零命中不能包装成失败；必需通道故障不能包装成合法零命中。

## 5. 解析算法

### 5.1 输入检查与格式路由

App 先保存原始文件和持久化任务，再交给执行器。执行器校验内容哈希、MIME 与大小，限制解压比例、页数、像素、转换时间、外链和主动内容；复杂转换在受限进程中执行。

路由按文件实际能力，而不是只按扩展名：

1. 原生 Office/HTML/结构化文本优先提取逻辑结构，必要时另渲染页面。
2. 数字 PDF 先取原生文字、版面与内嵌对象。
3. 扫描页采用 OCR；混合 PDF 按页或区域判断，不强制全文件 OCR。
4. 自然图片保留原图，OCR 和描述是辅助表示，不把整图等同于文字文档。
5. 表格、公式、代码使用专门结构，不一律转成无类型长文本。

Docling 是首个可保留适配器，不是中立合同。MinerU/PaddleOCR 可作为实验适配器，由同一评测决定具体语料的路由选择。

### 5.2 质量检测与局部修复

每页/区域计算质量向量，不只看文字长度：

- 原生字形、文字区域与提取文本覆盖是否一致。
- Unicode 异常、乱码、重复行、页眉页脚重复。
- 多栏阅读顺序、标题层级、正文连续性。
- 表格单元格覆盖、行列结构及续表一致性。
- 文字区域、图题和图片的空间对应。
- OCR/VLM 结果与图像及原生文字的对齐情况。

处理过程：

~~~text
基础提取
  -> 质量检测
  -> 仅对失败区域选择 OCR / 高精度解析 / VLM 转写
  -> 对齐、去重、保留候选来源
  -> 复评
  -> 达标、局部失败或整体失败
~~~

阈值和最多修复次数属于版本化 ParseProfile，须用标注集校准。预算到达或连续无改善后停止，不无限调用模型。OCR 自报置信度、模型输出流畅程度不能单独作为通过条件。

修复结果不能覆盖不可变原始提取记录。VLM 转写、解释、推断分别标注，未知内容保留未知，禁止把推测数值写成表格事实。

### 5.3 结构重建

- 结合原生样式、几何布局与阅读顺序识别标题；规则有歧义时可调用受约束模型建议。
- 去除检索文本中的重复页眉页脚，但原始节点仍可追溯。
- 表格保留标题、列头、单位、单元格与跨页关系。
- 图片与图题、正文指向、所在章节关联；独立图片和整页都可检索。
- 解析显式“见图/表/章节”和文献引用的出现位置，不在这一阶段生成全库语义图。
- 所有结构修复保留算法依据，不用单次 LLM 重写整份文档。

### 5.4 流式输出与完成语义

统一使用 ParseEvent 流；小文档的 parse() 只是将同一流汇总为 ParsedDocument。

事件族：RUN_STARTED、PAGE_BATCH、NODE_BATCH、ASSET_BATCH、REFERENCE_BATCH、STRUCTURE_FINALIZED、RUN_COMPLETED、RUN_FAILED。事件包含 run_id、sequence、batch_key、内容摘要及依赖键。

全局阅读顺序、跨页结构在 STRUCTURE_FINALIZED 前允许待定；App 可以暂存批次，但最终 Manifest 校验完成前不能将半成品发布为完整解析结果。结构补充以显式事件交付，不覆盖已提交批次内容。

生产者和消费者使用有界队列与背压。图片、页面和张量以字节流或受控临时制品交付，不将全部二进制 Base64 塞入一个 JSON。

流式接口本身不保证底层转换器低内存。每个适配器必须声明是否支持分段读取及其峰值内存边界；全量物化的转换器应在受限进程中运行，超出预算则切换分段适配器或明确拒绝，不能仅靠逐批发送结果声称已解决大文档内存问题。

同进程可注入临时制品端口；跨进程由标准任务/事件协议传输，不传 Python 回调或 App 数据库会话。临时制品的路径/桶由可信配置确定，调用者只能获得受限句柄。

App 只有在批次落库成功后确认消费。交付语义为至少一次，App 用任务代次、batch_key、摘要和事务内串行化保证幂等；同键不同内容必须冲突。取消和失效任务用租约/栅栏代次拒绝迟到写入。

若恢复时必须重新调用非确定性模型，而不是重放已保存批次，应建立新的 ParseRun；不能复用旧批次键再接受不同内容。局部失败页清单和可用模态进入 Manifest，由 App 的明确质量策略决定是否接收，不能静默跳页后宣称完整成功。

任务元数据和恢复点持久化，不放在唯一的进程内存里。制品完成持久接收后再按确认和保留策略清理；过期不应导致成功任务被宣称可恢复。不同 App 不共用解析缓存或临时制品命名空间。

## 6. App 接收、领域增强与存储

App 的接收用例分为四步：

1. 校验任务来源、源文件版本、事件顺序、摘要、引用完整性和资源预算。
2. 经 Repository/UoW 写入本 App 的暂存解析代次及批次，提交后确认消费。
3. 最终校验 Manifest，冻结规范解析结果，运行 App 领域增强与索引构建。
4. 仅当 App 所需通道就绪且业务门禁通过时，切换可检索快照。

App 可提取错误码、产品版本、尺寸、材料等领域信息，追加自己的表和搜索单元；原文及通用定位不可被覆盖。用于过滤的模型推断必须有来源和不确定性，不能无提示地排除可能相关资料。

App 的存储集合至少覆盖：document/version、parse_run、content_node、visual_asset、reference、search_unit、representation、index_generation、ingestion_job，以及需要可靠异步索引时的 outbox。

这是归属与逻辑集合，不要求每个集合都一对一对应独立表。最终表结构由 App 的数据量、查询方式和所在数据库决定，但不能回到“所有 App 共用解析正文表加 app_id”。

- PostgreSQL 使用登记的 App 表命名空间；Oracle 遵循 KBot 的规范 DDL。
- 不按用户输入、Domain 或每次安装动态建表，不暴露任意表名参数。
- 各 App 的对象目录、全文/向量/多向量索引和查询结果缓存均物理或引擎原生集合隔离。
- 表隔离之外仍校验 Tenant、Domain、Agent、文档状态、发布资格和安全等级。
- 应用层不执行 SQL；App 适配器实现查询，UoW 拥有事务。
- KBot 的业务幂等与唯一性不得依赖新增 UNIQUE/CHECK 或伪装成唯一索引；使用应用校验和适当锁定。
- 不在启动或业务请求中检测、建造或修复自身表结构；DDL 只在部署与验收流程执行。

脱敏必须发生在所有可检索表示构建之前。若 App 禁止向检索使用某字段，不能仅删除文本，却仍让原始截图、图片说明、向量或重排输入泄漏它。需要另存带来源映射的允许视图，并让所有相应表示绑定同一视图。

## 7. 索引构建算法

### 7.1 解析与切分解耦

ParseEngine 输出结构事实；ProjectionBuilder 根据版本化 ProjectionProfile 生成 SearchUnit。调整切分、embedding 或重排模型通常不重跑转换/OCR。

基础单元策略：

- 正文按章节、段落与语义边界组合，长段落在句子边界分割。
- 标题路径可加入检索文本，但不伪装成原文跨度。
- 表格同时建立表级、带列头和单位的行组单元，保留单元格映射；不把每格无条件独立化。
- 代码与公式尽量保持完整，必要时按可解释边界分割。
- 图片建立 FIGURE/PHOTO 单元，页面另建 PAGE 单元，两者不能用相同键覆盖。
- 区域单元保留父图片和 bbox；跨页内容以来源列表连接。

长度由目标模型 tokenizer 和 Profile 预算约束，不固定一套字符数适配所有语言与模态。上下文重叠保留映射，不能让重复副本在融合中获得多次加分。

### 7.2 并行表示

每个 SearchUnit 按已评测能力生成所需表示：

| 路线 | 输入与用途 |
|---|---|
| 词法 | 正文、编号、图题、允许的 OCR/领域字段；保留精确术语 |
| 文本 dense | 文本专用模型，覆盖文搜文与表达改写 |
| 跨模态 dense | 对齐模型的 text/image 两个方向，覆盖文搜图及语义图搜图 |
| 多向量 | 文档页/区域及适用的查询，保留 token/patch 矩阵做细粒度匹配 |
| 近重复 | 原始内容哈希、感知哈希或局部特征，服务找原图/近重复，不冒充语义相似度 |
| 显式关系 | 结构边及同 App 文档引用，为受控扩展服务 |

不是每个单元都强制生成全部表示。必需/可选路线由 App 的能力合同决定，未生成的能力必须可见，不能返回伪装为完整搜图的文本退化结果。

全文查询由数据库适配器安全生成，保留错误码、型号、数字、中文等信息；不将用户字符串直接拼接成 Oracle Text/ParadeDB 表达式。不在算法核中写数据库特有查询语法。

### 7.3 模型空间与张量合同

space_id 覆盖模型权重/适配器、revision、处理器、输入模板、维度、归一化及距离定义。查询只能进入完全匹配且支持当前方向的空间。

多向量表示包含 token_count × dimension、dtype、有效 token mask，必要时包含 token 到页面区域的映射。禁止用对 ColQwen/ColPali 输出直接求均值替代其原生算法。

MRL 截断、量化、token 聚类压缩均属于新表示配置，需要独立指纹与质量验证；不能把压缩后的索引仍标记为原空间的未经压缩输出。

只存向量不足以完成证据检索，必须能加载其对应的原文或图片。生成式描述、摘要和领域信息可以辅助召回，但不能取代原始证据。

### 7.4 依赖指纹与更新

~~~text
原始文件
  -> 规范解析代次
  -> App 允许视图 / 领域增强
  -> SearchUnit 代次
  -> Representation 代次
  -> IndexManifest
~~~

每条依赖记录输入摘要、算法/配置版本和实际模型快照。输入或依赖变化仅失效其下游。

同一 App 的重复计算缓存必须包含授权命名空间和完整指纹；不同 App 即使文件 Hash 相同也不跨 App 复用结果。换 reranker 不要求重做 embedding；换 embedding 必须建匹配的新索引；修改原图裁剪会失效对应视觉表示。

## 8. 召回算法

### 8.1 查询合同与计划

App 先得到 AuthorizedScope，再调用：

~~~text
app_engine.retrieve(QueryInput, AuthorizedScope, RetrievalProfile)
    -> EvidenceSet
~~~

QueryInput 支持纯文本、纯图片、图片加文字、显式区域；指定结果类型如 TEXT、FIGURE、PHOTO、TABLE、PAGE 或 MIXED，并区分语义相似与近重复查找。

用户指定的数量、排序和版本条件必须进入结构化合同，不能被改写或重排悄悄覆盖。结构化计划可用受约束 LLM 提议查询词和任务，不允许模型选择 App、扩大权限或删除硬条件。失败时使用确定性基础计划，返回告警。

多图片查询显式声明 ANY、ALL 或 COMPOSED。ANY 可按独立查询融合；ALL 必须检查同一对象或证据集合是否覆盖全部输入；COMPOSED 仅在有已评测组合能力时启用。不能把多张图向量直接平均后声称支持全部语义。

### 8.2 通道选择

| 任务 | 基础召回 | 精排与补充 |
|---|---|---|
| 文搜文 | 词法 + 文本 dense | 文本重排、章节/表格上下文 |
| 文搜图 | 跨模态 text->image + 图题/OCR 词法 | 跨模态重排，文档图表可加多向量 |
| 图搜图 | 支持该方向的跨模态 image->image | 区域匹配；找原图时加近重复路线 |
| 文搜视觉文档 | 跨模态 dense + 独立多向量召回 + 词法 | 页面/区域重排与原图证据 |
| 图文混合 | 经评测的组合模型，或显式多路策略 | 保留各输入覆盖信息，不宣称与组合模型等价 |

自然照片与文档截图使用各自的能力评测，不能因支持 TEXT_TO_PAGE 就默认支持高质量 IMAGE_TO_IMAGE。

多向量需要独立召回路线；若只对 dense Top-K 做 MaxSim，必须标记为“多向量重排”，不能宣称解决 dense 漏召回。小规模可以授权范围内精确扫描建立质量上限，大规模再评估 ANN/压缩索引。

### 8.3 并行召回、过滤和预算

1. 在同一授权与索引快照内并发执行已启用路线。
2. 每条路线的 App、Tenant/Domain、发布资格、版本和资源权限必须在候选产生前约束。
3. 按稳定 unit_key 合并，保留通道排名、对象类型与原始分数。
4. 候选截断按相关性预算，不先按上传时间截断可检索文档范围。
5. 后端无法在 ANN 中精确处理某授权条件时，须实现具备范围内 Top-K 正确性的执行策略或报告能力不可用，不能仅全库 Top-K 后过滤。
6. 预算、超时及覆盖不完整记录在响应里；不能把“只搜了一部分”伪装成全范围未找到。

在同 App 内搜索多个授权集合是允许的，不等于跨 App 检索。

### 8.4 候选融合

初始使用加权 RRF：

~~~text
fusion(u) = sum_c weight(c) / (rrf_constant + rank_c(u))
~~~

每条路线仅对同一单元计分一次。同一模型的重复改写、同一内容的重叠窗口应先在通道内融合，避免通道数量制造虚假相关性。

不同模型的 cosine、MaxSim、BM25 原始分数不能直接相加。权重与候选预算按任务版本化并通过离线实验选择，不把某个固定值写死为所有 App 最优。

去重键包括精确来源、对象身份、解析/投影代次。禁止按 Bundle 或“文档+页码”提前去重，PAGE、FIGURE、同页多个区域应保留各自身份。最终可分组展示，但不能吞掉有效独立证据。

### 8.5 多向量评分

采用模型规定的 late interaction，例如：

~~~text
L(Q, D) = sum_i weight_i * max_j similarity(q_i, d_j)
~~~

Q 和 D 是同一兼容模型空间的 token 矩阵；padding 不参与评分。归一化、query 权重和聚合必须遵循模型定义，不能自行用长度平均或对称化改变分数语义。

检验空输入、零范数、NaN、溢出、mask、矩阵维度、截断和批次间一致性。查询图片可否作为 Q 由模型的任务能力决定。匹配热区是定位辅助，不自动成为“结论被支持”的证明。

### 8.6 重排

融合后的候选进入适配其对象类型的专用重排：

- 文本查询与文本证据：文本 cross-encoder。
- 文本/图片查询与图片、图表、页面：跨模态重排。
- 多向量候选可补做高精度原生 MaxSim。
- 领域业务加分在 App 层配置，不能将语义无关但业务标签匹配的内容抬成事实证据。

重排只改变已授权候选的相关性，不得新开全库搜索或删除原始出处。文本和视觉精排分数如不能校准，使用保留配额的分组排名融合，不直接混加。

模型不可用时按 Profile 明确退化到融合排序，并记录通道状态；对该任务必须的视觉能力不可用时，应返回能力不足，而非空结果或伪装成功。质量优先的默认实验不受旧版移除 reranker 配置的限制，但不得偷偷恢复旧表字段。

### 8.7 上下文和引用扩展

以精排后的锚点扩展同章节邻段、表格列头、图题、原图，以及同 App 内的明确引用目标。每条扩展边都经过相同授权、版本和撤回校验。

扩展有边类型、深度、扇出、次数和 token/图片预算，保留来源锚点及扩展原因。扩展内容单独评分，不能因被引用就自动变成主要答案证据。

若覆盖不足，可在同一 App 与授权范围内做有界补充检索；达到预算后报告不完整。模型建议不能跳到其他 App，也不能将外部文献解析成其他 App 的资源。

### 8.8 证据组装

输出保留 PRIMARY、CONTEXT、REFERENCED 等角色，以及每个证据的版本、位置、内容类型和实际可读载荷。

- 找图片直接返回图片对象、受控预览和精确来源，不要求先生成自然语言答案。
- 图片问答将实际原图/区域送入获授权的 VLM；不能只传图片说明。
- 多证据问题保留一组或多组可用证据，不能强制一文档一引用而丢失细节。
- 相关性分数与答案支持判断分开；答案生成/claim 校验属于上层能力。
- 检索阶段没有答案，因而不能宣称某候选已支持一个尚未生成的结论。
- 结构化返回完整排名；App 最终展示的回答 Reference 应与实际使用的证据一致。

## 9. 同 App 文档引用算法

引用分两类：文内结构指向和跨文档原文引用。模型推断 RELATED_TO、SUPPORTS、CONTRADICTS 不是默认必建索引，后续须证明对真实任务有增益。

解析器提取 ReferenceOccurrence；App 绑定的 ReferenceCatalog 只在本 App 授权目录内解析目标：

1. 优先精确 DOI、ISBN、文档号、稳定业务标识。
2. 使用标题、作者、年份、文件名等组合召回候选。
3. 校验版本、章节/图表锚点及引用上下文。
4. 不确定时保留 AMBIGUOUS、UNRESOLVED 或 EXTERNAL，不硬连到最相似文档。
5. 记录匹配算法、候选依据和源位置，允许 App 人工确认。

关系身份同时包含源文档版本/解析代次和目标文档版本/位置。目标入库、修改或撤回后，App 驱动待解析引用重试或标记失效，不悄悄把历史引用改到最新版。

文档引用字符串可以指向未入库的外部文献，但不会查询其他 App，也不生成 target_app 路由或从其他 App 取正文。查询不到的授权内目标和不可访问目标不能通过响应泄漏存在性。

同 App 支持反向引用查询。显式引用边、模型语义关系、答案支持边分别保存，禁止混为一种“可信关系”。

## 10. 快照、一致性与任务恢复

源文件版本、解析代次、投影代次、表示空间、索引代次分开管理。检索快照引用完整 IndexManifest，并声明当前可用通道。

App 持久任务状态建议覆盖：RECEIVED、PARSING、PARSED、PROJECTING、INDEXING、SEARCHABLE、FAILED、CANCELLED；审批/发布状态另属 App，SEARCHABLE 不自动等于可被最终用户检索。

- 数据库变更与可靠索引事件在同一 App UoW 提交；外部引擎更新不与数据库伪装成一个分布式事务。
- 外部索引按 generation 幂等写入；必需表示核验完成后，App 原子切换活动指针。
- 新代失败不替换上一成功且仍获授权的快照；禁止原地半覆盖已发布索引。
- 可选视觉增强建立新可用性快照；不得混用不同源版本或不匹配的模型空间。
- 查询固定本次快照，取证和返回前复核发布/撤回权限，失效时拒绝或有界重试。
- 撤回首先在权威 App 读取门禁阻断，随后异步删除索引；所有读取端口都必须执行门禁，不能只靠陈旧向量 payload。
- 历史回答引用的原始解析与图片按审计策略保留；删除与法务要求由 App 明确协调，不默认重解析后直接销毁全部旧证据。
- 垃圾回收仅处理已确认无引用且达到保留期的本 App 产物，绝不扫描或清理其他 App 目录。
- 无全局 exactly-once 承诺，通过至少一次交付、代次、摘要、租约栅栏和幂等写入恢复。

## 11. 定制化规则

允许三类 App 扩展：

| 扩展 | 输入/输出 | 禁止事项 |
|---|---|---|
| ParseProfile | 路由、语言、质量与预算 | 修改其他 App、任意插件代码执行 |
| DomainEnricher | 不可变解析数据 -> 带来源的领域增量 | 覆盖原文，生成虚假 bbox |
| RetrievalProfile/DomainRanker | 授权候选 -> 受限计划与排名增量 | 扩大授权、跨 App 读取、把标签当证据 |

扩展是开发时登记的实现，不从请求加载脚本。扩展版本、输入指纹、失败和退化进入 Trace。

App 只选择经验证的模型与参数组合，不需要各自复制 RRF、MaxSim、结构切分或证据组装。通用默认能力必须可直接使用，不能接入一个新 App 就要求重写所有算法。

## 12. 模型与索引实验矩阵

首轮比较候选，而不是把论文排名写成固定部署结论：

- 文本：Qwen3 Embedding 等文本专用编码/重排对照。
- 统一跨模态：Qwen3-VL-Embedding/Reranker 对照文搜图、图搜图和截图。
- 多向量：原生 ColPali/ColQwen 路线、ColEmbed V2 等；先修正均值池化合同。
- 解析：当前 Docling、自适应 OCR，以及 MinerU/PaddleOCR 的困难页对照。
- 压缩：先保留高质量未压缩基线，再试 MRL、量化、token 聚类，分别测质量损失和成本。
- 图：先显式引用/结构关系，语义图仅在多跳任务消融实验通过后考虑。

模型选择要核对许可证、中文/业务覆盖、GPU 内存和 P95 延迟。自然照片、视觉文档分别评测，不能只依据一个截图检索榜单。

Oracle/PostgreSQL 继续作为各自 App 事务真源。是否使用独立多向量引擎由能力与压测决定；不得未经验证声称现有数据库天然满足大规模多向量 ANN。任何独立引擎仍按 App 独立索引，都是可重建投影。

## 13. 评测与验收

### 13.1 数据集

首批建议 200–500 条人工审核真实查询，包含中英文、数字 PDF、扫描、Office、图表、自然照片、近重复、版本冲突与同 App 跨文档问题。另建权限和故障集，不让它们被平均质量分掩盖。

评测允许一个问题有多组等价证据；标注文档、页面、具体图片/区域和完整证据集，不能只认一个黄金 chunk。训练、调参和验收数据按文档/近重复来源隔离，防止泄漏。

每个 App 的质量评测只使用其自己的数据。隔离测试可以在不同 App 放入相同或更相关的诱饵文档，但目标检索响应不得返回、访问、引用或缓存其他 App 的候选。

### 13.2 指标

- 解析：结构、阅读顺序、表格结构、公式、图片对应、locator 准确性及局部失败率。
- 检索：按任务统计 Recall@K、nDCG、MRR，单独报告精确术语与语义任务。
- 视觉：独立图片命中、区域定位、原图/近重复与语义相似分别统计。
- 多证据：证据集合覆盖、输入约束覆盖、同 App 引用扩展收益。
- 引用：目标解析 P/R、错链、歧义处理、版本锚定及反向关系一致性。
- 上层答案：claim 支持率、引用完整性；不能用召回分数替代。
- 工程：P50/P95、峰值内存、索引大小、模型成本、重试恢复、删除/撤回一致性。
- 安全：跨 App 访问与泄露为零，Tenant/Domain 越权为零，缓存/预览/重排同样覆盖。

### 13.3 消融顺序

A 当前实现；B 新结构单元与词法/文本 dense；C 加跨模态 dense；D 加独立多向量召回；E 加重排；F 加同 App 显式引用扩展。

使用相同语料、查询、授权范围、生成器和受控候选预算，另报告各配置实际算力预算。先跑检索指标，再评估固定回答器，不能靠更换生成模型掩盖召回问题。

验收阈值和各任务权重必须在看验收结果前冻结。质量优先也需要给定资源上限；本设计不宣称任何新模型已经取得本地提升，不以 LLM 自评单一分数作为验收。

## 14. 必须覆盖的自动化测试

1. 算法包无 ORM、App、服务 API、数据库和平台 IAM 导入；依赖方向可做 AST 守卫。
2. 解析不获取业务 UoW；同输入/配置的规则部分输出稳定，模型实际输出按摘要记录而非假定确定性。
3. 坐标旋转/裁剪往返、Office 逻辑定位、跨页表格和图题关系正确。
4. ParseEvent 乱序、重复、同键异内容、取消后迟到、进程崩溃、临时制品过期。
5. 流式大文档峰值内存和背压；使用合成负载，不通过固定 sleep 掩盖问题。
6. space_id 不匹配、图片方向不支持、mask/NaN、MaxSim 与参考实现一致。
7. 同页多图、PAGE/FIGURE 并存，不能被页或 Bundle 去重吞掉。
8. 混合查询 ANY/ALL/COMPOSED、用户数量/排序、候选预算截断和部分通道失败。
9. App 绑定固定，恶意 App 参数、跨 App 引用、其他 App 缓存和临时制品不可访问。
10. Domain/Agent 授权改变、撤回、发布变更发生于召回/重排/预览之间。
11. 索引新代失败、重放、原子切换及历史引用可读；无双写兼容依赖。
12. 前端/回答适配器保留图片原图、排名、证据身份和真实能力警告。

测试放 tests/unit、tests/contract、tests/integration、tests/evaluation；环境验收放 tests/acceptance，不放部署 scripts。

## 15. 现有代码到目标模块的实施映射

| 现有位置 | 目标处理 |
|---|---|
| knowledge_core/parsing/ir.py、pipeline.py | 提取中立结构合同、质量算法；不再直接将 DoclingDocument 暴露为核心输入 |
| parsing/evidence_planner.py | 拆成规范解析与 ProjectionBuilder；切分不要求重跑 OCR |
| application/parse_tasks.py、workers/parser | App 所有的任务与接收流程；通用执行器只生产结果 |
| application/indexing.py、visual_search.py | 表示构建移入算法核；读写与索引代次留给 App 适配器 |
| application/evidence_retrieval.py 的 SearchPort | 升级为 AppBoundBackend，支持各模态与精确对象，不固定 Bundle 作为门禁 |
| Discovery Profile 与 Bundle 两阶段 | 画像保留为可选辅助表示/展示组织，不再是所有证据的前置筛选 |
| entities/repositories/persistence | 解析/证据等业务数据迁入明确的 App 归属，不能继续使用公共正文表 |
| ColQwen2 模型适配器及视觉 schema | 原生多向量与 text/image 方向合同，不再只返回均值单向量 |
| Agent 文档技能和 Citation | 消费统一 EvidenceSet，保留图片/区域与原图，不以正文引用作为图片命中的唯一门禁 |
| KC API 与 platform_clients | 删除被替换的公共知识存储假设；实际远程计算合同单独定义，App 对外 API 由其自身拥有 |

知识检索 App 与 AIOps 是两个独立接入者。服务进程可继续复用现有 KC 的部署位置承载纯解析执行，但不能因此继续把所有 App 解析正文落入 KC 共用表。

其他 App 按实际存在的模块登记接入，本设计不创建新的 Asset App、共享搜索 App 或跨 App 知识目录。

## 16. 实施阶段与退出条件

### 阶段一：合同、隔离与基准

冻结中立合同、Parse/Projection/Retrieval Profile、App 绑定端口及黄金集。增加无跨 App 访问、模型空间和对象身份的合同测试。

退出条件：两仓库相同语义；纯内存 fake App 端口可演示解析结果接收与检索，不连业务数据库。

### 阶段二：解析与 App 接收

从现有 KC 提取解析核，落地流式结果、质量路由、临时制品和 App 接收；保留原文及位置。

退出条件：大文档内存与崩溃恢复通过；App 自有表保存结果；不保留公共正文双写。

### 阶段三：索引与多模态召回

实现 ProjectionBuilder、空间/张量合同、词法/文本/跨模态路线及独立多向量召回、融合与重排。

退出条件：真实依赖跑通三类检索，完成消融实验；图片结果不是正文说明替身。

### 阶段四：引用、快照与安全

实现同 App 引用解析、反向关系、原子索引代次、撤回门禁及历史证据保留。

退出条件：版本/权限竞争与断链测试通过，其他 App 诱饵文档零访问、零结果。

### 阶段五：App 与页面接入

知识检索 App 和 AIOps 分别接入自己的数据与策略；更新 Agent、API、图片预览及真实能力提示，随后处理既有页面对齐和 Domain 切换问题。

退出条件：逐 App 端到端与质量验收通过；两平台合同一致。

这些阶段用于开发和验收，不意味着生产长期维护两套搜索路径。最终发布应以已批准目标替换旧实现，移除废弃接口、配置、测试和文档；不新增旧版 fallback、双读双写或无授权数据回填。

任何现有数据重处理、表删除或部署切换都需单独确认精确范围。不得因新设计重建全库、覆盖其他未提交工作或自行重启服务。Oracle 规范 DDL 与 PostgreSQL 规范 Entity/初始化分别同步，各自运行边界和 Schema 验收。

## 17. 研究依据与采用边界

截至 2026-11-05 已查阅的主要公开依据：

- [Qwen3-VL-Embedding/Reranker](https://arxiv.org/abs/2601.04720)：统一跨模态检索与精排候选，不承诺对所有文本任务最优。
- [ColPali](https://arxiv.org/abs/2407.01449)、[ColEmbed V2](https://arxiv.org/abs/2602.03992)：保留多向量与 late interaction 的依据。
- [MinerU2.5-Pro](https://arxiv.org/abs/2604.04771)、[PaddleOCR-VL-1.6](https://arxiv.org/abs/2606.03264)、[Docling](https://github.com/docling-project/docling)：可替换解析与困难区域优化候选。
- [ViDoRe V3](https://arxiv.org/abs/2601.08620)：真实文档、多粒度证据及定位评测参考。
- [EVIE](https://arxiv.org/abs/2610.11553)：证据感知多向量与压缩研究，近期预印本，不能当作本地成本承诺。
- [GraphRAGProv](https://arxiv.org/abs/2610.37237)：来源约束与关系可追溯，借鉴原则，不默认引入 RDF/SPARQL 全栈。
- [Pix2Evidence](https://arxiv.org/abs/2610.41335)：图像/文字证据融合研究，代码与部署成熟度需另行核验。

论文与开源实现提供候选算法和评测启发；真正上线的模型、索引和参数以本 App 实测、许可证与资源约束为准。本文没有新增跨 App 检索，也没有承诺未经验证的召回提升。

## 18. 本次交付状态

- 已完成：底层算法边界、标准合同、解析/索引/召回流程、App 接入、隔离规则及分阶段验收设计。
- 未实施：算法包、App 表与索引、模型合同、公开/内部 API、页面和部署。
- 未执行：真实模型质量跑分、数据库变更、数据重处理、服务重启。
