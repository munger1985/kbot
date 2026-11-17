# App 独占知识基础能力编码与离线接入记录

日期：2026-11-06。范围：KBot4 / Ammolite 的基础组件源码、合同、受控合成回归与评测工具。不运行真实文档/模型/数据库测试，不部署、不重启、不切换业务入口；本轮不提交或推送。

## 1. 本轮可交付范围

K0/K1 之外，K3/K4 的可复用基础实现与 R1 可选组件已经编码，并通过本节所列离线测试。这里的“基础实现”包含各 App 独占的存储适配器、内部 HTTP/客户端、解析执行器、查询模型端口及 Runtime Skill/Composer，不只是几个算法函数。

六项产品代码已经进一步接入正式 App／Runtime／公开 API／双前端／AIOps，源码旧 KC／Citation 路径退出。这里仍不把离线通过当真实质量／数据库或上线门禁通过。正式落点及本轮可复现离线命令见[六项收尾与发布说明](app-owned-knowledge-release-readiness.md)。

## 2. 已编码能力和落点

| 模块 | 已有实现 | 重要约束 |
|---|---|---|
| 解析 | knowledge_core/adapters/document_parsing.py、local_ocr.py 与隔离计算 Worker；纯算法 ParseEngine | UTF-8、原生图片、Docling PDF/Office/HTML/Markdown；可选 Tesseract。模型离线预置，禁外部资源抓取和主动 HTML |
| 扫描页 | Docling 无可定位内容页面保留 PAGE 原始渲染图，可选图内 OCR | 不丢空白/扫描页；无法取得页面图时拒绝完整成功。没有 OCR 就没有 OCR 正文，不编造图片描述 |
| 规范投影 | ProjectionBuilder、App allowed_document/prepare | App 策略决定允许视图；保留原文跨度/标题路径。规范 Markdown 表格按完整行切分并重复真实表头，不能切开单元格 |
| 表示模型 | Model Serving embedding/visual 的内部 /represent、NativeRepresentationClient、各 App KnowledgeModelGateway | 单向量回执与真实模型实例绑定；原生 ColQwen 保留 token/mask，不做均值池化。冻结 revision、processor、template、方向、维度与范数 |
| 召回 | retrieval/exact.py、各 App PublishedKnowledgeBackend/KnowledgeRetrievalService | 全授权范围精确 BM25、dense/MaxSim、原图字节 SHA 精确重复、结构化全范围排序/分页；无 reranker/LLM rerank |
| 图文联动 | 文内 REFERS_TO、CAPTION_OF、HAS_ASSET、PART_OF 与双向 Manifest | 同字节图片的不同出现不合并身份；正文带原图，图命中带引用正文；图号不是主键 |
| 文档引用 | references/citations.py、允许视图 hydration | 明确 DOI/规范 URN 目录；正文 DOI 和明确参考文献章节的 [1]→DOI 精确链接，仅当前允许目录内连接。歧义不造边，脱敏后跨度映射回原文 |
| App 内部端口 | 各 App api/composition/knowledge_*、AppKnowledgeClient | Scope、检索、原文流、查询图、主结果/关系续取、pin、发布/撤回、文献标识登记。服务 JWT、AuthContext、专用 Run/Spec/Scope 委托及当前业务授权联合校验 |
| Runtime | composition/knowledge.py、两项 app_knowledge_skill.py、knowledge_delivery.py 与 adapters/knowledge_* | 单路径 Registry/DAG；实际正文/图像字节进入模型请求；引用闭包、PREPARED/SENT/FAILED/UNKNOWN_DELIVERY/NOT_APPLICABLE 分开 |
| 流水线 | 各 App application/knowledge_pipeline.py | 可恢复领取/续租/拉取/投影；固定授权队列接口与 outbox 构建入口；READY 不自动绕过 App 审核发布 |
| 可选 rg | packages/knowledge_text_scan + 各 App composition/knowledge_text_scan.py | 默认关闭，关闭时不导入/探测二进制；授权文本私有副本，literal，无 shell/任意目录；仅一份词法投票 |
| 评测工件 | tests/evaluation/knowledge_algorithms/audit_quality.py | 逐 App/分割/配置审计，逐查询及分任务指标、P95、身份/越权与静默零交付；不能删除失败查询 |

QueryInput.combination 是多图组合，rg 的 pattern_combination 是独立字面量组合，不能混用。近重复的当前生产适配实现是字节 SHA 精确重复，感知近重复需配置已准入表示/指纹端口，不能宣称已经测过照片相似度。

## 3. App 接入方式

- 纯计算进程使用 create_parse_computation_api 与 FixedAppSourceRegistry；只保存有期限的任务制品，不访问 App 业务表。
- 每 App 使用 create_knowledge_foundation_api，固定自己的 UoW、LocalKnowledgeStore、业务 authorization、Profile、tokenizer/image_cost、签名器、模型绑定。两个 App 的表/对象根/实例分离。
- authorization 必须按业务主体、Tenant/Domain、资料组织、审核与状态返回精确版本集合及 READ/PUBLISH/WITHDRAW 操作；不能把“同域所有文档”当默认授权。
- management_callers 单独配置，Runtime 只读身份不获得发布/撤回权限。纯计算 source 调用方不能兼任业务读取或管理调用方。
- App 上传/审核用例调用 register；可信队列驱动 KnowledgePipelineWorker；投影 READY 后仍由原 App 审核用例调用 publish。outbox 可通过 process_projection 接入已有事务重试。
- 索引图片表示读取 App 已提交允许制品；查询图片先通过 query-images 获取绑定主体、Scope/Run/Spec 的许可句柄，不能仅凭 SHA 或路径读取。
- 可信 Spec 编译器生成 KnowledgeExecutionBinding，并冻结 app_knowledge；build_app_knowledge_registry / build_app_knowledge_plan 可注入正式 AgentRuntimeWorker。它们不使用旧 KC Skill，也不会覆盖已有同版本注册槽。
- Runtime 内容客户端按每批 128 项复核，pin 的存储 owner 绑定 Run+Spec 指纹，且 pin 只延长保留，不赋予读取权、不绕过撤回或延长 Scope 有效期。
- Composer 的技术输出上限取 Model Serving 冻结配置，Prompt 由现有可信 Prompt 解析器注入。本轮未修改 Prompt 数据库。LLM 调 chat/completions；VLM 调已有 inference，不把图片发到错误路径。
- 可选 rg 单独安装 packages/knowledge_text_scan；App 通过 TextScanOptions(enabled=True, ...) 明确受控绝对路径、模式、pattern_combination、预算及可信模式规划器。WITHIN_CANDIDATES 还需显式候选规划器。默认不纳入核心依赖或自动安装。

这些组合根已用于正式 App API／Worker 和 Runtime API／Worker。业务改绑只在源码和离线工件完成，不添加新旧双读或 fallback；线上状态未改变。

## 4. 合同和基线变更

K0 的 22 个顶层合同保留，显式增加 knowledge.app_retrieval.v1 与 knowledge.composition.v1 两个正式 Runtime 包装合同；关系枚举新增 CITES_DOCUMENT。除这种精确跨文档关系外，不放宽文内图文关系的同版本/同解析代次约束。

schemas-v1.json 与 examples-v1.json 同步扩展；逐项核对后，原有嵌套合同没有被隐式更改。G-A 的 20 个黄金案例输出逐项完全不变，本轮只更新算法源码 input_manifest 指纹及实际峰值内存，未把失败期望改成成功。

## 5. 前一批基础组件验证记录（保留历史）

统一离线命令（分别在两仓根目录运行）：

    /home/chris/miniconda3/envs/cube/bin/python tests/acceptance/check_knowledge_foundation.py

命令强制以当前仓 packages/services 的 src 为 PYTHONPATH；不依赖可能串仓的 editable 安装。它只执行以下受控检查，不包含任何真实集成脚本：

| 测试组 | KBot4 | Ammolite |
|---|---:|---:|
| knowledge_algorithms | 168 | 168 |
| knowledge_ingestion | 26 | 26 |
| knowledge_core/test_parse_computation | 20 | 20 |
| platform_clients | 5 | 5 |
| knowledge_foundation | 45 | 45 |
| 已有 ColQwen 路径/适配/推理锁 | 不适用该仓既有测试文件 | 6 |
| 合计 | 264 | 270 |

另运行两仓架构边界、36 表物理隔离/规范 Schema 离线检查，以及 Oracle/PostgreSQL 规范建库静态检查。完成后再次运行编译和 diff --check。

新增关键覆盖：签名双 JWT/Run 委托的 ASGI HTTP；SQLite 内存 App 持久化→召回→Runtime Skill→字节交付；同字节两图身份、图文双向联动/关系续取；Run/Spec pin 隔离；索引重建不自动发布；部分读取失败/撤权重选/模型取消；原生 token mask；假 Docling 规范化/页面渲染图；OCR 合成 TSV；真实 rg 仅扫描合成文本；rg 退出码/协议错误/取消回收；关闭扫描不探测；表格行和表头源映射；全部评测请求统计。

SQLite 不验证 Oracle/PG 锁语义；合成图片、fake OCR/Docling/模型不证明实际 PDF、Office、OCR、语义图像检索质量。前轮 K2 的实库验证记录不算本轮重跑，本轮没有真实数据库写入，也没有修改 AIOps 数据库表。

## 6. 六项产品代码收尾

1. 知识检索 App 自有资料库／版本、幂等上传、可信后台摄取、人工审核／发布／撤回和 Agent 资料库绑定已经接入。
2. 正式 Runtime 可信 Spec／Registry／DAG、实际图文交付、引用闭包、父子执行、取消／撤权及历史精确内容读取已经接入。
3. 公开 DTO／BFF、内部 API、客户端和生成 OpenAPI 同步；浏览器不能覆盖内部 scope／owner／模型／策略。
4. 两前端提供文搜文／文搜图／图搜图的输入与主／关联结果、内容读取和续取。Domain／资料库切换清理旧请求及图片，迟到结果不能回填；KBot 切域失败恢复原域。
5. AIOps 使用自有文档、解析／索引代次，保留手册／案例资格、脱敏、发布和来源围栏；非目录知识问答委托正式 Runtime。
6. 源码公共 KC 存储／Bundle／旧 Citation 客户端与合同退出；规范建库、生成 SQL、拓扑／安装／离线检查和评测入口同步。

## 7. 仍未执行的真实门禁与能力限制

- 本轮未运行真实数据库／模型／文档测试；AIOps 数据库没有修改。K2 历史定向结果不计本轮重跑。
- 未部署、重启、注册模型、更新 Prompt 数据库、清理历史对象或切换线上流量。
- ExactIndex 是有界全授权精确基线，不是大规模 ANN；超限显式报错，不截断后冒充完整范围。
- 默认不配置 OCR／语义向量通道；图像字节相同检测不等于语义相似，未配置能力不能宣传可用。
- 文献引用只链接明确 DOI／URN 及可消歧标号，不猜标题／作者年。语义质量、规模上限和“最佳效果”仍待 K5 对照实验。
- 具体验证结果、已知非本任务回归与发布前置见[发布说明](app-owned-knowledge-release-readiness.md)。
