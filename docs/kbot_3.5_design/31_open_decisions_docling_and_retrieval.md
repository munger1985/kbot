# 待深化专题：Docling 解析与两阶段检索

## 当前已冻结与未冻结内容

3.5 已冻结的是系统边界：Parser 输出可追溯的 V2 Evidence DTO；KC 用 Discovery 缩小 Bundle/Document 范围，再用 Evidence 返回可引用内容。它们保证未来替换解析器、切块器、召回器或重排器时不需要重做数据模型和 API。

以下内容**尚未冻结**，必须用真实 Asset/PDF/PPT/Excel 样本、标注集和离线评测逐项定案；不能仅根据当前 Docling 输出或主观经验决定。

## Docling 与多模态解析决策

| 专题 | 待比较方案 | 决策依据 |
| --- | --- | --- |
| 文档路由 | 统一 Docling；按 MIME 路由；先轻量质量检测再 TEXT/VISUAL/HYBRID | 解析时延、标题树准确率、表格/图片召回、成本 |
| 版面与阅读顺序 | 原始 reading order；版面重排；VLM 修复局部区域 | 多栏 PDF、扫描件、页眉页脚的引用准确率 |
| Chunk 单元 | 固定 token；标题/语义段落；父子层级；表格/图文独立单元 | Evidence Recall、引用完整性、上下文冗余 |
| OCR/VLM | 仅 OCR；OCR + VLM 描述；置信度触发 VLM | 图表召回提升与生成描述幻觉风险 |
| 表格 | Markdown 表；HTML/CSV/JSON 工件；表头继承与子表切分 | 表格问答正确率、行列定位、Excel/问数复用 |
| Excel | 原生解析；Docling/VLM 增强；双表示融合 | Sheet/子表切分、cell range 准确率、数值保真 |
| 质量门 | 规则阈值；小模型/LLM 评审；人工抽检 | 错误 View 拦截率、误杀率和处理成本 |

建议先建立解析 benchmark：每种典型格式至少覆盖原生文本、扫描、多栏、复杂表格、图文混排、超长文档和坏附件。每个样本标注标题层级、关键表/图、正确页/单元格范围和应召回答案，分别测解析正确率、定位准确率、耗时和单位成本。

## 两阶段检索决策

| 阶段 | 待比较方案 | 重点指标 |
| --- | --- | --- |
| Query 理解 | 原问题；关键词抽取；查询改写/HyDE；Facet/时间/作者解析 | 过滤正确率、召回提升、改写漂移 |
| Discovery 召回 | 标题/Facet BM25；画像向量；混合 RRF；按附件聚合 | Bundle/Document Recall@K、跨 Collection 公平性 |
| Candidate 预算 | 固定 Top-K；按置信度动态预算；按 Collection 配额 | Evidence Recall、时延、长尾 Collection 覆盖 |
| Evidence 召回 | 全文/向量混合；章节父子扩展；表格专用检索；多视图去重 | Evidence Recall@K、引用定位、冗余率 |
| 重排 | 无重排；交叉编码器；LLM rerank；结构/Facet 特征融合 | NDCG、答案支撑率、成本/延迟 |
| 上下文组装 | Top-N；MMR；按 Document/证据类型多样化；覆盖约束 | 跨附件覆盖、token 利用率、答案完整性 |

不应预设“Discovery 必须只用摘要向量”或“Evidence 必须只用 chunk vector”。Discovery 可同时利用 Revision 标题、Facet、Manifest、附件目录和内容画像；Evidence 可同时利用正文、标题路径、表头、OCR/VLM 描述和结构关系。区别仅在于前者输出候选对象，后者输出可引用事实。

## 推荐的实验顺序

1. 先以当前/改造后的 Docling TEXT 输出建立无改写、BM25+Vector RRF 的可复现基线。
2. 单独对比层级切块、完整定位、表格/Excel 双表示，不同时引入 Query 改写或 LLM 重排。
3. 在固定解析产物上比较 Discovery 画像、候选预算和 Evidence 重排。
4. 最后评估 VLM、查询改写和 LLM rerank 等高成本能力，并记录质量增益是否覆盖成本/时延。

每次实验固定 Collection、来源 Revision、Embedding 模型、评测问题集和预算；输出按问题类型、文件类型、是否跨附件和是否包含表格拆分。任何进入生产默认策略的能力都必须保留可回退到上一已评测策略的配置版本，而不是回退 V1 数据模型。
