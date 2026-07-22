# 步骤 4 详细设计：质量评估与 Evidence 规划

## Evidence 不是固定大小 Chunk

`EvidencePlanner` 消费已通过基本不变量校验的 Structure IR。它的职责是选择可独立召回、可被答案引用且上下文充分的证据边界；不会修改章节结构或把生成式摘要写回原文。

基础类型与当前 KC 表保持一致：`DOCUMENT/SECTION/PARAGRAPH/TABLE/TABLE_ROW/IMAGE/SHEET/CELL_RANGE`。`DOCUMENT/SECTION/SHEET` 可作为父级导航和扩展上下文，叶级 Evidence 承载最终引用。`CAPTION` 默认附着到 TABLE/IMAGE，而不是生成无上下文的孤立结果；确需独立检索时通过子 Evidence 表达。

## 文本与层级规划规则

- 以完整语义段、列表项组和章节边界为第一优先级，以 token 预算为第二优先级；阈值由 parser policy 版本化，不写死在代码。
- 章节开头的短导语优先向后与首个正文块组合；明显承接上文的短尾向前组合；不同章节不因长度而强行合并。
- 短章节可以保留逻辑身份，但检索文本确定性补充文档标题与完整标题路径，避免其成为无语义短块。
- 超长内容按句子、列表项或 Atom 边界拆分，可使用小范围重叠；每个 fragment 必须保存精确 Atom/span，不允许只有合并后的 bbox。
- 父子关系来自 Structure IR 的真实结构，不使用“上一个 chunk”模拟父节点。`heading_level` 来自章节节点，而不是从正文块反推。

建议初始实验区间为正文目标 300–600 tokens、硬上限 800–1000 tokens；它只是 benchmark 起点，需按文件类型和召回结果定案。

## 表格、图像与 Excel

文档表格至少生成一个完整 `TABLE` Evidence；超长表按重复表头的行组生成 `TABLE_ROW` 子 Evidence，并保留表格标题、caption、列头和精确页/bbox。跨页表通过同构表头、列布局和连续性置信度连接，不能仅因相邻页面而合并。

图像 Evidence 的源文字、OCR 和视觉描述分开标明 provenance。视觉描述属于派生内容，引用时必须同时保留图片 locator，且不得覆盖 caption 或正文原文。

Excel 先生成 `SHEET` 结构，再识别子表并输出 `TABLE/CELL_RANGE`；保留工作表名、范围、合并单元格、表头层数、原始值与显示值。面向问文的 Evidence 使用可读表头和行组，规范化数据工件供未来 Data Query 服务消费，两者共享来源定位但不混淆职责。

## 检索文本与来源内容分离

数据库中的 `content` 是可验证来源内容；`retrieval_text` 由 KC INDEX 任务按固定版本生成：

```text
文档标题 + 标题路径 + 局部标题 + 来源内容 + 必要的继承表头
```

不得把全文摘要、LLM cross-reference 或“可能相关”描述复制到每个 Evidence。Discovery Object 可以使用独立摘要和画像；Evidence 检索只使用确定性结构上下文或明确标注的派生视觉内容。这样既减少相似度噪声，也能保证最终 `doc_results` 对应模型实际使用的原始证据。

## 分层质量报告

质量评估同时输出 Document、Page、Section 和 Evidence 四个层级：

| 维度 | 示例指标 |
| --- | --- |
| 提取完整性 | Atom 文本覆盖率、空白/乱码率、重复区域率、表格单元格保真 |
| 阅读顺序 | 栏内/跨栏倒序、caption 距离、跨页连续性低置信度数量 |
| 结构质量 | 标题置信度、编号一致性、非法层级跳变、孤立标题、目录匹配率 |
| Evidence 质量 | 孤立短块率、超限率、源跨度完整率、locator 完整率、重复内容率 |

内容丢失、树非法、稳定键冲突、缺少关键定位或清单 hash 不一致是硬拒绝。标题/阅读顺序低置信度进入质量告警；达到硬门时拒绝新 View并保留旧 ACTIVE View。当前 VLM 只生成有图片定位的独立视觉描述，不修改标题树；结构 VLM 必须在 golden benchmark 证明收益后才能加入新 policy 版本。

Worker 的完成报告至少使用以下激活字段：`passed=true`、空的 `hard_failures[]` 和分层 `metrics{}`。KC 不推断或覆盖 Worker 的质量结论，只验证硬门、工件完整性和 Evidence 清单一致性；任一条件不满足都不能调用激活逻辑。

## Benchmark 与测试门禁

实施前建立不少于 30 份代表性 golden documents，覆盖原生/扫描 PDF、多栏、错误标题、短章节、跨页表、图文混排、PPT、Word、复杂 Excel、损坏或加密附件。标注标题树、正文顺序、关键表图、页/bbox 或单元格范围，以及代表性检索问题。

评测至少包括标题识别 F1、层级准确率、正文保留率、阅读顺序错误率、表格保真率、孤立短 Evidence 比例、locator 完整率，以及端到端 Evidence Recall@K、Document Precision、答案支撑率和引用准确率。任何 VLM 增强必须单独报告增益、时延、成本和幻觉率。

## 实施状态

1. Atom/Structure IR、Docling Adapter、分页/逻辑/Spreadsheet 定位已实现。
2. Reading Order、Outline、Semantic Block、EvidencePlanner 和分层质量门已实现。
3. KC Worker、source URL、工件上传、服务端指纹复算和重解析替换已接入运行入口。
4. 单元/协议测试及真实 DOCX/PDF smoke test 已通过。
5. 生产默认阈值、VLM 模型和策略升级仍由不少于 30 份 golden documents 的离线报告决定。

旧 `HierarchyBuilder/SmartChunker/ChunkReflector` 不再是运行依赖，可在后续代码清理步骤直接删除。
