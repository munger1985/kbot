# K0 冻结清单与 K1 G-A 验收

日期：2026-11-05。适用：KBot4、Ammolite。本文是 D0/D1 的索引，不是数据、数据库或部署授权。

## 冻结工件

| 工件 | 内容及回归方式 |
|---|---|
| [worktree-v1.json](../../tests/contract/knowledge_algorithms/worktree-v1.json) | 两仓冻结 HEAD、脏文件、暂存区及本项允许面；不包含真实数据库/正文盘点 |
| [schemas-v1.json](../../tests/contract/knowledge_algorithms/schemas-v1.json) | K0 原有 22 个顶层 schema_version，后续显式增加 2 个 Runtime 包装合同（见基础编码记录）；所有可达嵌套字段/必需性/枚举、唯一展平 ParseEvent 及分事件载荷；单元测试逐项对比快照 |
| [examples-v1.json](../../tests/contract/knowledge_algorithms/examples-v1.json) | K0 原有 22 类完整线样例及后续 2 类 Runtime 包装样例；严格编码/解码、未知字段/版本、标量类型回归 |
| [inventory-v1.json](../../tests/contract/knowledge_algorithms/inventory-v1.json) | 两 App 各 18 表的静态目标名、关键列/非唯一访问索引；既有 Agent/资产版本映射；对象/临时/索引/缓存及 Scope 命名空间 |
| [experiments-v1.json](../../tests/contract/knowledge_algorithms/experiments-v1.json) | G-A 基线、数据分割、预算、许可及依赖准入账本；没有生产模型获准进入此门禁 |
| [知识检索合成黄金集](../../tests/evaluation/knowledge_retrieval_app/algorithm-golden-v1.json) | 本 App 独立 owner、数据版本、开发/验证/合同留出场景 |
| [AIOps 合成黄金集](../../tests/evaluation/aiops_agent/algorithm-golden-v1.json) | 独立运维材料，不复用知识检索的资料、范围或缓存 |
| [基准执行器](../../tests/evaluation/knowledge_algorithms/run_baseline.py) | 逐仓使用本仓源码/纯内存端口运行两 App；时间与 Python 分配内存预算，失败即非零退出 |
| [本仓结果](../../tests/evaluation/knowledge_algorithms/results-v1.json) | 20 个受控结果与 39 个输入摘要；单元测试重放比较，防止基线输入静默漂移 |

这些是 G-A 确定性黄金/权限/故障集，不是人工审核真实查询的质量黄金集。合同留出场景不用于数值调优；它们仍是公开 synthetic 反例，不冒充盲测。真实语料、旧实现真实召回率、模型理解能力、复杂文件正确率及规模上限均未测定，保留在 K5 的批准和预登记门禁。

表的目标名、关键列和索引方向已冻结；类型、空值/FK 与 DDL 工件属于 K2 同批实现/验收，不声称真实数据库已存在这些表。model_roles 是算法用途，不是新增 Model Serving 模型类别或网页配置项。生产模型的具体权重/revision/许可证没有选择、配置或执行，因此在账本明确为不准入；不能用空 revision 配置生产能力。Docling 的代码包声明不代表其权重许可或实际已安装版本。

## 合同边界

- Binding 只能来自组合根；Scope 是 App 自有不可变精确版本范围。默认落自己的 scope_snapshot/member，未使用公共 KC 范围实现。
- 委托 DTO 仅描述绑定；服务适配器先核验服务凭据、JWT、subject/Tenant/Domain/Agent/Run 和当前权限。纯合同 validate 不代替 JWT 验签或授权服务。
- 解析事件展平 owner/job/run/attempt/fence/源摘要/Profile 摘要。受控 artifact_handles 带对象归属、hash/MIME/大小/期限；随有序摘要覆盖，不接受路径、任意 URL 或无对象归属制品。大二进制不进入 JSON。App 仍负责读取校验、复制、落库后 ACK。
- QueryInput 支持相关性、资格范围原生排序、可执行谓词穷尽。纯算法 fake 验证三种模式；不把相关性候选排序当“最新 N”。
- QueryPlan/请求身份绑定完整 Profile 指纹和覆盖目标，不能只用 Profile 名称。关联分页保留受控续取和未完整原因；游标 HMAC 绑定 owner/subject/Scope/snapshot/query/Profile/角色/排序/期限，不授予访问权。
- EvidenceSet 区分展示主命中、关联清单和将交付的上下文选择。图像和来源没有正文也能保留身份；正文命中和图像命中双向取关联。
- RETRIEVAL_EVIDENCE、CitationLabelMap、MODEL_EVIDENCE_DELIVERY、GROUNDED_ANSWER 的字段/版本已冻结。没有实现 Composer 或实际发送；SENT 只能由 K4 真实发送路径产生，G-A 不填造发送时间。

## 算法实现范围与资源边界

ParseEngine 校验源流摘要，使用固定格式端口，按区域进行有次数/质量进展边界的修复；保留原文、修复候选、转写来源、歧义/缺失解析状态、阅读顺序和标题结构。当前可直接运行 UTF-8 文本适配器；复杂格式由服务端端口接入，G-A 使用 fake，并不把安装 Docling 当成已完成提取适配。

结构采用参数化分栏/通栏带和标题栈基线，不宣称它是所有复杂版面的最佳算法。规范节点/元数据先在明确预算内汇集，再冻结结构、分批输出；并非任意大文档的低内存解析器。转换器进程内存/强制终止、对象存储和背压属于 K2/K5。

ProjectionBuilder 消费 App 允许视图，区分原文和标题上下文；按 tokenizer/句子及表格行组切分，保留精确跨度与跨页来源。无法无损切分的超大单句/代码/公式/表格明确失败。图像表示、token/patch 矩阵不被均值池化；空间包含模型、processor、模板、维度、评分与方向。

RetrievalEngine 走固定后端和 Scope/snapshot：有界并行通道、全局归并、每信号族一次 RRF、读取/权限复核、图文直接关联、最终复核、覆盖及预算选择。可选通道失败不清空有效候选；全部失败、真实零命中、无可读载荷及预算不足分开报告。多图 ALL/ANY 保留子输入覆盖，不平均向量；未登记的 COMPOSED 或图文组合输入明确拒绝，不假装支持。未实施模型重排或可选 rg。

## 真实数据及历史证据处置（未授权）

| 对象 | 当前证据与保留决定 | 后续动作前提 |
|---|---|---|
| 当前文件/业务资产/版本 | 只检查仓库实体字段，未读真实行、文件或对象清单；全部保留 | 逐 App 的精确 owner/版本/权限清单，用户批准所需重新解析范围 |
| 旧 KC 正文/索引/对象 | 未枚举、搬运、删除或复用；全部保留 | 精确目标清单、容量、可恢复一致备份、一次性退出计划及批准 |
| 历史 Runtime Run/Citation/pin | 未读取历史正文，未改写身份/引用；全部保留 | 不依赖旧检索 API 的保留读取方案和批准；不能按名称/Hash猜归属 |
| App 资产与旧 KC 映射 | 记录源码中的精确字段含义，不伪造真实映射 | 逐版本核对；AIOps asset_version_id 不等于 document_version_id |

批准记录为空。纯代码 K0/K1 可以完成；上述未知数和缺少数据动作批准明确阻止 K7，不以此次编码授权自动补齐。也不为了完成 K0 而连接/扫描真实库。

## 验收命令

从各仓根目录执行：

~~~bash
/home/chris/miniconda3/envs/cube/bin/python -m unittest discover -s tests/unit/knowledge_algorithms
/home/chris/miniconda3/envs/cube/bin/python tests/evaluation/knowledge_algorithms/run_baseline.py
~~~

完整本次命令、通过数、打包/隔离导入证据及限制见[实施记录](app-owned-knowledge-k0-k1-progress.md)。后续版本更改线字段必须同步类型、版本决策、快照、样例与测试，不能绕过快照比较直接宣称兼容。
