# App 独占知识算法核

K0/K1 的 G-A 合同与纯算法门禁已通过。只共享算法、合同和端口，不提供公共文档库、跨 App 搜索或模型回答；尚未接入现有 App 运行路径。

包含 ParseEngine、ProjectionBuilder、RetrievalEngine、严格版本化 wire、精确身份/来源、区域质量修复、结构排序、保守图号消歧/双向关系、全局分片归并、加权 RRF、原生 MaxSim 及覆盖/载荷预算选择。标准库独立运行，无 ORM、FastAPI、IAM、Docling/GPU、reranker 或 ripgrep 依赖。

边界：

- 解析输出内存对象/事件；原文、修复备选和引用歧义状态保留。App Repository/UoW 在后续接入中拥有持久化、ACK、恢复与发布。
- 当前内置可执行 UTF-8 文本适配器；复杂转换器仍需服务端适配。规范节点/元数据在显式预算内汇集，不宣称任意大文件流式低内存或已实现转换器进程硬限制。
- 投影只消费 App 允许视图，tokenizer、模型空间和资源预算由调用端注入；保留脱敏及跨页来源，不用字符估算生产 token 或均值池化多向量。
- RELEVANCE/FILTERED_ORDER/EXHAUSTIVE 的原生后端合同均有 fake 验收；“最新 N”不是把相关性 Top-K 再按日期排序。
- 文搜文、文搜图、图搜图与正反向关联完成纯内存验收，不表示生产模型质量/方向已验证。未登记的 COMPOSED/图文组合输入明确拒绝。
- 失败、真实零命中、无可读载荷和预算不足分开；模型故障不清空其他合法候选。展示数量与将交付的上下文分离，DEFERRED 不代表模型已经收到。
- 委托/签名游标/Artifact DTO 不代替 JWT 核验、当前权限、真实内容读取或 Composer；这些接入属于 K2–K4。
- synthetic 参数不是“最佳”生产参数。真实模型 revision/许可、质量与规模属于 K5；页面/Domain 属于 K6，数据动作和发布属于 K7，rg 仍是独立可选包。

从各仓根目录执行，不需要安装包或配置数据库：

~~~bash
/home/chris/miniconda3/envs/cube/bin/python -m unittest discover -s tests/unit/knowledge_algorithms
/home/chris/miniconda3/envs/cube/bin/python tests/evaluation/knowledge_algorithms/run_baseline.py
~~~

安装/指纹/发布成员和架构守卫已登记本包；wheel 经源码逐字节核对及标准库隔离导入。阶段范围、冻结工件和实际验证见[实施记录](../../docs/operations/app-owned-knowledge-k0-k1-progress.md)及[冻结清单](../../docs/operations/app-owned-knowledge-k0-freeze.md)。没有创建真实表、部署、提交或推送。
