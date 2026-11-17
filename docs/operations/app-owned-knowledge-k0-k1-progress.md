# App 独占知识算法：K0/K1 实施与验收记录

日期：2026-11-05。状态：K0、K1 的 G-A 退出门禁通过；K2–K7、R1 未执行。

以上为 K0/K1 冻结时的历史状态；后续 K2 编码与受控数据库验证见[K2 实施记录](app-owned-knowledge-k2-progress.md)，不据此改写本记录的历史测试统计。

依据：[实施计划](app-owned-knowledge-implementation-plan.md)、[K0 冻结清单](app-owned-knowledge-k0-freeze.md)。两仓同步算法、线合同、fixture 和文档；平台发行包元数据保持各自版本。

## 1. D0：范围、工作区与权限

| 仓库 | 首批开始 HEAD | 本次冻结 HEAD | 本次冻结暂存区 |
|---|---|---|---|
| KBot4 | 73d87440 | 13355e09 | 空 |
| Ammolite | 0cd07928 | bef20928 | 空 |

完整源码文件清单、冻结时两仓脏文件及允许编辑面见 [worktree-v1.json](../../tests/contract/knowledge_algorithms/worktree-v1.json)。背景工作已推进 HEAD，因此以上是各自实际检查点，不冒充干净提交。本次未提交、推送、清理、覆盖或暂存通知、飞书、OEM/监控、前端等无关在途改动。

允许面：新增算法包、算法单元/合同/评测工件；工作区安装/指纹/发布包清单与算法导入架构守卫；本项设计/实施状态。安装脚本只编辑清单，未执行整仓安装。

未修改服务 API、业务用例、实体、Repository/UoW、数据库 Schema、Prompt、模型绑定或前端；未访问真实数据、数据库、模型、部署或重启入口。没有获得真实资料、历史 Run、旧 KC 对象及资产映射的数据动作批准；保留范围、未知项与 K7 阻止条件已冻结，不猜归属。

## 2. K0：合同和独立基准冻结

- 22 个顶层合同和全部可达嵌套字段、类型/必需性/枚举已形成版本化快照与完整线样例。ParseEvent 使用唯一展平线格式，含分事件载荷结构和受控制品描述，不支持旧嵌套信封。
- 固定 Binding、Scope、委托、三种 QueryInput、EvidenceSet、错误/签名游标、Runtime Artifact 与 CitationLabelMap。查询身份含完整策略指纹及目标；实际查询表示指纹进入计划。
- 两 App 分别冻结 18 张目标表的物理名、关键列/非唯一索引、Scope snapshot/member、对象/临时目录、索引空间和缓存命名空间。AIOps asset_version_id 保留原业务含义，旧 KC 字段只登记原子切换退出，不实现双读双写。
- 两 App 独立 synthetic 黄金/权限/故障集各 10 场景，分别含开发、验证、合同留出；本仓 runner 只用本仓源码/内存端口。baseline、实验预算、依赖 revision/许可证/准入限制和完整结果/输入摘要已记录。
- G-A 只准入 UTF-8 文本适配器、测试 tokenizer/fake 模型。复杂格式与真实模型未获准进入本次执行；真实质量黄金/盲测集和生产参数冻结属于 K5，不能以 synthetic 结果冒充。

具体路径和语义见[冻结清单](app-owned-knowledge-k0-freeze.md)。没有声称生成正式服务 OpenAPI、完成凭据验证或在真实数据库建表。

## 3. K1：算法、端口和进程内闭环

| 源码区域 | 完成行为 |
|---|---|
| contracts | 精确规范/投影证据身份、真实物理/逻辑定位、固定 owner 与快照、严格类型化 wire、版本注册、Runtime 交付/引用合同及签名续取 |
| parsing | 源流大小/摘要验证、固定格式路由、参数化结构/标题与分栏、区域质量/有界修复、原文和备选来源保留、图号消歧及事件/最终 Manifest |
| projections | 允许视图、精确脱敏/跨页来源映射、tokenizer/句子及表格行组切分、批量输出、独立单向量/多向量模型空间 |
| retrieval | 固定 App 后端、有界并行通道、同评分空间全局分片归并、每信号族一次 RRF、三查询模式、读取/复核身份账本、图文关联、覆盖与预算选择 |
| references | 图号作用域、原生锚点冲突、子图、未解决/歧义、直接双向图文及图题/原图关系；不按距离猜图、不无限扩散 |
| MaxSim/ports | token/patch mask/评分/方向/空间校验、原生排序端口、完整内存模型/解析/允许视图/检索端口 fixture |

闭环测试实际消费 ParseEngine 内存结果，经 ProjectionBuilder 产生精确投影，再由内存 App 后端供 RetrievalEngine 召回：正文带图题/原图，图像反向带引用正文；不是仅手填两个互不关联的 DTO。

边界保持：

- 只共享算法，候选/缓存/索引/数据与授权均不跨 App；无 ORM/FastAPI/IAM/Docling/GPU 依赖。
- 解析返回内存对象/事件，App 后续落库并在事务后 ACK；当前未接真实接收器。
- 规范节点/元数据在预算内汇集，超限显式失败；不宣称所有转换器流式低内存或已实施进程级硬限额。
- 可选通道失败不清空有效候选；必需故障、健康零命中、权限失效/无载荷和预算不足分开。关联分页保留续取/缺口，最终撤权不泄漏附件。
- 纯图不依赖正文门禁；展示数量与上下文预算分离。多图 ALL/ANY 保留子输入覆盖；未经登记的组合图文/COMPOSED 明确拒绝。
- 无模型重排、无 rg；不把 DEFERRED 的将交付清单说成模型已收到。

## 4. 本次实际验证

解释器：/home/chris/miniconda3/envs/cube/bin/python（3.12.13）。以下均对两仓各自工作树执行：

| 检查 | KBot4 | Ammolite |
|---|---|---|
| unittest discover -s tests/unit/knowledge_algorithms | 168 通过 | 168 通过 |
| 算法 + workspace_fingerprint + release_verifier + supply_chain 的 pytest 联跑 | 182 通过，另 144 subtests | 180 通过，另 144 subtests |
| 逐 App synthetic runner | 20/20 | 20/20 |
| 架构边界脚本 | 通过 | 通过 |
| pip wheel --no-deps --no-build-isolation --no-index | 成功 | 成功 |
| wheel 源码逐字节核对 | 34 个 Python 文件一致 | 34 个 Python 文件一致 |
| wheel 在 -I -S 隔离解释器中导入 | 根包及 33 个子模块通过 | 根包及 33 个子模块通过 |

架构命令：KBot python3 tests/acceptance/check_4_0_boundaries.py；Ammolite /home/chris/miniconda3/envs/cube/bin/python tests/acceptance/check_architecture_boundaries.py。

runner 的[本仓 results-v1.json](../../tests/evaluation/knowledge_algorithms/results-v1.json)含 20 个结果、39 个输入文件 SHA-256。本次记录的 Python 分配内存峰值：KBot 283477 字节、Ammolite 284301 字节，低于 fixture 的 64 MiB 上限；这不是进程 RSS、真实转换器或生产规模测试。时间/内存预算不作为生产 SLO。

构建产物仅在任务临时目录，不安装到共享解释器：

- KBot：kbot_knowledge_algorithms-4.0.0-py3-none-any.whl，SHA-256 0e2f7799b9ceedd6f3eed0030e5c15fd6e56ece5d4984363357e74bff64741d0。
- Ammolite：knowledge_algorithms-0.1.0-py3-none-any.whl，SHA-256 e7a0a0e904b230b6a35c362a17da72ed0bebb4076ed3ec585e77377a704ce5b0。

本次修复后重跑：跨 App 命中误归为普通通道故障、同 Profile 名不同策略身份碰撞、允许视图跨版本/重叠映射、关联续取丢失、内存解析结果丢失修复原文/未解决引用状态。初次失败不计通过，表中是修复后的结果。

补充离线核对包括两仓源码/测试/合同工件一致、安装脚本 bash -n、JSON/AST/本项文档本地链接与空白检查、git diff --check。Ruff 在该环境不可用，未声称 lint 通过。未运行全量 workspace 安装、真实 DB/模型/页面或 Runtime 交付验收。

## 5. 退出判定与下一阶段

K0 G-A：固定线合同/隔离清单/独立基准/数据前提和静态守卫齐备，通过。数据动作批准为空，按原门禁规则阻止 K7，不阻止纯代码开发。

K1 G-A：解析/投影/三模式召回、结构反例、双向关联、MaxSim、分片不变性、覆盖/预算和身份守恒，以及摘要/序列/超时/取消/fake 故障测试通过。只验收将交付的证据清单，不声称模型发送已实现。

后续入口是 K2：各 App Repository/UoW、表/对象/事件接收恢复和可靠索引。K3 才接真实检索/授权适配器与跨文档精确引用；K4 才接 Runtime 实际载荷/Composer；K5 才验证真实效果；K6 才对齐双页面/Domain；K7 仍需单独批准。没有长期兼容路径或提前切换现网。
