# 步骤 7 详细设计：评测、直接上线与 V1 退役

## 目标

V2 不迁移 V1 `KB/File/TxtChunk` 的解析产物；按来源重新入库，生成新的 Bundle/Revision/Document/Parse View/Evidence。完成测试环境验证后，KM Portal 和 Agent 问文路由直接改为 V2；不做长期影子运行、双写或按用户/流量灰度切换。

V1 接口、Skill 和旧表只在 V2 调试稳定前暂时保留，且生产 V2 请求绝不自动回退使用它们。稳定验收后，按清理清单一次性删除 V1 实现和对应表。

## 评测资产与标注

建立版本化评测集，至少覆盖：无附件 Asset、多个附件、下载失败附件、扫描 PDF、多栏文档、PPT、复杂表格、Excel 子表、长文档、同名/语义相近但不相关 Asset、跨附件问题和多 Collection 问题。

每个问题至少标注：允许 Domain/Collection、正确 Bundle/Asset、正确 Document（如适用）、支持 Evidence 的页/Sheet/cell range、可接受答案要点、不得展示的近似候选，以及“列出案例/Asset”问题的正确 Asset 集合。标注集和解析/索引输入都带来源 Revision，避免内容更新后用错版本评测。

## 分层指标与上线门槛

| 层次 | 指标 | 目的 |
| --- | --- | --- |
| 接收 | 受理成功率、重复投递幂等率、孤儿对象率 | Bundle 入库可靠性 |
| 解析 | 成功率、页/Sheet/cell 定位准确率、表格/图片抽取率、时延/成本 | Parser/Docling 质量 |
| Discovery | Bundle/Document Recall@K、NDCG、跨 Collection 覆盖 | 找对业务对象 |
| Evidence | Evidence Recall@K、定位准确率、跨附件覆盖、冗余率 | 找到可引用事实 |
| 回答 | claim 支撑率、引用有效率、答案正确率、证据不足诚实率 | Grounded answer |
| 前端展示 | `doc_results_v2` Precision/Recall、错误展示率 | 只展示真实采用来源 |
| 运行 | P50/P95 时延、队列延迟、失败重试、单位文件/查询成本 | 上线可用性 |

门槛应在样本基线后由产品、KM 和技术共同冻结；不能以“回答大多正确”替代引用、Asset 列表精度或定位准确率。对案例列表意图，`doc_results_v2` Precision 是独立阻断指标。

## 直接上线前验证

在测试/预生产环境对目标 Collection 全量或约定范围重新入库并完成离线评测、压测和故障演练。验证通过后，发布同一套 V2 Portal、KC、Parser 和 V2 Agent/Skill 代码，再一次性把生产 Portal intake URL 和 Agent 问文路由指向 V2。

上线前至少演练：Portal 网络超时后的幂等重投、全部附件下载失败但 Manifest 可检索、Parser 中断与租约重领、Parse View 重解析成功/失败替换、对象发布后数据库失败的 Receipt 清理、Bundle Revision 切换期间 Evidence 查询、Candidate Stale 重 Discovery、未绑定 Collection 拒绝和 V2 `doc_results` 只展示真实引用 Asset。

上线后稳定期只观察 V2 指标、错误和用户反馈；发现问题通过修复 V2、重解析或重新入库处理，不允许把单个请求悄然改走 V1。若发生阻断性故障，只能执行明确的发布回退/服务恢复操作，并记录为事故；这不是常规检索回退路径。

## V1 退役

V2 稳定验收通过后执行一次受控清理：

1. 确认没有 Portal、Agent、APEX 页面、定时任务或外部调用方仍使用 V1 API、V1 Skill 或旧表。
2. 导出必要审计/归档数据，并验证 V2 Collection 已覆盖需要保留的来源。
3. 删除 V1 路由、`TxtBaseSearch`/旧 ask-doc Skill、V1 上传/解析任务和 `KB/File/TxtChunk` 等对应表及索引。
4. 删除 V1 配置、监控和文档，运行 Schema/对象存储残留检查。

清理是独立、显式批准的数据库变更，不与 V2 上线同一事务执行。完成后系统只保留 KC V2 事实模型和引用契约，避免双模型长期维护。
