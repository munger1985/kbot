# AIOps 任务调度与事件流恢复

## 修复边界

- 独立 Blueprint 运行（包括定时巡检、告警和直接发起的 Run）没有 Conversation Turn，不向对话投影事件；自身 Task、Artifact、Run 事件和报告流程保持不变。
- 有 Turn 关联的运行继续投影领取、完成、回答增量、人工等待和终态事件。不按 workflow_kind 一刀切关闭巡检或告警的对话投影。
- 冻结快照声明了 turn_id 或 conversation_id 却没有关联，或关联指向不存在的 Turn，仍然报状态冲突，不掩盖关联损坏。
- 超期查询先处理交互运行，再按截止时间处理后台运行。已有未终结的过期巡检由正常 Reconciler 逐条收尾，不删除、不补造关联、不重跑过期取证。

## 事件流

AIOps Turn 和 Run 的 SSE 共用事件分页读取器，沿用各自入口的鉴权与 AuthContext。

- SSE 参数使用 Main API 已有的 sse_heartbeat_seconds、sse_poll_interval_seconds 和 sse_batch_size。
- 连接建立后立即发送注释帧；无新事件时发送 heartbeat 注释。上游请求尚未返回时也持续保活，同一连接最多只有一个事件读取请求。
- 注释不占用持久化事件序号，不改变 Last-Event-ID。
- 终态必须读完待传分页再发送 done，避免断线续传时丢掉回答增量。
- 客户端断开、取消或关闭迭代器时，取消并等待未完成的上游读取。
- 两端客户端收到 done 后停止读取；状态回读失败时提示“暂时无法确认后台状态”，不把连接中断解释为后台仍在执行。

回答正文继续沿用已存在的模型增量与 answer.delta；Top SQL 列表沿用证据驱动的确定性组装。心跳不延长机器执行预算，也不能替代任务期限收尾。

## 验证

新增回归：

- tests/unit/aiops_agent/test_aiops_standalone_run_lifecycle.py：独立运行、关联运行、安全校验、巡检后领取对话任务、连续过期收尾及 PostgreSQL/Oracle 查询优先级。
- tests/unit/main_api/test_aiops_event_stream.py：空闲/慢上游心跳、单请求约束、断连/取消清理、终态分页与两个公开入口接线。
- Ammolite apps/portal-web/tests/aiops-event-stream.spec.mjs：执行 Portal 事件消费代码和 KBot 流客户端，验证心跳、终态、续传游标及状态未知提示。

另执行任务内核、Worker、预算、巡检调度、Turn 应用、回答增量及未来观察回归，以及两仓架构边界和跨仓一致性检查。已有 Top SQL 标题断言将“已采集候选”与“已跟踪热点”视为不同文案，该项不属于本次修改。

## 上线后核验

代码提交不代表线上恢复。需要正常发布受影响的 Main API 和 AIOps 服务/Worker 后核验：

1. Worker、Reconciler 不再因独立巡检缺少 Primary Turn 关联反复回滚。
2. 已过截止时间的对话进入 PARTIAL/FAILED，而不是持续 ANSWERING；部分结果仅保留经过现有来源验证的证据。
3. 新 Top SQL 的回答任务能够领取，SSE 空闲和等待上游时均可见周期心跳，完成后不再重连。
4. 15 分钟窗口没有足够快照时如实展示缺口，累计榜不可冒充窗口榜单。

本修复不包含线上数据修改、服务重启、Schema 或 Prompt 变更。
