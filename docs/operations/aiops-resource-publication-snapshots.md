# 业务系统关系发布与周期 SQL 快照

## 发布合同

关系创建提交 relation_type_key、origin，以及名称、责任组、确认状态、属性、参与资源。
关系更新只提交后五类可变字段；创建专用字段不允许混入 PATCH，后端仍严格拒绝额外字段。

编辑关系类型或来源时，工作区先删除旧关系，再按新定义创建。每个已成功响应都会更新本地关系 ID、版本及原始详情；后续请求失败后，可在当前页面重试，不重复删除已删关系或重复创建已确认成功的关系。新增资源的参与关系使用发布后的 system_resource_id。

这不是整套发布的原子事务；刷新页面前应保存草稿。请求已在服务端提交但响应丢失时，仍需回读服务端状态后确认，不承诺网络层恰好执行一次。

## 快照取证

2026-11-06 线上只读取证时：

- 运行代码仍为 151ec347，未包含独立 Run 的 Turn 投影修复。
- 四个数据库的 SQL 快照策略均已开启、间隔为 1 分钟，但没有落库快照。
- 其中三个周期 WORKLOAD Run 持续停在 CREATED；它们为没有 Conversation Turn 的独立 INSPECTION Run，属于前一项任务生命周期修复覆盖的阻塞路径。
- 新增 Oracle 因错误的 Agent 前置依赖被阻塞，记录 ACTIVE_AGENT_BINDING_MISSING，没有创建采集 Run。该依赖已移除，不应以绑定 Agent 作为补救措施。
- 查询时 Target 均处于 DISABLED。该状态只说明当时不能采集，不能据此归因此前所有历史缺口；不要自动改变系统、Target 状态或授权。

开启策略仅表示配置意图，不表示任务已执行或数据已落库。增加等待时间或扩大“快照不足”的容忍度不能解决以上问题。

## 资源级采集边界

- SQL 快照和周期活动采样由 Target 策略独立驱动。没有 Agent、Agent 停用、未发布或没有绑定该资源，均不阻止采集。
- 使用明确的 RESOURCE_COLLECTION Run，agent_id、agent_version_id 为 NULL；不创建虚拟 Agent，不附加或扩大任何 Agent 权限。
- Worker 租约和运行查询允许资源采集无 Agent；对话、巡检、告警等 Agent 创建入口的 Agent 必填及授权规则保持不变。
- 固定工具通过同一只读执行器、签名授权、凭据发放和共享并发配额执行。执行中继续检查当前 Target、凭据、采集策略、固定工具和 Task 租约；关闭策略或停用资源后不得继续取证。
- 采集产物是资源事实，不路由为 Agent 诊断工作项，也不展示“继续深入诊断”入口。
- 旧 Agent 阻塞提示在正常调度成功创建任务后清除；其他执行错误及失败计数仍由实际采集结果更新。

## 上线后核验顺序

1. 先部署规范 Schema 的 Run 两个 Agent 列可空变更，再部署 AIOps 服务、Worker 和前端；仅重启旧 Schema 不足以支持无 Agent 采集。部署、拉取、重启及线上配置变更需另行授权。
2. 确认 Target 已启用、采集策略已开启、只读连接可用和采集服务运行。不检查 Agent 绑定。
3. 检查新的周期 Run 能从 CREATED 进入 RUNNING，最终完成；采集错误需按真实连接、只读查询能力、授权错误处理。历史过期 Run 由正常 Reconciler 收尾，不补造 Turn 或历史快照。
4. 核验真实快照落库、采集错误清除、时间及实例身份可比较。至少两次成功且可比较的快照才能形成增量区间；1 分钟策略不等于已经覆盖最近 15 分钟，也不回填开启之前的历史。
5. 完整 15 分钟窗口需要持续采集覆盖该窗口；缺口应继续如实显示，不能用累计榜单替代窗口指标。

## 回归

两仓各自执行：

- node --test tests/contract/aiops_relation_publication.mjs：真实关系创建/更新函数、不可变字段重建、参与者映射、失败重试与版本同步。
- tests/unit/aiops_agent/test_aiops_workload_collection_authorization.py：禁止访问 Agent 仓储，验证 Target 领取条件、三引擎 1 分钟资源采集、真实固定工具授权执行和连续快照落库后的窗口计算。
- tests/unit/aiops_agent/test_aiops_execution_fence.py：无 Agent 发放只读凭据，以及策略关闭、资源停用、凭据/租约变化、动态查询和工具身份错误的拒绝。
- tests/unit/aiops_agent/test_aiops_resource_collection_schema.py：Run 可空列、租约/Worker/查询合同及规范部署工件。
- tests/unit/aiops_agent/test_aiops_standalone_run_lifecycle.py：独立 Run 生命周期和过期收尾，不放松真实对话关联校验。

本次将待提交内容隔离导出后回归：Ammolite 251 项、KBot 255 项通过；Oracle 规范建库工件及重建生成器检查通过。Portal 构建和 18 项相关浏览器回归通过。

Ammolite 全量 PostgreSQL metadata 验收仍受既有字段摘要基线不一致影响；未修改前的 HEAD 在隔离环境中也返回相同摘要 6a1420e1f3db38bc3c638a05f0347af5e25bf7d028105ac2cd808dd2ac46b6d6。本次 Run 两列可空与显式初始化的专项验收通过，不以此宣称全量 Schema 验收成功。

PostgreSQL 规范实体和显式初始化器、Oracle 规范 DDL/清单/生成工件已同步允许这两列为空；不改写历史记录。不执行本地或线上 DDL，不修改 Prompt 或线上数据。
