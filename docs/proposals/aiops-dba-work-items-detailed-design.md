# AIOps DBA 工作项中心详细设计

## 1. 决策与阶段边界

DBA 工作项中心用于承接告警分析、日常巡检和人工诊断后需要持续处理的事项。它不替代正式报告：Finding 表达问题事实，Report 保存冻结输出，WorkItem 承载责任人、SLA、状态和审计，Proposal 承载审批、执行与回滚，Verification 承载处理效果。

当前版本在一期活动队列、详情、人工创建、责任分派、受控状态流转、SLA 和资源链基础上，已经接入告警诊断与日常巡检：严重告警先进入候选范围，Agent 自动诊断后再由确定性策略提炼真正需要 DBA 干预或观察的事项；日常巡检复用同一工作项决策内核，并通过独立的巡检状态适配器提供上下文，不照搬告警 Situation 状态机。

工作项不是原始告警或巡检结果的副本。Signal 表达监控系统观察到的事件，Situation 表达相关信号归并后的故障情境，Inspection Check Assessment 表达巡检检查项在本次证据窗口内的健康判断，Finding 表达 Agent 基于证据确认的技术事实，WorkItem 才表达需要 DBA 承担责任并完成闭环的工作。来源状态决定是否必须进入决策流程，Agent 诊断结论决定工作内容、处理模式和优先级。

## 2. 一期领域模型

- `aiops_work_item`：聚合根，保存优先级、责任组、责任人、SLA 和生命周期。
- `aiops_work_item_occurrence`：相同问题再次出现的事实，保存 Finding 快照和证据引用。
- `aiops_work_item_link`：关联 Run、Situation、Report、Proposal 与 Verification。
- `aiops_work_item_activity`：创建、归并、分派、状态变化和验证完成的审计时间线。
- `aiops_responsibility_group`：当前 Domain 内的 DBA 责任组，保存名称、说明、状态和治理审计。
- `aiops_responsibility_group_member`：责任组与已有 AIOps App 用户的成员关系，只保存稳定 `user_id` 和组内角色，不复制用户名、邮箱、密码或 App 权限。
- `WorkItemRoutingDecision`：应用层结构化决策值，不直接作为可变数据库实体。至少包含 `decision`、`recommended_priority`、`reason_codes`、`action_summary`、`impact`、`confirmation`、`evidence_refs`、`recommended_playbook_id` 和 `observation_window`；决策快照写入 Activity，禁止只保存模型自由文本。

所有内部标识使用 PostgreSQL `uuid`、Python `UUID` 和 UUIDv7。子记录以正式外键关联工作项，工作项以正式外键关联 AIOps Target；业务状态、组合规则和业务唯一性仍由应用层负责，不使用业务 `CHECK`、`UNIQUE` 或唯一索引。

## 3. 生命周期与 SLA

主流程为：

`PENDING_TRIAGE → OPEN → IN_PROGRESS / WAITING → PENDING_VERIFICATION → RESOLVED → CLOSED`

规则：

- `WAITING` 必须填写等待原因。
- DBA 通过明确的“标记工作完成”动作进入 `PENDING_VERIFICATION`，必须填写完成代码和完成说明，并记录完成人与完成时间。页面显示为“DBA 已完成 · 待验证”，不能用一个无审计语义的布尔字段代替。
- 完成代码至少覆盖 `FIXED / MITIGATED / OBSERVED_STABLE / ACCEPTED_RISK / FALSE_POSITIVE / DUPLICATE`。P1/P2 事项应关联处理或验证证据。
- Verification 通过后进入 `RESOLVED`；验证失败或问题再次发生时回到 `IN_PROGRESS` 并增加重开次数。`CLOSED` 只表达最终归档。
- 告警 `RESOLVED` 只说明监控信号已经恢复，不代表 DBA 工作完成，不得直接把工作项改为 `RESOLVED` 或 `CLOSED`。
- 已解决事项再次出现时重开并清除旧解决结论。
- `CLOSED` 与 `CANCELLED` 不进入默认活动队列或自动归并。

| 优先级 | 确认时限 | 解决时限 | 验证时限 |
| --- | ---: | ---: | ---: |
| P1 | 15 分钟 | 4 小时 | 8 小时 |
| P2 | 30 分钟 | 8 小时 | 12 小时 |
| P3 | 4 小时 | 72 小时 | 96 小时 |
| P4 | 8 小时 | 168 小时 | 216 小时 |

默认队列按“已超 SLA、P1、P2、P3、P4、解决时限、工作项 UUID”稳定排序，确保当前仍需人工干预的紧急事项优先于低优先级观察事项，不适用普通资源列表的创建时间倒序规则。

## 4. 告警诊断自动生成与归并

### 4.1 规范语义

Provider 原始严重度先由 Adapter 归一化。Alertmanager 的 `critical / error / fatal / panic` 均归一为 `CRITICAL`，工作项路由不得再次解析 Provider 原始字符串，也不增加第二套 `ERROR` 业务枚举。告警状态使用规范 `OPEN / RESOLVED`，路由必须在提交前读取并锁定 Situation 最新状态，不能使用 Run 启动时的过期快照。

Agent 继续通过证据编译器产生结构化 Finding。工作项决策器组合 Situation 最新状态、Finding、确认度、Target 重要级别、持续时间和重复次数，输出 `ACTION_REQUIRED / MANUAL_INVESTIGATION / OBSERVE / NO_WORK_ITEM`，而不是让模型直接写入工作项。

### 4.2 OPEN 严重告警

`OPEN + CRITICAL` 必须形成可审计决策：

| Agent 诊断结果 | 工作项决策 | 默认优先级 |
| --- | --- | --- |
| 已确认服务中断、数据损失风险、阻塞、容量耗尽、安全风险等，需要立即人工处置 | `ACTION_REQUIRED / INCIDENT_RESPONSE` | P1 |
| 结论为 `LIKELY`、影响受限，或仍需 DBA 进一步核验 | `MANUAL_INVESTIGATION / INCIDENT_RESPONSE` | P2 |
| 自动诊断失败、超时、证据不足或没有形成 Finding，但告警仍为 `OPEN` | `MANUAL_INVESTIGATION / INCIDENT_RESPONSE`，明确记录自动诊断缺口 | P2；核心实例或已确认重大影响可升为 P1 |

持续严重告警不能因为 Agent 没有形成 Finding 而静默消失。优先级由业务影响、确认度、Target 重要性、持续时间和重复次数共同决定，不能简单等同于告警严重度；系统只允许自动升级既有事项，不因告警恢复自动降低历史优先级。

### 4.3 RESOLVED 严重告警

`RESOLVED + CRITICAL` 仍需要保留恢复后的人工观察责任：

- 没有同问题活动工作项时，创建 `RECOVERY_OBSERVATION`，默认 P4；高频重复、持续时间长、核心实例或存在残余风险时为 P3。
- 已有 P1/P2 干预事项时，不再创建第二张观察事项；新增恢复 Occurrence 和 Activity，并将同一事项推进到观察或待验证阶段，保留完整故障时间线。
- 默认观察窗口为 24 小时或一个完整业务高峰周期，由策略配置。观察期内再次发生时自动重开并升级，不能创建重复事项。
- 恢复事件到达而诊断 Run 仍在执行时，等待 Run 完成后以 Situation 最新状态决策；不得先创建 P1 再并行创建 P4。

Situation 从活动转为恢复时发布事务 Outbox 事件。消费者按 Situation 查找已有工作项：有则追加恢复事实，无则在诊断结果可用后创建观察事项。重复恢复事件必须幂等。

### 4.4 其他来源与边界

- 数据质量 Gap 仍可建立 `OBSERVABILITY_GAP / PENDING_TRIAGE` 工作项，但不能冒充已确认故障。
- Chat Run 默认不自动建单，显式路由时可选择 Finding。
- Verification Run 只验证并推进既有事项，不创建新的故障工作项。
- 实施顺序先改造告警诊断，再按第 9 节切换日常巡检；切换前巡检保持当前行为，不做双写或兼容路由。

稳定指纹为：

`tenant_id + domain_id + target_id + finding_type + normalized_object_ref + condition_code`

对象引用规范化后计算 SHA-256。来源类型和处理模式不进入问题身份，避免同一数据库问题分别从告警和其他来源产生两张工作项；`source_kind`、`work_type` 和路由决策保留在 Occurrence、Link、Activity 及聚合根当前视图中。相同指纹且非终态时新增 Occurrence、Link 与 Activity，不新建聚合根。数据库不以唯一约束表达该业务规则，应用事务必须通过锁和幂等键抵御并发重复创建。

## 5. 责任组、成员与工单分派

### 5.1 责任组边界

责任组是 AIOps 内部的工作路由对象，不是新的用户、IAM 角色或权限组。Agent 可以由管理员统一创建，Agent 创建人不自动成为工单责任人；系统也不为每个 Agent 创建伪造的“系统用户”。自动动作的操作者是 AIOps 服务身份，实际处理责任只由责任组和真实 DBA 用户表达。

责任组按 `tenant_id + domain_id` 隔离。每组至少包含名称、说明、状态、可选组长、创建人与版本字段；成员关系包含 `responsibility_group_id`、`user_id`、`member_role=LEAD/MEMBER`、状态和审计字段。同一用户可以加入多个责任组，但同一组内只能有一条活动成员关系。停用采用软状态，历史工作项、Activity 和成员快照不得被删除。

`aiops_work_item` 使用正式 `responsibility_group_id` 关联责任组，保留 `assignee_user_id` 作为具体责任人，并增加 `assignment_source`、`assigned_by` 和 `assigned_at`。下一次完整数据库构建直接以这些字段替换当前自由文本 `assignment_group`，不保留双字段或兼容读写。

### 5.2 成员来源与资格

责任组成员只能从当前 Tenant 已有的有效 AIOps App 用户中选择。现有 IAM `platform_user_app_access(app_id=aiops)`、Tenant 成员状态、App 角色绑定和 Domain 范围是唯一资格来源；责任组不得另建账号、复制用户资料或授予 App 权限。

候选人必须同时满足：

1. 用户和 Tenant 成员状态有效。
2. AIOps App 访问状态为 `ACTIVE`。
3. App 角色在当前 Domain 有效并包含 `aiops:use`。
4. 未过角色有效期。

责任组成员身份只表示可以承接该组工作，不授予诊断、变更或成员管理权限。用户被停用、撤销 AIOps 访问或失去当前 Domain 授权后，立即禁止新分派与领取；历史姓名由 Main API 从 IAM 读取并展示，既有活动工单标记“责任人已失效”，由组长或管理员明确改派，系统不得静默改写责任历史。

跨服务边界上，Main API 负责组合 IAM 候选人与 AIOps 责任组数据，并在写入前校验资格；AIOps Agent 服务只持久化稳定 UUID，不直接查询 IAM 表。App 成员变化应通过事务 Outbox/投影使责任组资格失效可追踪，消费者必须幂等。

### 5.3 自动分派与领取

自动建单只自动确定责任组，不从组内任意挑选个人。路由优先级为：

`显式路由规则 → Target 默认责任组 → Agent 默认责任组 → 未分派队列`

Target 可覆盖 Agent 的默认责任组，以支持同一 Agent 管理多个数据库但由不同 DBA 团队负责。配置变化只影响新工单，不静默迁移已有活动工单。

组内 DBA 通过“领取”动作将当前登录用户写为 `assignee_user_id`；手工创建工单可以默认由当前用户领取。跨用户指派和改派必须由有管理权限的用户执行。领取、改派、退回组队列和成员失效分别写入 `CLAIMED / REASSIGNED / RETURNED_TO_GROUP / ASSIGNEE_BECAME_INELIGIBLE` Activity。

成员移出责任组前，页面必须显示其活动工单数量，并要求选择替代责任人或将这些工单退回原责任组队列；历史 Activity 中的原责任人保持不变。

### 5.4 权限与 API

新增 `aiops:work_item_handle` 和 `aiops:work_item_manage`：AIOps 操作员拥有处理、领取本人可见工单的权限；AIOps 管理员同时拥有责任组维护、跨用户分派和改派权限。`aiops:member_manage` 继续只管理 AIOps App 成员与角色，不用来隐式授予工单治理权限。

公开 API 位于当前 Domain 的 `/ops` 下：

- `GET /responsibility-groups`
- `POST /responsibility-groups`
- `GET /responsibility-groups/{group_id}`
- `PATCH /responsibility-groups/{group_id}`
- `GET /responsibility-groups/{group_id}/member-candidates`
- `PUT /responsibility-groups/{group_id}/members/{user_id}`
- `DELETE /responsibility-groups/{group_id}/members/{user_id}`
- `POST /work-items/{work_item_id}:claim`
- `PATCH /work-items/{work_item_id}/assignment`

候选接口只返回当前 Domain 符合资格的已有 AIOps 用户。写接口使用 `If-Match`、`Idempotency-Key` 和服务端资格复核，不能信任前端传入的用户名、角色或 Domain 范围。

## 6. 一期 Portal

“DBA 工作项”是 AIOps 业务导航的一部分，列表页使用共享 `table-card`，详情说明使用共享 `key-value-card / key-value-table`，标题来自导航元数据。

队列包含“我的待办、未分派、紧急干预、待研判、恢复观察、已超 SLA、待验证、已完成”等可操作切片。主表显示优先级、处理模式、事项、实例、Agent 诊断摘要、告警当前状态、责任人、解决 SLA 和发生次数。详情显示处理概览、诊断发生记录、证据与建议动作、关联资源、恢复观察和活动时间线。

告警诊断详情增加“DBA 干预决策”区域，展示 `需要人工干预 / 人工调查 / 恢复观察 / 无需建单`、决策理由、优先级理由、关联工作项及其状态。告警诊断与工作项之间必须双向可导航。

工作项详情提供独立的“标记工作完成”动作，不要求 DBA 理解内部状态枚举。确认框明确说明将进入待验证，提交完成代码、完成说明和证据后写入审计时间线。

管理导航增加“责任组”页面，展示组名、状态、组长、有效成员数、Agent/Target 绑定数和未领取工单数；新建或编辑抽屉从已有 AIOps 用户中搜索并添加成员。它不得跳转到 Tenant 用户候选列表，也不得在此页面创建用户或授予 App 访问。

工作项详情先选择责任组，再从该组当前有效成员中选择责任人；普通 DBA 显示“领取”与“退回组队列”，管理员显示跨用户指派。失效成员保留姓名和审计标识，但不能再次被选择。状态变化使用共享确认框，不使用浏览器原生确认。

## 7. API 与事务边界

公开 Main API 位于 `/api/v1/apps/aiops/domains/{domain_code}/ops/work-items`，内部服务 API 位于 `/internal/v1/aiops/work-items`。公开层负责业务 App 权限，内部层要求服务凭据和短期 AuthContext；Main API 不直接访问 AIOps 表。

Run 完成时的诊断路由与工作项写入位于同一 AIOps Unit of Work。Situation 恢复发生在独立事务时，通过 Outbox 驱动恢复投影，不能跨服务同步拼接事务。巡检按 Target Run 路由，Inspection Fire 只负责多 Target 汇总，不能在 Fire 收敛时重复建单。Repository 不提交事务，分派、标记完成与状态变化使用 `row_version` 做乐观并发控制。

路由器必须保存决策原因和使用的 Situation/Finding 版本。并发的 Run 完成、恢复事件和重复 Webhook 必须得到同一逻辑结果；重放不能增加重复工作项或重复发生次数。

## 8. 告警诊断改造验收

至少覆盖以下验收场景：

1. Provider 原始 `critical` 和 `error` 均按规范 `CRITICAL` 进入同一决策链。
2. `OPEN + CRITICAL` 且 Agent 确认需人工处置时创建 P1/P2 工作项，并从告警诊断页可达。
3. `OPEN + CRITICAL` 但自动诊断失败或证据不足时创建人工调查事项，不静默丢失。
4. Run 完成前 Situation 已恢复时只生成 P3/P4 观察事项。
5. 已有干预事项收到恢复事件时只追加恢复事实，不生成第二张事项。
6. 重复告警、重复恢复事件和并发 Run 完成保持幂等；同一问题只保留一个活动聚合根。
7. DBA 标记工作完成必须保存完成人、完成时间、代码和说明；验证失败或问题复发时自动重开。
8. Tenant、Domain、Target 权限隔离以及跨来源归并均有契约和集成测试。
9. 自动建单按 Target、Agent 和未分派队列顺序选择责任组，不把 Agent 创建人或系统身份设为责任人。
10. 只有当前 Domain 的有效 AIOps 用户可以加入责任组、领取或被指派；成员失效和移组不破坏历史审计。

## 9. 日常巡检复用实现

### 9.1 现状

自动巡检已经复用工作项链路：Scheduler 创建 Inspection Fire，按 Target 启动 `trigger_type=SCHEDULE` 的 Run；单个 Run 全部任务完成后编译 Finding、发布巡检报告，并调用统一工作项路由器。`SCHEDULE` 映射为来源 `INSPECTION`，普通 Finding 映射为 `RISK_REMEDIATION`，数据缺口映射为 `OBSERVABILITY_GAP`。检查项适配器结合模板 `selected_check_ids`、Finding、Gap 与 Run 完成状态区分 `UNHEALTHY / HEALTHY / UNKNOWN`；健康证据只验证已经由 DBA 标记完成的 `PENDING_VERIFICATION` 事项，未完成事项只记录 `HEALTH_OBSERVED`，Gap 不得作为恢复或验证证据。

### 9.2 统一决策上下文与来源适配

统一路由器接收 `WorkItemDecisionContext`，由来源适配器提供当前状态、证据完整性和来源资源：

- 告警适配器把 Situation 最新状态转换为 `OPEN / RESOLVED`。
- 巡检适配器按每个已选择检查项输出 `UNHEALTHY / HEALTHY / UNKNOWN`，并携带 `inspection_fire_id`、Run、Report、检查项 ID、模板版本、观测窗口和证据引用。
- 统一决策器消费来源状态、Finding、确认度、Target 重要级别、重复次数与历史活动工作项，仍只输出 `ACTION_REQUIRED / MANUAL_INVESTIGATION / OBSERVE / NO_WORK_ITEM`。

检查项状态定义：

| 状态 | 判定条件 | 工作项行为 |
| --- | --- | --- |
| `UNHEALTHY` | 检查项实际执行，证据足够且命中异常规则 | 创建或归并风险处置事项 |
| `HEALTHY` | 检查项实际执行，证据完整且明确未命中异常规则 | 默认不新建事项；可作为既有事项的恢复或验证证据 |
| `UNKNOWN` | 未执行、超时、步骤失败、证据过期、缺列或覆盖不足 | 不得声明恢复；保留既有事项，并创建或归并可观测性缺口 |

`FindingCompilation.empty_reasons` 只有在对应检查项确实执行且证据完整时才能支持 `HEALTHY`。Finding 不存在、Run 为 `PARTIAL / WAITING_USER / FAILED` 或存在影响判断的 Gap，一律不能作为恢复证据。

### 9.3 建单、观察与优先级

- `UNHEALTHY` 且确认存在中断、数据损失、安全、阻塞或容量耗尽风险时进入 `ACTION_REQUIRED`。核心实例的紧急风险为 P1，其他已确认高风险为 P2。
- 证据为 `LIKELY`、影响尚需核验或高频重复时进入 `MANUAL_INVESTIGATION`，通常为 P2；可计划治理的容量、性能和配置风险为 P3。
- `UNKNOWN` 产生 `OBSERVABILITY_GAP`，通常为 P3/P4；若它阻断了核心实例重大风险判断，可提升为 P2 人工调查，但不得伪装成已确认故障。
- 单纯健康检查不创建“正常”工作项。只有与历史异常或恢复事件相关、且需要跨观察窗口持续承担责任时，才创建或归并 P4 `RECOVERY_OBSERVATION`；重复、核心实例或残余风险可为 P3。
- 同一物理问题在告警与巡检中使用来源无关的稳定指纹归并。实现时必须移除当前指纹中的 `work_type`，并用规范 `condition_code` 映射同义告警与巡检规则；来源、处理模式和报告链接只作为发生记录与审计属性。

### 9.4 完成、恢复与验证边界

- 巡检 `HEALTHY` 不等于 DBA 已完成。工作项仍必须由 DBA 执行“标记工作完成”，系统不得代替责任人填写完成代码或说明。
- 活动事项处于 `OPEN / IN_PROGRESS / WAITING` 时，后续健康巡检只追加恢复证据和观察窗口，不自动关闭或降级。
- 事项已由 DBA 标记为 `PENDING_VERIFICATION` 后，覆盖原问题条件且证据完整的健康巡检可以触发或充当 Verification；验证通过进入 `RESOLVED`，验证失败或异常复发回到 `IN_PROGRESS`。
- `UNKNOWN` 永远不能通过 Verification，也不能清除旧风险；应记录缺口并等待下一次有效巡检或专项验证。

### 9.5 触发时机与幂等

- 以单个 Target Run 完成为路由时机，在该 Run 的 Unit of Work 内写入决策和工作项。无需等待同一 Inspection Fire 下所有 Target 完成，避免一个失败 Target 阻塞其他数据库的紧急事项。
- Inspection Fire 收敛为 `COMPLETED / PARTIAL / FAILED` 时只汇总覆盖率和状态，不再次路由工作项。周期汇总报告也只链接已有 Occurrence，不重复消费原 Run Finding。
- Run 重试以 `ops_run_id + finding_id/check_id + decision_version` 幂等；同一 Fire、同一 Target、同一检查项不得增加重复 Occurrence。
- `PARTIAL` Run 可路由已确认异常与数据缺口，但所有未完整执行检查项必须为 `UNKNOWN`，不得生成健康或恢复决策。

### 9.6 巡检复用验收

至少覆盖以下验收场景：

1. 完整巡检发现已确认高风险时创建 P1/P2，计划治理风险创建 P3，并关联 Fire、Run、Report、检查项和证据。
2. 完整巡检明确健康且无历史事项时不建单，避免每次巡检制造“正常工单”。
3. 下一次巡检没有 Finding，但检查未执行、证据缺列或 Run 部分失败时判为 `UNKNOWN`，既有事项不关闭。
4. 告警与巡检发现同一问题时归并到一个活动工作项，同时保留两个来源的独立 Occurrence 和链接。
5. 后续健康巡检只为既有事项补充恢复或验证证据；未经过 DBA“标记工作完成”不得自动解决。
6. Fire 包含多个 Target 且部分失败时，成功 Target 的紧急事项正常创建，失败 Target 记录缺口，Fire 汇总不重复建单。
7. Run 重放、Fire 重放和周期报告生成均保持幂等，不重复增加事项或发生次数。

## 10. 二期设计（未实现）

### 10.1 评论与关注人

增加 `WorkItemComment` 与 `WorkItemWatcher`。评论只追加，编辑以修订表达；提及用户和关注人必须通过服务端成员候选校验。评论、分派和状态变化通过平台通知中心生成通知。

拟议 API：

- `POST /work-items/{id}/comments`
- `GET /work-items/{id}/comments`
- `PUT /work-items/{id}/watchers/{user_id}`
- `DELETE /work-items/{id}/watchers/{user_id}`

### 10.2 批量操作

只允许同 Domain、同预期状态且逐项鉴权的批量分派、优先级调整和状态推进。响应返回逐项成功与失败，每个事项单独记录 Activity，不以部分成功伪装整体成功。

### 10.3 Problem 管理

Problem 是跨 WorkItem 的长期根因聚合，保存已知错误、根因、临时规避、永久修复计划和关联事项。Problem 与 WorkItem 生命周期独立，关闭 Problem 不自动关闭工作项。

### 10.4 运营指标

从 Activity 与 SLA 字段派生 MTTA、MTTR、超时率、重开率、积压年龄、自动归并率和验证通过率。统计响应必须携带口径、时区、窗口、样本量与生成时间。

### 10.5 外部 ITSM 集成

通过 Outbox 异步连接器同步，不在工作项事务内调用外部系统。映射保存外部系统、票号、同步版本和最后结果，并具备幂等、重放、死信和冲突审计能力。

## 11. 二期进入开发的前置条件

必须先确定通知事件合同、Problem 状态机、指标口径和保留期限，并准备至少一个 ITSM 沙箱完成契约验证。责任组和 AIOps 成员候选属于当前工作项流程范围，不依赖二期功能。在这些条件满足前，不新增二期表、API、任务或 Portal 页面。
