# AIOps DBA 工作项中心详细设计

## 1. 决策与阶段边界

DBA 工作项中心用于承接告警分析、日常巡检和人工诊断后需要持续处理的事项。它不是另一套报告系统：Finding 表达问题事实，Report 保存冻结输出，WorkItem 承载责任人、SLA、状态和审计，Proposal 承载审批、执行与回滚，Verification 承载处理效果。

一期已经实现以下能力：

- 按业务 Domain 隔离的活动工作队列、详情和 SLA 摘要。
- 人工创建、责任组与责任人分派、受控状态流转。
- Run 完成后从 Finding 自动生成工作项，重复问题归并发生记录。
- 关联 Run、Situation、Report、Proposal 和 Verification。
- 完整活动时间线和乐观并发版本控制。

二期只保留本文设计，不在本阶段实现：评论、关注人、批量操作、Problem 管理、运营指标和外部 ITSM 集成。

## 2. 一期领域模型

### 2.1 聚合结构

- `WorkItem`：工作项聚合根，保存优先级、责任、SLA 和生命周期。
- `Occurrence`：相同问题在不同 Run 中再次出现的事实记录，保存 Finding 快照和证据引用。
- `Link`：工作项与 Run、Situation、Report、Proposal、Verification 的有类型关联。
- `Activity`：创建、归并、分派、状态变化和验证完成的不可变审计记录。

### 2.2 状态机

主流程为：

`PENDING_TRIAGE → OPEN → IN_PROGRESS / WAITING → PENDING_VERIFICATION → RESOLVED → CLOSED`

约束：

- `WAITING` 必须填写等待原因。
- `RESOLVED` 和 `CLOSED` 必须填写解决代码和说明。
- Verification 完成只把非终态事项推进到 `PENDING_VERIFICATION`，不能自动关闭。
- 已解决事项再次出现时回到 `OPEN / DIAGNOSIS`，增加重开次数并清除旧解决结论。
- `CLOSED` 与 `CANCELLED` 是终态，不参与默认活动队列和自动归并。

### 2.3 SLA

| 优先级 | 确认时限 | 解决时限 | 验证时限 |
| --- | ---: | ---: | ---: |
| P1 | 15 分钟 | 4 小时 | 8 小时 |
| P2 | 30 分钟 | 8 小时 | 12 小时 |
| P3 | 4 小时 | 72 小时 | 96 小时 |
| P4 | 8 小时 | 168 小时 | 216 小时 |

默认队列排除终态事项，按解决时限升序、工作项 ID 升序排列，使最先到期事项稳定地排在前面。

## 3. 自动生成与归并

### 3.1 路由规则

- `CRITICAL / HIGH` Finding 自动建立 `OPEN` 工作项。
- `MEDIUM` Finding 自动建立 `PENDING_TRIAGE` 工作项。
- 数据质量 Gap 建立 `OBSERVABILITY_GAP / PENDING_TRIAGE` 工作项。
- Chat Run 默认不自动建单；用户显式路由 Run 时可以选择 Finding。
- Verification Run 只关联并推进既有事项，不独立制造“已解决”结论。

### 3.2 稳定指纹

指纹输入为：

`tenant/domain + target_id + work_type + finding_type + normalized_object_ref + condition_code`

对象引用按规范 JSON 序列化后计算 SHA-256。相同指纹且非终态的工作项复用聚合根，新 Run 只新增 Occurrence、Link 和 Activity。业务唯一性由应用事务负责，数据库不使用业务 `UNIQUE` 约束。

## 4. 一期页面与交互

### 4.1 工作队列

页面以紧凑表格为主体，显示优先级、事项、数据库实例、状态、责任人、解决 SLA 和发生次数。顶部摘要用于进入“我的待办、未分派、P1/P2、已超 SLA、待验证”五个业务切片，不使用无操作价值的装饰性 KPI 卡片。

### 4.2 工作项详情

详情按四组信息组织：

1. 处理概览：实例、责任、阶段、SLA 和解决结论。
2. 诊断证据：每次 Occurrence 的 Finding 快照与证据引用。
3. 资源链：Run、Situation、Report、Proposal 和 Verification。
4. 活动时间线：操作者、时间、前后状态和业务说明。

责任分派要求 `aiops:member_manage`。状态变化使用预定义状态机和行版本，冲突时要求刷新，不覆盖他人更新。

## 5. 数据与 API

KBot4 Oracle 规范表：

- `KBOT_OPS_WORK_ITEM`
- `KBOT_OPS_WORK_ITEM_OCCURRENCE`
- `KBOT_OPS_WORK_ITEM_LINK`
- `KBOT_OPS_WORK_ITEM_ACTIVITY`

公开 Main API 位于 `/api/v1/apps/aiops/work-items`，服务内部 API 位于 `/internal/v1/aiops/work-items`。API 提供列表、详情、创建、分派、状态流转和显式 Run 路由；所有查询继续受 Domain 与用户授权边界约束。

## 6. 二期设计（未实现）

### 6.1 评论与关注人

增加 `WorkItemComment` 和 `WorkItemWatcher`。评论只追加，不原地覆盖；编辑以修订记录表达。提及用户和关注人由服务端成员候选范围校验，新增评论与状态变化通过平台通知中心投递。

拟议 API：

- `POST /work-items/{id}/comments`
- `GET /work-items/{id}/comments`
- `PUT /work-items/{id}/watchers/{user_id}`
- `DELETE /work-items/{id}/watchers/{user_id}`

### 6.2 批量操作

仅支持同 Domain、同预期状态且逐项通过权限校验的批量分派、优先级调整和状态推进。服务端返回逐项结果，禁止“部分失败但前端显示全部成功”。每个工作项仍写独立 Activity。

### 6.3 Problem 管理

Problem 是跨多个 WorkItem 的长期根因聚合，不替代 WorkItem。它保存已知错误、根因、临时规避、永久修复计划和关联事项；关闭 Problem 不自动关闭关联事项。

### 6.4 运营指标

指标从 Activity 与 SLA 字段派生，包括 MTTA、MTTR、超时率、重开率、积压年龄、自动归并率和验证通过率。统计接口返回口径、时区、窗口和样本量，避免只有数字没有定义。

### 6.5 外部 ITSM 集成

采用 Outbox 驱动的异步连接器，不在工作项事务内同步调用外部系统。映射保存外部系统、外部票号、同步版本和最后结果；冲突策略默认“责任与状态以指定主系统为准、证据与活动双向追加”，并提供重放和死信处理。

## 7. 二期进入开发的前置条件

二期只有在以下条件满足后才能实施：成员候选接口确定、通知事件合同批准、Problem 状态机评审完成、至少一个 ITSM 沙箱可用于契约测试，并明确指标口径及数据保留期限。任何二期表、API、后台任务或页面都不在一期部署范围内。
