# AIOps 统一智能助手设计讨论基线

版本：0.4

状态：核心合同与第一阶段运行链已落地，后续能力持续演进

日期：2026-10-19

适用：KBot4、Ammolite Cube

## 1. 文档定位

本文固定 AIOps 统一智能助手架构，并记录 KBot4 与 Ammolite Cube 的共同实现基线。
目标产品语义是长期约束；“当前实现状态”章节只陈述本次已经由代码与测试验证的范围。

配套基线：

- [AIOps 业务系统资源拓扑总体设计](../product/aiops-business-system-resource-topology.md)
- [AIOps 业务系统资源图谱详细设计](aiops-business-system-resource-graph-detailed-design.md)
- [AIOps Agent 专业 DBA 调查详细设计](../architecture/aiops-agent-chat-diagnosis.md)
- [知识问答能力架构](../architecture/knowledge-answering-capability.md)

当本文与既有“创建会话时必须选择并冻结单一 Target”的设计冲突时，以本文为准：用户面向业务系统
表达问题，Target 仅作为内部诊断执行端点，由每个 Turn 的范围解析过程确定。

## 2. 已固定的总体方案

### 2.1 一个用户可见 Agent

AIOps 只向用户呈现一个统一问答窗口和一个用户可见的 AIOps Agent。知识 Agent、诊断流程、
监控查询、报告分析和后续资源能力是 AIOps Agent 委派的专业能力，不分别建立互相割裂的用户会话。

```text
统一 AIOps 问答窗口
        │
        ▼
用户可见的 AIOps Agent
  会话上下文 / 权限 / Agent版本 / 资源边界
        │
        ▼
Turn Router
 ├─ 知识问答 ──► Knowledge Specialist ──► Knowledge Core / 知识 Agent
 ├─ 运行观测 ──► Resource Specialist  ──► Prometheus / Zabbix / OEM / Loki
 ├─ 故障诊断 ──► Diagnostic Specialist ─► 数据库诊断 / 报告 / 监控 / 知识证据
 ├─ 综合问题 ──► 多专业能力协同
 └─ 变更请求 ──► Change Specialist ─────► 风险评估 / 审批 / 执行 / 验证
        │
        ▼
统一证据层与 Answer Composer
        │
        ▼
带来源、引用、置信度和操作边界的统一回答
```

底层执行引擎不要求合并，但必须统一 Conversation、Turn、事件、证据、引用和最终回答合同。
每个 Turn 重新判断问题类型和所需资源，不能把整个 Conversation 永久固定为知识问答或诊断。

统一 Router 只允许下列受控能力，不提供通用问数，也不允许模型针对 AIOps 内部表生成 SQL：

| 能力 | 用户问题范围 | 数据边界 |
| --- | --- | --- |
| `RESOURCE_CATALOG_QUERY` | 系统、主机、数据库、IP、拓扑和资源清单 | 业务系统资源图的类型化字段 |
| `KNOWLEDGE_QA` | 运维手册、原理、操作方法 | 当前 Agent 知识策略内的已发布版本 |
| `MONITORING_QUERY` | 当前状态、固定指标和趋势 | 已配置监控源的固定 Profile/View |
| `ALERT_QUERY` | 当前及历史告警情境 | 已冻结资源范围内的 Situation |
| `INSPECTION_QUERY` | 巡检执行和结果 | 已存在的 Inspection Fire/报告，不触发新巡检 |
| `REPORT_CASE_QUERY` | 正式报告和历史诊断案例 | 报告服务及已发布 `DIAGNOSIS_CASE` |
| `ACTIVE_DIAGNOSIS` | 主动取证、定位根因 | 内部 Resource→Target 解析后调用诊断内核 |
| `CONTROLLED_CHANGE` | 明确的变更请求 | 资源级白名单、审批、执行与验证链 |

### 2.2 知识 Agent 的定位

运维知识是 AIOps Agent 的专业能力，不是第二个用户可见助手：

- 纯知识问题委派给 Knowledge Specialist，并要求原文引用；
- 诊断问题由 Diagnostic Specialist 主导，运维知识作为解释、操作前提和历史案例证据；
- 综合问题可以并行取得实时事实与知识证据，再由统一回答层组合；
- 其他业务知识库通过受控的 Knowledge Retrieval Agent 绑定接入；
- 菜单中的“知识问答”可以保留为统一窗口的知识优先入口，但不再维护独立会话体系。

### 2.3 模型角色必须产品化配置

模型按语义角色配置，不在部署 TOML 中隐藏固定 Agent UUID 或模型 ID。

知识治理模型属于 Domain/App 级模型 Profile：

| 产品用途 | 模型角色 |
| --- | --- |
| 标签识别 | `tag_classifier_llm` |
| 诊断案例提炼 | `case_extractor_llm` |
| 后续离线治理任务 | 独立语义角色 |

Agent 运行时模型属于版本化 AIOps Agent：

| 产品用途 | 模型角色 |
| --- | --- |
| 问题理解、范围解析和任务规划 | `planner_llm` |
| 故障推理 | `diagnosis_llm` |
| 最终回答组织 | `composer_llm` |
| 会话摘要 | `memory_llm` |
| 长期记忆检索 | `memory_embedding` |
| 图片证据 | OCR / VLM 能力 |

同一模型可以承担多个角色，但角色必须分别声明。标签识别属于知识治理流程，不能再依赖知识问答
Agent 的 `composer_llm`。Agent 启用前根据已启用能力确定性检查所需模型和资源。

### 2.4 统一证据与安全边界

最终回答必须区分：

- 实时系统或工具观测；
- 用户提供材料；
- 运维知识原文引用；
- 模型推断。

知识证据不能冒充实时状态，模型推断不能冒充系统事实。变更请求始终经过独立风险策略、人工审批、
执行和验证，不因统一问答入口而扩大自动执行权限。

## 3. 本轮固定：Agent 绑定业务系统，而不是 Target

### 3.1 产品语义

用户配置 AIOps Agent 时选择其允许访问的一个或多个业务系统，不直接选择内部 Target。

```text
AIOps Agent
└─ 允许访问的业务系统
   ├─ MES 生产系统
   │  ├─ Oracle RAC 节点 1 → 内部 Target A
   │  ├─ Oracle RAC 节点 2 → 内部 Target B
   │  └─ Data Guard 备库   → 内部 Target C
   └─ 其他获授权业务系统
```

业务系统是授权、自然语言理解和用户展示边界；Managed Resource 是业务资源身份；Target 只是数据库
连接、监控映射、只读取证和受控执行所需的内部适配端点。公共页面和普通对话不要求用户理解 Target。

Agent 版本冻结 `allowed_business_system_ids` 及能力策略。新 Turn 开始时读取这些系统当前已启用的资源图，
形成当轮有效候选并保存不可变范围快照。业务系统或资源随后被停用时，不影响历史审计，但必须阻止新
Turn 再使用已失效资源。

系统级授权允许 Agent 发现和诊断该系统内当前有效、能力就绪的资源；受控变更仍必须按具体资源或
Target 显式配置动作白名单和审批策略，不能因绑定业务系统自动取得变更权限。

### 3.2 Conversation 与 Turn 边界

Conversation 绑定 Domain、用户可访问的 AIOps Agent 版本和持续上下文，不再强制绑定单一 Target。

- Conversation 可以保存最近确认的业务系统作为后续问题的上下文提示；
- 每个 Turn 独立解析业务系统和资源范围；
- 用户明确提到另一个获授权系统时，可以在同一 Conversation 的新 Turn 中切换；
- 用户未提系统时，可以继承最近一次已确认范围，但必须在事件和回答中明确展示；
- 纯知识问题可以不解析业务系统和 Target；
- 一期每个诊断 Turn 只解析一个业务系统，但允许该系统内包含多个资源和多个 Target；
- 跨系统比较作为后续能力，不通过隐式扩大范围实现。

### 3.3 诊断范围不是单一 Target

每个诊断 Turn 的范围解析结果必须是以下语义之一：

| 范围 | 含义 | 示例 |
| --- | --- | --- |
| `SYSTEM` | 整个业务系统的有界初筛 | “MES 系统现在有性能问题” |
| `TOPOLOGY_GROUP` | 一个拓扑关系组 | “MES 主库 RAC 性能怎么样” |
| `RESOURCE_SET` | 多个明确资源 | “比较 MES 主库和备库延迟” |
| `RESOURCE` | 一个明确资源 | “检查 MES 的 DG 备库” |
| `NONE` | 不需要运行资源 | 纯知识问答 |

不能因为底层诊断器按单一 Target 执行，就在用户输入阶段强迫选中一个 Target。系统级或拓扑组级问题
先由协调层做有预算的多资源初筛，再自动聚焦到需要深入诊断的资源。

## 4. 自然语言范围解析

### 4.1 解析输入

范围解析器只在以下可信候选中工作：

1. 当前用户有权使用的 AIOps Agent；
2. Agent 当前版本允许访问的业务系统；
3. 这些系统中状态有效的资源和已确认关系；
4. 满足本轮能力要求的内部 Target、监控绑定和诊断源。

模型不能生成并直接使用业务系统 ID、资源 ID 或 Target ID。模型只能从服务端给出的有界候选中选择，
最终由确定性策略再次校验 Domain、用户权限、Agent 授权、资源状态和能力状态。

业务系统至少需要规范名称、系统编码和显式别名。资源候选可以使用资源名称、资源编码、数据库唯一名、
服务名、拓扑角色和已确认关系。别名必须作为可治理数据维护，不能只依赖描述字段或模型模糊猜测。

### 4.2 分阶段解析

```text
用户问题
  ↓
识别业务系统提及
  ↓
在 Agent 允许范围内匹配名称、编码、别名和会话上下文
  ↓
读取该系统的资源图、RAC/DG 关系、角色和能力状态
  ↓
识别资源、角色、问题类型和时间范围提示
  ↓
确定 SYSTEM / TOPOLOGY_GROUP / RESOURCE_SET / RESOURCE
  ↓
自动执行、先做系统初筛，或进入交互式确认
```

系统匹配优先使用确定性名称、编码和别名；模型用于理解“主库”“备库”“RAC 第二节点”“生产库”等
业务表达和问题目标。LLM 置信度不能单独决定授权或执行范围。

### 4.3 决策状态

范围解析结果至少包含：

- `RESOLVED`：已有明确且可执行的范围；
- `SYSTEM_TRIAGE`：系统明确，但具体异常资源未知，先做多资源轻量初筛；
- `NEEDS_CLARIFICATION`：存在多个同等合理候选，继续执行可能产生错误结论；
- `NO_ELIGIBLE_RESOURCE`：系统存在，但没有满足本轮能力要求的资源；
- `NOT_FOUND_IN_ALLOWED_SCOPE`：在用户和 Agent 的允许范围内没有找到对应系统。

禁止用“猜测第一个候选”代替澄清，也不能向用户泄露未授权系统是否存在。

## 5. MES RAC + DG 示例

用户提问：

> 帮我查一下 MES 系统现在有性能问题。

处理过程：

1. 在 Agent 获授权系统中将“MES”唯一解析到 MES 生产系统；
2. 读取业务系统资源图，识别两个 RAC 成员和一个 DG 备库；
3. 因用户描述的是系统级性能问题，不立即要求用户从三个 Target 中选择；
4. 对三个资源执行有预算的轻量初筛，包括可用性、关键性能指标、数据库角色、复制状态和近期异常；
5. 如果证据表明 RAC 节点 2 异常，则自动把深入诊断聚焦到该资源，同时保留 RAC 与 DG 上下文；
6. 最终回答说明检查了哪些资源、为何聚焦、哪些证据支持结论，以及未检查或不可用的范围。

需要交互确认的示例：

- MES 中存在两套互不相关的数据库集群，而问题只说“数据库”；
- “备库”对应多个 DG Standby，且没有名称、站点或错误码提示；
- 监控身份与资源图角色冲突，无法安全确定主库；
- 用户要求执行高风险动作，但没有明确作用资源。

此时进入 `WAITING_USER`，展示用户可理解的候选项，例如：

```text
MES 生产系统中发现两个可能范围：
1. 订单库 RAC（2 个节点，当前主业务库）
2. 报表库单实例

你希望诊断哪一个？也可以选择“先检查全部数据库”。
```

用户选择结果保存为 `USER_CONFIRMED` 范围证据，并在同一个 Turn 中恢复执行，不要求重新创建会话
或重新提交原问题。

## 6. 多 Target 诊断执行模型

Conversation 和 Turn 是面向用户的协调根；现有单 Target 诊断 Run 保持为内部执行单元。

```text
Conversation
└─ Turn：MES 系统性能问题
   ├─ Scope Resolution：MES / SYSTEM_TRIAGE
   ├─ Triage Delegation
   │  ├─ Target Run A：RAC 节点 1
   │  ├─ Target Run B：RAC 节点 2
   │  └─ Target Run C：DG 备库
   ├─ Focused Delegation：RAC 节点 2 深入诊断
   ├─ Knowledge Delegation：相关手册与历史案例
   └─ Unified Answer：跨资源证据、结论和建议
```

协调层负责范围、预算、并行度、拓扑上下文、子运行状态和最终证据汇总。单 Target Run 继续负责该
Target 的监控、数据库只读工具、评估和可审计 Artifact。父 Turn 使用唯一 `RESOURCE_nnn` 关系绑定
每个子 Run；子 Run 不直接写父 Turn 的调查 Revision、Tool Invocation 或最终回答，避免并发覆盖。
任何一个子运行失败都不应抹掉其他资源的有效结果；最终 Turn 可以以 `PARTIAL` 返回并明确数据缺口。

当前一期执行语义如下：

1. 单资源继续走既有 `PRIMARY` Turn Run 和原生调查规划链；
2. 多资源按 `ExecutionBudgetV1.max_diagnosis_runs` 创建幂等的
   `RESOURCE_DIAGNOSIS_CHILD` Run，默认最多 4 个；
3. 子 Run 创建时与父 Turn 原子建立关系，Outbox 重放不会重复创建 Run 或回答；
4. 父 Turn 等待全部已启动子 Run 到达 `COMPLETED`、`PARTIAL`、`FAILED`、`CANCELLED` 或
   `EXPIRED` 后聚合，不因先完成的单个资源提前结束；
5. 全部预算内资源成功且证据完整时父 Turn 为 `COMPLETED`；部分失败、子结果不完整、证据缺失或
   超预算截断时为 `PARTIAL`；全部失败时为 `FAILED`；
6. 用户取消父 Turn 时取消其全部未终态子 Run；公开答案只展示业务资源名称、拓扑角色与证据引用，
   不展示内部 Target ID。

目标态的系统级初筛必须使用低成本能力，不能无条件对全部 Target 执行完整深度诊断。演进流程为：

1. 读取已有告警、监控摘要、角色和连接状态；
2. 根据异常证据选出少量焦点资源；
3. 只对焦点资源执行深度数据库诊断；
4. 达到预算、证据充分或需要用户确认时停止扩展。

RAC 和 DG 关系必须进入范围快照。回答不能把 RAC 节点的实例级事实误写成整个数据库集群事实，
也不能把 Standby 的正常只读或应用行为误判为主库异常。

## 7. 范围快照与审计

每个 Turn 保存不可变的 `AIOPS_SCOPE_RESOLUTION.v1`，至少包括：

- Agent ID 与 Agent Version ID；
- 解析出的业务系统及其当时状态；
- 原始系统提及、上下文来源和匹配依据；
- 范围类型和候选资源；
- 资源到内部 Target 的映射；
- RAC、DG 等相关关系与参与角色；
- 用户确认记录；
- 被能力、状态或权限过滤掉的数量和安全原因代码；
- 最终选定范围、置信度、策略版本和快照 Hash。

UI 展示业务系统、资源名称和拓扑角色，不展示内部 Target ID。运维审计可通过受控详情查看内部映射。

## 8. 交互原则

- 默认输入框不显示 Target 下拉框；
- 从业务系统详情进入聊天时，业务系统作为可信入口上下文，但用户仍可在问题中明确切换；
- 自动解析后，在回答头部展示“本轮范围：MES 生产系统 → 订单库 RAC → 节点 2”；
- 系统级初筛期间展示正在检查的资源数量和阶段，不暴露连接串或技术凭据；
- 歧义确认优先使用带环境、拓扑角色和资源说明的按钮，同时允许用户补充自然语言；
- 用户确认后恢复原 Turn，不生成重复问题和重复 Run；
- 低置信度、角色冲突和高风险动作必须确认，不能用模型猜测继续。

## 9. 当前实现状态

KBot4 与 Ammolite Cube 已同步完成：

1. Agent 版本以 `allowed_business_system_ids`、`capabilities`、`knowledge_policy` 和资源级动作策略冻结授权；
2. Conversation 不再保存或要求单一 Target，聊天入口只选择 Agent；
3. 每个 Turn 先生成 `TurnPlanV1`，再形成不含内部 Target 的 `ScopeResolutionV1`；
4. 多系统歧义保存 `ClarificationRequestV1`，候选或自由文字回答在同一 Turn 幂等恢复；
5. 资源、知识、监控、告警、巡检、报告/案例均通过受控 specialist 查询，纯查询不创建 Ops Run；
6. 知识能力使用当前 Agent 版本的知识策略和 `composer_llm`，并校验引用，不再依赖全局知识助手；
7. `ACTIVE_DIAGNOSIS` 将公开资源范围解析为内部 Target：单资源桥接既有 `PRIMARY` 诊断链，多资源
   由父 Turn 启动默认最多 4 个幂等子 Run，并在全部子 Run 终态后聚合系统级回答；
8. UI 展示能力计划、公开资源范围和澄清交互，不把 Target 作为会话前置选择；
9. 公共 Agent 投影不暴露内部 Target ID，旧 Target-Agent 管理入口已从公共 Main API 删除；
10. Oracle 与 PostgreSQL 的规范 Schema、Entity、初始化工件和跨项目验收已经同步更新；
11. 多资源本身不再构成 `PARTIAL`：仅子 Run 失败、子结果不完整、关键证据缺失或预算截断才降级，
    全部预算内资源失败则返回 `FAILED`。

本次没有重建或迁移已部署数据库；规范模型的落地需要按各项目既定部署流程完整重建。

## 10. 尚未完成的目标能力

以下内容仍属于后续实施范围，不能由第一阶段结果推断为已经具备：

1. 基于告警和轻量监控异常评分先筛选 Top-N 资源、再只对焦点资源执行深度诊断；当前一期是在预算内
   对全部候选资源复用单 Target 深度诊断内核，尚未实现轻量初筛后的自动聚焦；
2. `CONTROLLED_CHANGE` 作为统一 Router specialist 的完整委派；现有审批、执行和验证内核仍保持独立边界；
3. 业务系统和资源的可治理别名，以及主机/IP/服务地址的正式类型化网络地址模型；
4. 独立运维知识问答历史会话向统一 Conversation 的产品迁移；
5. 长期记忆、`memory_embedding` 检索和跨会话范围继承策略；
6. 多业务系统比较、RAC/DG 组级差异化预算策略及跨 Worker 的大规模并行限流；
7. 多资源子 Run 需要人工补证时的父 Turn 统一交互编排；当前子 Run 会保持等待状态，尚未形成一个
   合并后的父级补证请求。

## 11. 后续讨论清单

1. 新加入业务系统的资源是否立即纳入 Agent 只读范围，以及管理员变更提示；
2. 资源别名和网络地址的治理合同；
3. 多资源轻量初筛的指标集合、异常评分、Top-N 聚焦、并发和停止条件；
4. 告警、巡检、报告和业务系统详情入口的可信上下文优先级；
5. 受控变更如何消费同 Turn 诊断证据并保持逐条审批；
6. 独立知识入口和历史会话的迁移顺序。

## 12. 当前验收方向

- Agent 管理只要求选择允许访问的业务系统，不要求逐个选择内部 Target；
- 用户输入“MES 系统性能问题”可以在获授权 MES 范围内自动发现 RAC 与 DG 资源；
- 系统级问题先做有预算初筛，不因存在三个 Target 就立即打断用户；
- 问题明确指向主库、备库、节点或资源名时可以直接解析到正确范围；
- 候选确实无法区分时进入交互确认，并在同一 Turn 中继续；
- 未授权系统和资源不会进入模型候选、提示选项、日志正文或回答；
- 多 Target 部分失败仍返回可信的部分结果和明确缺口；
- 回答展示业务系统、资源和拓扑角色，不把内部 Target 作为用户概念；
- 所有子运行、范围决策、用户确认和最终引用可以追溯到同一 Turn；
- 纯知识问题不要求业务系统或 Target，诊断和变更问题按能力逐步收紧范围。
