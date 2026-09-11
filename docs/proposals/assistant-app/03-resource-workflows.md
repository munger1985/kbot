# 03 · Domain、KC、问数模型与 Agent 工作流

## 资源边界

Domain 是强隔离与授权边界，不是可由浏览器任意填写的标签。智能工作台虽允许创建 Domain，但创建动作必须委托平台的 Domain 生命周期用例：平台原子创建 Domain、建立 App-Domain 关系、创建初始授权范围，并返回新的业务 Token 上下文。App 不维护一份影子 Domain 表，也不能创建跨 Tenant 的 Domain。

首次部署先由 `scripts/db/initialize_assistant.py` 建立 `assistant_portal` 这个空白引导
Domain 和 `assistantadmin`。它只解决首次登录与 App 管理入口，不预置 KC、问数模型、
Agent、模型绑定或 OCI 配置；实际业务应在 App 内按下述流程新建业务 Domain 和资源。

```text
创建 Domain
  → 建立/验证 App-Domain 授权
  → 创建 Knowledge Core（可多个）
  → 创建并发布问数模型（可多个）
  → 创建 Agent
       ├─ 绑定一个 KC（必填）
       ├─ 绑定零至多个已发布问数模型（可选）
       ├─ 绑定路由/回答模型
       └─ 授予成员与角色后启用
```

首版一个 Agent 绑定一个 KC，避免单次问文跨 KC 后出现不透明的授权与引用混合。需要跨 KC 的场景应创建一个明确的 KC Collection Group 或新 Agent，不能在聊天页临时拼接 KC。

## Domain 页面

页面是表格与详情抽屉：名称、状态、KC 数、已发布问数模型数、Agent 数、成员范围和最后活动。创建 Dialog 至少需要名称、描述和初始管理员/角色范围；Domain ID 由服务端创建，不能手填。

创建成功后显示“切换到此 Domain”操作，且重新取得固定该 Domain 的业务 Token。停用前必须显示引用检查：仍启用的 Agent、KC、问数模型、进行中的 Run 和可访问媒体资产；不允许通过前端强行停用。

## Knowledge Core 页面

KC 页面复用现有 KC 的创建、文件/数据接入、解析、索引、Evidence 和引用能力。列表按当前 Domain 展示状态、Collection 数、Embedding、最近解析任务和被 Agent 引用数；详情页使用现有资源，而不是复制一套上传与检索流水线。

创建 KC 的向导：

1. 选择当前 Domain 并填写名称、用途、默认安全等级；
2. 选择当前可用的文本 Embedding，及可选视觉能力；
3. 创建后接入/上传受控资料，查看解析任务；
4. KC 处于 `ACTIVE` 且至少满足可用条件后，才可被 Agent 选择。

KC 模型变更、停用和删除遵守既有不可变 Revision、索引和反向引用规则。页面必须解释“被已启用 Agent 绑定时不能删除”，并跳转至关联 Agent，而不是隐去阻塞原因。

## 问数模型页面

“问数模型”是已发布的语义数据模型，不是 LLM 模型，也不是数据库连接。页面直接复用现有 Data Query 管理能力与生命周期：

```text
连接测试 → 创建数据源 → Schema 发现与对象选择 → 语义模型草稿
→ 测试问题 → 审核 → 发布 → Policy Binding → Agent Binding
```

列表显示名称、Domain、数据源、版本、`DRAFT/REVIEW/PUBLISHED/ARCHIVED` 状态、策略数和绑定 Agent 数。只有 `PUBLISHED` 版本可以在 Agent 向导中选取；草稿修改创建新版本，不能原地改变已运行 Run 所依赖的口径。

数据源凭据始终由 Data Query 加密存储。页面可展示连接测试成功、允许 Schema、快照时间和稳定错误摘要，但不能回显用户名、口令、完整 DSN 或原始驱动错误。

## Agent 页面

创建 Agent 用四步向导，而非一张堆满字段的表单：

1. **基本信息**：名称、描述、所属 Domain、指令和启用状态草稿；
2. **知识范围**：选择一个 `ACTIVE` KC；这是必填步骤；
3. **问数范围**：选择零至多个 `PUBLISHED` 问数模型，并检查每个模型已具备有效 Policy Binding；
4. **模型与授权**：选择模型目录中的 Router/回答模型、分配用户或角色、预览有效能力并启用。

能力预览由服务端计算，不由前端推断：

| 绑定状态 | 最终能力 |
| --- | --- |
| 仅 KC | 闲聊、问文 |
| KC + 已发布问数模型 | 闲聊、问文、问数及既有 Hybrid 路由 |
| 无 KC | 只能保存草稿，不能启用或出现在知识问答入口 |
| KC 停用、问数模型撤销发布、Policy 失效 | Agent 自动不可用或降级，用户获得明确原因 |

Agent 的运行版本冻结 KC、问数模型版本、模型引用与指令版本。后续编辑需创建草稿/新版本并进行显式启用，历史 Conversation 和 Run 不随之重写。
