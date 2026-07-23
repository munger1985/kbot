# 4.0 AIOps 步骤 3：配置与权限 API

## 目标与边界

本步骤只交付 Target、Agent Binding、Monitor Source/Binding、Policy 和 Inspection Plan 配置闭环，不创建 Ops Run，不调用 LLM，不查询监控指标或目标数据库。外部 `/api/v1/ops/*` 由 Main API/BFF 发布；AIOps API 仅暴露 `/internal/v1/aiops/config/*`，两者通过 `AIOpsManagementClient` 和版本化 DTO 映射，不能透传 HTTP 请求。

所有资源 ID 是 UUIDv7。Oracle Entity 使用 `RAW(16)`，DTO 使用 `uuid.UUID` 并序列化为规范字符串。`app_id/domain_id/actor` 只从已验证 `AuthContext` 与平台配置派生，请求体不得提交。

## 资源与 API

### Target

```text
POST   /api/v1/ops/targets
GET    /api/v1/ops/targets
GET    /api/v1/ops/targets/{target_id}
PATCH  /api/v1/ops/targets/{target_id}
POST   /api/v1/ops/targets/{target_id}/activate
POST   /api/v1/ops/targets/{target_id}/maintenance
POST   /api/v1/ops/targets/{target_id}/disable
```

创建后默认 `MAINTENANCE`、`HEALTH_STATUS=UNKNOWN`，避免未验证 Target 立即被自动流程使用。管理状态限定为 `ACTIVE/MAINTENANCE/DISABLED`；连通性单独使用 `UNKNOWN/HEALTHY/DEGRADED/UNREACHABLE`，不能用 `OFFLINE` 混合表达配置与运行事实。

Target 不提供在线 `DELETE`。从未产生 Run 的误配置也通过 `DISABLED` 保留审计；开发空库清理仅由离线 migration/reset 工具完成。`target_key` 创建后不可改，Display、Endpoint、SecretRef、能力和执行模式可以带版本修改。

### Agent Binding

```text
GET    /api/v1/ops/targets/{target_id}/agent-bindings
POST   /api/v1/ops/targets/{target_id}/agent-bindings
PATCH  /api/v1/ops/targets/{target_id}/agent-bindings/{binding_id}
POST   /api/v1/ops/targets/{target_id}/agent-bindings/{binding_id}/revoke
POST   /api/v1/ops/targets/{target_id}/agent-bindings/{binding_id}/restore
```

Binding 是带 Access Mode、Policy、Change Window、额度、状态和审计的关联实体，因此有自己的 UUIDv7 `binding_id`，但没有第二套数字 ID。唯一 `(target_id, agent_id)`；重复创建返回现有资源或 `409`，不能产生两条当前 Binding。

创建前由 AIOps Application Service 在事务外调用 `AgentRuntimeClient`，确认 Agent 存在、属于同一 Domain、状态可用且声明 AIOps 能力；随后在事务内重查 Target 版本并创建 Binding。AIOps 不查询 Agent 表，也不把远端验证放在数据库事务中。

### Monitor Source 与 Binding

```text
POST   /api/v1/ops/monitor-sources
GET    /api/v1/ops/monitor-sources
GET    /api/v1/ops/monitor-sources/{source_id}
PATCH  /api/v1/ops/monitor-sources/{source_id}
POST   /api/v1/ops/monitor-sources/{source_id}/enable
POST   /api/v1/ops/monitor-sources/{source_id}/disable
POST   /api/v1/ops/monitor-sources/{source_id}/health-checks
POST   /api/v1/ops/monitor-sources/{source_id}/webhook-key:rotate

GET    /api/v1/ops/targets/{target_id}/monitor-bindings
POST   /api/v1/ops/targets/{target_id}/monitor-bindings
PATCH  /api/v1/ops/targets/{target_id}/monitor-bindings/{binding_id}
POST   /api/v1/ops/targets/{target_id}/monitor-bindings/{binding_id}/disable
POST   /api/v1/ops/targets/{target_id}/monitor-bindings/{binding_id}/enable
```

Monitor Binding 有外部对象映射、优先级、指标覆盖、健康状态和审计，是独立配置实体，使用 UUIDv7 `binding_id`；唯一 `(source_id, external_target_key)`，避免同一外部对象映射多个 Target。一个 Target 仍可绑定该 Source 下多个不同对象。绑定时 Target 与 Source 必须属于同一 Domain。

新 Monitor Source 默认 `DISABLED/UNKNOWN`；完成 SecretRef/Endpoint 校验以及至少一次健康检查后再显式启用。仅 Webhook Source 可用一次已验证的签名测试事件代替主动健康检查。

健康检查是异步 Command：事务内写 `HEALTH_CHECK_REQUEST_ID/REQUESTED_AT` 和 Outbox，返回 `202`。Worker 在事务外解析 SecretRef 并连接 Provider，再以 Request ID + Config Row Version + Health Version 条件更新健康结果；旧检查结果不得覆盖新请求。`ROW_VERSION` 只保护配置 ETag，健康写入递增独立 `HEALTH_VERSION`，不会让后台检查持续使管理页面 ETag 失效。GET Source 返回最近结果以及 `health_check_pending` 派生字段。

### Policy

```text
POST   /api/v1/ops/policies
GET    /api/v1/ops/policies
GET    /api/v1/ops/policies/{policy_id}
POST   /api/v1/ops/policies/{policy_id}/activate
POST   /api/v1/ops/policies/{policy_id}/retire
```

Policy 内容不可 PATCH。创建相同 `policy_key` 的新版本时，服务端在锁定该 Key 后分配下一个 `version_no`，初始状态为 `DRAFT`。激活事务必须校验规则 Schema/Hash、将原 Active 版本设为 `RETIRED`、激活候选版本并写 Audit/Outbox；同一 Scope/Key 始终只有一个 Active。

Policy 增加 `ROW_VERSION` 和生命周期更新时间，用于激活/退役的 ETag；Rules 内容、Version 和 Hash 创建后不可变。被 Binding 或 Run Snapshot 引用的版本永不物理删除。

### Inspection Plan

```text
POST   /api/v1/ops/inspection-plans
GET    /api/v1/ops/inspection-plans
GET    /api/v1/ops/inspection-plans/{plan_id}
PATCH  /api/v1/ops/inspection-plans/{plan_id}
POST   /api/v1/ops/inspection-plans/{plan_id}/activate
POST   /api/v1/ops/inspection-plans/{plan_id}/pause
POST   /api/v1/ops/inspection-plans/{plan_id}/disable
POST   /api/v1/ops/inspection-plans/{plan_id}/targets
PATCH  /api/v1/ops/inspection-plans/{plan_id}/targets/{plan_target_id}
```

新 Plan 默认 `PAUSED`。激活前必须至少有一个 Active Target、合法 IANA 时区与规范五段 Cron、已登记模板/Resolver 版本、合法 `MISFIRE_POLICY/OVERLAP_POLICY` 和可计算的 `next_run_at`。有效 Target 数不得超过部署级 `max_targets_per_inspection_fire`；模板覆盖只能修改白名单阈值、窗口或可选 Check。Plan Target 是带覆盖配置、状态和审计的关联实体，使用独立 UUIDv7 ID；Plan/Target 必须同 Domain。步骤 3 只维护配置，直到步骤 10 才由 Scheduler 领取。

## PATCH、ETag 与幂等

配置响应返回 `row_version`，单资源 GET/PATCH/Command 使用强 ETag：

```text
ETag: "rv-7"
If-Match: "rv-7"
```

缺少 `If-Match` 返回 `428 PRECONDITION_REQUIRED`，版本不符返回 `412 ROW_VERSION_CHANGED`。PATCH 使用显式 Pydantic Partial DTO，不采用任意 JSON Patch；未出现字段表示不修改，显式 `null` 只允许清除 Schema 标记为可空的字段。

所有 POST Create、状态 Command、Key Rotation 和 Health Check 要求 `Idempotency-Key`。作用域为 `principal + domain + operation + parent_resource`；保存规范请求指纹和结果引用。相同 Key/同指纹返回原结果，不同指纹返回 `409 IDEMPOTENCY_CONFLICT`。

列表采用稳定 Keyset Cursor，默认排序为 `(updated_at DESC, id DESC)`。Cursor 是带签名的不透明值，包含最后排序键、过滤器 Hash、Domain/调用方 Hash、契约版本和短期过期时间；改变过滤器、Domain 或调用方后不得复用。

## 当前身份边界与执行能力

4.0 当前阶段不实现 Scope、角色或 Target ACL。Main API 校验门户 API Key，并把门户声明的 Domain 和操作人写入内部 AuthContext；AIOps 的所有配置读写必须限定在该 Domain。以下能力矩阵是业务与执行安全约束，不是用户权限模型：

| 操作 | 当前身份要求 | 附加校验 |
| --- | --- | --- |
| Target/Binding 读取和管理 | 已认证 Portal Client | AuthContext Domain |
| 启用 `AGENT_EXECUTE` | 已认证 Portal Client + 操作人留痕 | Execution SecretRef + Policy + 部署 Kill Switch |
| Monitor 管理/轮换 | 已认证 Portal Client + 操作人留痕 | Domain + Provider 类型允许 |
| Policy 读取/激活 | 已认证 Portal Client + 操作人留痕 | Domain + 规则 Schema/风险上限 |
| Inspection 管理 | 已认证 Portal Client + 操作人留痕 | Domain + Target 状态 |

Agent 的有效运行能力不是 Binding 单值，而是：

```text
Agent–Target Binding Access Mode
∩ Target Execution Mode
∩ Active Policy Decision
∩ 部署级 Capability/Kill Switch
```

`OBSERVE < DIAGNOSE < PROPOSE < EXECUTE` 是能力上限，不代表自动获批。即使结果为 `EXECUTE`，每条 Mutation 仍须独立 Proposal 和一次审批。客户端提交的 Access Mode 只能缩小管理者有权授予的能力。

跨 Domain 和不存在统一返回 `404 OPS_RESOURCE_NOT_FOUND`。审批还必须匹配待审记录中的 `asserted_user_id`。未来增加细粒度权限时，在 AuthContext 中扩展版本化 Scope，并重新启用操作类别与资源 ACL 求交，不改变当前 API 路径。

## Endpoint、SecretRef 与 Webhook Key

- Endpoint DTO 按 `db_type/source_type` 使用判别联合，不接受任意 URL 字符串；Host、Port、TLS、Service Name 等字段分别校验，并由部署 egress policy 再限制网络目标。
- `*_secret_ref` 只接受配置中允许的 Provider/Scheme 和规范路径；写入前通过 Secret Provider 做元数据/访问性检查，不读取或回显 Secret Value。
- Secret Store 不可用时不把未验证引用持久化；暂时性失败返回 `503`，请求可用相同 Idempotency Key 重试。
- Webhook Key 使用 CSPRNG 生成，明文只在创建/轮换响应中显示一次，数据库只保存 Hash。
- 轮换允许有限重叠窗口：保存 Current Hash 和 Previous Hash/Expiry；超过部署上限的宽限期拒绝。所有轮换写安全审计，日志只记录 Source ID 和 Key Fingerprint。
- Webhook Key 只负责不可枚举路由，Provider 签名/Secret 验证仍是认证条件。

## 状态与删除规则

| 资源 | 允许的管理迁移 | 删除 |
| --- | --- | --- |
| Target | `MAINTENANCE ↔ ACTIVE`；任意非终态→`DISABLED`；恢复先回 `MAINTENANCE` | 不提供 |
| Agent Binding | `ACTIVE ↔ REVOKED`，恢复重新校验 Agent/Policy | 不提供 |
| Monitor Source | `ACTIVE ↔ DISABLED` | 不提供 |
| Monitor Binding | `ACTIVE ↔ DISABLED` | 不提供 |
| Policy | `DRAFT → ACTIVE → RETIRED`；Active 切换原子完成 | 不提供 |
| Inspection Plan | `PAUSED ↔ ACTIVE`；任意→`DISABLED`；恢复先回 `PAUSED` | 不提供 |
| Plan Target | `ACTIVE ↔ DISABLED` | 不提供 |

停用 Target 会阻止新 Run/Task 领取，但不删除历史，也不强制取消正在执行的 Mutation；紧急停止由步骤 9 的 Kill Switch/Execution Reconciler 处理。停用 Source 只停止新采集，历史 Event/Artifact 仍可读。

## APEX 与返回投影

APEX 只读视图将 `RAW(16)` 通过 `RAW_TO_UUID` 输出为规范 ID，并从根资源透出 `APP_ID/DOMAIN_ID`。列表视图不输出 Endpoint、SecretRef、Webhook Hash、Policy 全量 JSON 或内部错误；详情和所有写操作经 Main API。APEX 页面提交 `app_id/domain_id` 不参与授权，BFF 从会话重新派生。

配置响应使用专用 DTO：

```text
TargetSummary / TargetDetail
AgentBindingSummary / AgentBindingDetail
MonitorSourceSummary / MonitorSourceDetail
MonitorBindingDetail
PolicySummary / PolicyDetail
InspectionPlanSummary / InspectionPlanDetail
```

Summary 不含 SecretRef 与大 JSON；Detail 只返回 SecretRef 的 provider、fingerprint 和 `configured=true`，不返回完整路径或 Secret。健康错误返回稳定 Code/摘要，Provider 原始错误只进入受控审计 Artifact。

## 事务与外部调用

所有跨服务/Secret/Provider 检查使用三段式用例：

```text
UoW-A: 读取 Domain 内资源与 row_version → rollback/close
外部: Agent Runtime / Secret Provider / Template Registry 校验
UoW-B: 重读并校验 row_version → 写配置 + Audit/Outbox → commit
```

外部检查结果必须携带资源指纹和短期时间戳；UoW-B 若发现资源已变化则返回 `412`，不能提交基于旧检查的配置。普通无外部依赖的 PATCH 在一个短 UoW 中条件更新。

## 代码归属

```text
aiops_agent/api/management/{targets,monitor_sources,policies,inspections}.py
aiops_agent/application/configuration/
  target_service.py
  binding_service.py
  monitor_service.py
  policy_service.py
  inspection_service.py
aiops_agent/application/dto/configuration.py
aiops_agent/ports/{agent_runtime,secret_store,template_registry}.py
platform_core/contracts/aiops/configuration.py
platform_clients/aiops.py
```

Main API Route 只把外部 AuthContext 映射为签名内部上下文并调用 Client；AIOps Application Service 执行最终授权和不变式。Controller、Client 和 Repository 都不能自行组合权限。

## 最小测试矩阵

- UUIDv7 Oracle `RAW(16)` 往返、API 规范字符串和跨 Domain 隐藏；
- Create 幂等、自然键冲突、Cursor Domain/Filter 变化拒绝；
- 缺失/过期 ETag、并发 PATCH、并发 Policy 激活只成功一个；
- Agent 不存在、跨 Domain、停用或不具备 AIOps 能力时 Binding 失败；
- `AGENT_EXECUTE` 缺操作人、Secret、Policy 或 Kill Switch 时失败；
- SecretRef 不回显，Provider 失败不写半配置，Webhook 旧 Key 按宽限期失效；
- Health Check 乱序完成不覆盖新结果；
- Target/Source/Plan 停用后历史仍可读且不能创建新自动任务；
- Repository 无外部 I/O/Commit，外部校验期间不持有数据库事务。

## 完成定义

- 六类配置资源均通过 Main API → AIOpsManagementClient → AIOps API 管理；
- AuthContext、Domain、Binding、Policy 和部署能力求交不可由请求体扩大；
- ETag、幂等、Cursor、SecretRef、Key Rotation 和异步健康检查行为稳定；
- 配置、运行健康和历史事实没有混用状态字段；
- APEX 只读投影不泄露 Secret/Endpoint，所有 Command 可审计；
- 本步骤不创建 Ops Run，也不包含 LLM、Monitor Query 或目标数据库 SQL。
