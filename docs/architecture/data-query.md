# KBot4 问数能力

## 目标与边界

KBot4 问数将自然语言转换为受语义模型约束的结构化 Query Plan，再由确定性编译器生成参数化只读查询。LLM 不生成或执行任意 SQL，浏览器不访问内部 Data Query 服务，也不持有数据库凭据。

问数服务支持 PostgreSQL、MySQL 和 Oracle，统一包含连接测试、Schema 自动发现、对象级结构采集、语义模型、Agent Binding 投影、查询运行、结果、审计和已验证问题。AIOps 的诊断与变更执行不通过该入口。

## 数据接入与自动发现

公开入口位于 `/api/v1/apps/knowledge-retrieval/data-query`，内部服务入口位于 `/internal/v1/data-query`。

创建 Data Source 时 `auto_discover_schema` 默认是 `true`。Data Query 在创建数据源和加密凭据的同一事务中创建 Schema Snapshot 请求，Worker 随后执行：

1. 只读取配置的 `allowed_schemas` 中可见的表和视图；
2. 等待管理员选择需要采集的对象；
3. 逐对象读取字段、类型、默认值、注释、约束和索引；
4. 单个对象失败不会丢失同批次其他对象，失败对象可重试或提交受限的人工 DDL；
5. 生成不可变 Snapshot，并在通知中心报告发现、选择、进度和完成状态。

如需先创建数据源、稍后人工发起发现，可显式传入：

```json
{
  "auto_discover_schema": false
}
```

随后调用 `POST /api/v1/apps/knowledge-retrieval/data-query/data-sources/{data_source_id}/snapshots`。

## 语义模型与治理

Schema Snapshot 只描述物理结构；问数 Agent 使用已发布的 Semantic Model。管理流程为：

```text
连接测试 → 创建数据源 → 自动发现 Schema → 选择对象 → 采集结构
→ 生成或编辑语义模型草稿 → 问题验证 → 提交审核 → 发布
→ 在业务 Agent 中选择已发布模型 → 保存或启用 Agent
```

智能工作台 Agent 的 `data_model_ids` 是问数能力的唯一配置事实。Main API 在 Agent 版本保存后调用内部同步契约，Data Query 自动生成 `consumer_app_id + agent_id + agent_version_id + semantic_model_id` 投影，并停用同一 Agent 的旧版本投影。用户不创建、不编辑 Agent Binding，也不需要重复配置用户或角色策略；访问边界由 App 权限、Domain 隔离、Agent ACTIVE 状态和当前版本绑定共同决定。

## 查询执行安全

- 所有外部数据库使用加密保存的只读凭据；响应、日志和 Prompt 不返回密码或连接引用。
- Query Plan 只能引用已发布语义模型中的数据集、维度和指标。
- PostgreSQL、MySQL、Oracle 均使用参数绑定和只读事务。
- Data Query 内部 Query Guardrail 统一控制最大行数、结果字节数、语句超时和并发上限，并在创建 Run 时冻结快照。
- Worker 租约、心跳和编译哈希防止重复执行或执行被冻结计划之外的查询。
- 结果按固定期限清理；Run、Execution、Audit 和通知记录保留可追溯关系。

## 运行组件

- `data_query.entrypoints.api`：端口 `18140`，仅提供内部 API。
- `data_query.entrypoints.worker`：探针端口 `18141`，负责 Schema、语义模型生成、查询执行和结果清理。
- Main API：公开 BFF，负责用户权限、Domain 上下文和公开请求契约。

部署时必须同时启动 API 与 Worker。只启动 API 可以管理资源，但不会执行自动 Schema 发现或问数运行。
