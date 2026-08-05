# 后台资源组合编排

Main API 是模型目录、知识库、Agent、问数和 Run 的组合层，但不拥有这些领域实体。模型由 Model Serving 管理，Collection 由 Knowledge Core 管理，Agent/Run 由 Agent Runtime 管理，Data Source、Semantic Model 与 Data Query Result 由 Data Query 管理。

## 命令协议

所有组合写命令必须携带 `Idempotency-Key`，更新命令还必须携带 `If-Match`。Main API 持久化 `KBOT_COMPOSITION_RECEIPT`，按以下顺序执行：

1. `PRECHECKING`：读取实际下游资源，校验模型状态、绑定和行版本。
2. `COMMAND_SUBMITTED`：先提交 Receipt，再调用资源归属服务。
3. `SUCCEEDED`：重新读取下游并验证预期状态后完成。
4. `FAILED_PRECHECK`：命令尚未发送，可在修复输入后使用新的幂等键重试。
5. `COMPENSATION_REQUIRED`：命令结果不确定；相同幂等键只会重做验证，不会重复发送命令。

创建 Agent 和 Collection 时，Main API 在发送命令前预分配资源 UUID，并将其写入 Receipt。即使下游提交后响应超时，后续验证仍能定位同一个资源。

## 引用与停用

`/api/v1/compositions/resources/{type}/{id}/references` 汇总模型、Collection、Agent、Semantic Model、Data Source 与 Run 的实际引用。每个节点都包含来源服务、来源版本、观察时间和可用性；下游不可用时响应标记为 partial，不把未知状态解释为“没有引用”。

停用前检查只返回阻断关系，不隐式级联删除。调用方必须先解除 Agent、Collection、模型或语义模型绑定，再调用资源归属服务的归档或删除接口。

## Run 追踪

Run 组合视图以 Agent Runtime 的 `config_snapshot`、任务和 Artifact provenance 为事实来源，关联实际 Agent、模型、Collection、Data Query Run/Result、Semantic Model、Data Source、Knowledge Evidence 和当前 Actor 可见通知。响应只返回结果结构、行数、截断标记、Hash 与 provenance 等安全摘要，不复制查询结果正文、提示词或凭据。
