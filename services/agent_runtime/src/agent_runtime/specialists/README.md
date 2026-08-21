# App 专属 Agent 实现边界

通用 Skill 只实现跨 App 一致的能力。某个 App 需要特殊规划、查询、检索或回答逻辑时，在 `specialists/<owner_app_id>/` 下提供专属实现，并由注册表按冻结 Agent Snapshot 中的 `owner_app_id` 显式分派。

新增特殊 Agent 时遵循以下约束：

1. 可以继承通用 Skill 或 Executor，复用稳定的协议解析、鉴权和通用调用流程。
2. App 的业务合同、提示词、筛选规则和回答口径必须留在自己的目录，不能向共享实现追加 `owner_app_id == ...` 分支。
3. Root 专属路由器通过 `RootAgentPlanner.app_route_planners` 注册；执行 Skill 通过 `AppScopedSkill.implementations` 注册。
4. 分派只信任冻结的 `config_snapshot.agent.owner_app_id`，不读取请求临时字段。
5. 未注册专属实现的 App 必须稳定使用通用实现，不能通过目录扫描或动态导入代码。

`km_asset/` 是首个完整示例：它拥有自己的 Asset Search Planner、语义问数执行器、Document Scope、KC 取证入口和结果组合入口。
