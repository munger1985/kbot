# Oracle DBA 操作资产

本目录保存必须由 DBA 显式确认后执行的重置、导出等 Oracle SQL。它们不属于空 Schema
初始化序列，也不会由应用启动或部署入口自动执行。

每个操作必须在文件头说明影响范围、前置条件、是否包含敏感数据以及失败后的恢复方式。

`apply_aiops_schema_20.sql` 用于已有测试数据不能重建时，将 AIOps Schema 19 原地升级到
Schema 20。执行前必须停止 AIOps API、Worker 和 DB Executor，并完成 Schema 备份；脚本保留
历史数据，但会关闭历史 Agent–Target 的受控执行，升级后需要在 Agent 页面重新明确授权。

`apply_aiops_schema_22.sql` 用于将 AIOps Schema 21 原地升级到 Schema 22，扩展附件受控
检索所需的工具调用分类约束。执行前必须停止 AIOps API、Worker、Scheduler 和 DB Executor，
并完成 Schema 备份；脚本不删除表、不删除业务行，也不修改既有诊断或审计数据。

`apply_aiops_schema_23.sql` 用于将 AIOps Schema 22 原地升级到 Schema 23，增加 Oracle
CDB/PDB 预期范围与连接实测字段。脚本保留全部业务行、Endpoint 和凭据引用；由于 Service
Name 不能可靠判断容器类型，既有 Oracle 直连 Target 会被安全停用为仅监控模式，升级后须在
Target 页面明确选择 `CDB Root`、`PDB` 或 `Non-CDB`，重新测试连接后再启用。

`enable_model_serving_capabilities.sql` 用于为已有模型目录新增 X Search、文生图和响应流的
能力验收列。脚本不会把任何模型标记为可用；所有新增列默认关闭，必须由受控 Canary 写入
验收结果后，业务入口才可使用对应能力。
