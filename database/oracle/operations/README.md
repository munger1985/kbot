# Oracle DBA 操作资产

本目录保存必须由 DBA 显式确认后执行的重置、导出等 Oracle SQL。它们不属于空 Schema
初始化序列，也不会由应用启动或部署入口自动执行。

每个操作必须在文件头说明影响范围、前置条件、是否包含敏感数据以及失败后的恢复方式。

`apply_aiops_schema_20.sql` 用于已有测试数据不能重建时，将 AIOps Schema 19 原地升级到
Schema 20。执行前必须停止 AIOps API、Worker 和 DB Executor，并完成 Schema 备份；脚本保留
历史数据，但会关闭历史 Agent–Target 的受控执行，升级后需要在 Agent 页面重新明确授权。
