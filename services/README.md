# 可部署服务

每个目录是一个可独立构建和部署的服务边界。`src/<package>/entrypoints/`
保存该服务的 HTTP、Worker 或 Scheduler 进程入口，业务代码和私有资源不得跨服务
直接导入。

当前服务：

- `main_api`：Portal 面向的公开 API/BFF。
- `agent_runtime`：对话、记忆、Skill 与任务运行时。
- `knowledge_core`：文档入库、解析、索引和检索。
- `aiops_agent`：监控、诊断、巡检、审批及数据库执行。
- `model_serving`：LLM、Embedding、VLM 和视觉模型托管。
