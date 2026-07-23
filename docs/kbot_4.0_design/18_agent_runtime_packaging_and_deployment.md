# 4.0 Agent Runtime 打包与部署

## 部署单元

Agent 是逻辑职责，Runtime 是执行宿主，API/Worker 是部署单元。4.0 采用以下逻辑服务：

```text
Main API / BFF
       ↓
Agent Runtime API ─────── Agent Runtime Worker
       │                         │
       │                         ├─ Root/Supervisor
       │                         ├─ Document Agent
       │                         └─ Response Composer
       │
       └────────────── AIOps Agent（独立服务）
```

开发环境可以让 Runtime API 和 Worker 在同一个进程中启动；生产环境建议拆成两个副本池：API 负责创建 Run、查询状态和 SSE，Worker 负责领取 Task、调用 Agent/Skill 和写入 Artifact。两者共享同一 Agent Runtime 代码和数据库，但不共享进程内状态。

AIOps Agent 不作为 Runtime Worker 中的一个普通模块运行。它拥有独立 `KBOT_OPS_*` 表、目标数据库权限、监控调度和 HITL 流程，通过窄 `AIOpsDelegationClient` 与 Runtime 通信。

## 推荐代码布局

```text
agent_runtime/
  api/                  # Run 查询、事件、取消、审批 DTO/路由
  application/          # Run/Task 用例、路由、组合和恢复
  domain/               # 状态机、Plan、Policy、Artifact 契约
  runtime/              # Scheduler、Lease、Worker、事件发布
  specialists/
    document/           # Document Agent 模块
    response_composer/  # 最终响应合成与 Grounding
  clients/              # KC、AIOps、MCP、Model Serving Client
  entities/             # KBOT_AGENT_*，包含跨服务 Delegation
  repositories/         # 仅访问 KBOT_AGENT_* 表
  tests/

aiops_agent/
  api/                  # Ops Run、Event、Proposal、Approval API
  application/          # 诊断/执行/验证用例
  domain/               # Ops 状态机和 Policy
  orchestration/        # Ops 专属流程编排
  repositories/         # 仅访问 KBOT_OPS_* 表
  adapters/             # Metrics、Logs、DB Executor Client、外部系统
  tests/

apps/agent_runtime_api/main.py
apps/agent_runtime_worker/main.py
apps/aiops_api/main.py
apps/aiops_worker/main.py
apps/aiops_scheduler/main.py
apps/aiops_db_executor/main.py
```

`agent_runtime_worker` 按独立进程装配自己的数据库 Runtime、KC Client、
Model Client、Skill Registry 和日志。API 进程只加载同一组 Manifest 用于
计划与 Artifact 校验，不初始化 KC/模型 Client；因此二者不会因为共享 Python
包而共享连接池或运行时状态。

4.0 不保留旧 Document Agent 或动态 Skill 实现。新的 Document Specialist
将在 `agent_runtime/specialists/document/` 中基于固定 Manifest 和
`DocumentQueryTask → DocumentRetrievalResult` 契约实现。

## API 与 Worker 边界

### Agent Runtime API

- 验证 AuthContext 和 Run 创建请求；
- 生成 Root Task 或调用 Router；
- 查询 Run、Task、Artifact 摘要；
- 从 Event 表读取 SSE；
- 接收取消、恢复和审批命令；
- 不执行 LLM、KC、MCP 或 AIOps 调用。

### Agent Runtime Worker

- 领取 `READY` Task 并维护租约；
- 执行 Root、Document 和 Response Composer Task；
- 调用 KC Client、Model Client 和 MCPDataClient；
- 通过窄 AIOpsDelegationClient 创建/对账跨服务子 Run；
- 校验 Skill/Artifact schema；
- 原子写入 Task、Artifact 和 Event；
- 处理超时、重试、取消和恢复。

API 与 Worker 都必须是无状态进程；Run 状态、事件、Artifact 和租约以数据库为准，不能使用 Python 全局变量、单机内存队列或本地文件作为可靠状态。

## AIOps Agent 边界

AIOps Agent 有自己的 API、Worker、Scheduler 和高权限 DB Executor 进程。它内部的诊断 Task、监控调度、ChangeProposal 和执行验证由 `KBOT_OPS_*` 表管理。Root Runtime 通过 `KBOT_AGENT_DELEGATION` 保存 `ops_run_id`、子事件游标、状态摘要和最终受限 Result Artifact。详细责任和代码布局见 [25_aiops_service_packaging_and_runtime.md](25_aiops_service_packaging_and_runtime.md)，父子集成见 [40_aiops_step11_root_main_api_and_apex_integration.md](40_aiops_step11_root_main_api_and_apex_integration.md)。

AIOps Agent 可以复用：

- `platform_core` 配置、日志、DB Runtime 和安全上下文；
- 通用 Run/Task/Event DTO 和租约语义；
- Model Serving、Knowledge Core 和 DB Executor Client。

但不能复用：

- Knowledge Core Repository；
- Agent Runtime 的 Ops 表写权限；
- Root Agent 的可变内存；
- 旧 `services`、旧 Agent Orchestrator 或旧 SkillRuntime。

## 配置与数据库

每个 App 有独立配置入口、日志资源名、健康检查和 Session Factory。4.0 阶段所有 App 指向同一 Oracle/APEX Schema，但只访问自己的表前缀和 API；未来拆库时仅替换连接配置、账号和连接池。

模型、KC、AIOps 和 MCP URL、超时、重试、服务身份和预算都通过配置注入，不能硬编码在 Specialist 或 Skill 中。生产环境 API/Worker 使用独立副本数和资源限制，开发环境启动脚本可以提供合并模式。

## 扩缩容和故障恢复

- Runtime API 按请求和 SSE 连接数扩缩；
- Runtime Worker 按 READY Task 数量扩缩；
- Document Task 和 Conversation Task 可设置不同并发/模型预算；
- AIOps Worker 按监控事件和诊断队列扩缩；
- Parser、KC Projection 和 Model Serving 继续独立扩缩；
- Worker 崩溃通过租约过期接管，API 重启不影响运行中的 Task；
- 同一 Task 不允许被两个 Worker 同时提交成功结果。

## 第一版交付顺序

1. 先在 Runtime 内实现 Root/Document/Conversation 模块和持久化 Run/Task/Artifact；
2. API 与 Worker 使用同一代码包提供两个入口，开发环境支持合并启动；
3. 实现 AIOps API/Worker/Scheduler/DB Executor 入口和 Management/Delegation Client；
4. 完成 MCP Data Adapter 接入，不创建 Data Agent；
5. 经过并发、租约、SSE 和跨服务契约测试后，再决定是否单独扩缩 Document Agent。
