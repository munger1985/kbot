# 4.0 目标架构与服务边界

## 运行拓扑

```text
Clients / KM Portal / Slack / MCP clients
                 │
                 ▼
          Main API / BFF
  auth · agent · ops · API composition
        │          │             │
        │ HTTP     │ HTTP         │ HTTP
        ▼          ▼             ▼
 Knowledge Core  Model Runtime  AIOps Agent
ingestion       LLM/Embed/VLM  events/diagnosis
discovery       Visual         HITL/orchestration
evidence              │
                      │ controlled execution
                      ▼
                  DB Executor
        │
        │ durable jobs / result records
        ▼
  Parser Worker ──► object/file storage

All processes ──► Oracle 26ai（单一 APEX Schema；表所有权受契约约束）
```

每个进程可独立重启、设置资源配额和水平扩缩，但保持一个 Git 仓库、统一配置格式、统一鉴权与观测规范。Agent 是逻辑职责，不要求每个 Agent 都单独部署；Root/Knowledge Agent 可先运行在 Agent Runtime 宿主内，AIOps Agent 因拥有独立 Ops 数据和高风险流程而单独部署。`start_kbot.sh` 可继续作为开发环境启动器；生产部署应按服务分别声明副本数、健康检查和资源限制。

每个可部署 App 的最小交付单元为：`app` 入口、所属领域包、`platform_core`、服务配置和必要的共享契约/客户端包。Agent Runtime 的 API 与 Worker 可以使用同一代码包提供两个入口；AIOps Agent 单独拥有自己的 App。当前所有 App 的 Session Factory 指向同一 APEX Schema；未来拆成完整微服务时，只替换各 App 的数据库配置、账号和连接池，不改变领域 API、任务协议或服务边界。

## 服务职责

| 服务 | 拥有的能力与写入权 | 禁止承担的职责 |
| --- | --- | --- |
| Main API / BFF | 认证、用户/API Key、Agent Runtime API、AIOps API 组合 | 直接查询或写入 Knowledge Core/Ops 表；实现检索算法或旧 API 适配 |
| Knowledge Core | Collection、Bundle、Document、Version、Evidence、Discovery、任务状态 | Agent 最终回答、Skill 路由、模型实例生命周期 |
| Parser Worker | 领取 `PARSE` 任务、生成 Parse View/Evidence 候选、报告结果 | 决定 Bundle READY、直接管理上传和版本切换 |
| Model Runtime | LLM/Embedding/VLM/Visual 推理与模型内存池 | 修改模型配置、写知识库业务表 |
| Platform Config | 模型配置、公共服务配置、身份和权限策略 | 业务领域数据的跨域写入 |
| AIOps Agent | Ops Event/Alert、目标资产、诊断 Run/Task、HITL、ChangeProposal、验证报告 | 知识库内容、业务数据查询、绕过 Policy 直接执行变更 |
| DB Executor | 经授权的目标数据库查询/运维执行 | 作为业务数据 Repository 或通用 SQL 后门 |

## 调用规则

- 跨服务只使用版本化 HTTP API 或 durable job/outbox；不得跨服务 import Service 或 Repository。
- 同一领域内部允许 Python 模块调用；`knowledge_core` 内的 API、application、domain、repository 不需要为了“微服务”再走 HTTP。
- 查询可同步 HTTP；耗时或可重试操作（解析、画像、索引、关系构建）必须异步任务化。
- API 的输入输出使用 Pydantic DTO；不得直接暴露 SQLAlchemy Entity。
- 服务 URL、超时、重试、内部令牌和调用指标统一封装在 `platform_clients/` 或各领域的 client 包中；4.0 新代码不得依赖通用 `utils` 工具包。

## 建议代码布局

```text
knowledge_core/
  api/                 # ingestion/discovery/evidence 路由与 DTO
  application/         # 用例、命令、查询、UoW 协调
  domain/              # 领域模型、状态机、策略、端口
  repositories/        # 仅访问 KB4 表
  workers/              # job claim、outbox dispatch、结果应用
  indexing/ retrieval/ parsing/
  tests/
apps/knowledge_core_api/main.py  # Knowledge Core 进程入口
main_api/                        # 公开 DTO、Domain Registry、BFF 路由
apps/main_api/main.py            # 唯一公开 `/api/v1` 入口
apps/aiops_api/main.py           # AIOps 对外/内部 API
apps/aiops_worker/main.py        # AIOps 诊断与流程 Worker
apps/aiops_scheduler/main.py     # 巡检调度与超时扫描
apps/aiops_db_executor/main.py   # 隔离的数据库执行面
```

旧通用 DAO 不进入 4.0，确认没有新消费者后直接删除；历史实现仅从 Git 查阅。
新知识库 Entity/Repository 不进入通用 DAO，避免再次形成全局数据层。

## 边界治理

4.0 因 APEX 约束保留**单一 Schema**，不将“独立 schema”作为前提。领域隔离通过严格表前缀（`KBOT_KC_*`、`KBOT_AGENT_*`、`KBOT_OPS_*`）、独立 migration 目录、表 Owner 清单、运行时数据库账号的最小授权（可行时）和代码依赖检查实现。若当前所有进程必须使用同一个数据库账号，则表权限无法成为隔离边界，必须以 API/UoW、代码审查和集成测试强制禁止跨域 DML。

4.0 不要求拆分 Schema 或数据库。未来若需要完整微服务化，只需为每个服务提供独立的数据库连接、账号和存储实例；服务 API、任务协议、表所有权和 `platform_core` 基础设施边界已经在 4.0 固定，不需要再次改造业务代码。APEX 继续通过受控视图或 API 访问其所属数据。
