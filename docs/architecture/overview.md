# KBot 4.0 系统架构

## 架构形态

KBot 4.0 是服务型单仓库。八个服务可独立构建、启动和
扩缩容，但受 APEX 约束，当前共享一个 Oracle 26ai Schema。每个服务只访问自己
拥有的 `KBOT_*` 表；共享 Schema 不代表共享 Repository 或事务。

这是有明确拆分边界的分布式单体。未来拆库时迁走对应 DDL、配置数据库连接并发布
共享契约包即可，业务服务之间不需要重写为直接数据库调用。

## 服务职责

| 服务 | 职责 | 主要进程 |
|---|---|---|
| Main API | Portal/APEX 的公开 BFF、认证、App RBAC、Domain 上下文、Slack 公共入口、通知和 SSE 转发 | API、Notification Worker |
| KM Asset App | KM 资产与 Agent 管理、Slack 验签/Inbox/Outbox/外部 Callback | API、Worker、Slack Worker |
| Knowledge Retrieval App | 知识检索 Agent、不可变版本和 Domain 隔离 | API |
| Agent Runtime | Execution Spec、Conversation、计划、Task、Skill、Artifact、事件和记忆 | API、Worker |
| Knowledge Core | Collection、Bundle、解析、Evidence、索引和检索 | API、Parser、Projection Worker |
| Data Query | 数据源、Schema 快照、语义模型、策略、绑定和受控查询运行 | API、Worker |
| AIOps Agent | 监控接入、数据库诊断、HITL、审批执行、巡检和报告 | API、Worker、Scheduler、DB Executor |
| Model Serving | 模型目录及 LLM、VLM、OCR、文本/视觉 Embedding 推理 | 五类模型进程 |

`packages/platform_core` 提供配置、日志、Oracle Session Factory、身份、Prompt
和跨服务 DTO；`packages/platform_clients` 提供 HTTP 客户端。共享包不能包含业务
Entity、Repository 或用例。

## 调用与数据边界

```text
Portal / APEX
      │ /api/v1 + 用户 Token / App API Key
      ▼
   Main API / App BFF
      │ /internal/v1 + Service Credential + AuthContext JWT
      ├────────► Knowledge Retrieval App ──► Agent Runtime
      ├────────► Agent Runtime ─────► Knowledge Core / Data Query / AIOps
      ├────────► KM Asset App ──────► Slack
      ├────────► Knowledge Core ────► Model Serving
      ├────────► Data Query ────────► Oracle / PostgreSQL / MySQL
      └────────► AIOps Agent ───────► DB Executor / Monitor Provider

Slack Events API ──► Main API 公共适配器 ──► KM Asset Inbox/Outbox
                                             │ App API Key + /api/v1
                                             ▼
                              Main API / KM Asset App BFF ──► Agent Runtime
```

跨服务调用只使用 HTTP、版本化 DTO 或持久化任务。Repository 不跨服务导入，
服务方法不把数据库 Session 注入其他服务。应用服务拥有 Unit of Work，Repository
只执行查询和 `flush()`，不得自行 `commit()`。

## 一致性与异步任务

服务内部用单个 Unit of Work 保证聚合、任务和事件的原子写入。耗时任务使用
状态机、幂等键、有限 Lease、Heartbeat 和有限重试。KC 通过 `DBMS_ALERT` 唤醒
Worker，并保留低频轮询作为通知丢失后的恢复路径。Agent Runtime、Data Query 与
AIOps 的执行事实均
持久化，进程重启只接管未完成任务。

## 部署原则

开发环境可由 `start_kbot.sh` 启动全部进程。生产环境应按服务构建，只携带目标
服务、必要共享包和配置；不得把整个仓库复制成一个应用进程。外部只暴露 Main API
以及明确启用的 OpenAI 兼容模型接口，所有 `/internal/v1` 均留在内部网络。
