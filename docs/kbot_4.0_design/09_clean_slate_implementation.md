# 4.0 Clean-slate 最终目标蓝图

## 不兼容重构决策

KBot 4.0 是新系统发布，不是 3.x 的渐进替换。它不加载 3.x 的表模型、API、Controller/Service 链、Agent 状态、Skill 自动发现、Prompt 规划协议或旧 Parser 轮询协议。唯一明确晋级的业务基线是 3.5 已实现的 Knowledge Core：其新表、领域包、Worker 和检索骨架直接归入 4.0，并按本文档继续加固，不再重写第二套实现。除此之外，3.x 只提供经过审查的算法思路、原始业务数据来源和回归评测样本；旧代码直接删除，需要时从 Git 历史查阅。

这项决策消除双写、双读、兼容 Adapter、旧状态迁移和新旧语义混合的长期成本，使团队可以围绕稳定领域契约重新设计。

## 最终运行时

```text
Portal / APEX / Slack / MCP / API clients
                  │
                  ▼
       Main API / BFF (`/api/v1`)
   identity · API composition · Agent Runtime API
       │              │                 │
       ▼              ▼                 ▼
       Knowledge Core    Agent Runtime       AIOps Agent
       ingest/discover   supervisor/tasks    events/diagnosis
       evidence/jobs     specialists/skills  HITL/executor policy
       │              │                 │
       └─────── durable jobs / outbox ─┘
                  │
      Parser Workers / Model Runtime / External Adapters
```

领域内部使用分层/端口适配器架构：`api → application → domain → repository/adapter`。跨领域只使用版本化 API、Artifact 或 durable event；同 Schema 不构成跨域直接读写的许可。

## 必须新建的核心

| 能力 | 4.0 实现 |
| --- | --- |
| 知识库 | 继承并完善现有 `knowledge_core`、`KBOT_KC_*` 模型、UoW、Outbox、Parse/Index Job 和 Discovery/Evidence API |
| Agent | `AgentRuntime`、持久化 Run/Task/Artifact、Supervisor + Specialist、预算/取消/恢复 |
| Skill | Manifest、typed DTO、Policy Gate、固定入口、契约/安全测试、隔离第三方执行 |
| 身份 | Portal API Key、AuthContext JWT、Domain 隔离、服务身份与不可变审计 |
| 运维 | 独立 AIOps Agent、Ops Event/Alert、诊断 Run/Task、HITL/ChangeProposal、Scheduler lease、DB Executor policy |
| 平台 | `/api/v1` 与 `/internal/v1` 契约、Schema Contract、对象存储 port、配置/密钥、观测/SLO |

## 可复用但必须重新封装的能力

- Docling、OCR、VLM、PDF 多视图、表格/PPT 解析算法。
- LLM、Embedding、Visual 模型 provider client 与 Oracle Vector/Text 查询参数绑定。
- Oracle 异步连接、日志、监控 Adapter、SQL 执行驱动和已有测试数据。

复用的代码必须先通过新接口、异常语义、权限、资源控制和测试要求；不能因为来自 3.x 就直接 import 到 4.0 领域代码。

## 设计质量门槛

每个新领域在实现前必须具备：Owner、边界图、API/Artifact schema、表与迁移、授权规则、UoW/任务语义、失败/重试策略、指标/SLO、测试集和发布/回滚说明。每个 Agent/Skill 在启用前必须具备：职责、输入输出、权限、模型/工具预算、风险等级、评测集和审计事件。

任何缺失上述要素的功能不得通过“先写进 Root Orchestrator、Service 或 Prompt，后续再治理”的方式进入 4.0。

最终验收不依赖人工演示。所有领域必须进入统一 Release Gate，提供版本化评测数据、Schema/OpenAPI Manifest、质量/安全报告、数据重建对账和可恢复切换证据；详见 [41_kbot4_step12_acceptance_release_and_cutover.md](41_kbot4_step12_acceptance_release_and_cutover.md)。
