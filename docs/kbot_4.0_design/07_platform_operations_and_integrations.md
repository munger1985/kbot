# 4.0 平台运行、观测与外部集成

## 平台运行模型

服务仍以独立进程部署：Main API/BFF、Knowledge Core、AIOps Agent、Parser Worker、LLM、Embedding、VLM、Visual、DB Executor、MCP，以及必要的 Scheduler/Outbox Dispatcher。开发环境可使用 `start_kbot.sh`；生产环境必须为每个服务单独声明副本数、资源规格、配置、密钥、Liveness、Readiness 和优雅终止时间。

`start_kbot.sh/stop_kbot.sh` 仅用于本地开发，不是生产发布或回滚工具。生产采用一次构建、多环境晋级的不可变镜像/包，并保存规范建库脚本校验和、配置 Schema、镜像 Digest、SBOM、测试和签名组成的 Release Evidence；完整流程见 [41_kbot4_step12_acceptance_release_and_cutover.md](41_kbot4_step12_acceptance_release_and_cutover.md)。

Readiness 不仅检查进程存活：需要验证关键配置已加载、数据库/对象存储可用、模型服务已接受请求、Worker 可领取任务。Liveness 只检查进程是否卡死，不能因为短暂下游故障不断重启服务。

## 统一可观测性与韧性

所有入站请求生成或继承 `request_id`，并在 HTTP、Job、Agent Task、MCP 与外部回调中传播。Trace、Metric、Log 使用相同资源属性：`service_name`、版本、环境、租户（脱敏）、`request_id`、`run_id`、`task_id`、`job_id`。

定义服务级 SLO：可用性、p95/p99 延迟、错误率、队列积压、Job 成功率、索引新鲜度、模型调用成本和 HITL 等待时间。每项 SLO 有告警阈值、负责人和 Runbook。日志需要结构化、脱敏、按级别采样；审计日志独立保留。

所有跨服务调用必须有超时、有限重试、指数退避、并发限制和熔断策略。读请求可以降级为“暂不可用/稍后重试”；写请求必须携带幂等键。任何后台任务的失败都必须进入可查询的重试/死信状态，而不是只写异常日志。

## AIOps 与 Scheduler

AIOps Agent 是独立领域服务，不应由定时脚本直接访问业务表或硬编码 BFF 地址。监控 Adapter 负责采集，AIOps Agent 负责 Ops Event/Alert、去重、关联、抑制、路由、资产绑定、诊断编排和 HITL 状态机；DB Executor 负责实际数据库操作。详见 [15_aiops_agent_scope_and_skills.md](15_aiops_agent_scope_and_skills.md)。

```text
Prometheus / Zabbix / OEM
  → Monitor Adapter → Ops Event
  → dedup/correlation → diagnostic task
  → AIOps Agent → ChangeProposal → Policy/HITL → DB Executor
```

Scheduler 使用配置化服务发现和内部 client，领取持久化巡检任务；同一规则在多副本下必须有租约或 leader 选举。告警、诊断、批准和执行均记录不可变审计链，禁止 Scheduler 直接绕过 Policy/HITL 写数据库或调用变更接口。

## MCP、Slack 与外部 Webhook

MCP、Slack、Dify、KM Portal 和监控平台均是 Adapter，不拥有核心业务逻辑或领域表。每个 Adapter 定义版本化 DTO、认证方式、超时、幂等键、速率限制、重放保护和失败重试策略。

- Webhook 在验证签名、时间戳和事件唯一 ID 后才入队；响应路径只做验证与接收，不同步运行 Agent。
- Slack/外部回调的重试通过 Event Inbox 去重，避免重复创建 Agent Run 或 Knowledge Job。
- MCP Tool 调用映射为受权限约束的领域 API；不将数据库 Session、内部 URL 或任意 SQL 暴露给 MCP 客户端。
- Portal 上传使用 Ingestion API 和稳定来源 ID；不共享服务器文件系统目录。

## 模型运行时治理

模型运行时只处理推理和内存池。配置领域负责模型启停策略、可用状态、供应商凭据和允许范围；每次请求记录模型、模型版本、参数快照、token/耗时与失败类别。模型切换采用健康检查、灰度路由和回退，不允许直接更新配置后让运行中的所有请求无版本地漂移。

GPU/CPU Worker 设置并发、显存/内存水位、队列上限和隔离策略。Parser 与 Visual/VLM 任务分别限流，避免大文件或单个模型加载耗尽同机资源。
