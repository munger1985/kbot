# 4.0 多 Agent 与 Skill 架构

## 目标与原则

4.0 将 Agent 体系从“一个大 Prompt 驱动的动态技能调用”演进为可控协作运行时。多 Agent 不等于每个请求都创建多个 LLM：默认走最短路径，只有任务跨领域、需要独立验证或包含高风险动作时才委派 Specialist。

- **Agent** 负责理解目标、选择协作路径、组合结果和对用户负责。
- **Skill** 是窄而确定的能力单元：检索、查询、计算、外部 API 调用或受控变更；它不负责自由规划。
- **Policy** 独立于 Planner/Prompt，决定谁可调用什么 Skill、是否需要审批、预算和数据范围。
- **Workflow** 是可恢复的执行记录，不是内存中可变的隐式上下文。

## 目标角色模型

```text
Request
  → Supervisor / Router
     ├─ Knowledge Specialist → Discovery/Evidence Skills
     ├─ Data Specialist      → Ask-data / analysis Skills
     ├─ Ops Specialist       → metric / diagnosis / change Skills
     └─ Conversation Specialist
  → Verifier / Policy Gate
  → Response Composer
```

Supervisor 维护会话级目标、委派、预算和最终编排；不得直接执行数据库变更或跨库检索。Specialist 接收最小化、类型化任务包，产出结构化 Artifact，而不是直接修改其他 Agent 的内存。Response Composer 只基于已验证的 Artifact 和 Evidence 组织回答。

Knowledge Specialist 必须先调用 Discovery，再在候选范围内调用 Evidence；禁止由 LLM 直接挑选全库 Chunk。Ops Specialist 的变更请求必须经过 Policy Gate 与 HITL；Planner 声称“允许执行”不构成授权。

## 协作协议

每次执行创建 `run_id`，每次委派创建 `task_id`。Agent 间使用版本化 Pydantic DTO/JSON Artifact：

```text
TaskEnvelope { run_id, task_id, parent_task_id, goal, input, permissions,
               budget, deadline, expected_artifact_type }
Artifact     { type, schema_version, producer, payload, evidence_refs,
               confidence, warnings, provenance }
```

Artifact 写入运行记录或短期状态存储；大文件和证据只保存 URI/稳定 ID。禁止跨 Agent 共享可任意写入的 `dict` 作为长期协议。现有 `ContextMemory`、`OpsContextMemory` 可在过渡期作为单个 Worker 的内部实现，但不能越过 Agent 边界。

委派必须声明输入、预期产物、超时、最大重试和取消传播。Supervisor 实施最大深度、最大并行数、token/模型/工具预算；无进展循环、相同 Skill 重复调用和无依据自我反思必须终止并返回可解释失败。

## Skill 标准

每个 Skill 是独立包，至少包含：

```text
skill.yaml              # 机器可校验 Manifest，运行时唯一注册来源
src/<package>/skill.py  # BaseSkill 实现
schemas.py              # 输入、输出和 Artifact Pydantic 模型
tests/                  # 单元、契约和安全测试
README.md               # 人类说明、示例与限制
```

Manifest 必须声明稳定 `skill_id`、语义版本、领域、输入/输出 schema、所需权限、运行模式（read-only / mutation）、幂等性、超时、重试策略、数据分类、外部依赖和 owner。Skill 名称使用 `kebab-case`，例如 `knowledge-discovery`、`ops-metric-query`；不得以 Python 类名或文件夹名作为隐式接口。

`BaseSkill` 统一接受已验证的输入 DTO 和 `ExecutionContext`，返回 typed Artifact 或标准事件流。Skill 不读取全局请求对象、不解析 Planner 自由文本、不自行打开跨域数据库 Session，也不直接调用其他 Skill。需要领域数据时调用所属领域 API/client；需要写入时由其 Application Service/UoW 完成。

4.0 不加载现有 `skill.md` 自动扫描、反射加载或“猜测 run/execute/call 入口”的适配器。所有 Skill 以新 Manifest、固定入口、依赖锁定和契约测试重新注册；上传的第三方代码默认不在主进程执行，须经审计并采用隔离运行环境。

## 策略、安全与 HITL

Policy Engine 在规划前过滤可见 Skill，在执行前再次强制检查。规则至少覆盖身份/租户、角色、资源范围、目标数据库、数据密级、运行模式、时间窗口、并发/费用预算和审计要求。

Mutation Skill 必须使用两阶段协议：先输出 `ChangeProposal`（影响范围、参数、回滚、依据和风险），经 Policy/HITL 批准后才获得一次性、带过期时间的执行令牌。执行结果必须包含目标、前后状态、操作者、批准记录和可追踪 `run_id`。禁止把“自动执行”规则仅写在 Prompt 中。

## 运行时、评测与迁移

新增 Agent Runtime，负责路由、委派、状态持久化、Skill 调用、超时/取消、预算、策略检查和事件流。Planner 仅生成受 schema 限制的候选计划；Plan Validator 校验 DAG、Skill 输入输出匹配、权限与预算后才能执行。简单请求可绕过 Planner，直接路由到单个 Specialist/Skill。

所有事件包含 `run_id`、`task_id`、`agent_id`、`skill_id`、`skill_version`、模型、耗时、token、重试、策略决定和 Artifact 引用。持续评测路由准确率、计划有效率、Skill 成功率、重复调用率、端到端时延/成本、Evidence 引用覆盖率、变更拦截率与人工批准率。

实施顺序：先定义 DTO、Run/Task 状态与新的 Agent Runtime；再用 Manifest、schema 和契约测试重写核心 Knowledge/Data/Ops Skill；随后实现 Specialist、Plan Validator 和 Policy Gate；最后接入 Portal/API。3.x SkillRuntime、动态反射适配器、Prompt 内授权规则和跨 Agent 可变全局上下文不迁入 4.0。
