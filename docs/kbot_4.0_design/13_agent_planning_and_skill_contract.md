# 4.0 Agent 规划与 Skill 契约

## 角色边界

```text
Supervisor
  → Router / Planner（可选）
     → Specialist
        → Skill
  → Verifier / Response Composer
```

- **Supervisor**：负责一次 Run 的目标、预算、委派、取消和最终结果，不直接查询 KC、数据库或执行外部动作。
- **Router**：判断是否单 Skill 直达，或需要生成多 Task DAG；只使用经过 Policy 过滤的能力目录。
- **Planner**：将复杂目标转换为结构化计划，不调用 Skill、不读取业务数据库、不执行副作用。
- **Plan Validator**：确定性校验 DAG、输入输出 schema、权限、预算、超时和并发限制。
- **Specialist**：Knowledge/Conversation 等通用领域协调者，只接收类型化 Task，选择本领域已批准 Skill；AIOps 由独立服务接管，不作为普通 Specialist 内嵌；问数暂由 MCP Adapter 提供。
- **Skill**：窄而确定的执行单元，调用所属领域 API/Client 并返回 Typed Artifact。
- **Verifier / Response Composer**：验证 Artifact 和 Evidence 后组织回答，不重新检索或修改原始产物。

任何角色都不得通过全局可变 Context 传递长期状态。跨角色数据只能使用版本化 Task、Artifact 和事件。

## 计划 DTO

Planner 只能返回以下结构，不能返回自由文本指令：

```text
PlanDraft {
  run_id
  plan_version
  objective
  tasks: [TaskSpec]
  final_task_key
  expires_at
}

TaskSpec {
  task_key
  task_type
  execution_kind: LOCAL_SKILL | DELEGATION
  specialist
  skill_id
  skill_version
  delegate_service
  delegate_capability
  depends_on: [task_key]
  input_refs: [artifact_type | task_output]
  expected_outputs: [artifact_type]
  required_scopes
  timeout_seconds
  max_retries
  completion_requirement: REQUIRED | OPTIONAL
  execution_mode: read_only | mutation | delegated
}
```

`LOCAL_SKILL` 必须填写 Skill ID/Version 且不能填写 Delegate 字段；`DELEGATION` 反之，并要求目标能力存在于受信 Service Capability Registry、Execution Mode 固定为 `delegated`。它只授权创建子 Run，不代表授权子服务执行变更；AIOps 仍独立执行 Policy/HITL。Planner 可以省略复杂步骤：简单问答直接生成一个 `knowledge-retrieval` 或 `conversation-answer` Task。只有跨领域、需要并行检索或需要独立验证时才生成多个 Task。

## Plan Validator

计划进入 Runtime 前必须通过以下检查：

1. `task_key` 唯一，依赖存在且 DAG 无环；
2. Local Skill ID/Version 或 Delegate Service/Capability 在对应 Registry 中存在，未被禁用；
3. 每个 `input_refs` 能由 Run 输入或前置 Artifact 提供；
4. 前置输出 schema 与后继输入 schema 兼容；
5. Specialist、Skill、domain、collection 和安全等级符合 AuthContext；
6. 总 Task 数、最大并行数、超时、重试和模型预算未超出 Run 上限；
7. Mutation Task 必须有 Policy Gate/HITL 节点，不能直接连接执行 Skill；
8. 最终 Task 的输出必须是允许暴露给 API 的 Artifact 类型。

Validator 返回结构化错误：`PLAN_CYCLE`、`SKILL_NOT_FOUND`、`SCHEMA_MISMATCH`、`SCOPE_DENIED`、`BUDGET_EXCEEDED`、`MUTATION_APPROVAL_REQUIRED`。校验失败时不创建可执行 Task，只生成错误 Artifact 和 `RUN_FAILED` 事件。

## Skill Manifest

Skill 不再依赖 `skill.md` 扫描或反射适配。每个 Skill 有固定入口、锁定版本和可验证 Manifest：

```yaml
skill_id: knowledge-retrieval
version: 1.0.0
owner: knowledge-core
specialist: knowledge
description: retrieve citable evidence from authorized collections
input_schema: KnowledgeRetrievalInput.v1
output_artifacts:
  - type: CITATION_PACK
    schema: CitationPack.v2
permissions:
  - knowledge.discovery.read
  - knowledge.evidence.read
execution_mode: read_only
idempotent: true
timeout_seconds: 30
max_retries: 2
data_classification: internal
external_dependencies:
  - knowledge_core_api
```

Manifest 必须声明稳定 `skill_id`、语义版本、owner、领域、输入 schema、输出 Artifact、权限、运行模式、幂等性、超时、重试上限、数据分类和外部依赖。Skill 名称使用 `kebab-case`；版本必须显式写入 Task 和 Artifact，不能随部署代码自动漂移。

Registry 在服务启动时加载经过审核的 Manifest 和固定实现映射。运行时只允许解析已注册的 `(skill_id, version)`；第三方 Skill 默认不进入主进程，未来需要隔离 Worker 和独立信任策略。

## 执行协议

```python
class Skill:
    manifest: SkillManifest

    async def execute(
        self,
        input: ValidatedInput,
        context: ExecutionContext,
    ) -> SkillResult:
        ...
```

SkillResult 只能包含 Typed Artifact、警告和进度事件。Skill 不能：

- 直接创建 Agent、Planner 或其他 Skill；
- 自行打开数据库 Session 或跨域访问 Repository；
- 修改 Run/Task 状态；
- 将未验证的 LLM 文本作为下一个 Task 的命令；
- 把敏感正文写入日志或事件 payload。

需要 KC 知识时调用 Knowledge Core Client；需要模型时调用 Model Serving Client；当前问数通过 MCPDataClient 调用受控 MCP Tool。未来若引入 Data Agent，再替换为 Data Query Client。领域写操作由对应 Application Service/UoW 完成。

## 规划到执行流程

```text
1. Supervisor 创建 Run 并冻结 AuthContext、Policy、预算
2. Router 选择单 Skill 快路或请求 Planner 生成 PlanDraft
3. Plan Validator 校验计划结构和契约
4. Policy Gate 对最终 Task 再次检查资源和运行模式
5. Runtime 持久化 Task DAG，产生 TASK_CREATED 事件
6. Local Task 调用固定 Skill；Delegate Task 幂等创建子 Run 后进入 `WAITING_EXTERNAL`
7. Runtime 校验 SkillResult 或 Result Envelope，原子写入 Artifact、状态和事件
8. Verifier/Composer 消费 Artifact，生成最终结果
```

Planner 产生的计划不是授权。每次重试、恢复或审批后重新执行前，都必须重新检查 Task 版本、租约和 Policy；策略变化时旧计划作废并重新规划。

## 领域 Skill 的最小集合

第一批只实现以下独立 Skill，不迁移旧动态技能库：

| Specialist | Skill | 输出 |
| --- | --- | --- |
| Knowledge | `knowledge-retrieval` | `CITATION_PACK` |
| Data（后续，当前跳过） | `data-query` | `QUERY_RESULT` |
| Conversation | `conversation-answer` | `ANSWER_DRAFT` |
| Verification | `evidence-grounding` | `GROUNDED_ANSWER` |
| AIOps Agent（独立服务能力，不注册为 Runtime Skill） | `aiops-diagnosis` | `DELEGATED_AIOPS_RESULT` |

Knowledge Retrieval 不直接生成最终答案；MCP 问数结果不返回文档引用；Grounding 只接受明确的 CitationPack、QueryResult、Delegated AIOps Result 和 AnswerDraft，保证混合回答的来源可追踪。
