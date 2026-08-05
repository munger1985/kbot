# KBot 后端能力补全迁移实施计划

配套执行文档：

- `docs/product/kbot-backend-capability-migration-steps.md`：实施批次、验收门和部署顺序；
- `docs/product/kbot-backend-capability-migration-detailed-design.md`：目录、契约、Oracle 表、
  状态机、接口和测试细节。

## 1. 目标、范围与顺序

本计划以 KBot 现有功能和 Oracle 运行环境为基准，选择性迁移 Ammolite 已成熟的后端
能力。目标是补全 KBot 的数据查询、Agent Runtime、Knowledge Core、模型目录、运行日志和
通知能力；不改变 KBot 的多 Domain 模型，不引入 Tenant/App、用户目录、RBAC 或 Portal
前端。

实施顺序固定如下：

0. 先按 Ammolite 的工作区组织和 editable package 安装方式收口 KBot；
1. Data Query 完整服务，并与现有 MCP 问数双模式兼容；
2. Agent Runtime 能力补足；
3. Knowledge Core 增强；
4. Model Serving 的目录一致性与生命周期补齐；
5. Main API 运行日志与开发运行记录迁移；
6. 无用户/权限依赖的通知中心；
7. 最后增加模型、知识库、Agent、Run、数据查询之间的组合编排接口。

### 不在范围内

- AIOps 及其 Target、凭据、监控、执行能力；
- Ammolite 的 Tenant、App、用户、角色、资源授权、SSO、服务账号和许可证功能；
- `apps/portal-web` 或任何前端页面；
- 将 KBot Oracle Schema 改为 PostgreSQL，或直接复制 Ammolite 的 PostgreSQL DDL；
- 为旧 3.x 接口保留适配层。现有 KBot 4.0 MCP 问数是本次需要保留的正式能力，不属于
  兼容遗留。

## 2. 阶段零：工作区组织与内部包安装收口

### 2.1 目标组织

以 Ammolite 的单仓库包边界作为 KBot 的目标结构：根目录只定义工作区元数据、全局测试和
格式工具；每个可运行服务和共享包都有独立 `pyproject.toml`、`src/<package>/`、独立
依赖声明及模块入口。禁止服务依赖当前目录、手工 `PYTHONPATH`、隐式源码相对导入或把其他
服务目录加入 `sys.path`。

KBot 现有 `platform_core`、`platform_clients`、`main_api`、`agent_runtime`、
`knowledge_core`、`model_serving`、`aiops_agent` 已具备独立包雏形；阶段零的工作是将其
作为唯一运行方式收口，并在新增 `data_query` 时遵从相同布局：

```text
packages/
  platform_core/       # 配置、安全、数据库原语、共享 contracts
  platform_clients/    # 受限内部 HTTP clients
services/
  main_api/
  agent_runtime/
  knowledge_core/
  model_serving/
  data_query/
  aiops_agent/         # 保持现状，不纳入本计划的业务改造
```

根 `pyproject.toml` 增加 KBot 专用 workspace members 清单，覆盖上述全部内部包。它用于
一致性校验和脚本生成，不作为可安装的伪聚合包；第三方依赖仍由 `requirements.txt` 统一锁定。

### 2.2 开发与生产安装契约

重写 `scripts/deployment/install_workspace.sh`，行为与 Ammolite 对齐：

1. 解析 `KBOT_PYTHON`；未设置时优先使用 `KBOT_CONDA_ENV`，否则自动选择 `kbot4`，
   开发机才允许回退 `cube`。通过 `conda run` 解析目标解释器，不要求调用者预先激活环境。
2. 默认开发模式先执行 `"$python_bin" -m pip install -r requirements.txt`，再按确定顺序执行：

   ```bash
   "$python_bin" -m pip install --no-deps -e packages/platform_core
   "$python_bin" -m pip install --no-deps -e packages/platform_clients
   "$python_bin" -m pip install --no-deps -e services/model_serving
   "$python_bin" -m pip install --no-deps -e services/knowledge_core
   "$python_bin" -m pip install --no-deps -e services/agent_runtime
   "$python_bin" -m pip install --no-deps -e services/aiops_agent
   "$python_bin" -m pip install --no-deps -e services/data_query
   "$python_bin" -m pip install --no-deps -e services/main_api
   ```

   在 `data_query` 尚未创建前，脚本必须明确报出缺失工作区成员；不得悄悄跳过。
3. 保留 `--production`：构建各内部包 wheel 到受控目录，再按依赖顺序使用本地 wheel 安装；
   生产环境绝不使用 `-e`。
4. 保留现有 `.env` 初始化逻辑，并将其置于依赖和 editable package 均成功之后；初始化失败
   时不伪称安装成功。
5. 启动脚本和测试命令只使用 `python -m <package>.entrypoints...`，验证它们来自已安装的
   editable/wheel 包，而非工作目录偶然可见的源码。

### 2.3 包元数据与验证

1. 每个内部 `pyproject.toml` 明确声明本包直接依赖的 KBot 内部发行名，例如
   `kbot-platform-core==4.0.0`；不在服务中复制共享依赖。
2. `setuptools` package discovery 限定本服务的顶级 Python 包，避免误打包测试、脚本和其他
   服务源码。
3. 新 `data_query` 包在创建的同一个提交中加入 root workspace、安装成员、生产 wheel 顺序、
   CI/Smoke 和发布清单。
4. 增加 `tests/acceptance/check_workspace_packages.py`：在干净虚拟环境执行安装后导入每个
   包，并确认 `importlib.metadata.distribution()`、版本和模块来源一致；开发模式模块路径应
   指向对应 `src/`，生产模式不得指向仓库源码。

阶段零验收：`bash scripts/deployment/install_workspace.sh` 能用全新环境安装全部内部包；
`python -m` 启动每个服务均可导入共享包；修改任一服务源码后无需再次安装即可被开发模式
加载；`--production` 只从构建的 wheel 安装；不存在通过 `PYTHONPATH`、裸脚本执行或
`sys.path` 注入才能运行的服务。

### 2.4 KBot 标识转换规则

迁移只复用架构和业务行为；任何新增或改动的 KBot 资产必须使用 KBot 标识，不能把
`ammolite` 当作中性前缀复制进来。转换规则如下：

| 资产类别 | KBot 要求 |
|---|---|
| Python distribution | 使用 `kbot-*`，版本保持 KBot 发布版本；不得安装或依赖 `ammolite-*` |
| Python import | 沿用 KBot 已有包名；新增服务使用 `data_query`，共享包仍为 `platform_core`、`platform_clients` |
| 环境变量与配置服务名 | 使用 `KBOT_`、`kbot-*`，例如 `KBOT_DATA_QUERY_*`、`kbot-data-query-api` |
| Oracle 对象 | 使用 `KBOT_*` 表、索引、约束、视图和序列名；不得出现 `AMMOLITE_*` |
| HTTP/OpenAPI/事件 | 使用 KBot service name、错误码、标题、日志字段和事件来源；保留 KBot 的 `/api/v1`、`/internal/v1` 规则 |
| 文档、脚本、容器与发布物 | 使用 KBot 文件名、说明、镜像/服务标签和 wheel 目录；不得引用 Ammolite 部署路径 |

`platform_core`、`platform_clients` 等未带产品前缀的既有 KBot import 是稳定内部名称，
不应为了表面统一改为 `kbot_platform_*`。同理，第三方协议名（OpenAI、MCP、Oracle、
PostgreSQL、MySQL）不做品牌替换。

每次迁移提交必须执行残留扫描：

```bash
rg -n -i "ammolite" packages services database scripts configuration docs tests \
  -g '!docs/product/kbot-ammolite-backend-migration-implementation-plan.md'
```

结果只允许经过审查的“迁移来源说明”；生产代码、DDL、配置、OpenAPI、测试夹具、日志和
用户可见错误中不得出现 `ammolite`。

## 3. 共同实施约束

### 3.1 归属与身份

所有新增业务对象继续以 `domain_id` 为强隔离边界。调用者身份仅使用已由 Main API 和
内部 AuthContext 传递的稳定 `actor_id` 字符串；不得新增用户表、角色表或权限判断。

需要“归属人”概念的记录使用：

```text
domain_id + actor_id
```

其中 `actor_id` 可以为空的系统任务只写 `created_by = "system"`，不得伪造用户记录。

### 3.2 服务边界

- Main API 只做外部 API Key、Domain 上下文、参数校验、错误映射和下游调用。
- 新 Data Query、Agent Runtime、Knowledge Core、Model Serving 各自保留应用服务和
  Repository/UoW；Router 不访问 SQLAlchemy Session 或表。
- 服务间调用使用受 audience 约束的短时 AuthContext JWT；不得转发 Portal API Key。
- Repository 不调用 `commit()`；事务由 UoW 所有者提交。

### 3.3 数据库与契约

- 新表放入 `database/oracle/<service>/`，UUIDv7 使用 `RAW(16)`，时间使用 UTC，JSON
  使用 Oracle 原生 JSON 约束。
- 每阶段同步实体、Canonical DDL、外键/索引、`schema_manifest.json`、OpenAPI、
  `platform_core` contracts、`platform_clients` 与测试；不留下只改 ORM 或只改 DDL 的状态。
- 所有异步任务必须有幂等键、租约、有限重试、可查询终态和不含明文/Prompt 的错误摘要。

## 4. 阶段一：Data Query 服务与双模式问数

### 4.1 目标架构

从 Ammolite 迁移独立 `services/data_query`，但用 KBot 的 `domain_id` 和 Oracle 持久化
替换其 Tenant/PostgreSQL 依赖。服务提供受控数据源、Schema Snapshot、语义模型、查询
计划、执行结果和审计；不接受任意 SQL 作为管理面输入。

新增统一能力名 `data_query`，并在 Agent Definition 中增加：

```json
{
  "data_query_mode": "MCP | SEMANTIC",
  "data_profile_name": "仅 MCP 模式必填"
}
```

`SEMANTIC` 模式的语义模型关系由 Data Query 的 Agent Binding 表管理，不在 Agent JSON 中
重复保存 ID 列表。

`MCP` 表示调用既有 SelectAI/AIReport `profile/user/ask`；`SEMANTIC` 表示调用新的
Data Query Runtime。一个 Agent 在同一时刻只能选择一种模式。两种模式都生成同一
`QUERY_RESULT.v1` Artifact、`data.query.completed` 事件和 ECharts 输入，Response Composer、
Conversation 历史和 Run 详情不得感知底层来源差异。

旧 `mcp_data` capability、路由名称和 `data_profile_name` 在本阶段一次性重命名到统一
`data_query` 契约；不保留双能力或双 Planner 路径。已配置的 Agent 数据通过一次明确的
Oracle 数据修复脚本设置 `data_query_mode = MCP`。

### 4.2 Data Query 领域模型

迁移并按 Domain 改造以下聚合：

| 聚合 | 关键内容 |
|---|---|
| Data Source | 类型、受限 Endpoint、加密凭据引用、状态、能力快照、行版本 |
| Schema Snapshot | 数据源版本、发现对象、DDL/元数据、选择状态、失败原因 |
| Semantic Model | 模型和不可变 Version、Dataset、Dimension、Measure、状态机 |
| Query Policy | 在 KBot 中简化为 Domain 内的执行预算和允许的语义模型；不包含角色/用户选择器 |
| Agent Binding | `agent_id` 与已发布 Semantic Model 的显式绑定 |
| Query Run/Result | 受控 Query Plan、执行状态、列/行结果、截断信息、到期清理和审计 |

数据源凭据不得复用 AIOps 实现。Data Query 自己使用用途隔离的 AES-256-GCM 密钥、
12 字节随机 nonce、`domain_id + data_source_id + credential_version` AAD；明文只出现在
请求处理和数据库驱动调用期间。

### 4.3 服务、接口与运行时

新增：

- `services/data_query/src/data_query/`：management、runtime、schema snapshot、semantic
  model generation、query run 和 result expiry Worker；
- `packages/platform_core/src/platform_core/contracts/data_query/`；
- `packages/platform_clients/src/platform_clients/data_query.py`；
- `database/oracle/data_query/` 与对应 Schema 校验。

内部管理接口至少包括：数据源连通性测试/创建/更新/停用，Snapshot 创建/查询/对象选择、
失败对象重试、受限手工 DDL，语义模型草稿/编辑/校验/提交审核/退回/发布/退役，以及
Agent 绑定。

内部 Runtime 接口只接收 `domain_id`、Agent、已绑定 Semantic Model、自然语言问题和
预算；它生成结构化 Query Plan，经 AST/Dialect Compiler 白名单校验后，以只读数据库
账号执行。禁止调用者直接提交 SQL、连接串、表名或凭据。

首次只移植 Ammolite 已支持的 Oracle、PostgreSQL、MySQL Connector 能力；每个 Connector
必须执行只读校验、Schema allowlist、单语句限制、超时、最大行数和最大结果字节数。

### 4.4 Agent Runtime 接入

1. 将 `MCPDataQuerySkill` 改名为统一的 `DataQuerySkill` 门面。
2. 门面依据 Agent 的 `data_query_mode` 选择 `MCPDataQueryExecutor` 或
   `SemanticDataQueryExecutor`，而不是由 LLM 决定后端。
3. MCP Executor 封装现有 `MCPDataClient`，保持其请求、超时、行数和响应字节限制。
4. Semantic Executor 通过 `DataQueryClient` 调用 Runtime，并把结果规范化为
   `QUERY_RESULT.v1`。
5. Planner 将 `MCP_DATA` 统一为 `DATA_QUERY`；所有能力名称、任务键、事件文档和测试
   在同一提交中切换。
6. 启用校验：`MCP` 必须有 `data_profile_name`；`SEMANTIC` 必须至少有一个有效、已发布且
   属于当前 Domain 的 Agent Binding。

### 4.5 数据库、配置与验收

- 新增 `KBOT_DATA_QUERY_CREDENTIAL_ENCRYPTION_KEY`、`KBOT_DATA_QUERY_CREDENTIAL_KEY_VERSION`；
  初始化脚本生成 Base64URL 32 字节密钥并写入 `.env`，`.env.example` 只保留占位符。
- 新增 `data_query` API、Worker、Connector 和模型生成配置；不使用 AIOps 配置项。
- 为现有 Agent 创建显式 `MCP` 模式数据修复脚本，并在脚本执行前检查列/数据状态。

验收：

1. 同一个 Agent 在 `MCP` 模式下结果与当前 MCP 问数契约一致。
2. `SEMANTIC` 模式只能查询已绑定、已发布语义模型中允许的对象，不能提交任意 SQL。
3. 两种模式均可产生相同版本的 Query Result、引用、图表和 SSE 事件。
4. 禁用数据源、退役模型、跨 Domain 绑定、未发布模型、凭据篡改和超预算查询均被拒绝。
5. Data Query 的凭据、SQL 参数和结果明细不进入日志、事件、幂等记录或异常文本。

## 5. 阶段二：Agent Runtime 能力补足

KBot 已有 Conversation、Run、Task、Artifact、Event，以及 Memory Snapshot、Item、Source、
Index Profile、Job 和 Consolidation Worker 基础。本阶段先逐文件审计差异，不覆盖或平行
重建现有聚合；在现有实现上补齐以下能力：

1. **长期记忆治理**：补齐受控记忆策略、提取、索引、召回、遗忘和会话删除级联中的实际
   缺口，并为已有流程补充分层与回归测试。
2. **记忆策略**：仅允许受控低敏偏好提升为跨会话共享；业务事实、联系方式、身份数据
   默认保持会话或 Agent 范围，不允许模型任意扩大范围。
3. **Specialist**：迁移 Hybrid Retrieval、Data Query、Visualization 三类 Skill；继续
   使用 Manifest 固定输入/输出 Schema，LLM 不得生成任意 Skill、SQL 或 URL。
4. **运行可靠性**：补齐 Worker 任务超时、租约恢复、结果保留期、记忆任务失败隔离和
   可审计公开事件；不暴露模型隐藏推理。
5. **配置校验**：Agent 激活时验证模型目录、Collection、Data Query 模式/绑定与
   Specialist 所需配置。

文件范围：`services/agent_runtime/` 的实体、Repository、UoW、conversation/runtime
application service、memory worker、specialists、internal API；以及 Agent contracts、
Main API 转发和 Agent Runtime OpenAPI。

验收：记忆失败不影响已完成回答；遗忘后不可再召回；运行重启后只接管未完成任务；
DATA_QUERY capability 以及 MCP/SEMANTIC 两种模式的配置错误均在 Agent 激活时得到确定错误；所有 Artifact
可追溯到输入、模型、Skill 版本和 Run。

## 6. 阶段三：Knowledge Core 增强

KBot 已有入库、审批、解析、索引和两阶段检索，本阶段只迁移增量能力：

1. **Bundle Revision 预览**：返回成员、MIME、大小、可读性；源文件经受控二进制流返回，
   绝不返回对象存储 URI。响应使用 `private, no-store`，禁止 MIME 嗅探。
2. **安全渲染规则**：只允许 PDF、常见位图和纯文本内嵌；HTML、SVG 和其他主动内容只能
   下载，不能同源执行。
3. **模型引用反查**：Model Serving 在更新、归档或删除模型前可查询被哪些 Collection/
   Index Profile 引用，以执行一致性检查。
4. **任务唤醒与通知接点**：迁移 Job Wakeup Repository/Listener 抽象，并向后续通知中心
   发表稳定业务事件；通知故障不得阻断解析、索引或清理。
5. **清理一致性**：完善 Collection Purge、Revision/Document 删除与对象存储清理的
   幂等性和可恢复状态。

所有 Collection 范围检查只校验 `domain_id` 和资源归属，不迁移 Ammolite 的 App 或权限
依赖。

验收：跨 Domain 预览返回未找到；预览不泄露存储 URI；被引用模型不能直接删除；通知
发布失败不影响 KC Job 终态；重复 Purge 不产生孤儿记录或错误删除。

## 7. 阶段四：Model Serving 一致性与生命周期

迁移 Ammolite 模型目录的完整管理语义，但保持 KBot 的四个推理进程和 OpenAI 兼容接口。

1. 引入统一的 `ModelServingUnitOfWork`、Repository 和目录实体映射，消除直接跨层访问。
2. 补齐模型 Provider 选项、创建、读取、更新、归档和删除契约；模型名称仍是稳定
   `served_model_name`，不是主键。
3. 目录变更后向 LLM/Embedding/VLM/Visual 进程发布失效通知，按模型粒度卸载/重载，不
   重启全部服务。
4. 删除前检查 Agent Definition、KC Collection/Index Profile、Data Query 语义模型生成
   配置和运行中任务的引用；引用存在时返回确定的冲突信息。
5. 对类别、Provider、模型参数、Embedding 维度、状态转换和并发行版本执行统一校验。

验收：更新后下一次请求使用新配置；归档模型不能被新 Agent/Collection/Data Query 引用；
已有引用阻止删除；并发更新产生行版本冲突；目录事件不包含连接密钥。

## 8. 阶段五：Main API 运行日志与开发运行记录

将 Ammolite 的后端日志查询能力整合进 KBot 的 `development_logs`，不迁移其 Portal 页面。

1. 统一服务目录、事件分页查询、时间/级别/服务/请求 ID/Run ID 筛选和单事件详情接口。
2. 详情可返回结构化字段与受限原始消息；对 Authorization、Cookie、API Key、数据库凭据、
   连接串和大字段递归脱敏。
3. 聚合 Agent Runtime、Knowledge Core、Model Serving、Data Query、Main API 的运行记录，
   保留源服务、请求 ID、关联 Run/Job、发生时间和保留期。
4. 新增查询游标、最大窗口和导出上限；日志查询失败不能影响业务服务启动。
5. 现有 `development_agent_runs` 与 Run 查询对齐为相同关联标识，不复制两套运行事实。

验收：按 `request_id` 和 `run_id` 可关联跨服务事件；分页稳定；敏感值无法通过列表、详情
或导出获取；缺失关联记录返回空结果而非 500。

## 9. 阶段六：无用户/权限依赖的通知中心

迁移通知的 Outbox、Inbox、投递、SSE、偏好、待办和后台任务观察机制，但删除所有用户、
角色、租户、权限、管理员策略和外部渠道依赖。

### 9.1 简化后的数据模型

| Ammolite 概念 | KBot 实现 |
|---|---|
| Tenant | `domain_id` |
| User/Recipient | `actor_id` 或 `system` |
| Role recipient | 不支持；事件发布者必须提供具体 `actor_id` 列表 |
| 用户偏好 | 可选，按 `domain_id + actor_id + event_type` 保存 |
| 管理通知策略/权限 | 不迁移 |
| Inbox | `domain_id + recipient_actor_id` 的站内消息 |

通知中心固定采用 `platform_core.notifications + main_api`：共享包定义事件目录和 Outbox
原语，各业务服务在自己的 UoW 事务中写统一 Oracle Outbox，Main API Worker 负责投影、
隔离重试和 SSE。当前不建立独立 Notification Service，也不维护第二套 Outbox。

### 9.2 必须能力

1. 业务事务内写 Outbox；Dispatcher 幂等投递 Inbox，并保存失败隔离与有限重试。
2. 支持通知摘要、列表、详情、单条/批量已读、SSE 重连和事件去重。
3. 支持后台 Operation/Runs 的关注/取消关注，并为解析、索引、Data Query、Agent Run
   发布完成、失败和需处理事件。
4. 没有 `actor_id` 的系统任务只记录 Domain 级运行事件；不得广播给虚构用户。
5. 通知正文只存安全摘要与资源 ID；详情由原业务服务读取，禁止复制凭据、SQL、文件正文、
   LLM Prompt、完整日志或敏感结果。

验收：重复投递不产生重复 Inbox；SSE 使用事件 ID 续传；失败事件进入隔离后可重试；
无用户目录时系统事件仍可审计但不生成无主通知；删除/遗忘 Actor 标识时可按稳定字符串
清理其偏好与 Inbox。

## 10. 阶段七：组合编排接口（最后实施）

前六阶段稳定后，再在 Main API 增加组合管理服务。该层不是新的领域所有者，而是对已有
内部服务进行一致的创建、修改、校验和查询编排。

### 提供的组合操作

1. 创建/更新 Agent 时原子校验其模型、Collection、Data Query 模式和语义模型绑定；
   失败不留下半配置 Agent。
2. 创建/发布 Collection 时验证解析/Embedding 模型目录和现有 Agent 引用。
3. 创建/发布 Semantic Model 时验证数据源 Snapshot、验证模型与可绑定 Agent。
4. 归档/删除模型、Collection、Semantic Model、数据源时返回所有阻塞引用，或执行显式
   级联停用，不提供隐式删除。
5. 为 Run 提供组合视图：Agent 配置快照、模型版本、Collection、Data Query Source/
   Semantic Model、任务、Artifact、Query Result 和通知摘要。

### 一致性规则

- 不引入跨服务分布式事务。先进行可重复的预校验，再提交领域命令；失败通过幂等键和
  显式补偿状态恢复。
- 所有组合命令必须携带 `Idempotency-Key`；更新携带 `If-Match`/行版本。
- 组合视图是只读 Projection，可延迟一致，但必须标注各来源的更新时间和缺失状态。

验收：任一依赖不可用时不产生静默半配置；冲突响应列出阻塞资源类型和 ID；Run 组合视图
可追溯所有实际使用的版本；重复请求不创建重复绑定、Run 或通知。

## 11. 每阶段交付与总体完成标准

每个阶段合入前必须完成：

1. 领域实体、Repository/UoW、Application Service、API、Client、DDL 和 Schema Manifest
   同步；
2. `docs/openapi/*` 重新生成；
3. 聚焦单元测试、契约测试、Oracle Schema 检查和至少一个真实依赖 Smoke；
4. `python3 -m compileall -q packages services tests` 通过；
5. 残留扫描确认没有旧能力名、旧契约或敏感字段双写；
6. 文档说明配置项、状态机、失败语义、数据保留与恢复步骤。

总体验收以阶段零和七个业务阶段的独立验收全部通过为准。特别是阶段一必须同时证明 MCP 和
SEMANTIC 两种问数模式可运行并产生同一输出契约；阶段七只能在前六阶段的领域对象、
引用反查和通知事件均稳定后开始。
