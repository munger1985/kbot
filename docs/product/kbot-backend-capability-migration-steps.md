# KBot 后端能力迁移步骤

## 1. 文档用途

本文把《KBot 后端能力补全迁移实施计划》拆成可按顺序提交、验证和部署的工作批次。
详细的数据模型、接口和代码设计见《KBot 后端能力迁移详细实施方案》。三个文档的关系为：

1. 总计划定义范围、原则和最终完成标准；
2. 本文定义实施顺序、每批输入输出、验收门和部署顺序；
3. 详细方案定义具体目录、契约、表、状态机和测试矩阵。

除明确的来源比对说明外，迁移结果中的包名、环境变量、Oracle 对象、服务名、日志、
OpenAPI 和文档标识全部使用 KBot，不得保留 Ammolite 产品标识。

## 2. 全局迁移规则

- 严格执行 `S0 → S1 → S2 → S3 → S4 → S5 → S6 → S7`，验收门未通过不得开始下一阶段。
- AIOps、多租户、用户目录、RBAC、SSO、Portal 前端均不在范围内。
- 每批先修改共享契约，再修改服务实现，随后修改 Main API 和 Client，最后生成 OpenAPI。
- Oracle Canonical DDL、ORM、Schema 检查必须在同一批完成。
- 不允许临时 `PYTHONPATH`、跨服务源码导入、Router 直接访问数据库或 Repository `commit()`。
- 不以 `compileall` 或静态 DDL 检查替代真实 Oracle/HTTP Smoke。
- 工作区可能已有未提交修改；每批只修改本阶段文件，不覆盖无关改动。

## 3. S0：工作区与安装方式收口

### S0.1 建立基线清单

输入：当前 KBot 工作树和 Ammolite 对应服务。

操作：

1. 保存 `git status --short`、内部包清单、服务入口、端口和 Oracle DDL 顺序。
2. 扫描所有 `sys.path`、`PYTHONPATH`、跨服务相对导入和直接源码启动方式。
3. 记录当前可运行的单元、契约、Acceptance 和 Smoke 命令，失败项单独标记为基线问题。
4. 建立迁移残留扫描，排除本迁移文档后检查 `ammolite` 标识。

输出：基线报告，不改业务行为。

### S0.2 创建 Data Query 最小包骨架

阶段零安装清单要求所有成员真实存在，因此先创建不含业务实现的最小包：

- `services/data_query/pyproject.toml`；
- `services/data_query/src/data_query/__init__.py`；
- `services/data_query/src/data_query/entrypoints/__init__.py`；
- 包版本 `4.0.0`，发行名 `kbot-data-query`，Python import 名 `data_query`。

此批不得复制 Ammolite 实体、路由或 PostgreSQL 依赖。

### S0.3 对齐 workspace 和 editable 安装

1. 根 `pyproject.toml` 增加 KBot workspace members，包含所有 packages/services。
2. 所有子包限定 `src` package discovery 和自己的顶级包名。
3. 重写 `scripts/deployment/install_workspace.sh`：
   - 默认开发模式安装第三方依赖后，逐包执行 `pip install --no-deps -e`；
   - `--production` 逐包构建 wheel，并只从本地 wheel 目录安装；
   - 支持 `KBOT_PYTHON`；
   - 所有包安装成功后才初始化 `.env`。
4. 启动脚本取消源码路径拼接，统一使用 `python -m <package>.entrypoints.<process>`。
5. 增加 `tests/acceptance/check_workspace_packages.py`。

### S0 验收门

```bash
bash -n scripts/deployment/install_workspace.sh
bash scripts/deployment/install_workspace.sh
python3 tests/acceptance/check_workspace_packages.py
python3 -m compileall -q packages services tests
rg -n "sys\.path|PYTHONPATH" packages services scripts tests
```

开发环境修改源码后无需重装即可导入；生产 wheel 安装不能从仓库源码解析模块。

## 4. S1：Data Query 完整迁移与双模式问数

### S1.1 移植纯领域契约

先迁移不依赖 FastAPI、SQLAlchemy、数据库驱动的部分：

- 状态机；
- `DataQueryPlanV1` 及 Filter/Measure/OrderBy；
- Management/Runtime DTO；
- 错误码与错误类型；
- Connector 抽象和 Query Budget。

转换要求：`tenant_id → domain_id`，UUID 用户字段改为稳定字符串 `actor_id`，移除
Role/User selector、App、Permission 依赖，所有产品标识改为 KBot。

### S1.2 建立 Oracle Schema 和 ORM

按以下顺序新增 Canonical DDL：

1. `001_data_sources.sql`：数据源与加密凭据；
2. `002_schema_snapshots.sql`：Snapshot 与对象采集任务；
3. `003_semantic_models.sql`：模型、Version、生成任务、验证样例；
4. `004_bindings_policies.sql`：Domain 策略和 Agent Binding；
5. `005_query_runtime.sql`：Run、Execution、Result、Event；
6. `006_audit_views.sql`：审计、外键、索引、只读视图和 Schema Version View。

同时完成实体、Repository、UoW、Manifest 和 `check_oracle_schema.py` 扩展。Oracle
函数唯一索引用于“每个语义模型仅一个 ACTIVE Version”等条件唯一约束。

### S1.3 数据源与凭据

1. 实现 Oracle/PostgreSQL/MySQL Endpoint 校验和连接测试。
2. 实现 Data Query 专用 AES-256-GCM 凭据加密，不复用 AIOps 密钥或表。
3. 实现数据源创建、更新、停用和凭据轮换；详情不返回凭据引用、密文或用户名。
4. `.env` 初始化生成 `KBOT_DATA_QUERY_CREDENTIAL_ENCRYPTION_KEY` 和版本。
5. 日志、异常、审计和幂等记录做敏感信息测试。

### S1.4 Schema Snapshot Worker

实现 `REQUESTED → DISCOVERING → WAITING_SELECTION → CAPTURING → READY/PARTIAL_READY/FAILED`：

1. 发现 allowlist Schema 下的表和视图；
2. 用户选择对象后创建逐对象采集任务；
3. 单对象失败可重试或使用受限手工 DDL；
4. Snapshot 内容、连接器版本和 Hash 不可变；
5. 新 Ready Snapshot 将旧 Snapshot 标为 `SUPERSEDED`，但不删除历史引用。

### S1.5 语义模型生命周期

1. 从 Snapshot 确定性生成 Dataset、Dimension、Measure 基础结构。
2. AI 仅补充显示名、同义词、敏感级别和警告，不得改变物理映射、类型、聚合或状态。
3. 实现 DRAFT、REVIEW、ACTIVE、REJECTED、RETIRED 状态机和行版本并发控制。
4. 实现 Verified Query、模型校验任务和发布前检查。
5. 一个语义模型最多一个 ACTIVE Version。

### S1.6 Runtime 查询执行

执行链固定为：

```text
Planning Context
  → LLM 生成 DataQueryPlanV1
  → 领域校验
  → Connector 编译参数化只读 SQL
  → Preflight
  → Worker 租约执行
  → Result/Events/Audit
```

禁止 API 接收任意 SQL。运行时必须限制 Schema、单语句、只读操作、行数、字节数、超时、
并发和结果保留期。

### S1.7 Agent Runtime 双模式切换

1. Agent 契约新增 `data_query_mode: MCP | SEMANTIC`。
2. 将 capability/Planner 路由统一为 `data_query`/`DATA_QUERY`。
3. 把现有 MCP 实现封装成 `MCPDataQueryExecutor`。
4. 新增 `SemanticDataQueryExecutor` 和 `DataQueryClient`。
5. `DataQuerySkill` 根据 Agent 配置确定 Executor，LLM 无权选择 Provider。
6. 两种 Executor 统一输出 `QUERY_RESULT.v1`、引用、图表输入和 SSE 事件。
7. 对现有 `mcp_data` Agent 执行一次性数据修复，设置 `data_query_mode=MCP`。

### S1.8 Main API 和契约收口

增加 Data Source、Snapshot、Semantic Model、Binding、Verified Query、Audit 和 Runtime
转发接口；Main API 不访问 Data Query 表。重生成 Data Query internal OpenAPI 和 Main API
public OpenAPI，并删除旧 `MCP_DATA` 契约残留。

### S1 验收门

- MCP 回归 Smoke；
- Semantic 模式真实 Oracle/PostgreSQL/MySQL 至少各一次连接测试，Oracle 至少一次完整查询；
- 跨 Domain、禁用源、未发布模型、超预算、超时、取消和凭据篡改测试；
- 两模式同一 `QUERY_RESULT.v1` Contract Test；
- Worker 崩溃租约接管与幂等重放测试；
- Oracle 全量建库与实体映射检查。

S1 未达到双模式运行态验收，不得开始 Agent Runtime Specialist 补足。

## 5. S2：Agent Runtime 差异补足

### S2.1 先审计、后迁移

KBot 已有 Conversation、Run、Task、Artifact、Event、Memory Snapshot/Item/Job 和 Worker。
先逐文件比较，不重建同名聚合。输出分为：完全相同、KBot 已扩展、Ammolite 独有、因
Tenant/Permission 被排除。

### S2.2 迁移实际缺口

1. 迁移并适配受控 `memory_policy.py`；补齐跨会话低敏偏好提升规则。
2. 补齐 Hybrid Retrieval、Data Query、Visualization Specialist。
3. 增强任务超时、租约恢复、保留期和公开事件；不得暴露隐藏推理。
4. Agent 激活时一次校验模型、Collection、Data Query Mode 与 Binding。
5. 只定义通知事件 Publisher 接口；S6 前使用显式 Null Publisher，不写伪 Outbox。

### S2 验收门

现有聊天、知识问答、MCP 问数全部回归通过；Semantic 问数和 Hybrid/Visualization 新增
测试通过；记忆失败不影响回答；遗忘后不再召回；Worker 重启不重复完成 Task。

## 6. S3：Knowledge Core 增强

### S3.1 Revision 和源文件预览

1. 增加 Preview Application Service 和内部 Router。
2. 查询链必须同时校验 Domain、Collection、Bundle、Revision、Document Version 归属。
3. 列表返回元数据；内容使用流式响应，不暴露对象 URI。
4. 设置 `private, no-store`、`nosniff` 和安全 Content-Disposition。

### S3.2 模型引用与清理一致性

1. 新增 Collection/Index Profile 的模型引用反查接口。
2. 完善 Collection Purge Repository 和对象存储清理补偿状态。
3. 重复 Purge、进程中断恢复和旧解析产物清理均做幂等测试。
4. 接入 S2 定义的通知 Publisher 接口，S6 再启用真实 Outbox。

### S3 验收门

预览越权、主动内容、Range/大文件、对象不存在和存储故障测试通过；模型引用反查能阻止
错误删除；Purge 中断后可恢复且不产生孤儿对象。

## 7. S4：Model Serving 一致性与生命周期

### S4.1 收口持久化和管理契约

1. 引入唯一 Model Serving UoW，移除重复 Repository 路径。
2. 补齐 Provider Options、创建、更新、归档、删除和行版本。
3. 管理路由调用 Application Service，不直接访问 Model Pool 或 Session。

### S4.2 引用检查和缓存失效

1. 聚合 Agent Runtime、Knowledge Core、Data Query 三方引用结果。
2. 更新/归档/删除前返回结构化阻塞引用。
3. 目录变更发送模型粒度失效事件；四类模型进程安全卸载或下一次请求重载。
4. Embedding 维度或向量空间不兼容时禁止原地变更。

### S4 验收门

并发更新、已有引用删除、归档后新绑定、缓存失效、模型进程重启和密钥脱敏测试通过；
OpenAI 兼容推理回归不变。

## 8. S5：Main API 运行日志与开发运行记录

### S5.1 升级现有接口

保留 KBot `development_logs` 所有权，迁移 Ammolite 的 bounded-tail 搜索与详情能力，不
创建第二套 `runtime_logs` 路由：

1. 服务目录；
2. 事件游标分页；
3. 时间、级别、关键字、request/trace/error/run/job ID 和 HTTP 状态筛选；
4. 单事件原始详情和 Traceback；
5. 统一递归脱敏和读取上限。

### S5.2 运行记录关联

让 `development_agent_runs` 使用 Agent Runtime 的 Run 事实，并关联 KC Job、DQ Run 和
模型调用标识。列表不复制完整 Artifact、Query Result 或日志正文。

### S5 验收门

跨服务关联、日志轮转、超大文件、损坏 JSON 行、全部级别取消、详情脱敏和路径穿越测试
通过；日志目录缺失不得阻止 Main API 启动。

## 9. S6：无用户权限的通知中心

### S6.1 固定归属架构

采用 `platform_core.notifications + main_api`：

- 共享包提供事件目录、Outbox DTO/Repository 和投影契约；
- 各业务服务在自己的 UoW 事务中写同一 Oracle `KBOT_NOTIFICATION_OUTBOX`；
- Main API Worker 投影 Inbox、Work Item 和 Background Operation；
- Main API 提供查询、已读、关注和 SSE。

不创建独立 notification service，不迁移 Role/Permission recipient、管理策略或外部渠道。

### S6.2 Oracle Schema 与投影

新增 `database/oracle/platform_core/003_notifications.sql`，包含 Outbox、Inbox、Preference、
Work Item、Background Operation、Operation Watch 及条件唯一索引。收件人为明确的
`domain_id + actor_id`；无 Actor 的系统事件只保留 Operation/Audit，不生成 Inbox。

### S6.3 事件接入

首批只启用：

- Agent Run completed/failed/input-required；
- KC ingestion completed/partial/failed 和 purge completed/failed；
- Data Query snapshot selection/complete/failed、semantic model generation/validation、
  query run complete/failed；
- Model catalog archive/delete blocked 或 reload failed。

移除所有 AIOps、License、API Key、Tenant、Role 和 Permission 事件。

### S6 验收门

真实业务事务到 Outbox、投影、Inbox、SSE 的 E2E 通过；重复、乱序、断线重连、隔离重试、
Actor 缺失和跨 Domain 测试通过。仅单元测试通过不能判定阶段完成。

## 10. S7：后台组合编排接口

### S7.1 建立引用图和只读 Projection

先实现模型、Collection、Agent、Semantic Model、Data Source 与 Run 的跨服务引用查询，
每个节点包含来源服务、资源版本、更新时间和缺失状态。

### S7.2 实现组合命令

按 `PRECHECK → COMMAND → VERIFY/COMPENSATE` 实现：

1. Agent 创建/更新组合校验；
2. Collection 创建/模型更新组合校验；
3. Semantic Model 发布和 Agent Binding；
4. 资源归档/删除阻塞引用查询；
5. Run 组合详情。

全部命令使用 `Idempotency-Key` 和行版本；不使用跨服务分布式事务，不隐式级联删除。

### S7 验收门

注入每个下游超时/409/500，确认不产生静默半配置；重放请求不产生重复对象；组合 Run
视图可追溯实际模型、Collection、Semantic Model、Query Result、Artifact 和通知。

### S7 实施记录（2026-08-04）

- Main API 已增加 Agent、Collection、Semantic Model 发布/绑定组合命令，以及模型、
  Collection、Agent、Semantic Model、Data Source、Run 引用图和停用前检查；
- Agent 组合创建支持同时提交 Collection 与 SEMANTIC 问数绑定，先以 DRAFT 建立资源和绑定，
  再按行版本启用；MCP 路径按本轮范围暂不扩展；
- `KBOT_COMPOSITION_RECEIPT` 已持久化预分配资源 ID、请求 Hash、命令状态、验证摘要和恢复错误；
  并发 `PRECHECKING` 请求不会重复发送命令，提交结果不确定时进入
  `COMPENSATION_REQUIRED`，相同幂等键只执行验证；
- Collection 模型更新已补齐 Knowledge Core 行版本校验和反向 Agent Binding 查询；
- Run 组合视图只返回模型、知识库、问数结果结构、Artifact Hash/provenance、Evidence 与通知，
  不复制结果行、提示词或凭据；
- 已通过故障注入、并发重放、完整 Run 追踪、OpenAPI、Oracle 全量 DDL、Entity 所有权及真实
  Oracle Receipt 恢复 Smoke。真实 Smoke 创建的测试 Receipt 已在结束时清理。

## 11. 最终收口与部署顺序

### 11.1 全量验证

```bash
python3 -m compileall -q packages services tests
python3 tests/acceptance/check_workspace_packages.py
python3 tests/acceptance/check_oracle_schema.py
python3 -m pytest tests/unit -q
python3 -m pytest tests/contract -q
git diff --check
```

另执行所有新增 Smoke，并确认排除迁移文档后没有 `ammolite` 标识、旧 `MCP_DATA` 能力名、
PostgreSQL 专用 ORM 类型或明文凭据残留。

### 11.2 部署顺序

1. 停止受影响 Worker；
2. 备份目标 Oracle Schema 和当前 `.env`；
3. 按 Platform Core、Data Query、Agent Runtime、Knowledge Core、Model Serving 的 DDL
   顺序执行已审核脚本；
4. 安装生产 wheels；
5. 启动 API，再启动 Worker；
6. 运行 Ready Probe、MCP/SEMANTIC 问数、KC Preview、模型引用、日志、通知 Smoke；
7. 完成 Agent MCP 模式数据修复后再开放写流量。

数据库脚本采用前向修复，不以删除新表或回滚已写业务数据作为普通回滚手段。应用回退前
必须确认旧版本不会写入或误读新增契约；否则保持新 Schema，修复应用并重新部署。
