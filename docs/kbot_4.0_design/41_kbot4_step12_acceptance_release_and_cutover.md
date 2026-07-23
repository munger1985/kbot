# KBot 4.0 步骤 12：统一验收、发布与切换

## 目标与原则

本步骤是 KBot 4.0 的统一发布门禁，覆盖 Platform、Model Serving、Knowledge Core、Parser、Agent Runtime、Main API/APEX、MCP Adapter 和 AIOps。测试通过必须能由版本化数据集、不可变构建物和机器可读报告证明，不能以一次人工演示替代。

4.0 不兼容 3.x，不做双写、双读、旧 DTO 适配或百分比切流。4.0 可以在独立新表上暗部署和重建数据；正式入口在维护窗口一次切换。Oracle DDL 只前向执行，生产故障优先关闭能力并前向修复，不删除已写 Run、Artifact、审批或执行事实。

## 当前基线与缺口

当前已有：

- `scripts/check_4_0_boundaries.py` 和 `scripts/check_kc_migrations.py`；
- KC/Parser/检索的大量 `unittest` 组件测试；
- Parser Golden Manifest 评测脚本和示例；
- KC 001–007 Migration；
- 本地 `start_kbot.sh/stop_kbot.sh`。

这些不是完整发布系统：现有 Golden Corpus 只是示例；旧 `test_ops_agent.py/test_hitl.py` 只提供场景，不证明新 AIOps 状态机；当前边界脚本尚未覆盖未来 `agent_runtime/aiops_agent/main_api`，Migration 检查也仅覆盖 KC；生产配置示例和本地启动脚本还未包含完整 4.0 拓扑。尚无统一 Test Dependency Lock、真实 Oracle Schema Manifest 校验、Agent/AIOps E2E、质量 Gate、OpenAPI 兼容检查、负载/故障演练和 Release Evidence Bundle。本地启动脚本不用于生产部署。

## 环境与测试数据

至少维护四类隔离环境：

| 环境 | 用途 | 数据与外部依赖 |
| --- | --- | --- |
| Unit/CI | 每次提交的确定性测试 | Fake Clock、Fake Model、内存/Stub Adapter，不访问共享数据库 |
| Integration | Migration、Repository、Client、对象存储 | 每个 Run 独立 Oracle Test Schema；可选 Oracle Free CI 实例 |
| Staging | 完整拓扑、真实模型与 APEX | 生产同版本 Oracle/代理配置，脱敏或合成数据 |
| Production | 单次切换与观测 | 只使用已签名 Release Artifact 和受控配置 |

测试 Schema 名称必须显式包含 Test Run ID，并在创建前验证当前连接不是生产。清理只能删除本次创建的精确 Schema/对象，不能使用宽目录或未解析环境变量。

评测数据集按版本保存 Manifest、Label Schema、内容 Hash、授权与负责人：

```text
quality/
  gates/v4.0.yaml
  datasets/parser/
  datasets/retrieval/
  datasets/routing/
  datasets/aiops/
  baselines/v4.0/
```

文档集覆盖 PDF、Word、PPT、Excel、图片、扫描件、错误标题、短段落、跨页表格、合并单元格、缺失附件和多文档 Bundle。AIOps 集覆盖 Oracle/MySQL、Prometheus/Zabbix/OEM、有/无只读连接、Chat/Alert/Schedule、性能/故障、证据不足和执行结果未知。测试数据不得包含生产密码、Token、未脱敏 SQL 结果或无权分发的文件。

## 统一测试入口

测试依赖放入独立锁定的 Test Dependency Group，不污染 Runtime 依赖。现有 `unittest` 用例可由统一 Runner 执行，新测试按 Marker 分层：

```text
unit, component, oracle, contract, e2e,
quality, security, load, chaos, mutation
```

目标命令：

```bash
python3 scripts/check_4_0_boundaries.py
python3 scripts/check_kc_migrations.py
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_release.py --profile rc --manifest release-manifest.json
```

Runner 启动时必须记录 Python 版本、解释器路径和 Dependency Lock Hash，不依赖开发机的 `python` 别名。`verify_release.py` 最终统一调用迁移、Schema、OpenAPI、测试、评测、安全和构建物校验，并生成机器可读 JSON/JUnit 报告。开发者可以运行子集；Release Candidate 必须运行完整 Profile，禁止只重跑失败用例后拼接旧报告。

## 测试分层

### T0：静态与供应链

- 编译/导入所有 Active 包，运行跨领域依赖守卫；
- 将边界守卫扩展到 Main API、Agent Runtime、AIOps 及其全部 App；
- 校验 Migration 连续性、Checksum、对象命名和禁止旧表引用；
- 比较 SQLAlchemy Entity、Oracle Catalog 与 Schema Manifest；
- 校验 Public/Internal/Executor OpenAPI Snapshot 和 Artifact JSON Schema；
- 校验 `base.toml.example` 覆盖所有必填生产配置且不含 Secret；
- 执行 Secret Scan、依赖漏洞扫描、许可证策略、SBOM 和镜像签名；
- High/Critical 漏洞、Secret、Schema Drift 和禁止依赖必须为零。

### T1：Unit 与领域不变量

使用 Fake Clock/UUID/Model/Provider，覆盖：

- UoW 提交/回滚、乐观锁、幂等键、租约 Fencing 和状态迁移；
- Bundle/Revision/Document/Parse View/Job 生命周期；
- Evidence 规划、检索融合、Citation Grounding；
- Agent Plan Validator、Budget、Delegation 和 Event Cursor；
- AIOps Alert、Run/Task、HITL、Proposal、Approval、Execution、Fire 和 Report；
- Policy、Hash、Grant、脱敏、窗口和 Comparison 判级。

安全不变量不能依赖概率模型：跨 Domain、未审批执行、Token 重放、任意 SQL、错误 Template Hash 和迟到 Lease 写回必须确定性拒绝。

### T2：Component 与真实 Oracle

- 在空 Schema 顺序应用所有 Platform/KC/Agent/Ops Migration；
- 验证列类型、Nullability、PK/UK/FK/Check、函数索引、Vector/Text 索引和 APEX View；
- 再次执行 Migration 必须按 Runner 规则明确拒绝，不能静默跳过；
- Repository/UoW 使用真实 Oracle，验证锁竞争、事务回滚和分页；
- Object Store、Model Client、Monitor Adapter、DB Executor 使用协议级 Test Double 或隔离实例；
- 从备份恢复 Test Schema 后再次运行只读一致性检查。

Migration Principal 与 Runtime Credential 分离。若受 APEX 限制必须共用 Schema Owner，必须记录该风险，并以 Repository 边界检查、数据库审计和禁止 App 自动 DDL 作为补偿；DB Executor 始终不得拥有 KBot Schema 权限。

### T3：契约与消费者验证

对每个方向冻结 Provider/Consumer Fixture：

- Portal → Main API/KC Ingestion；
- Parser Worker → KC Parse Result；
- Agent Runtime → KC Discovery/Evidence；
- Main API → Agent Runtime/AIOps；
- Agent Runtime → AIOps Delegation；
- AIOps → Monitor/Model/KC/DB Executor；
- Main API → APEX/SSE Client。

验证未知字段前向容忍、必填字段、枚举、错误码、Idempotency、ETag、AuthContext Audience、Artifact Hash 和最大 Payload。内部 OpenAPI 不暴露到外网；Public API 不返回 Entity、SecretRef、SQL、内部 URI 或租约字段。

### T4：完整端到端

必须从外部入口验证：

1. 普通文件与多文档 Bundle 上传 → Parse/Profile/Index → Discovery/Evidence → Grounded Answer；
2. KM Portal 来源 → Manifest/附件部分失败 → 可检索 Bundle 与正确 Reference Card；
3. Root Document/Data/AIOps/HYBRID 路由 → Delegation → 单父 SSE → 类型化引用；
4. Direct AIOps Chat 在数据库不可连接时多轮 SQL 补证；
5. Alert Webhook 验签/去重 → 自动诊断且不等待用户；
6. Inspection Fire → 日/周报告；
7. Advisory 人工结果 → Verification/Comparison；
8. 测试 Target 上逐命令审批 → At-most-once Execution → Result/Report。

E2E 断言状态、Artifact、引用、审计和副作用，不只匹配最终自然语言。

## 解析、检索与回答质量 Gate

LLM Judge 只能辅助分析，不能成为唯一发布判定。Parser/检索使用人工标注与确定性计算；回答同时执行引用合法性和人工抽检。模型 ID、Prompt Hash、Embedding 模型、参数和评测数据 Hash 必须冻结。

建议第一版将以下值写入 `quality/gates/v4.0.yaml`，经首轮人工基线评审后锁定：

| 指标 | 建议初始 Gate |
| --- | --- |
| Bundle Recall@10 | `>= 0.90` |
| 列表问题 Precision@10 | `>= 0.90` |
| Evidence Recall@20 | `>= 0.85` |
| 有效 Citation Precision | `1.00` |
| Unsupported Factual Claim Rate | `<= 0.01` |
| 支持格式 Locator Exact Accuracy | `>= 0.98` |
| 关键格式 Parser Hard Failure Rate | `<= 0.01` |

同时记录 Heading/Section Boundary F1、Table Cell Accuracy、nDCG、MRR、答案覆盖率、Bundle 去重准确率和各格式切片。总体通过不能掩盖关键切片失败；任何切片相对批准基线下降超过配置的百分点必须阻断。Collection 解析与检索必须使用同一冻结 Embedding Model Identity，不只比较维度。

## Agent 与 AIOps 质量 Gate

Router 使用人工标注意图集，记录 Macro-F1、HYBRID 召回、澄清率和越权拦截率。建议支持意图 Macro-F1 `>=0.95`；高风险 AIOps 意图误路由为普通对话/问文必须为零。

AIOps 不按“回答看起来合理”验收。每个 Scenario 声明：

```text
expected_target
allowed_tools and forbidden_tools
required_observations
acceptable_root_cause_family
maximum_root_cause_level
expected_gaps
allowed_actions and risk
expected_report/comparison result
```

门禁要求：

- Target/Domain 错配、越权工具和未经审批 Mutation 为零；
- `CONFIRMED/PROBABLE/POSSIBLE` 不得超过证据上限；
- Oracle/MySQL Catalog 的版本/能力选择完全匹配；
- Alert/Schedule 不创建人工 SQL HITL；
- 同一 Proposal 最多一次数据库投递；
- `UNKNOWN` 不自动重试 Mutation；
- Composer 不改变根因、风险、审批、Execution 或 Comparison 结果。

真实模型评测至少重复运行配置规定的次数，报告最差值和方差；CI 使用录制/合成响应保证确定性，RC 在 Staging 使用正式模型验证真实表现。

## 安全、负载与故障演练

安全测试至少覆盖 IDOR/跨 Domain、CSRF、SSRF、Webhook 重放、Prompt Injection、恶意文件、Zip Bomb、对象 URI 越权、SQL/命令注入、Grant/Token 重放、日志泄密、SSE 越权恢复和依赖漏洞。

负载目标来自版本化 SLO，不在测试代码中散落常量。分别压测：

- Intake 文件大小/并发与 Parser 队列；
- Discovery/Evidence p50/p95/p99 和模型预算；
- Agent/AIOps READY Task、Delegation 和 SSE 长连接；
- Webhook 风暴、Scheduler 多副本和 Report 生成；
- Oracle 连接池、Outbox/Inbox 积压及恢复速度。

Chaos 测试在每个外部调用前后和每个提交点终止 API/Worker/Scheduler/Executor，注入 Oracle/对象存储/Model/Monitor 网络故障、重复/乱序回调和时钟偏差。验收目标是“不丢事实、不重复副作用、可恢复或明确终止”，不是所有请求都成功。

Mutation 测试只允许隔离 Oracle/MySQL Target、白名单模板和测试凭据。Production 数据库不作为自动化 Mutation 测试环境。

## Release Evidence Bundle

每个 RC 生成不可修改的证据包：

```text
commit SHA and signed tag
source/archive/image digest
dependency lock, SBOM and signatures
migration manifest and SHA-256
Oracle schema manifest
OpenAPI/Artifact schema hashes
configuration schema/example hash
unit/integration/e2e/quality/security/load reports
data rebuild and reconciliation report
known limitations and accepted risks
approvers, timestamps and environment
```

构建物必须 Build Once、Promote Many；Staging 通过的同一镜像 Digest 才能进入生产，不能在生产重新 `pip install` 或现场修改源码/SQL。

## 发布 Gate

| Gate | 通过条件 |
| --- | --- |
| G0 Design Freeze | 设计、Owner、Schema、API、Runbook 和风险已评审 |
| G1 Build | T0 全通过，构建物签名，Schema/OpenAPI 无漂移 |
| G2 Correctness | T1/T2/T3 全通过，零 Critical Invariant 失败 |
| G3 Quality/Security | Parser/Retrieval/Agent/AIOps Gate 与安全扫描通过 |
| G4 Staging | T4、负载、Chaos、备份恢复和 APEX 验收通过 |
| G5 Cutover Rehearsal | 在生产等价副本完成全量重建、最终增量和切换演练 |
| G6 Production | Smoke、指标、审计和业务验收通过，进入 Soak |

Waiver 必须包含指标、影响、Owner、到期时间和补救计划；跨 Domain、未审批执行、数据丢失、无效 Citation、Critical 漏洞和不可恢复 Migration 不允许豁免。

## 数据重建与对账

KC 从原始业务来源和原始文件重建，不能把旧 TxtChunk、旧向量或旧 Parser 结果迁入新 Evidence。对每个 Source 记录：

```text
source system/type/id
source revision and content hash
expected/accepted/failed Bundle count
Document/Version count
Parse/Profile/Index terminal count
current Discovery/Evidence count
failure category and replay status
```

失败项可重放，但发布前必须分类为已恢复、明确不支持或经业务确认缺失。AIOps 不迁移旧运行内存、旧审批或执行状态；Target、Monitor、Policy、Inspection Plan 通过审查后的配置导入重新创建。Agent Run 历史不迁移。

## 生产切换顺序

1. 冻结 RC、Migration、配置 Schema、APEX Export 和 Release Evidence；
2. 验证备份可恢复，执行生产 Preflight；
3. 由 Migration Principal 前向部署 Platform/Model/KC/Agent/Ops DDL 和 APEX View；
4. 暗部署 Model Serving、KC API/Parser/Projection、AIOps API/Worker、只读 Executor、Agent Runtime 和 Main API；外部路由仍指向旧入口；
5. 保持 Mutation Kill Switch 关闭、Inspection Plan 暂停、Monitoring 新入口关闭；
6. 从原始来源全量重建 KC，运行对账和只读 Smoke；
7. 暂停旧上传/配置写入口，记录 Freeze Watermark，处理最终来源增量；
8. 原子切换 Portal/APEX/反向代理到 v4；旧 API 不再路由；
9. 验证上传、检索、Root SSE、Direct AIOps、HITL、Report 和 APEX；
10. 启用 Monitoring Intake，再启用已审核 Inspection Plan；
11. Production 初期保持 AIOps `ADVISORY`；Mutation 作为独立 Gate 启用。

不做 3.x/4.0 百分比流量或同一业务写入双发。当前 `main` 分支不因 RC 通过而自动修改；Release 从受保护 `4.0` 分支生成，合并或调整默认分支必须另行明确授权。

## Mutation 分级启用

```text
M0  全局 Kill Switch OFF
M1  Staging Advisory + 隔离 Target Mutation 测试
M2  Production Advisory
M3  Production Canary Target + 极小 Action Allowlist
M4  按 Target/Policy 扩大 AGENT_EXECUTE
```

每次生产命令仍需一次显式审批；发布 Gate 不能代替运行时 Approval。任一异常立即关闭全局 Kill Switch；已发出的命令继续按 Execution/UNKNOWN 对账，不伪造取消或自动回滚。

## 失败处理与回退

在入口切换前，任何 Gate 失败都中止发布；已部署的前向 DDL保留，修复后继续，不 Drop 半成品表。

入口切换并产生 4.0 新模型写入后：

- 先关闭 Mutation、Scheduler、Webhook 或受影响写入口；
- 保留 Run/Artifact/Inbox/Outbox 和来源 Freeze Watermark；
- 使用相同版本契约前向修复并重放幂等任务；
- 不让 3.x 接管同一请求，不把 v4 数据反写旧表；
- 无法保证写安全时进入只读/维护模式。

因此初始 4.0 切换没有自动“一键回滚到 3.x”。真正的风险控制来自切换前演练、写入 Freeze、可恢复任务和能力级 Kill Switch。

## Soak 与旧表退出

生产 Soak 至少覆盖配置规定的完整日报/周报周期，默认建议 14 天。期间要求：

- 所有预期调用方已迁移，旧 API 无成功响应；任何残留尝试均被监控和定位；
- 无旧表写入、旧 Worker 轮询或兼容 Import；
- 无未解释数据差异、越权事件、重复副作用或持续 SLO 超限；
- 所有 Failed/UNKNOWN/Partial 项已分类并有 Owner；
- 备份、审计、报告和 Runbook 可用。

旧代码、入口、配置和部署资源必须在开发阶段完成删除，不能把 Soak 当作代码清理阶段。Soak 通过后，旧表先导出归档并撤销写权限。物理 Drop 是单独的破坏性 Migration，必须列出精确对象、依赖和恢复介质，并在执行时再次获得明确批准，不能作为应用启动或普通发布脚本的一部分。

## 完成定义

4.0 只有在 G0–G6 全部通过、生产入口只使用 v4、数据重建对账完成、所有关键安全不变量为零失败、四类 AIOps 入口和 Root/Document 链路可恢复、Mutation 默认关闭且可独立受控启用时，才算发布完成。
