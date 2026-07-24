# KBot 4.0 当前验收状态

## 已完成门禁

截至 2026-07-24，开发者 Profile 已形成可重复执行的离线门禁：

```bash
conda run -n cube python scripts/verify_release.py \
  --profile developer \
  --output /tmp/kbot4-release-evidence.json
```

当前检查覆盖 Active Package 编译、架构依赖边界、五个服务的 Oracle DDL 静态
契约、Entity 表所有权、14 个进程的 App/配置/端口/启停覆盖、AIOps 诊断目录和
全量 Unit/Component/Contract 测试。配置门禁还会逐一核对 18 组实际 TOML 与
Example、开发/生产合并结果、五个服务配置模型和 `.env.example` Secret 声明。
OpenAPI 门禁覆盖 Main API、KC、Agent Runtime、四个模型进程和 AIOps 的 10 个
快照，并阻止 Public/Internal 路径越界和未冻结的契约漂移。
供应链基线要求直接依赖精确锁定且无重复，核对直接依赖 CycloneDX SBOM，并扫描
全部受 Git 跟踪文件中的常见 Secret 与敏感文件类型。
证据文件记录 Commit、Branch、解释器、Dirty Path、检查结果，以及 DDL、配置
样例、进程拓扑、OpenAPI、依赖声明和直接依赖 SBOM Hash。它不读取或记录环境
变量、数据库密码和 Provider Secret。

## Oracle 集成状态

真实 Oracle 验收使用：

```bash
conda run -n cube python scripts/verify_release.py \
  --profile developer \
  --oracle \
  --output /tmp/kbot4-release-evidence-oracle.json
```

Oracle Profile 先以 3 秒 TCP Preflight 检查配置的 Listener，再核对共享 Schema 中
全部 KBot 表、视图、对象类型和 VALID 状态，最后执行 AIOps Entity/Catalog 逐列
对照和可自动清理的 Repository/UoW Smoke。

2026-07-24 复验时 `KBOTDEV@KBOT4` 已恢复：53 个规范表/视图全部存在。列级检查
发现早期建库遗留四个 AIOps 缺列，受影响表均为空；已按规范 DDL 最小补齐
`KBOT_OPS_CHANGE_PROPOSAL.ROW_VERSION`、
`KBOT_OPS_EXECUTION.DEADLINE_AT`、
`KBOT_OPS_RUN.SOURCE_PROPOSAL_ID` 和
`KBOT_OPS_RUN.SOURCE_RESULT_ARTIFACT_ID`，同时补齐关联约束、索引并重建 10 个
AIOps 投影视图。修复后 21 张 AIOps 表逐列与 Entity 一致。
全服务检查进一步核对 Platform、Model、KC、Agent Runtime 与 AIOps 共 53 张表、
1001 列；KC Document Version 的 `RECEIVED_AT` 已补齐 Entity 映射，Agent
Definition 的 `DO_RERANK` 也已按规范 DDL 补入开发库并通过列级核对。

Oracle Profile 现在还执行跨服务 UoW、AIOps Persistence 和完整 Run 内核 Smoke，
覆盖显式提交、漏提交回滚、跨 Run 双 Worker `SKIP LOCKED`、租约栅栏、事件序列、
并发幂等、取消和租约接管。`http://localhost:9161/metrics` 已通过结构化校验：
67 个 HELP/TYPE、108 个 Sample、69 个指标族，其中 29 个数据库指标族。

## 尚未达到 RC 的项目

- 在空白 Test Schema 重放 21 份 DDL 并验证所有服务的 Repository/UoW；
- KC 上传、解析、检索、Grounding 的跨进程 E2E 与质量数据集；
- Root Document/AIOps、Direct Ops、HITL、审批、报告和 SSE 的完整 E2E；
- Oracle/MySQL 隔离 Target 上的只读诊断与受控 Mutation 演练；
- Secret/依赖扫描、SBOM、镜像签名、负载、Chaos、备份恢复和 APEX 验收；
- 使用干净工作树执行 `--profile rc --require-clean` 并冻结不可变构建物。

因此当前状态是“代码级 T0/T1 基线通过”，不是 Release Candidate 或生产可发布
声明。
