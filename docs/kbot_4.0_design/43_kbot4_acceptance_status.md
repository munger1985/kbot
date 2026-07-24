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
对照。每个数据库子检查还有 30 秒进程级上限。

本次执行时配置的本地 Listener 不可达，因此 Oracle 两项检查明确失败并生成证据；
没有执行 DDL，也没有修改数据库。Listener 恢复后直接重跑即可，不需要修改代码或
放宽超时。

## 尚未达到 RC 的项目

- 在空白 Test Schema 重放 17 份 DDL并验证所有服务的 Repository/UoW；
- KC 上传、解析、检索、Grounding 的跨进程 E2E 与质量数据集；
- Root Document/AIOps、Direct Ops、HITL、审批、报告和 SSE 的完整 E2E；
- Oracle/MySQL 隔离 Target 上的只读诊断与受控 Mutation 演练；
- Secret/依赖扫描、SBOM、镜像签名、负载、Chaos、备份恢复和 APEX 验收；
- 使用干净工作树执行 `--profile rc --require-clean` 并冻结不可变构建物。

因此当前状态是“代码级 T0/T1 基线通过”，不是 Release Candidate 或生产可发布
声明。
