# AIOps 首次初始化

AIOps 使用独立的固定业务范围，不复用 KM 的 Domain、账号或产品知识资产：

- Domain：`aiops_portal`
- 初始管理员：`aiopsadmin`
- 默认运维知识库：`默认运维知识库`
- 默认知识库内部索引集合：`operations-manuals`
- 内部诊断案例索引集合：`diagnosis-cases`

AIOps 在自身 Schema 中保存多个运维知识库、知识资产、不可变版本、适用范围、来源、索引引用和审核记录。
每个用户知识库拥有独立 KC Collection；`diagnosis-cases` 仍是内部固定集合。详细合同见
[AIOps 运维知识库详细设计](../proposals/aiops-operations-knowledge-base-detailed-design.md)。

初始化前必须完成平台、Knowledge Core、Main API 和 AIOps Agent 的 Schema
及服务部署。脚本不会伪造数据库目标、
诊断源、Target、巡检计划或 Agent，这些资源必须根据实际环境配置。执行策略不再
独立创建，而是随 Agent 创建和修改自动生成不可变版本。

创建 Agent 时至少选择一个已启用且可连接的监控源，可选择多个。数据库直连 Target
是可选项：选择后表示允许 Agent 使用该 Target 的只读诊断凭据；未选择时，Agent
仍可使用 Prometheus、Loki 等监控证据，但不会直连数据库。只有选择了 Target 后，
页面才允许勾选“允许人工审批后执行数据库变更”。该开关声明 Agent 的变更权限意图，
可以在执行凭据尚未配置时先保存；真正生成和执行变更仍要求独立执行凭据、系统支持的
动作模板、部署级变更开关和逐条人工审批全部就绪，绝不回退使用只读诊断凭据。

Target 需要配置 1 到 5 级重要程度，5 表示最重要。Agent 的“自动告警诊断最低级别”、
“最低 Target 重要程度”和“同一告警冷却时间”只约束告警自动触发的诊断，不影响聊天
诊断、人工运行和计划巡检。执行策略不提供独立配置或列表页面；系统仍随 Agent 版本保存
不可变 Policy 快照，后续由 Agent 详情和运行记录呈现相关审计信息。

在仓库根目录执行：

```bash
KBOT_CONFIG_FILE=configuration/kbot.toml \
python3 scripts/db/initialize_aiops.py
```

脚本在同一 Oracle 事务范围内幂等补齐 App、Domain、用户、角色、权限、默认运维知识库和
固定诊断案例集合，不自动上传或批准手册。初始化后，用户在“运维知识库”页面创建知识库，
在“运维知识”页面选择知识库上传资料、配置解析模型并完成审批发布。

只读复查完整结果：

```bash
python3 scripts/db/initialize_aiops.py --check-only
```

初始密码由脚本在终端输出。初始化脚本重复执行会恢复固定初始密码，因此生产环境完成
初始化并修改密码后，不应把它当作日常健康检查使用；日常检查使用 `--check-only`。

当前 AIOps 提供运维知识库增删改查、手册上传、诊断案例候选、版本审核、发布、退役和索引状态。
诊断 Agent 与知识问答 Agent 都持久化绑定 1–N 个知识库。普通用户不会看到 Collection、Bundle
或 Embedding；内部索引异常只以业务可理解的处理状态呈现。Agent 只通过
`ops.knowledge.search@1.0.0` 检索绑定知识库内的已发布版本，
未发布、失败、退役或索引失效的版本不会进入诊断上下文。
