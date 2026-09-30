# AIOps 首次初始化

AIOps 使用独立的固定业务范围，不复用 KM 的 Domain、账号或产品知识资产：

- Domain：`aiops_portal`
- 初始管理员：`aiopsadmin`
- 内部手册索引集合：`operations-manuals`
- 内部诊断案例索引集合：`diagnosis-cases`
- 固定手册：`services/aiops_agent/resources/knowledge/database-operations-manual.md`

AIOps 在自身 Schema 中保存运维知识资产、不可变版本、适用范围、来源、索引引用和审核记录。
两个内部集合只承担 KC 解析和索引，不作为普通用户直接管理的产品对象。详细合同见
[AIOps 运维知识库详细设计](../proposals/aiops-operations-knowledge-base-detailed-design.md)。

初始化前必须完成平台、模型目录、Knowledge Core、Main API 和 AIOps Agent 的 Schema
及服务部署，并至少启用一个 `CATEGORY=2` 的文本向量模型。脚本不会伪造数据库目标、
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

脚本先在同一 Oracle 事务范围内幂等补齐 App、Domain、用户、角色、权限和固定
索引集合，再以 `aiopsadmin` 登录 Main API，通过“运维知识库”业务接口上传内置手册。
AIOps 先登记资产版本，再驱动 KC 的正式 `user-files` ingestion；KC 批准只允许解析和索引，
不会把版本自动发布为 Agent 可检索知识。文件不会写入 Main API 本地目录。

如 Main API 地址与 `kbot.toml` 的 `[ui].main_api_base_url` 不同，可显式覆盖：

```bash
python3 scripts/db/initialize_aiops.py \
  --main-api-url http://127.0.0.1:18099
```

只读复查完整结果：

```bash
python3 scripts/db/initialize_aiops.py --check-only
```

仅补齐数据库资源、不上传手册：

```bash
python3 scripts/db/initialize_aiops.py --skip-manual-upload
```

初始密码由脚本在终端输出。初始化脚本重复执行会恢复固定初始密码，因此生产环境完成
初始化并修改密码后，不应把它当作日常健康检查使用；日常检查使用 `--check-only`。

当前 AIOps 页面统一使用“运维知识库”，提供手册上传、诊断案例候选、版本审核、发布、退役、
索引状态和检索测试。普通用户不会看到 Collection、Bundle 或 Embedding；内部索引异常只以
业务可理解的处理状态呈现。Agent 只通过 `ops.knowledge.search@1.0.0` 检索已发布版本，
未发布、失败、退役或索引失效的版本不会进入诊断上下文。
