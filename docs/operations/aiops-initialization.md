# AIOps 首次初始化

AIOps 使用独立的固定业务范围，不复用 KM 的 Domain、账号或 Knowledge Core Collection：

- Domain：`aiops_portal`
- 初始管理员：`aiopsadmin`
- 固定 Collection：`operations-manuals`
- 固定手册：`services/aiops_agent/resources/knowledge/database-operations-manual.md`

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

“自动告警诊断最低级别”和“同一告警冷却时间”只约束告警自动触发的诊断，不影响
聊天诊断、人工运行和计划巡检。执行策略不提供独立配置或列表页面；系统仍随 Agent
版本保存不可变 Policy 快照，后续由 Agent 详情和运行记录呈现相关审计信息。

在仓库根目录执行：

```bash
KBOT_CONFIG_FILE=configuration/kbot.toml \
python3 scripts/db/initialize_aiops.py
```

脚本先在同一 Oracle 事务范围内幂等补齐 App、Domain、用户、角色、权限和固定
Collection，再以 `aiopsadmin` 登录 Main API，通过 KC 的正式 `user-files` ingestion
上传手册并批准对应 revision。文件会进入 KC 的解析、切片和索引流程，不会写入
Main API 本地目录。

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

AIOps 页面中的 “Knowledge Core” 提供模型目录选择、KC 模型变更策略提示和运维手册
上传。文本向量或视觉向量模型在已有解析活动后是否允许更换，由 Knowledge Core 服务端
策略决定，前端和 Main API 都不通过硬编码绕过该约束。
