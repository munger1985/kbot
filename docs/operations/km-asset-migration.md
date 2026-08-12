# KM Asset 迁移与运行说明

## 目标架构

KM Asset 已迁移为 KBot 4.0 独立 App，应用标识为 `km_asset`。外部 Portal 只调用
Main API 的 `/api/v1/apps/km-asset/*`；MetaDB、SharePoint、Knowledge Core 和 Data
Query 都由 KBot 内部服务调用，不向浏览器暴露内部凭据或 `/internal/v1` 路由。

同步链路如下：

1. `km_asset_app` 定时读取 MetaDB 中 `processed=N` 的记录。
2. 原始 JSON 和规范化字段写入 `KBOT_KM_ASSET`，每次来源变更生成不可变 Revision。
3. Worker 下载记录声明的全部 SharePoint 附件；任一附件失败时不提交不完整 Bundle。
4. 完整 Bundle 通过内部服务身份提交给 Knowledge Core。
5. KC 解析完成后，Worker 将 MetaDB 状态更新为 `Y`；最终失败时更新为 `F`。
6. 对失败 Asset 调用重试接口后，Worker 先把 MetaDB 状态恢复为 `N`，再重新下载和入库。

## 建库与部署

空白 Schema 使用标准初始化器并启用 `km_asset_app`：

```bash
python scripts/db/apply_oracle_schema.py --help
```

既有测试 Schema 可在 SQL Developer 中直接执行：

```sql
-- 先执行 database/oracle/km_asset_app/001_km_asset.sql
-- 再按 database/oracle/main_api/002_access_control.sql 中 km_asset 项补充权限。
```

安装工作区后，`start_kbot.sh` 会启动：

- `km_asset_app.entrypoints.api`，端口 `18160`
- `km_asset_app.entrypoints.worker`，无监听端口

必须设置以下 Secret 环境变量：

- `KBOT_ORACLE_PASSWORD`
- `KBOT_MANAGED_CREDENTIAL_KEY`
- KBot 已有的内部 JWT 和服务身份 Secret

MetaDB Basic Auth 和 SharePoint Graph Client Secret 通过创建来源请求提交，随后只以
AES-256-GCM 密文保存在 `KBOT_MANAGED_CREDENTIAL`，查询接口不返回凭据内容。

## 首次配置

1. 给管理员分配 `km_asset/manager` App Role。
2. 创建一个 KC Collection。
3. 调用 `POST /api/v1/apps/km-asset/sources` 创建来源。
4. 创建来源时，Data Query 自动调和固定模型 `KM Asset 元数据（系统托管）`。
5. 调用 `POST /api/v1/apps/km-asset/sources/{source_id}/activate` 激活来源。
6. 创建 KM Agent，并选择聊天需要的模型；问文问数模式至少要配置 `router_llm`。
   Collection、语义模型和查询策略不由前端选择。DRAFT Agent 可通过激活接口重试并修复问数绑定。

正式 JavaScript 页面位于 `ui/km/`，包括工作台、MetaDB、数据来源、Asset、同步任务、
Agent 和智能问答。没有 APEX 用户管理页面时，
首次部署在 SQL Developer 中执行 `scripts/db/bootstrap_km_initial_admin.sql`
创建可登录用户及 KM 管理员权限。用户先在 `ui/km/login.html`
使用用户名和密码换取短期 Token，再访问其余 KM 页面。页面只
调用 Main API 公开 BFF，不能把 Portal API Key 或内部身份 Header 写入静态 JavaScript。

既有 Schema 不得重新执行完整的 `main_api/002_access_control.sql`。直接执行
`scripts/db/bootstrap_km_initial_admin.sql`，它会幂等创建用户凭据表、初始化
KM 权限和角色、创建 `kmadmin` 用户，并授予全部启用 Domain 的
`km_asset/manager`。

来源创建请求示例：

```json
{
  "display_name": "KM Asset MetaDB",
  "metadb_endpoint": "https://metadb.example.com/assets",
  "metadb_credentials": {
    "username": "service-account",
    "password": "secret"
  },
  "sharepoint_credentials": {
    "tenant_id": "tenant-id",
    "client_id": "client-id",
    "client_secret": "secret"
  },
  "sharepoint_site_path": "/sites/km",
  "collection_id": "01900000-0000-7000-8000-000000000001",
  "poll_interval_seconds": 60,
  "batch_size": 100
}
```

## 公开接口

- `GET /api/v1/apps/km-asset/sources`
- `POST /api/v1/apps/km-asset/sources`
- `PATCH /api/v1/apps/km-asset/sources/{source_id}`
- `POST /api/v1/apps/km-asset/sources/{source_id}/activate`
- `POST /api/v1/apps/km-asset/sources/{source_id}/sync`
- `GET /api/v1/apps/km-asset/sources/{source_id}/metadb/assets`
- `POST /api/v1/apps/km-asset/sources/{source_id}/metadb/assets/{asset_id}/retry`
- `GET /api/v1/apps/km-asset/sources/{source_id}/data-model`
- `POST /api/v1/apps/km-asset/sources/{source_id}/data-model/reconcile`
- `GET /api/v1/apps/km-asset/assets`
- `GET /api/v1/apps/km-asset/assets/{km_asset_id}`
- `POST /api/v1/apps/km-asset/assets/{km_asset_id}/retry`
- `GET /api/v1/apps/km-asset/jobs`
- `GET|POST /api/v1/apps/km-asset/agents`
- `GET /api/v1/apps/km-asset/agents/{agent_id}`
- `POST /api/v1/apps/km-asset/agents/{agent_id}/activate`
- `POST /api/v1/apps/km-asset/conversations`
- `GET|PATCH|DELETE /api/v1/apps/km-asset/conversations/{conversation_id}`
- `POST /api/v1/apps/km-asset/conversations/{conversation_id}/turns`
- `GET /api/v1/apps/km-asset/conversations/{conversation_id}/turns`
- `GET /api/v1/apps/km-asset/runs/{run_id}`
- `GET /api/v1/apps/km-asset/runs/{run_id}/result`
- `GET /api/v1/apps/km-asset/runs/{run_id}/events`（SSE）
- `GET /api/v1/apps/km-asset/runs/{run_id}/references/{citation_label}/preview`
- `GET /api/v1/apps/km-asset/runs/{run_id}/references/{citation_label}/content`

修改来源时必须提交 `expected_row_version`。可修改显示名称、MetaDB Endpoint、
SharePoint Site Path、`poll_interval_seconds`（10～86400 秒）和 `batch_size`（1～1000）；
也可以提交完整的新凭据组进行密文轮换。省略凭据字段表示保留原凭据。Collection 属于
Agent 版本的固定资源定位，不能通过该接口修改。

## 问文与问数

文档问题使用来源绑定的 KC Collection。结构化统计使用系统托管模型，对
`KBOT_V_KM_ASSET_CURRENT` 执行只读查询。数据集声明 `scope_column=DOMAIN_ID`，Data
Query 编译器会使用可信 AuthContext 中的 Domain ID 强制注入参数化条件，LLM 和前端
都不能移除或覆盖该条件。

## 下线旧 Portal Worker

新 App 完成一次真实 Asset 的端到端验收前，旧进程保持停止但保留代码。验收内容包括：

- MetaDB 原始记录可查询；
- 所有附件完整下载并形成一个 Bundle；
- KC Bundle 达到 `READY`；
- MetaDB 状态更新为 `Y`；
- Agent 能回答正文问题并给出文档来源；
- Agent 能正确回答按作者、主题、行业的数量和明细问题；
- 人工制造失败后，可以从 `F` 重置并重试成功。

验收通过后，不再运行 `/home/chris/km_portal/km_portal.py`，旧实现仅从 Git 历史恢复。
