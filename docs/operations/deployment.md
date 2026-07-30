# KBot 4.0 部署指南

## 前置条件

- Linux x86_64；
- Python 3.10 或更高版本；
- Oracle Database 26ai，应用表空间使用 ASSM；
- 运行模型托管进程时准备相应 CPU/GPU、模型文件和 Provider 凭据；
- Portal/APEX 只访问 Main API，内部服务端口位于受控网络。

生产环境应为每个服务构建独立运行单元。当前共享 Oracle Schema，但服务可以部署
在不同主机；仅在开发机使用 `start_kbot.sh` 一次启动全部进程。

## 安装

下载或克隆明确的 4.0 Release/Commit 后，仅安装锁定的第三方依赖：

```bash
bash scripts/deployment/install_workspace.sh
```

安装脚本不会执行 `pip install -e`，也不要求服务目录包含 `pyproject.toml`。
当前部署方式直接保留完整源码树；`start_kbot.sh` 统一设置 `PYTHONPATH` 后从
`services/*/src` 和 `packages/*/src` 加载模块。

## 配置与 Secret

复制唯一配置样例：

```bash
cp configuration/kbot.toml.example configuration/kbot.toml
cp .env.example .env
```

`configuration/kbot.toml` 只保存环境、数据/日志目录、Oracle 地址、全局 Embedding
维度、Portal/模型 Key 摘要和可选外部端点。服务端口、超时、Lease 和安全上限由
代码契约管理。生产可通过 `KBOT_CONFIG_FILE=/etc/kbot/kbot.toml` 指向外部文件。

至少注入：

```dotenv
KBOT_ORACLE_PASSWORD="..."
KBOT_MASTER_KEY="至少32字节的随机主密钥"
```

推荐使用 systemd `EnvironmentFile`、Kubernetes Secret 或企业 Secret Manager，
不要把生产 `.env` 写入仓库。Portal API Key 使用以下命令生成摘要：

```bash
python3 scripts/security/generate_portal_api_key.py --key-id portal-prod
```

脚本自动将摘要写入 `kbot.toml`；明文 Key 只交付 Portal Secret。

Main API 使用离线 Swagger UI，不依赖外部 CDN。在 `kbot.toml` 设置
`api_docs_enabled = true` 后，可访问 `http://<main-api-host>:18099/docs`，
OpenAPI JSON 位于 `/openapi.json`。

若门户页面在浏览器中直连 Main API，还必须在 `kbot.toml` 配置精确的跨域来源，例如：

```toml
api_allowed_origins = ["http://146.56.158.44:8080"]
```

修改后重启 Main API。该值必须与浏览器地址的协议、主机和端口完全一致，不能使用
末尾斜杠。若要临时允许任意网页来源，设置 `api_allowed_origins = ["*"]`。生产环境不要
将 Portal API Key 暴露给浏览器；应由门户服务端代理请求。

## 初始化 Oracle

在 `scripts/db/init_services.ini` 选择需要部署的业务服务。`platform_core` 基础表
始终创建。先预检：

```bash
python3 scripts/db/apply_oracle_schema.py \
  --config scripts/db/init_services.ini \
  --dry-run
```

KC 使用 `DBMS_ALERT` 通知 Worker，PDB 管理员需要直接授权：

```sql
GRANT EXECUTE ON SYS.DBMS_ALERT TO {KBOT_USER};
```

确认目标为空 Schema 后执行：

```bash
python3 scripts/db/apply_oracle_schema.py \
  --config scripts/db/init_services.ini
```

初始化器拒绝覆盖已有 `KBOT_%` 对象。4.0 不读取、迁移或兼容 3.x 表和数据。完整
权限、表空间和执行顺序见 [Oracle 初始化说明](../../database/oracle/README.md)。

## 启动前检查

```bash
python3 tests/acceptance/check_configuration_contract.py
python3 scripts/deployment/check_deployment.py
python3 tests/acceptance/check_oracle_schema.py
```

先启动 Oracle 和外部依赖，再按以下顺序启动：

1. Model Serving；
2. Knowledge Core API、Parser、Projection Worker；
3. Agent Runtime API、Worker；
4. AIOps API、Worker、Scheduler、DB Executor；
5. Main API。

各模块入口位于 `services/<service>/src/<package>/entrypoints/`。开发环境可执行：

```bash
bash start_kbot.sh
bash stop_kbot.sh
```

开发模式还会启动 `tools/dev_console/` 静态服务器；生产不得启用测试认证绕过或
开发日志接口。

## 数据与日志

`data_dir` 派生 KC 对象、Agent 附件、AIOps Artifact 和模型缓存目录。生产部署应
使用持久卷并纳入备份。每个服务仅写：

- `<log_dir>/<service>/runtime.log`；
- `<log_dir>/<service>/access.log`。

日志页面只在开发环境开放。生产接入日志采集系统时按服务目录采集，使用
`trace_id/run_id/task_id` 关联跨服务调用。

## 验收与发布

发布前执行：

```bash
python3 scripts/release/verify_release.py
```

验收至少覆盖健康检查、Oracle Schema、模型目录、文件入库解析、全文/向量检索、
Agent SSE 与引用、AIOps 受控执行和 Domain 隔离。公开调用方只使用 `/api/v1`；
`/internal/v1` 不进入外部网关。
