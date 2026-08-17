# KBot 4.0 部署指南

## 前置条件

- Linux x86_64；
- Python 3.10 或更高版本；
- Oracle Database 26ai，应用表空间使用 ASSM；
- 运行模型托管进程时准备相应 CPU/GPU、模型文件和 Provider 凭据；
- Portal/APEX 只访问 Main API，内部服务端口位于受控网络。

生产环境应为每个服务构建独立运行单元。当前共享 Oracle Schema，但服务可以部署
在不同主机；仅在开发机使用 `start_kbot.sh` 一次启动全部进程。

## 空白环境一键部署

先准备 `configuration/kbot.toml` 和 Secret。目标必须是没有任何 `KBOT_%` 表或
视图的空白 Schema，然后在仓库根目录执行：

```bash
bash scripts/deployment/bootstrap_kbot.sh --production
```

脚本依次安装锁定依赖和 9 个内部包、校验配置与 20 进程拓扑、解析 8 个服务的
规范 DDL、检查 Oracle 权限与空库条件、创建表/视图/索引/约束，并初始化默认 App
角色、权限、角色映射和 Prompt Catalog。它不会重置已有 Schema，也不会创建
Domain、用户、成员授权、业务 Agent、模型或知识库数据。

只验证安装、配置和 Schema 计划而不连接数据库：

```bash
bash scripts/deployment/bootstrap_kbot.sh --production --schema-dry-run
```

## 安装

下载或克隆明确的 4.0 Release/Commit 后，开发环境安装锁定的第三方依赖和全部
editable 内部包：

```bash
bash scripts/deployment/install_workspace.sh
```

每个服务和共享包都有独立 `pyproject.toml`。安装脚本按依赖顺序执行
`pip install --no-deps -e`，`start_kbot.sh` 不再设置 `PYTHONPATH`。

生产环境构建并安装本地 Wheel：

```bash
bash scripts/deployment/install_workspace.sh --production
```

可通过 `KBOT_PYTHON=/path/to/python` 指定解释器。未指定时，安装脚本与
`start_kbot.sh` 使用同一选择规则：优先安装到 `KBOT_CONDA_ENV`，否则自动选择
`kbot4`，仅当 `kbot4` 不存在时回退到 `cube`。安装开始时会打印目标解释器，安装后会
验证模块来源；发现其他工作区的同名 editable package 时直接失败。

例如明确安装到 `kbot4`：

```bash
KBOT_CONDA_ENV=kbot4 bash scripts/deployment/install_workspace.sh
```

### Conda OCR 依赖

Knowledge Core Parser 默认通过 Docling 的 `TesseractOcrOptions` 执行中英文 OCR。
`requirements.txt` 中的 Docling 不包含可选的 Tesseract Python 绑定，因此使用 Conda
环境部署时，还必须在 KBot 实际运行的同一个环境中安装 `tesseract` 和 `tesserocr`：

```bash
conda install -n kbot4 -c conda-forge tesseract tesserocr
```

激活环境后，将 Tesseract 语言数据目录保存为该环境的持久变量：

```bash
conda activate kbot4
conda env config vars set TESSDATA_PREFIX="$CONDA_PREFIX/share/tessdata/"
conda deactivate
conda activate kbot4
```

安装后必须确认 Python 绑定可导入，并且语言列表至少包含简体中文 `chi_sim` 和英文
`eng`：

```bash
tesseract --version
tesseract --list-langs
python -c "import tesserocr; print(tesserocr.tesseract_version()); print(tesserocr.get_languages())"
```

如果部署使用 `KBOT_CONDA_ENV` 指定了其他环境，应将以上命令中的 `kbot4` 替换为
实际环境名。Parser 进程由 systemd 或其他进程管理器启动时，也必须继承该环境保存的
`TESSDATA_PREFIX`，否则 Docling 可能能导入 `tesserocr`，但仍无法加载中文语言数据。

KBot 与其他使用 `platform_core`、`agent_runtime` 等相同 Import 名的项目不能同时在
一个 Python 环境中以 editable 模式安装。遇到来源冲突时应使用 KBot 专用环境，不能通过
调整 `PYTHONPATH` 或忽略来源检查绕过。
脚本会创建或补齐本地 `.env` 中的统一托管凭据加密密钥与版本；已有值不会覆盖。
生产使用外部 Secret 时设置 `KBOT_SKIP_LOCAL_ENV_INIT=1`。

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

`KBOT_MASTER_KEY` 会按用途派生 `KBOT_USER_JWT_SECRET`。如需单独轮换普通用户登录
Token 的签名密钥，可显式注入同名环境变量，长度不得少于 32 字节。

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

通知中心由 Main API 的 Notification Worker 投影共享 Oracle Outbox。默认配置适合单机，
需要调整批量、租约、重试或 SSE 心跳时，在 `kbot.toml` 增加 `[notifications]`，字段见
`configuration/kbot.toml.example`。Outbox 状态为 `PENDING → PROCESSING → PUBLISHED`；
租约过期会回到 `PENDING`，超过重试上限进入 `QUARANTINED`。修复事件内容或对应投影代码
后，通过 Main API 的隔离重试接口重新投递。不得直接删除未发布 Outbox；Inbox 默认按事件
目录保留 90 天。删除外部 Actor 标识时，应调用 Actor 数据清理接口同步删除其 Inbox、偏好、
待办和关注记录。

## 初始化 Oracle

全新数据库执行标准 Schema 初始化时，会自动幂等创建首次登录基础数据：默认业务域
`default`、App 目录、完整权限与角色模板、受保护的 `ADMIN` 凭据及其
`platform_admin` 平台角色。`ADMIN` 不会自动成为任何 App 的成员。普通用户安全等级
默认是 `1`，`ADMIN` 固定为最高等级 `3`。
初始账号为 `ADMIN`，初始密码为 `Admin@2026!`，部署完成后应
立即重置密码。模型、Collection、Agent、会话和 AIOps 业务数据不会初始化。

既有数据库缺少基础数据时，优先执行：

```bash
python3 scripts/db/apply_oracle_schema.py --foundation-only
python3 scripts/db/apply_oracle_schema.py --check-foundation
```

旧授权结构升级到平台/App 分层结构时，先停掉 Main API 和相关 Worker，备份当前 Schema，
若旧账号仍为小写 `admin`，先执行
`scripts/db/migrate_global_admin_to_uppercase.sql`；随后在 SQL Developer 中执行
`scripts/db/migrate_access_scope_model.sql`。Oracle DDL 会
自动提交，失败后不能依赖事务整体回滚。升级完成后部署同一版本代码，再运行
`--check-foundation`。迁移会保留现有用户、角色与 Domain 范围，但不会自行选择每个 App
的初始管理员；平台管理员登录后必须通过平台接口逐 App 创建。

仅需升级普通登录、用户管理和角色管理权限时，也可在 SQL Developer 中执行：

```text
scripts/db/bootstrap_platform_access_management.sql
```

脚本不要求输入参数，不修改 ADMIN 密码；它会幂等创建平台权限目录与
`platform_admin` 角色，并只在平台层为 ADMIN 授权。全新数据库无需再单独执行
`bootstrap_global_admin.sql`。

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
python3 scripts/db/apply_oracle_schema.py --check-foundation
python3 tests/acceptance/check_oracle_schema.py
```

`check_deployment.py` 会校验 UI 使用的 `[ui].main_api_base_url`；生产环境仍使用
`kbot.example.com` 等示例地址时会拒绝启动。基础数据校验失败会明确提示“系统尚未
初始化”，不会继续启动服务。

先启动 Oracle 和外部依赖，再按以下顺序启动：

1. Model Serving；
2. Knowledge Core API、Parser、Projection Worker；
3. Knowledge Retrieval App API；
4. Data Query API、Worker；
5. Agent Runtime API、Worker；
6. AIOps API、DB Executor、Worker、Scheduler；
7. KM Asset Slack Worker、Main API、Notification Worker。

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

发布候选环境应同时执行 Oracle 与 Data Query 外部数据库 Smoke：

```bash
export KBOT_DQ_SMOKE_POSTGRES_PASSWORD='由测试环境 Secret 提供'
export KBOT_DQ_SMOKE_POSTGRES_DATABASE='kbot_dq_smoke'
export KBOT_DQ_SMOKE_POSTGRES_USERNAME='kbot_smoke'
export KBOT_DQ_SMOKE_MYSQL_PASSWORD='由测试环境 Secret 提供'
python3 scripts/release/verify_release.py --profile rc
```

`rc` 档位自动强制 Oracle 实库、Data Query 外部数据库 Smoke 和干净工作树门禁；
缺少相应测试环境或 Secret 时会失败，不会降级成 developer 证据。

PostgreSQL/MySQL 的主机、端口、数据库、账号和 Schema 可通过同名前缀的
`KBOT_DQ_SMOKE_POSTGRES_*`、`KBOT_DQ_SMOKE_MYSQL_*` 环境变量覆盖。Smoke 只创建固定
测试表并在结束时删除，不将密码写入日志或发布证据。

验收至少覆盖健康检查、Oracle Schema、模型目录、文件入库解析、全文/向量检索、
Agent SSE 与引用、AIOps 受控执行和 Domain 隔离。公开调用方只使用 `/api/v1`；
`/internal/v1` 不进入外部网关。

## 下载docling模型

```bash
docling-tools models download --all -o models
```
