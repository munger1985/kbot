# KBot 4.0 部署配置

KBot 只使用一个部署文件：`configuration/kbot.toml`。服务端口、进程身份、内部
依赖、超时、租约、批次、重试和安全上限都是代码契约，部署人员无需填写。

## 本地开发

仓库已准备本机 `kbot.toml`。新环境可执行：

```bash
cp configuration/kbot.toml.example configuration/kbot.toml
cp .env.example .env
```

然后填写数据库地址、`KBOT_ORACLE_PASSWORD` 和 `KBOT_MASTER_KEY`。

配置完成后，可用一个入口安装依赖、检查配置并初始化空白 Schema：

```bash
bash scripts/deployment/bootstrap_kbot.sh
```

生产使用 `--production` 构建并安装 Wheel。只检查安装和建表计划、不连接数据库时
增加 `--schema-dry-run`。

## 生产配置

生产环境通常只需以下内容：

```toml
environment = "production"
data_dir = "/var/lib/kbot"
log_dir = "/var/log/kbot"
embedding_dimension = 2048
api_docs_enabled = false
# 仅浏览器直连 Main API 时需要配置；必须是精确 Origin，不带末尾斜杠。
api_allowed_origins = ["https://portal.example.com"]

[database]
host = "oracle.example.internal"
port = 1521
service_name = "kbot4"
username = "kbot"

[aiops]
# 部署级 Kill Switch；完成受控动作验收后按部署显式改为 true。
agent_execution_enabled = false
mutation_enabled = false

```

`data_dir` 自动派生 Knowledge Core、Agent附件、AIOps正文和模型缓存目录。Docling
默认使用 `<data_dir>/models/docling_models`。服务部署在同一主机时，内部地址由
`resources/topology.toml` 自动生成。

`api_docs_enabled = true` 会启用 Main API 的离线 Swagger UI（`/docs`）和
ReDoc（`/redoc`）；开发环境默认启用，生产环境默认关闭。

`[aiops]` 中两个开关必须同时为 `true`，审批后执行链才可签发一次性 Mutation Grant；
它们不会替代 Agent–Target 的动作选择、对象范围、执行凭据或逐条人工审批。默认关闭。

浏览器直连 Main API 时，使用 `api_allowed_origins` 列出允许的门户来源。每项必须
精确包含协议、主机和端口，例如 `https://portal.example.com` 或
`http://146.56.158.44:8080`，不带末尾斜杠。若需要允许任意来源，可设为
`api_allowed_origins = ["*"]`。用户页面使用短期用户 Token；第三方服务使用 App 管理员
在页面中签发的 App API Key。App API Key 不应下发到浏览器。

只有跨主机部署才增加对应端点：

```toml
[endpoints]
knowledge_core = "http://knowledge-core.internal:18090"
model_llm = "http://models.internal:18092"
```

未覆盖的端点继续使用本机地址。MCP问数和DeepSeek OCR等可选集成示例见
`kbot.toml.example`，未使用时不要填写。

## Secret

生产环境建议将以下变量放入权限为 `0600` 的 systemd EnvironmentFile，或通过
容器Secret注入：

```dotenv
KBOT_ORACLE_PASSWORD="数据库密码"
KBOT_MASTER_KEY="至少32字节的随机主密钥"
```

KBot从主密钥按用途派生内部JWT、API Pepper、统一托管凭据加密和AIOps签名密钥。
如需独立轮换托管凭据密钥，可显式配置 `KBOT_MANAGED_CREDENTIAL_KEY` 与
`KBOT_MANAGED_CREDENTIAL_KEY_VERSION`；AIOps、Data Query 和模型 Provider 共用
`KBOT_MANAGED_CREDENTIAL` 表，不再使用业务密钥文件或各模块专用凭据表。外部
问数服务启用时再增加 `KBOT_MCP_DATA_API_KEY`。模型厂商凭证仍由数据库加密
保存。

公开 Main API 不再读取配置型 Key。App 管理员登录对应 App 后，在“API 客户端”页面
绑定服务账号、Domain、Scope 和 Agent，并生成只显示一次的 `kbot_ak_...` Key。
数据库只保存摘要。外部系统直接调用模型 API 时仍使用独立的 `[[model_api_keys]]`。

## 启动前检查

```bash
python tests/acceptance/check_configuration_contract.py
python scripts/deployment/check_deployment.py
```

也可通过 `KBOT_CONFIG_FILE=/etc/kbot/kbot.toml` 使用外置配置，通过
`ENV_FILE=/etc/kbot/kbot.env` 加载Secret文件。
