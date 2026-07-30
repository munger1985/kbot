# KBot 4.0 部署配置

KBot 只使用一个部署文件：`configuration/kbot.toml`。服务端口、进程身份、内部
依赖、超时、租约、批次、重试和安全上限都是代码契约，部署人员无需填写。

## 本地开发

仓库已准备本机 `kbot.toml`。新环境可执行：

```bash
cp configuration/kbot.toml.example configuration/kbot.toml
cp .env.example .env
```

然后填写数据库地址、`KBOT_ORACLE_PASSWORD`、`KBOT_MASTER_KEY`，并生成
Portal API Key 摘要。

## 生产配置

生产环境通常只需以下内容：

```toml
environment = "production"
data_dir = "/var/lib/kbot"
log_dir = "/var/log/kbot"
embedding_dimension = 2560
api_docs_enabled = false
# 仅浏览器直连 Main API 时需要配置；必须是精确 Origin，不带末尾斜杠。
api_allowed_origins = ["https://portal.example.com"]

[database]
host = "oracle.example.internal"
port = 1521
service_name = "kbot4"
username = "kbot"

[[portal_api_keys]]
key_id = "portal-prod"
client_id = "portal"
key_digest = "生成的64位摘要"
```

`data_dir` 自动派生 Knowledge Core、Agent附件、AIOps正文和模型缓存目录。Docling
默认使用 `<data_dir>/models/docling_models`。服务部署在同一主机时，内部地址由
`resources/topology.toml` 自动生成。

`api_docs_enabled = true` 会启用 Main API 的离线 Swagger UI（`/docs`）和
ReDoc（`/redoc`）；开发环境默认启用，生产环境默认关闭。

浏览器直连 Main API 时，使用 `api_allowed_origins` 列出允许的门户来源。每项必须
精确包含协议、主机和端口，例如 `https://portal.example.com` 或
`http://146.56.158.44:8080`，不带末尾斜杠；不支持 `*`。Portal API Key 不应下发到
浏览器，生产环境应优先由门户服务端代理调用 KBot。

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

KBot从主密钥按用途派生内部JWT、API Pepper、凭证加密和AIOps签名密钥。外部
问数服务启用时再增加 `KBOT_MCP_DATA_API_KEY`。模型厂商凭证仍由数据库加密
保存。

Portal API Key 的明文只显示一次；设置主密钥后执行：

```bash
python scripts/security/generate_portal_api_key.py --key-id portal-prod
```

脚本会原子更新部署文件中同名 `key_id` 的 `[[portal_api_keys]]` 摘要记录，再只
显示一次明文 Key；将明文仅保存到 Portal Secret。外部系统需要直接调用模型 API
时，使用独立的
`[[model_api_keys]]`。

## 启动前检查

```bash
python tests/acceptance/check_configuration_contract.py
python scripts/deployment/check_deployment.py
```

也可通过 `KBOT_CONFIG_FILE=/etc/kbot/kbot.toml` 使用外置配置，通过
`ENV_FILE=/etc/kbot/kbot.env` 加载Secret文件。
