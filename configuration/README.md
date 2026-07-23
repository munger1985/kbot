# KBot 4.0 配置结构

配置按“共享平台 + 服务专属”分层加载：

1. `base.toml`
2. `${ENVIRONMENT}.toml`，默认 `development`
3. `services/<service>/base.toml`
4. `services/<service>/${ENVIRONMENT}.toml`

目前 `<service>` 可取 `main_api`、`knowledge_core`、`model_serving` 和
`agent_runtime`。服务代码只能读取自己的配置模型；`platform_core` 不导入
任何服务配置。

通过 `CONFIG_DIR` 指定配置根目录，通过 `ENVIRONMENT` 选择环境。例如：

```bash
export CONFIG_DIR=/app/configuration
export ENVIRONMENT=production
export KBOT_ORACLE_PASSWORD='由 Secret 管理系统注入'
python -m apps.knowledge_core_api.main
```

配置加载器默认读取当前工作目录的 `.env`，但不会覆盖进程中已经存在的
环境变量。也可以通过 `ENV_FILE=/etc/kbot/kbot.env` 显式指定环境文件。
因此推荐：

- 本地开发：复制 `.env.example` 为 `.env`，由 `python-dotenv` 加载；
- systemd：使用权限为 `0600` 的 `EnvironmentFile`，或直接声明
  `Environment`；
- Docker/Kubernetes：通过 Secret 注入进程环境或挂载 Secret 文件；
- 生产环境不要把 Secret 写入仓库、TOML、Shell Profile 或系统全局环境。

`base.toml` 只保存各服务共同遵守的配置：`app_id`、日志、安全参数名、
数据库连接与连接池，以及唯一向量维度。监听端口、依赖地址、Worker
租约、解析策略和模型推理默认值必须放在所属服务目录。

`example/` 与实际目录一一对应，用于服务器部署时对照。明文数据库密码、
API Key、Token、模型厂商 Key 和私钥不得写入任何 TOML；TOML 中只保存
环境变量名、Key 摘要或 Secret 引用。
