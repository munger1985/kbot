# KBot 4.0 配置结构

配置按“共享平台 + 服务专属”分层加载：

1. `base.toml`
2. `${ENVIRONMENT}.toml`，默认 `development`
3. `services/<service>/base.toml`
4. `services/<service>/${ENVIRONMENT}.toml`

目前 `<service>` 可取 `main_api`、`knowledge_core`、`model_serving`、
`agent_runtime` 和 `aiops_agent`。服务代码只能读取自己的配置模型；
`platform_core` 不导入任何服务配置。

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

AIOps 内部调用额外使用短期 Service Identity JWT。签名密钥通过
`KBOT_SERVICE_IDENTITY_JWT_SECRET` 注入，不写入 TOML；生产环境的 Target、
Monitor 和数据库凭据只保存 SecretRef，且禁止使用 `environment` Secret
Provider。

`process_topology.toml` 是 4.0 完整本地进程清单，不承载环境覆盖或 Secret。它将
14 个 `apps.*.main` 入口映射到所属服务配置段、进程类型和监听端口。执行：

```bash
python scripts/check_process_topology.py
```

可检查 App 入口、服务配置、Example、端口唯一性以及 `start_kbot.sh` /
`stop_kbot.sh` 覆盖关系。本地启动优先使用 `kbot3`，不存在时回退到开发机的
`cube`；也可通过 `KBOT_CONDA_ENV=<name> ./start_kbot.sh` 显式指定。环境不存在
或激活失败时脚本立即退出。生产部署不得依赖该脚本。

修改配置后还应执行：

```bash
python scripts/check_configuration_contract.py
```

该检查确保实际配置与 Example 字段严格对应，开发和生产配置可通过各服务的
Pydantic 模型，并且配置声明的 Secret 环境变量均已收录于 `.env.example`。

发布前还应执行 `python scripts/check_supply_chain.py`。该检查要求
`requirements.txt` 中的直接依赖全部精确锁定且不重复，验证直接依赖 CycloneDX
SBOM，并扫描受 Git 跟踪文件中的常见 Secret 与敏感文件类型。依赖变化后使用
`python scripts/check_supply_chain.py --write-sbom` 更新直接依赖 SBOM。

API 契约变化后使用 `python scripts/check_openapi_contracts.py --write` 重建全部
10 个受管理 OpenAPI 快照，再执行不带 `--write` 的命令检查契约漂移与
Public/Internal 路径边界。生成过程不会启动 Lifespan、连接数据库或加载模型。
