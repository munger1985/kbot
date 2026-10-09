# KBot 4.0 容器交付

本目录提供从同一 Commit 构建不可变镜像、渲染单机 Docker Compose 以及后续推送
OCI Container Registry（OCIR）的统一入口。开发机阶段不需要 OCI 账号、仓库或计算实例。

## 交付模型

- `kbot-ui` 是唯一发布宿主端口的容器，默认映射 `8080:8080`，并把 `/api/*`
  同源反向代理到 Main API；
- 9 个业务镜像对应 9 个 Python 服务包，同一服务的 API、Worker、Scheduler 复用同一镜像；
- Model Serving、Knowledge Core 和 Installer 显式安装固定版本的 CPU Torch/TorchVision，
  不会从普通 PyPI 意外带入 CUDA 运行时；
- `resources/topology.toml` 中 25 个进程分别作为 Compose Service 运行；
- 内部 HTTP 端点使用 Compose DNS，不使用 Host 网络，也不发布内部端口；
- Oracle 是外部受管数据库，不在 Compose 中创建；
- `kbot-installer` 仅在 `tools` Profile 下按需运行数据库初始化或校验工具；
- Oracle 密码和 KBot 主密钥从 Docker Secret 文件注入，生成目录内的副本固定为 `0600`。

当前镜像平台固定为 `linux/amd64`。在完成 ARM 依赖矩阵验证前，不将 OCI Ampere
实例作为正式目标。

## 开发机构建

复制配置，不要把实际配置或 Secret 提交到 Git：

```bash
cp installation/dev/release.ini.example installation/dev/release.ini
cp installation/dev/deployment.ini.example installation/dev/deployment.ini
mkdir -p installation/dev/secrets
printf '%s\n' 'Oracle密码' > installation/dev/secrets/oracle_password
python3 installation/release.py generate-master-key \
  --output installation/dev/secrets/master_key
chmod 600 installation/dev/secrets/oracle_password
```

在 `deployment.ini` 中填写外部 Oracle 地址，并保持 AIOps 变更 Kill Switch 默认关闭。
校验并渲染构建定义：

```bash
installation/dev/kbot-release validate \
  --release-config installation/dev/release.ini \
  --deployment-config installation/dev/deployment.ini
installation/dev/kbot-release render-build
installation/dev/kbot-release build --mode print
```

构建到本机 Docker：

```bash
installation/dev/kbot-release build --mode load
```

也可先验证较小的镜像：

```bash
installation/dev/kbot-release build --mode load --target ui
installation/dev/kbot-release build --mode load --target main_api
```

## 渲染与运行

```bash
installation/dev/kbot-release render-deployment
docker compose -f installation/generated/deployment/compose.yaml config
docker compose -f installation/generated/deployment/compose.yaml up -d
```

生成的 Compose 使用执行发布工具的用户 UID/GID 运行 Python 容器，使 `0600` Secret
保持可读，同时让数据和日志目录不需要开放给其他本机用户。生成目录已被 Git 忽略。

首次初始化必须先确认目标是空白 KBot Schema，再显式运行工具容器；渲染操作本身不会
连接数据库或执行 DDL：

```bash
docker compose -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py --help
```

实际初始化参数仍以 `scripts/db/apply_oracle_schema.py` 和
`scripts/deployment/bootstrap_kbot.sh` 的规范流程为准，不能对已有 Schema 直接重放空库脚本。

## OCI 阶段边界

只有执行以下事项时才需要申请 OCI 资源：

1. 创建 OCIR Repository，并取得区域 Registry 地址和推送凭据；
2. 创建 `linux/amd64` OCI Compute、VCN/子网、NSG、块存储和 DNS/TLS；
3. 从 Compute 拉取镜像并接入外部 Oracle；
4. 配置 OCI Vault 或等价 Secret 管理，以及镜像摘要锁定和回滚策略。

取得 OCIR 地址后，将两个 INI 中的 `registry` 改为
`<region-key>.ocir.io/<namespace>/kbot`，登录 Registry，再执行：

```bash
installation/dev/kbot-release build --mode push
```

正式部署应记录并使用 Registry 返回的镜像 Digest，而不是依赖可变 Tag。OCI NSG 默认只
开放管理所需的 SSH 和用户入口 80/443，所有 KBot 内部端口继续只在 Compose 网络可见。
`--mode push` 会拒绝存在未提交修改的工作区，镜像 Revision 标签不会伪装成可追溯 Commit。
