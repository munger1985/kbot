# KBot 4.0 容器交付

本目录提供从同一 Commit 构建不可变镜像、渲染单机 Docker Compose 以及后续推送
OCI Container Registry（OCIR）的统一入口。开发机阶段不需要 OCI 账号、仓库或计算实例。

## 正式版本 1.0

`1.0` 是本交付线的首个正式容器版本，最新修正版于 2026-10-22 从 `kbot4.1` 分支的
Commit `3b60f5342ac02e1d23b535ef7b8a69bac526feab` 构建并发布，平台为
`linux/amd64`。Registry 前缀为 `nrt.ocir.io/sehubjapacprod/kbot`；生产部署应按
下表使用不可变 Digest，不依赖可变 Tag。

| 镜像 | `1.0` Digest |
|---|---|
| `kbot-model-serving` | `sha256:0e15a89dffefebb41f19e0fd52f66e595a5f9dae02bc40ac9310adb72e6b7b1a` |
| `kbot-knowledge-core` | `sha256:525ffbb2f9d2ea2b3d64b869c05d9065b2a994b6eb37dda73aa5846c769b7d9c` |
| `kbot-knowledge-retrieval-app` | `sha256:2c6903f9ba7f83ab01268fba7b4412fdd26f6b02f9b8c57acee556913f96f02b` |
| `kbot-media-studio-app` | `sha256:6d4d9974ff2c8011f297f87ba18b033e1d3e5c14fab07520a9db799828d41ba9` |
| `kbot-agent-runtime` | `sha256:d013ae14c9e2aa2807921d7a206c203e3980a674d0bfbac70d48c7e12af3b0a3` |
| `kbot-data-query` | `sha256:b993d61320f0cbafcfad3815f1fe406f1b5fa986eeb28c420bf7b31c84f2bc01` |
| `kbot-aiops-agent` | `sha256:220989053f5601fcce3d2680389c968f95e41b7e03438df6788e4814852eb0e4` |
| `kbot-main-api` | `sha256:c0bf0da6a7bf5da732a09840513e8f3d7ddf78fb4b42487f6f38eab0f023b288` |
| `kbot-installer` | `sha256:def3b14698592917dbc17aac04fc97da39fc0e060e4431f1626019543a876ab5` |
| `kbot-ui` | `sha256:0ef51fbd2d3ee761a8650756e64bd2dcc73d38e7f133778cb506b36a29c6feb5` |

上一试发布标签 `4.0.0-e1b511db` 已从上述 10 个镜像仓库删除，不作为回滚版本保留。
正式环境的 OCI 基础设施、Digest 锁定、空库初始化、验收和恢复顺序见
[KBot 1.0 OCIR 生产部署与初始化方案](../docs/proposals/ocir-1.0-production-deployment-plan.md)。

## 交付模型

- `kbot-ui` 是唯一发布宿主端口的容器，默认映射 `8080:8080`，并把 `/api/*`
  同源反向代理到 Main API；
- 8 个业务服务镜像对应 8 个 Python 服务包，同一服务的 API、Worker、Scheduler
  复用同一镜像；另有 `kbot-installer` 工具镜像和 `kbot-ui` 前端镜像；
- Model Serving、Knowledge Core 和 Installer 显式安装固定版本的 CPU Torch/TorchVision，
  不会从普通 PyPI 意外带入 CUDA 运行时；
- `deployment.ini` 的 `[apps]` 开关决定实际部署哪些 App，发布器只渲染这些 App
  及其共享依赖进程；全选时覆盖 `resources/topology.toml` 中全部 23 个进程；
- 内部 HTTP 端点使用 Compose DNS，不使用 Host 网络，也不发布内部端口；
- Oracle 是外部受管数据库，不在 Compose 中创建；
- `kbot-installer` 仅在 `tools` Profile 下按需运行数据库初始化或校验工具；
- Oracle 密码和 KBot 主密钥从 Docker Secret 文件注入，生成目录内的副本固定为 `0600`。
- 选择 AIOps App 后可在同一配置单启用 Prometheus、Alertmanager、Loki、Alloy、
  Node Exporter 及数据库 Exporter；实时图表由 AIOps 原生页面提供，OEM 仍只在 App 内配置。

KBot 与 AIOps 观测栈保持两个独立 Compose Project，避免混淆业务生命周期和监控
生命周期；统一配置单和 `kbot-deploy` 会在变更容器前完成两套配置校验，再按顺序应用。

当前镜像平台固定为 `linux/amd64`。在完成 ARM 依赖矩阵验证前，不将 OCI Ampere
实例作为正式目标。

## 脱机交付

KBot 1.0 修正版离线归档为 `kbot-1.0-6743d1cb-linux-amd64.tar.zst`，大小
`3876383006` 字节，SHA-256 为
`2f54ed5ad20d3ed039ddac687e033c7163ff2092acaf3a2cf2ff720a12836400`。同名
`.sha256` 文件与归档一起交付。

脱机环境不需要 Git，也不从 OCIR 拉取镜像。联网发布机使用仓库内的正式镜像锁生成一个
完整目录，其中包含 10 个镜像的 `docker image save` 归档、Docling 2.114.0 运行所需的
Layout、TableFormer、EasyOCR 模型、自包含部署控制器、配置模板、逐文件模型清单、镜像
Image ID 清单和 `SHA256SUMS`：

```bash
installation/dev/kbot-release export-offline \
  --config installation/releases/1.0.ini \
  --docling-models-dir /srv/release-assets/docling_models \
  --output installation/generated/offline/kbot-1.0
```

制作前必须选择受版本控制的正式发布描述并在本机拉取全部镜像。制作器会核对每个
RepoDigest，并要求模型源目录完整包含正式模型规格列出的三个目录；工作区不干净、锁文件
缺失、模型缺失、镜像缺失或 Digest 不符都会拒绝导出。大体积镜像和模型归档属于生成
制品，不提交 Git；Git 只保存制作器、目标机工具、配置模板、正式镜像锁、模型规格和
操作文档。其他 Docling VLM、公式、图表和 OCR 模型没有被当前 KBot 解析路径调用，后续
启用相应能力时必须先扩充正式模型规格并重新验收离线包。

目标机复制整个生成目录后执行：

```bash
./kbot-offline verify-package
./kbot-offline load-images
./kbot-offline prepare
# 编辑 runtime/deployment.ini 中的 data_dir 后：
./kbot-offline install-models
./kbot-offline verify-models --full
```

完整配置、Secret、Installer 初始化和启动顺序见
[`installation/offline/README.md`](offline/README.md)。离线入口固定使用 `--pull never`。
核心离线包包含 AIOps App，但不包含 Prometheus、Loki 等第三方观测组件镜像；观测栈需要
另行制作对应的离线包。

## 开发机构建

复制配置，不要把实际配置或 Secret 提交到 Git：

```bash
cp installation/dev/release.ini.example installation/dev/release.ini
cp installation/dev/deployment.ini.example installation/dev/deployment.ini
chmod 600 installation/dev/deployment.ini
installation/dev/kbot-release prepare-secrets \
  --config installation/dev/deployment.ini
```

`prepare-secrets` 会从配置单的 `[secrets]` 自动确定文件位置，在终端无回显地要求输入
并确认 Oracle Schema 密码，同时首次生成 KBot 主密钥。两个文件权限均设置为 `0600`。
再次执行默认保留已有文件；数据库密码确需轮换时增加
`--replace-oracle-password`。主密钥不会被该命令轮换，升级和重新部署必须保留原文件，
否则平台内已有加密凭据将无法解密。密码不要作为命令参数输入。

在 `deployment.ini` 中填写外部 Oracle 地址、`[apps]` 开关和需要启用的观测组件，
并保持 AIOps 变更 Kill Switch 默认关闭。观测组件仍复用
`scripts/deployment/aiops_observability/` 的规范资源和逐服务 Secret 规则，不维护第二套实现。
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

只渲染和检查，不启动容器：

```bash
installation/dev/kbot-release render-deployment
docker compose --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml config
```

确认配置单后，用无参数入口执行真正部署：

```bash
installation/dev/kbot-deploy
```

它先重新渲染并校验，再启动所选 App；若 `[observability] enabled = true`，随后调用
既有零参数 `scripts/aiops-stack` 入口应用观测组件。任一配置在预检阶段不合法时不会启动
KBot 容器。生成的 `installation-input.json` 会记录 App、服务和观测组件的最终选择。

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
Installer 默认读取部署器按 App 选择生成的 `oracle_schema_services.ini`，不会为未选 App
创建业务表。

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
